#include "cocoons/FlatteningPass.h"
#include "cocoons/Config.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"
#include <cstdlib>
#include <vector>
#include <random>
#include <map>

using namespace llvm;

namespace cocoons {

bool FlatteningPass::isEnabled() {
    const Config &C = Config::get();
    return C.Fla != EnableMode::DefaultOff;
}

PreservedAnalyses FlatteningPass::run(Function &F, FunctionAnalysisManager &AM) {
    Config &Cfg = Config::get();
    unsigned BC = F.size();
    // armv7(32 位)目标整体跳过:demote 后的大量 alloca 触发 llvm@21
    // Thumb2 Greedy Register Allocator 段错误(sub+fla 组合时,单开不触发)。
    // 32 位老设备 fla 收益最小,保守跳过。
    StringRef Triple = F.getParent()->getTargetTriple().str();
    if (Triple.starts_with("armv7") || Triple.starts_with("thumbv7")) {
        return PreservedAnalyses::all();
    }
    if (!Cfg.shouldRunFla(F, {}, BC)) {
        return PreservedAnalyses::all();
    }

    if (!flatten(F)) {
        return PreservedAnalyses::all();
    }

    errs() << ">>> [Cocoons] Control Flow Flattening applied on " << F.getName() << "\n";
    return PreservedAnalyses::none();
}

bool FlatteningPass::flatten(Function &F) {
    // 含 invoke(ObjC EH / C++ 异常)的函数跳过:
    // 下面的重写只处理 Branch/Switch 终结器,invoke 会被 eraseFromParent
    // 留下悬空的返回值使用者,后续 pass(ConstantHoisting/CodeGenPrepare 等)
    // 读到坏 IR 直接段错误(@synchronized 的 objc_sync_exit 清理路径必中)。
    for (BasicBlock &BB : F) {
        if (BB.isEHPad() || isa<InvokeInst>(BB.getTerminator())) {
            errs() << ">>> [Cocoons] fla skip (has EH/invoke): " << F.getName() << "\n";
            return false;
        }
    }

    std::vector<PHINode *> PHIs;
    std::vector<Instruction *> Regs;
    for (BasicBlock &BB : F) {
        for (Instruction &I : BB) {
            if (auto *PN = dyn_cast<PHINode>(&I)) {
                PHIs.push_back(PN);
            } else {
                for (User *U : I.users()) {
                    if (auto *UI = dyn_cast<Instruction>(U)) {
                        if (UI->getParent() != &BB) {
                            Regs.push_back(&I);
                            break;
                        }
                    }
                }
            }
        }
    }
    for (PHINode *PN : PHIs) DemotePHIToStack(PN);
    for (Instruction *I : Regs) DemoteRegToStack(*I);

    // demote 会往入口块塞 alloca/load/store,数量一多(实测 >32 个)就会触发
    // Greedy Register Allocator 段错误(arm64e armv7 都会出现,sub+fla 组合尤甚)。
    // 超阈值直接放弃平坦化该函数。
    // 注意 demote 已经发生,即使返回 false 也无害(语义等价变换)。
    if (PHIs.size() + Regs.size() > 32) {
        errs() << ">>> [Cocoons] fla skip (too many demoted regs: "
               << PHIs.size() + Regs.size() << "): " << F.getName() << "\n";
        return false;
    }

    std::vector<BasicBlock *> OrigBBs;
    for (BasicBlock &BB : F) {
        OrigBBs.push_back(&BB);
    }

    BasicBlock *EntryBB = &F.getEntryBlock();
    OrigBBs.erase(OrigBBs.begin());

    if (EntryBB->getTerminator()->getNumSuccessors() > 1) {
        BasicBlock *NewBB = SplitBlock(EntryBB, EntryBB->getTerminator());
        OrigBBs.insert(OrigBBs.begin(), NewBB);
    }

    Config &Cfg = Config::get();
    std::uniform_int_distribution<uint32_t> Dist;
    std::map<BasicBlock *, uint32_t> BBKeys;
    for (BasicBlock *BB : OrigBBs) {
        BBKeys[BB] = Dist(Cfg.Rng);
    }

    BasicBlock *LoopHeader = BasicBlock::Create(F.getContext(), "fla.header", &F, EntryBB->getNextNode());
    BasicBlock *LoopEnd = BasicBlock::Create(F.getContext(), "fla.end", &F);
    BasicBlock *SwitchDefault = BasicBlock::Create(F.getContext(), "fla.default", &F, LoopEnd);

    IRBuilder<> EntryBuilder(&EntryBB->front());
    AllocaInst *SwitchVar = EntryBuilder.CreateAlloca(Type::getInt32Ty(F.getContext()), nullptr, "fla.switchVar");

    BasicBlock *FirstBB = EntryBB->getTerminator()->getSuccessor(0);
    EntryBB->getTerminator()->eraseFromParent();
    IRBuilder<> EntryTermBuilder(EntryBB);
    EntryTermBuilder.CreateStore(ConstantInt::get(Type::getInt32Ty(F.getContext()), BBKeys[FirstBB]), SwitchVar);
    EntryTermBuilder.CreateBr(LoopHeader);

    IRBuilder<> HeaderBuilder(LoopHeader);
    Value *LoadKey = HeaderBuilder.CreateLoad(Type::getInt32Ty(F.getContext()), SwitchVar, "fla.load");
    SwitchInst *SwitchI = HeaderBuilder.CreateSwitch(LoadKey, SwitchDefault);

    IRBuilder<>(LoopEnd).CreateBr(LoopHeader);
    IRBuilder<>(SwitchDefault).CreateBr(LoopEnd);

    for (BasicBlock *BB : OrigBBs) {
        SwitchI->addCase(ConstantInt::get(Type::getInt32Ty(F.getContext()), BBKeys[BB]), BB);
        Instruction *Term = BB->getTerminator();

        if (Term->getNumSuccessors() == 0) continue;

        IRBuilder<> TermBuilder(Term);
        if (auto *BI = dyn_cast<BranchInst>(Term)) {
            if (BI->isUnconditional()) {
                TermBuilder.CreateStore(
                    ConstantInt::get(Type::getInt32Ty(F.getContext()), BBKeys[BI->getSuccessor(0)]),
                    SwitchVar);
            } else {
                Value *SelectKey = TermBuilder.CreateSelect(BI->getCondition(),
                    ConstantInt::get(Type::getInt32Ty(F.getContext()), BBKeys[BI->getSuccessor(0)]),
                    ConstantInt::get(Type::getInt32Ty(F.getContext()), BBKeys[BI->getSuccessor(1)]));
                TermBuilder.CreateStore(SelectKey, SwitchVar);
            }
        }
        else if (auto *SI = dyn_cast<SwitchInst>(Term)) {
            Value *NextKey = ConstantInt::get(Type::getInt32Ty(F.getContext()), BBKeys[SI->getDefaultDest()]);
            for (auto Case : SI->cases()) {
                NextKey = TermBuilder.CreateSelect(
                    TermBuilder.CreateICmpEQ(SI->getCondition(), Case.getCaseValue()),
                    ConstantInt::get(Type::getInt32Ty(F.getContext()), BBKeys[Case.getCaseSuccessor()]),
                    NextKey
                );
            }
            TermBuilder.CreateStore(NextKey, SwitchVar);
        }

        TermBuilder.CreateBr(LoopEnd);
        Term->eraseFromParent();
    }

    return true;
}

} // namespace cocoons
