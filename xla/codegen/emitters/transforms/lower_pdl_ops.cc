/* Copyright 2026 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <iterator>
#include <memory>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "xla/backends/gpu/codegen/emitters/ir/xla_gpu_ops.h"
#include "xla/codegen/emitters/transforms/passes.h"

namespace xla {
namespace emitters {
namespace {

#define GEN_PASS_DEF_LOWERPDLWAITPASS
#include "xla/codegen/emitters/transforms/passes.h.inc"

constexpr char kPdlDependencyAttr[] = "xla.pdl_dependency";
constexpr int kMinPostLaunchWorkOps = 48;

bool IsLoadOp(mlir::Operation* op) {
  return mlir::isa<mlir::LLVM::LoadOp>(op) ||
         op->getName().getStringRef().ends_with(".load");
}

bool IsStoreOp(mlir::Operation* op) {
  return mlir::isa<mlir::LLVM::StoreOp>(op) ||
         op->getName().getStringRef().ends_with(".store");
}

bool IsKernelFunctionOp(mlir::Operation* op) {
  llvm::StringRef op_name = op->getName().getStringRef();
  return op_name == "func.func" || op_name == "tt.func";
}

bool IsBeforeInBlock(mlir::Operation* op, mlir::Operation* insertion_point) {
  return op->getBlock() != insertion_point->getBlock() ||
         op->isBeforeInBlock(insertion_point);
}

bool ValueTransitivelyUsesAny(
    mlir::Value value, llvm::ArrayRef<mlir::Value> targets,
    llvm::SmallPtrSetImpl<mlir::Operation*>& visited) {
  if (llvm::is_contained(targets, value)) {
    return true;
  }

  mlir::Operation* defining_op = value.getDefiningOp();
  if (defining_op == nullptr || !visited.insert(defining_op).second) {
    return false;
  }

  return llvm::any_of(defining_op->getOperands(), [&](mlir::Value operand) {
    return ValueTransitivelyUsesAny(operand, targets, visited);
  });
}

bool IsMovableAddressDependency(mlir::Operation* op) {
  if (op->getNumRegions() != 0 || op->hasTrait<mlir::OpTrait::IsTerminator>() ||
      mlir::isa<xla::gpu::PdlWaitOp, xla::gpu::PdlLaunchOp, mlir::LLVM::LoadOp,
                mlir::LLVM::StoreOp, mlir::LLVM::CallOp,
                mlir::LLVM::InlineAsmOp>(op)) {
    return false;
  }

  if (mlir::isa<mlir::LLVM::CallIntrinsicOp>(op)) {
    return op->getNumResults() != 0;
  }

  llvm::StringRef op_name = op->getName().getStringRef();
  if (op_name == "llvm.addrspacecast" || op_name == "llvm.getelementptr" ||
      op_name == "llvm.ptrtoint" || op_name == "llvm.inttoptr" ||
      op_name == "llvm.zext" || op_name == "llvm.sext" ||
      op_name == "llvm.trunc" || op_name == "llvm.add" ||
      op_name == "llvm.sub" || op_name == "llvm.mul" || op_name == "llvm.shl" ||
      op_name == "llvm.lshr" || op_name == "llvm.ashr" ||
      op_name == "llvm.and" || op_name == "llvm.or" ||
      op_name == "arith.index_cast" || op_name == "arith.index_castui") {
    return op->getNumResults() != 0;
  }

  auto memory_effect = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(op);
  return memory_effect && memory_effect.hasNoEffect();
}

bool CollectAddressOpsToMove(
    mlir::Value value, mlir::Operation* insertion_point,
    llvm::ArrayRef<mlir::Value> dependent_args,
    llvm::SmallPtrSetImpl<mlir::Operation*>& visited,
    llvm::SmallVectorImpl<mlir::Operation*>& ops_to_move) {
  mlir::Operation* defining_op = value.getDefiningOp();
  if (defining_op == nullptr || IsBeforeInBlock(defining_op, insertion_point)) {
    return true;
  }
  if (defining_op->getBlock() != insertion_point->getBlock() ||
      !visited.insert(defining_op).second) {
    return false;
  }
  llvm::SmallPtrSet<mlir::Operation*, 8> visited_dependency_ops;
  if (ValueTransitivelyUsesAny(value, dependent_args, visited_dependency_ops) ||
      !IsMovableAddressDependency(defining_op)) {
    return false;
  }

  for (mlir::Value operand : defining_op->getOperands()) {
    if (!CollectAddressOpsToMove(operand, insertion_point, dependent_args,
                                 visited, ops_to_move)) {
      return false;
    }
  }
  ops_to_move.push_back(defining_op);
  return true;
}

void HoistIndependentLlvmLoadsBeforeWait(mlir::FunctionOpInterface func) {
  llvm::SmallVector<mlir::Value, 8> dependent_args;
  for (int i = 0; i < func.getNumArguments(); ++i) {
    if (func.getArgAttr(i, kPdlDependencyAttr)) {
      dependent_args.push_back(func.getArgument(i));
    }
  }
  if (dependent_args.empty()) {
    return;
  }

  func.walk([&](xla::gpu::PdlWaitOp wait) {
    mlir::Block* block = wait->getBlock();
    llvm::SmallVector<mlir::Operation*, 16> ops_to_move;
    for (auto it = std::next(wait->getIterator()), end = block->end();
         it != end; ++it) {
      mlir::Operation* op = &*it;
      if (mlir::isa<xla::gpu::PdlLaunchOp, mlir::LLVM::StoreOp>(op) ||
          op->hasTrait<mlir::OpTrait::IsTerminator>()) {
        break;
      }

      auto load = mlir::dyn_cast<mlir::LLVM::LoadOp>(op);
      if (!load || load.getVolatile_() ||
          load.getOrdering() != mlir::LLVM::AtomicOrdering::not_atomic) {
        continue;
      }

      llvm::SmallPtrSet<mlir::Operation*, 8> visited_dependency_ops;
      if (ValueTransitivelyUsesAny(load.getAddr(), dependent_args,
                                   visited_dependency_ops)) {
        continue;
      }

      llvm::SmallVector<mlir::Operation*, 8> load_ops_to_move;
      llvm::SmallPtrSet<mlir::Operation*, 8> visited_address_ops;
      if (!CollectAddressOpsToMove(load.getAddr(), wait, dependent_args,
                                   visited_address_ops, load_ops_to_move)) {
        continue;
      }

      llvm::append_range(ops_to_move, load_ops_to_move);
      ops_to_move.push_back(load);
    }

    llvm::SmallPtrSet<mlir::Operation*, 16> moved;
    for (mlir::Operation* op : ops_to_move) {
      if (moved.insert(op).second) {
        op->moveBefore(wait);
      }
    }
  });
}

int PostLaunchWorkCost(mlir::Operation* op) {
  if (op->getNumRegions() != 0 || op->hasTrait<mlir::OpTrait::IsTerminator>() ||
      op->hasTrait<mlir::OpTrait::ConstantLike>() ||
      mlir::isa<xla::gpu::PdlWaitOp, xla::gpu::PdlLaunchOp, mlir::LLVM::CallOp,
                mlir::LLVM::CallIntrinsicOp, mlir::LLVM::InlineAsmOp,
                mlir::LLVM::LoadOp, mlir::LLVM::StoreOp>(op) ||
      IsLoadOp(op) || IsStoreOp(op)) {
    return 0;
  }

  llvm::StringRef op_name = op->getName().getStringRef();
  if (op_name.starts_with("arith.") || op_name.starts_with("math.") ||
      op_name.starts_with("tt.")) {
    return op->getNumResults() != 0 ? 1 : 0;
  }

  auto memory_effect = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(op);
  if (memory_effect) {
    return memory_effect.hasNoEffect() ? 1 : 0;
  }

  return op->getNumResults() != 0 ? 1 : 0;
}

bool IsSlowMathIntrinsic(mlir::Operation* op) {
  if (auto intrinsic = mlir::dyn_cast<mlir::LLVM::CallIntrinsicOp>(op)) {
    return intrinsic.getIntrin().contains("ex2") ||
           intrinsic.getIntrin().contains("exp");
  }

  llvm::StringRef op_name = op->getName().getStringRef();
  return op_name.starts_with("math.exp") || op_name.contains(".ex2");
}

bool HasEnoughTerminatorPostLaunchWork(int work) {
  return work >= kMinPostLaunchWorkOps;
}

bool ContainsTritonDot(mlir::Operation* op) {
  bool contains_dot = op->getName().getStringRef() == "tt.dot";
  op->walk([&](mlir::Operation* nested_op) {
    if (nested_op->getName().getStringRef() == "tt.dot") {
      contains_dot = true;
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });
  return contains_dot;
}

mlir::Operation* FindProfitableLaunchPointAfterWait(xla::gpu::PdlWaitOp wait) {
  mlir::Operation* last_load = nullptr;
  mlir::Operation* last_gemm = nullptr;
  int post_launch_work = 0;
  int post_gemm_work = 0;
  bool has_post_load_slow_math = false;
  bool has_post_gemm_slow_math = false;

  for (mlir::Operation* op = wait->getNextNode(); op != nullptr;
       op = op->getNextNode()) {
    if (IsStoreOp(op)) {
      if (last_gemm != nullptr && !has_post_gemm_slow_math &&
          post_gemm_work >= kMinPostLaunchWorkOps) {
        return last_gemm;
      }
      return last_load != nullptr && !has_post_load_slow_math &&
                     post_launch_work >= kMinPostLaunchWorkOps
                 ? last_load
                 : nullptr;
    }
    if (op->hasTrait<mlir::OpTrait::IsTerminator>()) {
      if (mlir::isa<mlir::func::ReturnOp>(op)) {
        return nullptr;
      }
      if (last_gemm != nullptr && !has_post_gemm_slow_math &&
          HasEnoughTerminatorPostLaunchWork(post_gemm_work)) {
        return last_gemm;
      }
      return last_load != nullptr && !has_post_load_slow_math &&
                     HasEnoughTerminatorPostLaunchWork(post_launch_work)
                 ? last_load
                 : nullptr;
    }
    if (mlir::isa<xla::gpu::PdlWaitOp, xla::gpu::PdlLaunchOp>(op)) {
      return nullptr;
    }
    if (ContainsTritonDot(op)) {
      last_load = nullptr;
      last_gemm = op;
      post_gemm_work = 0;
      has_post_gemm_slow_math = false;
      continue;
    }
    if (IsLoadOp(op)) {
      last_load = op;
      post_launch_work = 0;
      has_post_load_slow_math = false;
    }
    if (last_load != nullptr) {
      has_post_load_slow_math |= IsSlowMathIntrinsic(op);
      post_launch_work += PostLaunchWorkCost(op);
    }
    if (last_gemm != nullptr) {
      has_post_gemm_slow_math |= IsSlowMathIntrinsic(op);
      post_gemm_work += PostLaunchWorkCost(op);
    }
  }

  return nullptr;
}

mlir::Operation* FindProfitableLaunchPointBeforeWait(xla::gpu::PdlWaitOp wait) {
  mlir::Operation* last_gemm = nullptr;
  int post_gemm_work = 0;
  bool has_post_gemm_slow_math = false;

  for (mlir::Operation& op : *wait->getBlock()) {
    if (&op == wait.getOperation()) {
      break;
    }

    if (mlir::isa<xla::gpu::PdlLaunchOp>(&op) || IsStoreOp(&op) ||
        op.hasTrait<mlir::OpTrait::IsTerminator>()) {
      last_gemm = nullptr;
      post_gemm_work = 0;
      has_post_gemm_slow_math = false;
      continue;
    }

    if (ContainsTritonDot(&op)) {
      last_gemm = &op;
      post_gemm_work = 0;
      has_post_gemm_slow_math = false;
      continue;
    }

    if (last_gemm != nullptr) {
      has_post_gemm_slow_math |= IsSlowMathIntrinsic(&op);
      post_gemm_work += PostLaunchWorkCost(&op);
    }
  }

  if (last_gemm == nullptr) {
    return nullptr;
  }

  for (mlir::Operation* op = wait->getNextNode(); op != nullptr;
       op = op->getNextNode()) {
    if (IsStoreOp(op)) {
      return !has_post_gemm_slow_math && post_gemm_work >= kMinPostLaunchWorkOps
                 ? last_gemm
                 : nullptr;
    }
    if (op->hasTrait<mlir::OpTrait::IsTerminator>()) {
      if (mlir::isa<mlir::func::ReturnOp>(op)) {
        return nullptr;
      }
      return !has_post_gemm_slow_math &&
                     HasEnoughTerminatorPostLaunchWork(post_gemm_work)
                 ? last_gemm
                 : nullptr;
    }
    if (mlir::isa<xla::gpu::PdlWaitOp, xla::gpu::PdlLaunchOp>(op)) {
      return nullptr;
    }
    has_post_gemm_slow_math |= IsSlowMathIntrinsic(op);
    post_gemm_work += PostLaunchWorkCost(op);
  }

  return nullptr;
}

mlir::Operation* FindPreferredPdlLaunchPoint(mlir::Operation* kernel) {
  llvm::SmallVector<xla::gpu::PdlWaitOp, 4> waits;
  kernel->walk([&](xla::gpu::PdlWaitOp wait) { waits.push_back(wait); });
  for (xla::gpu::PdlWaitOp wait : waits) {
    mlir::Operation* launch_point = FindProfitableLaunchPointBeforeWait(wait);
    if (launch_point == nullptr) {
      launch_point = FindProfitableLaunchPointAfterWait(wait);
    }
    if (launch_point == nullptr) {
      continue;
    }
    return launch_point;
  }

  return nullptr;
}

void ReplaceWithPreferredPdlLaunch(mlir::Operation* kernel) {
  llvm::SmallVector<xla::gpu::PdlLaunchOp, 4> launches_to_erase;
  kernel->walk([&](xla::gpu::PdlLaunchOp launch) {
    launches_to_erase.push_back(launch);
  });
  for (xla::gpu::PdlLaunchOp launch : launches_to_erase) {
    launch.erase();
  }

  mlir::Operation* launch_point = FindPreferredPdlLaunchPoint(kernel);
  if (launch_point == nullptr) {
    return;
  }

  mlir::OpBuilder builder(launch_point);
  builder.setInsertionPointAfter(launch_point);
  builder.create<xla::gpu::PdlLaunchOp>(launch_point->getLoc());
}

bool HasEnoughPostLaunchWork(xla::gpu::PdlLaunchOp launch) {
  int post_launch_work = 0;
  bool has_post_launch_slow_math = false;
  for (mlir::Operation* op = launch->getNextNode(); op != nullptr;
       op = op->getNextNode()) {
    if (IsStoreOp(op)) {
      return !has_post_launch_slow_math &&
             post_launch_work >= kMinPostLaunchWorkOps;
    }
    if (op->hasTrait<mlir::OpTrait::IsTerminator>()) {
      return !mlir::isa<mlir::func::ReturnOp>(op) &&
             !has_post_launch_slow_math &&
             HasEnoughTerminatorPostLaunchWork(post_launch_work);
    }
    has_post_launch_slow_math |= IsSlowMathIntrinsic(op);
    post_launch_work += PostLaunchWorkCost(op);
  }
  return false;
}

void RemoveUnprofitablePdlLaunches(mlir::Operation* kernel) {
  llvm::SmallVector<xla::gpu::PdlLaunchOp, 4> launches_to_erase;
  kernel->walk([&](xla::gpu::PdlLaunchOp launch) {
    if (!HasEnoughPostLaunchWork(launch)) {
      launches_to_erase.push_back(launch);
    }
  });

  for (xla::gpu::PdlLaunchOp launch : launches_to_erase) {
    launch.erase();
  }
}

bool HasPriorLoadInBlock(mlir::Operation* op) {
  for (mlir::Operation* prev = op->getPrevNode(); prev != nullptr;
       prev = prev->getPrevNode()) {
    if (IsLoadOp(prev)) {
      return true;
    }
  }
  return false;
}

void CreateUnanchoredPdlInlineAsm(mlir::PatternRewriter& rewriter,
                                  mlir::Location loc,
                                  mlir::StringRef pdl_instruction,
                                  bool memory_clobber = false) {
  auto asm_dialect_attr = mlir::LLVM::AsmDialectAttr::get(
      rewriter.getContext(), mlir::LLVM::AsmDialect::AD_ATT);
  mlir::LLVM::InlineAsmOp::create(
      rewriter, loc, mlir::LLVM::LLVMVoidType::get(rewriter.getContext()),
      mlir::ValueRange{}, (pdl_instruction + ";").str(),
      memory_clobber ? "~{memory}" : "",
      /*has_side_effects=*/true,
      /*is_align_stack=*/false, mlir::LLVM::TailCallKind::None,
      asm_dialect_attr,
      /*operand_attrs=*/mlir::ArrayAttr());
}

struct LowerPdlWaitPattern
    : public mlir::OpRewritePattern<xla::gpu::PdlWaitOp> {
  using OpRewritePattern<xla::gpu::PdlWaitOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      xla::gpu::PdlWaitOp op, mlir::PatternRewriter& rewriter) const override {
    const bool needs_memory_barrier = HasPriorLoadInBlock(op);
    if (!needs_memory_barrier) {
      rewriter.replaceOpWithNewOp<mlir::NVVM::GriddepcontrolOp>(
          op, mlir::NVVM::GridDepActionKind::wait);
      return mlir::success();
    }

    CreateUnanchoredPdlInlineAsm(rewriter, op.getLoc(),
                                 "membar.cta; griddepcontrol.wait",
                                 /*memory_clobber=*/true);
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

struct LowerPdlLaunchPattern
    : public mlir::OpRewritePattern<xla::gpu::PdlLaunchOp> {
  using OpRewritePattern<xla::gpu::PdlLaunchOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      xla::gpu::PdlLaunchOp op,
      mlir::PatternRewriter& rewriter) const override {
    CreateUnanchoredPdlInlineAsm(rewriter, op.getLoc(),
                                 "membar.cta; griddepcontrol.launch_dependents",
                                 /*memory_clobber=*/true);
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

class LowerPdlWaitPass : public impl::LowerPdlWaitPassBase<LowerPdlWaitPass> {
 public:
  void runOnOperation() override {
    llvm::SmallVector<mlir::Operation*, 8> kernels;
    getOperation()->walk([&](mlir::Operation* op) {
      if (IsKernelFunctionOp(op)) {
        kernels.push_back(op);
      }
    });

    for (mlir::Operation* kernel : kernels) {
      if (auto func = mlir::dyn_cast<mlir::FunctionOpInterface>(kernel)) {
        HoistIndependentLlvmLoadsBeforeWait(func);
      }
      RemoveUnprofitablePdlLaunches(kernel);
      ReplaceWithPreferredPdlLaunch(kernel);
      RemoveUnprofitablePdlLaunches(kernel);
    }

    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<LowerPdlWaitPattern, LowerPdlLaunchPattern>(&getContext());
    if (mlir::failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> CreateLowerPdlWaitPass() {
  return std::make_unique<LowerPdlWaitPass>();
}

}  // namespace emitters
}  // namespace xla
