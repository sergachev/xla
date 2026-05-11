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

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "xla/backends/gpu/codegen/emitters/ir/xla_gpu_ops.h"
#include "xla/backends/gpu/codegen/emitters/transforms/passes.h"
#include "xla/backends/gpu/codegen/triton/ir/triton_xla_ops.h"
#include "xla/codegen/emitters/ir/xla_ops.h"

namespace xla {
namespace gpu {
namespace {

#define GEN_PASS_DEF_INSERTPDLPASS
#include "xla/backends/gpu/codegen/emitters/transforms/passes.h.inc"

constexpr char kPdlDependencyAttr[] = "xla.pdl_dependency";

bool IsPrivateFunction(mlir::FunctionOpInterface func) {
  if (auto func_op = mlir::dyn_cast<mlir::func::FuncOp>(func.getOperation())) {
    return func_op.isPrivate();
  }
  return false;
}

bool ReadsDependentTensor(mlir::Operation* op) {
  return mlir::isa<mlir::tensor::ExtractOp, mlir::vector::TransferReadOp,
                   mlir::triton::xla::ExtractOp>(op);
}

bool ReadsTensor(mlir::Operation* op) {
  return mlir::isa<mlir::tensor::ExtractOp, mlir::vector::TransferReadOp,
                   mlir::triton::xla::ExtractOp>(op);
}

bool WritesTensor(mlir::Operation* op) {
  return mlir::isa<mlir::tensor::InsertOp, mlir::vector::TransferWriteOp,
                   mlir::triton::xla::InsertOp>(op);
}

bool UsesAny(mlir::Operation* op,
             const llvm::SmallVector<mlir::Value, 16>& values) {
  return llvm::any_of(op->getOperands(), [&](auto value) {
    return llvm::is_contained(values, value);
  });
}

bool OperandsAvailableBefore(mlir::Operation* op,
                             mlir::Operation* insertion_point) {
  mlir::Block* block = insertion_point->getBlock();
  return llvm::all_of(op->getOperands(), [&](mlir::Value value) {
    auto block_arg = mlir::dyn_cast<mlir::BlockArgument>(value);
    if (block_arg) {
      return block_arg.getOwner() == block;
    }

    mlir::Operation* defining_op = value.getDefiningOp();
    return defining_op == nullptr || defining_op->getBlock() != block ||
           defining_op->isBeforeInBlock(insertion_point);
  });
}

mlir::Operation* FindFirstDependentRead(mlir::FunctionOpInterface func) {
  llvm::SmallVector<mlir::Value, 16> dependent_values;
  for (int i = 0; i < func.getNumArguments(); ++i) {
    if (func.getArgAttr(i, kPdlDependencyAttr)) {
      dependent_values.push_back(func.getArgument(i));
    }
  }

  mlir::Operation* first_read = nullptr;
  func.walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation* op) {
    bool uses_dependent_value =
        llvm::any_of(op->getOperands(), [&](auto value) {
          return llvm::is_contained(dependent_values, value);
        });
    if (!uses_dependent_value) {
      return mlir::WalkResult::advance();
    }

    // Private callees may read dependent arguments. Put a single wait at the
    // public call site instead of inserting waits into private helpers that may
    // later be inlined many times.
    if (mlir::isa<mlir::func::CallOp>(op)) {
      first_read = op;
      return mlir::WalkResult::interrupt();
    }

    if (ReadsDependentTensor(op)) {
      first_read = op;
      return mlir::WalkResult::interrupt();
    }

    for (mlir::Value result : op->getResults()) {
      dependent_values.push_back(result);
    }
    return mlir::WalkResult::advance();
  });
  return first_read;
}

mlir::Operation* GetEntryBlockAncestor(mlir::FunctionOpInterface func,
                                       mlir::Operation* op) {
  mlir::Block& entry_block = func.getFunctionBody().front();
  while (op != nullptr && op->getBlock() != &entry_block) {
    op = op->getParentOp();
  }
  return op;
}

bool IsNestedOnlyInIfOps(mlir::FunctionOpInterface func, mlir::Operation* op) {
  mlir::Block& entry_block = func.getFunctionBody().front();
  while (op != nullptr && op->getBlock() != &entry_block) {
    op = op->getParentOp();
    if (op == nullptr || !mlir::isa<mlir::scf::IfOp>(op)) {
      return false;
    }
  }
  return true;
}

void HoistIndependentReadsBefore(mlir::FunctionOpInterface func,
                                 mlir::Operation* insertion_point) {
  mlir::Block* block = insertion_point->getBlock();
  llvm::SmallVector<mlir::Value, 16> dependent_values;
  for (int i = 0; i < func.getNumArguments(); ++i) {
    if (func.getArgAttr(i, kPdlDependencyAttr)) {
      dependent_values.push_back(func.getArgument(i));
    }
  }

  for (mlir::Operation& op : *block) {
    if (&op == insertion_point) {
      break;
    }
    if (!UsesAny(&op, dependent_values)) {
      continue;
    }
    for (mlir::Value result : op.getResults()) {
      dependent_values.push_back(result);
    }
  }

  llvm::SmallVector<mlir::Operation*, 8> reads_to_move;
  for (auto it = std::next(insertion_point->getIterator()), end = block->end();
       it != end; ++it) {
    mlir::Operation* op = &*it;
    if (WritesTensor(op) || op->hasTrait<mlir::OpTrait::IsTerminator>()) {
      break;
    }

    bool uses_dependent_value = UsesAny(op, dependent_values);
    if (uses_dependent_value) {
      for (mlir::Value result : op->getResults()) {
        dependent_values.push_back(result);
      }
      continue;
    }

    if (ReadsTensor(op) && OperandsAvailableBefore(op, insertion_point)) {
      reads_to_move.push_back(op);
    }
  }

  for (mlir::Operation* op : reads_to_move) {
    op->moveBefore(insertion_point);
  }
}

bool InsertPdlWait(mlir::FunctionOpInterface func, bool fallback_at_begin) {
  if (func.getFunctionBody().empty()) {
    return false;
  }

  if (mlir::Operation* first_read = FindFirstDependentRead(func)) {
    mlir::Operation* insertion_point =
        IsNestedOnlyInIfOps(func, first_read)
            ? first_read
            : GetEntryBlockAncestor(func, first_read);
    if (insertion_point == nullptr) {
      mlir::Block& entry_block = func.getFunctionBody().front();
      mlir::OpBuilder::atBlockBegin(&entry_block)
          .create<xla::gpu::PdlWaitOp>(func.getLoc());
      return true;
    }
    HoistIndependentReadsBefore(func, insertion_point);
    mlir::OpBuilder(insertion_point).create<xla::gpu::PdlWaitOp>(func.getLoc());
    return true;
  }

  if (fallback_at_begin) {
    mlir::Block& entry_block = func.getFunctionBody().front();
    mlir::OpBuilder::atBlockBegin(&entry_block)
        .create<xla::gpu::PdlWaitOp>(func.getLoc());
    return true;
  }

  return false;
}

void InsertPdlLaunch(mlir::FunctionOpInterface func) {
  // Keep launch completion top-level so it executes at most once per kernel.
  // Only place it after a read and before a later write; late lowering prunes
  // cases that do not have enough lowered work after final launch placement.
  mlir::Block& entry_block = func.getFunctionBody().front();
  mlir::Operation* first_write = nullptr;
  mlir::Operation* last_read_before_write = nullptr;

  for (mlir::Operation& op : entry_block) {
    if (first_write != nullptr) {
      break;
    }

    if (WritesTensor(&op)) {
      first_write = &op;
      continue;
    }

    if (ReadsTensor(&op)) {
      last_read_before_write = &op;
    }
  }

  if (first_write == nullptr || last_read_before_write == nullptr) {
    return;
  }

  mlir::OpBuilder builder(last_read_before_write);
  builder.setInsertionPointAfter(last_read_before_write);
  builder.create<xla::gpu::PdlLaunchOp>(last_read_before_write->getLoc());
}

class InsertPDLPass : public impl::InsertPDLPassBase<InsertPDLPass> {
 public:
  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();

    module.walk([&](mlir::FunctionOpInterface func) {
      if (func.getFunctionBody().empty()) {
        return;
      }

      const bool is_private = IsPrivateFunction(func);
      if (is_private) {
        return;
      }

      const bool inserted_wait =
          InsertPdlWait(func, /*fallback_at_begin=*/true);
      if (inserted_wait) {
        InsertPdlLaunch(func);
      }
    });
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> CreateInsertPDLPass() {
  return std::make_unique<InsertPDLPass>();
}

}  // namespace gpu
}  // namespace xla
