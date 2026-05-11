// RUN: emitters_opt %s --allow-unregistered-dialect -split-input-file \
// RUN: -xla-gpu-insert-pdl | FileCheck %s --check-prefix=INSERT
// RUN: emitters_opt %s --allow-unregistered-dialect -split-input-file \
// RUN: -xla-gpu-insert-pdl \
// RUN: -xla-lower-tensors="gpu_device_info='cuda_compute_capability {major: 9}'" \
// RUN: -xla-lower-pdl-wait \
// RUN: | FileCheck %s --check-prefix=LOWER

func.func @pdl_entry_store(%arg0: tensor<8xf32> {xla.slice_index = 0}) -> tensor<8xf32> attributes {xla.entry} {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 1.0 : f32
  %out = tensor.insert %cst into %arg0[%c0] : tensor<8xf32>
  func.return %out : tensor<8xf32>
}

// INSERT-LABEL: func.func @pdl_entry_store(
// INSERT: xla_gpu.pdl_wait
// INSERT: tensor.insert %cst into %arg0[%c0]
// INSERT-NOT: xla_gpu.pdl_launch
// INSERT: return

// LOWER-LABEL: func.func @pdl_entry_store(
// LOWER: nvvm.griddepcontrol wait
// LOWER-NOT: griddepcontrol.wait
// LOWER: llvm.store
// LOWER-NOT: griddepcontrol.launch_dependents
// LOWER: return

// -----

func.func private @helper(%arg0: tensor<8xf32> {xla.slice_index = 0}, %arg1: index) -> f32 {
  %v = tensor.extract %arg0[%arg1] : tensor<8xf32>
  func.return %v : f32
}

func.func @entry_calls_helper(%arg0: tensor<8xf32> {xla.slice_index = 0}, %arg1: index) -> f32 attributes {xla.entry} {
  %v0 = call @helper(%arg0, %arg1) : (tensor<8xf32>, index) -> f32
  %v1 = tensor.extract %arg0[%arg1] : tensor<8xf32>
  %sum = arith.addf %v0, %v1 : f32
  func.return %sum : f32
}

// INSERT-LABEL: func.func private @helper(
// INSERT-NOT: pdl
// INSERT: tensor.extract
// INSERT: return
// INSERT-LABEL: func.func @entry_calls_helper(
// INSERT: xla_gpu.pdl_wait
// INSERT: call @helper
// INSERT: tensor.extract
// INSERT-NOT: xla_gpu.pdl_launch
// INSERT: return

// LOWER-LABEL: func.func private @helper(
// LOWER-NOT: griddepcontrol
// LOWER: llvm.load
// LOWER: return
// LOWER-LABEL: func.func @entry_calls_helper(
// LOWER: nvvm.griddepcontrol wait
// LOWER-NOT: griddepcontrol.wait
// LOWER: call @helper
// LOWER: llvm.load
// LOWER-NOT: griddepcontrol.launch_dependents
// LOWER: return

// -----

func.func @launch_before_first_write(
    %dep_out: tensor<8xf32> {xla.slice_index = 0},
    %tail_out: tensor<8xf32> {xla.slice_index = 1}) -> (tensor<8xf32>, tensor<8xf32>) attributes {xla.entry} {
  %c0 = arith.constant 0 : index
  %cst0 = arith.constant 1.0 : f32
  %cst1 = arith.constant 2.0 : f32
  %updated_dep = tensor.insert %cst0 into %dep_out[%c0] : tensor<8xf32>
  %updated_tail = tensor.insert %cst1 into %tail_out[%c0] : tensor<8xf32>
  func.return %updated_dep, %updated_tail : tensor<8xf32>, tensor<8xf32>
}

// INSERT-LABEL: func.func @launch_before_first_write(
// INSERT: tensor.insert
// INSERT-NOT: xla_gpu.pdl_launch
// INSERT: tensor.insert
// INSERT: return

// LOWER-LABEL: func.func @launch_before_first_write(
// LOWER: llvm.store
// LOWER-NOT: griddepcontrol.launch_dependents
// LOWER: llvm.store
// LOWER: return

// -----

func.func @mixed_memory_access(%arg0: tensor<4xf32> {xla.slice_index = 0}) -> (f32, f32) attributes {xla.entry} {
  %c0 = arith.constant 0 : index
  %t = arith.constant dense<1.0> : tensor<4xf32>
  %v_local = tensor.extract %t[%c0] : tensor<4xf32>
  %v_global = tensor.extract %arg0[%c0] : tensor<4xf32>
  func.return %v_local, %v_global : f32, f32
}

// INSERT-LABEL: func.func @mixed_memory_access
// INSERT: xla_gpu.pdl_wait
// INSERT: arith.constant 0
// INSERT: arith.constant dense
// INSERT: tensor.extract
// INSERT: tensor.extract

// -----

func.func @sink_wait_to_dependent_read(
    %dep: tensor<8xf32> {xla.slice_index = 0, xla.pdl_dependency},
    %independent: tensor<8xf32> {xla.slice_index = 1}) -> (f32, f32) attributes {xla.entry} {
  %c0 = arith.constant 0 : index
  %v_independent = tensor.extract %independent[%c0] : tensor<8xf32>
  %v_dep = tensor.extract %dep[%c0] : tensor<8xf32>
  func.return %v_independent, %v_dep : f32, f32
}

// INSERT-LABEL: func.func @sink_wait_to_dependent_read(
// INSERT: tensor.extract %arg1
// INSERT: xla_gpu.pdl_wait
// INSERT: tensor.extract %arg0
// INSERT-NOT: xla_gpu.pdl_launch
// INSERT: return

// LOWER-LABEL: func.func @sink_wait_to_dependent_read(
// LOWER: llvm.load
// LOWER: llvm.inline_asm {{.*}}membar.cta; griddepcontrol.wait;
// LOWER-NOT: griddepcontrol.wait
// LOWER: llvm.load
// LOWER-NOT: griddepcontrol.launch_dependents
// LOWER: return

// -----

func.func @launch_after_reads_before_compute(
    %dep: tensor<8xf32> {xla.slice_index = 0, xla.pdl_dependency},
    %independent: tensor<8xf32> {xla.slice_index = 1},
    %out: tensor<8xf32> {xla.slice_index = 2}) -> tensor<8xf32> attributes {xla.entry} {
  %c0 = arith.constant 0 : index
  %v_independent = tensor.extract %independent[%c0] : tensor<8xf32>
  %v_dep = tensor.extract %dep[%c0] : tensor<8xf32>
  %sum = arith.addf %v_independent, %v_dep : f32
  %updated = tensor.insert %sum into %out[%c0] : tensor<8xf32>
  func.return %updated : tensor<8xf32>
}

// INSERT-LABEL: func.func @launch_after_reads_before_compute(
// INSERT: tensor.extract %arg1
// INSERT: xla_gpu.pdl_wait
// INSERT: tensor.extract %arg0
// INSERT: xla_gpu.pdl_launch
// INSERT: arith.addf
// INSERT: tensor.insert
// INSERT-NOT: xla_gpu.pdl_launch
// INSERT: return

// LOWER-LABEL: func.func @launch_after_reads_before_compute(
// LOWER: llvm.load
// LOWER: llvm.inline_asm {{.*}}membar.cta; griddepcontrol.wait;
// LOWER-NOT: griddepcontrol.wait
// LOWER: llvm.load
// LOWER: arith.addf
// LOWER: llvm.store
// LOWER-NOT: griddepcontrol.launch_dependents
// LOWER: return

// -----

func.func @hoist_independent_read_before_wait(
    %dep: tensor<8xf32> {xla.slice_index = 0, xla.pdl_dependency},
    %independent: tensor<8xf32> {xla.slice_index = 1},
    %out: tensor<8xf32> {xla.slice_index = 2}) -> tensor<8xf32> attributes {xla.entry} {
  %c0 = arith.constant 0 : index
  %v_dep = tensor.extract %dep[%c0] : tensor<8xf32>
  %v_independent = tensor.extract %independent[%c0] : tensor<8xf32>
  %sum = arith.addf %v_independent, %v_dep : f32
  %updated = tensor.insert %sum into %out[%c0] : tensor<8xf32>
  func.return %updated : tensor<8xf32>
}

// INSERT-LABEL: func.func @hoist_independent_read_before_wait(
// INSERT: tensor.extract %arg1
// INSERT: xla_gpu.pdl_wait
// INSERT: tensor.extract %arg0
// INSERT: xla_gpu.pdl_launch
// INSERT: arith.addf
// INSERT: tensor.insert
// INSERT-NOT: xla_gpu.pdl_launch
// INSERT: return

// LOWER-LABEL: func.func @hoist_independent_read_before_wait(
// LOWER: llvm.load
// LOWER: llvm.inline_asm {{.*}}membar.cta; griddepcontrol.wait;
// LOWER-NOT: griddepcontrol.wait
// LOWER: llvm.load
// LOWER: arith.addf
// LOWER: llvm.store
// LOWER-NOT: griddepcontrol.launch_dependents
// LOWER: return

// -----

func.func @wait_inside_if_before_dependent_read(
    %dep: tensor<8xf32> {xla.slice_index = 0, xla.pdl_dependency},
    %independent: tensor<8xf32> {xla.slice_index = 1, xla.invariant},
    %out: tensor<8xf32> {xla.slice_index = 2}) -> tensor<8xf32> attributes {xla.entry} {
  %c0 = arith.constant 0 : index
  %true = arith.constant true
  %updated = scf.if %true -> (tensor<8xf32>) {
    %independent_value = tensor.extract %independent[%c0] : tensor<8xf32>
    %dep_value = tensor.extract %dep[%c0] : tensor<8xf32>
    %sum = arith.addf %independent_value, %dep_value : f32
    %next = tensor.insert %sum into %out[%c0] : tensor<8xf32>
    scf.yield %next : tensor<8xf32>
  } else {
    scf.yield %out : tensor<8xf32>
  }
  func.return %updated : tensor<8xf32>
}

// INSERT-LABEL: func.func @wait_inside_if_before_dependent_read(
// INSERT: scf.if
// INSERT: tensor.extract %arg1
// INSERT-NEXT: xla_gpu.pdl_wait
// INSERT-NEXT: tensor.extract %arg0
// INSERT-NOT: xla_gpu.pdl_wait
// INSERT: return

// -----

func.func @wait_before_region_read(
    %dep: tensor<8xf32> {xla.slice_index = 0, xla.pdl_dependency},
    %out: tensor<8xf32> {xla.slice_index = 1}) -> tensor<8xf32> attributes {xla.entry} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %zero = arith.constant 0.0 : f32
  %sum = scf.for %i = %c0 to %c2 step %c1 iter_args(%acc = %zero) -> f32 {
    %v = tensor.extract %dep[%c0] : tensor<8xf32>
    %next = arith.addf %acc, %v : f32
    scf.yield %next : f32
  }
  %updated = tensor.insert %sum into %out[%c0] : tensor<8xf32>
  func.return %updated : tensor<8xf32>
}

// INSERT-LABEL: func.func @wait_before_region_read(
// INSERT: xla_gpu.pdl_wait
// INSERT-NEXT: scf.for
// INSERT-NOT: xla_gpu.pdl_wait
// INSERT: tensor.insert
// INSERT-NOT: xla_gpu.pdl_wait
// INSERT: return

// -----

func.func private @lower_hoists_independent_llvm_load(
    %dep: !llvm.ptr {xla.pdl_dependency},
    %independent: !llvm.ptr) -> f32 {
  "xla_gpu.pdl_wait"() : () -> ()
  %dep_addr = llvm.getelementptr inbounds %dep[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<8 x f32>
  %dep_value = llvm.load %dep_addr : !llvm.ptr -> f32
  %independent_addr = llvm.getelementptr inbounds %independent[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<8 x f32>
  %independent_value = llvm.load %independent_addr : !llvm.ptr -> f32
  %sum = arith.addf %independent_value, %dep_value : f32
  "xla_gpu.pdl_launch"() : () -> ()
  func.return %sum : f32
}

// LOWER-LABEL: func.func private @lower_hoists_independent_llvm_load(
// LOWER: llvm.getelementptr{{.*}}%arg1
// LOWER: llvm.load
// LOWER: llvm.inline_asm {{.*}}membar.cta; griddepcontrol.wait;
// LOWER: llvm.getelementptr{{.*}}%arg0
// LOWER: llvm.load
// LOWER: arith.addf
// LOWER-NOT: griddepcontrol.launch_dependents
// LOWER: return

// -----

func.func private @lower_prefers_launch_after_gemm_loop(
    %dep: !llvm.ptr {xla.pdl_dependency},
    %independent: !llvm.ptr,
    %out: !llvm.ptr) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %zero = arith.constant 0.0 : f32
  %acc = scf.for %i = %c0 to %c2 step %c1 iter_args(%iter = %zero) -> f32 {
    %lhs = "tt.load"(%independent) : (!llvm.ptr) -> f32
    %rhs = "tt.load"(%independent) : (!llvm.ptr) -> f32
    %dot = "tt.dot"(%lhs, %rhs, %iter) : (f32, f32, f32) -> f32
    scf.yield %dot : f32
  }
  %pre_wait = "tt.load"(%independent) : (!llvm.ptr) -> f32
  "xla_gpu.pdl_wait"() : () -> ()
  %dep_value = "tt.load"(%dep) : (!llvm.ptr) -> f32
  "xla_gpu.pdl_launch"() : () -> ()
  %e01 = arith.addf %pre_wait, %dep_value : f32
  %e02 = arith.addf %e01, %acc : f32
  %e03 = arith.addf %e02, %dep_value : f32
  %e04 = arith.addf %e03, %acc : f32
  %e05 = arith.addf %e04, %dep_value : f32
  %e06 = arith.addf %e05, %acc : f32
  %e07 = arith.addf %e06, %dep_value : f32
  %e08 = arith.addf %e07, %acc : f32
  %e09 = arith.addf %e08, %dep_value : f32
  %e10 = arith.addf %e09, %acc : f32
  %e11 = arith.addf %e10, %dep_value : f32
  %e12 = arith.addf %e11, %acc : f32
  %e13 = arith.addf %e12, %dep_value : f32
  %e14 = arith.addf %e13, %acc : f32
  %e15 = arith.addf %e14, %dep_value : f32
  %e16 = arith.addf %e15, %acc : f32
  %e17 = arith.addf %e16, %dep_value : f32
  %e18 = arith.addf %e17, %acc : f32
  %e19 = arith.addf %e18, %dep_value : f32
  %e20 = arith.addf %e19, %acc : f32
  %e21 = arith.addf %e20, %dep_value : f32
  %e22 = arith.addf %e21, %acc : f32
  %e23 = arith.addf %e22, %dep_value : f32
  %e24 = arith.addf %e23, %acc : f32
  %e25 = arith.addf %e24, %dep_value : f32
  %e26 = arith.addf %e25, %acc : f32
  %e27 = arith.addf %e26, %dep_value : f32
  %e28 = arith.addf %e27, %acc : f32
  %e29 = arith.addf %e28, %dep_value : f32
  %e30 = arith.addf %e29, %acc : f32
  %e31 = arith.addf %e30, %dep_value : f32
  %e32 = arith.addf %e31, %acc : f32
  %e33 = arith.addf %e32, %dep_value : f32
  %e34 = arith.addf %e33, %acc : f32
  %e35 = arith.addf %e34, %dep_value : f32
  %e36 = arith.addf %e35, %acc : f32
  %e37 = arith.addf %e36, %dep_value : f32
  %e38 = arith.addf %e37, %acc : f32
  %e39 = arith.addf %e38, %dep_value : f32
  %e40 = arith.addf %e39, %acc : f32
  %e41 = arith.addf %e40, %dep_value : f32
  %e42 = arith.addf %e41, %acc : f32
  %e43 = arith.addf %e42, %dep_value : f32
  %e44 = arith.addf %e43, %acc : f32
  %e45 = arith.addf %e44, %dep_value : f32
  %e46 = arith.addf %e45, %acc : f32
  %e47 = arith.addf %e46, %dep_value : f32
  %e48 = arith.addf %e47, %acc : f32
  llvm.store %e48, %out : f32, !llvm.ptr
  func.return
}

// LOWER-LABEL: func.func private @lower_prefers_launch_after_gemm_loop(
// LOWER: scf.for
// LOWER: "tt.dot"
// LOWER: scf.yield
// LOWER: llvm.inline_asm {{.*}}griddepcontrol.launch_dependents
// LOWER-NOT: griddepcontrol.launch_dependents
// LOWER: "tt.load"(%arg1)
// LOWER: llvm.inline_asm {{.*}}membar.cta; griddepcontrol.wait;
// LOWER: "tt.load"(%arg0)
// LOWER-NOT: griddepcontrol.launch_dependents
// LOWER: llvm.store
// LOWER-NOT: griddepcontrol.launch_dependents
// LOWER: return

// -----

func.func private @lower_drops_launch_without_post_gemm_epilogue(
    %dep: !llvm.ptr {xla.pdl_dependency},
    %independent: !llvm.ptr,
    %out: !llvm.ptr) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %zero = arith.constant 0.0 : f32
  %acc = scf.for %i = %c0 to %c2 step %c1 iter_args(%iter = %zero) -> f32 {
    %lhs = "tt.load"(%independent) : (!llvm.ptr) -> f32
    %rhs = "tt.load"(%independent) : (!llvm.ptr) -> f32
    %dot = "tt.dot"(%lhs, %rhs, %iter) : (f32, f32, f32) -> f32
    scf.yield %dot : f32
  }
  "xla_gpu.pdl_wait"() : () -> ()
  %dep_value = "tt.load"(%dep) : (!llvm.ptr) -> f32
  %sum = arith.addf %acc, %dep_value : f32
  "xla_gpu.pdl_launch"() : () -> ()
  llvm.store %sum, %out : f32, !llvm.ptr
  func.return
}

// LOWER-LABEL: func.func private @lower_drops_launch_without_post_gemm_epilogue(
// LOWER: scf.for
// LOWER: "tt.dot"
// LOWER: scf.yield
// LOWER-NOT: griddepcontrol.launch_dependents
// LOWER: nvvm.griddepcontrol wait
// LOWER-NOT: griddepcontrol.launch_dependents
// LOWER: llvm.store
// LOWER-NOT: griddepcontrol.launch_dependents
// LOWER: return
