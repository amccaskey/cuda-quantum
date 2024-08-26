# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
import os

import pytest
import numpy as np

import cudaq
from typing import Callable 

def test_mergeExternal():

    @cudaq.kernel
    def kernel(i : int):
        q = cudaq.qvector(i)
        h(q[0])

    kernel.compile() 
    kernel(10)

    otherMod = '''module attributes {quake.mangled_name_map = {__nvqpp__mlirgen__test = "__nvqpp__mlirgen__test_PyKernelEntryPointRewrite"}} {
  func.func @__nvqpp__mlirgen__test() attributes {"cudaq-entrypoint"} {
    %0 = quake.alloca !quake.veq<2>
    %1 = quake.extract_ref %0[0] : (!quake.veq<2>) -> !quake.ref
    quake.h %1 : (!quake.ref) -> ()
    return
  }
}''' 
    newMod = cudaq.jit.mergeExternalMLIRWithKernel(kernel, otherMod)
    print(newMod)
    assert '__nvqpp__mlirgen__test' in str(newMod) and '__nvqpp__mlirgen__kernel' in str(newMod)
  
def test_synthCallable():
    
    @cudaq.kernel 
    def callee(q : cudaq.qview):
        x(q[0])
        x(q[1])

    callee.compile() 

    otherMod = '''module attributes {quake.mangled_name_map = {__nvqpp__mlirgen__caller = "__nvqpp__mlirgen__caller_PyKernelEntryPointRewrite"}} {
  func.func @__nvqpp__mlirgen__caller(%arg0: !cc.callable<(!quake.veq<?>) -> ()>) attributes {"cudaq-entrypoint"} {
    %0 = quake.alloca !quake.veq<2>
    %1 = quake.relax_size %0 : (!quake.veq<2>) -> !quake.veq<?>
    %2 = cc.callable_func %arg0 : (!cc.callable<(!quake.veq<?>) -> ()>) -> ((!quake.veq<?>) -> ())
    call_indirect %2(%1) : (!quake.veq<?>) -> ()
    return
  }
}'''

    # Merge the external code with the current pure device kernel
    newMod = cudaq.jit.mergeExternalMLIRWithKernel(callee, otherMod)
    # Synthesize away the callable arg with the pure device kernel 
    cudaq.jit.synthesizeCallableBlockArgument(newMod, 'callee')
    
    # Create a new kernel from it 
    k = cudaq.PyKernelDecorator(None, kernelName='caller', module=newMod) 
    counts = cudaq.sample(k) 
    assert len(counts) == 1 and '11' in counts

 
def test_synthCallableCCCallCallableOp():
    
    @cudaq.kernel 
    def callee(q : cudaq.qview):
        x(q[0])
        x(q[1])

    callee.compile() 

    otherMod = '''module attributes {quake.mangled_name_map = {__nvqpp__mlirgen__caller = "__nvqpp__mlirgen__caller_PyKernelEntryPointRewrite"}} {
  func.func @__nvqpp__mlirgen__adapt_caller(%arg0: i64, %arg1: !cc.callable<(!quake.veq<?>) -> ()>, %arg2: !cc.stdvec<f64>, %arg3: !cc.stdvec<f64>, %arg4: !cc.stdvec<!cc.charspan>) attributes {"cudaq-entrypoint", "cudaq-kernel", no_this} {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %0 = quake.alloca !quake.veq<?>[%arg0 : i64]
    cc.call_callable %arg1, %0 : (!cc.callable<(!quake.veq<?>) -> ()>, !quake.veq<?>) -> ()
    %1 = cc.loop while ((%arg5 = %c0_i64) -> (i64)) {
      %2 = cc.stdvec_size %arg2 : (!cc.stdvec<f64>) -> i64
      %3 = arith.cmpi ult, %arg5, %2 : i64
      cc.condition %3(%arg5 : i64)
    } do {
    ^bb0(%arg5: i64):
      %2 = cc.loop while ((%arg6 = %c0_i64) -> (i64)) {
        %3 = cc.stdvec_size %arg4 : (!cc.stdvec<!cc.charspan>) -> i64
        %4 = arith.cmpi ult, %arg6, %3 : i64
        cc.condition %4(%arg6 : i64)
      } do {
      ^bb0(%arg6: i64):
        %3 = cc.stdvec_data %arg2 : (!cc.stdvec<f64>) -> !cc.ptr<!cc.array<f64 x ?>>
        %4 = cc.compute_ptr %3[%arg5] : (!cc.ptr<!cc.array<f64 x ?>>, i64) -> !cc.ptr<f64>
        %5 = cc.load %4 : !cc.ptr<f64>
        %6 = cc.stdvec_data %arg3 : (!cc.stdvec<f64>) -> !cc.ptr<!cc.array<f64 x ?>>
        %7 = cc.compute_ptr %6[%arg6] : (!cc.ptr<!cc.array<f64 x ?>>, i64) -> !cc.ptr<f64>
        %8 = cc.load %7 : !cc.ptr<f64>
        %9 = arith.mulf %5, %8 : f64
        %10 = cc.stdvec_data %arg4 : (!cc.stdvec<!cc.charspan>) -> !cc.ptr<!cc.array<!cc.charspan x ?>>
        %11 = cc.compute_ptr %10[%arg6] : (!cc.ptr<!cc.array<!cc.charspan x ?>>, i64) -> !cc.ptr<!cc.charspan>
        %12 = cc.load %11 : !cc.ptr<!cc.charspan>
        quake.exp_pauli %9, %0, %12 : (f64, !quake.veq<?>, !cc.charspan) -> ()
        cc.continue %arg6 : i64
      } step {
      ^bb0(%arg6: i64):
        %3 = arith.addi %arg6, %c1_i64 : i64
        cc.continue %3 : i64
      }
      cc.continue %arg5 : i64
    } step {
    ^bb0(%arg5: i64):
      %2 = arith.addi %arg5, %c1_i64 : i64
      cc.continue %2 : i64
    }
    return
  }
}'''

    # Merge the external code with the current pure device kernel
    newMod = cudaq.jit.mergeExternalMLIRWithKernel(callee, otherMod)
    print(newMod)
    # Synthesize away the callable arg with the pure device kernel 
    cudaq.jit.synthesizeCallableBlockArgument(newMod, 'callee')
    print(newMod)
    assert '!cc.callable' not in str(newMod) 