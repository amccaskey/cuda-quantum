// RUN: cudaq-opt --quake-to-qir-dialect %s | FileCheck %s --check-prefix=QIRDIALECT
// RUN: cudaq-opt --quake-to-qir-dialect --convert-qir-to-llvm %s | FileCheck %s --check-prefix=LLVMIR

// Test: Bell circuit in reference semantics Quake → cudaq::qir dialect → LLVM IR.

module {
  func.func @bell() {
    %q0 = quake.alloca !quake.ref
    %q1 = quake.alloca !quake.ref
    quake.h %q0 : (!quake.ref) -> ()
    quake.x [%q0] %q1 : (!quake.ref, !quake.ref) -> ()
    %m0 = quake.mz %q0 name "r0" : (!quake.ref) -> !quake.measure
    %m1 = quake.mz %q1 name "r1" : (!quake.ref) -> !quake.measure
    quake.dealloc %q0 : !quake.ref
    quake.dealloc %q1 : !quake.ref
    return
  }
}

// QIRDIALECT-LABEL: func.func @bell
// QIRDIALECT:         [[Q0:%.*]] = qir.alloc_qubit : !qir.qubit
// QIRDIALECT:         [[Q1:%.*]] = qir.alloc_qubit : !qir.qubit
// QIRDIALECT:         qir.h [[Q0]]
// QIRDIALECT:         qir.cnot [[Q0]], [[Q1]]
// QIRDIALECT:         {{.*}} = qir.measure [[Q0]]
// QIRDIALECT:         {{.*}} = qir.measure [[Q1]]
// QIRDIALECT:         qir.release_qubit [[Q0]]
// QIRDIALECT:         qir.release_qubit [[Q1]]

// LLVMIR-LABEL: func.func @bell
// LLVMIR:         {{.*}} = llvm.call @__quantum__rt__qubit_allocate
// LLVMIR:         {{.*}} = llvm.call @__quantum__rt__qubit_allocate
// LLVMIR:         llvm.call @__quantum__qis__h__body
// LLVMIR:         llvm.call @__quantum__qis__cnot__body
// LLVMIR:         {{.*}} = llvm.call @__quantum__qis__mz__alloc
// LLVMIR:         {{.*}} = llvm.call @__quantum__qis__mz__alloc
// LLVMIR:         llvm.call @__quantum__rt__qubit_release
// LLVMIR:         llvm.call @__quantum__rt__qubit_release
