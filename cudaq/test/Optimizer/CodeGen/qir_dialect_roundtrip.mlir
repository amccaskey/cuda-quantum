// RUN: cudaq-opt %s | FileCheck %s
// RUN: cudaq-opt --convert-qir-to-llvm %s | FileCheck %s --check-prefix=LLVMIR

// Round-trip test: emit cudaq::qir ops directly and verify they print/parse
// correctly, then lower to LLVM.

module {
  func.func @bell_qir() {
    // Allocate two qubits
    %q0 = qir.alloc_qubit : !qir.qubit
    %q1 = qir.alloc_qubit : !qir.qubit
    // Bell circuit
    qir.h %q0
    qir.cnot %q0, %q1
    // Measure
    %r0 = qir.measure %q0
    %r1 = qir.measure %q1
    // Release
    qir.release_qubit %q0
    qir.release_qubit %q1
    return
  }

  func.func @rotation_test() {
    %q = qir.alloc_qubit : !qir.qubit
    %pi = arith.constant 3.14159265358979 : f64
    qir.rx %pi, %q
    qir.ry %pi, %q
    qir.rz %pi, %q
    qir.r1 %pi, %q
    %r = qir.measure %q
    qir.release_qubit %q
    return
  }

  func.func @static_resource_test() {
    // Static (Base/Adaptive) qubit and result addressing
    %q = qir.static_qubit 0
    %slot = qir.static_result 0
    qir.h %q
    qir.measure_named %q -> %slot
    qir.record_bool %slot, "m0"
    return
  }
}

// CHECK-LABEL: func.func @bell_qir
// CHECK:         qir.alloc_qubit : !qir.qubit
// CHECK:         qir.h
// CHECK:         qir.cnot
// CHECK:         qir.measure
// CHECK:         qir.release_qubit

// CHECK-LABEL: func.func @static_resource_test
// CHECK:         qir.static_qubit 0
// CHECK:         qir.static_result 0
// CHECK:         qir.measure_named
// CHECK:         qir.record_bool

// LLVMIR-LABEL: func.func @bell_qir
// LLVMIR:         llvm.call @__quantum__rt__qubit_allocate
// LLVMIR:         llvm.call @__quantum__qis__h__body
// LLVMIR:         llvm.call @__quantum__qis__cnot__body
// LLVMIR:         llvm.call @__quantum__qis__mz__alloc
// LLVMIR:         llvm.call @__quantum__rt__qubit_release

// LLVMIR-LABEL: func.func @static_resource_test
// LLVMIR:         llvm.inttoptr
// LLVMIR:         llvm.call @__quantum__qis__h__body
// LLVMIR:         llvm.call @__quantum__qis__mz__body
// LLVMIR:         llvm.call @__quantum__rt__result_record_output
