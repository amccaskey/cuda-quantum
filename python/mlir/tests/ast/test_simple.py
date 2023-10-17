# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
import os

import pytest
import numpy as np

import cudaq

def test_bell():
    @cudaq.kernel(jit=True)
    def bell():
        q = cudaq.qvector(2)
        h(q[0])
        x.ctrl(q[0], q[1])

    print(bell)
    bell()
    counts = cudaq.sample(bell)
    assert '00' in counts and '11' in counts 

def test_ghz():

    @cudaq.kernel(jit=True, verbose=True)
    def ghz(N:int):
        q = cudaq.qvector(N)
        h(q[0])
        for i in range(N-1):
            x.ctrl(q[i], q[i+1])

    print(ghz)

    counts = cudaq.sample(ghz, 5)
    assert '0'*5 in counts and '1'*5 in counts 

def test_no_annotations():
    with pytest.raises(RuntimeError) as error:
        @cudaq.kernel(jit=True, verbose=True)
        def ghz(N):
            q = cudaq.qvector(N)
            h(q[0])
            for i in range(N-1):
                x.ctrl(q[i], q[i+1])
    
def test_kernel_composition():

    @cudaq.kernel(jit=True, verbose=True)
    def iqft(qubits:cudaq.qview):
        N = qubits.size()
        for i in range(N//2):
            swap(qubits[i], qubits[N-i-1])

        for i in range(N-1):
            h(qubits[i])
            j = i + 1
            for y in range(i, -1, -1):
                r1.ctrl(-np.pi/ 2**(j-y), qubits[j], qubits[y])

        h(qubits[N-1])

    print(iqft)

    def testme():
        i = 1 

    @cudaq.kernel(jit=True, verbose=True)
    def entryPoint():
        q = cudaq.qvector(3)
        iqft(q)
    
    print(entryPoint)
    entryPoint()

def test_qreg_iter():
    @cudaq.kernel(jit=True, verbose=True)
    def foo(N:int):
        q = cudaq.qvector(N)
        for r in q:
            x(r)

    foo(10)

def test_control_kernel():
    @cudaq.kernel(jit=True)
    def applyX(q:cudaq.qubit):
        x(q)
    
    @cudaq.kernel(jit=True, verbose=True)
    def bell():
        q = cudaq.qvector(2)
        h(q[0])
        cudaq.control(applyX, [q[0]], q[1])
        cudaq.control(applyX, q[0], q[1])
        cudaq.control(applyX, q[0], q[1])

    print(bell)
    bell()


def test_simple_sampling_qpe():
    """Test that we can build up a set of kernels, compose them, and sample."""
    @cudaq.kernel(jit=True)
    def iqft(qubits:cudaq.qview):
        N = qubits.size()
        for i in range(N//2):
            swap(qubits[i], qubits[N-i-1])

        for i in range(N-1):
            h(qubits[i])
            j = i + 1
            for y in range(i, -1, -1):
                r1.ctrl(-np.pi / 2**(j-y), qubits[j], qubits[y])

        h(qubits[N-1])

    @cudaq.kernel(jit=True)
    def tGate(qubit:cudaq.qubit):
        t(qubit)

    @cudaq.kernel(jit=True)
    def xGate(qubit:cudaq.qubit):
        x(qubit)

    @cudaq.kernel(jit=True, verbose=True)
    def qpe(nC:int, nQ:int):
        q = cudaq.qvector(nC+nQ)
        countingQubits = q.front(nC)
        stateRegister = q.back()
        xGate(stateRegister)
        # Fixme add quantum op broadcast
        for i in range(nC):
            h(countingQubits[i])
        for i in range(nC):
            for j in range(2**i):
                cudaq.control(tGate, [countingQubits[i]], stateRegister)
        iqft(countingQubits)
        # mz(countingQubits)

    print(qpe)
    qpe(3,1)

# TODO adjoint kernels, if stmts, while loop,
#  async, exp_pauli, common kernels