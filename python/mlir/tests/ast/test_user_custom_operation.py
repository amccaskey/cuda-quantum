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


def test_custom_op():

    custom_h = cudaq.register_operation(1./np.sqrt(2.) * np.array([[1, 1], [1, -1]]))
    custom_x = cudaq.register_operation(np.array([[0,1],[1,0]]))

    @cudaq.kernel(jit=True)
    def bell():
        q,r = cudaq.qubit(), cudaq.qubit()
        custom_h(q)
        custom_x.ctrl(q,r)

    counts = cudaq.sample(bell, shots_count=100)
    counts.dump()
    assert '00' in counts and '11' in counts and len(counts) == 2

    # Also support multi-target unitary
    hMat = 1./np.sqrt(2.) * np.array([[1, 1], [1, -1]])
    cxMat = np.array([[1,0,0,0], [0,1,0,0], [0,0,0,1], [0,0,1,0]])
    bellMat = np.dot(cxMat, np.kron(hMat, np.eye(2)))

    custom_bell = cudaq.register_operation(bellMat)

    @cudaq.kernel(jit=True)
    def bell2():
        q,r = cudaq.qubit(), cudaq.qubit()
        custom_bell(q, r)
    
    print(bell2)
    
    counts = cudaq.sample(bell2, shots_count=100)
    counts.dump()
    assert '00' in counts and '11' in counts and len(counts) == 2


    @cudaq.kernel(jit=True)
    def bell3(turnOn:bool):
        q,r,s = cudaq.qubit(), cudaq.qubit(), cudaq.qubit()
        if turnOn:
            x(q)
        custom_bell.ctrl(q, r, s)
    
    counts = cudaq.sample(bell3, True, shots_count=100)
    counts.dump()

    assert '100' in counts and '111' in counts and len(counts) == 2

    counts = cudaq.sample(bell3, False, shots_count=100)
    counts.dump()
    assert '000' in counts and len(counts) == 1