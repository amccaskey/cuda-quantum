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

# FIXME implement in eager mode...

@pytest.fixture(autouse=True)
def do_something():
    cudaq.__clearKernelRegistries()
    yield 
    return 

def test_from_state():
    @cudaq.kernel(jit=True)
    def bell():
        q = cudaq.qvector(np.array([1./np.sqrt(2),0,0,1./np.sqrt(2)]))

    print(bell)
    counts = cudaq.sample(bell)
    counts.dump()
    assert '00' in counts and '11' in counts 

def test_from_state_again():
    @cudaq.kernel(jit=True)
    def bell():
        q = cudaq.qvector(2)
        cudaq.initialize_state(q, np.array([1./np.sqrt(2),0,0,1./np.sqrt(2)]))

    print(bell)
    counts = cudaq.sample(bell)
    counts.dump()
    assert '00' in counts and '11' in counts 


def test_from_state_again2():
    @cudaq.kernel(jit=True)
    def bell():
        q = cudaq.qvector(2)
        vector = np.array([1./np.sqrt(2),0,0,1./np.sqrt(2)])
        cudaq.initialize_state(q, vector)

    print(bell)
    counts = cudaq.sample(bell)
    counts.dump()
    assert '00' in counts and '11' in counts 

# WILL TAKE SOME WORK
# def test_from_state_again3():
#     @cudaq.kernel(jit=True)
#     def bell():
#         q = cudaq.qvector(2)
#         vector = 1/np.sqrt(2) * np.array([1,0,0,1])
#         cudaq.initialize_state(q, vector)

#     print(bell)
#     counts = cudaq.sample(bell)
#     counts.dump()
#     assert '00' in counts and '11' in counts 
