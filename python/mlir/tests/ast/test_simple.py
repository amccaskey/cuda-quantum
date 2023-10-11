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

# TODO sample / observe async