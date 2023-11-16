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


def test_state_vector_simple():
    """
    A simple end-to-end test of the state class on a state vector
    backend. Begins with a kernel, converts to state, then checks
    its member functions.
    """

    @cudaq.kernel
    def bell():
        qubits = cudaq.qvector(2)
        h(qubits[0])
        x.ctrl(qubits[0], qubits[1])

    # Get the quantum state, which should be a vector.
    got_state = cudaq.get_state(bell)

    want_state = np.array([1. / np.sqrt(2.), 0., 0., 1. / np.sqrt(2.)],
                          dtype=np.complex128)

    # Check the indexing operators on the State class
    # while also checking their values
    np.isclose(want_state[0], got_state[0].real)
    np.isclose(want_state[1], got_state[1].real)
    np.isclose(want_state[2], got_state[2].real)
    np.isclose(want_state[3], got_state[3].real)

    # Check the entire vector with numpy.
    got_vector = np.array(got_state, copy=False)
    for i in range(len(want_state)):
        assert np.isclose(want_state[i], got_vector[i])
        # if not np.isclose(got_vector[i], got_vector_b[i]):
        print(f"want = {want_state[i]}")
        print(f"got = {got_vector[i]}")
    assert np.allclose(want_state, np.array(got_state))


def test_state_density_matrix_simple():
    """
    A simple end-to-end test of the state class on a density matrix
    backend. Begins with a kernel, converts to state, then checks
    its member functions.
    """
    cudaq.set_target('density-matrix-cpu')

    # Create the bell state
    @cudaq.kernel
    def bell():
        qubits = cudaq.qvector(2)
        h(qubits[0])
        x.ctrl(qubits[0], qubits[1])

    got_state = cudaq.get_state(bell)
    print(got_state)

    want_state = np.array([[0.5, 0.0, 0.0, 0.5], [0.0, 0.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0, 0.0], [0.5, 0.0, 0.0, 0.5]],
                          dtype=np.complex128)

    # Check the indexing operators on the State class
    # while also checking their values
    np.isclose(.5, got_state[0, 0].real)
    np.isclose(.5, got_state[0, 3].real)
    np.isclose(.5, got_state[3, 0].real)
    np.isclose(.5, got_state[3, 3].real)

    # Check the entire matrix with numpy.
    assert np.allclose(want_state, np.array(got_state))

    cudaq.reset_target()
