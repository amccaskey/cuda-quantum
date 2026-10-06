# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Greedy and exact Tanner-graph edge coloring for CSS schedules."""

import pytest

import cudaq.logical as cql

_HX = ((0,), (0, 1), (1, 2))
_HZ = ((3,), (3, 4), (4, 5))


def _suboptimal_greedy_code():
    """Two disjoint five-edge paths whose greedy coloring uses three colors."""

    return cql.codes.CSSCode(
        name="suboptimal_greedy_coloring",
        n=6,
        k=0,
        hx=_HX,
        hz=_HZ,
    )


def _maximum_degree(checks):
    check_degree = max((len(support) for support in checks), default=0)
    data_degree = {}
    for support in checks:
        for data in support:
            data_degree[data] = data_degree.get(data, 0) + 1
    return max(check_degree, max(data_degree.values(), default=0))


def _assert_optimal_edge_coloring(checks, layers):
    expected = sorted((check, data)
                      for check, support in enumerate(checks)
                      for data in support)
    assert sorted(edge for layer in layers for edge in layer) == expected
    for layer in layers:
        layer_checks = [check for check, _ in layer]
        layer_data = [data for _, data in layer]
        assert len(layer_checks) == len(set(layer_checks))
        assert len(layer_data) == len(set(layer_data))
    assert len(layers) == _maximum_degree(checks)


def test_colored_schedule_defaults_to_the_existing_greedy_algorithm():
    code = _suboptimal_greedy_code()

    implicit = code.colored_schedule()
    explicit = code.colored_schedule(algorithm="greedy")

    assert implicit == explicit
    assert explicit == (
        (
            ((0, 0), (2, 1)),
            ((1, 0), (2, 2)),
            ((1, 1),),
        ),
        (
            ((0, 3), (2, 4)),
            ((1, 3), (2, 5)),
            ((1, 4),),
        ),
    )


def test_exact_colored_schedule_with_no_seed_is_deterministic():
    code = _suboptimal_greedy_code()

    first = code.colored_schedule(algorithm="exact")
    second = code.colored_schedule(algorithm="exact")
    explicit_none = code.colored_schedule(algorithm="exact", rng_seed=None)
    rebuilt = _suboptimal_greedy_code().colored_schedule(algorithm="exact")

    assert first == second == explicit_none == rebuilt
    x_layers, z_layers = first
    _assert_optimal_edge_coloring(code.hx, x_layers)
    _assert_optimal_edge_coloring(code.hz, z_layers)
    assert len(x_layers) == len(z_layers) == 2


def test_seeded_exact_colored_schedule_is_reproducible_varied_and_optimal():
    code = _suboptimal_greedy_code()
    schedules = {}

    for seed in range(16):
        schedule = code.colored_schedule(algorithm="exact", rng_seed=seed)
        assert schedule == code.colored_schedule(algorithm="exact",
                                                 rng_seed=seed)
        assert schedule == _suboptimal_greedy_code().colored_schedule(
            algorithm="exact", rng_seed=seed)
        x_layers, z_layers = schedule
        _assert_optimal_edge_coloring(code.hx, x_layers)
        _assert_optimal_edge_coloring(code.hz, z_layers)
        schedules[schedule] = seed

    assert len(schedules) > 1


@pytest.mark.parametrize("algorithm", (None, "greedy"))
def test_colored_schedule_rejects_rng_seed_with_greedy(algorithm):
    kwargs = {"rng_seed": 0}
    if algorithm is not None:
        kwargs["algorithm"] = algorithm
    with pytest.raises(ValueError, match="rng_seed"):
        _suboptimal_greedy_code().colored_schedule(**kwargs)


@pytest.mark.parametrize("algorithm", (None, "EXACT", "random", 1))
def test_colored_schedule_rejects_an_unknown_algorithm(algorithm):
    with pytest.raises(ValueError, match="algorithm"):
        _suboptimal_greedy_code().colored_schedule(algorithm=algorithm)
