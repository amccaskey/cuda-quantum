# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Pinnacle PLAQ Fermi--Hubbard P3 paper-evaluation driver.

This example writes the resource-relevant plaquette-Trotter circuit from
Campbell, Quantum 4, 296 (2020), as a normal ``@qlx.program``.  It then lowers
that program through Pinnacle's real GB-code, WSC, PBC, BRS-RUS, scratch, and
bounded-retry protocols to a physical schedule and a generic Tier-SCHEDULE
estimate.

Two device routes are intentionally available.  ``protocol`` uses the typed
Pinnacle scheduled-macro producer.  ``factory_model`` supplies the same paper
magic engine as an explicitly opaque :class:`qlx.devices.FactoryModel`.  The
compute program and every compute-side protocol are identical between them.

This is a resource-relevant phase-estimation workload, not a state-preparation
or energy-sampling implementation.  The source papers do not specify a concrete
approximate ground-state preparation and cost repeated PLAQ applications rather
than spelling a controlled phase-estimation circuit.  Those limits are reported
instead of being filled with invented operations.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time

import cudaq.logical as qlx


QLX_SOURCE = {
    "module": str(qlx.__file__),
    "version": getattr(qlx, "__version__", "unknown"),
}

from cudaq.mlir import ir as mlir_ir
from cudaq.logical.architectures import pinnacle

# Campbell, arXiv:2012.09238v4, Table III.  The pair is
# (||H_h|| / tau, ||[[H_h^p, H_h^g], H_h^p]|| / tau^3).
_PLAQ_NORMS = {
    4: (24, 0),
    6: (56, 110),
    8: (100, 190),
    10: (160, 300),
    12: (230, 440),
    14: (320, 630),
    16: (410, 810),
    18: (520, 1000),
    20: (650, 1300),
    22: (780, 1600),
    24: (930, 1800),
    26: (1100, 2200),
    28: (1300, 2500),
    30: (1500, 2900),
    32: (1700, 3300),
}
_HOPPING = 1.0
_INTERACTION = 4.0
_RELATIVE_PRECISION = 0.005
_ENERGY_PER_SITE = 1.02
_TARGET_OUTPUT_INFIDELITY = 1.0e-9
_FAILURE_BUDGET = 1.0
_ARCHITECTURE_BY_DISTANCE = {
    4: "gb30",
    6: "gb62",
    10: "gb126",
    16: "gb254",
    24: "gb510",
}
_MATRIX_LATTICES = (4, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32)
_MATRIX_RATES = (1.0e-3, 1.0e-4)
_MIN_AVAILABLE_GIB = {
    4: 12, 8: 12, 10: 12, 12: 12, 14: 12, 16: 12,
    18: 24, 20: 32, 22: 48, 24: 80, 26: 100, 28: 140,
    30: 190, 32: 190,
}
_MATRIX_EXPECTED = {
    (4, 1.0e-3): (10_890, 7.200, 398.5, 218.4, 246.2, 1.127),
    (4, 1.0e-4): (2_400, 7.200, 398.5, 106.6, 129.0, 1.210),
    (8, 1.0e-3): (20_610, 7.396, 408.8, 224.3, 241.7, 1.078),
    (8, 1.0e-4): (6_016, 7.396, 408.8, 109.5, 119.3, 1.090),
    (10, 1.0e-3): (27_090, 7.410, 409.5, 224.7, 255.6, 1.138),
    (10, 1.0e-4): (8_728, 7.410, 409.5, 109.7, 130.0, 1.186),
    (12, 1.0e-3): (36_810, 7.412, 409.7, 224.8, 255.1, 1.135),
    (12, 1.0e-4): (12_344, 7.412, 409.7, 109.7, 134.2, 1.223),
    (14, 1.0e-3): (46_530, 7.432, 410.7, 225.4, 199.0, 0.883),
    (14, 1.0e-4): (15_960, 7.432, 410.7, 110.0, 102.7, 0.934),
    (16, 1.0e-3): (59_490, 7.420, 410.1, 225.0, 294.1, 1.307),
    (16, 1.0e-4): (20_480, 7.420, 410.1, 109.8, 139.9, 1.274),
    (18, 1.0e-3): (72_450, 7.416, 409.9, 224.9, 245.4, 1.091),
    (18, 1.0e-4): (25_904, 7.416, 409.9, 109.8, 128.0, 1.166),
    (20, 1.0e-3): (88_650, 7.432, 410.7, 225.4, 261.2, 1.159),
    (20, 1.0e-4): (31_328, 7.432, 410.7, 110.0, 125.6, 1.142),
    (22, 1.0e-3): (104_850, 7.431, 410.7, 225.4, 216.4, 0.960),
    (22, 1.0e-4): (37_656, 7.431, 410.7, 110.0, 110.3, 1.002),
    (24, 1.0e-3): (124_290, 7.422, 410.2, 225.1, 276.8, 1.230),
    (24, 1.0e-4): (44_888, 7.422, 410.2, 109.8, 143.2, 1.303),
    (26, 1.0e-3): (143_730, 7.433, 410.8, 225.4, 257.5, 1.142),
    (26, 1.0e-4): (52_120, 7.433, 410.8, 110.0, 131.2, 1.192),
    (28, 1.0e-3): (166_410, 7.439, 411.1, 225.6, 269.4, 1.194),
    (28, 1.0e-4): (60_256, 7.439, 411.1, 110.1, 133.2, 1.210),
    (30, 1.0e-3): (189_090, 7.444, 411.3, 225.7, 243.2, 1.077),
    (30, 1.0e-4): (69_296, 7.444, 411.3, 110.2, 127.3, 1.155),
    (32, 1.0e-3): (215_010, 7.442, 411.2, 225.7, 310.6, 1.376),
    (32, 1.0e-4): (78_336, 7.442, 411.2, 110.1, 148.3, 1.346),
}


@dataclass(frozen=True, slots=True)
class PlaquetteProblem:
    """One specialized executable program plus its independent paper facts."""

    definition: qlx.programs.ProgramDefinition
    lattice_size: int
    sites: int
    hopping: float
    interaction: float
    relative_precision: float
    energy_per_site: float
    periodic: bool
    p0_logical_qubits: int
    pinnacle_logical_occupants: int
    phase_steps_continuous: float
    phase_steps: int
    timestep: float
    rotation_precision: float
    synthesis_error_fraction: float
    fermionic_swaps_per_step: int
    exact_clifford_rotations_per_step: int
    direct_t_per_step: int
    rotations_per_step: int
    paper_t_per_rotation: float
    paper_t_states: float
    paper_rus_measurements: float
    paper_logical_cycles: float


@dataclass(frozen=True, slots=True)
class PhysicalStudy:
    """The compiler products and auditable scalar result of one P3 route."""

    factory_mode: str
    clifford_semantics: str
    rus_readout_model: str
    termination_semantics: str
    p_phys: float
    target_output_infidelity: float
    failure_budget: float
    cycle_time_ns: float
    architecture: str
    logical_capacity: int
    processing_blocks: int
    p0_sha256: str
    compute_placement_sha256: str
    compute_selection_sha256: str
    p1_sha256: str
    p2_sha256: str
    p3_sha256: str
    schedule_sha256: str
    factory_service_sha256: str
    schedule_entries: int
    processing_physical_qubits: int
    scratch_physical_qubits: int
    factory_physical_qubits: int
    physical_qubits: int
    paper_physical_qubits: int
    physical_qubit_overhead: int
    factory_acceptance_probability: float
    factory_startup_ns: float
    factory_attempt_interval_ns: float
    factory_output_interval_ns: float
    service_slot_ns: float
    compiler_service_slots_per_step: float
    paper_service_slots_per_step: float
    makespan_ns: float
    expected_makespan_ns: float
    maximum_makespan_ns: float
    exhaustion_probability: float
    paper_expected_logical_cycles: float
    paper_runtime_ns: float
    compiler_to_paper_runtime_ratio: float


@dataclass(frozen=True, slots=True)
class _FactoryServiceEvidence:
    """Normalized scalars read from retained, verifier-closed P3 facts."""

    commitment: str
    acceptance_probability: float
    startup_ns: float
    attempt_interval_ns: float
    output_interval_ns: float
    physical_qubits: int


def _plaq_trotter_bound(lattice_size: int) -> float:
    try:
        hopping_norm, nested_norm = _PLAQ_NORMS[lattice_size]
    except KeyError as error:
        supported = ", ".join(map(str, _PLAQ_NORMS))
        raise ValueError(
            "the paper-calibrated study requires a source-tabulated even "
            f"lattice; supported values are {supported}") from error
    sites = lattice_size * lattice_size
    split_operator = ((_INTERACTION * _HOPPING**2 / 6.0) * sites *
                      (math.sqrt(5.0) + 8.0) +
                      (_INTERACTION**2 * _HOPPING / 24.0) * hopping_norm)
    return split_operator + 3.0 * nested_norm / 24.0


def _paper_terms(lattice_size: int, synthesis_fraction: float):
    sites = lattice_size * lattice_size
    epsilon = _RELATIVE_PRECISION * _ENERGY_PER_SITE * sites
    bound = _plaq_trotter_bound(lattice_size)
    retained = 1.0 - synthesis_fraction
    phase_steps = 6.203 * math.sqrt(bound) / (retained * epsilon)**1.5
    rotations = 4 * sites
    direct_t = 12 * sites
    log_argument = (rotations * math.sqrt(3.0 * bound) /
                    (synthesis_fraction * math.sqrt(retained) * epsilon**1.5))
    t_per_rotation = 1.15 * math.log2(log_argument) + 9.2
    t_states = phase_steps * (direct_t + rotations * t_per_rotation)
    return phase_steps, t_per_rotation, t_states, epsilon, bound


def _optimized_synthesis_fraction(lattice_size: int) -> float:
    """Minimize Eq. (16) without introducing a numerical dependency."""

    lower = 1.0e-12
    upper = 1.0 - 1.0e-12
    ratio = (math.sqrt(5.0) - 1.0) / 2.0

    def objective(value):
        return _paper_terms(lattice_size, value)[2]

    left = upper - ratio * (upper - lower)
    right = lower + ratio * (upper - lower)
    left_value = objective(left)
    right_value = objective(right)
    for _ in range(160):
        if left_value < right_value:
            upper, right, right_value = right, left, left_value
            left = upper - ratio * (upper - lower)
            left_value = objective(left)
        else:
            lower, left, left_value = left, right, right_value
            right = lower + ratio * (upper - lower)
            right_value = objective(right)
    return (lower + upper) / 2.0


def _quarter_turn(qubit, *, adjoint: bool):
    """Apply one exact T or T-dagger as a Pinnacle-selectable RPP site."""

    angle = -qlx.types.pi / 4 if adjoint else qlx.types.pi / 4
    return qlx.ops.rotate(qlx.types.Z(qubit), angle=angle)[0]


def _rotate_ordered(product, operands, *, angle, precision=None):
    """Return rotation owners in caller order, not canonical Pauli order."""

    canonical_inputs = product.operands
    outputs = qlx.ops.rotate(product, angle=angle, precision=precision)
    return tuple(
        next(output
             for source, output in zip(canonical_inputs, outputs)
             if source is requested)
        for requested in operands)


def _clifford_rz(qubit, numerator: int, denominator: int = 2):
    angle = numerator * qlx.types.pi / denominator
    return qlx.ops.rotate(qlx.types.Z(qubit), angle=angle)[0]


def _clifford_rx(qubit, numerator: int, denominator: int = 2):
    angle = numerator * qlx.types.pi / denominator
    return qlx.ops.rotate(qlx.types.X(qubit), angle=angle)[0]


def _clifford_h(qubit):
    # Rz(pi/2) Rx(pi/2) Rz(pi/2) = H up to an irrelevant global phase.
    qubit = _clifford_rz(qubit, 1)
    qubit = _clifford_rx(qubit, 1)
    return _clifford_rz(qubit, 1)


def _clifford_s(qubit, *, adjoint: bool):
    return _clifford_rz(qubit, -1 if adjoint else 1)


def _clifford_x(qubit):
    return _clifford_rx(qubit, 1, 1)


def _clifford_cz(left, right):
    # Rz(pi/2) x Rz(pi/2) x Rzz(-pi/2) is CZ up to global phase.
    left = _clifford_rz(left, 1)
    right = _clifford_rz(right, 1)
    product = qlx.types.Z(left) @ qlx.types.Z(right)
    left, right = _rotate_ordered(
        product,
        (left, right),
        angle=-qlx.types.pi / 2,
    )
    return left, right


def _clifford_cx(control, target):
    target = _clifford_h(target)
    control, target = _clifford_cz(control, target)
    target = _clifford_h(target)
    return control, target


def _v_basis(qubit, *, adjoint: bool):
    """The one-T basis change V with V Z V-dagger = H."""

    qubit = _clifford_s(qubit, adjoint=True)
    qubit = _clifford_h(qubit)
    qubit = _quarter_turn(qubit, adjoint=adjoint)
    qubit = _clifford_h(qubit)
    return _clifford_s(qubit, adjoint=False)


def _controlled_h(control, target):
    # CH = V CZ V-dagger.  In time order that is V-dagger, CZ, V.
    target = _v_basis(target, adjoint=True)
    control, target = _clifford_cz(control, target)
    target = _v_basis(target, adjoint=False)
    return control, target


def _fermionic_fourier(left, right, *, adjoint: bool):
    """The F gate of Campbell Eq. (E11), using exactly two T sites."""

    if not adjoint:
        left, right = _clifford_cx(left, right)
        right, left = _controlled_h(right, left)
        left, right = _clifford_cx(left, right)
        left, right = _clifford_cz(left, right)
    else:
        # Reverse the Hermitian primitive sequence instead of relying on the
        # mathematical coincidence F == F-dagger.
        left, right = _clifford_cz(left, right)
        left, right = _clifford_cx(left, right)
        right, left = _controlled_h(right, left)
        left, right = _clifford_cx(left, right)
    return left, right


def _fermionic_swap(left, right):
    """Adjacent fSWAP from four exact Clifford rotations.

    ``RXX(pi/2) RYY(pi/2) RZ(pi/2) RZ(pi/2)`` is ``-i`` times fSWAP.  Every
    supported gather has even length and is replayed during scatter, so that
    phase cancels exactly.  The compact identity keeps routing explicit
    without expanding every fSWAP into three CX gates and one CZ.
    """

    product = qlx.types.X(left) @ qlx.types.X(right)
    left, right = _rotate_ordered(
        product,
        (left, right),
        angle=qlx.types.pi / 2,
    )
    product = qlx.types.Y(left) @ qlx.types.Y(right)
    left, right = _rotate_ordered(
        product,
        (left, right),
        angle=qlx.types.pi / 2,
    )
    left = _clifford_rz(left, 1)
    right = _clifford_rz(right, 1)
    return left, right


def _plaquette_gather_swaps(mode_count: int, corners):
    """Return the adjacent swaps that gather modes as ``[1, 3, 2, 4]``."""

    target = (corners[0], corners[2], corners[1], corners[3])
    permutation = list(range(mode_count))
    base = min(corners)
    swaps = []
    for offset, mode in enumerate(target):
        source = permutation.index(mode)
        destination = base + offset
        if source < destination:
            raise AssertionError("plaquette gather crossed its fixed prefix")
        while source > destination:
            lower = source - 1
            permutation[lower], permutation[source] = (
                permutation[source],
                permutation[lower],
            )
            swaps.append(lower)
            source -= 1
    if tuple(permutation[base:base + 4]) != target:
        raise AssertionError("plaquette gather produced the wrong mode order")
    return base, tuple(swaps)


def _apply_fermionic_swap(modes, permutation, lower: int) -> None:
    modes[lower], modes[lower + 1] = _fermionic_swap(
        modes[lower],
        modes[lower + 1],
    )
    permutation[lower], permutation[lower + 1] = (
        permutation[lower + 1],
        permutation[lower],
    )


def _plaquette(modes, corners, *, angle: float, precision: float):
    """Apply Campbell Eq. (E10)--(E13) with exact JW routing."""

    target = (corners[0], corners[2], corners[1], corners[3])
    base, swaps = _plaquette_gather_swaps(len(modes), corners)
    if len(swaps) % 2:
        raise AssertionError(
            "even-lattice plaquette routing must have even parity")
    permutation = list(range(len(modes)))
    for lower in swaps:
        _apply_fermionic_swap(modes, permutation, lower)
    if tuple(permutation[base:base + 4]) != target:
        raise AssertionError("fermionic routing did not gather the plaquette")

    one, three, two, four = range(base, base + 4)
    modes[three], modes[one] = _fermionic_fourier(modes[three],
                                                  modes[one],
                                                  adjoint=False)
    modes[two], modes[four] = _fermionic_fourier(modes[two],
                                                 modes[four],
                                                 adjoint=False)
    left, right = modes[two], modes[three]
    product = qlx.types.X(left) @ qlx.types.X(right)
    modes[two], modes[three] = _rotate_ordered(
        product,
        (left, right),
        angle=angle,
        precision=precision,
    )
    left, right = modes[two], modes[three]
    product = qlx.types.Y(left) @ qlx.types.Y(right)
    modes[two], modes[three] = _rotate_ordered(
        product,
        (left, right),
        angle=angle,
        precision=precision,
    )
    modes[two], modes[four] = _fermionic_fourier(modes[two],
                                                 modes[four],
                                                 adjoint=True)
    modes[three], modes[one] = _fermionic_fourier(modes[three],
                                                  modes[one],
                                                  adjoint=True)
    for lower in reversed(swaps):
        _apply_fermionic_swap(modes, permutation, lower)
    if permutation != list(range(len(modes))):
        raise AssertionError("plaquette routing did not restore JW owner order")


def _site(lattice_size: int, x: int, y: int) -> int:
    return (y % lattice_size) * lattice_size + (x % lattice_size)


def _spin_mode(lattice_size: int, site: int, spin: int) -> int:
    return spin * lattice_size * lattice_size + site


def _tile_anchors(lattice_size: int, colour: str):
    offset = 0 if colour == "pink" else 1
    return tuple((x, y)
                 for y in range(offset, lattice_size, 2)
                 for x in range(offset, lattice_size, 2))


def _fermionic_swaps_per_step(lattice_size: int) -> int:
    """Exact gather-plus-scatter count for pink/gold/pink and both spins."""

    return 10 * lattice_size**3 - 10 * lattice_size**2 - 4 * lattice_size


def _tile(modes, lattice_size: int, colour: str, *, angle, precision):
    for x, y in _tile_anchors(lattice_size, colour):
        sites = (
            _site(lattice_size, x, y),
            _site(lattice_size, x + 1, y),
            _site(lattice_size, x + 1, y + 1),
            _site(lattice_size, x, y + 1),
        )
        for spin in (0, 1):
            corners = tuple(
                _spin_mode(lattice_size, site, spin) for site in sites)
            _plaquette(modes, corners, angle=angle, precision=precision)


def _trotter_step(
    values,
    *,
    lattice_size: int,
    timestep: float,
    precision: float,
):
    modes = list(values[:-1])
    phase = values[-1]
    sites = lattice_size * lattice_size

    # Campbell Eq. (6): the chemical-potential-shifted interaction has one
    # ZZ rotation per site.  The known fixed-particle-number energy shift is a
    # classical correction and therefore is not a P0 operation.
    onsite_angle = -0.5 * _INTERACTION * timestep
    for site in range(sites):
        up = _spin_mode(lattice_size, site, 0)
        down = _spin_mode(lattice_size, site, 1)
        left, right = modes[up], modes[down]
        product = qlx.types.Z(left) @ qlx.types.Z(right)
        modes[up], modes[down] = _rotate_ordered(
            product,
            (left, right),
            angle=onsite_angle,
            precision=precision,
        )

    # e^(i t H) uses angle -2t in QLX's exp(-i angle P / 2) convention.
    _tile(
        modes,
        lattice_size,
        "pink",
        angle=-timestep,
        precision=precision,
    )
    _tile(
        modes,
        lattice_size,
        "gold",
        angle=-2.0 * timestep,
        precision=precision,
    )
    _tile(
        modes,
        lattice_size,
        "pink",
        angle=-timestep,
        precision=precision,
    )
    return (*modes, phase)


def plaquette_phase_estimation_problem(
    lattice_size: int = 16,
    *,
    periodic: bool = True,
):
    """Return the paper-specialized pure-QLX PLAQ workload and its oracle."""

    if not isinstance(lattice_size, int) or isinstance(lattice_size, bool):
        raise TypeError("lattice_size must be an even source-tabulated int")
    if not isinstance(periodic, bool):
        raise TypeError("periodic must be a bool")
    if not periodic:
        raise ValueError("the paper-calibrated PLAQ study is periodic")
    if lattice_size not in _PLAQ_NORMS:
        supported = ", ".join(map(str, _PLAQ_NORMS))
        raise ValueError(
            "PLAQ reproduction requires a periodic, even, source-tabulated "
            f"lattice; supported values are {supported}")

    sites = lattice_size * lattice_size
    synthesis_fraction = _optimized_synthesis_fraction(lattice_size)
    phase_continuous, t_per_rotation, paper_t, epsilon, bound = _paper_terms(
        lattice_size, synthesis_fraction)
    phase_steps = math.ceil(phase_continuous)
    retained_error = (1.0 - synthesis_fraction) * epsilon
    timestep = math.sqrt(retained_error / (3.0 * bound))
    rotation_precision = synthesis_fraction * epsilon * timestep / (4 * sites)

    def plaquette_resource_program() -> None:
        modes = qlx.allocate(2 * sites, state=qlx.types.zero, name="modes")
        phase = qlx.prepare_zero()
        phase = _clifford_h(phase)

        # Half filling is a resource-neutral starting fixture.  A scientifically
        # useful approximate ground-state preparation is outside both sources.
        for y in range(lattice_size):
            for x in range(lattice_size):
                spin = (x + y) & 1
                index = _spin_mode(
                    lattice_size,
                    _site(lattice_size, x, y),
                    spin,
                )
                modes[index] = _clifford_x(modes[index])

        carries = qlx.ops.repeat(
            phase_steps,
            carries=(*modes, phase),
            body=lambda _iteration, *values: _trotter_step(
                values,
                lattice_size=lattice_size,
                timestep=timestep,
                precision=rotation_precision,
            ),
        )
        # The paper explicitly neglects the one final phase-estimation
        # measurement relative to millions of inner logical cycles.  Closing
        # the whole packed register at one site also preserves the GB block's
        # linear lifetime contract.
        qlx.discard(carries)

    name = f"pinnacle_plaquette_fermi_hubbard_l{lattice_size}"
    definition = qlx.program(plaquette_resource_program, name=name)
    measurements = phase_continuous * 2.0 * (4 * sites)
    fermionic_swaps = _fermionic_swaps_per_step(lattice_size)
    return PlaquetteProblem(
        definition=definition,
        lattice_size=lattice_size,
        sites=sites,
        hopping=_HOPPING,
        interaction=_INTERACTION,
        relative_precision=_RELATIVE_PRECISION,
        energy_per_site=_ENERGY_PER_SITE,
        periodic=periodic,
        p0_logical_qubits=2 * sites + 1,
        pinnacle_logical_occupants=2 * sites + 2,
        phase_steps_continuous=phase_continuous,
        phase_steps=phase_steps,
        timestep=timestep,
        rotation_precision=rotation_precision,
        synthesis_error_fraction=synthesis_fraction,
        fermionic_swaps_per_step=fermionic_swaps,
        exact_clifford_rotations_per_step=(240 * sites + 4 * fermionic_swaps),
        direct_t_per_step=12 * sites,
        rotations_per_step=4 * sites,
        paper_t_per_rotation=t_per_rotation,
        paper_t_states=paper_t,
        paper_rus_measurements=measurements,
        paper_logical_cycles=paper_t + measurements,
    )


def _architecture_for(
    engine,
    rus_readout_model: pinnacle.RUSReadoutModel = (
        pinnacle.RUSReadoutModel.EXPLICIT_WSC),
):
    if not isinstance(rus_readout_model, pinnacle.RUSReadoutModel):
        raise TypeError("rus_readout_model must be a Pinnacle RUSReadoutModel")
    canonical = getattr(
        pinnacle,
        _ARCHITECTURE_BY_DISTANCE[engine.gb_distance],
    )
    if rus_readout_model is pinnacle.RUSReadoutModel.EXPLICIT_WSC:
        return canonical
    return pinnacle.for_code(
        canonical.encoding,
        rus_readout_model=rus_readout_model,
    )


def pinnacle_fermi_hubbard_device(
    *,
    logical_capacity: int,
    p_phys: float,
    cycle_time_ns: float,
    factory_mode: str,
    rus_readout_model: pinnacle.RUSReadoutModel,
    clifford_semantics: str,
):
    """Build the complete compute/scratch/factory Pinnacle P3 device."""

    profile = os.getenv("QLX_PROFILE_PINNACLE_FERMI_HUBBARD") is not None

    def timed(label, function):
        started = time.perf_counter()
        result = function()
        if profile:
            print(
                f"pinnacle-fermi-hubbard {factory_mode} device-{label} "
                f"{time.perf_counter() - started:.6f}s",
                flush=True,
            )
        return result

    if factory_mode not in {"protocol", "factory_model"}:
        raise ValueError("factory_mode must be 'protocol' or 'factory_model'")
    if not isinstance(rus_readout_model, pinnacle.RUSReadoutModel):
        raise TypeError("rus_readout_model must be a Pinnacle RUSReadoutModel")
    if clifford_semantics not in {"execute", "frame"}:
        raise ValueError("clifford_semantics must be 'execute' or 'frame'")
    if (not isinstance(logical_capacity, int) or
            isinstance(logical_capacity, bool) or logical_capacity <= 0):
        raise TypeError("logical_capacity must be a positive int")
    if not math.isfinite(cycle_time_ns) or cycle_time_ns <= 0.0:
        raise ValueError("cycle_time_ns must be finite and positive")
    engine = timed(
        "magic-engine",
        lambda: pinnacle.magic_engine(
            p_phys=p_phys,
            target_output_infidelity=_TARGET_OUTPUT_INFIDELITY,
        ),
    )
    architecture = _architecture_for(engine, rus_readout_model)
    processing_blocks = math.ceil(logical_capacity / architecture.code.k)
    block = timed("processing-block",
                  lambda: pinnacle.processing_block(architecture))

    builder = qlx.devices.DeviceBuilder(
        f"Pinnacle{architecture.name.upper()}P{p_phys:g}"
        f"{factory_mode.title().replace('_', '')}")
    compute = builder.logical.add_compute(
        capacity=logical_capacity,
        name="compute",
    )
    factory = builder.logical.add_factory(
        produces=qlx.standard.T_STATE,
        via=engine.producer,
        capacity=1,
        buffer_size=1,
        name="magic_engine",
        stream_name="t_states",
    )
    compute_qec = builder.qec.bind(compute, architecture=architecture)
    factory_qec = builder.qec.bind(factory, encoding=qlx.codes.BareQubit)
    actions = qlx.architecture.physical_actions
    native_actions = (
        actions.H,
        actions.S,
        actions.SDG,
        actions.X,
        actions.Z,
        actions.CX,
        actions.CZ,
        actions.RESET,
    )
    capabilities = (
        qlx.architecture.NATIVE_PAULI_PRODUCT_ROTATION,
    )
    native_instruments = (qlx.architecture.physical_instruments.MPP,)
    compute_qubits = builder.physical.add_qubits(
        processing_blocks * block.physical_qubits,
        name="processing_blocks",
        native_actions=native_actions,
        native_instruments=native_instruments,
        capabilities=capabilities,
    )
    scratch_qubits = builder.physical.add_qubits(
        block.physical_qubits,
        name="rus_scratch_block",
        native_actions=native_actions,
        native_instruments=native_instruments,
        capabilities=capabilities,
    )
    factory_qubits = builder.physical.add_qubits(
        engine.physical_qubits,
        name="magic_engine_qubits",
    )
    builder.physical.bind(compute_qec, to=compute_qubits)
    (scratch_qec,) = compute_qec.auxiliary_regions
    builder.physical.bind(scratch_qec, to=scratch_qubits)
    if factory_mode == "factory_model":
        accepted_cadence = (engine.cycles_per_attempt /
                            engine.acceptance_probability)
        model = qlx.devices.FactoryModel(
            # The compact one-buffer model owns one finite startup before its
            # first accepted output.  Thereafter the scheduler may advance the
            # next output during compute gaps.  The protocol route retains
            # explicit on-demand scheduled-macro requests instead, so equal
            # cadence and footprint do not imply identical whole-program
            # makespans.  The report exposes that abstraction delta.
            startup_cycles=accepted_cadence,
            output_interval_cycles=accepted_cadence,
            evidence=qlx.analysis.user_assertion(
                "Pinnacle arXiv:2602.11457v2 Section V.B.2, Eqs. "
                "(4)-(11): one accepted T-state cadence including the "
                "published rejection probability"),
        )
        builder.physical.bind(
            factory_qec,
            to=factory_qubits,
            factory_model=model,
        )
    else:
        builder.physical.bind(factory_qec, to=factory_qubits)
    timing = {
        "cycle_ns": cycle_time_ns,
        "surface_cycle_ns": cycle_time_ns,
        "condition_ns": 10.0 * cycle_time_ns,
        "decode_bit_ns": 10.0 * cycle_time_ns,
    }
    if rus_readout_model is pinnacle.RUSReadoutModel.PAPER_LOGICAL_CYCLE:
        timing["mpp_ns"] = max(
            (architecture.code.d.conservative_value + 2) * cycle_time_ns,
            engine.cycles_per_attempt * cycle_time_ns,
        )
    if clifford_semantics == "frame":
        # P0 frame normalization conjugates and removes authored Cliffords.
        # Exact Clifford actions introduced later by selected P2 protocols
        # remain typed schedule events, but this operating point prices them
        # as controller-side frame updates rather than quantum execution.
        timing.update({
            f"{action}_ns": 0.0
            for action in ("rpp", "h", "s", "sdg", "x", "z", "cx", "cz")
        })
    builder.physical.set_operating_point(
        timing=timing,
        calibration={
            "physical_error": p_phys,
            "magic_engine_output_infidelity": engine.output_infidelity,
        },
        name="paper_operating_point",
    )
    return (
        timed("finalize", builder.build),
        architecture,
        engine,
        processing_blocks,
    )


def paper_physical_qubits(problem: PlaquetteProblem, *, p_phys: float) -> int:
    engine = pinnacle.magic_engine(
        p_phys=p_phys,
        target_output_infidelity=_TARGET_OUTPUT_INFIDELITY,
    )
    architecture = _architecture_for(engine)
    blocks = math.ceil(problem.pinnacle_logical_occupants / architecture.code.k)
    return (blocks * pinnacle.processing_block(architecture).physical_qubits +
            engine.physical_qubits)


def _paper_expected_logical_cycles(
    problem: PlaquetteProblem,
    *,
    factory_acceptance: float,
) -> float:
    """Apply magic-engine rejection only to T-state deliveries."""

    if (not math.isfinite(factory_acceptance) or
            not 0.0 < factory_acceptance <= 1.0):
        raise ValueError("factory_acceptance must be finite and in (0, 1]")
    return (problem.paper_t_states / factory_acceptance +
            problem.paper_rus_measurements)


def _route_neutral_qec_selection_sha256(p2) -> str:
    """Commit every field of the route-independent typed P2 witness."""

    selection = p2.qec_selection
    if selection is None:
        raise RuntimeError("P2 route has no QEC selection witness")
    payload = json.dumps(
        asdict(selection),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(b"qlx.pinnacle-compute-selection/v2\0")
    digest.update(payload)
    return "sha256:" + digest.hexdigest()


def _route_neutral_placement_sha256(p1) -> str:
    """Commit the full P1 witness while normalizing the route device name."""

    placement = p1.placement
    if placement is None:
        raise RuntimeError("P1 route has no placement witness")
    payload = asdict(placement)
    # The two comparison devices intentionally have different symbols because
    # their factory bindings differ.  Machine identity is consequently the
    # only route-specific P1 field; every placement, slot, source path,
    # objective, and tie-break remains committed verbatim.
    payload.pop("machine")
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(b"qlx.pinnacle-compute-placement/v1\0")
    digest.update(encoded)
    return "sha256:" + digest.hexdigest()


def _walk_ir(operation):
    """Walk one structured MLIR operation without printing or reparsing it."""

    yield operation
    for region in operation.regions:
        for block in region.blocks:
            for child in block.operations:
                yield from _walk_ir(child.operation)


def _string_attribute(attribute, *, field: str) -> str:
    if not isinstance(attribute, mlir_ir.StringAttr):
        raise RuntimeError(f"P3 {field} must be a string attribute")
    return attribute.value


def _flat_symbol_name(attribute, *, field: str) -> str:
    if not isinstance(attribute, mlir_ir.FlatSymbolRefAttr):
        raise RuntimeError(
            f"P3 {field} must be an unqualified flat symbol reference")
    return attribute.value


def _provider_factory_facts(p3, provider_attribute) -> dict[str, object]:
    provider = p3.definitions.get(_flat_symbol_name(
        provider_attribute, field="factory provider"))
    if provider is None or provider.kind != "fabric.protocol":
        raise RuntimeError("P3 factory evidence names no retained provider")
    metadata = provider.op.attributes.get("metadata")
    if metadata is None:
        raise RuntimeError("P3 factory provider has no retained metadata")
    try:
        return {
            "provider":
                provider.symbol,
            "cycles_per_attempt":
                float(_string_attribute(
                    metadata["cycles_per_attempt"],
                    field="factory cycles_per_attempt")),
            "acceptance_probability":
                float(_string_attribute(
                    metadata["acceptance_probability"],
                    field="factory acceptance_probability")),
            "pipeline_depth":
                int(_string_attribute(
                    metadata["pipeline_depth"],
                    field="factory pipeline_depth")),
            "physical_qubits":
                int(_string_attribute(
                    metadata["physical_qubits"],
                    field="factory physical_qubits")),
        }
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(
            "P3 factory provider has incomplete scheduled-macro evidence"
        ) from error


def _p3_factory_service_evidence(
    p3,
    *,
    factory_mode: str,
) -> _FactoryServiceEvidence:
    """Read normalized factory service facts from the retained P3 graph."""

    graphs = tuple(value for value in p3.definitions.values()
                   if value.kind == "phys.graph")
    if len(graphs) != 1:
        raise RuntimeError("P3 route must retain exactly one physical graph")
    requests = tuple(operation for operation in _walk_ir(graphs[0].op)
                     if operation.name == "phys.resource_request" and
                     "provider" in operation.attributes)
    if not requests:
        raise RuntimeError("P3 route has no backed factory request")

    if factory_mode == "protocol":
        records = []
        for request in requests:
            if "factory_model" in request.attributes:
                raise RuntimeError(
                    "protocol route unexpectedly retained a FactoryModel")
            provider = _provider_factory_facts(p3,
                                               request.attributes["provider"])
            record = {
                **provider,
                "attempt_interval_ns":
                    float(request.attributes["factory_attempt_duration_ns"]),
                "output_interval_ns":
                    float(request.attributes["duration_ns"]),
                "factory_mode":
                    _string_attribute(
                        request.attributes["factory_mode"],
                        field="factory request mode"),
            }
            records.append(record)
        canonical = {
            json.dumps(record, sort_keys=True, separators=(",", ":"))
            for record in records
        }
        if len(canonical) != 1:
            raise RuntimeError(
                "protocol route retained inconsistent factory request evidence")
        record = records[0]
        if record["factory_mode"] != "scheduled_macro":
            raise RuntimeError("protocol route is not a scheduled macro")
        if record["pipeline_depth"] != 1:
            raise RuntimeError("paper comparison requires one factory lane")
        startup_ns = 0.0
        commitment_record = {
            "schema": "qlx.pinnacle-factory-service/protocol-v1",
            "request_count": len(requests),
            **record,
        }
    elif factory_mode == "factory_model":
        models = tuple(value for value in p3.definitions.values()
                       if value.kind == "phys.factory_model")
        if len(models) != 1:
            raise RuntimeError(
                "FactoryModel route must retain exactly one physical model")
        model = models[0]
        attributes = model.op.attributes
        if any("factory_model" not in request.attributes
               for request in requests):
            raise RuntimeError(
                "FactoryModel route has a provider-backed request without "
                "the retained model")
        references = {
            _flat_symbol_name(
                request.attributes["factory_model"],
                field="factory request model")
            for request in requests
        }
        if references != {model.symbol} or len(references) != 1:
            raise RuntimeError(
                "FactoryModel requests do not all reference the retained model")
        provider = _provider_factory_facts(p3, attributes["provider"])
        startup_ns = float(attributes["startup_ns"])
        output_interval_ns = float(attributes["output_interval_ns"])
        physical_qubits = int(attributes["physical_units"])
        if physical_qubits != provider["physical_qubits"]:
            raise RuntimeError(
                "FactoryModel footprint differs from its retained provider")
        if provider["pipeline_depth"] != 1:
            raise RuntimeError("paper comparison requires one factory lane")
        record = {
            **provider,
            "attempt_interval_ns":
                output_interval_ns * provider["acceptance_probability"],
            "output_interval_ns":
                output_interval_ns,
            "physical_qubits":
                physical_qubits,
        }
        commitment_record = {
            "schema": "qlx.pinnacle-factory-service/opaque-v1",
            "model": model.symbol,
            "request_count": len(requests),
            "startup_ns": startup_ns,
            **record,
        }
    else:
        raise ValueError("factory_mode must be 'protocol' or 'factory_model'")

    encoded = json.dumps(
        commitment_record,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _FactoryServiceEvidence(
        commitment="sha256:" + hashlib.sha256(encoded).hexdigest(),
        acceptance_probability=float(record["acceptance_probability"]),
        startup_ns=startup_ns,
        attempt_interval_ns=float(record["attempt_interval_ns"]),
        output_interval_ns=float(record["output_interval_ns"]),
        physical_qubits=int(record["physical_qubits"]),
    )


def compile_physical_study(
    problem: PlaquetteProblem,
    *,
    p0=None,
    logical_capacity: int | None = None,
    p_phys: float,
    cycle_time_ns: float,
    factory_mode: str,
    clifford_semantics: str = "frame",
    rus_readout_model: pinnacle.RUSReadoutModel = (
        pinnacle.RUSReadoutModel.PAPER_LOGICAL_CYCLE),
    termination: qlx.estimate.ScheduleTermination = (
        qlx.estimate.ScheduleTermination.FULL_WORKLOAD),
) -> PhysicalStudy:
    """Run the normal P0 -> P1 -> P2 -> P3 -> schedule -> estimate path."""

    if clifford_semantics not in {"execute", "frame"}:
        raise ValueError("clifford_semantics must be 'execute' or 'frame'")
    if not isinstance(rus_readout_model, pinnacle.RUSReadoutModel):
        raise TypeError("rus_readout_model must be a Pinnacle RUSReadoutModel")
    if not isinstance(termination, qlx.estimate.ScheduleTermination):
        raise TypeError("termination must be a ScheduleTermination value")

    profile = os.getenv("QLX_PROFILE_PINNACLE_FERMI_HUBBARD") is not None

    def timed(label, function):
        started = time.perf_counter()
        result = function()
        if profile:
            print(
                f"pinnacle-fermi-hubbard {factory_mode} {label} "
                f"{time.perf_counter() - started:.6f}s",
                flush=True,
            )
        return result

    if p0 is None:
        raw_p0 = timed("p0", lambda: qlx.compile(problem.definition))
        p0 = (timed(
            "clifford-frame",
            lambda: qlx.compiler.absorb_clifford_frame(raw_p0),
        ) if clifford_semantics == "frame" else raw_p0)
    frame_evidence = tuple(item for item in p0.evidence
                           if item.kind == "clifford_frame_normalization")
    if clifford_semantics == "frame" and len(frame_evidence) != 1:
        raise ValueError(
            "frame semantics requires one authenticated Clifford-frame P0")
    if clifford_semantics == "execute" and frame_evidence:
        raise ValueError(
            "execute semantics requires the authored P0 before frame absorption"
        )
    if logical_capacity is None:
        logical_capacity = timed(
            "logical-estimate",
            lambda: qlx.estimate(
                p0,
                tier=qlx.estimate.Tier.LOGICAL,
            ).logical_qubits_peak,
        )
    device, architecture, engine, processing_blocks = timed(
        "device",
        lambda: pinnacle_fermi_hubbard_device(
            logical_capacity=logical_capacity,
            p_phys=p_phys,
            cycle_time_ns=cycle_time_ns,
            factory_mode=factory_mode,
            rus_readout_model=rus_readout_model,
            clifford_semantics=clifford_semantics,
        ),
    )
    p1 = timed("p1-place", lambda: qlx.compiler.place(p0, device=device))
    p2 = timed(
        "p2-qec",
        lambda: qlx.compile(
            p1,
            pipeline=qlx.compiler.pipelines.qec(),
            device=device,
        ),
    )
    p3 = timed(
        "p3-physical",
        lambda: qlx.compile(
            p2,
            pipeline=qlx.compiler.pipelines.physical(),
            device=device,
        ),
    )
    estimate = timed(
        "fused-schedule-estimate",
        lambda: qlx.estimate(
            p3,
            tier=qlx.estimate.Tier.SCHEDULE,
            p_phys=p_phys,
            failure_budget=_FAILURE_BUDGET,
            termination=termination,
        ),
    )
    factory_service = timed(
        "factory-service-evidence",
        lambda: _p3_factory_service_evidence(
            p3,
            factory_mode=factory_mode,
        ),
    )
    # The estimate-only path keeps schedule verification and estimation in one
    # native module.  Re-estimating a portable paper-scale PhysicalSchedule
    # would clone its full row artifact and rematerialize the analytical lower
    # tier, which is substantially slower and holds both giant modules alive.
    # Materialize the independently authenticated portable schedule only after
    # the scalar estimate has been reduced.
    schedule = timed("schedule-artifact", lambda: qlx.compiler.schedule(p3))
    if estimate.event_count != len(schedule.entries):
        raise RuntimeError(
            "fused estimate event count differs from the retained schedule")
    if estimate.makespan_ns != schedule.makespan_ns:
        raise RuntimeError(
            "fused estimate makespan differs from the retained schedule")
    factory_attempt_interval_ns = factory_service.attempt_interval_ns
    expected_attempt_interval_ns = engine.cycles_per_attempt * cycle_time_ns
    if not math.isclose(
            factory_attempt_interval_ns,
            expected_attempt_interval_ns,
            rel_tol=0.0,
            abs_tol=1.0e-9,
    ):
        raise RuntimeError(
            "retained P3 factory attempt cadence differs from the selected "
            "Pinnacle engine")
    logical_cycle_ns = max(
        (architecture.code.d.conservative_value + 2) * cycle_time_ns,
        factory_attempt_interval_ns,
    )
    factory_interval_ns = factory_service.output_interval_ns
    factory_startup_ns = factory_service.startup_ns
    paper_qubits = paper_physical_qubits(problem, p_phys=p_phys)
    paper_expected_cycles = _paper_expected_logical_cycles(
        problem,
        factory_acceptance=factory_service.acceptance_probability,
    )
    paper_runtime_ns = paper_expected_cycles * logical_cycle_ns
    p0_sha256 = timed("p0-commitment", lambda: p0.content_sha256)
    compute_placement_sha256 = timed(
        "compute-placement-commitment",
        lambda: _route_neutral_placement_sha256(p1),
    )
    compute_selection_sha256 = timed(
        "compute-selection-commitment",
        lambda: _route_neutral_qec_selection_sha256(p2),
    )
    p1_sha256 = timed("p1-commitment", lambda: p1.content_sha256)
    p2_sha256 = timed("p2-commitment", lambda: p2.content_sha256)
    p3_sha256 = timed("p3-commitment", lambda: p3.content_sha256)
    schedule_sha256 = timed(
        "schedule-commitment",
        lambda: schedule.build.content_sha256,
    )
    block_qubits = pinnacle.processing_block(architecture).physical_qubits
    processing_physical_qubits = processing_blocks * block_qubits
    scratch_physical_qubits = block_qubits
    factory_physical_qubits = factory_service.physical_qubits
    if (processing_physical_qubits + scratch_physical_qubits +
            factory_physical_qubits != estimate.physical_qubits):
        raise RuntimeError(
            "P3 physical footprint differs from route provisioning components")
    return PhysicalStudy(
        factory_mode=factory_mode,
        clifford_semantics=clifford_semantics,
        rus_readout_model=rus_readout_model.value,
        termination_semantics=termination.value,
        p_phys=p_phys,
        target_output_infidelity=_TARGET_OUTPUT_INFIDELITY,
        failure_budget=_FAILURE_BUDGET,
        cycle_time_ns=cycle_time_ns,
        architecture=architecture.name,
        logical_capacity=logical_capacity,
        processing_blocks=processing_blocks,
        p0_sha256=p0_sha256,
        compute_placement_sha256=compute_placement_sha256,
        compute_selection_sha256=compute_selection_sha256,
        p1_sha256=p1_sha256,
        p2_sha256=p2_sha256,
        p3_sha256=p3_sha256,
        schedule_sha256=schedule_sha256,
        factory_service_sha256=factory_service.commitment,
        schedule_entries=len(schedule.entries),
        processing_physical_qubits=processing_physical_qubits,
        scratch_physical_qubits=scratch_physical_qubits,
        factory_physical_qubits=factory_physical_qubits,
        physical_qubits=estimate.physical_qubits,
        paper_physical_qubits=paper_qubits,
        physical_qubit_overhead=estimate.physical_qubits - paper_qubits,
        factory_acceptance_probability=(factory_service.acceptance_probability),
        factory_startup_ns=factory_startup_ns,
        factory_attempt_interval_ns=factory_attempt_interval_ns,
        factory_output_interval_ns=factory_interval_ns,
        service_slot_ns=logical_cycle_ns,
        compiler_service_slots_per_step=(
            (estimate.expected_makespan_ns - factory_startup_ns) /
            problem.phase_steps / logical_cycle_ns),
        paper_service_slots_per_step=(paper_expected_cycles /
                                      problem.phase_steps_continuous),
        makespan_ns=estimate.makespan_ns,
        expected_makespan_ns=estimate.expected_makespan_ns,
        maximum_makespan_ns=estimate.maximum_makespan_ns,
        exhaustion_probability=estimate.exhaustion_probability,
        paper_expected_logical_cycles=paper_expected_cycles,
        paper_runtime_ns=paper_runtime_ns,
        compiler_to_paper_runtime_ratio=(estimate.expected_makespan_ns /
                                         paper_runtime_ns),
    )


def _hours(nanoseconds: float) -> float:
    return nanoseconds / 1.0e9 / 3600.0


def _print_result(problem: PlaquetteProblem, study: PhysicalStudy) -> None:
    print(f"L={problem.lattice_size}, {study.factory_mode}, "
          f"Cliffords={study.clifford_semantics}, "
          f"RUS-readout={study.rus_readout_model}, "
          f"termination={study.termination_semantics}, "
          f"p={study.p_phys:g}, {study.cycle_time_ns:g} ns code cycle")
    print(f"  PLAQ repeat: ceil({problem.phase_steps_continuous:.6f}) "
          f"= {problem.phase_steps:,} steps")
    print(f"  one step: {problem.fermionic_swaps_per_step:,} fSWAP + "
          f"{problem.direct_t_per_step:,} exact T/Tdg + "
          f"{problem.rotations_per_step:,} arbitrary rotations")
    print(f"  P3 schedule: {study.schedule_entries:,} entries, "
          f"{study.physical_qubits:,} provisioned qubits")
    print(f"  compiled P0 peak: {study.logical_capacity:,} logical owners in "
          f"{study.processing_blocks:,} processing blocks")
    print(f"  paper-packed footprint: {study.paper_physical_qubits:,} qubits "
          f"({study.physical_qubit_overhead:+,} compiler overhead)")
    print("  compiler runtime first/expected/max: "
          f"{_hours(study.makespan_ns):.6g} / "
          f"{_hours(study.expected_makespan_ns):.6g} / "
          f"{_hours(study.maximum_makespan_ns):.6g} h")
    print(f"  independent Pinnacle Eq. (17) runtime: "
          f"{_hours(study.paper_runtime_ns):.6g} h")
    print("  compiler/paper expected-runtime ratio: "
          f"{study.compiler_to_paper_runtime_ratio:.6g}")
    print(f"  factory startup/attempt/accepted cadence: "
          f"{study.factory_startup_ns:g} / "
          f"{study.factory_attempt_interval_ns:g} / "
          f"{study.factory_output_interval_ns:g} ns")
    print("  compiler/paper service slots per step: "
          f"{study.compiler_service_slots_per_step:.6g} / "
          f"{study.paper_service_slots_per_step:.6g}")


def _problem_record(problem: PlaquetteProblem) -> dict[str, object]:
    """Return scalar problem facts without copying the program definition."""

    return {
        field.name: getattr(problem, field.name)
        for field in fields(problem)
        if field.name != "definition"
    }


def _result_payload(
    problem: PlaquetteProblem,
    studies: list[PhysicalStudy],
) -> dict[str, object]:
    return {
        "schema": "qlx.pinnacle-fermi-hubbard-p3/v2",
        "source": QLX_SOURCE,
        "study_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "problem": _problem_record(problem),
        "studies": [asdict(study) for study in studies],
    }


def _write_json(
    destination: Path,
    problem: PlaquetteProblem,
    studies: list[PhysicalStudy],
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            _result_payload(problem, studies),
            indent=2,
            sort_keys=True,
        ) + "\n")
    os.replace(temporary, destination)


_SCIENTIFIC_ARTIFACT_NAMES = (
    "pinnacle_fermi_hubbard_p3.json",
    "pinnacle_fermi_hubbard_p3.csv",
    "physical_qubits_log.png",
    "runtime_hours.png",
)
_PUBLIC_BUNDLE_NAMES = _SCIENTIFIC_ARTIFACT_NAMES


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_artifact_bundle(bundle: Path) -> None:
    entries = {entry.name for entry in bundle.iterdir()}
    expected = set(_PUBLIC_BUNDLE_NAMES)
    if entries != expected:
        raise RuntimeError("artifact bundle is incomplete")
    if any(not (bundle / name).is_file() or (bundle / name).is_symlink()
           for name in expected):
        raise RuntimeError("artifact bundle must contain direct files")
    payload = json.loads(
        (bundle / "pinnacle_fermi_hubbard_p3.json").read_text())
    if payload.get("schema") != "qlx.pinnacle-fermi-hubbard-p3/v2":
        raise RuntimeError("artifact bundle has an unsupported result schema")
    if not (bundle / "pinnacle_fermi_hubbard_p3.csv").read_bytes():
        raise RuntimeError("artifact bundle has an empty CSV result")
    for name in ("physical_qubits_log.png", "runtime_hours.png"):
        if not (bundle / name).read_bytes().startswith(b"\x89PNG\r\n\x1a\n"):
            raise RuntimeError(f"artifact bundle has an invalid PNG {name!r}")


def _validate_replaceable_output_dir(output_dir: Path) -> None:
    """Fail closed before replacing anything except our flat result bundle."""

    if output_dir.is_symlink() or (output_dir.exists() and
                                   not output_dir.is_dir()):
        raise RuntimeError("artifact output path must be a real directory")
    if not output_dir.exists():
        return
    entries = {entry.name for entry in output_dir.iterdir()}
    if not entries:
        return
    expected = set(_PUBLIC_BUNDLE_NAMES)
    if entries != expected:
        raise RuntimeError(
            "artifact output directory must be empty or contain exactly the "
            f"managed flat bundle; found {sorted(entries)!r}")
    if any((output_dir / name).is_symlink() for name in expected):
        raise RuntimeError("artifact output bundle must contain direct files")
    _validate_artifact_bundle(output_dir)


def _publish_artifact_bundle(
    output_dir: Path,
    transaction: Path,
    staging: Path,
) -> None:
    """Replace one flat bundle, restoring the previous directory on failure."""

    _validate_artifact_bundle(staging)
    for name in _PUBLIC_BUNDLE_NAMES:
        with (staging / name).open("rb") as stream:
            os.fsync(stream.fileno())
    _fsync_directory(staging)
    _validate_replaceable_output_dir(output_dir)

    previous = transaction / "previous"
    failed = transaction / "failed"
    had_previous = output_dir.exists()
    if had_previous:
        os.replace(output_dir, previous)
    installed = False
    try:
        os.replace(staging, output_dir)
        installed = True
        _fsync_directory(output_dir.parent)
    except Exception as publish_error:
        try:
            if installed and output_dir.exists():
                os.replace(output_dir, failed)
            if had_previous and previous.exists():
                os.replace(previous, output_dir)
            _fsync_directory(output_dir.parent)
        except OSError as rollback_error:
            raise RuntimeError(
                "artifact publication failed and the previous flat bundle "
                "could not be restored") from rollback_error
        raise publish_error


def _write_result_artifacts(
    output_dir: Path,
    problem: PlaquetteProblem,
    studies: list[PhysicalStudy],
) -> None:
    """Render, validate, and transactionally replace one flat result bundle."""

    _validate_replaceable_output_dir(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    transaction = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}-transaction-",
            dir=output_dir.parent,
        ))
    staging = transaction / "next"
    staging.mkdir()
    try:
        _write_json(staging / "pinnacle_fermi_hubbard_p3.json", problem,
                    studies)

        rows = []
        problem_record = _problem_record(problem)
        for study in studies:
            rows.append({**problem_record, **asdict(study)})
        if rows:
            with (staging / "pinnacle_fermi_hubbard_p3.csv").open(
                    "w", newline="") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=tuple(rows[0]),
                    lineterminator="\n",
                )
                writer.writeheader()
                writer.writerows(rows)

        # Import plotting only when artifacts were requested; ordinary
        # compilation does not require matplotlib.
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/qlx-matplotlib")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        labels = [study.factory_mode for study in studies]
        qubits = [study.physical_qubits for study in studies]
        if studies:
            labels.append("paper packed")
            qubits.append(studies[0].paper_physical_qubits)
        figure, axis = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
        bars = axis.bar(
            labels,
            [value - 1 for value in qubits],
            bottom=1,
            color=("#76B900", "#4C78A8", "#888888")[:len(labels)],
        )
        axis.bar_label(bars,
                       labels=[f"{value:,}" for value in qubits],
                       padding=3)
        axis.set_yscale("log")
        upper = 10**math.ceil(math.log10(max(qubits) * 1.25))
        axis.set_ylim(1, upper)
        axis.set_ylabel("Physical qubits (log scale)")
        axis.set_title(
            f"Pinnacle PLAQ Fermi--Hubbard, L={problem.lattice_size}")
        axis.grid(axis="y", which="both", alpha=0.25)
        figure.savefig(staging / "physical_qubits_log.png", dpi=180)
        plt.close(figure)

        runtime_labels = [study.factory_mode for study in studies]
        runtime_hours = [
            _hours(study.expected_makespan_ns) for study in studies
        ]
        if studies:
            runtime_labels.append("paper Eq. (17)")
            runtime_hours.append(_hours(studies[0].paper_runtime_ns))
        figure, axis = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
        bars = axis.bar(
            runtime_labels,
            runtime_hours,
            color=("#76B900", "#4C78A8", "#888888")[:len(runtime_labels)],
        )
        axis.bar_label(
            bars,
            labels=[f"{value:.6f}" for value in runtime_hours],
            padding=3,
        )
        axis.set_ylabel("Expected runtime per shot (hours)")
        axis.set_title(f"Pinnacle PLAQ P3 schedule, L={problem.lattice_size}")
        axis.grid(axis="y", alpha=0.25)
        figure.savefig(staging / "runtime_hours.png", dpi=180)
        plt.close(figure)

        _publish_artifact_bundle(output_dir, transaction, staging)
    finally:
        if transaction.exists():
            shutil.rmtree(transaction)


def _validated_factory_routes(
    first: PhysicalStudy,
    second: PhysicalStudy,
) -> tuple[PhysicalStudy, PhysicalStudy]:
    routes = {first.factory_mode: first, second.factory_mode: second}
    if set(routes) != {"protocol", "factory_model"}:
        raise RuntimeError(
            "factory-route comparison requires protocol and factory_model")
    protocol = routes["protocol"]
    opaque = routes["factory_model"]
    exact_invariants = (
        "p0_sha256",
        "compute_placement_sha256",
        "compute_selection_sha256",
        "clifford_semantics",
        "rus_readout_model",
        "termination_semantics",
        "architecture",
        "logical_capacity",
        "processing_blocks",
        "processing_physical_qubits",
        "scratch_physical_qubits",
        "factory_physical_qubits",
        "physical_qubits",
        "paper_physical_qubits",
        "p_phys",
        "target_output_infidelity",
        "failure_budget",
        "cycle_time_ns",
        "factory_acceptance_probability",
        "exhaustion_probability",
    )
    for name in exact_invariants:
        if getattr(protocol, name) != getattr(opaque, name):
            raise RuntimeError(f"factory routes changed route invariant {name}")
    floating_invariants = (
        "factory_attempt_interval_ns",
        "factory_output_interval_ns",
        "service_slot_ns",
    )
    for name in floating_invariants:
        if not math.isclose(
                getattr(protocol, name),
                getattr(opaque, name),
                rel_tol=0.0,
                abs_tol=1.0e-9,
        ):
            raise RuntimeError(f"factory routes changed route invariant {name}")
    return protocol, opaque


def _available_gib() -> float | None:
    path = Path("/proc/meminfo")
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / (1024 * 1024)
    return None


def _matrix_name(lattice: int, rate: float) -> str:
    return f"l{lattice}_p{rate:g}.json"


def _validate_matrix_point(
    path: Path, lattice: int, rate: float
) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "qlx.pinnacle-fermi-hubbard-p3/v2":
        raise RuntimeError(f"{path} has an unsupported result schema")
    problem = payload.get("problem", {})
    studies = payload.get("studies", [])
    if problem.get("lattice_size") != lattice or len(studies) != 1:
        raise RuntimeError(f"{path} does not contain one requested matrix point")
    study = studies[0]
    physical, t_millions, rus_thousands, analytical_s, p3_s, ratio = (
        _MATRIX_EXPECTED[(lattice, rate)]
    )
    checks = {
        "source": payload.get("source") == QLX_SOURCE,
        "study": payload.get("study_sha256")
        == hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "p_phys": math.isclose(study["p_phys"], rate),
        "physical_qubits": study["physical_qubits"] == physical,
        "paper_t_states_millions": (
            round(problem["paper_t_states"] / 1e6, 3) == t_millions
        ),
        "paper_rus_thousands": (
            round(problem["paper_rus_measurements"] / 1e3, 1)
            == rus_thousands
        ),
        "analytical_seconds": (
            round(study["paper_runtime_ns"] / 1e9, 1) == analytical_s
        ),
        "p3_seconds": (
            round(study["expected_makespan_ns"] / 1e9, 1) == p3_s
        ),
        "ratio": (
            round(study["compiler_to_paper_runtime_ratio"], 3) == ratio
        ),
        "factory_mode": study.get("factory_mode") == "protocol",
        "clifford_semantics": study.get("clifford_semantics") == "frame",
        "termination": study.get("termination_semantics") == "full_workload",
        "architecture": study.get("architecture")
        == ("pinnacle_gb510" if rate == 1.0e-3 else "pinnacle_gb126"),
        "stage_hashes": all(
            isinstance(study.get(name), str)
            and len(study[name]) == 71
            and study[name].startswith("sha256:")
            and all(
                character in "0123456789abcdef"
                for character in study[name][7:]
            )
            for name in (
                "p0_sha256",
                "p1_sha256",
                "p2_sha256",
                "p3_sha256",
                "schedule_sha256",
            )
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"{path} disagrees with the paper matrix: {failed}")
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "lattice": lattice,
        "p_phys": rate,
        "checks": checks,
    }


def _run_matrix(arguments: argparse.Namespace) -> None:
    import signal
    import subprocess
    import sys

    def run_worker(command, stream):
        if (
            len(command) < 2
            or Path(command[1]).resolve() != Path(__file__).resolve()
        ):
            raise RuntimeError("workers may execute only this study file")
        process = subprocess.Popen(
            command,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=(os.name == "posix"),
        )
        try:
            returncode = process.wait(
                timeout=arguments.worker_timeout_seconds
            )
        except subprocess.TimeoutExpired as error:
            if os.name == "posix":
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            else:
                process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                if os.name == "posix":
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                else:
                    process.kill()
                process.wait()
            stream.write(
                f"\nWORKER_TIMEOUT_SECONDS="
                f"{arguments.worker_timeout_seconds}\n"
            )
            stream.flush()
            raise subprocess.TimeoutExpired(
                command, arguments.worker_timeout_seconds
            ) from error
        return subprocess.CompletedProcess(command, returncode)

    if arguments.output_dir is None:
        raise RuntimeError("--matrix requires --output-dir")
    if not arguments.acknowledge_large_run:
        raise RuntimeError(
            "--matrix requires --acknowledge-large-run; archived high-L "
            "points used up to 178.7 GiB"
        )
    output_dir = arguments.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = output_dir / "matrix-receipt.json"
    if any(output_dir.iterdir()) and not (arguments.resume or arguments.force):
        raise FileExistsError(
            f"refusing to use nonempty {output_dir}; pass --resume or --force"
        )
    if receipt_path.exists() and not arguments.force:
        if not arguments.resume:
            raise FileExistsError(f"refusing to overwrite {receipt_path}")

    records = []
    for lattice in _MATRIX_LATTICES:
        for rate in _MATRIX_RATES:
            output = output_dir / _matrix_name(lattice, rate)
            log_path = output.with_suffix(".log")
            if output.exists() and arguments.resume:
                records.append(_validate_matrix_point(output, lattice, rate))
                continue
            available = _available_gib()
            required = _MIN_AVAILABLE_GIB[lattice]
            if available is not None and available < required:
                raise RuntimeError(
                    f"only {available:.1f} GiB available before L={lattice}, "
                    f"p={rate:g}; this point requires at least {required} GiB"
                )
            if (output.exists() or log_path.exists()) and not (
                arguments.force or arguments.resume
            ):
                raise FileExistsError(
                    f"refusing to overwrite {output} or {log_path}"
                )
            command = (
                sys.executable,
                str(Path(__file__).resolve()),
                "--lattice-size",
                str(lattice),
                "--p-phys",
                str(rate),
                "--factory-mode",
                "protocol",
                "--termination-semantics",
                "full_workload",
                "--json",
                str(output),
                "--force",
            )
            with log_path.open("w", encoding="utf-8") as stream:
                completed = run_worker(command, stream)
            if completed.returncode:
                raise subprocess.CalledProcessError(completed.returncode, command)
            records.append(_validate_matrix_point(output, lattice, rate))

    if len(records) != len(_MATRIX_EXPECTED):
        raise RuntimeError("the Fermi--Hubbard matrix is incomplete")
    receipt = {
        "schema": "qlx.paper.pinnacle-fermi-hubbard-matrix/v2",
        "source": QLX_SOURCE,
        "lattices": list(_MATRIX_LATTICES),
        "p_phys": list(_MATRIX_RATES),
        "worker_timeout_seconds": arguments.worker_timeout_seconds,
        "records": records,
        "acceptance": {"complete_28_point_matrix": True},
    }
    temporary = receipt_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, receipt_path)
    print(json.dumps({"receipt": str(receipt_path), "validated": 28}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lattice-size", type=int, default=16)
    parser.add_argument(
        "--p-phys",
        type=float,
        choices=(1.0e-3, 1.0e-4),
        default=1.0e-3,
    )
    parser.add_argument("--cycle-time-ns", type=float, default=1_000.0)
    parser.add_argument(
        "--factory-mode",
        choices=("protocol", "factory_model", "both"),
        default="both",
    )
    parser.add_argument(
        "--clifford-semantics",
        choices=("frame", "execute"),
        default="frame",
        help=("frame absorbs exact Cliffords before P1 (the paper/PBC costing "
              "model); execute schedules every authored Clifford"),
    )
    parser.add_argument(
        "--rus-readout-model",
        choices=tuple(model.value for model in pinnacle.RUSReadoutModel),
        default=pinnacle.RUSReadoutModel.PAPER_LOGICAL_CYCLE.value,
        help=(
            "paper_logical_cycle uses the paper's native one-cycle encoded "
            "MPP; explicit_wsc schedules the constructive serial WSC circuit"),
    )
    parser.add_argument(
        "--termination-semantics",
        choices=tuple(
            value.value for value in qlx.estimate.ScheduleTermination),
        default=qlx.estimate.ScheduleTermination.FULL_WORKLOAD.value,
        help=("full_workload prices the complete resource-estimation workload "
              "and reports exhaustion separately (the paper comparison); "
              "program models early termination on retry abort"),
    )
    parser.add_argument(
        "--json",
        type=Path,
        help="write the compiler-derived and paper-oracle records",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="write JSON, CSV, and plots derived from completed estimates",
    )
    parser.add_argument(
        "--matrix",
        action="store_true",
        help="run and validate the complete 28-point paper matrix",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--acknowledge-large-run", action="store_true")
    parser.add_argument(
        "--worker-timeout-seconds",
        type=int,
        default=21_600,
        help="positive wall-clock limit for each isolated matrix point",
    )
    arguments = parser.parse_args()
    if arguments.worker_timeout_seconds <= 0:
        parser.error("--worker-timeout-seconds must be positive")
    if arguments.matrix:
        _run_matrix(arguments)
        return
    if arguments.json is not None and arguments.json.exists() and not arguments.force:
        parser.error(f"refusing to overwrite {arguments.json}; pass --force")
    if (
        arguments.output_dir is not None
        and arguments.output_dir.exists()
        and any(arguments.output_dir.iterdir())
        and not arguments.force
    ):
        parser.error(
            f"refusing to replace nonempty {arguments.output_dir}; pass --force"
        )
    rus_readout_model = pinnacle.RUSReadoutModel(arguments.rus_readout_model)
    termination = qlx.estimate.ScheduleTermination(
        arguments.termination_semantics)

    problem = plaquette_phase_estimation_problem(arguments.lattice_size)
    raw_p0 = qlx.compile(problem.definition)
    p0 = (qlx.compiler.absorb_clifford_frame(raw_p0)
          if arguments.clifford_semantics == "frame" else raw_p0)
    logical_capacity = qlx.estimate(
        p0,
        tier=qlx.estimate.Tier.LOGICAL,
    ).logical_qubits_peak
    modes = (("protocol",
              "factory_model") if arguments.factory_mode == "both" else
             (arguments.factory_mode,))
    studies = []
    for mode in modes:
        study = compile_physical_study(
            problem,
            p0=p0,
            logical_capacity=logical_capacity,
            p_phys=arguments.p_phys,
            cycle_time_ns=arguments.cycle_time_ns,
            factory_mode=mode,
            clifford_semantics=arguments.clifford_semantics,
            rus_readout_model=rus_readout_model,
            termination=termination,
        )
        if studies:
            # Keep the first independently completed route available, but do
            # not publish the combined flat bundle until the route-pair
            # comparison contract itself has passed.
            _validated_factory_routes(studies[0], study)
        studies.append(study)
        _print_result(problem, study)
        if arguments.json is not None:
            # Persist each completed route so a later independent route cannot
            # erase already-produced compiler evidence if it fails closed.
            _write_json(arguments.json, problem, studies)
        if arguments.output_dir is not None:
            _write_result_artifacts(arguments.output_dir, problem, studies)
    if len(studies) == 2:
        protocol, opaque = _validated_factory_routes(*studies)
        runtime_delta = (opaque.expected_makespan_ns -
                         protocol.expected_makespan_ns)
        print("factory-route comparison: identical footprint and input "
              "cadence; opaque - protocol expected-runtime delta = "
              f"{runtime_delta:g} ns")


if __name__ == "__main__":
    main()
