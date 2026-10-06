# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Compiler-driven Gidney--Ekera RSA-2048 paper-evaluation driver.

This example keeps every consequential input visible.  Standard CUDA-Q kernels
define the folded arithmetic.  Idiomatic QLX protocols define the distance-27
surface-code AutoCCZ producer and consumer.  A typed device describes the
198-by-62 patch board, its 28 factory lanes, and its operating point.  The
ordinary QLX compiler then produces P0, P1, P2, P3, a physical schedule, and a
Tier.SCHEDULE estimate; no paper-specific estimator is called.

The arithmetic is a resource-faithful kernel, not a factoring application: the
QROM controls and measurement fixups preserve the paper recurrence but do not
contain modulus-dependent table data or classical factor recovery.  The
compiler first projects and schedules one complete 15-to-1 -> CCZ -> AutoCCZ
lane, then derives the compact model used by the paper-scale factory bank from
that verified P3 artifact. The nine-patch AutoCCZ ring, adaptive injection
protocol, arithmetic recurrence, workspace, factory capacity, and schedule are
compiler-visible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
from time import perf_counter

import cudaq.logical as qlx
import cudaq


QLX_SOURCE = {
    "module": str(qlx.__file__),
    "version": getattr(qlx, "__version__", "unknown"),
}

import cudaq.logical.architectures.surface as surface_architecture

_definition_started = perf_counter()
# Paper-selected arithmetic and machine facts.
ACCUMULATOR_WIDTH = 2_124
PIECE_LENGTH = 1_062
LOOKUP_COUNT = 505_965
ADDRESS_WIDTH = 10
TABLE_ROWS = (1 << ADDRESS_WIDTH) - 1
FIXUP_COUNT = 64

LEVEL_1_CODE_DISTANCE = 15
LEVEL_2_CODE_DISTANCE = 27
CARRY_PIECES = 2
FACTORY_LANES = 28
FACTORY_ROWS_PER_PIECE = 2
AUTOCZZ_ROUTING_PATCHES = 6
DETAILED_LEVEL1_LANES = 6
DETAILED_LEVEL1_PATCHES = DETAILED_LEVEL1_LANES * 5
DETAILED_AUTOCZZ_WORKSPACES = 2
DETAILED_LEVEL2_PATCHES = 11 + DETAILED_AUTOCZZ_WORKSPACES * 9
DETAILED_FACTORY_PATCHES = ((LEVEL_1_CODE_DISTANCE - 2) *
                            ((LEVEL_1_CODE_DISTANCE - 1) // 2))
RAW_T_INJECTION_LANES = 8 * 15
PHYSICAL_ERROR_RATE = 1.0e-3

SURFACE_L1 = surface_architecture.definitions(LEVEL_1_CODE_DISTANCE)
SURFACE = surface_architecture.definitions(LEVEL_2_CODE_DISTANCE)
LEVEL_1_CODE = SURFACE_L1.code
CODE = SURFACE.code
LEVEL_1_PATCH_FOOTPRINT = SURFACE_L1.square_patch_footprint.units
PATCH_FOOTPRINT = SURFACE.square_patch_footprint.units
# The selected ancillary layout shrinks the level-1 factory boxes from the
# rounded 15-by-8 construction in the paper text to 13-by-7.  Two rows of
# seven factories occupy each arithmetic piece.  The provider independently
# recomputes these formulas from the P2 proof, typed factory model, and machine.
FACTORY_BOX_WIDTH = LEVEL_1_CODE_DISTANCE - 2
FACTORY_BOX_HEIGHT = (LEVEL_1_CODE_DISTANCE - 1) // 2
FACTORY_COLUMNS_PER_PIECE = (FACTORY_LANES //
                             (CARRY_PIECES * FACTORY_ROWS_PER_PIECE))
PIECE_WIDTH = (FACTORY_COLUMNS_PER_PIECE * FACTORY_BOX_WIDTH +
               FACTORY_COLUMNS_PER_PIECE + 1)
OPERATING_ROWS = 2 * FACTORY_BOX_HEIGHT + 2 * 3 + 3 + 6
REGISTER_ROWS_PER_REGISTER = math.ceil(PIECE_LENGTH / (PIECE_WIDTH - 2))
SELECTED_REGISTER_ROWS = 3 * REGISTER_ROWS_PER_REGISTER
PIECE_HEIGHT = OPERATING_ROWS + SELECTED_REGISTER_ROWS
BOARD_PATCHES = CARRY_PIECES * PIECE_WIDTH * PIECE_HEIGHT
FACTORY_PATCHES = (FACTORY_LANES * FACTORY_BOX_WIDTH * FACTORY_BOX_HEIGHT)
ROUTING_POOL_PATCHES = CARRY_PIECES * AUTOCZZ_ROUTING_PATCHES
COMPUTE_PATCHES = BOARD_PATCHES - FACTORY_PATCHES - ROUTING_POOL_PATCHES
logical_z = qlx.gadgets.logical_pauli(
    CODE,
    basis=qlx.architecture.Basis.Z,
)
logical_x = qlx.gadgets.logical_pauli(
    CODE,
    basis=qlx.architecture.Basis.X,
)


def _pauli_product(pauli, patches, indices):
    """Build one typed product over named logical ports."""

    selected = tuple(patches[index] for index in indices)
    product = pauli(selected[0][0])
    for patch in selected[1:]:
        product = product @ pauli(patch[0])
    return product


def _replace_product_successors(values, product, successors):
    """Advance owners in the canonical order carried by ``product``."""

    for factor, successor in zip(product.factors, successors, strict=True):
        owner = getattr(factor.operand, "patch", factor.operand)
        index = next(
            index for index, value in enumerate(values) if value is owner)
        values[index] = successor


def _conditional_z(bit, patch):
    return qlx.ops.cond(
        bit,
        then=lambda live: (logical_z(live),),
        else_=lambda live: (live,),
        carries=(patch,),
    )[0]


# Raw state injection is the physical leaf of this detailed factory.  It is a
# typed P2 producer with an explicit 120-lane P3 timing/resource binding below;
# the distillation circuit therefore has no unresolved external supply.
@qlx.protocol(implements=qlx.logical.produce(qlx.standard.RAW_T_STATE))
def surface_raw_t_injection() -> qlx.types.resource[qlx.standard.RAW_T_STATE]:
    return qlx.ops.produce(qlx.standard.RAW_T_STATE)


# One explicit level-1 lane. The first four raw T states initialize the four
# check rows of the compressed triorthogonal circuit; the other eleven power
# its Z-product pi/4 rotations. Nothing here is a timing or footprint model:
# these are ordinary P2 allocations, resource flows, rotations, measurements,
# and postselections that the selected native projector must carry into P3.
@qlx.protocol(implements=qlx.logical.produce(qlx.standard.T_STATE))
def surface_distill_15to1() -> qlx.types.resource[qlx.standard.T_STATE]:
    raw = qlx.ops.request_many(qlx.standard.RAW_T_STATE, count=15)
    output = SURFACE_L1.prepare_plus(
        qlx.ops.allocate_patch(LEVEL_1_CODE, region="factory"))

    checks = []
    for state in raw[:4]:
        output, check = qlx.ops.unpack_resource(
            state,
            like=output,
            encoding=LEVEL_1_CODE,
        )
        checks.append(check)

    values = [*checks, output]
    for state, support in zip(raw[4:],
                              qlx.protocols.FIFTEEN_TO_ONE_ROTATION_SUPPORTS):
        product = _pauli_product(qlx.types.Z, values, support)
        updated = qlx.ops.resource_rotate(
            state,
            product,
            angle=math.pi / 4.0,
        )
        _replace_product_successors(values, product, updated)

    # This positive-angle convention yields T-dagger on the odd row; logical S
    # converts it to the canonical T|+> resource.
    values[4] = SURFACE_L1.fold_s(values[4])
    for check in values[:4]:
        qlx.ops.postselect(SURFACE_L1.measure_x(check), expected=False)
    return qlx.ops.pack_resource(values[4], kind=qlx.standard.T_STATE)


# Gidney--Fowler Figure 5, expressed directly as typed QLX. The four initial
# X-product measurements are the four syndrome lines in the figure. The eight
# distilled T resources drive the eight quarter rotations on a..h; their
# measurement results then enact the exact classical correction matrix.
@qlx.protocol(implements=qlx.logical.produce(qlx.standard.CCZ_STATE))
def surface_ccz_8to1() -> qlx.types.resource[qlx.standard.CCZ_STATE]:
    distilled_t = tuple(surface_distill_15to1() for _ in range(8))
    patches = [
        SURFACE.prepare_zero(
            qlx.ops.allocate_patch(CODE, region="factory_level2"))
        for _ in range(11)
    ]

    checks = []
    for support in qlx.protocols.CCZ_8TO1_CHECK_SUPPORTS:
        product = _pauli_product(qlx.types.X, patches, support)
        updated = qlx.ops.mpp(product)
        _replace_product_successors(patches, product, updated[:-1])
        checks.append(updated[-1])

    injection_bits = []
    for state, index in zip(distilled_t,
                            qlx.protocols.CCZ_8TO1_INJECTION_TARGETS):
        (patches[index],) = qlx.ops.resource_rotate(
            state,
            qlx.types.Z(patches[index][0]),
            angle=math.pi / 4.0,
        )
        patches[index] = SURFACE.h(patches[index])
        injection_bits.append(SURFACE.measure_z(patches[index]))

    parity = checks[1]
    for bit in injection_bits:
        parity = qlx.ops.xor(parity, bit)
    qlx.ops.postselect(parity, expected=False)

    # a..h -> output-Z masks 111, 110, 101, 100, 011, 010, 001, 000.
    for bit, mask in zip(injection_bits,
                         qlx.protocols.CCZ_8TO1_OUTPUT_CORRECTION_MASKS):
        for output in range(3):
            if mask & (1 << (2 - output)):
                patches[output] = _conditional_z(bit, patches[output])

    # The three remaining syndrome measurements correct outputs 1, 3, and 2.
    for check, output in zip((checks[0], checks[2], checks[3]),
                             qlx.protocols.CCZ_8TO1_SYNDROME_OUTPUTS):
        patches[output] = _conditional_z(check, patches[output])
    patches[:3] = [logical_x(patch) for patch in patches[:3]]
    return qlx.ops.pack_resource(patches[:3], kind=qlx.standard.CCZ_STATE)


# The outer producer consumes the actual Figure-5 CCZ result, unpacks its three
# output patches, adds six |+> routing patches, and applies every encoded-CZ
# edge of the exact nine-patch AutoCCZ ring.
@qlx.protocol(implements=qlx.logical.produce(qlx.standard.AUTO_CCZ_STATE))
def surface_autoccz_factory(
) -> qlx.types.resource[qlx.standard.AUTO_CCZ_STATE]:
    main_anchors = tuple(
        qlx.ops.allocate_patch(CODE, region="factory_level2") for _ in range(3))
    routing = tuple(
        qlx.ops.prepare_plus(
            qlx.ops.allocate_patch(CODE, region="factory_level2"))
        for _ in range(AUTOCZZ_ROUTING_PATCHES))
    ccz_state = surface_ccz_8to1()
    empty_anchors, main_payloads = qlx.ops.unpack_resource(
        ccz_state,
        like=main_anchors,
        logical_ports=((0,), (0,), (0,)),
    )
    qlx.ops.discard(empty_anchors)

    ring = [
        main_payloads[0],
        routing[0],
        routing[1],
        main_payloads[1],
        routing[2],
        routing[3],
        main_payloads[2],
        routing[4],
        routing[5],
    ]
    # An odd nine-cycle has edge-chromatic number three.  Authoring its edges
    # as three disjoint matchings exposes the actual encoded-CZ concurrency to
    # the generic P3 scheduler without attaching a duration model here.
    for matching in ((0, 3, 6), (1, 4, 7), (2, 5, 8)):
        updated = {}
        for left in matching:
            right = (left + 1) % len(ring)
            updated[left], updated[right] = SURFACE.cz(ring[left], ring[right])
        for index, patch in updated.items():
            ring[index] = patch
    return qlx.ops.pack_resource(
        (
            ring[0],
            ring[3],
            ring[6],
            ring[1],
            ring[2],
            ring[4],
            ring[5],
            ring[7],
            ring[8],
        ),
        kind=qlx.standard.AUTO_CCZ_STATE,
    )


# Figure 4's adaptive consumer injects the three main payloads into the two
# controls and target, resolves the six delayed-choice routing measurements,
# applies the linear and quadratic Z corrections, and returns the three live
# data patches.
@qlx.protocol
def consume_surface_autoccz_as_toffoli(
    control_a: qlx.patch[CODE],
    control_b: qlx.patch[CODE],
    target: qlx.patch[CODE],
    main_a: qlx.patch[CODE],
    main_b: qlx.patch[CODE],
    main_c: qlx.patch[CODE],
    ab_a: qlx.patch[CODE],
    ab_b: qlx.patch[CODE],
    bc_b: qlx.patch[CODE],
    bc_c: qlx.patch[CODE],
    ca_c: qlx.patch[CODE],
    ca_a: qlx.patch[CODE],
) -> tuple[qlx.patch[CODE], qlx.patch[CODE], qlx.patch[CODE]]:

    def measure_delayed_pair(choice, left, right):
        left, right = qlx.ops.cond(
            choice,
            then=lambda a, b: (SURFACE.h(a), SURFACE.h(b)),
            else_=lambda a, b: (a, b),
            carries=(left, right),
        )
        left_bit = SURFACE.measure_z(left)
        right_bit = SURFACE.measure_z(right)
        return qlx.ops.cond(
            choice,
            then=lambda a, b: (b, a),
            else_=lambda a, b: (a, b),
            carries=(left_bit, right_bit),
        )

    def apply_z_if(bit, block):
        return qlx.ops.cond(
            bit,
            then=lambda live: (logical_z(live),),
            else_=lambda live: (live,),
            carries=(block,),
        )[0]

    def z_if_both(first, second, block):
        return qlx.ops.cond(
            first,
            then=lambda live: (qlx.ops.cond(
                second,
                then=lambda nested: (logical_z(nested),),
                else_=lambda nested: (nested,),
                carries=(live,),
            )[0],),
            else_=lambda live: (live,),
            carries=(block,),
        )[0]

    target = SURFACE.h(target)
    control_a, main_a = SURFACE.transversal_cx(control_a, main_a)
    control_b, main_b = SURFACE.transversal_cx(control_b, main_b)
    target, main_c = SURFACE.transversal_cx(target, main_c)
    outcome_a = SURFACE.measure_z(main_a)
    outcome_b = SURFACE.measure_z(main_b)
    outcome_c = SURFACE.measure_z(main_c)

    bit_ab_a, bit_ab_b = measure_delayed_pair(outcome_c, ab_a, ab_b)
    bit_bc_b, bit_bc_c = measure_delayed_pair(outcome_a, bc_b, bc_c)
    bit_ca_c, bit_ca_a = measure_delayed_pair(outcome_b, ca_c, ca_a)

    control_a = apply_z_if(bit_ab_a, control_a)
    control_b = apply_z_if(bit_ab_b, control_b)
    control_b = apply_z_if(bit_bc_b, control_b)
    target = apply_z_if(bit_bc_c, target)
    target = apply_z_if(bit_ca_c, target)
    control_a = apply_z_if(bit_ca_a, control_a)
    target = z_if_both(outcome_a, outcome_b, target)
    control_a = z_if_both(outcome_b, outcome_c, control_a)
    control_b = z_if_both(outcome_a, outcome_c, control_b)
    target = SURFACE.h(target)
    return control_a, control_b, target


def routing_region(context, placement):
    primary = context.qec_region_for(placement)
    binding = next(item for item in context.device.logical_to_qec
                   if item.qec_region == primary)
    if (len(binding.auxiliary_regions) != 1 or
            binding.auxiliary_regions[0].role != "scratch"):
        raise ValueError(
            "AutoCCZ lowering requires one auxiliary scratch QEC region")
    return binding.auxiliary_regions[0]


# Selection is explicit and typed: every logical CCX on this code requests an
# AutoCCZ state, unpacks the nine roles against three live data owners and six
# fresh routing anchors, and invokes the adaptive consumer above.
@qlx.compiler.qec_lowering(
    objective=qlx.logical.ccx,
    codes=(CODE,),
    dependencies=(consume_surface_autoccz_as_toffoli,),
    consumes=(qlx.standard.AUTO_CCZ_STATE,),
    plugin="example.surface_autoccz",
    version="1",
)
def inject_surface_autoccz(site, context):
    encoding = context.encoding
    scratch = routing_region(context, site.placements[0])

    @qlx.protocol(implements=qlx.logical.ccx)
    def surface_autoccz_toffoli(
        control_a: qlx.patch[encoding],
        control_b: qlx.patch[encoding],
        target: qlx.patch[encoding],
    ) -> tuple[
            qlx.patch[encoding],
            qlx.patch[encoding],
            qlx.patch[encoding],
    ]:
        state = qlx.ops.event_await(qlx.ops.request(
            qlx.standard.AUTO_CCZ_STATE))
        anchors = tuple(
            qlx.ops.allocate_patch(encoding, region=scratch)
            for _ in range(AUTOCZZ_ROUTING_PATCHES))
        successors, payloads = qlx.ops.unpack_resource(
            state,
            like=(control_a, control_b, target, *anchors),
            logical_ports=((0,), (0,), (0,), (), (), (), (), (), ()),
        )
        control_a, control_b, target = successors[:3]
        qlx.ops.discard(successors[3:])
        return consume_surface_autoccz_as_toffoli(
            control_a,
            control_b,
            target,
            *payloads,
        )

    return surface_autoccz_toffoli


# These are ordinary CUDA-Q kernels. QLX retains their typed helper-call graph
# through P0, P1, and P2, then lowers each selected logical action normally.
# Neither the device nor a compiler pass assigns algorithm-specific roles to
# these helpers.
QEC_DEFINITION_SECONDS = perf_counter() - _definition_started
_cudaq_definition_started = perf_counter()


@cudaq.kernel
def toffoli(
    control_a: cudaq.qubit,
    control_b: cudaq.qubit,
    target: cudaq.qubit,
):
    x.ctrl([control_a, control_b], target)


@cudaq.kernel
def maj(carry: cudaq.qubit, addend: cudaq.qubit, accumulator: cudaq.qubit):
    x.ctrl(accumulator, addend)
    x.ctrl(accumulator, carry)
    toffoli(carry, addend, accumulator)


@cudaq.kernel
def uma(carry: cudaq.qubit, addend: cudaq.qubit, accumulator: cudaq.qubit):
    toffoli(carry, addend, accumulator)
    x.ctrl(accumulator, carry)
    x.ctrl(carry, addend)


@cudaq.kernel
def maj_pair(
    lower_carry: cudaq.qubit,
    lower_addend: cudaq.qubit,
    lower_accumulator: cudaq.qubit,
    upper_carry: cudaq.qubit,
    upper_addend: cudaq.qubit,
    upper_accumulator: cudaq.qubit,
):
    """Two independent carry-piece reactions exposed in one helper graph."""

    maj(lower_carry, lower_addend, lower_accumulator)
    maj(upper_carry, upper_addend, upper_accumulator)


@cudaq.kernel
def uma_pair(
    lower_carry: cudaq.qubit,
    lower_addend: cudaq.qubit,
    lower_accumulator: cudaq.qubit,
    upper_carry: cudaq.qubit,
    upper_addend: cudaq.qubit,
    upper_accumulator: cudaq.qubit,
):
    """Two independent unmajority reactions exposed in one helper graph."""

    uma(lower_carry, lower_addend, lower_accumulator)
    uma(upper_carry, upper_addend, upper_accumulator)


@cudaq.kernel
def qrom_access_step(
    address: cudaq.qubit,
    workspace_a: cudaq.qubit,
    workspace_b: cudaq.qubit,
    target: cudaq.qubit,
):
    """One unary-iteration reaction followed by an external row access.

    The helper boundary is ordinary CUDA-Q.  At P2 its exact owner graph has
    one selected AutoCCZ application connected by CX to a fourth owner; the
    surface-code P3 provider can therefore derive one alternating access layer
    without trusting this function's name or attaching an arithmetic role.
    """

    toffoli(address, workspace_a, workspace_b)
    x.ctrl(workspace_b, target)


@cudaq.kernel
def lookup_addition(
    accumulator: cudaq.qview,
    address: cudaq.qview,
    bus: cudaq.qview,
    runways: cudaq.qview,
    ancillas: cudaq.qview,
    unlookup_access: cudaq.qubit,
):
    for table_index in range(1023):
        qrom_access_step(
            address[table_index % 10],
            ancillas[0],
            ancillas[1],
            bus[table_index % 2124],
        )

    # The two carry-runway pieces are independent spatial sweeps.  Express
    # them in lockstep so a stable list/ASAP scheduler can expose the intended
    # two-piece concurrency without assigning a semantic "MAJ phase" or
    # "UMA phase" to either the device or the dialect.
    maj_pair(
        runways[0],
        bus[0],
        accumulator[0],
        runways[1],
        bus[1062],
        accumulator[1062],
    )
    for offset in range(1061):
        lower = offset + 1
        upper = offset + 1063
        maj_pair(
            accumulator[lower - 1],
            bus[lower],
            accumulator[lower],
            accumulator[upper - 1],
            bus[upper],
            accumulator[upper],
        )
    for offset in range(1061):
        lower = 1061 - offset
        upper = 2123 - offset
        uma_pair(
            accumulator[lower - 1],
            bus[lower],
            accumulator[lower],
            accumulator[upper - 1],
            bus[upper],
            accumulator[upper],
        )

    # The 64 alternating accesses are the retained physical recurrence for
    # measurement-based QROM uncomputation.  The persistent bus is not torn
    # down here: its final measurement belongs to the board lifetime, not to
    # every arithmetic iteration.
    for fixup in range(64):
        qrom_access_step(
            address[fixup % 10],
            ancillas[0],
            ancillas[1],
            unlookup_access,
        )


@cudaq.kernel
def rsa2048_resource_kernel():
    accumulator = cudaq.qvector(2124)
    # The paper's arithmetic board is persistent.  These workspaces are
    # prepared once, retained through every folded lookup addition, and
    # released only after the complete arithmetic recurrence.  Allocating them
    # inside lookup_addition would incorrectly charge board startup/cleanup to
    # all 505,965 iterations.
    address = cudaq.qvector(10)
    bus = cudaq.qvector(2124)
    runways = cudaq.qvector(2)
    ancillas = cudaq.qvector(2)
    unlookup_access = cudaq.qubit()
    h(address)
    for _ in range(505965):
        lookup_addition(
            accumulator,
            address,
            bus,
            runways,
            ancillas,
            unlookup_access,
        )
    for bit in range(2124):
        mx(bus[bit])
    for bit in range(10):
        mx(address[bit])
    for bit in range(2):
        mz(runways[bit])
    for bit in range(2):
        mz(ancillas[bit])
    mz(unlookup_access)
    mx(accumulator[0])


CUDAQ_DEFINITION_SECONDS = perf_counter() - _cudaq_definition_started


def _surface_resource_options(surface):
    return {
        "granularity":
            qlx.architecture.ResourceGranularity.PATCH,
        "footprint":
            surface.square_patch_footprint,
        "native_actions":
            qlx.architecture.physical_actions.clifford_set(),
        "native_instruments": (
            qlx.architecture.physical_instruments.MX,
            qlx.architecture.physical_instruments.MZ,
            qlx.architecture.physical_instruments.MPP,
        ),
    }


def _surface_timing():
    cycle = 1 * qlx.devices.us

    def encoded_patch_layer(distance):
        layer = distance * cycle
        return {
            # These are patch-level lattice-surgery macro durations. Pauli
            # corrections are frame updates; every other encoded operation is
            # one distance-round layer in this explicit operating-point model.
            "prepare_ns": layer,
            "h_ns": layer,
            "cx_ns": layer,
            "cz_ns": layer,
            "x_ns": 0 * qlx.devices.ns,
            "z_ns": 0 * qlx.devices.ns,
            "mpp_ns": layer,
            "rpp_ns": layer,
            "resource_rpp_ns": layer,
            "measure_z_instrument_ns": layer,
            "measure_x_instrument_ns": layer,
            "reset_ns": layer,
            # Packing changes typed ownership but performs no extra physical
            # evolution beyond the surrounding compiled operations.
            "pack_resource_ns": 0 * qlx.devices.ns,
            "unpack_resource_ns": 0 * qlx.devices.ns,
        }

    return qlx.devices.TimingModel(
        {
            "surface_cycle_ns": cycle,
            "reaction_time_ns": 10 * qlx.devices.us,
            "condition_ns": 10 * qlx.devices.us,
            "postselect_ns": 10 * qlx.devices.us,
            "xor_ns": 0 * qlx.devices.ns,
        },
        by_code_distance={
            LEVEL_1_CODE_DISTANCE: encoded_patch_layer(LEVEL_1_CODE_DISTANCE),
            LEVEL_2_CODE_DISTANCE: encoded_patch_layer(LEVEL_2_CODE_DISTANCE),
        },
        source=("surface-code patch macro model: 1us code cycle and one "
                "distance-round layer per encoded operation; informed by "
                "arXiv:1812.01238"),
    )


def build_detailed_factory_device():
    """Build one explicit level-1/level-2 AutoCCZ production lane."""

    architecture = qlx.devices.QECArchitecture(
        "surface_autoccz_factory_d15_d27",
        LEVEL_1_CODE.default_encoding,
        auxiliary_regions=(qlx.devices.QECRegion(
            "level2",
            CODE.default_encoding,
            DETAILED_LEVEL2_PATCHES,
            role="scratch",
        ),),
    )
    builder = qlx.devices.DeviceBuilder("DetailedSurfaceAutoCCZLane")
    factory = builder.logical.add_factory(
        produces=qlx.standard.AUTO_CCZ_STATE,
        via=surface_autoccz_factory,
        capacity=1,
        name="factory",
        stream_name="autoccz_states",
    )
    raw_factory = builder.logical.add_factory(
        produces=qlx.standard.RAW_T_STATE,
        via=surface_raw_t_injection,
        capacity=RAW_T_INJECTION_LANES,
        name="raw_t_injection",
        stream_name="raw_t_states",
    )
    factory_qec = builder.qec.bind(
        factory,
        architecture=architecture,
        block_capacity=DETAILED_LEVEL1_PATCHES,
    )
    raw_qec = builder.qec.bind(
        raw_factory,
        architecture=qlx.devices.QECArchitecture(
            "surface_raw_t_injection_d15",
            LEVEL_1_CODE.default_encoding,
        ),
        block_capacity=RAW_T_INJECTION_LANES,
    )
    factory_layout = builder.physical.add_resources(
        "surface_code_patch",
        DETAILED_FACTORY_PATCHES,
        name="factory_layout_patches",
        **_surface_resource_options(SURFACE),
    )
    raw_injection_qubits = builder.physical.add_qubits(
        RAW_T_INJECTION_LANES,
        name="raw_t_injection_qubits",
    )
    builder.physical.bind(factory_qec, to=factory_layout)
    builder.physical.bind(
        factory_qec.auxiliary_regions[0],
        to=factory_layout,
    )
    builder.physical.bind(
        raw_qec,
        to=raw_injection_qubits,
        factory_model=qlx.devices.FactoryModel(
            startup_cycles=1,
            output_interval_cycles=1,
            evidence=qlx.analysis.user_assertion(
                "one physical injection qubit and one surface cycle per raw "
                "T state"),
        ),
    )
    builder.physical.set_operating_point(
        timing=_surface_timing(),
        calibration={
            "physical_error": PHYSICAL_ERROR_RATE,
            "surface_scaling_prefactor": 0.03,
            "surface_threshold": 0.01,
        },
    )
    return builder.build()


def build_paper_device(factory_model):
    """Build the workload-neutral distance-27 board and factory model."""

    architecture = qlx.devices.QECArchitecture(
        "surface_autoccz_d27",
        CODE.default_encoding,
        link_roots=(
            *SURFACE.wsc().link_roots,
            logical_z,
            inject_surface_autoccz,
        ),
        auxiliary_regions=(qlx.devices.QECRegion(
            "autoccz_routing",
            CODE.default_encoding,
            AUTOCZZ_ROUTING_PATCHES,
            role="scratch",
        ),),
    )
    builder = qlx.devices.DeviceBuilder("GidneyEkeraSurfaceBoard")
    compute = builder.logical.add_compute(
        capacity=COMPUTE_PATCHES,
        name="compute",
    )
    factory = builder.logical.add_factory(
        produces=qlx.standard.AUTO_CCZ_STATE,
        via=surface_autoccz_factory,
        capacity=FACTORY_LANES,
        name="autoccz_factory",
        stream_name="autoccz_states",
    )
    builder.logical.add_stream(
        qlx.standard.RAW_T_STATE,
        name="raw_t_states",
        external=True,
    )
    compute_qec = builder.qec.bind(compute, architecture=architecture)
    factory_qec = builder.qec.bind(
        factory,
        encoding=CODE,
        block_capacity=9 * FACTORY_LANES,
    )
    resource_options = _surface_resource_options(SURFACE)
    compute_patches = builder.physical.add_resources(
        "surface_code_patch",
        COMPUTE_PATCHES,
        name="compute_patches",
        **resource_options,
    )
    routing_patches = builder.physical.add_resources(
        "surface_code_patch",
        ROUTING_POOL_PATCHES,
        name="routing_patches",
        **resource_options,
    )
    # One schedulable member is one complete, independently compiled factory
    # lane.  Its base-unit footprint comes from the characterized P3 schedule,
    # including the explicit raw-injection leaf; the RSA device does not
    # recreate or round this value from paper geometry.
    factory_lanes = builder.physical.add_resources(
        "autoccz_factory_lane",
        FACTORY_LANES,
        name="factory_lanes",
        granularity=qlx.architecture.ResourceGranularity.PATCH,
        footprint=qlx.architecture.PhysicalFootprint(
            factory_model.characterization.physical_unit_kind,
            factory_model.characterization.physical_units,
            "compiler-characterized detailed AutoCCZ factory P3 schedule",
        ),
    )
    builder.physical.bind(compute_qec, to=compute_patches)
    builder.physical.bind(
        compute_qec.auxiliary_regions[0],
        to=routing_patches,
    )
    builder.physical.bind(
        factory_qec,
        to=factory_lanes,
        factory_model=factory_model,
    )
    builder.physical.set_operating_point(
        timing=_surface_timing(),
        calibration={
            "physical_error": 1.0e-3,
            "surface_scaling_prefactor": 0.03,
            "surface_threshold": 0.01,
        },
    )
    return builder.build()


def peak_rss_mib():
    raw_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw_peak / (1024 * 1024 if sys.platform == "darwin" else 1024)


def timed(label, operation, timings):
    started = perf_counter()
    result = operation()
    elapsed = perf_counter() - started
    peak_mib = peak_rss_mib()
    timings.append((label, elapsed, peak_mib))
    print(
        f"[{label}] {elapsed:.3f} s; process peak {peak_mib:.1f} MiB",
        file=sys.stderr,
        flush=True,
    )
    return result


def _print_detailed_factory(schedule, model, estimate, timings) -> None:
    """Print the independently compiled factory evidence."""

    print("Detailed surface-code AutoCCZ factory:")
    print("  eight 15-to-1 invocations on six recurring lanes -> CCZ -> "
          "two-workspace AutoCCZ stage")
    print(f"  scheduled P3 events: {len(schedule.entries):,}")
    print(f"  single-shot latency: {model.startup_cycles:,.3f} "
          "surface cycles")
    print(f"  compiler-derived steady-state cadence: "
          f"{model.output_interval_cycles:,.3f} surface cycles")
    print(f"  physical footprint: "
          f"{model.characterization.physical_units:,} "
          f"{model.characterization.physical_unit_kind}s")
    print(f"  independently estimated P3 footprint: "
          f"{estimate.physical_qubits:,} physical qubits")
    print(f"  selected code distances: "
          f"{model.characterization.code_distances}")
    print("  policy: single-shot accepted path; no retry model")
    print("  lowering performance:")
    for label, seconds, peak_mib in timings:
        print(f"    {label}: {seconds:.3f} s; peak RSS {peak_mib:.1f} MiB")


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def main(
    *,
    emit_p3: Path | None = None,
    emit_factory_p3: Path | None = None,
    factory_only: bool = False,
) -> dict[str, object]:
    timings = [
        (
            "define QEC protocols",
            QEC_DEFINITION_SECONDS,
            peak_rss_mib(),
        ),
        (
            "define exact CUDA-Q kernels",
            CUDAQ_DEFINITION_SECONDS,
            peak_rss_mib(),
        ),
    ]
    detailed_factory_device = timed(
        "build detailed factory device",
        build_detailed_factory_device,
        timings,
    )
    detailed_factory_p3 = timed(
        "compile detailed factory to P3",
        lambda: qlx.compile(
            surface_autoccz_factory,
            pipeline=qlx.compiler.pipelines.physical(),
            device=detailed_factory_device,
        ),
        timings,
    )
    detailed_factory_schedule = timed(
        "schedule detailed factory",
        lambda: qlx.compiler.schedule(detailed_factory_p3),
        timings,
    )
    detailed_factory_estimate = timed(
        "estimate detailed factory schedule",
        lambda: qlx.estimate(
            detailed_factory_schedule,
            tier=qlx.estimate.Tier.SCHEDULE,
            p_phys=PHYSICAL_ERROR_RATE,
            failure_budget=0.8,
            scaling=qlx.estimate.Scaling(
                prefactor=0.03,
                threshold=0.01,
            ),
            cycle_time=1.0e-6,
        ),
        timings,
    )
    factory_model = timed(
        "characterize compact factory model",
        lambda: qlx.compiler.factory_model(
            detailed_factory_schedule,
            produces=qlx.standard.AUTO_CCZ_STATE,
        ),
        timings,
    )
    assert (detailed_factory_estimate.physical_qubits ==
            factory_model.characterization.physical_units)
    if emit_factory_p3 is not None:
        assembly = timed(
            "serialize detailed factory P3",
            detailed_factory_schedule.build.to_mlir,
            timings,
        )
        _atomic_write(emit_factory_p3, assembly)
    payload = {
        "schema": "qlx.paper.rsa2048/v1",
        "status": "passed",
        "source": QLX_SOURCE,
        "study_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "factory_only": factory_only,
        "detailed_factory": {
            "p3_sha256": detailed_factory_p3.content_sha256,
            "schedule_sha256": detailed_factory_schedule.build.content_sha256,
            "schedule_entries": len(detailed_factory_schedule.entries),
            "makespan_ns": detailed_factory_schedule.makespan_ns,
            "physical_qubits": detailed_factory_estimate.physical_qubits,
            "startup_cycles": factory_model.startup_cycles,
            "output_interval_cycles": factory_model.output_interval_cycles,
        },
    }
    if factory_only:
        _print_detailed_factory(
            detailed_factory_schedule,
            factory_model,
            detailed_factory_estimate,
            timings,
        )
        if emit_factory_p3 is not None:
            print(f"  scheduled P3 MLIR: {emit_factory_p3}")
        return payload
    device = timed(
        "build paper device from compiled factory",
        lambda: build_paper_device(factory_model),
        timings,
    )
    p0 = timed(
        "CUDA-Q -> P0",
        lambda: qlx.compiler.import_cudaq(rsa2048_resource_kernel),
        timings,
    )
    p1 = timed(
        "P0 -> P1 placement",
        lambda: qlx.compiler.place(p0, device=device),
        timings,
    )
    p2 = timed(
        "P1 -> selected P2",
        lambda: qlx.compile(
            p1,
            pipeline=qlx.compiler.pipelines.qec(),
            device=device,
        ),
        timings,
    )
    p3 = timed(
        "P2 -> physical P3",
        lambda: qlx.compile(
            p2,
            pipeline=qlx.compiler.pipelines.physical(),
            device=device,
        ),
        timings,
    )
    schedule = timed(
        "P3 -> schedule",
        lambda: qlx.compiler.schedule(p3),
        timings,
    )
    estimate = timed(
        "generic Tier.SCHEDULE estimate",
        lambda: qlx.estimate(
            schedule,
            tier=qlx.estimate.Tier.SCHEDULE,
            p_phys=PHYSICAL_ERROR_RATE,
            failure_budget=0.8,
            scaling=qlx.estimate.Scaling(
                prefactor=0.03,
                threshold=0.01,
            ),
            cycle_time=1.0e-6,
            # The paper selects a distance-27 surface-code construction. QLX
            # retains that as a claimed code-distance profile rather than
            # pretending an exhaustive distance proof was run at this scale.
            evidence_policy=qlx.estimate.EvidencePolicy(),
        ),
        timings,
    )
    # Consume the already verified immutable schedule.  The native lower-tier
    # and Tier-3 passes therefore analyze exactly the artifact printed below
    # without rerunning the comparatively expensive P3 scheduler.
    assert math.isclose(
        estimate.makespan_ns,
        schedule.makespan_ns,
        rel_tol=0.0,
        abs_tol=1.0e-3,
    )
    cycle_ns = _surface_timing()["surface_cycle_ns"]
    bank_interval_ns = (factory_model.output_interval_cycles * cycle_ns /
                        FACTORY_LANES)
    qrom_step_ns = max(CODE.d.conservative_value * cycle_ns / 2.0,
                       bank_interval_ns)
    addition_step_ns = max(
        _surface_timing()["reaction_time_ns"],
        CARRY_PIECES * bank_interval_ns,
    )
    lookup_period_ns = ((TABLE_ROWS + FIXUP_COUNT) * qrom_step_ns +
                        (PIECE_LENGTH + PIECE_LENGTH - 1) * addition_step_ns)
    final_measure_ns = _surface_timing(
    ).by_code_distance[LEVEL_2_CODE_DISTANCE]["measure_x_instrument_ns"]
    expected_ns = (factory_model.startup_cycles * cycle_ns +
                   LOOKUP_COUNT * lookup_period_ns + final_measure_ns)
    # The paper board counts every factory box as distance-27 patch area.  The
    # compiler additionally retains the 120 physical raw-injection leaves in
    # each of the 28 characterized lanes instead of hiding them in that area.
    expected_qubits = (BOARD_PATCHES * PATCH_FOOTPRINT +
                       FACTORY_LANES * RAW_T_INJECTION_LANES)
    assert math.isclose(
        estimate.makespan_ns,
        expected_ns,
        rel_tol=0.0,
        abs_tol=1.0e-3,
    ), (f"compiler makespan {estimate.makespan_ns} ns differs from the "
        f"independent paper recurrence {expected_ns} ns")
    assert estimate.physical_qubits == expected_qubits

    if emit_p3 is not None:
        assembly = timed("serialize scheduled P3", schedule.build.to_mlir,
                         timings)
        _atomic_write(emit_p3, assembly)

    payload["paper_compilation"] = {
        "p0_sha256": p0.content_sha256,
        "p1_sha256": p1.content_sha256,
        "p2_sha256": p2.content_sha256,
        "p3_sha256": p3.content_sha256,
        "schedule_sha256": schedule.build.content_sha256,
        "schedule_entries": len(schedule.entries),
        "makespan_ns": estimate.makespan_ns,
        "physical_qubits": estimate.physical_qubits,
        "expected_recurrence_ns": expected_ns,
        "expected_recurrence_qubits": expected_qubits,
    }

    print("Gidney--Ekera RSA-2048 through the generic QLX compiler:")
    print(f"  P0 root: @{p0.root.symbol}")
    print(f"  P1 placed logical owners: {len(p1.placement.bindings):,}")
    print("  P2 arithmetic: ordinary typed helper protocols and selected "
          "logical actions")
    print("  P2 realization: distance-27 surface-code AutoCCZ")
    print("  detailed factory: eight 15-to-1 protocols -> CCZ -> AutoCCZ")
    print(f"  detailed factory P3 events: "
          f"{len(detailed_factory_schedule.entries):,}")
    print(f"  detailed factory single-shot latency: "
          f"{factory_model.startup_cycles:,.3f} surface cycles")
    print(f"  detailed factory periodic cadence: "
          f"{factory_model.output_interval_cycles:,.3f} surface cycles")
    print(f"  detailed factory physical footprint: "
          f"{factory_model.characterization.physical_units:,} "
          f"{factory_model.characterization.physical_unit_kind}s")
    print("  compact factory policy: single-shot accepted path; no retry model")
    print("  timing evidence: sourced distance-qualified patch-macro model")
    print("  distance evidence: conditional paper-selected d=15/d=27 claims")
    bottleneck = ("factory" if bank_interval_ns > min(
        CODE.d.conservative_value * cycle_ns / 2.0,
        _surface_timing()["reaction_time_ns"] / CARRY_PIECES,
    ) else "reaction/code depth")
    print(f"  independent paper-model bottleneck: {bottleneck}")
    print(f"  aggregate factory output interval: "
          f"{bank_interval_ns / 1_000.0:.6f} us")
    print(f"  independent paper-model lookup period: "
          f"{lookup_period_ns / 1_000_000.0:.6f} ms")
    print(f"  folded lookup additions: {LOOKUP_COUNT:,}")
    print(f"  compact scheduled events: {estimate.event_count:,}")
    print(f"  physical qubits: {estimate.physical_qubits:,}")
    print("  compiler result is a single-shot arithmetic schedule;")
    print("  retry probability and classical factor recovery are not scheduled")
    print(f"  compiler single-shot makespan: "
          f"{estimate.makespan_ns / 3.6e12:.6f} h")
    print("  lowering performance:")
    for label, seconds, peak_mib in timings:
        print(f"    {label}: {seconds:.3f} s; peak RSS {peak_mib:.1f} MiB")
    if emit_p3 is not None:
        print(f"  scheduled P3 MLIR: {emit_p3}")
    return payload


if __name__ == "__main__":
    if not __debug__:
        raise RuntimeError("run this experiment without Python -O")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--emit-p3",
        type=Path,
        help="optional destination for the verified scheduled P3 MLIR",
    )
    parser.add_argument(
        "--emit-factory-p3",
        type=Path,
        help="optional destination for the verified detailed-factory P3 MLIR",
    )
    parser.add_argument(
        "--factory-only",
        action="store_true",
        help="compile, schedule, and report only the detailed factory",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--force", action="store_true")
    options = parser.parse_args()
    for destination in (options.emit_p3, options.emit_factory_p3, options.output):
        if destination is not None and destination.exists() and not options.force:
            parser.error(f"refusing to overwrite {destination}")
    payload = main(
        emit_p3=options.emit_p3,
        emit_factory_p3=options.emit_factory_p3,
        factory_only=options.factory_only,
    )
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if options.output is not None:
        _atomic_write(options.output, encoded)
    print(encoded, end="")
