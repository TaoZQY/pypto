# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: F722, F821

"""Tests for the ``MaterializeCommDomainScopes`` pass.

The pass walks each ``host_orch`` function, traces
``pld.tensor.alloc_window_buffer → pld.tensor.window → dispatch(device=r)``
chains, and:

* constructs one :class:`ir.WindowBuffer` per alloc,
* rewrites every ``pld.tensor.window`` result Var so its
  :class:`ir.DistributedTensorType` carries a ``window_buffer`` back-reference,
* clusters allocs with the same device descriptor into a single comm domain and
  wraps the host_orch body in nested :class:`ir.CommDomainScopeStmt` nodes
  (outer = first declared domain, inner = last).

The tests exercise SSA conversion, signal synthesis, and communication-domain
materialization. Complete programs round-trip through the private dump syntax;
slot identity assertions additionally check that views share their scope's
WindowBuffer objects. Per-slot structural expectations cover device inference
independently of the printer and parser.

"""

import ast
import textwrap
from typing import cast

import pypto
import pypto.language as pl
import pypto.language.distributed as pld
import pytest
from pypto.ir import IRBuilder
from pypto.ir.op.distributed import tensor_ops as dist_tensor_ops
from pypto.language.parser.diagnostics import ParserTypeError
from pypto.pypto_core import DataType, ir, passes

_OP_PLD_TENSOR_ALLOC_WINDOW_BUFFER = ir.get_op("pld.tensor.alloc_window_buffer").name
_OP_PLD_TENSOR_ALLREDUCE = ir.get_op("pld.tensor.allreduce").name
_OP_PLD_TENSOR_WINDOW = ir.get_op("pld.tensor.window").name


def _get_func(program: ir.Program, name: str) -> ir.Function:
    gvar = program.get_global_var(name)
    assert gvar is not None
    return program.functions[gvar]


def _as_call(expr: ir.Expr) -> ir.Call:
    assert isinstance(expr, ir.Call)
    return expr


def _as_var(expr: ir.Expr) -> ir.Var:
    assert isinstance(expr, ir.Var)
    return expr


def _find_window_calls(func: ir.Function) -> list[ir.AssignStmt]:
    """Return all ``AssignStmt``s whose RHS is a ``pld.tensor.window`` Call."""

    found: list[ir.AssignStmt] = []

    def walk(stmt: ir.Stmt) -> None:
        if isinstance(stmt, ir.AssignStmt):
            if isinstance(stmt.value, ir.Call) and stmt.value.op.name == _OP_PLD_TENSOR_WINDOW:
                found.append(stmt)
        if isinstance(stmt, ir.SeqStmts):
            for s in stmt.stmts:
                walk(s)
        if isinstance(stmt, ir.ForStmt):
            walk(stmt.body)
        if isinstance(stmt, ir.ScopeStmt):
            walk(stmt.body)

    walk(func.body)
    return found


def _view_var_types(func: ir.Function) -> list[ir.DistributedTensorType]:
    """Return the type of each pld.tensor.window result Var, in source order."""
    return [
        stmt.var.type
        for stmt in _find_window_calls(func)
        if isinstance(stmt.var.type, ir.DistributedTensorType)
    ]


def _get_comm_domain_scopes(func: ir.Function) -> list[ir.CommDomainScopeStmt]:
    """Walk ``func.body`` and return every ``CommDomainScopeStmt`` it carries,
    in nesting order (outermost first, innermost last)."""
    scopes: list[ir.CommDomainScopeStmt] = []

    def walk(stmt: ir.Stmt) -> None:
        if isinstance(stmt, ir.CommDomainScopeStmt):
            scopes.append(stmt)
        if isinstance(stmt, ir.SeqStmts):
            for s in stmt.stmts:
                walk(s)
        if isinstance(stmt, ir.ForStmt):
            walk(stmt.body)
        if isinstance(stmt, ir.ScopeStmt):
            walk(stmt.body)

    walk(func.body)
    return scopes


def _flatten_stmts(stmt: ir.Stmt) -> list[ir.Stmt]:
    result = [stmt]
    if isinstance(stmt, ir.SeqStmts):
        for child in stmt.stmts:
            result.extend(_flatten_stmts(child))
    if isinstance(stmt, ir.ScopeStmt):
        result.extend(_flatten_stmts(stmt.body))
    if isinstance(stmt, ir.ForStmt):
        result.extend(_flatten_stmts(stmt.body))
    if isinstance(stmt, ir.WhileStmt):
        result.extend(_flatten_stmts(stmt.body))
    return result


def _apply(program: ir.Program) -> ir.Program:
    # Run ConvertToSSA first, matching the real pipeline (pass_manager.py runs
    # it before these passes): loop-carried data becomes an IterArg, the exact
    # shape the passes must handle — without this the loop tests assert against
    # IR the compiler never produces (review #2504).
    program = cast(ir.Program, passes.convert_to_ssa()(program))
    return cast(ir.Program, passes.materialize_comm_domain_scopes()(_synthesize(program)))


def _synthesize(program: ir.Program) -> ir.Program:
    return cast(ir.Program, passes.synthesize_allreduce_signals()(program))


def _mul(left: int, right: int) -> ir.Expr:
    """Build ``Mul(ConstInt(left, INDEX), ConstInt(right, INDEX))``.

    Matches the expression the parser produces when a ``@pl.program`` body
    contains ``N * pl.<DType>.get_byte()`` (e.g. ``256 * pl.FP32.get_byte()``
    → ``256 * 4``, a ``Mul`` of two ``ConstInt`` with dtype ``INDEX``).
    """
    span = ir.Span.unknown()
    return ir.Mul(
        ir.ConstInt(left, DataType.INDEX, span),
        ir.ConstInt(right, DataType.INDEX, span),
        DataType.INDEX,
        span,
    )


def _expected_slot(name: str, size: ir.Expr) -> ir.WindowBuffer:
    """Hand-build the WindowBuffer the pass should mint for one alloc.

    Each ``pld.alloc_window_buffer(size, *, name)`` materialises
    ``WindowBuffer(base=Var(name, PtrType), size=size, load_from_host=False,
    store_to_host=False)`` with ``name_hint`` inherited from the base Ptr Var.
    The ``size`` is the alloc's first arg passed through unchanged — the DSL
    emits it as a ``ConstInt`` (bare literal) or a ``Mul`` of two ``ConstInt``
    (``N * dtype.get_byte()`` form), always with dtype ``index``.
    """
    base = ir.Var(name, ir.PtrType(), ir.Span.unknown())
    return ir.WindowBuffer(base, size)


def _assert_scope_fields(
    actual: ir.CommDomainScopeStmt,
    expected_devices: list[int],
    expected_slots: list[ir.WindowBuffer],
) -> None:
    """Compare a produced ``CommDomainScopeStmt`` against hand-derived fields.

    ``enable_auto_mapping=True`` lets the freshly-built slot Vars in
    ``expected_slots`` match the pass-produced ones by structural isomorphism
    (name_hint + size + flags) rather than by ``shared_ptr`` identity.
    """
    assert list(actual.devices) == expected_devices
    assert len(actual.slots) == len(expected_slots)
    for actual_slot, expected_slot in zip(actual.slots, expected_slots, strict=True):
        ir.assert_structural_equal(actual_slot, expected_slot, enable_auto_mapping=True)


def _assert_rank2_signal_type(value: ir.Expr, world_size_name: str) -> None:
    signal_type = value.type
    assert isinstance(signal_type, ir.DistributedTensorType)
    assert len(signal_type.shape) == 2
    assert isinstance(signal_type.shape[0], ir.Var)
    assert signal_type.shape[0].name_hint == world_size_name
    assert isinstance(signal_type.shape[1], ir.ConstInt)
    assert signal_type.shape[1].value == 1


# ---------------------------------------------------------------------------
# Single alloc / ALL devices
# ---------------------------------------------------------------------------


def test_single_alloc_all_devices_world_size_loop():
    """``for r in pl.range(pld.world_size())`` + ``device=r`` ⇒ kAll."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(self, data: pld.DistributedTensor[[256], pl.FP32]):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            buf = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
            data = pld.window(buf, [256], dtype=pl.FP32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(data, device=r)
            return 0

    result = _apply(P)
    host = _get_func(result, "host_orch")

    # The world_size loop bound resolves the device descriptor to kAll, encoded
    # on the wire as an empty ``devices`` list, with a single slot for ``buf``.
    scopes = _get_comm_domain_scopes(host)
    assert len(scopes) == 1
    _assert_scope_fields(scopes[0], [], [_expected_slot("buf", _mul(256, 4))])

    # The view's window_buffer back-reference now points to the (same) slot.
    # Pointer-identity between the scope's slot and the view type's
    # window_buffer is a load-bearing invariant that structural comparison
    # alone cannot express.
    wb = scopes[0].slots[0]
    view_types = _view_var_types(host)
    assert len(view_types) == 1
    assert view_types[0].window_buffer is wb


def test_defining_call_type_carries_the_window_buffer_back_reference():
    """The re-typed view Var and the Call that defines it must agree (#1285).

    Phase 7 substitutes Var *references*, so without an explicit re-mint the
    ``pld.tensor.window`` Call keeps the deducer's ``window_buffer``-less
    ``DistributedTensorType`` while its LHS Var carries the back-reference —
    an asymmetric AssignStmt that ``AssignTypeSymmetry`` rejects. The same
    holds for a collective result Var that inherits the window's lineage.
    """

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(self, data: pld.DistributedTensor[[256], pl.FP32]):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            data_buf = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
            signal_buf = pld.alloc_window_buffer(4 * pl.INT32.get_byte())
            data = pld.window(data_buf, [256], dtype=pl.FP32)
            signal = pld.window(signal_buf, [4], dtype=pl.INT32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(data, device=r)
            pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum)
            return 0

    result = _apply(P)
    host = _get_func(result, "host_orch")

    # Both sides of every window assignment point at the SAME WindowBuffer
    # object the scope slot holds — structural equality alone cannot state this.
    window_assigns = _find_window_calls(host)
    assert len(window_assigns) == 2
    for stmt in window_assigns:
        assert isinstance(stmt.var.type, ir.DistributedTensorType)
        assert isinstance(stmt.value.type, ir.DistributedTensorType)
        assert stmt.value.type.window_buffer is stmt.var.type.window_buffer
        assert stmt.value.type.window_buffer is not None

    props = passes.IRPropertySet()
    props.insert(passes.IRProperty.AssignTypeSymmetry)
    assert passes.PropertyVerifierRegistry.verify(props, result) == []


def test_allreduce_signal_inherits_data_comm_domain():
    """Signal buffers used only by pld.tensor.allreduce still become scope slots."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(self, data: pld.DistributedTensor[[256], pl.FP32]):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            data_buf = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
            signal_buf = pld.alloc_window_buffer(4 * pl.INT32.get_byte())
            data = pld.window(data_buf, [256], dtype=pl.FP32)
            signal = pld.window(signal_buf, [4], dtype=pl.INT32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(data, device=r)
            pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum)
            return 0

    result = _apply(P)
    host = _get_func(result, "host_orch")
    scopes = _get_comm_domain_scopes(host)
    assert len(scopes) == 1
    _assert_scope_fields(
        scopes[0], [], [_expected_slot("data_buf", _mul(256, 4)), _expected_slot("signal_buf", _mul(4, 4))]
    )


def test_explicit_allreduce_assignment_keeps_user_signal():
    """Direct explicit-signal assignments stay explicit and join the data domain."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(self, data: pld.DistributedTensor[[256], pl.FP32]):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            data_buf = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
            signal_buf = pld.alloc_window_buffer(4 * pl.INT32.get_byte())
            data = pld.window(data_buf, [256], dtype=pl.FP32)
            signal = pld.window(signal_buf, [pld.world_size(), 1], dtype=pl.INT32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(data, device=r)
            data = pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum)
            return data

    result = _apply(P)
    host = _get_func(result, "host_orch")
    stmts = _flatten_stmts(host.body)
    synthetic_assigns = [
        stmt
        for stmt in stmts
        if isinstance(stmt, ir.AssignStmt) and stmt.var.name_hint.startswith("__allreduce_signal_")
    ]
    allreduces = [
        stmt.value
        for stmt in stmts
        if isinstance(stmt, ir.AssignStmt)
        and isinstance(stmt.value, ir.Call)
        and stmt.value.op.name == _OP_PLD_TENSOR_ALLREDUCE
    ]

    assert synthetic_assigns == []
    assert len(allreduces) == 1
    assert len(allreduces[0].args) == 2
    assert _as_var(allreduces[0].args[1]).name_hint == "signal__ssa_v0"

    scopes = _get_comm_domain_scopes(host)
    assert len(scopes) == 1
    assert [slot.base.name_hint for slot in scopes[0].slots] == ["data_buf__ssa_v0", "signal_buf__ssa_v0"]


def test_implicit_allreduce_signal_is_materialized_in_data_comm_domain():
    """A host allreduce without signal gets a private signal slot in the data domain."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(self, data: pld.DistributedTensor[[256], pl.FP32]):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            data_buf = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
            data = pld.window(data_buf, [256], dtype=pl.FP32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(data, device=r)
            data = pld.tensor.allreduce(data, op=pld.ReduceOp.Sum)
            return data

    result = _apply(P)
    host = _get_func(result, "host_orch")
    stmts = _flatten_stmts(host.body)
    signal_windows = [
        stmt
        for stmt in stmts
        if isinstance(stmt, ir.AssignStmt)
        and isinstance(stmt.value, ir.Call)
        and stmt.value.op.name == _OP_PLD_TENSOR_WINDOW
        and stmt.var.name_hint.startswith("__allreduce_signal_")
    ]
    allreduces = [
        stmt.value
        for stmt in stmts
        if isinstance(stmt, ir.AssignStmt)
        and isinstance(stmt.value, ir.Call)
        and stmt.value.op.name == _OP_PLD_TENSOR_ALLREDUCE
    ]

    assert len(signal_windows) == 1
    assert len(allreduces) == 1
    assert len(allreduces[0].args) == 2
    assert _as_var(allreduces[0].args[1]).name_hint == signal_windows[0].var.name_hint

    _assert_rank2_signal_type(signal_windows[0].var, "__allreduce_signal_world_size_0")

    scopes = _get_comm_domain_scopes(host)
    assert len(scopes) == 1
    # Shared signal binding is hoisted to the top of the body, so its slot precedes data_buf.
    assert [slot.base.name_hint for slot in scopes[0].slots] == [
        "__allreduce_signal_buf_0",
        "data_buf__ssa_v0",
    ]


def test_implicit_allreduce_over_distinct_subsets_gets_distinct_signals():
    """One shared signal per data-buffer lineage, not per function.

    Two data buffers dispatched to disjoint device subsets ({0,1} and {2,3}),
    each reduced implicitly, must each get its own synthesized signal. A single
    per-function shared signal would merge both subsets into the signal's
    coverage while each data buffer keeps its own domain, so
    LowerHostTensorCollectives would reject every call (data and signal in
    different comm-domain scopes).
    """

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(self, data: pld.DistributedTensor[[256], pl.FP32]):
            return data

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch_other(self, data: pld.DistributedTensor[[256], pl.FP32]):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            buf_a = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
            buf_b = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
            data_a = pld.window(buf_a, [256], dtype=pl.FP32)
            data_b = pld.window(buf_b, [256], dtype=pl.FP32)
            self.chip_orch(data_a, device=0)
            self.chip_orch(data_a, device=1)
            self.chip_orch_other(data_b, device=2)
            self.chip_orch_other(data_b, device=3)
            data_a = pld.tensor.allreduce(data_a, op=pld.ReduceOp.Sum)
            data_b = pld.tensor.allreduce(data_b, op=pld.ReduceOp.Sum)
            return data_a

    result = _apply(P)
    host = _get_func(result, "host_orch")
    stmts = _flatten_stmts(host.body)
    signal_windows = [
        stmt
        for stmt in stmts
        if isinstance(stmt, ir.AssignStmt)
        and isinstance(stmt.value, ir.Call)
        and stmt.value.op.name == _OP_PLD_TENSOR_WINDOW
        and stmt.var.name_hint.startswith("__allreduce_signal_")
    ]
    allreduces = [
        stmt.value
        for stmt in stmts
        if isinstance(stmt, ir.AssignStmt)
        and isinstance(stmt.value, ir.Call)
        and stmt.value.op.name == _OP_PLD_TENSOR_ALLREDUCE
    ]

    # Two distinct data lineages -> two distinct synthesized signals.
    assert len(signal_windows) == 2
    assert len(allreduces) == 2
    signal_names = {stmt.var.name_hint for stmt in signal_windows}
    assert len(signal_names) == 2
    for ar in allreduces:
        assert len(ar.args) == 2
        assert _as_var(ar.args[1]).name_hint in signal_names

    # After materialization, each (data, signal) pair lands in its own
    # comm-domain scope: {0,1} for buf_a and {2,3} for buf_b.
    scopes = _get_comm_domain_scopes(host)
    assert len(scopes) == 2
    assert list(scopes[0].devices) == [0, 1]
    assert [slot.base.name_hint for slot in scopes[0].slots] == ["__allreduce_signal_buf_0", "buf_a__ssa_v0"]
    assert list(scopes[1].devices) == [2, 3]
    assert [slot.base.name_hint for slot in scopes[1].slots] == ["__allreduce_signal_buf_1", "buf_b__ssa_v0"]


def test_optional_signal_allreduce_round_trips_before_materialization():
    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(self, data: pld.DistributedTensor[[256], pl.FP32]):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            data_buf = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
            data = pld.window(data_buf, [256], dtype=pl.FP32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(data, device=r)
            data = pld.tensor.allreduce(data, op=pld.ReduceOp.Sum)
            return data

    printed = P.as_python()
    reparsed = cast(ir.Program, pl.parse(printed))
    host = _get_func(reparsed, "host_orch")
    allreduces = [
        stmt.value
        for stmt in _flatten_stmts(host.body)
        if isinstance(stmt, ir.AssignStmt)
        and isinstance(stmt.value, ir.Call)
        and stmt.value.op.name == _OP_PLD_TENSOR_ALLREDUCE
    ]

    assert "pld.tensor.allreduce(" in printed
    assert "op=" in printed
    assert "__allreduce_signal_" not in printed
    assert len(allreduces) == 1
    allreduce = _as_call(allreduces[0])
    assert len(allreduce.args) == 1
    assert allreduce.kwargs["op"] == int(pld.ReduceOp.Sum)


def test_synthesize_allreduce_signals_normalizes_host_allreduce():
    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(self, data: pld.DistributedTensor[[256], pl.FP32]):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            data_buf = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
            data = pld.window(data_buf, [256], dtype=pl.FP32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(data, device=r)
            data = pld.tensor.allreduce(data, op=pld.ReduceOp.Sum)
            return data

    synthesized = _synthesize(P)
    host = _get_func(synthesized, "host_orch")
    stmts = _flatten_stmts(host.body)
    allreduces = [
        stmt.value
        for stmt in stmts
        if isinstance(stmt, ir.AssignStmt)
        and isinstance(stmt.value, ir.Call)
        and stmt.value.op.name == _OP_PLD_TENSOR_ALLREDUCE
    ]
    signal_windows = [
        stmt
        for stmt in stmts
        if isinstance(stmt, ir.AssignStmt)
        and isinstance(stmt.value, ir.Call)
        and stmt.value.op.name == _OP_PLD_TENSOR_WINDOW
        and stmt.var.name_hint.startswith("__allreduce_signal_")
    ]

    assert len(allreduces) == 1
    assert len(allreduces[0].args) == 2
    assert len(signal_windows) == 1
    assert _as_var(allreduces[0].args[1]).name_hint == signal_windows[0].var.name_hint


def test_synthesize_multicore_allreduce_signal_uses_core_lanes_and_bytes():
    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(self, data: pld.DistributedTensor[[256], pl.FP32]):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            data_buf = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
            data = pld.window(data_buf, [256], dtype=pl.FP32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(data, device=r)
            data = pld.tensor.allreduce(data, core_num=4)
            return data

    synthesized = _synthesize(P)
    host = _get_func(synthesized, "host_orch")
    stmts = _flatten_stmts(host.body)
    window_stmt = next(
        stmt
        for stmt in stmts
        if isinstance(stmt, ir.AssignStmt)
        and isinstance(stmt.value, ir.Call)
        and stmt.value.op.name == _OP_PLD_TENSOR_WINDOW
        and stmt.var.name_hint.startswith("__allreduce_signal_")
    )
    alloc_stmt = next(
        stmt
        for stmt in stmts
        if isinstance(stmt, ir.AssignStmt)
        and isinstance(stmt.value, ir.Call)
        and stmt.value.op.name == _OP_PLD_TENSOR_ALLOC_WINDOW_BUFFER
        and stmt.var.name_hint.startswith("__allreduce_signal_buf_")
    )
    allreduce = next(
        stmt.value
        for stmt in stmts
        if isinstance(stmt, ir.AssignStmt)
        and isinstance(stmt.value, ir.Call)
        and stmt.value.op.name == _OP_PLD_TENSOR_ALLREDUCE
    )

    signal_type = window_stmt.var.type
    assert isinstance(signal_type, ir.DistributedTensorType)
    assert isinstance(signal_type.shape[1], ir.ConstInt)
    assert signal_type.shape[1].value == 4
    assert allreduce.kwargs["core_num"] == 4

    assert isinstance(alloc_stmt.value, ir.Call)
    alloc_size = alloc_stmt.value.args[0]
    assert isinstance(alloc_size, ir.Mul)
    assert isinstance(alloc_size.right, ir.ConstInt)
    assert alloc_size.right.value == pl.INT32.get_byte()
    assert isinstance(alloc_size.left, ir.Mul)
    assert isinstance(alloc_size.left.left, ir.Var)
    assert alloc_size.left.left.name_hint.startswith("__allreduce_signal_world_size_")
    assert isinstance(alloc_size.left.right, ir.ConstInt)
    assert alloc_size.left.right.value == 4


def test_implicit_allreduce_core_num_widening_sizes_signal_to_max():
    """Two implicit allreduces over the same data buffer with different
    core_num share ONE synthesized signal, sized to the widest core_num (the
    allow_wider_lanes contract in LowerHostTensorCollectives)."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(self, data: pld.DistributedTensor[[256], pl.FP32]):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            data_buf = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
            data = pld.window(data_buf, [256], dtype=pl.FP32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(data, device=r)
            a = pld.tensor.allreduce(data, core_num=1)
            b = pld.tensor.allreduce(data, core_num=4)
            return a, b

    result = _apply(P)
    host = _get_func(result, "host_orch")
    stmts = _flatten_stmts(host.body)
    signal_windows = [
        stmt
        for stmt in stmts
        if isinstance(stmt, ir.AssignStmt)
        and isinstance(stmt.value, ir.Call)
        and stmt.value.op.name == _OP_PLD_TENSOR_WINDOW
        and stmt.var.name_hint.startswith("__allreduce_signal_")
    ]
    allreduces = [
        stmt
        for stmt in stmts
        if isinstance(stmt, ir.AssignStmt)
        and isinstance(stmt.value, ir.Call)
        and stmt.value.op.name == _OP_PLD_TENSOR_ALLREDUCE
    ]
    assert len(signal_windows) == 1
    assert len(allreduces) == 2
    signal_type = signal_windows[0].var.type
    assert isinstance(signal_type, ir.DistributedTensorType)
    assert isinstance(signal_type.shape[1], ir.ConstInt)
    assert signal_type.shape[1].value == 4  # widened to the max core_num
    # Both calls reuse the one shared signal.
    signals = {_as_var(_as_call(stmt.value).args[1]).name_hint for stmt in allreduces}
    assert signals == {signal_windows[0].var.name_hint}


def test_synthesized_allreduce_signal_round_trips_after_materialization():
    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(self, data: pld.DistributedTensor[[256], pl.FP32]):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            data_buf = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
            data = pld.window(data_buf, [256], dtype=pl.FP32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(data, device=r)
            data = pld.tensor.allreduce(data, op=pld.ReduceOp.Sum)
            return data

    materialized = _apply(P)
    printed = materialized.as_python()
    reparsed = cast(ir.Program, pl.parse(printed))
    host = _get_func(reparsed, "host_orch")
    stmts = _flatten_stmts(host.body)
    allocs = [
        stmt
        for stmt in stmts
        if isinstance(stmt, ir.AssignStmt)
        and isinstance(stmt.value, ir.Call)
        and stmt.value.op.name == _OP_PLD_TENSOR_ALLOC_WINDOW_BUFFER
        and stmt.var.name_hint.startswith("__allreduce_signal_buf_")
    ]
    windows = [
        stmt
        for stmt in stmts
        if isinstance(stmt, ir.AssignStmt)
        and isinstance(stmt.value, ir.Call)
        and stmt.value.op.name == _OP_PLD_TENSOR_WINDOW
        and stmt.var.name_hint.startswith("__allreduce_signal_")
    ]
    allreduces = [
        stmt.value
        for stmt in stmts
        if isinstance(stmt, ir.AssignStmt)
        and isinstance(stmt.value, ir.Call)
        and stmt.value.op.name == _OP_PLD_TENSOR_ALLREDUCE
    ]

    printed_ast = ast.parse(printed)
    alloc_call_nodes = [
        node
        for node in ast.walk(printed_ast)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "alloc_window_buffer"
    ]
    assert alloc_call_nodes
    for node in alloc_call_nodes:
        assert "name" in {keyword.arg for keyword in node.keywords}
    assert len(allocs) == 1
    alloc_call = _as_call(allocs[0].value)
    assert alloc_call.kwargs["name"] == allocs[0].var.name_hint
    assert len(windows) == 1
    window_call = _as_call(windows[0].value)
    assert _as_var(window_call.args[0]).name_hint == allocs[0].var.name_hint
    signal_type = windows[0].var.type
    assert isinstance(signal_type, ir.DistributedTensorType)
    assert len(signal_type.shape) == 2
    assert isinstance(signal_type.shape[0], ir.Var)
    assert signal_type.shape[0].name_hint.startswith("__allreduce_signal_world_size_")
    assert isinstance(signal_type.shape[1], ir.ConstInt)
    assert signal_type.shape[1].value == 1
    assert len(allreduces) == 1
    assert len(allreduces[0].args) == 2
    assert _as_var(allreduces[0].args[1]).name_hint == windows[0].var.name_hint


def test_implicit_allreduce_signal_names_are_program_unique():
    program = cast(
        ir.Program,
        pl.parse(
            textwrap.dedent(
                """
            @pl.program
            class P:
                @pl.function(type=pl.FunctionType.Orchestration)
                def chip_orch(self, data: pld.DistributedTensor[[256], pl.FP32]):
                    return data

                @pl.function(type=pl.FunctionType.Orchestration)
                def chip_orch_other(self, data: pld.DistributedTensor[[256], pl.FP32]):
                    return data

                @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
                def host_orch_a(self):
                    data_buf = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
                    __allreduce_signal_world_size_0 = pld.world_size()
                    __allreduce_signal_buf_0 = pld.alloc_window_buffer(pld.world_size() * pl.INT32.get_byte())
                    __allreduce_signal_0 = pld.window(
                        __allreduce_signal_buf_0,
                        [pld.world_size(), 1],
                        dtype=pl.INT32,
                    )
                    __allreduce_result_0 = pld.window(data_buf, [256], dtype=pl.FP32)
                    data = pld.window(data_buf, [256], dtype=pl.FP32)
                    for r in pl.range(pld.world_size()):
                        self.chip_orch(data, device=r)
                    data = pld.tensor.allreduce(data, __allreduce_signal_0, op=pld.ReduceOp.Sum)
                    data = pld.tensor.allreduce(data, op=pld.ReduceOp.Sum)
                    return data

                @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
                def host_orch_b(self):
                    data_buf_b = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
                    data = pld.window(data_buf_b, [256], dtype=pl.FP32)
                    for r in pl.range(pld.world_size()):
                        self.chip_orch_other(data, device=r)
                    data = pld.tensor.allreduce(data, op=pld.ReduceOp.Sum)
                    return data
            """
            )
        ),
    )

    # NOTE: deliberately NOT `_apply` (which runs ConvertToSSA first) — this
    # test pre-defines user vars named `__allreduce_*`, which SSA cannot rename
    # (reserved `__` delimiter in auto-name bases) and which are exactly what
    # the synthesized-name-uniqueness contract must dodge.
    result = cast(ir.Program, passes.materialize_comm_domain_scopes()(_synthesize(program)))
    hosts = [_get_func(result, "host_orch_a"), _get_func(result, "host_orch_b")]
    allreduce_signal_args = [
        _as_var(_as_call(stmt.value).args[1]).name_hint
        for host in hosts
        for stmt in _flatten_stmts(host.body)
        if isinstance(stmt, ir.AssignStmt)
        and isinstance(stmt.value, ir.Call)
        and stmt.value.op.name == _OP_PLD_TENSOR_ALLREDUCE
    ]
    world_size_vars = [
        stmt.var.name_hint
        for host in hosts
        for stmt in _flatten_stmts(host.body)
        if isinstance(stmt, ir.AssignStmt) and stmt.var.name_hint.startswith("__allreduce_signal_world_size_")
    ]
    signal_slots = [
        slot.base.name_hint
        for host in hosts
        for scope in _get_comm_domain_scopes(host)
        for slot in scope.slots
        if slot.base.name_hint.startswith("__allreduce_signal_buf_")
    ]

    assert allreduce_signal_args == [
        "__allreduce_signal_0",
        "__allreduce_signal_1",
        "__allreduce_signal_2",
    ]
    # One shared signal per data-buffer lineage (one per function here, since
    # each reduces a single buffer), hoisted to the top of the body:
    # host_orch_a's shared binding is __allreduce_signal_1 (skipping the
    # user-pre-used _0 names) and precedes the user's hand-written signal.
    assert world_size_vars == [
        "__allreduce_signal_world_size_1",
        "__allreduce_signal_world_size_0",
        "__allreduce_signal_world_size_2",
    ]
    assert signal_slots == [
        "__allreduce_signal_buf_1",
        "__allreduce_signal_buf_0",
        "__allreduce_signal_buf_2",
    ]
    assert len(signal_slots) == len(set(signal_slots))


def test_consecutive_implicit_allreduces_share_one_signal_slot():
    """Consecutive implicit allreduces over the same buffer share ONE
    synthesized signal slot (safe since the host builtin kernels self-clear
    their barrier cells after each call)."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(self, data: pld.DistributedTensor[[256], pl.FP32]):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            data_buf = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
            data = pld.window(data_buf, [256], dtype=pl.FP32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(data, device=r)
            data = pld.tensor.allreduce(data, op=pld.ReduceOp.Sum)
            data = pld.tensor.allreduce(data, op=pld.ReduceOp.Sum)
            return data

    result = _apply(P)
    host = _get_func(result, "host_orch")
    scopes = _get_comm_domain_scopes(host)
    assert len(scopes) == 1
    # Shared binding is hoisted to the top, so the signal buffer slot precedes data_buf.
    assert [slot.base.name_hint for slot in scopes[0].slots] == [
        "__allreduce_signal_buf_0",
        "data_buf__ssa_v0",
    ]
    allreduces = [
        _as_call(stmt.value)
        for stmt in _flatten_stmts(host.body)
        if isinstance(stmt, ir.AssignStmt)
        and isinstance(stmt.value, ir.Call)
        and stmt.value.op.name == _OP_PLD_TENSOR_ALLREDUCE
    ]
    assert len(allreduces) == 2
    assert _as_var(allreduces[0].args[1]).name_hint == "__allreduce_signal_0"
    assert _as_var(allreduces[1].args[1]).name_hint == "__allreduce_signal_0"


def test_implicit_allreduce_inside_loop_uses_shared_signal():
    """An implicit-signal allreduce inside a for loop is accepted (previously
    rejected as a single-use signal) and reuses the buffer's one shared
    signal, hoisted outside the loop."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(self, data: pld.DistributedTensor[[256], pl.FP32]):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            data_buf = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
            data = pld.window(data_buf, [256], dtype=pl.FP32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(data, device=r)
            for it in pl.range(2):
                data = pld.tensor.allreduce(data, op=pld.ReduceOp.Sum)
            return data

    result = _apply(P)
    host = _get_func(result, "host_orch")
    signal_windows = [
        stmt
        for stmt in _flatten_stmts(host.body)
        if isinstance(stmt, ir.AssignStmt)
        and isinstance(stmt.value, ir.Call)
        and stmt.value.op.name == _OP_PLD_TENSOR_WINDOW
        and stmt.var.name_hint.startswith("__allreduce_signal_")
    ]
    allreduces = [
        stmt
        for stmt in _flatten_stmts(host.body)
        if isinstance(stmt, ir.AssignStmt)
        and isinstance(stmt.value, ir.Call)
        and stmt.value.op.name == _OP_PLD_TENSOR_ALLREDUCE
    ]
    # One shared signal hoisted to the top; the (single) in-loop allreduce reuses it.
    assert len(signal_windows) == 1
    assert len(allreduces) == 1
    assert _as_var(_as_call(allreduces[0].value).args[1]).name_hint == signal_windows[0].var.name_hint


def test_synthesize_allreduce_signals_reserves_existing_alloc_names():
    span = ir.Span.unknown()
    existing_alloc = dist_tensor_ops.alloc_window_buffer(
        16,
        name="__allreduce_signal_buf_0",
        span=span,
    )
    existing_buf = ir.Var("user_named_buf", existing_alloc.type, span)
    existing_assign = ir.AssignStmt(existing_buf, existing_alloc, span)

    data_type = ir.DistributedTensorType([ir.ConstInt(256, DataType.INT64, span)], DataType.FP32)
    data = ir.Var("data", data_type, span)
    allreduce = dist_tensor_ops.allreduce(data, op=ir.ReduceOp.Sum, span=span)
    allreduce_assign = ir.AssignStmt(ir.Var("data_out", allreduce.type, span), allreduce, span)

    host = ir.Function(
        "host_orch",
        [data],
        [],
        ir.SeqStmts([existing_assign, allreduce_assign], span),
        span,
        ir.FunctionType.Opaque,
        ir.Level.HOST,
        ir.Role.Orchestrator,
    )
    synthesized = _synthesize(ir.Program([host], "P", span))
    alloc_names = [
        stmt.value.kwargs["name"]
        for stmt in _flatten_stmts(_get_func(synthesized, "host_orch").body)
        if isinstance(stmt, ir.AssignStmt)
        and isinstance(stmt.value, ir.Call)
        and stmt.value.op.name == _OP_PLD_TENSOR_ALLOC_WINDOW_BUFFER
    ]

    # Shared binding is hoisted to the top, so the synthesized alloc precedes the
    # user's pre-existing __allreduce_signal_buf_0 alloc.
    assert alloc_names == ["__allreduce_signal_buf_1", "__allreduce_signal_buf_0"]


def test_ssa_allreduce_result_keeps_source_window_lineage():
    """Post-SSA allreduce result Vars still resolve to the original data window."""

    program = cast(
        ir.Program,
        pl.parse(
            textwrap.dedent(
                """
                @pl.program
                class P:
                    @pl.function(type=pl.FunctionType.Orchestration)
                    def chip_orch(self, data: pld.DistributedTensor[[256], pl.FP32]):
                        return data

                    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
                    def host_orch(self):
                        data_buf = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
                        signal_buf = pld.alloc_window_buffer(4 * pl.INT32.get_byte())
                        data = pld.window(data_buf, [256], dtype=pl.FP32)
                        signal = pld.window(signal_buf, [pld.world_size(), 1], dtype=pl.INT32)
                        for r in pl.range(pld.world_size()):
                            self.chip_orch(data, device=r)
                        data_1 = pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum)
                        data_2 = pld.tensor.allreduce(data_1, op=pld.ReduceOp.Sum)
                        return data_2
                """
            )
        ),
    )

    result = _apply(program)
    host = _get_func(result, "host_orch")
    scopes = _get_comm_domain_scopes(host)

    assert len(scopes) == 1
    # Shared signal binding is hoisted to the top of the body, so its slot precedes data_buf.
    assert [slot.base.name_hint for slot in scopes[0].slots] == [
        "__allreduce_signal_buf_0",
        "data_buf__ssa_v0",
        "signal_buf__ssa_v0",
    ]


def test_return_implicit_allreduce_is_lifted_for_host_lowering():
    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(self, data: pld.DistributedTensor[[256], pl.FP32]):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            data_buf = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
            data = pld.window(data_buf, [256], dtype=pl.FP32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(data, device=r)
            return pld.tensor.allreduce(data, op=pld.ReduceOp.Sum)

    result = _apply(P)
    host = _get_func(result, "host_orch")
    stmts = _flatten_stmts(host.body)
    allreduce_assigns = [
        stmt
        for stmt in stmts
        if isinstance(stmt, ir.AssignStmt)
        and isinstance(stmt.value, ir.Call)
        and stmt.value.op.name == _OP_PLD_TENSOR_ALLREDUCE
    ]
    returns = [stmt for stmt in stmts if isinstance(stmt, ir.ReturnStmt)]

    assert len(allreduce_assigns) == 1
    assert allreduce_assigns[0].var.name_hint.startswith("__allreduce_result_")
    allreduce_call = _as_call(allreduce_assigns[0].value)
    assert len(allreduce_call.args) == 2
    assert _as_var(allreduce_call.args[1]).name_hint == "__allreduce_signal_0"
    _assert_rank2_signal_type(allreduce_call.args[1], "__allreduce_signal_world_size_0")
    assert len(returns) == 1
    assert isinstance(returns[0].value[0], ir.Var)
    assert returns[0].value[0] is allreduce_assigns[0].var

    scopes = _get_comm_domain_scopes(host)
    assert len(scopes) == 1
    # Shared signal binding is hoisted to the top of the body, so its slot precedes data_buf.
    assert [slot.base.name_hint for slot in scopes[0].slots] == [
        "__allreduce_signal_buf_0",
        "data_buf__ssa_v0",
    ]


def test_return_explicit_allreduce_is_lifted_for_host_lowering():
    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(self, data: pld.DistributedTensor[[256], pl.FP32]):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            data_buf = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
            signal_buf = pld.alloc_window_buffer(4 * pl.INT32.get_byte())
            data = pld.window(data_buf, [256], dtype=pl.FP32)
            signal = pld.window(signal_buf, [pld.world_size(), 1], dtype=pl.INT32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(data, device=r)
            return pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum)

    result = _apply(P)
    host = _get_func(result, "host_orch")
    stmts = _flatten_stmts(host.body)
    synthetic_assigns = [
        stmt
        for stmt in stmts
        if isinstance(stmt, ir.AssignStmt) and stmt.var.name_hint.startswith("__allreduce_signal_")
    ]
    allreduce_assigns = [
        stmt
        for stmt in stmts
        if isinstance(stmt, ir.AssignStmt)
        and isinstance(stmt.value, ir.Call)
        and stmt.value.op.name == _OP_PLD_TENSOR_ALLREDUCE
    ]
    returns = [stmt for stmt in stmts if isinstance(stmt, ir.ReturnStmt)]

    assert synthetic_assigns == []
    assert len(allreduce_assigns) == 1
    assert allreduce_assigns[0].var.name_hint.startswith("__allreduce_result_")
    assert _as_var(_as_call(allreduce_assigns[0].value).args[1]).name_hint == "signal__ssa_v0"
    assert len(returns) == 1
    assert isinstance(returns[0].value[0], ir.Var)
    assert returns[0].value[0] is allreduce_assigns[0].var


def test_implicit_allreduce_eval_stmt_gets_signal():
    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(self, data: pld.DistributedTensor[[256], pl.FP32]):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            data_buf = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
            data = pld.window(data_buf, [256], dtype=pl.FP32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(data, device=r)
            pld.tensor.allreduce(data, op=pld.ReduceOp.Sum)
            return 0

    result = _apply(P)
    host = _get_func(result, "host_orch")
    stmts = _flatten_stmts(host.body)
    allreduces = [
        stmt.expr
        for stmt in stmts
        if isinstance(stmt, ir.EvalStmt)
        and isinstance(stmt.expr, ir.Call)
        and stmt.expr.op.name == _OP_PLD_TENSOR_ALLREDUCE
    ]

    assert len(allreduces) == 1
    assert len(allreduces[0].args) == 2
    assert _as_var(allreduces[0].args[1]).name_hint == "__allreduce_signal_0"
    _assert_rank2_signal_type(allreduces[0].args[1], "__allreduce_signal_world_size_0")

    scopes = _get_comm_domain_scopes(host)
    assert len(scopes) == 1
    # Shared signal binding is hoisted to the top of the body, so its slot precedes data_buf.
    assert [slot.base.name_hint for slot in scopes[0].slots] == [
        "__allreduce_signal_buf_0",
        "data_buf__ssa_v0",
    ]


def test_explicit_allreduce_eval_stmt_keeps_user_signal():
    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(self, data: pld.DistributedTensor[[256], pl.FP32]):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            data_buf = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
            signal_buf = pld.alloc_window_buffer(4 * pl.INT32.get_byte())
            data = pld.window(data_buf, [256], dtype=pl.FP32)
            signal = pld.window(signal_buf, [pld.world_size(), 1], dtype=pl.INT32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(data, device=r)
            pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum)
            return 0

    result = _apply(P)
    host = _get_func(result, "host_orch")
    stmts = _flatten_stmts(host.body)
    synthetic_assigns = [
        stmt
        for stmt in stmts
        if isinstance(stmt, ir.AssignStmt) and stmt.var.name_hint.startswith("__allreduce_signal_")
    ]
    allreduces = [
        stmt.expr
        for stmt in stmts
        if isinstance(stmt, ir.EvalStmt)
        and isinstance(stmt.expr, ir.Call)
        and stmt.expr.op.name == _OP_PLD_TENSOR_ALLREDUCE
    ]

    assert synthetic_assigns == []
    assert len(allreduces) == 1
    assert _as_var(allreduces[0].args[1]).name_hint == "signal__ssa_v0"


def test_implicit_allreduce_in_while_loop_uses_shared_signal():
    """An implicit-signal allreduce inside a while loop is accepted: ONE shared
    signal is synthesized (hoisted before the loop) and the in-loop call reuses
    it — safe because the host builtin kernels self-clear their barrier cells
    after each call."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(self, data: pld.DistributedTensor[[256], pl.FP32]):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            data_buf = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
            data = pld.window(data_buf, [256], dtype=pl.FP32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(data, device=r)
            for r in pl.range(pld.world_size()):
                self.chip_orch(data, device=r)
            while True:
                data = pld.tensor.allreduce(data, op=pld.ReduceOp.Sum)
            return data

    synthesized = _synthesize(P)
    host = _get_func(synthesized, "host_orch")
    signal_windows = [
        stmt
        for stmt in _flatten_stmts(host.body)
        if isinstance(stmt, ir.AssignStmt)
        and isinstance(stmt.value, ir.Call)
        and stmt.value.op.name == _OP_PLD_TENSOR_WINDOW
        and stmt.var.name_hint.startswith("__allreduce_signal_")
    ]
    allreduces = [
        stmt
        for stmt in _flatten_stmts(host.body)
        if isinstance(stmt, ir.AssignStmt)
        and isinstance(stmt.value, ir.Call)
        and stmt.value.op.name == _OP_PLD_TENSOR_ALLREDUCE
    ]
    assert len(signal_windows) == 1
    assert len(allreduces) == 1
    assert _as_var(_as_call(allreduces[0].value).args[1]).name_hint == signal_windows[0].var.name_hint


def test_explicit_allreduce_in_loop_is_accepted():
    """An explicit signal is reusable across iterations, so a `for` loop is allowed."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(self, data: pld.DistributedTensor[[256], pl.FP32]):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            data_buf = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
            signal_buf = pld.alloc_window_buffer(4 * pl.INT32.get_byte())
            data = pld.window(data_buf, [256], dtype=pl.FP32)
            signal = pld.window(signal_buf, [pld.world_size(), 1], dtype=pl.INT32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(data, device=r)
            for _ in pl.range(2):
                data = pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum)
            return data

    # An explicit signal needs no synthesis, and the lowered kernel restores its
    # cells to zero before returning, so carrying it across iterations is safe.
    _apply(P)


def test_explicit_allreduce_in_while_loop_is_accepted():
    """Same as the `for` case: only a synthesized signal is loop-bound."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(self, data: pld.DistributedTensor[[256], pl.FP32]):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            data_buf = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
            signal_buf = pld.alloc_window_buffer(4 * pl.INT32.get_byte())
            data = pld.window(data_buf, [256], dtype=pl.FP32)
            signal = pld.window(signal_buf, [pld.world_size(), 1], dtype=pl.INT32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(data, device=r)
            while True:
                data = pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum)
            return data

    # An explicit signal needs no synthesis, and the lowered kernel restores its
    # cells to zero before returning, so carrying it across iterations is safe.
    _apply(P)


def test_all_to_all_v_signal_and_recv_counts_inherit_data_comm_domain():
    """signal and recv_counts (both device-descriptor-less) each register their
    own CollectiveConsumer entry sharing all_to_all_v's target alloc, so both
    inherit target's device coverage — send_counts does not (its coverage comes
    from its own dispatch site, like symmetric all_to_all's `input`).

    chip_orch deliberately does NOT take `signal`/`recv` as params (unlike the
    lowering-rule tests) — if it did, the per-rank dispatch loop would give
    them their own device descriptors directly, and this test would pass even
    with the two new CollectiveConsumer entries removed. Matches
    test_allreduce_signal_inherits_data_comm_domain's discipline of leaving
    the signal-like arg out of the dispatch site so only the collective's own
    inheritance mechanism can supply its coverage.
    """

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            inp: pld.DistributedTensor[[8, 64], pl.FP32],
            data: pld.DistributedTensor[[8, 64], pl.FP32],
            counts: pld.DistributedTensor[[4, 1], pl.INT32],
        ):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            input_buf = pld.alloc_window_buffer(8 * 64 * pl.FP32.get_byte())
            data_buf = pld.alloc_window_buffer(8 * 64 * pl.FP32.get_byte())
            signal_buf = pld.alloc_window_buffer(4 * pl.INT32.get_byte())
            counts_buf = pld.alloc_window_buffer(4 * pl.INT32.get_byte())
            recv_buf = pld.alloc_window_buffer(4 * pl.INT32.get_byte())
            inp = pld.window(input_buf, [8, 64], dtype=pl.FP32)
            data = pld.window(data_buf, [8, 64], dtype=pl.FP32)
            counts = pld.window(counts_buf, [4, 1], dtype=pl.INT32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(inp, data, counts, device=r)
            signal = pld.window(signal_buf, [4, 1], dtype=pl.INT32)
            recv = pld.window(recv_buf, [4, 1], dtype=pl.INT32)
            pld.tensor.all_to_all_v(inp, data, signal, counts, recv)
            return 0

    result = _apply(P)
    host = _get_func(result, "host_orch")
    scopes = _get_comm_domain_scopes(host)
    assert len(scopes) == 1
    _assert_scope_fields(
        scopes[0],
        [],
        [
            _expected_slot("input_buf", _mul(8 * 64, 4)),
            _expected_slot("data_buf", _mul(8 * 64, 4)),
            _expected_slot("signal_buf", _mul(4, 4)),
            _expected_slot("counts_buf", _mul(4, 4)),
            _expected_slot("recv_buf", _mul(4, 4)),
        ],
    )


def test_all_to_all_v_in_loop_is_rejected():
    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            inp: pld.DistributedTensor[[8, 64], pl.FP32],
            data: pld.DistributedTensor[[8, 64], pl.FP32],
            sig: pld.DistributedTensor[[4, 1], pl.INT32],
            counts: pld.DistributedTensor[[4, 1], pl.INT32],
            recv: pld.DistributedTensor[[4, 1], pl.INT32],
        ):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            input_buf = pld.alloc_window_buffer(8 * 64 * pl.FP32.get_byte())
            data_buf = pld.alloc_window_buffer(8 * 64 * pl.FP32.get_byte())
            signal_buf = pld.alloc_window_buffer(4 * pl.INT32.get_byte())
            counts_buf = pld.alloc_window_buffer(4 * pl.INT32.get_byte())
            recv_buf = pld.alloc_window_buffer(4 * pl.INT32.get_byte())
            inp = pld.window(input_buf, [8, 64], dtype=pl.FP32)
            data = pld.window(data_buf, [8, 64], dtype=pl.FP32)
            signal = pld.window(signal_buf, [4, 1], dtype=pl.INT32)
            counts = pld.window(counts_buf, [4, 1], dtype=pl.INT32)
            recv = pld.window(recv_buf, [4, 1], dtype=pl.INT32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(inp, data, signal, counts, recv, device=r)
            for _ in pl.range(2):
                data = pld.tensor.all_to_all_v(inp, data, signal, counts, recv)
            return 0

    with pytest.raises(ValueError, match="all_to_all_v is not supported inside a for/while loop"):
        _apply(P)


def test_all_to_all_v_in_while_loop_is_rejected():
    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            inp: pld.DistributedTensor[[8, 64], pl.FP32],
            data: pld.DistributedTensor[[8, 64], pl.FP32],
            sig: pld.DistributedTensor[[4, 1], pl.INT32],
            counts: pld.DistributedTensor[[4, 1], pl.INT32],
            recv: pld.DistributedTensor[[4, 1], pl.INT32],
        ):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            input_buf = pld.alloc_window_buffer(8 * 64 * pl.FP32.get_byte())
            data_buf = pld.alloc_window_buffer(8 * 64 * pl.FP32.get_byte())
            signal_buf = pld.alloc_window_buffer(4 * pl.INT32.get_byte())
            counts_buf = pld.alloc_window_buffer(4 * pl.INT32.get_byte())
            recv_buf = pld.alloc_window_buffer(4 * pl.INT32.get_byte())
            inp = pld.window(input_buf, [8, 64], dtype=pl.FP32)
            data = pld.window(data_buf, [8, 64], dtype=pl.FP32)
            signal = pld.window(signal_buf, [4, 1], dtype=pl.INT32)
            counts = pld.window(counts_buf, [4, 1], dtype=pl.INT32)
            recv = pld.window(recv_buf, [4, 1], dtype=pl.INT32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(inp, data, signal, counts, recv, device=r)
            while True:
                data = pld.tensor.all_to_all_v(inp, data, signal, counts, recv)
            return 0

    with pytest.raises(ValueError, match="all_to_all_v is not supported inside a for/while loop"):
        _apply(P)


def test_nested_explicit_allreduce_expression_is_rejected():
    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(self, data: pld.DistributedTensor[[256], pl.FP32]):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            data_buf = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
            signal_buf = pld.alloc_window_buffer(4 * pl.INT32.get_byte())
            data = pld.window(data_buf, [256], dtype=pl.FP32)
            signal = pld.window(signal_buf, [4], dtype=pl.INT32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(data, device=r)
            wrapped = (pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum),)
            return wrapped

    with pytest.raises(ValueError, match="must be a direct assignment"):
        _apply(P)


def test_nested_implicit_allreduce_expression_is_rejected():
    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(self, data: pld.DistributedTensor[[256], pl.FP32]):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            data_buf = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
            data = pld.window(data_buf, [256], dtype=pl.FP32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(data, device=r)
            wrapped = (pld.tensor.allreduce(data, op=pld.ReduceOp.Sum),)
            return wrapped

    with pytest.raises(ValueError, match="must be a direct assignment"):
        _apply(P)


# ---------------------------------------------------------------------------
# Single alloc / explicit ConstInt subset
# ---------------------------------------------------------------------------


def test_single_alloc_subset_const_int_devices():
    """Two separate ``device=0`` / ``device=1`` dispatches ⇒ subset {0,1}."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(self, data: pld.DistributedTensor[[256], pl.FP32]):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            buf = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
            data = pld.window(buf, [256], dtype=pl.FP32)
            self.chip_orch(data, device=0)
            self.chip_orch(data, device=1)
            return 0

    result = _apply(P)
    host = _get_func(result, "host_orch")
    # Two ConstInt dispatches contribute {0} and {1}; merged into subset {0,1}
    # over a single ``buf`` slot.
    scopes = _get_comm_domain_scopes(host)
    assert len(scopes) == 1
    _assert_scope_fields(scopes[0], [0, 1], [_expected_slot("buf", _mul(256, 4))])


# ---------------------------------------------------------------------------
# Single alloc / bounded loop
# ---------------------------------------------------------------------------


def test_single_alloc_bounded_loop_devices():
    """``for r in pl.range(2)`` + ``device=r`` ⇒ subset {0, 1}."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(self, data: pld.DistributedTensor[[256], pl.FP32]):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            buf = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
            data = pld.window(buf, [256], dtype=pl.FP32)
            for r in pl.range(2):
                self.chip_orch(data, device=r)
            return 0

    result = _apply(P)
    host = _get_func(result, "host_orch")
    # ``pl.range(2)`` expands the induction-var descriptor to subset {0, 1}.
    scopes = _get_comm_domain_scopes(host)
    assert len(scopes) == 1
    _assert_scope_fields(scopes[0], [0, 1], [_expected_slot("buf", _mul(256, 4))])


# ---------------------------------------------------------------------------
# Two allocs / same descriptor → one scope, two slots in alloc order
# ---------------------------------------------------------------------------


def test_two_allocs_same_descriptor_one_group():
    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            data: pld.DistributedTensor[[256], pl.FP32],
            signal: pld.DistributedTensor[[8], pl.INT32],
        ):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            buf_data = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
            buf_signal = pld.alloc_window_buffer(8 * pl.INT32.get_byte())
            data = pld.window(buf_data, [256], dtype=pl.FP32)
            signal = pld.window(buf_signal, [8], dtype=pl.INT32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(data, signal, device=r)
            return 0

    result = _apply(P)
    host = _get_func(result, "host_orch")
    # Both allocs are dispatched over the same world_size loop ⇒ identical kAll
    # descriptor ⇒ a single scope whose slots follow source/alloc order
    # (buf_data, then buf_signal) per Phase-6 clustering.
    scopes = _get_comm_domain_scopes(host)
    assert len(scopes) == 1
    _assert_scope_fields(
        scopes[0],
        [],  # kAll
        [_expected_slot("buf_data", _mul(256, 4)), _expected_slot("buf_signal", _mul(8, 4))],
    )


# ---------------------------------------------------------------------------
# Two allocs / different descriptors → two nested scopes
# ---------------------------------------------------------------------------


def test_two_allocs_different_descriptors_two_groups():
    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch_a(self, a: pld.DistributedTensor[[256], pl.FP32]):
            return a

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch_b(self, b: pld.DistributedTensor[[256], pl.FP32]):
            return b

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            buf_a = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
            buf_b = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
            a = pld.window(buf_a, [256], dtype=pl.FP32)
            b = pld.window(buf_b, [256], dtype=pl.FP32)
            self.chip_orch_a(a, device=0)
            self.chip_orch_a(a, device=1)
            self.chip_orch_b(b, device=2)
            self.chip_orch_b(b, device=3)
            return 0

    result = _apply(P)
    host = _get_func(result, "host_orch")
    # buf_a is dispatched to {0,1}; buf_b to {2,3}. Distinct descriptors ⇒ two
    # nested scopes. The pass walks allocs in source order and opens a domain
    # on first descriptor mismatch, so scope order follows alloc order:
    # outermost = buf_a, innermost = buf_b.
    scopes = _get_comm_domain_scopes(host)
    assert len(scopes) == 2
    _assert_scope_fields(scopes[0], [0, 1], [_expected_slot("buf_a", _mul(256, 4))])
    _assert_scope_fields(scopes[1], [2, 3], [_expected_slot("buf_b", _mul(256, 4))])


def test_four_allocs_three_descriptors_three_groups():
    """Regression for the Phase-6 clustering fix: a linear scan of `pending`
    per alloc would still pass with two groups, but a third distinct
    descriptor exercises the third `pending` slot the map-based lookup must
    also resolve correctly (not just append-and-never-find). `buf_d` reuses
    `buf_c`'s descriptor ({4, 5}) so `desc_to_index.find` must also succeed on
    a *nonzero* index (index 2, not just index 0 as the two-group test above
    already covers) and route `buf_d` into `pending[2]` instead of appending a
    fourth, spurious domain."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch_a(self, a: pld.DistributedTensor[[256], pl.FP32]):
            return a

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch_b(self, b: pld.DistributedTensor[[256], pl.FP32]):
            return b

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch_c(
            self, c: pld.DistributedTensor[[256], pl.FP32], d: pld.DistributedTensor[[256], pl.FP32]
        ):
            return c

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            buf_a = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
            buf_b = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
            buf_c = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
            buf_d = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
            a = pld.window(buf_a, [256], dtype=pl.FP32)
            b = pld.window(buf_b, [256], dtype=pl.FP32)
            c = pld.window(buf_c, [256], dtype=pl.FP32)
            d = pld.window(buf_d, [256], dtype=pl.FP32)
            self.chip_orch_a(a, device=0)
            self.chip_orch_a(a, device=1)
            self.chip_orch_b(b, device=2)
            self.chip_orch_b(b, device=3)
            self.chip_orch_c(c, d, device=4)
            self.chip_orch_c(c, d, device=5)
            return 0

    result = _apply(P)
    host = _get_func(result, "host_orch")
    scopes = _get_comm_domain_scopes(host)
    assert len(scopes) == 3
    _assert_scope_fields(scopes[0], [0, 1], [_expected_slot("buf_a", _mul(256, 4))])
    _assert_scope_fields(scopes[1], [2, 3], [_expected_slot("buf_b", _mul(256, 4))])
    _assert_scope_fields(
        scopes[2],
        [4, 5],
        [_expected_slot("buf_c", _mul(256, 4)), _expected_slot("buf_d", _mul(256, 4))],
    )


# ---------------------------------------------------------------------------
# Single alloc / multiple views share the same WindowBuffer instance
# ---------------------------------------------------------------------------


def test_multi_view_per_alloc_shares_wb_instance():
    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            v1: pld.DistributedTensor[[256], pl.FP32],
            v2: pld.DistributedTensor[[256], pl.FP32],
        ):
            return v1

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            buf = pld.alloc_window_buffer(512 * pl.FP32.get_byte())
            view_a = pld.window(buf, [256], dtype=pl.FP32)
            view_b = pld.window(buf, [256], dtype=pl.FP32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(view_a, view_b, device=r)
            return 0

    result = _apply(P)
    host = _get_func(result, "host_orch")
    types = _view_var_types(host)
    assert len(types) == 2
    wb_a = types[0].window_buffer
    wb_b = types[1].window_buffer
    assert wb_a is not None and wb_b is not None
    # Both views materialise the same alloc → identical WindowBuffer shared_ptr.
    assert wb_a is wb_b
    # And that same WindowBuffer is the (single) slot of the (single) scope.
    scopes = _get_comm_domain_scopes(host)
    assert len(scopes) == 1
    assert len(scopes[0].slots) == 1
    assert scopes[0].slots[0] is wb_a


# ---------------------------------------------------------------------------
# Dispatch formals inherit the owner's exact window identity.
# ---------------------------------------------------------------------------


def test_c2a_chip_orch_formal_and_body_share_owner_window_identity():
    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(self, data: pld.DistributedTensor[[256], pl.FP32]):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            buf = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
            data = pld.window(buf, [256], dtype=pl.FP32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(data, device=r)
            return 0

    result = _apply(P)
    chip = _get_func(result, "chip_orch")
    # The formal and every use must reference the existing owner slot; creating
    # an equal-looking second WindowBuffer would lose allocation ownership.
    assert len(chip.params) == 1
    pt = chip.params[0].type
    assert isinstance(pt, ir.DistributedTensorType)
    slot = _get_comm_domain_scopes(_get_func(result, "host_orch"))[0].slots[0]
    assert pt.window_buffer is slot
    returns = [stmt for stmt in _flatten_stmts(chip.body) if isinstance(stmt, ir.ReturnStmt)]
    assert returns[0].value[0] is chip.params[0]
    assert returns[0].value[0].type.window_buffer is slot


# ---------------------------------------------------------------------------
# Dead alloc — alloc with no pld.tensor.window materialisation
# ---------------------------------------------------------------------------


def test_dead_alloc_no_window_materialisation_raises():
    """An alloc with no ``pld.tensor.window`` view is a dead allocation.

    Phase-3 sanity check (source ``CHECK(!allocs_with_windows[...].empty())``,
    doc "Sanity checks" bullet 1): downstream codegen would have nothing to
    point a comm-domain buffer slot at, so the pass rejects it. This is a
    distinct branch from ``test_dead_alloc_no_dispatch_raises`` (which has a
    view but no consuming dispatch).
    """

    @pl.program
    class P:
        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            buf = pld.alloc_window_buffer(256 * pl.FP32.get_byte())  # noqa: F841  # never windowed
            return 0

    with pytest.raises(pypto.InternalError, match=r"no pld\.tensor\.window materialisation"):
        _apply(P)


# ---------------------------------------------------------------------------
# Dead alloc — alloc + window but no dispatch consumer
# ---------------------------------------------------------------------------


def test_dead_alloc_no_dispatch_raises():
    @pl.program
    class P:
        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            buf = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
            _ = pld.window(buf, [256], dtype=pl.FP32)
            return 0

    with pytest.raises(pypto.InternalError, match="not consumed by any chip_orch dispatch"):
        _apply(P)


# ---------------------------------------------------------------------------
# Unsupported device= — induction var over a non-unit-step loop
# ---------------------------------------------------------------------------


def test_non_unit_step_device_loop_raises():
    """``device=r`` over ``pl.range(0, 4, 2)`` is rejected.

    The device resolver only supports unit-step ``pl.range`` induction vars
    (source ``CHECK(step == 1)``, doc device-descriptor table "other ⇒
    ValueError"). A stride-2 loop has no well-defined contiguous coverage, so
    the pass raises rather than guessing.
    """

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(self, data: pld.DistributedTensor[[256], pl.FP32]):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            buf = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
            data = pld.window(buf, [256], dtype=pl.FP32)
            for r in pl.range(0, 4, 2):
                self.chip_orch(data, device=r)
            return 0

    with pytest.raises(pypto.InternalError, match="non-unit-step loop is not supported"):
        _apply(P)


# ---------------------------------------------------------------------------
# Idempotence — running the pass twice produces an equivalent program
# ---------------------------------------------------------------------------


def test_idempotent():
    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(self, data: pld.DistributedTensor[[256], pl.FP32]):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            buf = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
            data = pld.window(buf, [256], dtype=pl.FP32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(data, device=r)
            return 0

    first = _apply(P)
    second = passes.materialize_comm_domain_scopes()(first)
    ir.assert_structural_equal(first, second)
    first_host = _get_func(first, "host_orch")
    second_host = _get_func(second, "host_orch")
    first_scopes = _get_comm_domain_scopes(first_host)
    second_scopes = _get_comm_domain_scopes(second_host)
    assert len(first_scopes) == 1
    assert len(second_scopes) == 1
    # Same device coverage; same number of slots.
    assert list(first_scopes[0].devices) == list(second_scopes[0].devices)
    assert len(first_scopes[0].slots) == len(second_scopes[0].slots)


# ---------------------------------------------------------------------------
# Programs without any pld.tensor.alloc_window_buffer pass straight through
# ---------------------------------------------------------------------------


def test_no_alloc_window_buffer_no_op():
    @pl.program
    class Before:
        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self, x: pl.Tensor[[64], pl.FP32]):
            return x

    After = _apply(Before)
    # No alloc_window_buffer chains ⇒ the pass emits no scope wrapping and
    # rewrites nothing, so the program is unchanged.
    ir.assert_structural_equal(After, Before)


# ---------------------------------------------------------------------------
# Loop-bound through SSA temp (regression for the second N4 bug)
# ---------------------------------------------------------------------------


def test_loop_bound_via_assigned_temp_world_size():
    """``n = pld.world_size(); for r in pl.range(n): ... device=r`` ⇒ kAll.

    Mirrors the post-ConvertToSSA / NormalizeStmtStructure shape where
    ``pl.range(pld.world_size())`` has been CSE-hoisted into a named temp.
    The pass must follow the AssignStmt def chain back to the
    ``pld.system.world_size`` Call, otherwise the dispatch is rejected.
    """

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(self, data: pld.DistributedTensor[[256], pl.FP32]):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            buf = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
            data = pld.window(buf, [256], dtype=pl.FP32)
            n = pld.world_size()
            for r in pl.range(n):
                self.chip_orch(data, device=r)
            return 0

    result = _apply(P)
    host = _get_func(result, "host_orch")
    scopes = _get_comm_domain_scopes(host)
    assert len(scopes) == 1
    assert list(scopes[0].devices) == [], "world_size loop bound must resolve to kAll"


def test_loop_bound_via_assigned_temp_const_int():
    """``n = 2; for r in pl.range(n): ... device=r`` ⇒ subset {0, 1}.

    Same indirection as the world_size case but with a ConstInt at the
    end of the def chain — exercises the integer-bound branch of
    ``UnwrapStopExpr``.
    """

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(self, data: pld.DistributedTensor[[256], pl.FP32]):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            buf = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
            data = pld.window(buf, [256], dtype=pl.FP32)
            n = 2
            for r in pl.range(n):
                self.chip_orch(data, device=r)
            return 0

    result = _apply(P)
    host = _get_func(result, "host_orch")
    scopes = _get_comm_domain_scopes(host)
    assert len(scopes) == 1
    assert list(scopes[0].devices) == [0, 1]


# ---------------------------------------------------------------------------
# CommDomainScopesMaterialized property verifier
# ---------------------------------------------------------------------------


def _build_pass_output_with_two_groups() -> ir.Program:
    """Run the pass on a 2-domain program; return its (uniqueness-correct) output."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(self, data: pld.DistributedTensor[[256], pl.FP32]):
            return data

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch_other(self, data: pld.DistributedTensor[[256], pl.FP32]):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            buf_a = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
            buf_b = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
            data_a = pld.window(buf_a, [256], dtype=pl.FP32)
            data_b = pld.window(buf_b, [256], dtype=pl.FP32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(data_a, device=r)
            self.chip_orch_other(data_b, device=0)
            return 0

    return _apply(P)


def test_verifier_passes_on_pass_output():
    """Verifier emits no errors on the pass's own output (slot-uniqueness holds)."""
    program = _build_pass_output_with_two_groups()
    host = _get_func(program, "host_orch")
    assert len(_get_comm_domain_scopes(host)) == 2

    props = passes.IRPropertySet()
    props.insert(passes.IRProperty.CommDomainScopesMaterialized)
    diagnostics = passes.PropertyVerifierRegistry.verify(props, program)
    errors = [d for d in diagnostics if d.severity == passes.DiagnosticSeverity.Error]
    assert errors == []


def test_verifier_flags_duplicate_slot_across_scopes():
    """A WindowBuffer reused across two CommDomainScopeStmts is rejected with a clear error."""
    program = _build_pass_output_with_two_groups()
    host = _get_func(program, "host_orch")
    outer, inner = _get_comm_domain_scopes(host)

    # Build a sibling scope that reuses the outer scope's slot while the outer
    # still owns it. The body is the original (substituted) body lifted out
    # from under ``inner`` — it's only present to satisfy the ScopeStmt ctor;
    # the verifier only inspects ``slots_``.
    duplicated = ir.CommDomainScopeStmt(
        list(outer.devices),
        [outer.slots[0]],
        "comm_d_dup",
        body=inner.body,
        span=outer.span,
    )

    # Splice the duplicate into the function body in place of the original
    # scope chain so the IR carries both ``outer`` (with the slot) and
    # ``duplicated`` (reusing it).
    bad_body = ir.SeqStmts([outer, duplicated], outer.span)
    bad_host_func = ir.Function(
        host.name,
        list(host.params),
        list(host.return_types),
        bad_body,
        host.span,
        type=host.func_type,
        level=host.level,
        role=host.role,
    )
    new_functions = [bad_host_func if f.name == host.name else f for f in program.functions.values()]
    bad_program = ir.Program(new_functions, program.name, program.span)

    props = passes.IRPropertySet()
    props.insert(passes.IRProperty.CommDomainScopesMaterialized)
    diagnostics = passes.PropertyVerifierRegistry.verify(props, bad_program)
    errors = [d for d in diagnostics if d.severity == passes.DiagnosticSeverity.Error]
    assert len(errors) == 1
    assert "appears in multiple CommDomainScopeStmts" in errors[0].message
    assert errors[0].rule_name == "CommDomainScopesMaterialized"


@pytest.mark.parametrize("explicit_layout", [False, True])
def test_materialized_program_text_roundtrip(explicit_layout):
    """Print and parse a complete program without re-running any lowering pass."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip(self, x: pld.DistributedTensor[[4], pl.FP32]):
            return x

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_other(self, x: pld.DistributedTensor[[4], pl.FP32]):
            return x

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host(self, n: pl.Scalar[pl.INDEX]):
            size = n * 4
            buf = pld.alloc_window_buffer(size)
            other_buf = pld.alloc_window_buffer(16)
            x = pld.window(buf, [4], dtype=pl.FP32)
            alias = pld.window(buf, [4], dtype=pl.FP32)
            other = pld.window(other_buf, [4], dtype=pl.FP32)
            for r in pl.range(2):
                self.chip(x, device=r)
                self.chip(alias, device=r)
            self.chip_other(other, device=2)
            return 0

    ir.assert_structural_equal(P, pl.parse(ir.python_print(P, format=False)))
    original = _apply(P)
    printed = ir.python_print(original, format=False, explicit_layout=explicit_layout)
    restored = pl.parse_program(printed)
    ir.assert_structural_equal(original, restored)
    assert ir.structural_hash(original) == ir.structural_hash(restored)
    scopes = _get_comm_domain_scopes(_get_func(restored, "host"))
    views = _view_var_types(_get_func(restored, "host"))
    assert [list(scope.devices) for scope in scopes] == [[0, 1], [2]]
    assert views[0].window_buffer is views[1].window_buffer is scopes[0].slots[0]
    assert views[2].window_buffer is scopes[1].slots[0]
    assert views[0].window_buffer is not views[2].window_buffer


def test_explicit_comm_domain_metadata_roundtrip():
    """Restore explicitly recorded metadata, including values not inferred by the pass."""
    original = pl.parse_program("""
import pypto.language as pl
import pypto.language.distributed as pld

@pl.program
class P:
    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def host(self):
        with pl._dump.comm_domain(
            name="saved_domain", devices=[],
            slots=[pl._dump.window_buffer(7, "buf", 16, load_from_host=True, store_to_host=True)],
            attrs={"saved_metadata": True},
        ):
            buf: pl.Ptr = pld.alloc_window_buffer(16)
            x: pld.DistributedTensor[
                [4], pl.FP32, pl._dump.window_ref(7)
            ] = pld.window(buf, [4], dtype=pl.FP32)
            return x
""")
    restored = pl.parse_program(ir.python_print(original, format=False))
    ir.assert_structural_equal(original, restored)
    assert ir.structural_hash(original) == ir.structural_hash(restored)
    scope = _get_comm_domain_scopes(_get_func(restored, "host"))[0]
    assert scope.name_hint == "saved_domain"
    assert list(scope.devices) == []
    assert scope.attrs == {"saved_metadata": True}
    assert scope.slots[0].load_from_host is True
    assert scope.slots[0].store_to_host is True
    assert _view_var_types(_get_func(restored, "host"))[0].window_buffer is scope.slots[0]


def test_signature_window_reference_resolves_body_allocation():
    """The complete function supplies definitions referenced by its signature."""
    original = pl.parse_program("""
@pl.program
class P:
    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def host(self, x: pld.DistributedTensor[
        [4], pl.FP32, pl.TensorView(stride=[2]), pl._dump.window_ref("input_window")
    ]):
        with pl._dump.comm_domain(name="saved", devices=[0], slots=[
            pl._dump.window_buffer("input_window", base="buf", size=32)
        ]):
            buf: pl.Ptr = pld.alloc_window_buffer(32)
            return x
""")
    restored = pl.parse_program(ir.python_print(original, format=False))
    ir.assert_structural_equal(original, restored)
    host = _get_func(restored, "host")
    scope = _get_comm_domain_scopes(host)[0]
    param_type = host.params[0].type
    assert isinstance(param_type, ir.DistributedTensorType)
    assert param_type.window_buffer is scope.slots[0]
    assert param_type.tensor_view is not None
    stride = param_type.tensor_view.stride[0]
    assert isinstance(stride, ir.ConstInt)
    assert stride.value == 2
    assert isinstance(scope.body, ir.SeqStmts)
    assert isinstance(scope.body.stmts[0], ir.AssignStmt)
    assert scope.slots[0].base is scope.body.stmts[0].var


def test_unused_window_slot_and_body_local_scope_attr_roundtrip():
    """Slots define window identities independently of whether a tensor uses them."""
    original = pl.parse_program("""
@pl.program
class P:
    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def host(self):
        with pl._dump.comm_domain(
            name="saved", devices=[0], slots=[pl._dump.window_buffer(0, "buf", size)],
            attrs={"saved_size": size},
        ):
            size: pl.Scalar[pl.INDEX] = 16
            buf: pl.Ptr = pld.alloc_window_buffer(size)
            return 0
""")
    restored = pl.parse_program(ir.python_print(original, format=False))
    ir.assert_structural_equal(original, restored)
    assert ir.structural_hash(original) == ir.structural_hash(restored)
    scope = _get_comm_domain_scopes(_get_func(restored, "host"))[0]
    assert scope.attrs["saved_size"] is scope.slots[0].size


def test_undefined_window_reference_is_rejected():
    """An absent definition cannot be repaired by silently rerunning a pass."""
    with pytest.raises(ParserTypeError, match="Undefined window_buffer ID 9"):
        pl.parse_program("""
@pl.program
class P:
    @pl.function
    def f(self, x: pld.DistributedTensor[[4], pl.FP32, pl._dump.window_ref(9)]):
        return x
""")


def test_named_windows_preserve_distinct_identity_with_shared_base():
    """Window labels identify objects, not their base allocation or slot index."""
    original = pl.parse_program("""
@pl.program
class P:
    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def host(self):
        with pl._dump.comm_domain(name="saved", devices=[0], slots=[
            pl._dump.window_buffer("first", base="buf", size=16),
            pl._dump.window_buffer("second", base="buf", size=16),
        ]):
            buf: pl.Ptr = pl._dump.alloc_window_buffer(16, name="buf")
            x = pl._dump.call("pld.tensor.window", [buf, [4]],
                pld.DistributedTensor[[4], pl.FP32, pl._dump.window_ref("first")],
                kwargs={"dtype": pl.FP32})
            y = pl._dump.call("pld.tensor.window", [buf, [4]],
                pld.DistributedTensor[[4], pl.FP32, pl._dump.window_ref("second")],
                kwargs={"dtype": pl.FP32})
            return x, y
""")
    printed = ir.python_print(original, format=False)
    assert 'window_buffer("buf_window", base="buf", size=16' in printed
    assert 'window_buffer("buf_window_1", base="buf", size=16' in printed
    restored = pl.parse_program(printed)
    ir.assert_structural_equal(original, restored)
    assert ir.structural_hash(original) == ir.structural_hash(restored)
    assert ir.python_print(restored, format=False) == printed
    host = _get_func(restored, "host")
    first, second = _get_comm_domain_scopes(host)[0].slots
    assert first is not second
    assert first.base is second.base
    x_type, y_type = _view_var_types(host)
    assert x_type.window_buffer is first
    assert y_type.window_buffer is second


@pytest.mark.parametrize(
    ("slots", "reference", "message"),
    [
        ('pl._dump.window_buffer("known", base="buf", size=16)', '"missing"', "Undefined window_buffer"),
        (
            'pl._dump.window_buffer("same", base="buf", size=16), '
            'pl._dump.window_buffer("same", base="buf", size=32)',
            '"same"',
            "Conflicting window_buffer descriptors",
        ),
        ('pl._dump.window_buffer("", base="buf", size=16)', '""', "non-empty string"),
        ('pl._dump.window_buffer(True, base="buf", size=16)', "True", "non-empty string"),
        ('pl._dump.window_buffer("w", "buf", base="buf", size=16)', '"w"', "Duplicate window_buffer"),
    ],
)
def test_named_window_reference_diagnostics(slots, reference, message):
    """Missing or ambiguous identity must fail instead of binding a different window."""
    with pytest.raises(ParserTypeError, match=message):
        pl.parse_program(f"""
@pl.program
class P:
    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def host(self):
        with pl._dump.comm_domain(name="saved", devices=[0], slots=[{slots}]):
            buf: pl.Ptr = pl._dump.alloc_window_buffer(16, name="buf")
            x = pl._dump.call("pld.tensor.window", [buf, [4]],
                pld.DistributedTensor[[4], pl.FP32, pl._dump.window_ref({reference})],
                kwargs={{"dtype": pl.FP32}})
            return x
""")


@pytest.mark.parametrize("trailing_return", [False, True])
def test_empty_comm_domain_scope_roundtrip(trailing_return):
    """An empty scope needs a pass statement, including before a following sibling."""
    suffix = "\n        return 0" if trailing_return else ""
    original = pl.parse_program(
        """
@pl.program
class P:
    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def host(self):
        with pl._dump.comm_domain(name="empty", devices=[], slots=[]):
            pass
"""
        + suffix
    )
    restored = pl.parse_program(ir.python_print(original, format=False))
    ir.assert_structural_equal(original, restored)
    assert ir.structural_hash(original) == ir.structural_hash(restored)
    scope = _get_comm_domain_scopes(_get_func(restored, "host"))[0]
    assert isinstance(scope.body, ir.SeqStmts)
    assert not scope.body.stmts


@pytest.mark.parametrize("value", [-1, -2.5, -0.0])
def test_signed_scope_attribute_roundtrip(value):
    """A negative scalar attribute remains distinct from a typed IR constant."""
    span = ir.Span.unknown()
    builder = IRBuilder()
    index_expr = ir.ConstInt(-1, DataType.INDEX, span)
    with builder.function("host", span, level=ir.Level.HOST, role=ir.Role.Orchestrator) as fn:
        with builder.comm_domain_scope(
            [], [], "saved", span, attrs=[("saved", value), ("index_expr", index_expr)]
        ):
            builder.return_stmt(0, span)
    original = ir.Program([fn.get_result()], "P", span)
    restored = pl.parse_program(ir.python_print(original, format=False))
    ir.assert_structural_equal(original, restored)
    assert ir.structural_hash(original) == ir.structural_hash(restored)
    scope = _get_comm_domain_scopes(_get_func(restored, "host"))[0]
    assert type(scope.attrs["saved"]) is type(value)
    assert scope.attrs["saved"] == value
    assert isinstance(scope.attrs["index_expr"], ir.ConstInt)
    assert scope.attrs["index_expr"].value == -1


def _c1b_dynamic_dispatch_program(query, *, hoisted=False, start="0", step="1"):
    bound = "n_alias" if hoisted else f"pld.{query}()"
    hoist = f"        n = pld.{query}()\n        n_alias = n\n" if hoisted else ""
    return pl.parse_program(
        "import pypto.language as pl\n"
        "import pypto.language.distributed as pld\n"
        "@pl.program\nclass P:\n"
        "    @pl.function(type=pl.FunctionType.Orchestration)\n"
        "    def chip(self, data: pld.DistributedTensor[[256], pl.FP32]):\n"
        "        return data\n"
        "    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)\n"
        "    def host(self, begin: pl.Scalar[pl.INT64], stride: pl.Scalar[pl.INT64]):\n"
        "        buf = pld.alloc_window_buffer(1024)\n"
        "        data = pld.window(buf, [256], dtype=pl.FP32)\n"
        f"{hoist}"
        f"        for r in pl.range({start}, {bound}, {step}):\n"
        "            self.chip(data, worker=r)\n"
        "        return 0\n"
    )


@pytest.mark.parametrize("hoisted", [False, True])
def test_c1b_num_workers_full_coverage_loop(hoisted):
    # ConvertToSSA runs inside _apply; both direct and aliased bounds must resolve.
    program = _apply(_c1b_dynamic_dispatch_program("num_workers", hoisted=hoisted))
    scopes = _get_comm_domain_scopes(_get_func(program, "host"))
    assert len(scopes) == 1
    assert list(scopes[0].devices) == []
    assert len(scopes[0].slots) == 1
    assert _view_var_types(_get_func(program, "host"))[0].window_buffer is scopes[0].slots[0]


@pytest.mark.parametrize("query", ["world_size", "num_workers"])
@pytest.mark.parametrize("hoisted", [False, True])
@pytest.mark.parametrize(("start", "step"), [("1", "1"), ("0", "2"), ("begin", "1"), ("0", "stride")])
def test_c1b_dynamic_full_coverage_requires_zero_start_unit_step(query, hoisted, start, step):
    # A dynamic stop alone proves no coverage: skipping rank zero or striding
    # cannot allocate an all-workers domain. Dynamic starts/steps are unproven too.
    before = _c1b_dynamic_dispatch_program(query, hoisted=hoisted, start=start, step=step)
    with pytest.raises(ValueError, match="start|step|full.coverage"):
        _apply(before)


def _c2a_stmts(stmt):
    yield stmt
    if isinstance(stmt, ir.SeqStmts):
        for child in stmt.stmts:
            yield from _c2a_stmts(child)
    elif isinstance(stmt, (ir.ScopeStmt, ir.ForStmt, ir.WhileStmt)):
        yield from _c2a_stmts(stmt.body)
    elif isinstance(stmt, ir.IfStmt):
        yield from _c2a_stmts(stmt.then_body)
        if stmt.else_body is not None:
            yield from _c2a_stmts(stmt.else_body)


def _c2a_exprs(expr):
    yield expr
    if isinstance(expr, (ir.Call, ir.Submit)):
        for arg in expr.args:
            yield from _c2a_exprs(arg)
        if isinstance(expr, ir.Submit):
            for dep in expr.deps:
                yield from _c2a_exprs(dep)
    elif isinstance(expr, ir.MakeTuple):
        for element in expr.elements:
            yield from _c2a_exprs(element)
    elif isinstance(expr, ir.TupleGetItemExpr):
        yield from _c2a_exprs(expr.tuple)


def _c2a_dist_types(ty):
    if isinstance(ty, ir.DistributedTensorType):
        yield ty
    elif isinstance(ty, ir.TupleType):
        for element in ty.types:
            yield from _c2a_dist_types(element)


def _c2a_assert_function_identity(func, slot):
    """Check signatures, defining expressions, projections, and SSA carries."""
    types = [param.type for param in func.params] + list(func.return_types)
    for stmt in _c2a_stmts(func.body):
        exprs = []
        if isinstance(stmt, ir.AssignStmt):
            exprs = [stmt.var, stmt.value]
        elif isinstance(stmt, ir.EvalStmt):
            exprs = [stmt.expr]
        elif isinstance(stmt, (ir.ReturnStmt, ir.YieldStmt)):
            exprs = list(stmt.value)
        elif isinstance(stmt, (ir.ForStmt, ir.WhileStmt)):
            exprs = list(stmt.iter_args) + list(stmt.return_vars)
            exprs += [arg.initValue for arg in stmt.iter_args]
        elif isinstance(stmt, ir.IfStmt):
            exprs = list(stmt.return_vars)
        for expr in exprs:
            types.extend(value.type for value in _c2a_exprs(expr))
    distributed = [dt for ty in types for dt in _c2a_dist_types(ty)]
    assert distributed, f"{func.name} must contain distributed values"
    assert all(dt.window_buffer is slot for dt in distributed)


def _c2a_three_level_program(names=("z_root", "a_host", "m_chip"), *, hoisted=False, start="0", step="1"):
    root, host, chip = names
    bound = "count_alias" if hoisted else "pld.num_workers()"
    hoist = "        count = pld.num_workers()\n        count_alias = count\n" if hoisted else ""
    return pl.parse_program(
        f"""
@pl.program
class ThreeLevels:
    @pl.function(level=pl.Level.CHIP, role=pl.Role.Orchestrator)
    def {chip}(self, data: pld.DistributedTensor[[4], pl.FP32]) -> pld.DistributedTensor[[4], pl.FP32]:
        return data

    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def {host}(self, data: pld.DistributedTensor[[4], pl.FP32]) -> pld.DistributedTensor[[4], pl.FP32]:
        forwarded = self.{chip}(data, worker=0)
        return forwarded

    @pl.function(level=pl.Level.CLUSTER_0, role=pl.Role.Orchestrator)
    def {root}(self, begin: pl.Scalar[pl.INDEX], stride: pl.Scalar[pl.INDEX]):
        buf = pld.alloc_window_buffer(16)
        data = pld.window(buf, [4], dtype=pl.FP32)
{hoist}        for worker in pl.range({start}, {bound}, {step}):
            self.{host}(data, worker=worker)
        return 0
"""
    )


@pytest.mark.parametrize("names", [("z_root", "a_host", "m_chip"), ("a_root", "z_host", "b_chip")])
@pytest.mark.parametrize("hoisted", [False, True])
def test_c2a_three_levels_share_identity_independent_of_function_order(names, hoisted):
    before = _c2a_three_level_program(names, hoisted=hoisted)
    after = _apply(before)
    root, host, chip = [_get_func(after, name) for name in names]
    scopes = _get_comm_domain_scopes(root)
    assert len(scopes) == 1
    assert list(scopes[0].devices) == []
    assert len(scopes[0].slots) == 1
    slot = scopes[0].slots[0]
    for func in (root, host, chip):
        _c2a_assert_function_identity(func, slot)
    assert _get_comm_domain_scopes(host) == []
    assert _get_comm_domain_scopes(chip) == []
    props = passes.IRPropertySet()
    props.insert(passes.IRProperty.AssignTypeSymmetry)
    props.insert(passes.IRProperty.TypeChecked)
    assert passes.PropertyVerifierRegistry.verify(props, after) == []


@pytest.mark.parametrize("roundtrip", ["idempotence", "text", "serde"])
def test_c2a_cross_function_identity_survives_roundtrip(roundtrip):
    before = _apply(_c2a_three_level_program())
    if roundtrip == "idempotence":
        after = passes.materialize_comm_domain_scopes()(before)
    elif roundtrip == "text":
        after = pl.parse_program(ir.python_print(before, format=False))
    else:
        after = ir.deserialize(ir.serialize(before))
    ir.assert_structural_equal(before, after)
    slot = _get_comm_domain_scopes(_get_func(after, "z_root"))[0].slots[0]
    for name in ("z_root", "a_host", "m_chip"):
        _c2a_assert_function_identity(_get_func(after, name), slot)


def test_c2a_local_and_inherited_windows_have_separate_owners():
    before = pl.parse_program("""
@pl.program
class LocalAndInherited:
    @pl.function(level=pl.Level.CHIP, role=pl.Role.Orchestrator)
    def chip(
        self, inherited: pld.DistributedTensor[[4], pl.FP32], local: pld.DistributedTensor[[4], pl.FP32]
    ):
        return inherited, local
    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def host(self, inherited: pld.DistributedTensor[[4], pl.FP32]):
        local_buf = pld.alloc_window_buffer(16)
        local = pld.window(local_buf, [4], dtype=pl.FP32)
        self.chip(inherited, local, worker=0)
        return inherited
    @pl.function(level=pl.Level.CLUSTER_0, role=pl.Role.Orchestrator)
    def root(self):
        buf = pld.alloc_window_buffer(16)
        inherited = pld.window(buf, [4], dtype=pl.FP32)
        self.host(inherited, worker=0)
        return 0
""")
    after = _apply(before)
    root, host, chip = [_get_func(after, name) for name in ("root", "host", "chip")]
    root_scope, host_scope = _get_comm_domain_scopes(root)[0], _get_comm_domain_scopes(host)[0]
    assert len(root_scope.slots) == len(host_scope.slots) == 1
    global_slot, local_slot = root_scope.slots[0], host_scope.slots[0]
    assert global_slot is not local_slot
    assert host.params[0].type.window_buffer is chip.params[0].type.window_buffer is global_slot
    assert _view_var_types(host)[0].window_buffer is chip.params[1].type.window_buffer is local_slot
    assert _get_comm_domain_scopes(chip) == []


@pytest.mark.parametrize("lineage", ["alias_view", "writeback", "for", "while", "branch"])
def test_c2a_inherited_identity_through_local_lineage(lineage):
    bodies = {
        "alias_view": """
        alias = data
        sliced = pl.slice(alias, [4], [0])
        packed = (alias, sliced)
        result = packed[1]
""",
        "writeback": """
        sliced = pl.slice(data, [4], [0])
        result = pl.assemble(data, sliced, [0])
""",
        "for": """
        for _, (carried,) in pl.range(2, init_values=(data,)):
            result = pl.yield_(carried)
""",
        "while": """
        for (carried,) in pl.while_(init_values=(data,)):
            pl.cond(flag)
            result = pl.yield_(carried)
""",
        "branch": """
        if flag:
            result = pl.yield_(data)
        else:
            result = pl.yield_(data)
""",
    }
    before = pl.parse_program(
        """
@pl.program
class Lineage:
    @pl.function(level=pl.Level.CHIP, role=pl.Role.Orchestrator)
    def chip(self, data: pld.DistributedTensor[[4], pl.FP32]):
        return data
    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def host(self, data: pld.DistributedTensor[[4], pl.FP32], flag: pl.Scalar[pl.BOOL]):
"""
        + bodies[lineage]
        + """
        returned = self.chip(result, worker=0)
        return returned
    @pl.function(level=pl.Level.CLUSTER_0, role=pl.Role.Orchestrator)
    def root(self, flag: pl.Scalar[pl.BOOL]):
        buf = pld.alloc_window_buffer(16)
        data = pld.window(buf, [4], dtype=pl.FP32)
        self.host(data, flag, worker=0)
        return 0
"""
    )
    after = _apply(before)
    slot = _get_comm_domain_scopes(_get_func(after, "root"))[0].slots[0]
    for name in ("host", "chip"):
        _c2a_assert_function_identity(_get_func(after, name), slot)


@pytest.mark.parametrize("conflict", [False, True])
def test_c2a_multiple_callers_require_one_object_identity(conflict):
    # Equal shapes do not make separately owned windows interchangeable.
    extra = "        other_buf = pld.alloc_window_buffer(16)\n"
    extra += "        other = pld.window(other_buf, [4], dtype=pl.FP32)\n"
    before = pl.parse_program(
        """
@pl.program
class MultipleCallers:
    @pl.function(level=pl.Level.CHIP, role=pl.Role.Orchestrator)
    def chip(self, data: pld.DistributedTensor[[4], pl.FP32]):
        return data
    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def host(self):
        buf = pld.alloc_window_buffer(16)
        data = pld.window(buf, [4], dtype=pl.FP32)
"""
        + (extra if conflict else "        other = data\n")
        + """
        self.chip(data, worker=0)
        self.chip(other, worker=0)
        return 0
"""
    )
    if conflict:
        with pytest.raises(ValueError, match="[Cc]onflict.*[Ww]indow|[Ww]indow.*[Cc]onflict"):
            _apply(before)
    else:
        after = _apply(before)
        slot = _get_comm_domain_scopes(_get_func(after, "host"))[0].slots[0]
        _c2a_assert_function_identity(_get_func(after, "chip"), slot)


def test_c2a_distinct_existing_windows_with_same_name_and_shape_conflict():
    span = ir.Span.unknown()
    shape = [ir.ConstInt(4, DataType.INDEX, span)]
    size = ir.ConstInt(16, DataType.INDEX, span)
    unbound_type = ir.DistributedTensorType(shape, pl.FP32)
    formal = ir.Var("data", unbound_type, span)
    chip = ir.Function(
        "chip",
        [formal],
        [unbound_type],
        ir.ReturnStmt([formal], span),
        span,
        level=ir.Level.CHIP,
        role=ir.Role.Orchestrator,
    )
    owners = []
    for name in ("first_host", "second_host"):
        base = ir.Var("buf", ir.PtrType(), span)
        slot = ir.WindowBuffer(base, size)
        ty = ir.DistributedTensorType(shape, pl.FP32, slot)
        actual = ir.Var("data", ty, span)
        call = ir.Call(ir.GlobalVar("chip"), [actual], {}, {}, unbound_type, span)
        body = ir.CommDomainScopeStmt(
            [0],
            [slot],
            "saved",
            body=ir.EvalStmt(call, span),
            span=span,
        )
        owners.append(
            ir.Function(
                name,
                [actual],
                [],
                body,
                span,
                level=ir.Level.HOST,
                role=ir.Role.Orchestrator,
            )
        )
    # Pre-existing owner metadata is sufficient to identify both parameters;
    # body allocations are irrelevant to the distinct-object conflict.
    before = ir.Program([chip, *owners], "SameNameConflict", span)
    with passes.PassContext([], verification_level=passes.VerificationLevel.NONE):
        with pytest.raises(ValueError, match="[Cc]onflict.*[Ww]indow|[Ww]indow.*[Cc]onflict"):
            passes.materialize_comm_domain_scopes()(before)


@pytest.mark.parametrize("reverse_names", [False, True])
def test_c2a_shared_formal_object_rewrites_every_function_owner(reverse_names):
    """Native functions may share a formal/body DAG; neither owner may be skipped."""
    span = ir.Span.unknown()
    shape = [ir.ConstInt(4, DataType.INDEX, span)]
    ty = ir.DistributedTensorType(shape, pl.FP32)
    shared_formal = ir.Var("shared_formal", ty, span)
    shared_body = ir.SeqStmts([ir.EvalStmt(shared_formal, span), ir.ReturnStmt([], span)], span)
    names = ["z_first", "a_second"] if reverse_names else ["a_first", "z_second"]
    children = [
        ir.Function(
            name, [shared_formal], [], shared_body, span, level=ir.Level.CHIP, role=ir.Role.Orchestrator
        )
        for name in names
    ]
    assert children[0].params[0] is children[1].params[0]
    assert children[0].body is children[1].body
    buffer = ir.Var("buf", ir.PtrType(), span)
    data = ir.Var("data", ty, span)
    alloc = ir.create_op_call(
        "pld.tensor.alloc_window_buffer", [ir.ConstInt(16, DataType.INDEX, span)], {"name": "buf"}, span
    )
    window = ir.create_op_call(
        "pld.tensor.window", [buffer, ir.MakeTuple(shape, span)], {"dtype": pl.FP32}, span
    )
    calls = [
        ir.EvalStmt(
            ir.Call(
                ir.GlobalVar(name),
                [data],
                {},
                {"worker": ir.ConstInt(0, DataType.INDEX, span)},
                ir.TupleType([]),
                span,
            ),
            span,
        )
        for name in names
    ]
    host = ir.Function(
        "host",
        [],
        [],
        ir.SeqStmts(
            [
                ir.AssignStmt(buffer, alloc, span),
                ir.AssignStmt(data, window, span),
                *calls,
                ir.ReturnStmt([], span),
            ],
            span,
        ),
        span,
        level=ir.Level.HOST,
        role=ir.Role.Orchestrator,
    )
    before = ir.Program([*children, host], "SharedFunctionOwners", span)
    # This legal native input already satisfies SSA. Preprocessing could clone
    # the shared formal and hide the multiple function owners being tested.
    with passes.PassContext([passes.VerificationInstrument(passes.VerificationMode.BEFORE_AND_AFTER)]):
        after = passes.materialize_comm_domain_scopes()(before)
    slot = _get_comm_domain_scopes(_get_func(after, "host"))[0].slots[0]
    for name in names:
        child = _get_func(after, name)
        assert child.params[0].type.window_buffer is slot
        assert child.body.stmts[0].expr is child.params[0]
        assert child.body.stmts[0].expr.type.window_buffer is slot
        assert child.return_types == []


@pytest.mark.parametrize("reverse_names", [False, True])
def test_c2a_shared_scalar_attr_root_rewrites_nested_owner_call_in_every_function(reverse_names):
    """A scalar root can contain a changed DT call in a shared native attr DAG."""
    span = ir.Span.unknown()
    shape = [ir.ConstInt(4, DataType.INDEX, span)]
    ty = ir.DistributedTensorType(shape, pl.FP32)
    formal = ir.Var("formal", ty, span)
    chip = ir.Function(
        "chip",
        [formal],
        [],
        ir.ReturnStmt([], span),
        span,
        level=ir.Level.CHIP,
        role=ir.Role.Orchestrator,
    )
    buffer = ir.Var("buf", ir.PtrType(), span)
    data = ir.Var("data", ty, span)
    alloc = ir.create_op_call(
        "pld.tensor.alloc_window_buffer", [ir.ConstInt(16, DataType.INDEX, span)], {"name": "buf"}, span
    )
    window = ir.create_op_call(
        "pld.tensor.window", [buffer, ir.MakeTuple(shape, span)], {"dtype": pl.FP32}, span
    )
    dispatch = ir.Call(
        ir.GlobalVar("chip"),
        [data],
        {},
        {"worker": ir.ConstInt(0, DataType.INDEX, span)},
        ir.TupleType([]),
        span,
    )
    owner = ir.Function(
        "owner",
        [],
        [ty],
        ir.SeqStmts(
            [
                ir.AssignStmt(buffer, alloc, span),
                ir.AssignStmt(data, window, span),
                ir.EvalStmt(dispatch, span),
                ir.ReturnStmt([data], span),
            ],
            span,
        ),
        span,
        level=ir.Level.HOST,
        role=ir.Role.Orchestrator,
    )
    owner_call = ir.Call(ir.GlobalVar("owner"), [], {}, {}, ty, span)
    context_query = ir.create_op_call("pld.system.get_comm_ctx", [owner_call], {}, span)
    scalar_root = ir.create_op_call("pld.system.rank", [context_query], {}, span)
    names = ["z_first", "a_second"] if reverse_names else ["a_first", "z_second"]
    users = [
        ir.Function(
            name,
            [],
            [],
            ir.ReturnStmt([], span),
            span,
            level=ir.Level.HOST,
            role=ir.Role.Orchestrator,
            attrs={"nested_owner": scalar_root, "ordinary": "kept", "count": 7},
        )
        for name in names
    ]
    assert users[0].attrs["nested_owner"] is users[1].attrs["nested_owner"]
    before = ir.Program([*users, owner, chip], "SharedScalarAttribute", span)
    # No preprocessing: preserve the shared expression object and verify this
    # parameter-free, closed attribute expression as legal native IR directly.
    with passes.PassContext([passes.VerificationInstrument(passes.VerificationMode.BEFORE_AND_AFTER)]):
        after = passes.materialize_comm_domain_scopes()(before)
    rewritten_owner = _get_func(after, "owner")
    slot = _get_comm_domain_scopes(rewritten_owner)[0].slots[0]
    assert rewritten_owner.return_types[0].window_buffer is slot
    assert _get_func(after, "chip").params[0].type.window_buffer is slot
    for name in names:
        user = _get_func(after, name)
        assert user.params == [] and user.return_types == []
        assert isinstance(user.body, ir.ReturnStmt) and user.body.value == []
        assert user.attrs["ordinary"] == "kept" and user.attrs["count"] == 7
        root = user.attrs["nested_owner"]
        assert isinstance(root, ir.Call) and root.op.name == ir.get_op("pld.system.rank").name
        assert root.type.dtype == DataType.INT32
        query = root.args[0]
        assert isinstance(query, ir.Call) and query.op.name == ir.get_op("pld.system.get_comm_ctx").name
        assert isinstance(query.type, ir.CommCtxType)
        nested = query.args[0]
        assert isinstance(nested, ir.Call) and isinstance(nested.op, ir.GlobalVar)
        assert nested.op.name == "owner"
        assert nested.type.window_buffer is slot


def _c2a_scalar_tuple_calls_program(width, calls, mixed, kind):
    """Share one scalar result type across calls to a flat-return native callee."""
    span = ir.Span.unknown()
    index_type = ir.ScalarType(DataType.INDEX)
    shape = [ir.ConstInt(4, DataType.INDEX, span)]
    ty = ir.DistributedTensorType(shape, pl.FP32)
    formal = ir.Var("formal", ty, span)
    child = ir.Function(
        "scalar_child",
        [formal] if mixed else [],
        [index_type] * width,
        ir.ReturnStmt([ir.ConstInt(i, DataType.INDEX, span) for i in range(width)], span),
        span,
        level=ir.Level.CHIP,
        role=ir.Role.Orchestrator,
    )
    buffer = ir.Var("buf", ir.PtrType(), span)
    data = ir.Var("data", ty, span)
    dependency = ir.Var("dependency", ir.ScalarType(DataType.TASK_ID), span)
    statements = []
    if mixed:
        alloc = ir.create_op_call(
            "pld.tensor.alloc_window_buffer", [ir.ConstInt(16, DataType.INDEX, span)], {"name": "buf"}, span
        )
        window = ir.create_op_call(
            "pld.tensor.window", [buffer, ir.MakeTuple(shape, span)], {"dtype": pl.FP32}, span
        )
        statements.extend([ir.AssignStmt(buffer, alloc, span), ir.AssignStmt(data, window, span)])
    result_fields = [index_type] * width
    if kind == "submit":
        result_fields.append(ir.ScalarType(DataType.TASK_ID))
    shared_result_type = ir.TupleType(result_fields)
    launches = []
    for i in range(calls):
        attrs = {"ordinary": "kept"}
        if mixed:
            attrs["worker"] = ir.ConstInt(0, DataType.INDEX, span)
        if kind == "submit":
            launch = ir.Submit(
                ir.GlobalVar("scalar_child"),
                [data] if mixed else [],
                [dependency],
                {},
                attrs,
                shared_result_type,
                span,
                core_num=ir.ConstInt(2, DataType.INDEX, span),
                sync_start=True,
                allow_early_resolve=True,
                predicate=ir.ConstBool(True, span),
            )
        else:
            launch = ir.Call(
                ir.GlobalVar("scalar_child"), [data] if mixed else [], {}, attrs, shared_result_type, span
            )
        packed = ir.Var(f"packed_{i}", shared_result_type, span)
        launches.append(ir.AssignStmt(packed, launch, span))
        if kind == "submit":
            tid = ir.Var(f"tid_{i}", result_fields[-1], span)
            launches.append(ir.AssignStmt(tid, ir.TupleGetItemExpr(packed, width, span), span))
    if kind == "submit":
        statements.append(ir.RuntimeScopeStmt(manual=True, body=ir.SeqStmts(launches, span), span=span))
    else:
        statements.extend(launches)
    statements.append(ir.ReturnStmt([], span))
    host = ir.Function(
        "host",
        [dependency] if kind == "submit" else [],
        [],
        ir.SeqStmts(statements, span),
        span,
        level=ir.Level.HOST,
        role=ir.Role.Orchestrator,
    )
    return ir.Program([child, host], "ScalarTupleCalls", span)


@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("kind", ["call", "submit"])
def test_c2a_shared_scalar_tuple_call_results_keep_abi_and_changed_dt_arguments(mixed, kind):
    before = _c2a_scalar_tuple_calls_program(4, 3, mixed, kind)
    with passes.PassContext([passes.VerificationInstrument(passes.VerificationMode.BEFORE_AND_AFTER)]):
        after = passes.materialize_comm_domain_scopes()(before)
    host, child = _get_func(after, "host"), _get_func(after, "scalar_child")
    assert len(child.return_types) == 4
    assert all(ty.dtype == DataType.INDEX for ty in child.return_types)
    calls = [
        stmt.value
        for stmt in _c2a_stmts(host.body)
        if isinstance(stmt, ir.AssignStmt)
        and isinstance(stmt.value, (ir.Call, ir.Submit))
        and isinstance(stmt.value.op, ir.GlobalVar)
    ]
    assert len(calls) == 3
    if mixed:
        slot = _get_comm_domain_scopes(host)[0].slots[0]
        assert child.params[0].type.window_buffer is slot
        window_var = _find_window_calls(host)[0].var
    else:
        assert after is before
        assert child.params == []
    for call in calls:
        assert call.attrs["ordinary"] == "kept"
        assert len(call.type.types) == (5 if kind == "submit" else 4)
        assert all(ty.dtype == DataType.INDEX for ty in call.type.types[:4])
        if mixed:
            assert call.args[0] is window_var
            assert call.args[0].type.window_buffer is slot
        else:
            assert call.args == []
        if kind == "submit":
            assert isinstance(call, ir.Submit)
            assert call.deps[0] is host.params[0]
            assert call.type.types[4].dtype == DataType.TASK_ID
            assert call.core_num.value == 2
            assert call.sync_start and call.allow_early_resolve and call.predicate.value
        else:
            assert isinstance(call, ir.Call)
    if kind == "submit":
        projections = [
            stmt
            for stmt in _c2a_stmts(host.body)
            if isinstance(stmt, ir.AssignStmt) and isinstance(stmt.value, ir.TupleGetItemExpr)
        ]
        assert len(projections) == 3
        assert all(stmt.value.index == 4 and stmt.var.type.dtype == DataType.TASK_ID for stmt in projections)


def _c2a_ordinary_tuple_program(width, aliases, mixed):
    """Build a shared type DAG with O(width + aliases) ordinary IR nodes."""
    span = ir.Span.unknown()
    scalar = ir.ScalarType(DataType.INT32)
    shared_type = ir.TupleType([scalar] * width)
    parameter = ir.Var("ordinary_input", shared_type, span)
    previous = parameter
    statements = []
    for index in range(aliases):
        alias = ir.Var(f"ordinary_alias_{index}", shared_type, span)
        statements.append(ir.AssignStmt(alias, previous, span))
        previous = alias
    statements.append(ir.ReturnStmt([previous], span))
    ordinary = ir.Function(
        "ordinary",
        [parameter],
        [shared_type],
        ir.SeqStmts(statements, span),
        span,
        attrs={"ordinary": "kept", "count": 7, "reference": parameter},
    )
    functions = [ordinary]
    if mixed:
        owner = pl.parse_program("""
@pl.program
class UnrelatedWindow:
    @pl.function(level=pl.Level.CHIP, role=pl.Role.Orchestrator)
    def chip(self, data: pld.DistributedTensor[[4], pl.FP32]):
        return data
    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def host(self):
        unrelated_buf = pld.alloc_window_buffer(16)
        data = pld.window(unrelated_buf, [4], dtype=pl.FP32)
        self.chip(data, worker=0)
        return 0
""")
        functions.extend(owner.functions.values())
    return ir.Program(functions, "OrdinaryTypeDag", span)


@pytest.mark.parametrize("mixed", [False, True])
def test_c2a_shared_ordinary_tuple_dag_retains_function_and_metadata(mixed):
    before = _c2a_ordinary_tuple_program(16, 24, mixed)
    ordinary = _get_func(before, "ordinary")
    shared_type = ordinary.params[0].type
    # The input is already legal SSA. Avoid a preprocessing clone so exact
    # object preservation measures only this materializer's behavior.
    with passes.PassContext([passes.VerificationInstrument(passes.VerificationMode.BEFORE_AND_AFTER)]):
        after = passes.materialize_comm_domain_scopes()(before)
    preserved = _get_func(after, "ordinary")
    assert preserved is ordinary
    assert preserved.params[0].type is shared_type
    assert preserved.return_types[0] is shared_type
    assert preserved.attrs["ordinary"] == "kept"
    assert preserved.attrs["count"] == 7
    assert preserved.attrs["reference"] is preserved.params[0]
    assignments = [stmt for stmt in preserved.body.stmts if isinstance(stmt, ir.AssignStmt)]
    assert len(assignments) == 24
    assert all(stmt.var.type is shared_type and stmt.value.type is shared_type for stmt in assignments)
    if mixed:
        slot = _get_comm_domain_scopes(_get_func(after, "host"))[0].slots[0]
        _c2a_assert_function_identity(_get_func(after, "chip"), slot)
    else:
        assert after is before


@pytest.mark.parametrize("payload_kind", ["formal", "reference_list", "view", "tuple_projection"])
def test_c2a_function_expr_attrs_share_rewritten_formal_identity(payload_kind):
    """Function attr expressions are live references, including attr-only views."""
    span = ir.Span.unknown()
    size = ir.ConstInt(16, DataType.INDEX, span)
    shape = [ir.ConstInt(4, DataType.INDEX, span)]
    ty = ir.DistributedTensorType(shape, pl.FP32)
    formal = ir.Var("formal", ty, span)
    ctx = ir.Var("ctx", ir.CommCtxType(), span)
    view = ir.create_op_call(
        "tensor.slice",
        [formal, ir.MakeTuple(shape, span), ir.MakeTuple([ir.ConstInt(0, DataType.INDEX, span)], span)],
        {},
        span,
    )
    # The expression exists only in attrs, so merely substituting the formal
    # cannot fix its result/tuple types without indexing attr lineage as well.
    projection = ir.TupleGetItemExpr(ir.MakeTuple([view, ctx], span), 0, span)
    payloads = {"formal": formal, "reference_list": [formal], "view": view, "tuple_projection": projection}
    chip = ir.Function(
        "chip",
        [formal, ctx],
        [ty],
        ir.ReturnStmt([formal], span),
        span,
        level=ir.Level.CHIP,
        role=ir.Role.Orchestrator,
        attrs={"custom_reference": payloads[payload_kind], "ordinary": "kept", "count": 7},
    )
    buffer = ir.Var("buf", ir.PtrType(), span)
    data = ir.Var("data", ty, span)
    alloc = ir.create_op_call("pld.tensor.alloc_window_buffer", [size], {"name": "buf"}, span)
    window = ir.create_op_call(
        "pld.tensor.window", [buffer, ir.MakeTuple(shape, span)], {"dtype": pl.FP32}, span
    )
    call = ir.Call(
        ir.GlobalVar("chip"), [data, ctx], {}, {"worker": ir.ConstInt(0, DataType.INDEX, span)}, ty, span
    )
    host = ir.Function(
        "host",
        [ctx],
        [],
        ir.SeqStmts(
            [
                ir.AssignStmt(buffer, alloc, span),
                ir.AssignStmt(data, window, span),
                ir.EvalStmt(call, span),
                ir.ReturnStmt([], span),
            ],
            span,
        ),
        span,
        level=ir.Level.HOST,
        role=ir.Role.Orchestrator,
    )
    # Generic Expr-valued function attrs exceed the private DSL printer's
    # supported syntax. Keep native verification for this native API fixture.
    with passes.PassContext([passes.VerificationInstrument(passes.VerificationMode.BEFORE_AND_AFTER)]):
        prepared = _synthesize(passes.convert_to_ssa()(ir.Program([chip, host], "FunctionAttrs", span)))
        prepared_chip = _get_func(prepared, "chip")
        prepared_formal, prepared_ctx = prepared_chip.params
        prepared_type = prepared_formal.type
        after = passes.materialize_comm_domain_scopes()(prepared)
    rewritten = _get_func(after, "chip")
    slot = _get_comm_domain_scopes(_get_func(after, "host"))[0].slots[0]
    assert rewritten.params[0] is not prepared_formal
    assert rewritten.params[0].type.window_buffer is slot
    expected_type = ir.DistributedTensorType(
        prepared_type.shape, prepared_type.dtype, prepared_type.memref, prepared_type.tensor_view, slot
    )
    ir.assert_structural_equal(rewritten.params[0].type, expected_type)
    assert rewritten.params[1] is prepared_ctx
    assert rewritten.return_types[0].window_buffer is slot
    assert isinstance(rewritten.body, ir.ReturnStmt)
    assert rewritten.body.value[0] is rewritten.params[0]
    assert rewritten.attrs["ordinary"] == "kept"
    assert rewritten.attrs["count"] == 7
    payload = rewritten.attrs["custom_reference"]
    if payload_kind == "reference_list":
        assert list(payload) == [rewritten.params[0]]
        assert payload[0] is rewritten.params[0]
    elif payload_kind == "formal":
        assert payload is rewritten.params[0]
    else:
        assert payload.type.window_buffer is slot
        if payload_kind == "tuple_projection":
            assert isinstance(payload, ir.TupleGetItemExpr)
            assert payload.tuple.type.types[0].window_buffer is slot
            assert payload.tuple.elements[1] is rewritten.params[1]
            payload = payload.tuple.elements[0]
        assert isinstance(payload, ir.Call)
        assert payload.type.window_buffer is slot
        assert payload.args[0] is rewritten.params[0]


@pytest.mark.parametrize("kind", ["call", "submit"])
def test_c2a_sparse_tuple_return_positions_preserve_scalar_gaps_and_task_id(kind):
    span = ir.Span.unknown()
    index_type = ir.ScalarType(DataType.INDEX)
    tensor_type = ir.DistributedTensorType([ir.ConstInt(4, DataType.INDEX, span)], pl.FP32)
    formal = ir.Var("formal", tensor_type, span)
    returns = [index_type, tensor_type, index_type, tensor_type]
    chip = ir.Function(
        "chip",
        [formal],
        returns,
        ir.ReturnStmt(
            [ir.ConstInt(0, DataType.INDEX, span), formal, ir.ConstInt(2, DataType.INDEX, span), formal], span
        ),
        span,
        level=ir.Level.CHIP,
        role=ir.Role.Orchestrator,
    )
    buffer = ir.Var("buf", ir.PtrType(), span)
    data = ir.Var("data", tensor_type, span)
    dependency = ir.Var("dependency", ir.ScalarType(DataType.TASK_ID), span)
    size = ir.ConstInt(16, DataType.INDEX, span)
    alloc = ir.create_op_call("pld.tensor.alloc_window_buffer", [size], {"name": "buf"}, span)
    window = ir.create_op_call(
        "pld.tensor.window", [buffer, ir.MakeTuple(tensor_type.shape, span)], {"dtype": pl.FP32}, span
    )
    attrs = {"worker": ir.ConstInt(0, DataType.INDEX, span), "ordinary": "kept"}
    if kind == "submit":
        returns = [*returns, ir.ScalarType(DataType.TASK_ID)]
        invocation = ir.Submit(
            ir.GlobalVar("chip"),
            [data],
            [dependency],
            {},
            attrs,
            ir.TupleType(returns),
            span,
            core_num=ir.ConstInt(2, DataType.INDEX, span),
            sync_start=True,
            allow_early_resolve=True,
            predicate=ir.ConstBool(True, span),
        )
    else:
        invocation = ir.Call(ir.GlobalVar("chip"), [data], {}, attrs, ir.TupleType(returns), span)
    packed = ir.Var("packed", invocation.type, span)
    projections = [ir.Var(f"projection_{index}", ty, span) for index, ty in enumerate(returns)]
    launch_statements = [ir.AssignStmt(packed, invocation, span)] + [
        ir.AssignStmt(var, ir.TupleGetItemExpr(packed, index, span), span)
        for index, var in enumerate(projections)
    ]
    if kind == "submit":
        launch_statements = [
            ir.RuntimeScopeStmt(
                manual=True,
                body=ir.SeqStmts(launch_statements, span),
                span=span,
            )
        ]
    host = ir.Function(
        "host",
        [dependency] if kind == "submit" else [],
        [tensor_type, tensor_type],
        ir.SeqStmts(
            [
                ir.AssignStmt(buffer, alloc, span),
                ir.AssignStmt(data, window, span),
                *launch_statements,
                ir.ReturnStmt([projections[1], projections[3]], span),
            ],
            span,
        ),
        span,
        level=ir.Level.HOST,
        role=ir.Role.Orchestrator,
    )
    with passes.PassContext([passes.VerificationInstrument(passes.VerificationMode.BEFORE_AND_AFTER)]):
        after = _apply(ir.Program([chip, host], "SparseReturnPositions", span))
    rewritten_host, rewritten_chip = _get_func(after, "host"), _get_func(after, "chip")
    slot = _get_comm_domain_scopes(rewritten_host)[0].slots[0]
    assert rewritten_chip.params[0].type.window_buffer is slot
    assert rewritten_chip.body.value[1] is rewritten_chip.params[0]
    assert rewritten_chip.body.value[3] is rewritten_chip.params[0]
    assert len(rewritten_chip.return_types) == 4
    for index in [0, 2]:
        assert rewritten_chip.return_types[index].dtype == DataType.INDEX
    for index in [1, 3]:
        assert rewritten_chip.return_types[index].window_buffer is slot
    assignments = [stmt for stmt in _c2a_stmts(rewritten_host.body) if isinstance(stmt, ir.AssignStmt)]
    launch = next(
        stmt.value
        for stmt in assignments
        if isinstance(stmt.value, (ir.Call, ir.Submit)) and isinstance(stmt.value.op, ir.GlobalVar)
    )
    assert launch.attrs["ordinary"] == "kept"
    assert len(launch.type.types) == len(returns)
    projected = [stmt for stmt in assignments if isinstance(stmt.value, ir.TupleGetItemExpr)]
    assert [stmt.value.index for stmt in projected] == list(range(len(returns)))
    for index, stmt in enumerate(projected):
        ir.assert_structural_equal(stmt.var.type, launch.type.types[index])
        ir.assert_structural_equal(stmt.value.type, launch.type.types[index])
        if index in [1, 3]:
            assert stmt.var.type.window_buffer is slot
            assert stmt.value.type.window_buffer is slot
        else:
            expected_dtype = DataType.TASK_ID if index == 4 else DataType.INDEX
            assert stmt.var.type.dtype == expected_dtype
    assert all(ty.window_buffer is slot for ty in rewritten_host.return_types)
    if kind == "submit":
        assert isinstance(launch, ir.Submit)
        assert launch.deps[0] is rewritten_host.params[0]
        assert launch.type.types[4].dtype == DataType.TASK_ID
        assert launch.core_num.value == 2
        assert launch.sync_start and launch.allow_early_resolve
        assert launch.predicate.value is True
    else:
        assert isinstance(launch, ir.Call)


@pytest.mark.parametrize("kind,supplied_out", [("call", True), ("submit", True), ("submit", False)])
@pytest.mark.parametrize("with_ctx", [False, True])
def test_c2a_call_submit_binding_retains_local_metadata_and_task_fields(kind, supplied_out, with_ctx):
    """An omitted Out is a gap, not a caller value or the trailing ctx argument.

    Native nodes expose launch fields and deliberately different callee MemRef
    metadata; only the WindowBuffer field may be inherited from the caller.
    """
    span = ir.Span.unknown()
    size = ir.ConstInt(16, DataType.INDEX, span)
    shape = [ir.ConstInt(4, DataType.INDEX, span)]
    local_type = ir.DistributedTensorType(shape, pl.FP32)
    memref = ir.MemRef("callee_storage", 0, 16, span)
    view = ir.TensorView(stride=[1], layout=ir.TensorLayout.ND)
    formal_type = ir.DistributedTensorType(shape, pl.FP32, memref, view)
    formal = ir.Var("data", formal_type, span)
    output = ir.Var("output", local_type, span)
    ctx = ir.Var("ctx", ir.CommCtxType(), span)
    callee_params = [(formal, ir.ParamDirection.In), (output, ir.ParamDirection.Out)]
    if with_ctx:
        callee_params.append((ctx, ir.ParamDirection.In))
    callee = ir.Function(
        "chip",
        callee_params,
        [formal_type, local_type],
        ir.ReturnStmt([formal, output], span),
        span,
        level=ir.Level.CHIP,
        role=ir.Role.Orchestrator,
    )
    buffer = ir.Var("buf", ir.PtrType(), span)
    data = ir.Var("data", local_type, span)
    alloc = ir.create_op_call("pld.tensor.alloc_window_buffer", [size], {"name": "buf"}, span)
    window = ir.create_op_call(
        "pld.tensor.window", [buffer, ir.MakeTuple(shape, span)], {"dtype": pl.FP32}, span
    )
    dep = ir.Var("dependency", ir.ScalarType(DataType.TASK_ID), span)
    args = [data] + ([data] if supplied_out else []) + ([ctx] if with_ctx else [])
    attrs = {"worker": ir.ConstInt(0, DataType.INDEX, span), "saved": "identity-test"}
    returned = [local_type, local_type]
    if kind == "submit":
        returned.append(ir.ScalarType(DataType.TASK_ID))
        invocation = ir.Submit(
            ir.GlobalVar("chip"),
            args,
            [dep],
            {},
            attrs,
            ir.TupleType(returned),
            span,
            core_num=ir.ConstInt(2, DataType.INDEX, span),
            sync_start=True,
            allow_early_resolve=True,
            predicate=ir.ConstBool(True, span),
        )
    else:
        invocation = ir.Call(ir.GlobalVar("chip"), args, {}, attrs, ir.TupleType(returned), span)
    packed = ir.Var("packed", invocation.type, span)
    first = ir.Var("first", local_type, span)
    launch_body = ir.SeqStmts(
        [
            ir.AssignStmt(packed, invocation, span),
            ir.AssignStmt(first, ir.TupleGetItemExpr(packed, 0, span), span),
        ],
        span,
    )
    if kind == "submit":
        launch_body = ir.RuntimeScopeStmt(manual=True, body=launch_body, span=span)
    launch_stmts = list(launch_body.stmts) if isinstance(launch_body, ir.SeqStmts) else [launch_body]
    caller_params = ([ctx] if with_ctx else []) + ([dep] if kind == "submit" else [])
    caller = ir.Function(
        "host",
        caller_params,
        [local_type],
        ir.SeqStmts(
            [
                ir.AssignStmt(buffer, alloc, span),
                ir.AssignStmt(data, window, span),
                *launch_stmts,
                ir.ReturnStmt([first], span),
            ],
            span,
        ),
        span,
        level=ir.Level.HOST,
        role=ir.Role.Orchestrator,
    )
    # This native ABI fixture includes a runtime-allocated Out gap on a HOST
    # Submit and low-level launch fields not authored by the current DSL. Keep
    # native verification; textual round-trip coverage lives in its own test.
    with passes.PassContext([passes.VerificationInstrument(passes.VerificationMode.BEFORE_AND_AFTER)]):
        after = _apply(ir.Program([callee, caller], "Binding", span))
    host, chip = _get_func(after, "host"), _get_func(after, "chip")
    slot = _get_comm_domain_scopes(host)[0].slots[0]
    data_type, output_type = chip.params[0].type, chip.params[1].type
    assert data_type.window_buffer is slot
    assert data_type.memref is memref
    # TensorView is a value type: its constructor input and the type's cached
    # Python getter wrapper need not be the same object. Check every recorded
    # field structurally, retaining separate MemRef/WindowBuffer identity checks.
    expected_type = ir.DistributedTensorType(shape, pl.FP32, memref, formal_type.tensor_view, slot)
    ir.assert_structural_equal(data_type, expected_type)
    assert data_type.dtype == pl.FP32
    assert [dim.value for dim in data_type.shape] == [4]
    assert output_type.window_buffer is (slot if supplied_out else None)
    assert chip.return_types[0].window_buffer is slot
    assert chip.return_types[1].window_buffer is (slot if supplied_out else None)
    assignments = [stmt for stmt in _c2a_stmts(host.body) if isinstance(stmt, ir.AssignStmt)]
    launch = next(
        stmt.value
        for stmt in assignments
        if isinstance(stmt.value, (ir.Call, ir.Submit)) and isinstance(stmt.value.op, ir.GlobalVar)
    )
    assert launch.attrs["saved"] == "identity-test"
    assert len(launch.args) == len(args)
    if with_ctx:
        assert isinstance(launch.args[-1].type, ir.CommCtxType)
    assert launch.type.types[0].window_buffer is slot
    assert launch.type.types[1].window_buffer is (slot if supplied_out else None)
    projection = next(stmt for stmt in assignments if isinstance(stmt.value, ir.TupleGetItemExpr))
    assert projection.var.type.window_buffer is projection.value.type.window_buffer is slot
    if kind == "submit":
        assert isinstance(launch, ir.Submit)
        assert len(launch.deps) == 1
        assert launch.deps[0] is host.params[-1]
        assert launch.type.types[-1].dtype == DataType.TASK_ID
        assert launch.core_num.value == 2
        assert launch.sync_start and launch.allow_early_resolve
        assert launch.predicate.value is True
    else:
        assert isinstance(launch, ir.Call)


def test_c2a_orphan_typed_formal_is_rejected_but_nullable_device_formal_is_allowed():
    span = ir.Span.unknown()
    shape = [ir.ConstInt(4, DataType.INDEX, span)]
    slot = ir.WindowBuffer(ir.Var("orphan", ir.PtrType(), span), ir.ConstInt(16, DataType.INDEX, span))
    for window in (None, slot):
        ty = ir.DistributedTensorType(shape, pl.FP32, None, None, window)
        param = ir.Var("data", ty, span)
        func = ir.Function(
            "chip",
            [param],
            [ty],
            ir.ReturnStmt([param], span),
            span,
            level=ir.Level.CHIP,
            role=ir.Role.Orchestrator,
        )
        before = ir.Program([func], "Standalone", span)
        if window is None:
            after = passes.materialize_comm_domain_scopes()(before)
            ir.assert_structural_equal(before, after)
        else:
            # An intentionally invalid identity cannot round-trip: the printer
            # has no scope definition to emit. Pin the pass's own diagnostic.
            with passes.PassContext([], verification_level=passes.VerificationLevel.NONE):
                with pytest.raises(ValueError, match="[Oo]rphan|[Oo]wner|[Ss]cope"):
                    passes.materialize_comm_domain_scopes()(before)


@pytest.mark.parametrize("start,step", [("1", "1"), ("0", "2"), ("begin", "1"), ("0", "stride")])
def test_c2a_cluster_dynamic_coverage_rejects_noncanonical_range(start, step):
    before = _c2a_three_level_program(start=start, step=step)
    with pytest.raises(ValueError, match="start|step|full.coverage"):
        _apply(before)


@pytest.mark.parametrize("binding", ["nullable", "orphan", "owned"])
def test_c2a_property_verifier_checks_formal_window_ownership_without_running_pass(binding):
    """A serialized/externally constructed program must satisfy the property too."""
    span = ir.Span.unknown()
    shape = [ir.ConstInt(4, DataType.INDEX, span)]
    size = ir.ConstInt(16, DataType.INDEX, span)
    base = ir.Var("buf", ir.PtrType(), span)
    slot = ir.WindowBuffer(base, size)
    ty = ir.DistributedTensorType(shape, pl.FP32, None, None, None if binding == "nullable" else slot)
    formal = ir.Var("data", ty, span)
    chip = ir.Function(
        "chip",
        [formal],
        [ty],
        ir.ReturnStmt([formal], span),
        span,
        level=ir.Level.CHIP,
        role=ir.Role.Orchestrator,
    )
    functions = [chip]
    if binding == "owned":
        data = ir.Var("data", ty, span)
        alloc = ir.create_op_call("pld.tensor.alloc_window_buffer", [size], {"name": "buf"}, span)
        window = ir.Call(
            ir.get_op("pld.tensor.window"),
            [base, ir.MakeTuple(shape, span)],
            {"dtype": pl.FP32},
            {},
            ty,
            span,
        )
        call = ir.Call(
            ir.GlobalVar("chip"), [data], {}, {"worker": ir.ConstInt(0, DataType.INDEX, span)}, ty, span
        )
        body = ir.CommDomainScopeStmt(
            [0],
            [slot],
            "owned",
            body=ir.SeqStmts(
                [
                    ir.AssignStmt(base, alloc, span),
                    ir.AssignStmt(data, window, span),
                    ir.EvalStmt(call, span),
                    ir.ReturnStmt([], span),
                ],
                span,
            ),
            span=span,
        )
        functions.append(
            ir.Function(
                "host",
                [],
                [],
                body,
                span,
                level=ir.Level.HOST,
                role=ir.Role.Orchestrator,
            )
        )
    props = passes.IRPropertySet()
    props.insert(passes.IRProperty.CommDomainScopesMaterialized)
    diagnostics = passes.PropertyVerifierRegistry.verify(props, ir.Program(functions, "Ownership", span))
    errors = [item for item in diagnostics if item.severity == passes.DiagnosticSeverity.Error]
    if binding == "orphan":
        assert errors, "the verifier must reject a typed formal whose WindowBuffer has no owner"
        assert all(item.rule_name == "CommDomainScopesMaterialized" for item in errors)
        assert any("orphan" in item.message.lower() or "owner" in item.message.lower() for item in errors)
    else:
        assert errors == []


@pytest.mark.parametrize("repetitions", [1, 32])
def test_c2a_repeated_collective_edges_observe_updated_source_coverage(repetitions):
    """A repeated A -> S edge must see A's new coverage after B -> A.

    This pins directed evidence union while repeated consumers are deduplicated;
    widening the data domains as an undirected component would also be wrong.
    """
    dispatches = "        self.chip_a(a, worker=0)\n        self.chip_b(b, worker=1)\n" * repetitions
    consume = "        a = pld.tensor.allreduce(a, signal, op=pld.ReduceOp.Sum)\n" * repetitions
    before = pl.parse_program(
        """
@pl.program
class CoverageSnapshots:
    @pl.function(type=pl.FunctionType.Orchestration)
    def chip_a(self, data: pld.DistributedTensor[[4], pl.FP32]):
        return data
    @pl.function(type=pl.FunctionType.Orchestration)
    def chip_b(self, data: pld.DistributedTensor[[4], pl.FP32]):
        return data
    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def host(self):
        a_buf = pld.alloc_window_buffer(16)
        b_buf = pld.alloc_window_buffer(16)
        signal_buf = pld.alloc_window_buffer(16)
        a = pld.window(a_buf, [4], dtype=pl.FP32)
        b = pld.window(b_buf, [4], dtype=pl.FP32)
        a_signal = pld.window(a_buf, [4], dtype=pl.INT32)
        signal = pld.window(signal_buf, [4], dtype=pl.INT32)
"""
        + dispatches
        + consume
        + "        b = pld.tensor.allreduce(b, a_signal, op=pld.ReduceOp.Sum)\n"
        + consume
        + "        return 0\n"
    )
    after = _apply(before)
    host = _get_func(after, "host")
    scopes = _get_comm_domain_scopes(host)
    assert [list(scope.devices) for scope in scopes] == [[0, 1], [1]]
    assert [len(scope.slots) for scope in scopes] == [2, 1]
    a_type, b_type, a_signal_type, signal_type = _view_var_types(host)
    assert a_type.window_buffer is scopes[0].slots[0]
    assert a_signal_type.window_buffer is a_type.window_buffer
    assert signal_type.window_buffer is scopes[0].slots[1]
    assert b_type.window_buffer is scopes[1].slots[0]


def test_c2a_host_owner_keeps_chip_graph_out_of_dispatch_coverage():
    before = pl.parse_program("""
@pl.program
class GraphCompatibility:
    @pl.function(level=pl.Level.CHIP, role=pl.Role.Orchestrator)
    def chip(self, data: pld.DistributedTensor[[4], pl.FP32]):
        return data
    @pl.function(type=pl.FunctionType.Graph)
    def graph(self, data: pld.DistributedTensor[[4], pl.FP32]):
        return data
    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def host(self):
        buf = pld.alloc_window_buffer(16)
        data = pld.window(buf, [4], dtype=pl.FP32)
        self.chip(data, worker=0)
        self.graph(data)
        return 0
""")
    after = _apply(before)
    scope = _get_comm_domain_scopes(_get_func(after, "host"))[0]
    assert list(scope.devices) == [0]
    _c2a_assert_function_identity(_get_func(after, "chip"), scope.slots[0])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
