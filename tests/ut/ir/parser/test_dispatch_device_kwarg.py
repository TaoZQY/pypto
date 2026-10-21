# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: F722, F821

"""Compatibility and canonical worker-placement tests for hierarchical dispatch.

``self.<orch>(..., device=r)`` is the deprecated spelling for a logical child
dispatch. The kwarg is only legal when the callee is an orchestrator. The parser is intentionally
permissive about the value form — any IR expression is parsed and stashed
on ``call.attrs['worker']``. Downstream passes (simplify, comm-collection)
inspect the resolved expression to derive the per-dispatch device subset:

* ``ConstInt`` → fixed-device dispatch
* IR Var that's a ``for``-loop induction variable → subset / kAll
* Anything else → rejected by the comm-collection pass with full
  def-use context
"""

import warnings

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
from pypto.language.parser.diagnostics import ParserError, ParserTypeError
from pypto.pypto_core import ir


def _get_func(program: ir.Program, name: str) -> ir.Function:
    gvar = program.get_global_var(name)
    assert gvar is not None
    return program.functions[gvar]


def _find_callee_call(func: ir.Function, callee_name: str) -> ir.Call:
    found: list[ir.Call] = []

    def maybe_record(expr: ir.Expr | None) -> None:
        if isinstance(expr, ir.Call) and expr.op.name == callee_name:
            found.append(expr)

    def walk(stmt: ir.Stmt) -> None:
        if isinstance(stmt, ir.AssignStmt):
            maybe_record(stmt.value)
        if isinstance(stmt, ir.EvalStmt):
            maybe_record(stmt.expr)
        if isinstance(stmt, ir.SeqStmts):
            for s in stmt.stmts:
                walk(s)
        if isinstance(stmt, ir.ForStmt):
            walk(stmt.body)

    walk(func.body)
    assert found, f"no call to '{callee_name}' found in function body"
    return found[0]


# ---------------------------------------------------------------------------
# Positive cases — parser stashes the IR expression on ``call.attrs['worker']``
# ---------------------------------------------------------------------------


def test_device_int_literal():
    """``device=0`` lifts to a ConstInt stored on call.attrs['worker']."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(self, x: pl.Tensor[[64], pl.FP32]):
            return x

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self, x: pl.Tensor[[64], pl.FP32]):
            self.chip_orch(x, device=0)
            return 0

    func = _get_func(P, "host_orch")
    call = _find_callee_call(func, "chip_orch")
    device = call.attrs["worker"]
    assert isinstance(device, ir.ConstInt)
    assert device.value == 0


def test_device_uses_for_loop_induction_var():
    """``device=r`` parses to the IR Var of the enclosing ``for r in pl.range(...)``."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(self, x: pl.Tensor[[64], pl.FP32]):
            return x

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self, x: pl.Tensor[[64], pl.FP32]):
            for r in pl.range(pld.world_size()):  # type: ignore[arg-type]
                self.chip_orch(x, device=r)
            return 0

    func = _get_func(P, "host_orch")
    call = _find_callee_call(func, "chip_orch")
    device = call.attrs["worker"]
    assert isinstance(device, ir.Var)
    assert device.name_hint == "r"
    assert isinstance(device.type, ir.ScalarType)


def test_device_accepts_locally_bound_constant():
    """``other = 3; self.chip_orch(..., device=other)`` is accepted — ``other``
    parses to a Var bound to ConstInt(3); the comm-collection pass folds it
    via simplify before deriving the device subset."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(self, x: pl.Tensor[[64], pl.FP32]):
            return x

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self, x: pl.Tensor[[64], pl.FP32]):
            other = 3
            self.chip_orch(x, device=other)
            return 0

    func = _get_func(P, "host_orch")
    call = _find_callee_call(func, "chip_orch")
    device = call.attrs["worker"]
    # The parser hands back the Var; downstream simplify can fold it to
    # ConstInt(3) when the comm-collection pass needs a literal.
    assert isinstance(device, ir.Var)
    assert device.name_hint == "other"


def test_device_accepts_arithmetic_expression():
    """``device=r + 1`` parses to an arithmetic IR expression. The
    comm-collection pass is responsible for any further validation."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(self, x: pl.Tensor[[64], pl.FP32]):
            return x

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self, x: pl.Tensor[[64], pl.FP32]):
            for r in pl.range(pld.world_size()):  # type: ignore[arg-type]
                self.chip_orch(x, device=r + 1)
            return 0

    func = _get_func(P, "host_orch")
    call = _find_callee_call(func, "chip_orch")
    device = call.attrs["worker"]
    # Parser does not constrain shape — comm-collection pass will validate.
    assert isinstance(device, ir.Expr)


def test_device_accepts_explicit_level_role_orchestrator():
    """``FunctionType.Orchestration`` and ``level=..., role=Orchestrator`` are
    two ways to declare the same role. Both must accept ``device=`` on
    dispatch sites — the parser predicate is on ``role``, not ``func_type``."""

    @pl.program
    class P:
        @pl.function(level=pl.Level.CHIP, role=pl.Role.Orchestrator)
        def chip_orch(self, x: pl.Tensor[[64], pl.FP32]):
            return x

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self, x: pl.Tensor[[64], pl.FP32]):
            self.chip_orch(x, device=2)
            return 0

    func = _get_func(P, "host_orch")
    call = _find_callee_call(func, "chip_orch")
    device = call.attrs["worker"]
    assert isinstance(device, ir.ConstInt)
    assert device.value == 2


# ---------------------------------------------------------------------------
# Negative cases — only the callee-type check stays in the parser
# ---------------------------------------------------------------------------


def test_device_rejected_on_non_orchestrator_callee():
    """A callee that is not an Orchestrator (default ``Opaque`` /
    ``SubWorker``) does not accept ``device=``. Predicate is on ``role``,
    not ``func_type``."""
    with pytest.raises(ParserTypeError, match="'device'"):

        @pl.program
        class P:  # noqa: F841
            @pl.function
            def kernel(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
            def host_orch(self, x: pl.Tensor[[64], pl.FP32]):
                self.kernel(x, device=0)
                return 0


def _parse_worker_dispatch(placement: str, *, submit: bool = False) -> tuple[ir.Program, ir.Expr]:
    """Parse one HOST-to-CHIP launch using the public placement spelling."""
    launch = (
        f"result, task = pl.submit(self.chip_orch, x, {placement})"
        if submit
        else f"self.chip_orch(x, {placement})"
    )
    program = pl.parse_program(f"""
import pypto.language as pl

@pl.program
class P:
    @pl.function(level=pl.Level.CHIP, role=pl.Role.Orchestrator)
    def chip_orch(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        return x

    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def host_orch(self, x: pl.Tensor[[64], pl.FP32]):
        for r in pl.range(4):
            {launch}
        return 0
""")
    launches: list[ir.Expr] = []

    class FindLaunch(ir.IRVisitor):
        def visit_call(self, op):
            if isinstance(op.op, ir.GlobalVar) and op.op.name == "chip_orch":
                launches.append(op)
            super().visit_call(op)

        def visit_submit(self, op):
            if op.op.name == "chip_orch":
                launches.append(op)
            super().visit_submit(op)

    FindLaunch().visit_program(program)
    assert len(launches) == 1
    return program, launches[0]


@pytest.mark.parametrize("submit", [False, True], ids=["call", "submit"])
@pytest.mark.parametrize(
    ("placement", "expr_type"),
    [
        ("worker=0", ir.ConstInt),
        ("worker=r", ir.Var),
        ("worker=r + 1", ir.Add),
        ('attrs={"worker": r}', ir.Var),
    ],
)
def test_worker_dispatch_uses_canonical_expr_attr(submit, placement, expr_type):
    """The child selector is a typed Expr on both call-like IR nodes."""
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always", DeprecationWarning)
        program, launch = _parse_worker_dispatch(placement, submit=submit)
    assert not [warning for warning in recorded if issubclass(warning.category, DeprecationWarning)]
    assert isinstance(launch, ir.Submit if submit else ir.Call)
    assert "device" not in launch.attrs
    assert isinstance(launch.attrs["worker"], expr_type)
    assert "worker" not in launch.kwargs
    text = program.as_python()
    assert "device=" not in text and '"device":' not in text
    assert "__FREE_VAR" not in text
    ir.assert_structural_equal(program, pl.parse_program(text))


@pytest.mark.parametrize("submit", [False, True], ids=["call", "submit"])
def test_worker_dispatch_device_alias_warns_and_normalizes(submit):
    with pytest.warns(DeprecationWarning, match=r"device.*worker"):
        _, launch = _parse_worker_dispatch("device=r", submit=submit)
    assert "device" not in launch.attrs
    assert isinstance(launch.attrs["worker"], ir.Var)
    assert launch.attrs["worker"].name_hint == "r"


@pytest.mark.parametrize("submit", [False, True], ids=["call", "submit"])
def test_worker_dispatch_legacy_attrs_normalize(submit):
    _, launch = _parse_worker_dispatch('attrs={"device": r}', submit=submit)
    assert "device" not in launch.attrs
    assert isinstance(launch.attrs["worker"], ir.Var)


@pytest.mark.parametrize("submit", [False, True], ids=["call", "submit"])
@pytest.mark.parametrize(
    "placement",
    [
        "worker=0, device=0",
        'worker=0, attrs={"device": 1}',
        'device=0, attrs={"worker": 1}',
        'worker=0, attrs={"worker": 1}',
        'device=0, attrs={"device": 1}',
        'attrs={"worker": 0, "device": 0}',
    ],
)
def test_worker_dispatch_rejects_competing_placement_sources(submit, placement):
    """Even equal aliases are ambiguous; attrs cannot silently override a kwarg."""
    with pytest.raises(ParserError, match=r"worker|device"):
        _parse_worker_dispatch(placement, submit=submit)


def test_worker_dispatch_accepts_host_callee_metadata_at_cluster_level():
    """Parsing uses hierarchy metadata; this does not assert POD codegen support."""
    program = pl.parse_program("""
import pypto.language as pl

@pl.program
class P:
    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def host_orch(self, x: pl.Tensor[[64], pl.FP32]):
        return x

    @pl.function(level=pl.Level.CLUSTER_0, role=pl.Role.Orchestrator)
    def pod_orch(self, x: pl.Tensor[[64], pl.FP32]):
        self.host_orch(x, worker=0)
        return 0
""")
    launch = _find_callee_call(_get_func(program, "pod_orch"), "host_orch")
    assert "worker" in launch.attrs and "device" not in launch.attrs
    assert launch.attrs["worker"].value == 0


def test_worker_dispatch_preserves_ordinary_kernel_parameter_names():
    """Positional scalar parameters called worker/device are ordinary values."""
    program = pl.parse_program("""
import pypto.language as pl

@pl.program
class P:
    @pl.function
    def kernel(self, worker: pl.Scalar[pl.INDEX], device: pl.Scalar[pl.INDEX]):
        return worker + device

    @pl.function
    def caller(self, worker: pl.Scalar[pl.INDEX], device: pl.Scalar[pl.INDEX]):
        result = self.kernel(worker, device)
        return result
""")
    launch = _find_callee_call(_get_func(program, "caller"), "kernel")
    assert not launch.attrs
    assert [arg.name_hint for arg in launch.args] == ["worker", "device"]


def _non_orchestrator_dispatch_source(attrs: str, *, submit: bool) -> str:
    launch = (
        f"out, task = pl.submit(self.kernel, x, attrs={attrs})"
        if submit
        else f"self.kernel(x, attrs={attrs})"
    )
    return f"""
import pypto.language as pl

@pl.program
class P:
    @pl.function
    def kernel(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        return x

    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def host_orch(self, x: pl.Tensor[[64], pl.FP32]):
        {launch}
        return 0
"""


@pytest.mark.parametrize("submit", [False, True], ids=["call", "submit"])
@pytest.mark.parametrize("key", ["worker", "device"])
def test_worker_dispatch_attrs_reject_non_orchestrator_callee(submit, key):
    """Machine attrs cannot bypass the role check on public placement kwargs."""
    with pytest.raises(ParserTypeError, match=r"worker|device"):
        pl.parse_program(_non_orchestrator_dispatch_source(f'{{"{key}": 0}}', submit=submit))


@pytest.mark.parametrize("submit", [False, True], ids=["call", "submit"])
def test_worker_dispatch_non_orchestrator_keeps_ordinary_custom_attrs(submit):
    """Restrict placement metadata without closing the generic attr namespace."""
    program = pl.parse_program(_non_orchestrator_dispatch_source('{"label": "ordinary"}', submit=submit))
    launches: list[ir.Call | ir.Submit] = []

    class FindLaunch(ir.IRVisitor):
        def visit_call(self, op):
            if isinstance(op.op, ir.GlobalVar) and op.op.name == "kernel":
                launches.append(op)
            super().visit_call(op)

        def visit_submit(self, op):
            if op.op.name == "kernel":
                launches.append(op)
            super().visit_submit(op)

    FindLaunch().visit_program(program)
    assert len(launches) == 1
    assert dict(launches[0].attrs) == {"label": "ordinary"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
