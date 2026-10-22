# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: F722, F821

"""Tests for ``MaterializeDistTensorCtx``."""

import pypto
import pypto.language as pl
import pypto.language.distributed as pld
import pytest
from pypto import DataType
from pypto.pypto_core import ir, passes


@pytest.fixture(autouse=True)
def _basic_verification_context():
    with passes.PassContext([passes.VerificationInstrument(passes.VerificationMode.BEFORE_AND_AFTER)]):
        yield


def _get_func(program: ir.Program, name: str) -> ir.Function:
    gvar = program.get_global_var(name)
    assert gvar is not None
    return program.functions[gvar]


def _collect_assign_stmts(stmt: ir.Stmt) -> list[ir.AssignStmt]:
    found: list[ir.AssignStmt] = []

    def walk(s: ir.Stmt) -> None:
        if isinstance(s, ir.AssignStmt):
            found.append(s)
        if isinstance(s, ir.SeqStmts):
            for child in s.stmts:
                walk(child)
        if isinstance(s, ir.ForStmt):
            walk(s.body)
        if isinstance(s, ir.ScopeStmt):
            walk(s.body)

    walk(stmt)
    return found


def _collect_calls(stmt: ir.Stmt, op_name: str) -> list[ir.Call]:
    calls: list[ir.Call] = []

    def visit_expr(expr: ir.Expr) -> None:
        if isinstance(expr, ir.Call):
            if expr.op.name == op_name:
                calls.append(expr)
            for arg in expr.args:
                visit_expr(arg)

    for assign in _collect_assign_stmts(stmt):
        visit_expr(assign.value)

    def walk_eval(s: ir.Stmt) -> None:
        if isinstance(s, ir.EvalStmt):
            visit_expr(s.expr)
        if isinstance(s, ir.SeqStmts):
            for child in s.stmts:
                walk_eval(child)
        if isinstance(s, ir.ForStmt):
            walk_eval(s.body)
        if isinstance(s, ir.ScopeStmt):
            walk_eval(s.body)

    walk_eval(stmt)
    return calls


def _span() -> ir.Span:
    return ir.Span("test_materialize_dist_tensor_ctx.py", 1, 1)


def _dist_ty() -> ir.DistributedTensorType:
    return ir.DistributedTensorType([4], pl.FP32)


def _apply(program: ir.Program) -> ir.Program:
    program = passes.materialize_comm_domain_scopes()(program)
    program = passes.lower_host_tensor_collectives()(program)
    program = passes.derive_call_directions()(program)
    return passes.materialize_dist_tensor_ctx()(program)


def _derive_and_materialize(program: ir.Program) -> ir.Program:
    """DeriveCallDirections (pass 41) then MaterializeDistTensorCtx (pass 47).

    Pass 40 is what stamps ``arg_directions`` on each call, so running it first
    lets a DSL-authored ``Before`` reach pass 46 in the shape the pipeline
    actually delivers, with no hand-written direction attrs.
    """
    return passes.materialize_dist_tensor_ctx()(passes.derive_call_directions()(program))


def test_host_dispatch_materializes_comm_ctx_args():
    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            data: pld.DistributedTensor[[256], pl.FP32],
            signal: pld.DistributedTensor[[4], pl.INT32],
        ):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            data_buf = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
            signal_buf = pld.alloc_window_buffer(4 * pl.INT32.get_byte())
            data = pld.window(data_buf, [256], dtype=pl.FP32)
            signal = pld.window(signal_buf, [4], dtype=pl.INT32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(data, signal, device=r)
            return 0

    result = _apply(P)
    chip = _get_func(result, "chip_orch")
    host = _get_func(result, "host_orch")

    assert [type(param.type) for param in chip.params][-2:] == [ir.CommCtxType, ir.CommCtxType]
    assert list(chip.param_directions)[-2:] == [ir.ParamDirection.In, ir.ParamDirection.In]

    calls = _collect_calls(host.body, "chip_orch")
    assert len(calls) == 1
    call = calls[0]
    assert len(call.args) == 4
    assert isinstance(call.args[2].type, ir.CommCtxType)
    assert isinstance(call.args[3].type, ir.CommCtxType)
    assert list(call.arg_directions)[-2:] == [ir.ArgDirection.Scalar, ir.ArgDirection.Scalar]

    get_ctx_calls = _collect_calls(host.body, "pld.system.get_comm_ctx")
    assert len(get_ctx_calls) == 2
    assert get_ctx_calls[0].args[0] is call.args[0]
    assert get_ctx_calls[1].args[0] is call.args[1]
    ctx_assigns = [
        assign for assign in _collect_assign_stmts(host.body) if isinstance(assign.var.type, ir.CommCtxType)
    ]
    assert [assign.var.name_hint for assign in ctx_assigns] == ["data_ctx", "signal_ctx"]


# ---------------------------------------------------------------------------
# Manual Spmd / Group wrapper chain: main -> wrapper -> inner.
#
# Every function takes the same DistributedTensor, so the pass must thread one
# materialized ``data_ctx`` param through all three and forward it at both call
# sites. ``derive_call_directions`` (pass 41) runs first, exactly as in the
# pipeline, so the Before programs carry no hand-written ``arg_directions``.
#
# Spmd and Group get their own program pair because ``@pl.function(type=...)``
# is read from the decorator's AST, so the value cannot come from a parameter.
# ---------------------------------------------------------------------------


@pl.program
class _SpmdWrapper:
    @pl.function(type=pl.FunctionType.InCore)
    def inner(self, data: pld.DistributedTensor[[4], pl.FP32]):
        return

    @pl.function(type=pl.FunctionType.Spmd)
    def wrapper(self, data: pld.DistributedTensor[[4], pl.FP32]):
        self.inner(data)

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, data: pld.DistributedTensor[[4], pl.FP32]):
        self.wrapper(data)


@pl.program
class _SpmdWrapperExpected:
    @pl.function(type=pl.FunctionType.InCore)
    def inner(self, data: pld.DistributedTensor[[4], pl.FP32], data_ctx: pld.CommCtx):
        return

    @pl.function(type=pl.FunctionType.Spmd)
    def wrapper(self, data: pld.DistributedTensor[[4], pl.FP32], data_ctx: pld.CommCtx):
        self.inner(data, data_ctx, attrs={"arg_directions": [pl.adir.input, pl.adir.scalar]})

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, data: pld.DistributedTensor[[4], pl.FP32], data_ctx: pld.CommCtx):
        self.wrapper(data, data_ctx, attrs={"arg_directions": [pl.adir.input, pl.adir.scalar]})


@pl.program
class _GroupWrapper:
    @pl.function(type=pl.FunctionType.InCore)
    def inner(self, data: pld.DistributedTensor[[4], pl.FP32]):
        return

    @pl.function(type=pl.FunctionType.Group)
    def wrapper(self, data: pld.DistributedTensor[[4], pl.FP32]):
        self.inner(data)

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, data: pld.DistributedTensor[[4], pl.FP32]):
        self.wrapper(data)


@pl.program
class _GroupWrapperExpected:
    @pl.function(type=pl.FunctionType.InCore)
    def inner(self, data: pld.DistributedTensor[[4], pl.FP32], data_ctx: pld.CommCtx):
        return

    @pl.function(type=pl.FunctionType.Group)
    def wrapper(self, data: pld.DistributedTensor[[4], pl.FP32], data_ctx: pld.CommCtx):
        self.inner(data, data_ctx, attrs={"arg_directions": [pl.adir.input, pl.adir.scalar]})

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, data: pld.DistributedTensor[[4], pl.FP32], data_ctx: pld.CommCtx):
        self.wrapper(data, data_ctx, attrs={"arg_directions": [pl.adir.input, pl.adir.scalar]})


@pytest.mark.parametrize(
    ("Before", "Expected"),
    [(_SpmdWrapper, _SpmdWrapperExpected), (_GroupWrapper, _GroupWrapperExpected)],
    ids=["spmd", "group"],
)
def test_wrapper_calls_forward_materialized_comm_ctx_params(Before, Expected):
    """A manual Spmd / Group wrapper forwards the materialized ctx at every hop."""
    ir.assert_structural_equal(_derive_and_materialize(Before), Expected)


def test_materialized_comm_ctx_param_name_avoids_existing_param():
    """A param already named ``data_ctx`` pushes the materialized one to ``data_ctx_1``."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(self, data: pld.DistributedTensor[[4], pl.FP32], data_ctx: pl.Scalar[pl.INDEX]):
            return

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self, data: pld.DistributedTensor[[4], pl.FP32], data_ctx: pl.Scalar[pl.INDEX]):
            self.kernel(data, data_ctx)

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            data: pld.DistributedTensor[[4], pl.FP32],
            data_ctx: pl.Scalar[pl.INDEX],
            data_ctx_1: pld.CommCtx,
        ):
            return

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            data: pld.DistributedTensor[[4], pl.FP32],
            data_ctx: pl.Scalar[pl.INDEX],
            data_ctx_1: pld.CommCtx,
        ):
            self.kernel(
                data,
                data_ctx,
                data_ctx_1,
                attrs={"arg_directions": [pl.adir.input, pl.adir.scalar, pl.adir.scalar]},
            )

    ir.assert_structural_equal(_derive_and_materialize(Before), Expected)


def test_materialized_local_comm_ctx_name_avoids_existing_local():
    """A window allocated in the host body has no context to inherit, so host
    orchestration synthesizes the query — under a name that dodges the local
    ``data_ctx`` already bound above it."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def callee(self, data: pld.DistributedTensor[[4], pl.FP32]):
            return

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def main(self):
            data_ctx: pl.Scalar[pl.INDEX] = 0  # noqa: F841 - the name collision is the point
            buffer = pld.alloc_window_buffer(16)
            data = pld.window(buffer, [4], dtype=pl.FP32)
            self.callee(data)

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.InCore)
        def callee(self, data: pld.DistributedTensor[[4], pl.FP32], data_ctx: pld.CommCtx):
            return

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def main(self):
            data_ctx: pl.Scalar[pl.INDEX] = 0  # noqa: F841 - the name collision is the point
            buffer: pl.Ptr = pld.tensor.alloc_window_buffer(16)
            data: pld.DistributedTensor[[4], pl.FP32] = pld.tensor.window(buffer, [4], dtype=pl.FP32)
            data_ctx_1: pld.CommCtx = pld.system.get_comm_ctx(data)
            self.callee(data, data_ctx_1, attrs={"arg_directions": [pl.adir.input, pl.adir.scalar]})

    ir.assert_structural_equal(_derive_and_materialize(Before), Expected)


def test_materialized_comm_ctx_param_name_avoids_existing_local():
    """A *local* named ``data_ctx`` also pushes the materialized param to ``data_ctx_1``."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(self, data: pld.DistributedTensor[[4], pl.FP32]):
            data_ctx: pl.Scalar[pl.INDEX] = 0  # noqa: F841 - the name collision is the point
            return  # noqa: PLR1711 - DSL spelling of an empty body

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(self, data: pld.DistributedTensor[[4], pl.FP32], data_ctx_1: pld.CommCtx):
            data_ctx: pl.Scalar[pl.INDEX] = 0  # noqa: F841 - the name collision is the point
            return  # noqa: PLR1711 - DSL spelling of an empty body

    ir.assert_structural_equal(_derive_and_materialize(Before), Expected)


def test_param_alias_forwards_materialized_comm_ctx_param():
    """An alias of a DistributedTensor param inherits that param's context —
    the call forwards the materialized param, no local query is synthesized."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(self, data: pld.DistributedTensor[[4], pl.FP32]):
            return

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self, data: pld.DistributedTensor[[4], pl.FP32]):
            alias = data
            self.kernel(alias)

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(self, data: pld.DistributedTensor[[4], pl.FP32], data_ctx: pld.CommCtx):
            return

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self, data: pld.DistributedTensor[[4], pl.FP32], data_ctx: pld.CommCtx):
            alias: pld.DistributedTensor[[4], pl.FP32] = data
            self.kernel(alias, data_ctx, attrs={"arg_directions": [pl.adir.input, pl.adir.scalar]})

    ir.assert_structural_equal(_derive_and_materialize(Before), Expected)


def test_returned_mixed_values_use_reordered_distributed_param_contexts():
    """Return positions, rather than a tail heuristic, select each CommCtx."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def reorder(
            self,
            first: pl.InOut[pld.DistributedTensor[[4], pl.FP32]],
            marker: pl.InOut[pl.Tensor[[4], pl.INT32]],
            second: pl.InOut[pld.DistributedTensor[[4], pl.FP32]],
        ) -> tuple[
            pl.Tensor[[4], pl.INT32],
            pld.DistributedTensor[[4], pl.FP32],
            pld.DistributedTensor[[4], pl.FP32],
        ]:
            return marker, second, first

        @pl.function(type=pl.FunctionType.InCore)
        def consume(self, data: pld.DistributedTensor[[4], pl.FP32]):
            pld.system.wait(data, offsets=[0], expected=1, cmp=pld.WaitCmp.Eq)

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            first: pl.InOut[pld.DistributedTensor[[4], pl.FP32]],
            second: pl.InOut[pld.DistributedTensor[[4], pl.FP32]],
            marker: pl.InOut[pl.Tensor[[4], pl.INT32]],
        ):
            result = self.reorder(first, marker, second)
            returned_second = result[1]
            returned_first = result[2]
            self.consume(returned_second)
            self.consume(returned_first)

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
        def reorder(
            self,
            first: pl.InOut[pld.DistributedTensor[[4], pl.FP32]],
            marker: pl.InOut[pl.Tensor[[4], pl.INT32]],
            second: pl.InOut[pld.DistributedTensor[[4], pl.FP32]],
            first_ctx: pld.CommCtx,
            second_ctx: pld.CommCtx,
        ) -> tuple[
            pl.Tensor[[4], pl.INT32],
            pld.DistributedTensor[[4], pl.FP32],
            pld.DistributedTensor[[4], pl.FP32],
        ]:
            return marker, second, first

        @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
        def consume(self, data: pld.DistributedTensor[[4], pl.FP32], data_ctx: pld.CommCtx):
            pld.system.wait(data, offsets=[0], expected=1, cmp=pld.WaitCmp.Eq)

        @pl.function(type=pl.FunctionType.Orchestration, level=pl.Level.CHIP, role=pl.Role.Orchestrator)
        def main(
            self,
            first: pl.InOut[pld.DistributedTensor[[4], pl.FP32]],
            second: pl.InOut[pld.DistributedTensor[[4], pl.FP32]],
            marker: pl.InOut[pl.Tensor[[4], pl.INT32]],
            first_ctx: pld.CommCtx,
            second_ctx: pld.CommCtx,
        ):
            result = self.reorder(
                first,
                marker,
                second,
                first_ctx,
                second_ctx,
                attrs={
                    "arg_directions": [
                        pl.adir.inout,
                        pl.adir.inout,
                        pl.adir.inout,
                        pl.adir.scalar,
                        pl.adir.scalar,
                    ]
                },
            )
            returned_second = result[1]
            returned_first = result[2]
            # `reorder` returns (marker, second, first), so position 1 carries
            # `second`'s context and position 2 carries `first`'s — a tail
            # heuristic would swap them.
            self.consume(
                returned_second, second_ctx, attrs={"arg_directions": [pl.adir.input, pl.adir.scalar]}
            )
            self.consume(returned_first, first_ctx, attrs={"arg_directions": [pl.adir.input, pl.adir.scalar]})

    ir.assert_structural_equal(_apply(passes.convert_to_ssa()(Before)), Expected)


def test_loop_carried_returned_distributed_tensors_reuse_contexts():
    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def comm(
            self,
            data: pl.InOut[pld.DistributedTensor[[4], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[2], pl.INT32]],
        ) -> tuple[pld.DistributedTensor[[4], pl.FP32], pld.DistributedTensor[[2], pl.INT32]]:
            return data, signal

        @pl.function(type=pl.FunctionType.InCore)
        def compute(self, value: pl.InOut[pl.Tensor[[4], pl.FP32]]) -> pl.Tensor[[4], pl.FP32]:
            return value

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            data: pl.InOut[pld.DistributedTensor[[4], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[2], pl.INT32]],
            value: pl.InOut[pl.Tensor[[4], pl.FP32]],
        ) -> pl.Tensor[[4], pl.FP32]:
            for _layer in pl.range(2):
                data, signal = self.comm(data, signal)
                value = self.compute(value)
            data, signal = self.comm(data, signal)
            return value

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
        def comm(
            self,
            data: pl.InOut[pld.DistributedTensor[[4], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[2], pl.INT32]],
            data_ctx: pld.CommCtx,
            signal_ctx: pld.CommCtx,
        ) -> tuple[pld.DistributedTensor[[4], pl.FP32], pld.DistributedTensor[[2], pl.INT32]]:
            return data, signal

        @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
        def compute(self, value: pl.InOut[pl.Tensor[[4], pl.FP32]]) -> pl.Tensor[[4], pl.FP32]:
            return value

        @pl.function(type=pl.FunctionType.Orchestration, level=pl.Level.CHIP, role=pl.Role.Orchestrator)
        def main(
            self,
            data: pl.InOut[pld.DistributedTensor[[4], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[2], pl.INT32]],
            value: pl.InOut[pl.Tensor[[4], pl.FP32]],
            data_ctx: pld.CommCtx,
            signal_ctx: pld.CommCtx,
        ) -> pl.Tensor[[4], pl.FP32]:
            # Both call sites forward the caller's ctx params: the loop carry is
            # seeded from them and the yield keeps naming the same windows.
            for _layer, (data_i, signal_i, value_i) in pl.range(2, init_values=(data, signal, value)):
                carried = self.comm(
                    data_i,
                    signal_i,
                    data_ctx,
                    signal_ctx,
                    attrs={
                        "arg_directions": [
                            pl.adir.inout,
                            pl.adir.inout,
                            pl.adir.scalar,
                            pl.adir.scalar,
                        ]
                    },
                )
                data_next = carried[0]
                signal_next = carried[1]
                value_next = self.compute(value_i, attrs={"arg_directions": [pl.adir.inout]})
                data_out, signal_out, value_out = pl.yield_(data_next, signal_next, value_next)
            tail = self.comm(
                data_out,
                signal_out,
                data_ctx,
                signal_ctx,
                attrs={
                    "arg_directions": [
                        pl.adir.inout,
                        pl.adir.inout,
                        pl.adir.scalar,
                        pl.adir.scalar,
                    ]
                },
            )
            _tail_data = tail[0]
            _tail_signal = tail[1]
            return value_out

    ir.assert_structural_equal(_apply(passes.convert_to_ssa()(Before)), Expected)


def test_buffer_aliasing_view_inherits_source_context():
    """A zero-copy view keeps the source window, so it forwards the source ctx."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def consume(self, data: pld.DistributedTensor[[2, 2], pl.FP32]):
            pld.system.wait(data, offsets=[0, 0], expected=1, cmp=pld.WaitCmp.Eq)

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self, data: pl.InOut[pld.DistributedTensor[[4], pl.FP32]]):
            view = pl.tensor.view(data, [2, 2])
            self.consume(view)

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
        def consume(self, data: pld.DistributedTensor[[2, 2], pl.FP32], data_ctx: pld.CommCtx):
            pld.system.wait(data, offsets=[0, 0], expected=1, cmp=pld.WaitCmp.Eq)

        @pl.function(type=pl.FunctionType.Orchestration, level=pl.Level.CHIP, role=pl.Role.Orchestrator)
        def main(
            self,
            data: pl.InOut[pld.DistributedTensor[[4], pl.FP32]],
            data_ctx: pld.CommCtx,
        ):
            view = pl.tensor.view(data, [2, 2])
            # `view` is a fresh Var, but it names `data`'s window, so the call
            # forwards `data_ctx` rather than querying a context for it.
            self.consume(view, data_ctx, attrs={"arg_directions": [pl.adir.input, pl.adir.scalar]})

    ir.assert_structural_equal(_apply(Before), Expected)


def test_loop_rebinding_carry_to_another_distributed_tensor_is_rejected():
    """A carry seeded from its init must still yield back the same context."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.InCore)
        def pick(self, data: pld.DistributedTensor[[4], pl.FP32]) -> pld.DistributedTensor[[4], pl.FP32]:
            return data

        @pl.function(type=pl.FunctionType.InCore)
        def consume(self, data: pld.DistributedTensor[[4], pl.FP32]):
            pld.system.wait(data, offsets=[0], expected=1, cmp=pld.WaitCmp.Eq)

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            first: pl.InOut[pld.DistributedTensor[[4], pl.FP32]],
            second: pl.InOut[pld.DistributedTensor[[4], pl.FP32]],
        ):
            data = first
            for _step in pl.range(2):
                self.consume(data)
                # Rebinds the carry to `second`: it would enter the loop with
                # `first`'s context and leave with `second`'s.
                data = self.pick(second)
            self.consume(data)

    with pytest.raises(ValueError, match="Rebinding loop-carried DistributedTensor"):
        _apply(passes.convert_to_ssa()(P))


def test_submit_result_forwards_returned_param_ctx():
    """A Submit result position maps back to the arg it writes back, like Call."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def stage(
            self, data: pl.InOut[pld.DistributedTensor[[4], pl.FP32]]
        ) -> pld.DistributedTensor[[4], pl.FP32]:
            return data

        @pl.function(type=pl.FunctionType.InCore)
        def consume(self, data: pld.DistributedTensor[[4], pl.FP32]):
            pld.system.wait(data, offsets=[0], expected=1, cmp=pld.WaitCmp.Eq)

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self, data: pl.InOut[pld.DistributedTensor[[4], pl.FP32]]):
            produced, _tid = pl.submit(self.stage, data)
            self.consume(produced)

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
        def stage(
            self,
            data: pl.InOut[pld.DistributedTensor[[4], pl.FP32]],
            data_ctx: pld.CommCtx,
        ) -> pld.DistributedTensor[[4], pl.FP32]:
            return data

        @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
        def consume(self, data: pld.DistributedTensor[[4], pl.FP32], data_ctx: pld.CommCtx):
            pld.system.wait(data, offsets=[0], expected=1, cmp=pld.WaitCmp.Eq)

        @pl.function(type=pl.FunctionType.Orchestration, level=pl.Level.CHIP, role=pl.Role.Orchestrator)
        def main(
            self,
            data: pl.InOut[pld.DistributedTensor[[4], pl.FP32]],
            data_ctx: pld.CommCtx,
        ):
            # The submit gets its ctx arg like any Call, and `produced` — result
            # position 0, ahead of the trailing TASK_ID — resolves back through
            # `stage`'s return to `data`, so `consume` reuses the same ctx.
            produced, _tid = pl.submit(
                self.stage, data, data_ctx, attrs={"arg_directions": [pl.adir.inout, pl.adir.scalar]}
            )
            self.consume(produced, data_ctx, attrs={"arg_directions": [pl.adir.input, pl.adir.scalar]})

    ir.assert_structural_equal(_apply(Before), Expected)


def test_distributed_if_keeps_current_no_context_yield_behavior():
    """The `if` is left alone: no CommCtxType is added to its return vars."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            data: pld.DistributedTensor[[4], pl.FP32],
            cond: pl.Scalar[pl.BOOL],
        ):
            result = data
            if cond:
                result = data
            ctx = pld.system.get_comm_ctx(result)
            _rank = pld.system.rank(ctx)

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
        def kernel(
            self,
            data: pld.DistributedTensor[[4], pl.FP32],
            cond: pl.Scalar[pl.BOOL],
            data_ctx: pld.CommCtx,
        ):
            # `_result` keeps the `if` return var the Before program has; the
            # pass adds no CommCtxType yield alongside it, and the query below
            # resolves straight to the materialized parameter. Underscored
            # because nothing reads it once `get_comm_ctx` is gone.
            _result = data
            if cond:
                _result = data
            ctx = data_ctx
            _rank = pld.system.rank(ctx)

    ir.assert_structural_equal(_apply(Before), Expected)


def test_device_get_comm_ctx_is_replaced_by_materialized_context():
    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(self, data: pld.DistributedTensor[[4], pl.FP32]):
            ctx = pld.system.get_comm_ctx(data)
            _rank = pld.system.rank(ctx)
            pld.system.wait(data, offsets=[0], expected=1, cmp=pld.WaitCmp.Eq)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host(self):
            buffer = pld.alloc_window_buffer(4 * pl.FP32.get_byte())
            data = pld.window(buffer, [4], dtype=pl.FP32)
            self.chip_orch(data, device=0)

    @pl.program
    class ExpectedChipOrch:
        @pl.function(type=pl.FunctionType.Orchestration, level=pl.Level.CHIP, role=pl.Role.Orchestrator)
        def chip_orch(self, data: pld.DistributedTensor[[4], pl.FP32], data_ctx: pld.CommCtx):
            # The query is gone: `ctx` is now a plain alias of the materialized
            # parameter, which the final Simplify folds away.
            ctx = data_ctx
            _rank = pld.system.rank(ctx)
            pld.system.wait(data, offsets=[0], expected=1, cmp=pld.WaitCmp.Eq)

    result = _apply(Before)
    chip = _get_func(result, "chip_orch")
    expected_chip = _get_func(ExpectedChipOrch, "chip_orch")
    old_data = expected_chip.params[0]
    old_type = old_data.type
    window = chip.params[0].type.window_buffer
    assert window is not None
    expected_data = ir.Var(
        old_data.name_hint,
        ir.DistributedTensorType(
            list(old_type.shape), old_type.dtype, old_type.memref, old_type.tensor_view, window
        ),
        old_data.span,
    )

    class BindExpectedWindow(ir.IRMutator):
        def visit_var(self, var):
            return expected_data if var is old_data else super().visit_var(var)

    # C2a now preserves the owner's window on the CHIP formal and every body
    # use. Keep the exact ctx-query/body structural expectation with that one
    # additional type field, rather than dropping the structural comparison.
    expected_chip = BindExpectedWindow().visit_function(expected_chip)
    ir.assert_structural_equal(chip, expected_chip)

    # `host` is compared by assertion rather than structurally: its body is
    # wrapped in a CommDomainScopeStmt by MaterializeCommDomainScopes, and that
    # node has no DSL surface to spell out in an Expected program.
    host_queries = _collect_calls(_get_func(result, "host").body, "pld.system.get_comm_ctx")
    assert len(host_queries) == 1, "host orchestration must keep the runtime query"


# ---------------------------------------------------------------------------
# The one Before below is hand-built rather than DSL-authored, and deliberately
# so: the program under test is one the DSL frontend refuses to author. It
# allocates a window inside a *chip* orchestration function, and the parser
# rejects ``pld.alloc_window_buffer`` outside HOST/CLUSTER_0 orchestration with
# its own diagnostic. Native verification now rejects the same invalid IR before
# the pass; disabling instrumentation separately pins the materializer's error.
# Constructing that unsupported shape therefore requires raw ``ir.*``.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("instrumented", [False, True])
def test_device_call_without_materialized_context_rejects_synthesized_prefix(instrumented):
    span = ir.Span("test_materialize_dist_tensor_ctx.py", 1, 1)
    data_ty = ir.DistributedTensorType([4], pl.FP32)

    # Returns nothing: a task cannot return a scalar (#631), and this fixture is
    # about the missing communication context, not about the return value.
    predicate_data = ir.Var("data", data_ty, span)
    predicate = ir.Function(
        "predicate",
        [(predicate_data, ir.ParamDirection.In)],
        [],
        ir.ReturnStmt([], span),
        span,
        ir.FunctionType.InCore,
    )

    # `local_data` is a window created inside a chip-orchestration function, so
    # it is a genuine new DistributedTensor rather than a view of a parameter —
    # there is no context for it to inherit, and outside host orchestration the
    # pass cannot query one either.
    buffer = ir.Var("buffer", ir.PtrType.get(), span)
    local_data = ir.Var("local_data", data_ty, span)
    alloc = ir.create_op_call(
        "pld.tensor.alloc_window_buffer", [ir.ConstInt(16, DataType.INDEX, span)], {"name": "buf"}, span
    )
    window = ir.create_op_call(
        "pld.tensor.window",
        [buffer, ir.MakeTuple([ir.ConstInt(4, DataType.INDEX, span)], span)],
        {"dtype": pl.FP32},
        span,
    )
    predicate_call = ir.Call(
        ir.GlobalVar("predicate"),
        [local_data],
        {},
        {"arg_directions": [ir.ArgDirection.Input]},
        ir.TupleType([]),
        span,
    )
    main = ir.Function(
        "main",
        [],
        [],
        ir.SeqStmts(
            [
                ir.AssignStmt(buffer, alloc, span),
                ir.AssignStmt(local_data, window, span),
                ir.EvalStmt(predicate_call, span),
            ],
            span,
        ),
        span,
        ir.FunctionType.Orchestration,
    )

    program = ir.Program([predicate, main], "unsupported_prefix_context", span)
    instruments = (
        [passes.VerificationInstrument(passes.VerificationMode.BEFORE_AND_AFTER)] if instrumented else []
    )
    error_type = pypto.Error if instrumented else ValueError
    match = "Pre-verification failed" if instrumented else "Cannot determine the communication context"
    with passes.PassContext(instruments, verification_level=passes.VerificationLevel.NONE):
        with pytest.raises(error_type, match=match) as excinfo:
            passes.materialize_dist_tensor_ctx()(program)
    if instrumented:
        assert "DistributedOperationContext" in str(excinfo.value)
        assert "pld.tensor.alloc_window_buffer" in str(excinfo.value)


def test_submit_prefix_runtime_out_keeps_ctx_after_passed_args():
    """The materialized ctx param lands *after* the runtime-allocated ``Out``
    param, and the Submit forwards it right after the caller-supplied args."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def stage(
            self,
            data: pld.DistributedTensor[[4], pl.FP32],
            scratch: pl.Out[pl.Tensor[[4], pl.FP32]],
        ) -> pl.Tensor[[4], pl.FP32]:
            return scratch

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self, data: pld.DistributedTensor[[4], pl.FP32]):
            with pl.manual_scope():
                out, _tid = pl.submit(self.stage, data)
            return out

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.InCore)
        def stage(
            self,
            data: pld.DistributedTensor[[4], pl.FP32],
            scratch: pl.Out[pl.Tensor[[4], pl.FP32]],
            data_ctx: pld.CommCtx,
        ) -> pl.Tensor[[4], pl.FP32]:
            return scratch

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self, data: pld.DistributedTensor[[4], pl.FP32], data_ctx: pld.CommCtx):
            with pl.manual_scope():
                out, _tid = pl.submit(
                    self.stage,
                    data,
                    data_ctx,
                    attrs={"arg_directions": [pl.adir.input, pl.adir.scalar]},
                )
            return out

    ir.assert_structural_equal(_derive_and_materialize(Before), Expected)


def test_return_call_reuses_returned_param_ctx():
    """A call in return position reuses the context of the param whose value the
    producer returned, rather than synthesizing a second query."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def producer(
            self, source_data: pld.DistributedTensor[[4], pl.FP32]
        ) -> pld.DistributedTensor[[4], pl.FP32]:
            return source_data

        @pl.function(type=pl.FunctionType.InCore)
        def callee(self, data: pld.DistributedTensor[[4], pl.FP32]) -> pld.DistributedTensor[[4], pl.FP32]:
            return data

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self, source_data: pld.DistributedTensor[[4], pl.FP32]
        ) -> pld.DistributedTensor[[4], pl.FP32]:
            data = self.producer(source_data)
            return self.callee(data)

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.InCore)
        def producer(
            self, source_data: pld.DistributedTensor[[4], pl.FP32], source_data_ctx: pld.CommCtx
        ) -> pld.DistributedTensor[[4], pl.FP32]:
            return source_data

        @pl.function(type=pl.FunctionType.InCore)
        def callee(
            self, data: pld.DistributedTensor[[4], pl.FP32], data_ctx: pld.CommCtx
        ) -> pld.DistributedTensor[[4], pl.FP32]:
            return data

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self, source_data: pld.DistributedTensor[[4], pl.FP32], source_data_ctx: pld.CommCtx
        ) -> pld.DistributedTensor[[4], pl.FP32]:
            data: pld.DistributedTensor[[4], pl.FP32] = self.producer(
                source_data, source_data_ctx, attrs={"arg_directions": [pl.adir.input, pl.adir.scalar]}
            )
            return self.callee(
                data, source_data_ctx, attrs={"arg_directions": [pl.adir.input, pl.adir.scalar]}
            )

    ir.assert_structural_equal(_derive_and_materialize(Before), Expected)


@pytest.mark.parametrize("kind", ["call", "submit"])
def test_c2b_supplied_distributed_out_return_uses_output_ctx(kind):
    """Supplied Out has caller provenance distinct from the input's context."""
    span = _span()
    memref = ir.MemRef("output_storage", 0, 16, span)
    dt = ir.DistributedTensorType([4], pl.FP32, memref, ir.TensorView(stride=[1], layout=ir.TensorLayout.ND))
    source, output = ir.Var("source", _dist_ty(), span), ir.Var("output", dt, span)
    stage = ir.Function(
        "stage",
        [(source, ir.ParamDirection.In), (output, ir.ParamDirection.Out)],
        [source.type, dt],
        ir.ReturnStmt([source, output], span),
        span,
        type=ir.FunctionType.InCore,
    )
    invocation = _c2b_invocation(kind, [source, output], [source.type, dt], span)
    attrs = {
        **invocation.attrs,
        "arg_directions": [
            ir.ArgDirection.Input,
            ir.ArgDirection.OutputExisting,
        ],
    }
    if kind == "call":
        invocation = ir.Call(invocation.op, invocation.args, invocation.kwargs, attrs, invocation.type, span)
    else:
        invocation = ir.Submit(
            invocation.op,
            invocation.args,
            [],
            invocation.kwargs,
            attrs,
            invocation.type,
            span,
            core_num=invocation.core_num,
            sync_start=True,
            allow_early_resolve=True,
            predicate=invocation.predicate,
        )
    packed = ir.Var("packed", invocation.type, span)
    result = ir.Var("result", dt, span)
    ctx = ir.Var("ctx", ir.CommCtxType(), span)
    body = ir.SeqStmts(
        [
            ir.AssignStmt(packed, invocation, span),
            ir.AssignStmt(result, ir.TupleGetItemExpr(packed, 1, span), span),
            ir.AssignStmt(ctx, ir.create_op_call("pld.system.get_comm_ctx", [result], {}, span), span),
            ir.ReturnStmt([], span),
        ],
        span,
    )
    if kind == "submit":
        body = ir.RuntimeScopeStmt(manual=True, body=body, span=span)
    caller = ir.Function(
        "caller",
        [source, output],
        [],
        body,
        span,
        type=ir.FunctionType.Orchestration,
    )
    after = passes.materialize_dist_tensor_ctx()(ir.Program([stage, caller], "SuppliedDTOut", span))
    caller, stage = _get_func(after, "caller"), _get_func(after, "stage")
    body = caller.body.body if kind == "submit" else caller.body
    launch = body.stmts[0].value
    assert list(launch.args[-2:]) == list(caller.params[-2:])
    assert body.stmts[2].value is caller.params[-1]
    assert body.stmts[2].value is not caller.params[-2]
    assert stage.params[1].type.memref is memref
    ir.assert_structural_equal(stage.params[1].type, dt)
    assert body.stmts[1].var.type is dt
    assert body.stmts[1].value.type is dt


def _c2b_invocation(kind, args, returns, span, dependency=None):
    """Keep launch metadata visible while testing only the ctx ABI mutation."""
    kwargs = {"saved_kwarg": 7}
    attrs = {
        "saved_attr": "kept",
        "arg_directions": [
            ir.ArgDirection.Scalar if isinstance(arg.type, ir.CommCtxType) else ir.ArgDirection.Input
            for arg in args
        ],
    }
    if kind == "submit":
        return ir.Submit(
            ir.GlobalVar("stage"),
            args,
            [dependency] if dependency is not None else [],
            kwargs,
            attrs,
            ir.TupleType([*returns, ir.ScalarType(DataType.TASK_ID)]),
            span,
            core_num=ir.ConstInt(2, DataType.INDEX, span),
            sync_start=True,
            allow_early_resolve=True,
            predicate=ir.ConstBool(True, span),
        )
    return_type = returns[0] if len(returns) == 1 else ir.TupleType(returns)
    return ir.Call(ir.GlobalVar("stage"), args, kwargs, attrs, return_type, span)


@pytest.mark.parametrize("supplied_out", [False, True])
def test_c2b_submit_ordinary_out_gap_before_existing_ctx_suffix(supplied_out):
    """Only an Out tail before the ctx suffix may be omitted from Submit args."""
    span = _span()
    dt = _dist_ty()
    ordinary = ir.TensorType([4], pl.FP32)
    data, output = ir.Var("data", dt, span), ir.Var("output", ordinary, span)
    existing = ir.Var("existing", ir.CommCtxType(), span)
    stage = ir.Function(
        "stage",
        [(data, ir.ParamDirection.In), (output, ir.ParamDirection.Out), existing],
        [ordinary],
        ir.ReturnStmt([output], span),
        span,
        type=ir.FunctionType.InCore,
    )
    source, target = ir.Var("source", dt, span), ir.Var("target", ordinary, span)
    dep = ir.Var("dep", ir.ScalarType(DataType.TASK_ID), span)
    args = [source, *([target] if supplied_out else []), existing]
    invocation = _c2b_invocation("submit", args, [ordinary], span, dep)
    if supplied_out:
        invocation = ir.Submit(
            invocation.op,
            invocation.args,
            invocation.deps,
            invocation.kwargs,
            {
                **invocation.attrs,
                "arg_directions": [
                    ir.ArgDirection.Input,
                    ir.ArgDirection.OutputExisting,
                    ir.ArgDirection.Scalar,
                ],
            },
            invocation.type,
            span,
            core_num=invocation.core_num,
            sync_start=True,
            allow_early_resolve=True,
            predicate=invocation.predicate,
        )
    packed = ir.Var("packed", invocation.type, span)
    caller = ir.Function(
        "caller",
        [source, target, existing, dep],
        [],
        ir.RuntimeScopeStmt(
            manual=True,
            body=ir.SeqStmts(
                [
                    ir.AssignStmt(packed, invocation, span),
                    ir.ReturnStmt([], span),
                ],
                span,
            ),
            span=span,
        ),
        span,
        type=ir.FunctionType.Orchestration,
    )
    after = passes.materialize_dist_tensor_ctx()(ir.Program([caller, stage], "OutGap", span))
    caller, stage = _get_func(after, "caller"), _get_func(after, "stage")
    launch = caller.body.body.stmts[0].value
    assert isinstance(launch, ir.Submit)
    assert list(launch.args[:-1]) == args
    assert launch.args[-1] is caller.params[-1]
    assert list(stage.param_directions) == [
        ir.ParamDirection.In,
        ir.ParamDirection.Out,
        ir.ParamDirection.In,
        ir.ParamDirection.In,
    ]
    assert len(stage.params) == 4
    assert list(launch.arg_directions) == [*invocation.arg_directions, ir.ArgDirection.Scalar]
    assert launch.kwargs["saved_kwarg"] == 7 and launch.attrs["saved_attr"] == "kept"
    assert launch.deps[0] is dep
    assert launch.core_num.value == 2
    assert launch.sync_start and launch.allow_early_resolve and launch.predicate.value
    assert launch.type.types[0] is ordinary
    assert launch.type.types[-1].dtype == DataType.TASK_ID


@pytest.mark.parametrize("with_existing_ctx", [False, True])
def test_c2b_omitted_distributed_out_has_no_caller_ctx(with_existing_ctx):
    """A suffix CommCtx is never provenance for an omitted distributed Out."""
    span = _span()
    data, output = ir.Var("data", _dist_ty(), span), ir.Var("output", _dist_ty(), span)
    existing = ir.Var("existing", ir.CommCtxType(), span)
    stage = ir.Function(
        "stage",
        [
            (data, ir.ParamDirection.In),
            (output, ir.ParamDirection.Out),
            *([existing] if with_existing_ctx else []),
        ],
        [output.type],
        ir.ReturnStmt([output], span),
        span,
        type=ir.FunctionType.InCore,
    )
    invocation = _c2b_invocation(
        "submit",
        [data, *([existing] if with_existing_ctx else [])],
        [output.type],
        span,
    )
    caller = ir.Function(
        "caller",
        [data, *([existing] if with_existing_ctx else [])],
        [],
        ir.RuntimeScopeStmt(manual=True, body=ir.EvalStmt(invocation, span), span=span),
        span,
        type=ir.FunctionType.Orchestration,
    )
    # This is valid Submit ABI, but no runtime-allocation ctx contract exists.
    # Pin a user-facing limitation rather than an INTERNAL_CHECK or bogus arg.
    with pytest.raises(ValueError, match="(?i)(omitted|runtime.allocated).*([Cc]omm|context|[Dd]istributed)"):
        passes.materialize_dist_tensor_ctx()(ir.Program([stage, caller], "OmittedDTOut", span))


@pytest.mark.parametrize("kind", ["call", "submit"])
@pytest.mark.parametrize("result_shape", ["no_dt", "sparse"])
def test_c2b_result_ctx_preserves_scalar_gaps_and_task_id(kind, result_shape):
    """Result positions stay stable when only sparse results carry a ctx."""
    span = _span()
    data = ir.Var("data", _dist_ty(), span)
    marker = ir.Var("marker", ir.ScalarType(DataType.INDEX), span)
    values = [marker, marker] if result_shape == "no_dt" else [marker, data, marker, data]
    returns = [value.type for value in values]
    stage = ir.Function(
        "stage",
        [data, marker],
        returns,
        ir.ReturnStmt(values, span),
        span,
        level=ir.Level.CHIP,
        role=ir.Role.Orchestrator,
    )
    invocation = _c2b_invocation(kind, [data, marker], returns, span)
    # Scalar arguments keep their direction even when ctx scalar args append.
    attrs = {**invocation.attrs, "arg_directions": [ir.ArgDirection.Input, ir.ArgDirection.Scalar]}
    if kind == "call":
        invocation = ir.Call(invocation.op, invocation.args, invocation.kwargs, attrs, invocation.type, span)
    else:
        invocation = ir.Submit(
            invocation.op,
            invocation.args,
            [],
            invocation.kwargs,
            attrs,
            invocation.type,
            span,
            core_num=invocation.core_num,
            sync_start=True,
            allow_early_resolve=True,
            predicate=invocation.predicate,
        )
    packed = ir.Var("packed", invocation.type, span)
    stmts = [ir.AssignStmt(packed, invocation, span)]
    if result_shape == "sparse":
        for index in (1, 3):
            projected = ir.Var(f"result_{index}", data.type, span)
            ctx = ir.Var(f"ctx_{index}", ir.CommCtxType(), span)
            stmts.extend(
                [
                    ir.AssignStmt(projected, ir.TupleGetItemExpr(packed, index, span), span),
                    ir.AssignStmt(
                        ctx, ir.create_op_call("pld.system.get_comm_ctx", [projected], {}, span), span
                    ),
                ]
            )
    stmts.append(ir.ReturnStmt([], span))
    body = ir.SeqStmts(stmts, span)
    if kind == "submit":
        body = ir.RuntimeScopeStmt(manual=True, body=body, span=span)
    caller = ir.Function(
        "caller",
        [data, marker],
        [],
        body,
        span,
        type=ir.FunctionType.Orchestration,
    )
    after = passes.materialize_dist_tensor_ctx()(ir.Program([caller, stage], "SparseResults", span))
    caller = _get_func(after, "caller")
    rewritten_body = caller.body.body if kind == "submit" else caller.body
    launch = rewritten_body.stmts[0].value
    assert isinstance(launch, ir.Submit if kind == "submit" else ir.Call)
    assert launch.type is invocation.type
    assert list(launch.arg_directions) == [
        ir.ArgDirection.Input,
        ir.ArgDirection.Scalar,
        ir.ArgDirection.Scalar,
    ]
    assert launch.args[-1] is caller.params[-1]
    assert len(launch.type.types) == len(returns) + (kind == "submit")
    if kind == "submit":
        assert launch.type.types[-1].dtype == DataType.TASK_ID
    if result_shape == "sparse":
        for offset in (2, 4):
            assert rewritten_body.stmts[offset].value is caller.params[-1]


@pytest.mark.parametrize("payload_kind", ["call", "submit", "query", "query_tuple"])
def test_c2b_function_expr_attrs_rewrite_ctx_abi(payload_kind):
    """Expr-valued function attrs must retain the updated ctx ABI."""
    span = _span()
    data = ir.Var("data", _dist_ty(), span)
    stage = ir.Function(
        "stage",
        [data],
        [data.type],
        ir.ReturnStmt([data], span),
        span,
        type=ir.FunctionType.InCore,
    )
    query = ir.create_op_call("pld.system.get_comm_ctx", [data], {}, span)
    payload = (
        query
        if payload_kind.startswith("query")
        else _c2b_invocation(
            payload_kind,
            [data],
            [data.type],
            span,
        )
    )
    caller = ir.Function(
        "caller",
        [data],
        [],
        ir.ReturnStmt([], span),
        span,
        type=ir.FunctionType.Orchestration,
        attrs={
            "saved": ir.MakeTuple([payload], span) if payload_kind == "query_tuple" else payload,
            "ordinary": "kept",
        },
    )
    after = passes.materialize_dist_tensor_ctx()(ir.Program([stage, caller], "FunctionAttrs", span))
    caller = _get_func(after, "caller")
    rewritten = caller.attrs["saved"]
    assert caller.attrs["ordinary"] == "kept"
    if payload_kind == "query_tuple":
        assert isinstance(rewritten, ir.MakeTuple)
        assert len(rewritten.elements) == 1
        rewritten = rewritten.elements[0]
    if payload_kind.startswith("query"):
        assert rewritten is caller.params[-1]
    else:
        assert isinstance(rewritten, ir.Submit if payload_kind == "submit" else ir.Call)
        assert rewritten.args[0] is caller.params[0]
        assert rewritten.args[-1] is caller.params[-1]
        assert len(rewritten.args) == 2
        assert list(rewritten.arg_directions) == [ir.ArgDirection.Input, ir.ArgDirection.Scalar]
        assert rewritten.kwargs["saved_kwarg"] == 7
        assert rewritten.attrs["saved_attr"] == "kept"
        assert rewritten.type is payload.type


def test_c2b_function_attr_requiring_ctx_prefix_reports_unsupported_context():
    """Attrs have no statement insertion site for a synthesized runtime query."""
    span = _span()
    data = ir.Var("data", _dist_ty(), span)
    stage = ir.Function(
        "stage",
        [data],
        [data.type],
        ir.ReturnStmt([data], span),
        span,
        type=ir.FunctionType.InCore,
    )
    buffer = ir.Var("buffer", ir.PtrType(), span)
    window = ir.create_op_call(
        "pld.tensor.window",
        [buffer, ir.MakeTuple([ir.ConstInt(4, DataType.INDEX, span)], span)],
        {"dtype": pl.FP32},
        span,
    )
    caller = ir.Function(
        "caller",
        [buffer],
        [],
        ir.ReturnStmt([], span),
        span,
        level=ir.Level.HOST,
        role=ir.Role.Orchestrator,
        attrs={"saved": _c2b_invocation("call", [window], [data.type], span)},
    )
    with pytest.raises(ValueError, match="(?i)(attribute|expression context|prefix)"):
        passes.materialize_dist_tensor_ctx()(ir.Program([caller, stage], "AttrPrefix", span))


@pytest.mark.parametrize("kind", ["call", "submit"])
def test_c2b_nested_result_ctx_uses_materialized_signature(kind):
    """Result lookup sees nested ctx suffixes after the base mutator visits args."""
    span = _span()
    data = ir.Var("data", _dist_ty(), span)
    scratch = ir.Var("scratch", ir.TensorType([4], pl.FP32), span)
    existing = ir.Var("existing", ir.CommCtxType(), span)
    stage = ir.Function(
        "stage",
        [(data, ir.ParamDirection.In), (scratch, ir.ParamDirection.Out), existing],
        [data.type, scratch.type],
        ir.ReturnStmt([data, scratch], span),
        span,
        type=ir.FunctionType.InCore,
    )
    consume = ir.Function(
        "consume",
        [data],
        [],
        ir.ReturnStmt([], span),
        span,
        type=ir.FunctionType.InCore,
    )
    args = [data, *([scratch] if kind == "call" else []), existing]
    nested = _c2b_invocation(kind, args, [data.type, scratch.type], span)
    if kind == "call":
        nested = ir.Call(
            nested.op,
            nested.args,
            nested.kwargs,
            {
                **nested.attrs,
                "arg_directions": [
                    ir.ArgDirection.Input,
                    ir.ArgDirection.OutputExisting,
                    ir.ArgDirection.Scalar,
                ],
            },
            nested.type,
            span,
        )
    outer = ir.Call(
        ir.GlobalVar("consume"),
        [ir.TupleGetItemExpr(nested, 0, span)],
        {},
        {"arg_directions": [ir.ArgDirection.Input]},
        ir.TupleType([]),
        span,
    )
    body = ir.EvalStmt(outer, span)
    if kind == "submit":
        body = ir.RuntimeScopeStmt(manual=True, body=body, span=span)
    caller = ir.Function(
        "caller",
        [data, scratch, existing],
        [],
        body,
        span,
        type=ir.FunctionType.Orchestration,
    )
    after = passes.materialize_dist_tensor_ctx()(ir.Program([stage, caller, consume], "NestedCtx", span))
    caller = _get_func(after, "caller")
    body = caller.body.body if kind == "submit" else caller.body
    outer = body.expr
    rewritten = outer.args[0].tuple
    assert isinstance(rewritten, ir.Submit if kind == "submit" else ir.Call)
    assert list(rewritten.args[:-1]) == args
    assert rewritten.args[-1] is outer.args[-1] is caller.params[-1]
    assert list(outer.arg_directions) == [ir.ArgDirection.Input, ir.ArgDirection.Scalar]
    assert rewritten.type is nested.type


@pytest.mark.parametrize("kind", ["call", "submit"])
@pytest.mark.parametrize("layout", ["empty", "all_scalar", "mixed", "repeated_dt"])
def test_c2b_wide_shared_result_type_and_tuple_aliases_keep_ctx_positions(kind, layout):
    """Result width and alias-chain length do not change parameter provenance.

    The same immutable result type is shared by several invocations. Repeated
    DT positions all return the second input, so confusing return position with
    input position or capturing the first input's ctx cannot satisfy this case.
    Scalar returns belong to CHIP orchestration helpers; HOST dispatches to
    those helpers and forwards the selected contexts to a second CHIP helper.
    """
    span = _span()
    width = 64
    first, second = ir.Var("first", _dist_ty(), span), ir.Var("second", _dist_ty(), span)
    marker = ir.Var("marker", ir.ScalarType(DataType.INDEX), span)
    existing = ir.Var("existing", ir.CommCtxType(), span)
    dependency = ir.Var("dependency", ir.ScalarType(DataType.TASK_ID), span)
    values = [] if layout == "empty" else [marker] * width
    query_positions = []
    if layout == "mixed":
        values[1], values[-2] = first, second
        query_positions = [(1, -2), (width - 2, -1)]
    elif layout == "repeated_dt":
        values = [second] * width
        query_positions = [(0, -1), (width - 1, -1)]
    returns = [value.type for value in values]
    stage = ir.Function(
        "stage",
        [first, second, marker, existing],
        returns,
        ir.ReturnStmt(values, span),
        span,
        type=ir.FunctionType.Orchestration,
    )
    consumed = ir.Var("consumed", _dist_ty(), span)
    consumed_ctx = ir.Var("consumed_ctx", ir.CommCtxType(), span)
    consume = ir.Function(
        "consume",
        [consumed],
        [],
        ir.SeqStmts(
            [
                ir.AssignStmt(
                    consumed_ctx,
                    ir.create_op_call("pld.system.get_comm_ctx", [consumed], {}, span),
                    span,
                ),
                ir.ReturnStmt([], span),
            ],
            span,
        ),
        span,
        type=ir.FunctionType.Orchestration,
    )
    result_type = ir.TupleType([*returns, *([dependency.type] if kind == "submit" else [])])
    statements = []
    expected_contexts = {}
    directions = [
        ir.ArgDirection.Input,
        ir.ArgDirection.Input,
        ir.ArgDirection.Scalar,
        ir.ArgDirection.Scalar,
    ]
    for site in range(3):
        args = [first, second, marker, existing]
        attrs = {
            "arg_directions": directions,
            "ordinary": "kept",
            "worker": ir.ConstInt(0, DataType.INDEX, span),
        }
        if kind == "submit":
            invocation = ir.Submit(
                ir.GlobalVar("stage"),
                args,
                [dependency],
                {},
                attrs,
                result_type,
                span,
            )
        else:
            invocation = ir.Call(ir.GlobalVar("stage"), args, {}, attrs, result_type, span)
        packed = ir.Var(f"packed_{site}", result_type, span)
        statements.append(ir.AssignStmt(packed, invocation, span))
        alias = packed
        for depth in range(6):
            fresh = ir.Var(f"alias_{site}_{depth}", result_type, span)
            statements.append(ir.AssignStmt(fresh, alias, span))
            alias = fresh
        for position, ctx_param in query_positions:
            projected = ir.Var(f"result_{site}_{position}", values[position].type, span)
            statements.extend(
                [
                    ir.AssignStmt(projected, ir.TupleGetItemExpr(alias, position, span), span),
                    ir.EvalStmt(
                        ir.Call(
                            ir.GlobalVar("consume"),
                            [projected],
                            {},
                            {
                                "arg_directions": [ir.ArgDirection.Input],
                                "worker": ir.ConstInt(0, DataType.INDEX, span),
                            },
                            ir.TupleType([]),
                            span,
                        ),
                        span,
                    ),
                ]
            )
            expected_contexts[projected.name_hint] = ctx_param
    statements.append(ir.ReturnStmt([], span))
    body = ir.SeqStmts(statements, span)
    if kind == "submit":
        body = ir.RuntimeScopeStmt(manual=True, body=body, span=span)
    caller = ir.Function(
        "caller",
        [first, second, marker, existing, dependency],
        [],
        body,
        span,
        level=ir.Level.HOST,
        role=ir.Role.Orchestrator,
    )
    after = passes.materialize_dist_tensor_ctx()(ir.Program([stage, caller, consume], "SharedResults", span))
    caller = _get_func(after, "caller")
    assignments = _collect_assign_stmts(caller.body)
    launches = [assign.value for assign in assignments if assign.var.name_hint.startswith("packed_")]
    assert len(launches) == 3
    for launch in launches:
        assert isinstance(launch, ir.Submit if kind == "submit" else ir.Call)
        assert launch.type is result_type
        assert list(launch.args[-2:]) == list(caller.params[-2:])
        assert list(launch.arg_directions) == [*directions, ir.ArgDirection.Scalar, ir.ArgDirection.Scalar]
        assert launch.attrs["ordinary"] == "kept"
        if kind == "submit":
            assert launch.deps[0] is dependency
            assert launch.type.types[-1].dtype == DataType.TASK_ID
    consumers = _collect_calls(caller.body, "consume")
    assert len(consumers) == len(expected_contexts)
    for call in consumers:
        assert call.args[-1] is caller.params[expected_contexts[call.args[0].name_hint]]
        assert call.attrs["worker"].value == 0
    consume = _get_func(after, "consume")
    assert consume.body.stmts[0].value is consume.params[-1]
    aliases = [assign for assign in assignments if assign.var.name_hint.startswith("alias_")]
    assert len(aliases) == 18
    assert all(assign.var.type is assign.value.type is result_type for assign in aliases)


@pytest.mark.parametrize("kind", ["call", "submit"])
@pytest.mark.parametrize("metadata", ["shape", "memref", "view"])
def test_c2b_result_types_preserve_dynamic_metadata_on_ctx_rewrite(kind, metadata):
    """Nonliteral type fields retain normal base-mutator traversal semantics."""
    span = _span()
    dim = ir.Var("dim", ir.ScalarType(DataType.INDEX), span)
    base = ir.Var("storage", ir.PtrType(), span)
    data = ir.Var("data", _dist_ty(), span)
    memref = ir.MemRef(base, dim, 64, span) if metadata == "memref" else None
    view = (
        ir.TensorView(
            stride=[dim], layout=ir.TensorLayout.ND, valid_shape=[ir.ConstInt(2, DataType.INDEX, span)]
        )
        if metadata == "view"
        else None
    )
    ordinary_type = ir.TensorType([dim] if metadata == "shape" else [4], pl.FP32, memref, view)
    ordinary = ir.Var("ordinary", ordinary_type, span)
    params = [dim, base, data, ordinary]
    directions = [
        ir.ArgDirection.Scalar,
        ir.ArgDirection.Scalar,
        ir.ArgDirection.Input,
        ir.ArgDirection.Input,
    ]
    stage = ir.Function(
        "stage",
        params,
        [ordinary_type, data.type],
        ir.ReturnStmt([ordinary, data], span),
        span,
        type=ir.FunctionType.InCore,
    )
    result_type = ir.TupleType(
        [
            ordinary_type,
            data.type,
            *([ir.ScalarType(DataType.TASK_ID)] if kind == "submit" else []),
        ]
    )
    attrs = {"arg_directions": directions}
    invocation = (
        ir.Submit(ir.GlobalVar("stage"), params, [], {}, attrs, result_type, span)
        if kind == "submit"
        else ir.Call(ir.GlobalVar("stage"), params, {}, attrs, result_type, span)
    )
    packed = ir.Var("packed", result_type, span)
    output = ir.Var("output", ordinary_type, span)
    returned = ir.Var("returned", data.type, span)
    context = ir.Var("context", ir.CommCtxType(), span)
    body = ir.SeqStmts(
        [
            ir.AssignStmt(packed, invocation, span),
            ir.AssignStmt(output, ir.TupleGetItemExpr(packed, 0, span), span),
            ir.AssignStmt(returned, ir.TupleGetItemExpr(packed, 1, span), span),
            ir.AssignStmt(context, ir.create_op_call("pld.system.get_comm_ctx", [returned], {}, span), span),
            ir.ReturnStmt([], span),
        ],
        span,
    )
    if kind == "submit":
        body = ir.RuntimeScopeStmt(manual=True, body=body, span=span)
    caller = ir.Function("caller", params, [], body, span, type=ir.FunctionType.Orchestration)
    before = ir.Program([stage, caller], "DynamicMetadata", span)
    props = passes.IRPropertySet()
    for prop in (
        passes.IRProperty.TypeChecked,
        passes.IRProperty.AssignTypeSymmetry,
        passes.IRProperty.NoScalarKernelReturn,
    ):
        props.insert(prop)
    assert passes.PropertyVerifierRegistry.verify(props, before) == []
    after = passes.materialize_dist_tensor_ctx()(before)
    assert passes.PropertyVerifierRegistry.verify(props, after) == []
    caller = _get_func(after, "caller")
    body = caller.body.body if kind == "submit" else caller.body
    launch = body.stmts[0].value
    assert isinstance(launch, ir.Submit if kind == "submit" else ir.Call)
    assert launch.type is result_type
    assert launch.args[-1] is body.stmts[3].value is caller.params[-1]
    assert list(launch.arg_directions) == [*directions, ir.ArgDirection.Scalar]
    restored_type = launch.type.types[0]
    assert restored_type is body.stmts[1].var.type is body.stmts[1].value.type is ordinary_type
    if metadata == "shape":
        assert restored_type.shape[0] is caller.params[0]
    elif metadata == "memref":
        assert restored_type.memref is memref
        assert restored_type.memref.base_ is caller.params[1]
        assert restored_type.memref.byte_offset_ is caller.params[0]
    else:
        assert restored_type.tensor_view.stride[0] is caller.params[0]
        assert restored_type.tensor_view.valid_shape[0].value == 2
        assert restored_type.tensor_view.layout == ir.TensorLayout.ND
    if kind == "submit":
        assert launch.type.types[-1].dtype == DataType.TASK_ID


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
