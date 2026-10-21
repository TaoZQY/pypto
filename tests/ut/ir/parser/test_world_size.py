# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: F722, F821

"""Parser tests for ``pld.world_size()`` / ``pld.system.world_size()``.

The short form (``pld.world_size()``) is the unified-dispatch entry point
and the long form (``pld.system.world_size()``) is the canonical 3-segment
surface. Both lift to ``ir.OpExpr('pld.system.world_size')`` returning a
scalar INT64. Host-only enforcement happens at parse time: invocations
outside a ``level=pl.Level.HOST`` function body are rejected with a clear
error message.
"""

import ast
import textwrap
from importlib import import_module

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
from pypto import DataType
from pypto.language.parser.ast_parser import ASTParser
from pypto.language.parser.diagnostics import InvalidOperationError, ParserSyntaxError
from pypto.pypto_core import ir, passes

_OP_PLD_SYSTEM_WORLD_SIZE = ir.get_op("pld.system.world_size").name


def _get_func(program: ir.Program, name: str) -> ir.Function:
    gvar = program.get_global_var(name)
    assert gvar is not None
    return program.functions[gvar]


def _find_world_size_calls(func: ir.Function) -> list[ir.Call]:
    """Return every ``pld.system.world_size()`` call appearing anywhere in ``func``.

    Walks both statements and expression subtrees so we catch calls nested
    inside another call's argument list (e.g.
    ``pld.tensor.alloc_window_buffer(pld.world_size() * 4)``).
    """
    found: list[ir.Call] = []

    def visit_expr(expr: ir.Expr | None) -> None:
        if expr is None or not isinstance(expr, ir.Call):
            return
        if expr.op.name == _OP_PLD_SYSTEM_WORLD_SIZE:
            found.append(expr)
        for sub in expr.args:
            visit_expr(sub)

    def walk(stmt: ir.Stmt) -> None:
        if isinstance(stmt, ir.AssignStmt):
            visit_expr(stmt.value)
        if isinstance(stmt, ir.SeqStmts):
            for s in stmt.stmts:
                walk(s)
        if isinstance(stmt, ir.ForStmt):
            visit_expr(stmt.start)
            visit_expr(stmt.stop)
            visit_expr(stmt.step)
            walk(stmt.body)

    walk(func.body)
    return found


def test_world_size_call_returns_int64_scalar():
    """A bare ``pld.world_size()`` call lifts to a ScalarType(INT64) IR call."""

    @pl.program
    class P:
        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            n = pld.world_size()
            return n

    func = _get_func(P, "host_orch")
    calls = _find_world_size_calls(func)
    assert len(calls) == 1
    call = calls[0]
    assert isinstance(call.type, ir.ScalarType)
    assert call.type.dtype == DataType.INT64
    assert call.args == []
    assert call.kwargs == {}


def test_world_size_can_drive_pl_range_bound():
    """``pl.range(pld.world_size())`` produces a for-loop whose stop bound is
    the IR call (no DynVar / sentinel rewrite)."""

    @pl.program
    class P:
        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            for _ in pl.range(pld.world_size()):  # type: ignore[arg-type]
                pass
            return 0

    func = _get_func(P, "host_orch")

    def find_for(stmt: ir.Stmt) -> ir.ForStmt | None:
        if isinstance(stmt, ir.ForStmt):
            return stmt
        if isinstance(stmt, ir.SeqStmts):
            for s in stmt.stmts:
                hit = find_for(s)
                if hit is not None:
                    return hit
        return None

    for_stmt = find_for(func.body)
    assert for_stmt is not None
    # The stop bound is the result of pld.world_size() — a Var assigned from
    # the world_size call, since the parser threads expressions through SSA.
    stop = for_stmt.stop
    assert isinstance(stop.type, ir.ScalarType)
    assert stop.type.dtype == DataType.INT64


def test_world_size_rejects_positional_args():
    with pytest.raises(InvalidOperationError, match=r"no positional arguments|takes 0 positional"):

        @pl.program
        class P:  # noqa: F841
            @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
            def host_orch(self):
                n = pld.world_size(4)  # type: ignore[call-arg]  # noqa: F841
                return 0


def test_world_size_rejects_kwargs():
    with pytest.raises(
        InvalidOperationError, match=r"does not accept keyword arguments|unexpected keyword argument"
    ):

        @pl.program
        class P:  # noqa: F841
            @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
            def host_orch(self):
                n = pld.world_size(rank=0)  # type: ignore[call-arg]  # noqa: F841
                return 0


def test_world_size_rejected_outside_host_function():
    """``pld.world_size()`` is host-only — calling it from a CORE_GROUP-level
    function body is a parse error."""
    with pytest.raises(ParserSyntaxError, match="HOST"):

        @pl.program
        class P:  # noqa: F841
            @pl.function
            def kernel(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    n = pld.world_size()  # noqa: F841
                return x


def test_world_size_rejected_in_nested_device_scope_within_host_function():
    """Even inside a HOST orchestrator, ``pld.world_size()`` must be rejected
    when nested inside a device-side scope (InCore / SPMD), since
    the call is not lowerable there."""
    with pytest.raises(ParserSyntaxError, match="InCore"):

        @pl.program
        class P:  # noqa: F841
            @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
            def host_orch(self):
                with pl.at(level=pl.Level.CORE_GROUP):
                    n = pld.world_size()  # noqa: F841
                return 0


def test_world_size_call_used_as_size_in_alloc():
    """``pld.world_size()`` can flow into ``pld.tensor.alloc_window_buffer(size)``
    as a per-rank byte-size operand."""

    @pl.program
    class P:
        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            buf = pld.alloc_window_buffer(pld.world_size())
            return buf

    func = _get_func(P, "host_orch")
    calls = _find_world_size_calls(func)
    assert len(calls) == 1
    # The alloc op consumed the world_size call as its size argument.
    body = func.body
    assert isinstance(body, ir.SeqStmts)
    alloc_call = next(
        c
        for stmt in body.stmts
        if isinstance(stmt, ir.AssignStmt)
        and isinstance(stmt.value, ir.Call)
        and stmt.value.op.name == ir.get_op("pld.tensor.alloc_window_buffer").name
        for c in [stmt.value]
    )
    assert alloc_call.args[0] is calls[0]


def test_long_form_world_size_call():
    """The canonical ``pld.system.world_size()`` long form parses to the same op
    as the unified short form."""

    @pl.program
    class P:
        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            n = pld.system.world_size()
            return n

    func = _get_func(P, "host_orch")
    calls = _find_world_size_calls(func)
    assert len(calls) == 1
    assert calls[0].op.name == _OP_PLD_SYSTEM_WORLD_SIZE


def _c1b_program_source(body: str, *, level: str = "HOST", preamble: str = "") -> str:
    """Keep operand types valid so legality failures identify the execution context."""
    return (
        "import pypto.language as pl\n"
        "import pypto.language.distributed as pld\n"
        f"{preamble}\n"
        "@pl.program\nclass P:\n"
        f"    @pl.function(level=pl.Level.{level}, role=pl.Role.Orchestrator, auto_scope=False)\n"
        "    def main(self, data: pld.DistributedTensor[[8, 8], pl.FP32], "
        "signal: pld.DistributedTensor[[4], pl.INT32]):\n"
        + textwrap.indent(body, "        ")
        + "\n        return 0\n"
    )


@pytest.mark.parametrize(
    "module_name",
    [
        "pypto.ir.op.distributed.system_ops",
        "pypto.language.distributed.op.system_ops",
        "pypto.language.distributed.op.unified_ops",
        "pypto.language.distributed.op",
        "pypto.language.distributed",
    ],
)
def test_c1b_num_workers_is_exported(module_name):
    module = import_module(module_name)
    assert callable(getattr(module, "num_workers", None)), module_name
    assert "num_workers" in module.__all__


@pytest.mark.parametrize("level", ["HOST", "CLUSTER_0"])
@pytest.mark.parametrize("query", ["pld.num_workers", "pld.system.num_workers", "child_count"])
def test_c1b_num_workers_parser_and_roundtrip(level, query):
    # Resolve an aliased callable too: spelling alone must not determine legality.
    preamble = "child_count = pld.num_workers" if query == "child_count" else ""
    program = pl.parse_program(_c1b_program_source(f"n = {query}()", level=level, preamble=preamble))
    func = _get_func(program, "main")
    assert func.level == getattr(ir.Level, level)
    assert func.role == ir.Role.Orchestrator
    calls = [
        stmt.value
        for stmt in func.body.stmts
        if isinstance(stmt, ir.AssignStmt) and isinstance(stmt.value, ir.Call)
    ]
    assert len(calls) == 1
    assert calls[0].op.name == ir.get_op("pld.system.num_workers").name
    assert isinstance(calls[0].type, ir.ScalarType)
    assert calls[0].type.dtype == DataType.INT64
    assert calls[0].args == []
    assert calls[0].kwargs == {}
    ir.assert_structural_equal(program, pl.parse_program(str(program)))


@pytest.mark.parametrize("arguments", ["1", "count=1"])
def test_c1b_num_workers_rejects_arguments(arguments):
    with pytest.raises(InvalidOperationError, match="positional|keyword|kwargs"):
        pl.parse_program(_c1b_program_source(f"n = pld.num_workers({arguments})"))


@pytest.mark.parametrize("query", ["pld.num_workers", "pld.system.num_workers", "child_count"])
@pytest.mark.parametrize(
    ("level", "scope"),
    [
        ("CHIP", None),
        ("HOST", "with pl.at(level=pl.Level.CORE_GROUP):"),
        ("CLUSTER_0", "with pl.spmd(2):"),
        ("CLUSTER_0", "with pl.at(level=pl.Level.CHIP, role=pl.Role.Orchestrator):"),
    ],
)
def test_c1b_num_workers_rejects_device_context(level, scope, query):
    body = f"n = {query}()"
    if scope == "with pl.spmd(2):":
        # SPMD validates that each block can observe its own index before parsing
        # the body; satisfy that independent contract to reach query legality.
        body = f"block = pl.tile.get_block_idx()\n{body}"
    if scope:
        body = f"{scope}\n{textwrap.indent(body, '    ')}"
    preamble = "child_count = pld.num_workers" if query == "child_count" else ""
    with pytest.raises(ParserSyntaxError, match="orchestration context"):
        pl.parse_program(_c1b_program_source(body, level=level, preamble=preamble))


@pytest.mark.parametrize("query", ["pld.world_size", "pld.system.world_size", "local_count"])
@pytest.mark.parametrize("nested", [False, True])
def test_c1b_world_size_rejects_cluster_context(query, nested):
    body = f"n = {query}()"
    if nested:
        body = f"with pl.at(level=pl.Level.CLUSTER_0, role=pl.Role.Orchestrator):\n    {body}"
    preamble = "local_count = pld.world_size" if query == "local_count" else ""
    with pytest.raises(ParserSyntaxError, match="HOST"):
        pl.parse_program(
            _c1b_program_source(body, level="HOST" if nested else "CLUSTER_0", preamble=preamble)
        )


@pytest.mark.parametrize("query", ["pld.world_size", "pld.num_workers"])
@pytest.mark.parametrize("outer_level", ["CLUSTER_0", "CHIP"])
@pytest.mark.parametrize("generic_scope", ["pl.scope()", "pl.manual_scope()"])
def test_c1b_nested_host_context_accepts_local_queries(outer_level, query, generic_scope):
    # Hierarchy scopes select their own context even inside a generic device function.
    body = (
        "with pl.at(level=pl.Level.HOST, role=pl.Role.Orchestrator):\n"
        f"    with {generic_scope}:\n"
        f"        n = {query}()"
    )
    program = pl.parse_program(_c1b_program_source(body, level=outer_level))
    assert ir.get_op(f"pld.system.{query.split('.')[-1]}").name in str(program)
    ir.assert_structural_equal(program, pl.parse_program(str(program)))


@pytest.mark.parametrize("level", ["HOST", "CLUSTER_0"])
def test_c1b_orchestration_window_creation_is_legal(level):
    program = pl.parse_program(
        _c1b_program_source(
            "buf = pld.alloc_window_buffer(256)\nwindow = pld.window(buf, [8, 8], dtype=pl.FP32)",
            level=level,
        )
    )

    class CollectOps(ir.IRVisitor):
        def __init__(self):
            super().__init__()
            self.names = []

        def visit_call(self, op):
            self.names.append(op.op.name)
            super().visit_call(op)

    collector = CollectOps()
    collector.visit_program(program)
    # Allocation prints through private dump syntax; inspect its actual IR op.
    assert ir.get_op("pld.tensor.alloc_window_buffer").name in collector.names
    assert ir.get_op("pld.tensor.window").name in collector.names
    ir.assert_structural_equal(program, pl.parse_program(str(program)))


@pytest.mark.parametrize(
    ("preamble", "expression"),
    [
        ("", "pld.tensor.barrier(signal)"),
        ("sync = pld.tensor.barrier", "sync(signal)"),
        ("", "pld.tensor.allreduce(data, signal)"),
        ("reduce_data = pld.tensor.allreduce", "reduce_data(data, signal)"),
        ("", "pld.tile.remote_load(data, 0, offsets=[0, 0], shape=[8, 8])"),
        ("", "pld.remote_load(data, 0, offsets=[0, 0], shape=[8, 8])"),
        (
            "load_peer = pld.tile.remote_load",
            "load_peer(data, 0, offsets=[0, 0], shape=[8, 8])",
        ),
        (
            "",
            'pl.builtin.tensor.barrier(signal, attrs={"worker": 0, "arg_directions": [pl.adir.inout]})',
        ),
    ],
)
@pytest.mark.parametrize("nested", [False, True])
def test_c1b_cluster_communication_rejects_resolved_aliases(preamble, expression, nested):
    body = f"result = {expression}"
    if nested:
        body = f"with pl.at(level=pl.Level.CLUSTER_0, role=pl.Role.Orchestrator):\n    {body}"
    with pytest.raises(ParserSyntaxError, match="CLUSTER_0"):
        pl.parse_program(
            _c1b_program_source(body, level="HOST" if nested else "CLUSTER_0", preamble=preamble)
        )


def test_c1b_cluster_scope_allows_nested_host_collective():
    program = pl.parse_program(
        _c1b_program_source(
            "with pl.at(level=pl.Level.HOST, role=pl.Role.Orchestrator):\n"
            "    result = pld.tensor.barrier(signal)",
            level="CLUSTER_0",
        )
    )
    ir.assert_structural_equal(program, pl.parse_program(str(program)))


def test_c1b_nested_host_scope_restores_cluster_query_context():
    source = _c1b_program_source(
        "with pl.at(level=pl.Level.HOST, role=pl.Role.Orchestrator):\n"
        "    n = pld.world_size()\n"
        "invalid = pld.world_size()",
        level="CLUSTER_0",
    )
    with pytest.raises(ParserSyntaxError, match="HOST") as excinfo:
        pl.parse_program(source)
    # The inner HOST query must succeed; the diagnostic belongs to the outer one.
    expected_line = source.splitlines().index("        invalid = pld.world_size()") + 1
    assert excinfo.value.span["line"] == expected_line


@pytest.mark.parametrize("level", ["HOST", "CLUSTER_0"])
def test_c1b_num_workers_python_worker_body_is_opaque(level):
    source = (
        _c1b_program_source("n = pld.num_workers()", level=level)
        .replace("role=pl.Role.Orchestrator", "role=pl.Role.SubWorker")
        .replace("def main(self, ", "def main(")
    )
    program = pl.parse_program(source)
    func = _get_func(program, "main")
    assert func.role == ir.Role.SubWorker
    # HOST-and-higher SubWorkers capture Python source rather than parse DSL.
    # Keep that contract; native Call legality is covered by direct/serde IR tests.
    assert isinstance(func.body, ir.InlineStmt)
    assert func.body.language == ir.InlineLanguage.Python
    assert func.body.body.splitlines() == ["n = pld.num_workers()", "return 0"]
    calls = []

    class CollectCalls(ir.IRVisitor):
        def visit_call(self, op):
            calls.append(op)
            super().visit_call(op)

    CollectCalls().visit_function(func)
    assert calls == []


@pytest.mark.parametrize("query", ["world_size", "num_workers"])
def test_c1b_spmd_launch_count_query_uses_parent_host_context(query):
    program = pl.parse_program(
        _c1b_program_source(f"with pl.spmd(pld.{query}()):\n    block = pl.tile.get_block_idx()")
    )
    ir.assert_structural_equal(program, pl.parse_program(str(program)))
    properties = passes.IRPropertySet()
    properties.insert(passes.IRProperty.TypeChecked)
    errors = passes.PropertyVerifierRegistry.verify(properties, program)
    # The launch width is computed by HOST before entering the device body.
    assert errors == [], [error.message for error in errors]


class _C1bScopeProbeStack(list):
    """Count deterministic linear membership work without relying on timing."""

    def __init__(self):
        super().__init__()
        self.membership_scan_work = 0

    def __contains__(self, item):
        self.membership_scan_work += len(self)
        return super().__contains__(item)


@pytest.mark.parametrize("query", ["world_size", "num_workers"])
@pytest.mark.parametrize("depth", [1, 12])
def test_c1b_scope_context_membership_does_not_scan_nesting_stack(query, depth):
    body = "\n".join(f"n{i} = pld.{query}()" for i in range(8))
    for _ in range(depth):
        body = "with pl.scope():\n" + textwrap.indent(body, "    ")
    source = "def main():\n" + textwrap.indent(body + "\nreturn 0", "    ")
    parser = ASTParser("<c1b-scope-membership>", source.splitlines())
    stack = _C1bScopeProbeStack()
    parser._scope_kind_stack = stack
    function = parser.parse_function(
        ast.parse(source).body[0],
        func_level=ir.Level.HOST,
        func_role=ir.Role.Orchestrator,
        func_attrs={"auto_scope": False},
    )
    assert function.level == ir.Level.HOST
    # Every count call checks device context. Scanning the enclosing generic
    # scopes here creates depth * call-count work despite a linear source body.
    assert stack.membership_scan_work == 0
    assert stack == []


@pytest.mark.parametrize("kind", [ir.ScopeKind.InCore, ir.ScopeKind.Spmd])
def test_c1b_scope_membership_restores_repeated_kind_after_exception(kind):
    parser = ASTParser("<c1b-scope-restoration>", [])
    assert not parser._is_inside_scope(kind)
    with parser._scope_kind_context(kind):
        with parser._scope_kind_context(ir.ScopeKind.Runtime):
            with pytest.raises(RuntimeError, match="unwind repeated scope"):
                with parser._scope_kind_context(kind):
                    assert parser._is_inside_scope(kind)
                    raise RuntimeError("unwind repeated scope")
            # Leaving one occurrence cannot erase the still-active outer kind.
            assert parser._is_inside_scope(kind)
            assert parser._is_inside_scope(ir.ScopeKind.Runtime)
        assert parser._is_inside_scope(kind)
        assert not parser._is_inside_scope(ir.ScopeKind.Runtime)
    assert not parser._is_inside_scope(kind)
    assert parser._scope_kind_stack == []


@pytest.mark.parametrize("annotated", [False, True])
@pytest.mark.parametrize(
    ("level", "scope"),
    [
        ("CHIP", None),
        ("HOST", "with pl.at(level=pl.Level.CORE_GROUP):"),
        ("HOST", "with pl.spmd(2):"),
    ],
)
def test_c1b_printed_alloc_rejects_device_context(annotated, level, scope):
    target = "buf: pl.Ptr" if annotated else "buf"
    body = f'{target} = pl._dump.alloc_window_buffer(16, name="buf")'
    if scope == "with pl.spmd(2):":
        body = "block = pl.tile.get_block_idx()\n" + body
    if scope:
        body = f"{scope}\n" + textwrap.indent(body, "    ")
    with pytest.raises(ParserSyntaxError, match="orchestration context"):
        pl.parse_program(_c1b_program_source(body, level=level))


@pytest.mark.parametrize("annotated", [False, True])
@pytest.mark.parametrize("level", ["HOST", "CLUSTER_0"])
def test_c1b_printed_alloc_preserves_orchestration_roundtrip(annotated, level):
    target = "buf: pl.Ptr" if annotated else "buf"
    program = pl.parse_program(
        _c1b_program_source(f'{target} = pl._dump.alloc_window_buffer(16, name="buf")', level=level)
    )
    alloc = next(
        stmt.value
        for stmt in _get_func(program, "main").body.stmts
        if isinstance(stmt, ir.AssignStmt) and isinstance(stmt.value, ir.Call)
    )
    assert alloc.op.name == ir.get_op("pld.tensor.alloc_window_buffer").name
    assert alloc.kwargs["name"] == "buf"
    assert isinstance(alloc.type, ir.PtrType)
    ir.assert_structural_equal(program, pl.parse_program(str(program)))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
