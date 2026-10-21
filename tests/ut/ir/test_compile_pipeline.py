# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for the shared IR pass pipeline."""

import json
import re
import warnings
from contextlib import nullcontext
from pathlib import Path

import pytest
from pypto import DataType, ir
from pypto.compile_profiling import CompileProfiler
from pypto.ir.compile import _run_pass_pipeline
from pypto.pypto_core import passes
from pypto.runtime import ExecutionMode


def _scalar_program() -> ir.Program:
    span = ir.Span.unknown()
    dtype = ir.ScalarType(DataType.INT64)
    x = ir.Var("x", dtype, span)
    y = ir.Var("y", dtype, span)
    body = ir.SeqStmts([ir.AssignStmt(y, x, span), ir.ReturnStmt([y], span)], span)
    fn = ir.Function("main", [x], [dtype], body, span)
    return ir.Program([fn], "lower_test", span)


def test_run_pass_pipeline_orders_outer_before_extra_instruments():
    seen: list[tuple[str, str]] = []
    outer_instrument = passes.CallbackInstrument(
        before_pass=lambda pass_obj, _program: seen.append(("outer", pass_obj.get_name())),
        name="outer",
    )
    extra_instrument = passes.CallbackInstrument(
        before_pass=lambda pass_obj, _program: seen.append(("extra", pass_obj.get_name())),
        name="extra",
    )
    with passes.PassContext([outer_instrument]):
        result = _run_pass_pipeline(
            _scalar_program(),
            operation="lower",
            extra_instruments=(extra_instrument,),
        )
    assert isinstance(result.transformed_program, ir.Program)
    assert seen
    assert len(seen) % 2 == 0
    for outer_event, extra_event in zip(seen[::2], seen[1::2], strict=True):
        assert outer_event[0] == "outer"
        assert extra_event == ("extra", outer_event[1])


@pytest.mark.parametrize("mode", ["plain", "dump", "profiling", "dump_and_profiling"])
@pytest.mark.parametrize("enabled", [False, True])
def test_buffer_ir_option_survives_nested_pipeline_contexts(tmp_path, mode, enabled):
    """Compilation, dump and profiling wrappers inherit the same storage option."""
    seen: list[bool] = []

    def observe(_pass: passes.Pass, _program: ir.Program) -> None:
        ctx = passes.PassContext.current()
        assert ctx is not None
        seen.append(ctx.get_enable_buffer_ir())

    instrument = passes.CallbackInstrument(before_pass=observe, name="ObserveBufferIR")
    profiling = CompileProfiler() if "profiling" in mode else nullcontext()
    with passes.PassContext([instrument], enable_buffer_ir=enabled), profiling:
        _run_pass_pipeline(
            _scalar_program(),
            operation="lower",
            dump_passes="dump" in mode,
            passes_dump_dir=str(tmp_path / "passes"),
        )
        ctx = passes.PassContext.current()
        assert ctx is not None and ctx.get_enable_buffer_ir() == enabled
    assert seen and all(value == enabled for value in seen)


def test_run_pass_pipeline_names_diagnostic_conflict_for_lower():
    with passes.PassContext([]):
        with pytest.raises(RuntimeError, match=r"lower\(\).*diagnostic_phase"):
            _run_pass_pipeline(
                _scalar_program(),
                operation="lower",
                diagnostic_phase=passes.DiagnosticPhase.POST_PASS,
            )


def test_compile_validates_platform_before_creating_output(tmp_path):
    output_dir = tmp_path / "must_not_exist"
    with pytest.raises(ValueError, match="Invalid platform"):
        ir.compile(
            _scalar_program(),
            output_dir=str(output_dir),
            platform="invalid",
            dump_passes=False,
            skip_ptoas=True,
        )
    assert not output_dir.exists()


def test_compile_creates_output_before_pass_context_conflict(tmp_path):
    output_dir = tmp_path / "compile_output"
    with passes.PassContext([]):
        with pytest.raises(RuntimeError, match=r"compile\(\).*diagnostic_phase"):
            ir.compile(
                _scalar_program(),
                output_dir=str(output_dir),
                diagnostic_phase=passes.DiagnosticPhase.POST_PASS,
                dump_passes=False,
                skip_ptoas=True,
            )
    assert output_dir.is_dir()
    assert not (output_dir / "report").exists()


def test_compile_preserves_dump_and_report_artifacts(tmp_path):
    output_dir = tmp_path / "compile_output"
    ir.compile(
        _scalar_program(),
        output_dir=str(output_dir),
        dump_passes=True,
        skip_ptoas=True,
        dump_ptoas_passes=True,
    )

    passes_dump = output_dir / "passes_dump"
    assert (passes_dump / "00_frontend.py").is_file()
    assert list(passes_dump.glob("*_after_*.py"))
    assert (output_dir / "report").is_dir()
    assert not (output_dir / "ptoas_passes").exists()


def test_compile_owned_profiler_writes_nested_pass_and_codegen_stages(tmp_path):
    output_dir = tmp_path / "owned_profile"
    assert CompileProfiler.current() is None
    ir.compile(
        _scalar_program(),
        output_dir=str(output_dir),
        dump_passes=False,
        skip_ptoas=True,
        profiling=True,
    )
    assert CompileProfiler.current() is None

    report_path = output_dir / "report" / "pipeline_profile.json"
    profile = json.loads(report_path.read_text())
    stages = profile["stages"]
    assert [stage["name"] for stage in stages] == ["passes", "codegen"]
    assert stages[0]["children"]


def test_compile_outer_profiler_retains_ownership(tmp_path):
    output_dir = tmp_path / "outer_profile"
    with CompileProfiler() as profiler:
        ir.compile(
            _scalar_program(),
            output_dir=str(output_dir),
            dump_passes=False,
            skip_ptoas=True,
            profiling=True,
        )
        assert CompileProfiler.current() is profiler
    assert CompileProfiler.current() is None

    stages = profiler.to_dict()["stages"]
    assert [stage["name"] for stage in stages] == ["passes", "codegen"]
    assert stages[0]["children"]
    assert not (output_dir / "report" / "pipeline_profile.json").exists()


def test_default_output_dirs_are_unique_per_compile(tmp_path, monkeypatch):
    """Two compiles of one program never share a directory.

    The default used to be ``<program name>_<timestamp>`` at one-second
    resolution, created with ``exist_ok=True``. Two same-named programs compiled
    inside one second therefore landed in one directory and the second's kernels
    overwrote the first's, silently: a caller that compiled a batch up front and
    dispatched afterwards got the wrong kernel with no error, and only a numeric
    assertion could catch it.

    Asserted on distinctness rather than on the naming scheme, so it still holds
    if the scheme changes again.
    """
    monkeypatch.setenv("PYPTO_PROG_BUILD_DIR", str(tmp_path))

    dirs = [ir.compile(_scalar_program(), dump_passes=False, skip_ptoas=True).output_dir for _ in range(3)]

    assert len({str(d) for d in dirs}) == 3, f"compiles shared an output directory: {dirs}"
    for d in dirs:
        assert Path(d).is_dir()


def test_default_output_dir_name_leads_with_timestamp(tmp_path, monkeypatch):
    """The default directory reads ``<name>_<YYYYmmdd_HHMMSS>_<random>``.

    The timestamp keeps ``build_output/`` human-readable and chronologically
    sortable; the random tail (not the timestamp) is what keeps it unique.
    """
    monkeypatch.setenv("PYPTO_PROG_BUILD_DIR", str(tmp_path))
    program = _scalar_program()

    output_dir = Path(ir.compile(program, dump_passes=False, skip_ptoas=True).output_dir)

    assert output_dir.parent == tmp_path
    assert re.fullmatch(rf"{re.escape(program.name)}_\d{{8}}_\d{{6}}_\w+", output_dir.name), output_dir.name


@pytest.mark.parametrize("mode", ["sim", 1, True])
def test_compile_rejects_invalid_mode_before_creating_output(tmp_path, mode):
    output_dir = tmp_path / "invalid_target"
    with pytest.raises(TypeError, match="execution_mode"):
        ir.compile(_scalar_program(), execution_mode=mode, output_dir=str(output_dir))
    assert not output_dir.exists()


def test_compile_rejects_conflicting_mode_before_creating_output(tmp_path):
    output_dir = tmp_path / "conflicting_target"
    with pytest.raises(ValueError, match="conflicts"):
        ir.compile(
            _scalar_program(), platform="a5", execution_mode=ExecutionMode.SIM, output_dir=str(output_dir)
        )
    assert not output_dir.exists()


@pytest.mark.parametrize("legacy", [True, False])
def test_compile_platform_deprecation_does_not_leak_into_new_api(tmp_path, legacy):
    from pypto.runtime import CompileOptions  # noqa: PLC0415

    kwargs = CompileOptions(execution_mode=ExecutionMode.ONBOARD).as_compile_kwargs()
    kwargs.update(output_dir=str(tmp_path), skip_ptoas=True, dump_passes=False)
    if legacy:
        with pytest.warns(DeprecationWarning, match=r"ir.compile\(platform=") as caught:
            compiled = ir.compile(_scalar_program(), platform="a2a3", **kwargs)
        assert len(caught) == 1
        assert caught[0].filename == __file__
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            compiled = ir.compile(_scalar_program(), **kwargs)
    assert compiled.platform == "a2a3"


def _c1d1_level_program(level):
    span = ir.Span.unknown()
    func = ir.Function("main", [], [], ir.SeqStmts([], span), span, level=level, role=ir.Role.Orchestrator)
    return ir.Program([func], "c1d1_cluster_config", span)


@pytest.mark.parametrize("level", [ir.Level.CHIP, ir.Level.HOST])
def test_c1d1_compile_cluster_config_rejects_noncluster_before_backend(tmp_path, monkeypatch, level):
    import importlib  # noqa: PLC0415

    from pypto.ir import (  # noqa: PLC0415
        ClusterConfig,
        ClusterNodeSpec,
    )

    compile_module = importlib.import_module("pypto.ir.compile")

    def no_backend(**kwargs):
        pytest.fail("Invalid cluster placement must fail before backend selection")

    monkeypatch.setattr(compile_module, "_select_backend", no_backend)
    destination = tmp_path / "must_not_exist"
    with pytest.raises(ValueError, match="(?i)cluster_config.*(POD|CLUSTER_0|L4|cluster)"):
        ir.compile(
            _c1d1_level_program(level),
            output_dir=str(destination),
            cluster_config=ClusterConfig([ClusterNodeSpec("node-a")], "/env/bin/python"),
        )
    assert not destination.exists()


def test_c1d1_compile_cluster_distributed_conflict_rejected_before_backend(tmp_path, monkeypatch):
    import importlib  # noqa: PLC0415

    from pypto.ir import (  # noqa: PLC0415
        ClusterConfig,
        ClusterNodeSpec,
        DistributedConfig,
    )

    compile_module = importlib.import_module("pypto.ir.compile")

    def no_backend(**kwargs):
        pytest.fail("Conflicting launch configs must fail before backend selection")

    monkeypatch.setattr(compile_module, "_select_backend", no_backend)
    with pytest.raises(
        ValueError, match="(?i)(cluster_config.*distributed_config|distributed_config.*cluster_config)"
    ):
        ir.compile(
            _c1d1_level_program(ir.Level.HOST),
            output_dir=str(tmp_path / "conflict"),
            cluster_config=ClusterConfig([ClusterNodeSpec("node-a")], "/env/bin/python"),
            distributed_config=DistributedConfig(),
        )
    assert not (tmp_path / "conflict").exists()


def test_c1d1_compile_cluster_config_retains_temporary_pod_guard(tmp_path, monkeypatch):
    import importlib  # noqa: PLC0415

    from pypto.ir import (  # noqa: PLC0415
        ClusterConfig,
        ClusterNodeSpec,
    )

    compile_module = importlib.import_module("pypto.ir.compile")

    def no_backend(**kwargs):
        pytest.fail("Full POD compilation remains unsupported before C3")

    monkeypatch.setattr(compile_module, "_select_backend", no_backend)
    with pytest.raises(NotImplementedError, match="(?i)(POD|CLUSTER_0|L4)"):
        ir.compile(
            _c1d1_level_program(ir.Level.CLUSTER_0),
            output_dir=str(tmp_path / "pod"),
            cluster_config=ClusterConfig([ClusterNodeSpec("node-a")], "/env/bin/python"),
        )
    assert not (tmp_path / "pod").exists()


@pytest.mark.parametrize("has_config", [False, True])
def test_c1d2_postpass_cluster_entry_selects_cluster_wrapper_contract(tmp_path, monkeypatch, has_config):
    import importlib  # noqa: PLC0415
    from types import SimpleNamespace  # noqa: PLC0415

    from pypto.backend import BackendType  # noqa: PLC0415
    from pypto.ir import ClusterConfig, ClusterNodeSpec, DistributedCompiledProgram  # noqa: PLC0415

    module = importlib.import_module("pypto.ir.compile")
    span = ir.Span.unknown()
    body = ir.SeqStmts([], span)
    host = ir.Function("host_dep", [], [], body, span, level=ir.Level.HOST, role=ir.Role.Orchestrator)
    cluster = ir.Function(
        "cluster_entry", [], [], body, span, level=ir.Level.CLUSTER_0, role=ir.Role.Orchestrator
    )
    transformed = ir.Program([host, cluster], "c1d2_postpass", span)
    monkeypatch.setattr(module, "_check_supported_orchestration_level", lambda *_: None)
    monkeypatch.setattr(
        module,
        "_run_pass_pipeline",
        lambda *args, **kwargs: SimpleNamespace(
            transformed_program=transformed,
            memory_planner=passes.MemoryPlanner.PYPTO,
            backend_type=BackendType.Ascend910,
            runtime=passes.RuntimeKind.TENSORMAP_AND_RINGBUFFER,
        ),
    )
    monkeypatch.setattr(
        module,
        "generate",
        lambda *args, **kwargs: {
            "orchestration/cluster_orch.py": "raise AssertionError('metadata-only fixture')\n",
        },
    )
    kwargs = (
        {"cluster_config": ClusterConfig([ClusterNodeSpec("node-a")], "/env/bin/python")}
        if has_config
        else {}
    )
    if not has_config:
        with pytest.raises(ValueError, match=r"(?i)(cluster_config|ClusterConfig)"):
            module.compile(transformed, output_dir=str(tmp_path), skip_ptoas=True, **kwargs)
    else:
        compiled = module.compile(transformed, output_dir=str(tmp_path), skip_ptoas=True, **kwargs)
        assert isinstance(compiled, DistributedCompiledProgram)
        assert compiled.program is transformed
        assert compiled.entry_kind == "cluster" and compiled.entry_name == "cluster_entry"
        assert compiled.cluster_config == kwargs["cluster_config"]
        record = json.loads((tmp_path / "distributed_meta.json").read_text())
        assert record["schema"] == 4 and record["entry_name"] == "cluster_entry"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
