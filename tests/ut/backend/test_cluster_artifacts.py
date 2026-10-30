# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Recursive CLUSTER artifact generation, cold contracts and runtime source closure."""

import ast
import copy
import ctypes
import gzip
import hashlib
import importlib
import io
import json
import os
import re
import subprocess
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pypto.language as pl
import pytest
import torch
from pypto import backend, codegen, ir
from pypto.backend import BackendType, pto_backend
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.jit._artifact_manifest import ArtifactSpec, ArtifactState, BuildKind
from pypto.runtime import _artifact_sources, _prebuilt


def _config():
    return ir.ClusterConfig(
        [ir.ClusterNodeSpec("node-a", [0, 1]), ir.ClusterNodeSpec("node-b", [2, 3])],
        "/env/bin/python",
    )


def _program(*, lower=True):
    @pl.program
    class Recursive:
        @pl.function(type=pl.FunctionType.Orchestration, level=pl.Level.CHIP, role=pl.Role.Orchestrator)
        def chip(
            self, x: pl.Tensor[[8], pl.FP32], count: pl.Scalar[pl.INT32], out: pl.Out[pl.Tensor[[8], pl.FP32]]
        ):
            return 0

        @pl.function(type=pl.FunctionType.Orchestration, level=pl.Level.CHIP, role=pl.Role.Orchestrator)
        def unused_chip(self, x: pl.Tensor[[8], pl.FP32]):
            return 0

        @pl.function(level=pl.Level.HOST, role=pl.Role.SubWorker)
        def observe(x: pl.Tensor[[8], pl.FP32]):
            assert x.numel() == 8

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def first(
            self, x: pl.Tensor[[8], pl.FP32], count: pl.Scalar[pl.INT32], out: pl.Out[pl.Tensor[[8], pl.FP32]]
        ) -> pl.Tensor[[8], pl.FP32]:
            self.observe(x)
            self.chip(x, count, out, worker=0)
            return x

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def second(
            self, x: pl.Tensor[[8], pl.FP32], count: pl.Scalar[pl.INT32], out: pl.Out[pl.Tensor[[8], pl.FP32]]
        ):
            self.chip(x, count, out, worker=1)
            return 0

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def unused_host(self, x: pl.Tensor[[8], pl.FP32]):
            self.unused_chip(x, worker=0)
            return 0

        @pl.function(level=pl.Level.CLUSTER_0, role=pl.Role.Orchestrator)
        def root(
            self,
            x: pl.Tensor[[8], pl.FP32],
            count: pl.Scalar[pl.INT32],
            left: pl.Out[pl.Tensor[[8], pl.FP32]],
            right: pl.Out[pl.Tensor[[8], pl.FP32]],
        ):
            self.first(x, count, left, worker=0)
            self.second(x, count, right, worker=1)
            return 0

    if not lower:
        return Recursive
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910)
    try:
        program = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(Recursive)
        assert program.functions[program.get_global_var("chip")].func_type == ir.FunctionType.Orchestration
        # Verify the real native complete-program source before testing packaging.
        compile(codegen.DistributedCodegen().generate(program), "cluster_orch.py", "exec")
        for name in ("first", "second"):
            source = codegen.DistributedCodegen()._generate_embedded(program, entry_name=name)
            compile(source, f"{name}.py", "exec")
        return program
    finally:
        backend.reset_for_testing()


def _emit(program, root):
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910)
    try:
        files = pto_backend.generate(program, str(root), skip_ptoas=True)
    finally:
        backend.reset_for_testing()
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    compiled = ir.DistributedCompiledProgram(program, str(root), cluster_config=_config())
    return compiled, files, json.loads((root / "distributed_meta.json").read_text())


@pytest.fixture(scope="module")
def cluster_program():
    return _program()


@pytest.fixture
def cluster_artifact(tmp_path, cluster_program):
    root = tmp_path / "generated"
    return root, *_emit(cluster_program, root)


def _layout(source):
    assignment = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute) and target.attr == "_pypto_embedded_layout"
            for target in node.targets
        )
    )
    assert isinstance(assignment.value, ast.Call)
    return json.loads(json.dumps(ast.literal_eval(assignment.value.args[0])))


def test_c3c_recursive_entries_keep_full_logical_chip_paths(cluster_artifact):
    root, _, files, metadata = cluster_artifact
    assert "orchestration/cluster_orch.py" in files
    assert "orchestration/host_orch.py" not in files
    assert not any("unused_" in path for path in files)
    closure = metadata["l3_entries"]
    assert closure["version"] == 1
    entries = closure["entries"]
    assert [entry["entry_name"] for entry in entries] == ["first", "second"]
    expected = {}
    for entry in entries:
        name = entry["entry_name"]
        prefix = f"next_levels/{name}"
        assert entry["logical_path"] == prefix
        assert entry["module"] == f"next_levels.{name}.orchestration.host_orch"
        assert entry["qualname"] == name
        assert entry["chip_paths"] == {"chip": f"{prefix}/next_levels/chip"}
        assert entry["layout"] == _layout(files[f"{prefix}/orchestration/host_orch.py"])
        assert entry["layout"]["version"] == 1
        assert [row[0:3:2] for row in entry["layout"]["tensor_slots"]] == [[0, 0], [2, 1]]
        assert [row[0:3:2] for row in entry["layout"]["scalar_slots"]] == [[1, 0]]
        assert entry["host_tensor_indices"] == [0, 1]
        assert entry["local_device_count"] == 2
        assert entry["minimum_sub_workers"] == (2 if name == "first" else 0)
        assert entry["sub_worker_paths"] == (
            {"observe": f"{prefix}/sub_workers/observe.py"} if name == "first" else {}
        )
        assert entry["return_copy_callable"] == (
            {"module": "orchestration._distributed_support", "qualname": "_copy_return_tensor"}
            if name == "first"
            else None
        )
        assert entry["view_geometry"] == {"version": 1, "window_slots": []}
        expected[f"{prefix}/next_levels/chip"] = root / prefix / "next_levels/chip"
    # Equal CHIP basenames are two distinct registration paths, not one flattened key.
    assert _prebuilt.chip_directories(root, BuildKind.DISTRIBUTED) == expected
    # A valid-looking stale subtree must never become part of this build.
    stale = root / "next_levels/stale/next_levels/chip/kernel_config.py"
    stale.parent.mkdir(parents=True)
    stale.write_text("raise AssertionError('stale config executed')\n")
    assert _prebuilt.chip_directories(root, BuildKind.DISTRIBUTED) == expected


def test_c3c_cold_restore_does_not_import_entries_or_reconstruct_program(cluster_artifact, monkeypatch):
    root, _, files, metadata = cluster_artifact
    for path in files:
        if path.endswith(".py"):
            (root / path).write_text("raise AssertionError('cold restore executed generated code')\n")

    def forbidden(*args, **kwargs):
        pytest.fail("Cold metadata restoration invoked compilation or source execution")

    monkeypatch.setattr(ir, "compile", forbidden)
    monkeypatch.setattr(_artifact_sources, "read_kernel_config", forbidden)
    restored = ir.DistributedCompiledProgram.from_dir(root)
    assert restored.program is None and restored.entry_kind == "cluster"
    assert restored.entry_name == "root" and restored.cluster_config == _config()
    assert json.loads((root / "distributed_meta.json").read_text()) == metadata
    # A restored cluster can use its dedicated runner, while the legacy HOST
    # execution seam still refuses to consume schema-4 entries.
    with pytest.raises(NotImplementedError, match=r"(?i)(cluster|POD|L4).*(execution|runner|support)"):
        restored._require_supported_execution()


@pytest.mark.parametrize(
    "path,value",
    [
        (("extra",), True),
        (("version",), True),
        (("version",), 2),
        (("entries", 0, "extra"), True),
        (("entries", 0, "logical_path"), "../escape"),
        (("entries", 0, "logical_path"), "/escape"),
        (("entries", 0, "logical_path"), "next_levels\\first"),
        (("entries", 0, "chip_paths", "chip"), "next_levels/second/next_levels/chip"),
        (("entries", 0, "layout", "tensor_slots", 1, 2), 0),
        (("entries", 0, "layout", "tensor_slots", 0, 2), True),
        (("entries", 0, "layout", "scalar_slots", 0, 0), 0),
        (("entries", 0, "layout", "extra"), []),
        (("entries", 0, "view_geometry", "extra"), []),
        (("entries", 0, "view_geometry", "version"), True),
        (("entries", 0, "local_device_count"), 3),
        (("entries", 0, "minimum_sub_workers"), -1),
        (("entries", 0, "module"), "next_levels.second.orchestration.host_orch"),
        (("entries", 0, "qualname"), "first()"),
    ],
)
def test_c3c_cold_entries_reject_malformed_contract(cluster_artifact, path, value):
    root, _, _, metadata = cluster_artifact
    closure = copy.deepcopy(metadata["l3_entries"])
    parent = closure
    for component in path[:-1]:
        parent = parent[component]
    parent[path[-1]] = value
    metadata["l3_entries"] = closure
    (root / "distributed_meta.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match=r"distributed_meta\.json.*(l3_entries|entry|layout|geometry)"):
        ir.DistributedCompiledProgram.from_dir(root)


@pytest.mark.parametrize("mutation", ["duplicate_entry", "missing_layout_field"])
def test_c3c_cold_entries_reject_duplicate_entries_and_incomplete_layout(cluster_artifact, mutation):
    root, _, _, metadata = cluster_artifact
    closure = metadata["l3_entries"]
    if mutation == "duplicate_entry":
        closure["entries"].append(copy.deepcopy(closure["entries"][0]))
    else:
        closure["entries"][0]["layout"].pop("prefork_requirements")
    (root / "distributed_meta.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match=r"distributed_meta\.json.*(l3_entries|entry|layout)"):
        ir.DistributedCompiledProgram.from_dir(root)


def test_c3c_parent_packages_every_nested_chip_before_prebuilt_promotion(cluster_artifact, monkeypatch):
    root, _, _, _ = cluster_artifact
    _artifact_sources.package_generated_sources(root, BuildKind.DISTRIBUTED)
    chips = _prebuilt.chip_directories(root, BuildKind.DISTRIBUTED)
    assert len(chips) == 2
    for chip in chips.values():
        record = json.loads((chip / "kernel_config.json").read_text())
        assert not Path(record["orchestration"]["source"]).is_absolute()
        assert (chip / record["orchestration"]["source"]).is_file()
        _prebuilt.write_chip_binaries(
            chip,
            "a2a3sim",
            record["orchestration"],
            [],
            b"native chip orchestration",
            "tensormap_and_ringbuffer",
            record["runtime"],
        )
    (root / "binary_manifest.json").write_text(
        json.dumps({"schema": 1, "platform": "a2a3sim", "chips": list(chips)})
    )

    def forbidden(*args, **kwargs):
        pytest.fail("Prebuilt metadata reconstruction executed kernel_config or compiler")

    monkeypatch.setattr(_artifact_sources, "read_kernel_config", forbidden)
    records = _prebuilt.read_prebuilt(root, "a2a3sim", BuildKind.DISTRIBUTED)
    assert set(records) == set(chips)
    spec = ArtifactSpec(
        ArtifactState.GENERATED,
        BuildKind.DISTRIBUTED,
        ("distributed_meta.json", "orchestration/cluster_orch.py"),
        entry_kind="cluster",
    )
    ready = _prebuilt.ready_spec(root, spec, metadata_only=True)
    assert ready.state is ArtifactState.BINARY_READY and ready.effective_entry_kind == "cluster"
    assert all(f"{path}/binary_manifest.json" in ready.required_files for path in chips)


def test_c3c_identity_only_cluster_metadata_cannot_claim_artifact_readiness(cluster_artifact):
    root, _, _, metadata = cluster_artifact
    metadata.pop("l3_entries")
    (root / "distributed_meta.json").write_text(json.dumps(metadata))
    assert ir.DistributedCompiledProgram.from_dir(root).entry_kind == "cluster"
    with pytest.raises((ValueError, NotImplementedError), match=r"(?i)(entry|l3_entries|closure|cluster)"):
        _prebuilt.chip_directories(root, BuildKind.DISTRIBUTED)


def test_c3c_source_bundle_is_deterministic_owned_ustar_with_one_helper(cluster_artifact, tmp_path):
    root, _, files, _ = cluster_artifact
    digest, payload = _artifact_sources.build_cluster_source_bundle(root)
    assert digest == hashlib.sha256(payload).hexdigest()
    assert _artifact_sources.build_cluster_source_bundle(root) == (digest, payload)
    # Archive identity cannot depend on output location, file timestamps or digest package name.
    other = tmp_path / "relocated"
    _emit(_program(), other)
    for path in other.rglob("*"):
        if path.is_file():
            os.utime(path, (12345, 12345))
    assert _artifact_sources.build_cluster_source_bundle(other) == (digest, payload)
    unpacked = gzip.decompress(payload)
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        assert names == sorted(names) and len(names) == len(set(names))
        assert all(member.isfile() and member.uid == member.gid == member.mtime == 0 for member in members)
        assert all(not Path(name).is_absolute() and ".." not in Path(name).parts for name in names)
        assert names.count("orchestration/_distributed_support.py") == 1
        assert "next_levels/first/sub_workers/observe.py" in names
        assert "next_levels/second/sub_workers/observe.py" not in names
        assert not any(name.endswith("kernel_config.py") or name.endswith(".cpp") for name in names)
        for member in members:
            assert unpacked[member.offset + 257 : member.offset + 265] == b"ustar\x0000"
            data = archive.extractfile(member).read()
            assert digest.encode() not in data
            if member.name.endswith(".py"):
                module = ast.parse(data)
                imports = [node for node in ast.walk(module) if isinstance(node, ast.ImportFrom)]
                assert not any(
                    node.module and (node.module == "pypto" or node.module.startswith("pypto."))
                    for node in imports
                )
        for name in ("first", "second"):
            member = archive.getmember(f"next_levels/{name}/orchestration/host_orch.py")
            assert archive.extractfile(member).read().decode() == files[member.name]
            assert f"next_levels/{name}/orchestration/__init__.py" in names


_RUNTIME_IMPORT_PROOF = r"""
import importlib
import importlib.abc
import sys
class BlockPyPTO(importlib.abc.MetaPathFinder):
    attempts = []
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'pypto' or fullname.startswith('pypto.'):
            self.attempts.append(fullname)
            raise AssertionError('Runtime sources imported ' + fullname)
guard = BlockPyPTO()
sys.meta_path.insert(0, guard)
namespace, digest, archive_path, cache_root = sys.argv[1:]
from pathlib import Path
from simpler.startup_artifacts import (
    StartupArtifactSpec, read_startup_artifacts, install_startup_artifacts, activate_startup_artifacts,
)
assert not Path(cache_root).exists()
spec = StartupArtifactSpec(archive_path, digest, namespace)
descriptions = read_startup_artifacts([spec.to_manifest()])
installed = install_startup_artifacts(descriptions, cache_root)
assert len(installed) == 1 and installed[0].digest == digest
activate_startup_artifacts(installed)
helper = importlib.import_module(namespace + '.orchestration._distributed_support')
for name in ('first', 'second'):
    module = importlib.import_module(namespace + '.next_levels.' + name + '.orchestration.host_orch')
    entry = getattr(module, name)
    assert entry._pypto_distributed_entry and entry._pypto_embedded_layout['version'] == 1
    shim = importlib.import_module(namespace + '.next_levels.' + name + '.orchestration')
    assert shim._distributed_support is helper
observer = importlib.import_module(namespace + '.next_levels.first.sub_workers.observe')
from simpler.buffer import create_host_shared_buffer, mint_owner_instance_id
from simpler.task_interface import CallConfig, DataType, TaskArgs, TensorArgType
from simpler.worker import Worker
owner = mint_owner_instance_id()
buffers = [create_host_shared_buffer(32, owner, index + 1) for index in range(2)]
worker = None
try:
    buffers[0].shm.buf[:] = bytes(32)
    buffers[1].shm.buf[:] = bytes([0xcd]) * 32
    source = buffers[0].tensor(shapes=(8,), dtype=DataType.FLOAT32)
    destination = buffers[1].tensor(shapes=(8,), dtype=DataType.FLOAT32)
    worker = Worker(level=3, num_sub_workers=2)
    observe_handle = worker.register(observer.observe)
    copy_handle = worker.register(helper._copy_return_tensor)
    worker.init()
    def orchestration(orch, args, config):
        observation = TaskArgs()
        observation.add_tensor(source, TensorArgType.INPUT)
        orch.submit_sub(observe_handle, observation)
        copying = TaskArgs()
        copying.add_tensor(source, TensorArgType.INPUT)
        copying.add_tensor(destination, TensorArgType.OUTPUT_EXISTING)
        orch.submit_sub(copy_handle, copying)
    worker.run(orchestration, args=None, config=CallConfig())
    assert bytes(buffers[1].shm.buf) == bytes(32)
finally:
    if worker is not None:
        worker.close()
    for backing in buffers:
        backing.close()
assert not guard.attempts
assert not any(name == 'pypto' or name.startswith('pypto.') for name in sys.modules)
print('C3C_RUNTIME_SOURCES_PASS')
"""


def test_c3c_installed_sources_import_in_arbitrary_digest_namespace_without_pypto(cluster_artifact, tmp_path):
    runtime_python = os.environ.get("L4_RUNTIME_ONLY_PYTHON")
    runtime_pythonpath = os.environ.get("L4_RUNTIME_ONLY_PYTHONPATH")
    if not runtime_python or not runtime_pythonpath:
        pytest.skip("Set L4_RUNTIME_ONLY_PYTHON and L4_RUNTIME_ONLY_PYTHONPATH for real runtime imports")
    root, _, _, _ = cluster_artifact
    digest, payload = _artifact_sources.build_cluster_source_bundle(root)
    namespace = "installed_" + digest
    archive = tmp_path / "sources.tar.gz"
    archive.write_bytes(payload)
    cache_root = tmp_path / "r1-empty-cache"
    # The real R1 loader validates compressed digest, USTAR members and install
    # ownership before activating its import finder. No manual extraction or
    # package path setup can conceal an incompatible producer archive.
    result = subprocess.run(
        [runtime_python, "-c", _RUNTIME_IMPORT_PROOF, namespace, digest, str(archive), str(cache_root)],
        cwd=tmp_path,
        env=os.environ
        | {
            "PYTHONPATH": os.pathsep.join((str(tmp_path), runtime_pythonpath)),
            "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
        },
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "C3C_RUNTIME_SOURCES_PASS" in result.stdout


def test_c3c_full_compile_routes_cluster_to_dedicated_runner(tmp_path, monkeypatch):
    # Exercise the public compile entry and actual pass/codegen path, rather
    # than bypassing the guard or replacing pipeline output with a fake program.
    program = _program(lower=False)
    compiled = ir.compile(
        program, output_dir=str(tmp_path / "compiled"), skip_ptoas=True, cluster_config=_config()
    )
    assert compiled.entry_kind == "cluster"
    metadata = json.loads((compiled.output_dir / "distributed_meta.json").read_text())
    assert len(metadata["l3_entries"]["entries"]) == 2
    runner = importlib.import_module("pypto.runtime.cluster_runner")
    calls = []
    sentinel = object()

    def prepare(programs, config, **kwargs):
        calls.append(programs)
        return sentinel

    monkeypatch.setattr(runner, "ClusterWorker", prepare)
    assert compiled.prepare() is sentinel
    assert calls == [[compiled]]


@pytest.mark.parametrize("level", [ir.Level.CLUSTER_1, ir.Level.CLUSTER_2, ir.Level.GLOBAL])
def test_c3c_cluster_one_and_higher_compile_remains_unsupported(level):
    module = importlib.import_module("pypto.ir.compile")
    with pytest.raises(NotImplementedError, match=r"(?i)(CLUSTER|higher|support|L5|L6|global)"):
        module._check_supported_orchestration_level(level)


def test_c3c_source_bundle_rejects_oversize_member_before_parsing(cluster_artifact, monkeypatch):
    from pypto.ir import _cluster_artifacts  # noqa: PLC0415

    root, _, _, _ = cluster_artifact
    # A small fixture-relative limit admits metadata but rejects this HOST
    # source. The parser must never see its deliberately oversized contents.
    limit = (root / "distributed_meta.json").stat().st_size + 1
    (root / "next_levels/first/orchestration/host_orch.py").write_text("#" + "x" * limit)
    monkeypatch.setattr(_artifact_sources, "_CLUSTER_MAX_FILE_BYTES", limit, raising=False)

    def forbidden(*args, **kwargs):
        pytest.fail("Oversize source reached AST/layout parsing before its read bound")

    monkeypatch.setattr(_cluster_artifacts, "source_layout", forbidden)
    with pytest.raises(ValueError, match=r"(?i)(source|member|file).*(size|limit|large)"):
        _artifact_sources.build_cluster_source_bundle(root)


def test_c3c_source_bundle_rejects_member_count_before_parsing(cluster_artifact, monkeypatch):
    from pypto.ir import _cluster_artifacts  # noqa: PLC0415

    root, _, _, _ = cluster_artifact
    monkeypatch.setattr(_artifact_sources, "_CLUSTER_MAX_MEMBERS", 3, raising=False)

    def forbidden(*args, **kwargs):
        pytest.fail("Source parsing started before checking the declared closure member count")

    monkeypatch.setattr(_cluster_artifacts, "source_layout", forbidden)
    with pytest.raises(ValueError, match=r"(?i)(members|member count)"):
        _artifact_sources.build_cluster_source_bundle(root)


def test_c3c_source_bundle_rejects_aggregate_content_limit(cluster_artifact, monkeypatch):
    root, _, _, _ = cluster_artifact
    monkeypatch.setattr(_artifact_sources, "_CLUSTER_MAX_UNPACKED_BYTES", 1, raising=False)
    with pytest.raises(ValueError, match=r"(?i)(aggregate|unpacked|content).*(size|limit|large)"):
        _artifact_sources.build_cluster_source_bundle(root)


def test_c3c_source_bundle_rejects_final_compressed_payload_limit(cluster_artifact, monkeypatch):
    root, _, _, _ = cluster_artifact
    monkeypatch.setattr(_artifact_sources, "_CLUSTER_MAX_PAYLOAD_BYTES", 1, raising=False)

    with pytest.raises(ValueError, match=r"(?i)(payload|archive|gzip).*(size|limit|large)"):
        _artifact_sources.build_cluster_source_bundle(root)


@pytest.mark.parametrize("old,new", [("first", "con"), ("second", "First")])
def test_c3c_source_bundle_rejects_receiver_path_aliases(cluster_program, tmp_path, old, new):
    # Both names are legal IR/Python identifiers. The producer must diagnose
    # archive portability before handing an unusable payload to the R1 cache.
    source = ir.python_print(cluster_program, format=False)
    renamed = re.sub(rf"\bself\.{old}\b", f"self.{new}", source.replace(f"def {old}(", f"def {new}("))
    program = pl.parse_program(renamed)
    root = tmp_path / "unsafe-receiver-path"
    _emit(program, root)
    with pytest.raises(ValueError, match=r"(?i)(reserved|unsafe|alias|casefold)"):
        _artifact_sources.build_cluster_source_bundle(root)


@pytest.fixture
def cluster_runner_harness(cluster_artifact, monkeypatch):
    """Keep native generated root/metadata and replace only remote process/device I/O."""
    runner = importlib.import_module("pypto.runtime.cluster_runner")
    from simpler.callable_identity import CallableHandle  # noqa: PLC0415
    from simpler.task_interface import RemoteAddressSpace, RemoteBufferHandle  # noqa: PLC0415
    from simpler.worker import RemoteCallable  # noqa: PLC0415

    root, compiled, _, metadata = cluster_artifact
    events, registrations, tasks = [], [], []
    chips = {
        path: object() for entry in metadata["l3_entries"]["entries"] for path in entry["chip_paths"].values()
    }
    state = SimpleNamespace(events=events, registrations=registrations, tasks=tasks, fail=False)

    class Worker:
        def __init__(self, *, level, **kwargs):
            assert level == 4
            self.memory = {}
            self.in_graph = False
            state.worker = self

        def register(self, target, *, workers=None):
            registrations.append((target, workers))
            remote = isinstance(target, RemoteCallable)
            return CallableHandle(
                "sha256:" + hashlib.sha256(str(len(registrations)).encode()).hexdigest(),
                "PYTHON_IMPORT" if remote else "CHIP_CALLABLE",
                "REMOTE_TASK_DISPATCHER" if remote else "LOCAL_CHIP",
            )

        def add_mpirun_worker_group(self, spec):
            state.spec = spec
            return (7, 13)

        def init(self, **kwargs):
            events.append("ready")

        def remote_malloc(self, *, worker, nbytes):
            assert not self.in_graph, "Remote allocation must precede graph submission"
            handle = RemoteBufferHandle._from_remote_allocation(
                worker_id=worker,
                buffer_id=len(self.memory) + 1,
                generation=1,
                address_space=RemoteAddressSpace.REMOTE_DEVICE,
                nbytes=nbytes,
            )
            self.memory[handle] = bytearray(nbytes)
            events.append(("malloc", worker))
            return handle

        def remote_copy_to(self, handle, ptr, nbytes, *, offset=0):
            assert not self.in_graph, "Remote upload must precede graph submission"
            events.append(("upload", handle.worker_id))
            self.memory[handle][offset : offset + nbytes] = ctypes.string_at(ptr, nbytes)

        def remote_copy_from(self, handle, ptr, nbytes, *, offset=0):
            events.append(("copyback", handle.worker_id))
            ctypes.memmove(ptr, bytes(self.memory[handle][offset : offset + nbytes]), nbytes)

        def remote_free(self, handle):
            events.append(("free", handle.worker_id))
            del self.memory[handle]
            handle._mark_released()

        def run(self, fn, args=None, config=None):
            def submit(handle, args, config, *, worker):
                tasks.append((handle, args, worker))

            self.in_graph = True
            try:
                fn(SimpleNamespace(submit_next_level=submit), args, config)
            finally:
                self.in_graph = False
            if state.fail:
                events.append("failed")
                raise RuntimeError("remote DAG failed")
            events.append("drained")

        def close(self):
            events.append("closed")
            self.memory.clear()

    monkeypatch.setattr(runner, "Worker", Worker)
    monkeypatch.setattr(
        runner,
        "_assemble_cluster_chips",
        lambda compiled: {path: (chip, compiled.cluster_config.runtime, {}) for path, chip in chips.items()},
    )
    # C3c's metadata fixture intentionally uses the real A3 fabric profile.
    # The process/device boundary is replaced here, so select its matching
    # real platform rather than weakening MpiL3GroupSpec's profile validation.
    state.compiled = ir.DistributedCompiledProgram.from_dir(root, platform="a2a3")
    state.chips = chips
    return runner, state


def test_i0_cluster_implicit_return_storage_is_preallocated_before_graph(cluster_artifact):
    root, compiled, files, metadata = cluster_artifact
    module = {}
    exec(compile(files["orchestration/cluster_orch.py"], "cluster_orch.py", "exec"), module)
    tensors = {"x": torch.ones(8), "count": 2, "left": torch.zeros(8), "right": torch.zeros(8)}
    originals = {
        value.untyped_storage().data_ptr() for value in tensors.values() if isinstance(value, torch.Tensor)
    }
    module["_alloc_intermediates"](tensors, nodes=[7, 13])
    implicit = [
        value
        for value in tensors.values()
        if isinstance(value, torch.Tensor) and value.untyped_storage().data_ptr() not in originals
    ]
    assert len(implicit) == 1, "Native CLUSTER allocator must materialize its implicit HOST return slot"
    assert implicit[0].shape == (8,) and implicit[0].dtype == torch.float32 and implicit[0].is_shared()
    body = ast.parse(files["orchestration/cluster_orch.py"])
    entry = next(node for node in body.body if isinstance(node, ast.FunctionDef) and node.name == "root")
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "torch"
        and node.func.attr in {"zeros", "empty", "ones", "full"}
        for node in ast.walk(entry)
    ), "Generated CLUSTER graph must not allocate storage after remote control staging"


def test_i0_cluster_public_prepare_uses_native_root_and_all_rank_remote_buffers(cluster_runner_harness):
    runner, state = cluster_runner_harness
    compiled = state.compiled
    assert compiled.program is None
    x, left, right = torch.arange(8, dtype=torch.float32), torch.zeros(8), torch.zeros(8)
    assert not list(compiled.output_dir.rglob("__pycache__"))
    with compiled.prepare() as worker:
        assert not list(compiled.output_dir.rglob("__pycache__")), (
            "Loading a cluster root must not write bytecode inside an immutable artifact"
        )
        assert isinstance(worker, runner.ClusterWorker)
        for _ in range(2):
            worker.run(compiled, x, -7, left, right)
        assert not state.worker.memory
    assert [task[2] for task in state.tasks] == [7, 13, 7, 13]
    assert [task[1].tensor_count() for task in state.tasks] == [3, 2, 3, 2]
    assert all(task[1].scalar_count() == 1 for task in state.tasks)
    assert state.spec.hosts == ("node-a", "node-b")
    assert len(state.spec.startup_hooks_by_rank) == 2
    assert len(state.registrations) == 4
    assert all(workers == [7, 13] for _, workers in state.registrations[2:])
    assert {event[1] for event in state.events if isinstance(event, tuple) and event[0] == "malloc"} == {
        7,
        13,
    }
    for index, event in enumerate(state.events):
        if isinstance(event, tuple) and event[0] == "copyback":
            assert "drained" in state.events[:index]
    assert state.events.index(("copyback", 13)) < state.events.index(("free", 7))


def test_i0_failure_never_copies_partial_outputs_and_frees_after_teardown(cluster_runner_harness):
    _, state = cluster_runner_harness
    state.fail = True
    with state.compiled.prepare() as worker:
        with pytest.raises(RuntimeError, match="remote DAG failed"):
            worker(torch.ones(8), 2, torch.zeros(8), torch.zeros(8))
    assert not any(isinstance(event, tuple) and event[0] == "copyback" for event in state.events)
    assert not state.worker.memory


def test_i0_failed_close_retains_invocation_frames_until_safe_teardown(cluster_runner_harness, monkeypatch):
    _, state = cluster_runner_harness
    worker = state.compiled.prepare()
    original_close = state.worker.close
    state.fail = True

    def fail_close():
        raise RuntimeError("hierarchy close failed")

    monkeypatch.setattr(state.worker, "close", fail_close)
    try:
        with pytest.raises(RuntimeError, match="hierarchy close failed"):
            worker(torch.ones(8), 2, torch.zeros(8), torch.zeros(8))
        assert worker._active_io is not None and worker._active_io.storages
        assert worker._keep, "An unconfirmed hierarchy close must retain submitted TaskArgs frames"
        assert not any(isinstance(event, tuple) and event[0] == "copyback" for event in state.events)
        previous_io, previous_frames = worker._active_io, tuple(worker._keep)
        before_retry = list(state.events)
        with pytest.raises(RuntimeError, match=r"(?i)(previous|pending|close)"):
            worker(torch.ones(8), 2, torch.zeros(8), torch.zeros(8))
        assert worker._active_io is previous_io, (
            "A new run must not replace an unconfirmed invocation journal"
        )
        assert tuple(worker._keep) == previous_frames
        assert state.events == before_retry, "Unsafe admission must fail before additional remote I/O"
    finally:
        monkeypatch.setattr(state.worker, "close", original_close)
        worker.close()
    assert worker._active_io is None and worker._keep == []
    assert not state.worker.memory


def test_i0_scalar_only_failed_close_retains_frames_without_storage(
    cluster_runner_harness, monkeypatch, tmp_path
):
    runner, state = cluster_runner_harness

    @pl.program
    class ScalarEntries:
        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def node(self, count: pl.Scalar[pl.INT32]):
            return 0

        @pl.function(level=pl.Level.CLUSTER_0, role=pl.Role.Orchestrator)
        def root(self, count: pl.Scalar[pl.INT32]):
            self.node(count, worker=0)
            return 0

    compiled = ir.compile(
        ScalarEntries,
        output_dir=str(tmp_path / "scalar-only"),
        skip_ptoas=True,
        cluster_config=_config(),
        platform="a2a3",
    )
    assert compiled._get_metadata()[0][0].shape is None
    assert not compiled.l3_entries["entries"][0]["layout"]["tensor_slots"]
    monkeypatch.setattr(runner, "_assemble_cluster_chips", lambda compiled: {})
    worker = compiled.prepare()
    original_close = state.worker.close
    state.fail = True

    def fail_close():
        raise RuntimeError("hierarchy close failed")

    monkeypatch.setattr(state.worker, "close", fail_close)
    try:
        with pytest.raises(RuntimeError, match="hierarchy close failed"):
            worker(3)
        assert worker._active_io is not None, "No storage is not proof that the accepted graph drained"
        assert worker._active_io.storages == {}
        assert worker._keep, "Scalar-only TaskArgs frames must survive an unconfirmed hierarchy close"
        previous_io, previous_frames = worker._active_io, tuple(worker._keep)
        before_retry = list(state.events)
        with pytest.raises(RuntimeError, match=r"(?i)(previous|pending|close)"):
            worker(4)
        assert worker._active_io is previous_io and tuple(worker._keep) == previous_frames
        assert state.events == before_retry
    finally:
        monkeypatch.setattr(state.worker, "close", original_close)
        worker.close()
    assert worker._active_io is None and worker._keep == []


def test_i0_contiguous_offset_aliases_share_remote_storage_and_cross_node_writes_fail(cluster_runner_harness):
    _, state = cluster_runner_harness
    with state.compiled.prepare() as worker:
        backing = torch.arange(16, dtype=torch.float32)
        view = backing[4:12]
        worker(view, 2, torch.zeros(8), torch.zeros(8))
        # One tensor may be read on two nodes. A write followed by a peer read
        # requires explicit transfer/fences, unavailable in ordinary I/O.
        with pytest.raises(NotImplementedError, match=r"(?i)(cross.node|forward|ordinary)"):
            worker(view, 2, view, torch.zeros(8))
        with pytest.raises(ValueError, match=r"(?i)contiguous"):
            worker(backing[::2], 2, torch.zeros(8), torch.zeros(8))


_BOOTSTRAP_PROOF = (
    _RUNTIME_IMPORT_PROOF.split("helper = importlib.import_module", 1)[0]
    + r"""
from types import SimpleNamespace
import json
import torch
from simpler.startup_preparation import NodeStartupContext
from simpler.task_interface import CallConfig, TaskArgs, TensorArgType
bootstrap = importlib.import_module(namespace + '.orchestration.cluster_bootstrap')
metadata = json.loads((installed[0].root / 'distributed_meta.json').read_text())
records = metadata['l3_entries']['entries']
paths = {path for entry in records for path in entry['chip_paths'].values()}
digests = {path: str(index + 1).zfill(64) for index, path in enumerate(sorted(paths))}
local_handles = {digest: object() for digest in digests.values()}
registered = []
class Worker:
    def register(self, function):
        registered.append(function)
        return object()
context = NodeStartupContext(
    rank=0, artifact_roots={digest: installed[0].root}, inner_worker=Worker(),
    chip_handles=local_handles,
    parameters={'module_prefix': namespace, 'entries': records, 'chip_digests': digests,
                'chip_configs': {path: {} for path in paths}},
)
state = bootstrap.prepare_node(context)
try:
    assert len(registered) == 2  # concrete observe SUB and return-copy SUB
    calls = []
    orch = SimpleNamespace(_worker=object(), submit_sub=lambda *a, **k: None,
                           submit_next_level=lambda *a, **k: calls.append((a, k)))
    from simpler.buffer import create_host_shared_buffer, mint_owner_instance_id
    from simpler.task_interface import DataType
    owner = mint_owner_instance_id()
    buffers = [create_host_shared_buffer(32, owner, index + 1) for index in range(3)]
    try:
        for record in records:
            module = importlib.import_module(namespace + '.' + record['module'])
            entry = getattr(module, record['qualname'])
            binding = entry._pypto_embedded_state
            assert binding.callables['chip'] is local_handles[digests[record['chip_paths']['chip']]]
            assert binding.callables.host_tensor_indices['chip'] == frozenset()
            args = TaskArgs()
            for index in range(2 + len(record['layout']['implicit_outputs'])):
                args.add_tensor(buffers[index].tensor(shapes=(8,), dtype=DataType.FLOAT32),
                                TensorArgType.INPUT if index == 0 else TensorArgType.OUTPUT_EXISTING)
            args.add_scalar(-7)
            for repeat in range(3):
                entry(orch, args, CallConfig())
                assert binding._frame is not None
                try:
                    entry(orch, args, CallConfig())
                    raise AssertionError('Premature entry reuse accepted')
                except RuntimeError as error:
                    assert 'completion' in str(error)
                state.task_completed()
                assert binding._frame is None
    finally:
        for backing in buffers:
            backing.close()
finally:
    state.close()
assert not guard.attempts
print('I0_BOOTSTRAP_PASS')
"""
)


_CPU_BOOTSTRAP_PROOF = (
    _RUNTIME_IMPORT_PROOF.split("helper = importlib.import_module", 1)[0]
    + r"""
import json
from simpler.startup_preparation import NodeStartupContext
from simpler.buffer import create_host_shared_buffer, mint_owner_instance_id
from simpler.task_interface import CallConfig, DataType, TaskArgs, TensorArgType
from simpler.worker import Worker
records = json.loads((installed[0].root / 'distributed_meta.json').read_text())['l3_entries']['entries']
bootstrap = importlib.import_module(namespace + '.orchestration.cluster_bootstrap')
worker = Worker(level=3, num_sub_workers=4)
state = None
owner = mint_owner_instance_id()
buffers = [create_host_shared_buffer(32, owner, index + 1) for index in range(2)]
try:
    state = bootstrap.prepare_node(NodeStartupContext(
        rank=0, artifact_roots={digest: installed[0].root}, inner_worker=worker, chip_handles={},
        parameters={'module_prefix': namespace, 'entries': records, 'chip_digests': {}, 'chip_configs': {}},
    ))
    worker.init()
    buffers[0].shm.buf[:] = bytes(range(32))
    source, destination = [b.tensor(shapes=(8,), dtype=DataType.FLOAT32) for b in buffers]
    for record in records:
        entry = getattr(importlib.import_module(namespace + '.' + record['module']), record['qualname'])
        args = TaskArgs()
        args.add_tensor(source, TensorArgType.INPUT)
        args.add_tensor(destination, TensorArgType.OUTPUT_EXISTING)
        for repeat in range(3):
            buffers[1].shm.buf[:] = bytes(32)
            worker.run(entry, args, CallConfig())
            assert entry._pypto_embedded_state._frame is not None
            state.task_completed()
            assert entry._pypto_embedded_state._frame is None
            assert bytes(buffers[1].shm.buf) == bytes(buffers[0].shm.buf)
finally:
    worker.close()
    if state is not None:
        state.close()
    for backing in buffers:
        backing.close()
assert not guard.attempts
print('I0_BOOTSTRAP_PASS')
"""
)


@pytest.mark.parametrize("real_cpu", [False, True])
def test_i0_bundle_bootstrap_binds_native_entries_without_pypto(cluster_artifact, tmp_path, real_cpu):
    runtime_python = os.environ.get("L4_RUNTIME_ONLY_PYTHON")
    runtime_pythonpath = os.environ.get("L4_RUNTIME_ONLY_PYTHONPATH")
    if not runtime_python or not runtime_pythonpath:
        pytest.skip("Set L4_RUNTIME_ONLY_PYTHON and L4_RUNTIME_ONLY_PYTHONPATH")
    root, _, _, _ = cluster_artifact
    if real_cpu:

        @pl.program
        class CpuEntries:
            @pl.function(level=pl.Level.HOST, role=pl.Role.SubWorker)
            def observe(x: pl.Tensor[[8], pl.FP32]):
                assert x.numel() == 8

            @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
            def first(self, x: pl.Tensor[[8], pl.FP32]) -> pl.Tensor[[8], pl.FP32]:
                self.observe(x)
                return x

            @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
            def second(self, x: pl.Tensor[[8], pl.FP32]) -> pl.Tensor[[8], pl.FP32]:
                self.observe(x)
                return x

            @pl.function(level=pl.Level.CLUSTER_0, role=pl.Role.Orchestrator)
            def root(self, x: pl.Tensor[[8], pl.FP32]):
                self.first(x, worker=0)
                self.second(x, worker=1)
                return 0

        # Full public compilation proves serialization/codegen premises before
        # the optional runtime-only child exercises actual CPU SUB completion.
        compiled = ir.compile(
            CpuEntries, output_dir=str(tmp_path / "cpu"), skip_ptoas=True, cluster_config=_config()
        )
        root = compiled.output_dir
    digest, payload = _artifact_sources.build_cluster_source_bundle(root)
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        assert "orchestration/cluster_bootstrap.py" in archive.getnames()
    archive = tmp_path / "startup.tar.gz"
    archive.write_bytes(payload)
    result = subprocess.run(
        [
            runtime_python,
            "-c",
            _CPU_BOOTSTRAP_PROOF if real_cpu else _BOOTSTRAP_PROOF,
            "installed_" + digest,
            digest,
            str(archive),
            str(tmp_path / "installed"),
        ],
        cwd=tmp_path,
        env=os.environ | {"PYTHONPATH": runtime_pythonpath, "TORCH_DEVICE_BACKEND_AUTOLOAD": "0"},
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "I0_BOOTSTRAP_PASS" in result.stdout


@pytest.fixture
def i1_global_artifact(tmp_path):
    # Both HOST entries receive one physical global window. Only first owns
    # another window, so its local scope must coexist with the inherited ABI.
    program = pl.parse_program("""
@pl.program
class GlobalEntries:
    @pl.function(type=pl.FunctionType.Orchestration, level=pl.Level.CHIP, role=pl.Role.Orchestrator)
    def mixed_chip(self, x: pl.Tensor[[8], pl.FP32], count: pl.Scalar[pl.INT32],
                   data: pld.DistributedTensor[[8], pl.FP32],
                   local: pld.DistributedTensor[[8], pl.FP32],
                   out: pl.Out[pl.Tensor[[8], pl.FP32]]):
        return 0
    @pl.function(type=pl.FunctionType.Orchestration, level=pl.Level.CHIP, role=pl.Role.Orchestrator)
    def global_chip(self, x: pl.Tensor[[8], pl.FP32], count: pl.Scalar[pl.INT32],
                    data: pld.DistributedTensor[[8], pl.FP32],
                    out: pl.Out[pl.Tensor[[8], pl.FP32]]):
        return 0
    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def first(self, x: pl.Tensor[[8], pl.FP32], count: pl.Scalar[pl.INT32],
              data: pld.DistributedTensor[[8], pl.FP32],
              out: pl.Out[pl.Tensor[[8], pl.FP32]], original: pld.CommCtx):
        local_buffer = pld.alloc_window_buffer(32)
        local = pld.window(local_buffer, [8], dtype=pl.FP32)
        self.mixed_chip(x, count, data, local, out, worker=1)
        return 0
    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def second(self, x: pl.Tensor[[8], pl.FP32], count: pl.Scalar[pl.INT32],
               data: pld.DistributedTensor[[8], pl.FP32],
               out: pl.Out[pl.Tensor[[8], pl.FP32]], original: pld.CommCtx):
        self.global_chip(x, count, data, out, worker=1)
        return 0
    @pl.function(level=pl.Level.CLUSTER_0, role=pl.Role.Orchestrator)
    def root(self, x: pl.Tensor[[8], pl.FP32], count: pl.Scalar[pl.INT32],
             left: pl.Out[pl.Tensor[[8], pl.FP32]], right: pl.Out[pl.Tensor[[8], pl.FP32]]):
        global_buffer = pld.alloc_window_buffer(32)
        data = pld.window(global_buffer, [8], dtype=pl.FP32)
        original = pld.system.get_comm_ctx(data)
        self.first(x, count, data, left, original, worker=0)
        self.second(x, count, data, right, original, worker=1)
        return 0
""")
    compiled = ir.compile(
        program,
        output_dir=str(tmp_path / "global"),
        skip_ptoas=True,
        cluster_config=_config(),
        platform="a2a3",
    )
    return compiled


def test_i1_global_full_native_pipeline_preserves_cold_domain_contract(i1_global_artifact):
    from contextlib import nullcontext  # noqa: PLC0415

    from pypto.runtime._distributed_support import _decode_scalar  # noqa: PLC0415
    from simpler.buffer import create_host_shared_buffer, mint_owner_instance_id  # noqa: PLC0415
    from simpler.task_interface import CallConfig, DataType  # noqa: PLC0415

    compiled = i1_global_artifact
    restored = ir.DistributedCompiledProgram.from_dir(compiled.output_dir, platform="a2a3")
    assert restored.program is None and restored.l3_entries == compiled.l3_entries
    records = compiled.l3_entries["entries"]
    assert [record["entry_name"] for record in records] == ["first", "second"]
    for record in records:
        layout = record["layout"]
        assert layout["version"] == 1 and len(layout["window_slots"]) == 1
        assert len(layout["tensor_slots"]) == 2  # Received DT is absent from dense tensor wire.
        assert [row[3] for row in layout["scalar_slots"]] == ["int32", "comm_ctx", "comm_ctx"]
        assert ["inherited_domain_tokens", 2] in layout["prefork_requirements"]
        assert ["domain_dispatch", 2, [1]] in layout["prefork_requirements"]
        local = [row for row in layout["prefork_requirements"] if row[0] == "local_domain_tokens"]
        assert bool(local) == (record["entry_name"] == "first")
        source = (compiled.output_dir / record["logical_path"] / "orchestration/host_orch.py").read_text()
        compile(source, record["module"], "exec")
    source = (compiled.output_dir / "orchestration/cluster_orch.py").read_text()
    assert ".domain_id" in source and ".device_ctx" not in source
    namespace = {"_io_arg": lambda binding, tensor, tag: descriptor}
    exec(compile(source, "cluster_orch.py", "exec"), namespace)
    backing = create_host_shared_buffer(32, mint_owner_instance_id(), 1)
    try:
        descriptor = backing.tensor(shapes=(8,), dtype=DataType.FLOAT32)
        calls = []
        domain_id = (1 << 63) + 907
        orch = SimpleNamespace(
            allocate_global_domain=lambda **kwargs: nullcontext(SimpleNamespace(domain_id=domain_id)),
            submit_next_level=lambda handle, incoming, cfg, *, worker: calls.append(incoming),
        )
        infos = compiled._get_metadata()[0]
        tensors = {info.name: (-7 if info.shape is None else torch.zeros(8)) for info in infos}
        namespace[compiled.entry_name](
            orch,
            None,
            CallConfig(),
            tensors=tensors,
            callables={record["entry_name"]: [object(), object()] for record in records},
            nodes=[7, 13],
            node_sizes=[2, 2],
            node_bindings=[None, None],
            _keep=[],
            _retain=False,
        )
        assert len(calls) == 2
        assert all(_decode_scalar(args.scalar(0), "int32") == -7 for args in calls)
        assert all([args.scalar(i) for i in (1, 2)] == [domain_id, domain_id] for args in calls)
    finally:
        backing.close()


def test_i1_parent_allocates_global_domain_only_in_active_graph(
    i1_global_artifact, cluster_runner_harness, monkeypatch
):
    # Replace process/device transport only. The entry and all TaskArgs below
    # are real output of the complete native compiler, including ctx suffixes.
    from contextlib import contextmanager  # noqa: PLC0415

    from pypto.runtime._distributed_support import _decode_scalar  # noqa: PLC0415

    runner, state = cluster_runner_harness
    base = runner.Worker
    allocations = []

    class Worker(base):
        def run(self, function, args=None, config=None):
            @contextmanager
            def allocate(**kwargs):
                assert self.in_graph, "R1 allocation requires the active parent graph"
                assert kwargs["members"] == [(7, 0), (7, 1), (13, 0), (13, 1)]
                assert kwargs.get("retain_after_run", False) is False
                allocations.append(kwargs)
                yield SimpleNamespace(domain_id=(1 << 63) + len(allocations))

            self.in_graph = True
            try:
                function(
                    SimpleNamespace(
                        allocate_global_domain=allocate,
                        submit_next_level=lambda handle, incoming, cfg, *, worker: state.tasks.append(
                            (handle, incoming, worker)
                        ),
                    ),
                    args,
                    config,
                )
            finally:
                self.in_graph = False
            state.events.append("drained")

    monkeypatch.setattr(runner, "Worker", Worker)
    monkeypatch.setattr(
        runner,
        "_assemble_cluster_chips",
        lambda compiled: {
            path: (object(), compiled.cluster_config.runtime, {})
            for record in compiled.l3_entries["entries"]
            for path in record["chip_paths"].values()
        },
    )
    with i1_global_artifact.prepare() as prepared:
        for _ in range(3):
            prepared(torch.ones(8), -7, torch.zeros(8), torch.zeros(8))
    assert len(allocations) == 3 and len(state.tasks) == 6
    for index, (_, incoming, worker) in enumerate(state.tasks):
        assert worker == (7, 13)[index % 2]
        assert incoming.tensor_count() == 2 and incoming.scalar_count() == 3
        assert _decode_scalar(incoming.scalar(0), "int32") == -7
        assert [incoming.scalar(i) for i in (1, 2)] == [
            (1 << 63) + index // 2 + 1,
            (1 << 63) + index // 2 + 1,
        ]


_I1_BOOTSTRAP_PROOF = (
    _RUNTIME_IMPORT_PROOF.split("helper = importlib.import_module", 1)[0]
    + r"""
import ctypes
from contextlib import contextmanager
from types import SimpleNamespace
from unittest import mock
import json
import torch
import simpler.buffer as buffer_api
from simpler.buffer import create_host_shared_buffer, mint_owner_instance_id
from simpler.startup_preparation import NodeStartupContext
from simpler.task_interface import (
    CallConfig, ChipDomainContext, CommDomainHandle, DataType, GlobalCommDomainView,
    GlobalDomainMember, TaskArgs, TensorArgType,
)
from simpler.worker import Worker
bootstrap = importlib.import_module(namespace + '.orchestration.cluster_bootstrap')
support = importlib.import_module(namespace + '.orchestration._distributed_support')
records = json.loads((installed[0].root / 'distributed_meta.json').read_text())['l3_entries']['entries']
paths = {path for record in records for path in record['chip_paths'].values()}
digests = {path: str(index + 1).zfill(64) for index, path in enumerate(sorted(paths))}
handles = {digest: object() for digest in digests.values()}
# Physical IDs are not local worker IDs. The CPU worker below supplies the
# real prefork Tensor conversion; CHIP execution remains a recorded boundary.
inner = SimpleNamespace(_config={'device_ids': [12, 15]}, register=lambda fn: object())
state = bootstrap.prepare_node(NodeStartupContext(
    rank=0, artifact_roots={digest: installed[0].root}, inner_worker=inner, chip_handles=handles,
    parameters={'module_prefix': namespace, 'entries': records, 'chip_digests': digests,
                'chip_configs': {path: {} for path in paths}},
))
entries = [getattr(importlib.import_module(namespace + '.' + r['module']), r['qualname']) for r in records]
node = entries[0]._pypto_embedded_state.node_state
assert node is entries[1]._pypto_embedded_state.node_state
assert node.local_worker_ids == (0, 1)
capacity = len(node._tokens)
assert 1 <= capacity <= sum(row[1] for r in records for row in r['layout']['prefork_requirements']
                            if row[0] == 'inherited_domain_tokens')
token_ptrs = {t.untyped_storage().data_ptr() for bundle in node._tokens for t in bundle}
assert len(token_ptrs) == capacity * 2
local_tokens = [t for entry in entries for spec in entry._pypto_embedded_state._local_specs.values()
                for t in spec[1].values()]
assert not token_ptrs.intersection(t.untyped_storage().data_ptr() for t in local_tokens)
owner = mint_owner_instance_id()
buffers = [create_host_shared_buffer(64, owner, index + 1) for index in range(5)]
worker = Worker(level=3, num_sub_workers=0)
calls, local_domains = [], []
view = None
try:
    worker.init()
    x, out = [buffer.tensor(shapes=(8,), dtype=DataType.FLOAT32) for buffer in buffers[:2]]
    name = records[0]['layout']['window_slots'][0][4]
    members = tuple(GlobalDomainMember(node_worker_id=7 if rank in (4, 9) else 100 + rank,
                    local_worker_id=0 if rank == 4 else 1 if rank == 9 else 0,
                    global_device_rank=rank, domain_rank=rank) for rank in range(12))

    def domain(domain_id, generation):
        result = GlobalCommDomainView(name='received', members=members,
            contexts={local: ChipDomainContext(name='received', domain_rank=rank, domain_size=12,
                device_ctx=80000 + local, local_window_base=0, actual_window_size=64,
                buffers={name: buffers[2 + local]}) for local, rank in ((0, 4), (1, 9))},
            domain_id=domain_id, generation=generation, mapping_size=64)
        result._committed = True
        return result

    @contextmanager
    def allocate_local(**kwargs):
        assert kwargs['workers'] == [1]
        local = CommDomainHandle(name=kwargs['name'], workers=(1,), allocation_id=601,
            contexts={1: ChipDomainContext(name=kwargs['name'], domain_rank=0, domain_size=1,
                device_ctx=90001, local_window_base=0, actual_window_size=64,
                buffers={spec.name: buffers[4] for spec in kwargs['buffers']})},
            _release_fn=lambda handle: None)
        local_domains.append(local)
        with local:
            yield local

    def submit(handle, incoming, config, *, worker):
        assert worker == 1
        calls.append((handle, incoming))

    def resolve(domain_id):
        assert domain_id == view.domain_id
        return view

    orch = SimpleNamespace(_worker=worker, get_global_domain=resolve,
                           allocate_domain=allocate_local, submit_next_level=submit)
    def incoming(domain_id):
        args = TaskArgs()
        args.add_tensor(x, TensorArgType.INPUT)
        args.add_tensor(out, TensorArgType.OUTPUT_EXISTING)
        for value in (ctypes.c_int32(-7), ctypes.c_uint64(domain_id), ctypes.c_uint64(domain_id)):
            args.add_scalar(value)
        return args

    def forbidden(*args, **kwargs):
        raise AssertionError('Storage allocation after prefork preparation')

    with mock.patch.object(torch, 'zeros', forbidden), mock.patch.object(torch, 'empty', forbidden), \
         mock.patch.object(torch.Tensor, 'share_memory_', forbidden), \
         mock.patch.object(buffer_api, 'create_host_shared_buffer', forbidden):
        stable = None
        for generation in range(1, capacity + 5):
            view = domain((1 << 63) + generation // 2, generation)
            if generation == 1:
                for failure in ('uncommitted', 'missing_context'):
                    before = dict(node._bindings)
                    calls.clear()
                    missing = view.contexts.pop(1) if failure == 'missing_context' else None
                    if failure == 'uncommitted':
                        view._committed = False
                    try:
                        entries[0](orch, incoming(view.domain_id), CallConfig())
                    except (ValueError, RuntimeError, KeyError):
                        pass
                    else:
                        raise AssertionError('Invalid received domain reached CHIP submission')
                    finally:
                        view._committed = True
                        if missing is not None:
                            view.contexts[1] = missing
                    assert not calls and node._bindings == before
                    assert entries[0]._pypto_embedded_state._frame is None
            shared = None
            for index, entry in enumerate(entries):
                calls.clear()
                entry(orch, incoming(view.domain_id), CallConfig())
                assert len(calls) == 1
                _, sent = calls[0]
                assert sent.scalar_count() == (3 if index == 0 else 2)
                assert support._decode_scalar(sent.scalar(0), 'int32') == -7
                assert sent.scalar(1) == view[1].device_ctx
                assert sent.scalar(sent.scalar_count() - (2 if index == 0 else 1)) == view[1].device_ctx
                expected = view[1].buffers[name].tensor(shapes=(8,), dtype=DataType.FLOAT32)
                assert sent.tensor(1) == expected
                token_index = 4 if index == 0 else 3
                global_token = sent.tensor(token_index)
                assert sent.tag(token_index) == TensorArgType.INOUT
                if shared is not None:
                    assert global_token == shared, 'Live physical domain changed token between HOST entries'
                shared = global_token
                if index == 0:
                    assert sent.scalar(sent.scalar_count() - 1) == 90001
                    assert sent.tensor(2).buffer != expected.buffer
                    assert sent.tensor(token_index + 1) != global_token
                    assert local_domains[-1].released
                state.task_completed()
                assert entry._pypto_embedded_state._frame is None
            if stable is None:
                stable = shared
            assert len(node._tokens) == capacity
            # R1 invalidates this same view before destructive release fanout.
            # Completion alone must preserve its token; invalidation plus drain
            # allows reuse on the next acquisition without a manual retirement.
            assert (view.domain_id, view.generation) in node._bindings
            view._committed = False
        assert len(node._bindings) <= capacity
    assert not guard.attempts
    print('I1_GLOBAL_BOOTSTRAP_PASS')
finally:
    worker.close()
    state.close()
    for backing in buffers:
        backing.close()
"""
)


def test_i1_global_bootstrap_uses_canonical_prefork_tokens_and_bounded_retirement(
    i1_global_artifact, tmp_path
):
    runtime_python = os.environ.get("L4_RUNTIME_ONLY_PYTHON")
    runtime_pythonpath = os.environ.get("L4_RUNTIME_ONLY_PYTHONPATH")
    if not runtime_python or not runtime_pythonpath:
        pytest.skip("Set L4_RUNTIME_ONLY_PYTHON and L4_RUNTIME_ONLY_PYTHONPATH")
    digest, payload = _artifact_sources.build_cluster_source_bundle(i1_global_artifact.output_dir)
    archive = tmp_path / "global-sources.tar.gz"
    archive.write_bytes(payload)
    result = subprocess.run(
        [
            runtime_python,
            "-c",
            _I1_BOOTSTRAP_PROOF,
            "installed_" + digest,
            digest,
            str(archive),
            str(tmp_path / "installed-global"),
        ],
        cwd=tmp_path,
        env=os.environ | {"PYTHONPATH": runtime_pythonpath, "TORCH_DEVICE_BACKEND_AUTOLOAD": "0"},
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "I1_GLOBAL_BOOTSTRAP_PASS" in result.stdout


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
