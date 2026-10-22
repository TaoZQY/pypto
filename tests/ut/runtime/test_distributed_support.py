# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Runtime-neutral embedded ABI contracts, including real optional CPU wire tests."""

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pypto.language as pl
import pytest
from pypto import backend, codegen, ir, passes
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy, PassManager


@pytest.fixture
def mixed_embedded_source():
    @pl.program
    class Program:
        @pl.function(level=pl.Level.CHIP, role=pl.Role.Orchestrator)
        def chip(
            self,
            x: pl.Tensor[[8], pl.FP32],
            count: pl.Scalar[pl.INT8],
            out: pl.Out[pl.Tensor[[8], pl.FP32]],
            enabled: pl.Scalar[pl.BOOL],
            limit: pl.Scalar[pl.INDEX],
            gain: pl.Scalar[pl.FP32],
        ):
            return 0

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def node(
            self,
            x: pl.Tensor[[8], pl.FP32],
            count: pl.Scalar[pl.INT8],
            out: pl.Out[pl.Tensor[[8], pl.FP32]],
            enabled: pl.Scalar[pl.BOOL],
            limit: pl.Scalar[pl.INDEX],
            gain: pl.Scalar[pl.FP32],
        ):
            self.chip(x, count, out, enabled, limit, gain)

    program = passes.convert_to_ssa()(Program)
    compile(codegen.DistributedCodegen().generate(program), "host_orch.py", "exec")
    source = codegen.DistributedCodegen()._generate_embedded(program, entry_name="node")
    compile(source, "embedded_node.py", "exec")
    assert "def node(orch, args, config):" in source and "_submit_chip(" in source
    return source


@pytest.fixture
def canonical_support_path(mixed_embedded_source):
    # Native generation and its full verification context must succeed before
    # a missing canonical helper can be classified as the intended RED failure.
    assert mixed_embedded_source
    path = Path(__file__).resolve().parents[3] / "python/pypto/runtime/_distributed_support.py"
    return path


def test_c3b_canonical_support_exports_embedded_and_dispatch_contracts(canonical_support_path, monkeypatch):
    assert canonical_support_path.is_file(), "Missing canonical runtime-neutral support source"
    name = "_c3b_standalone_support_contract"
    spec = importlib.util.spec_from_file_location(name, canonical_support_path)
    assert spec is not None and spec.loader is not None
    support = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, support)
    spec.loader.exec_module(support)
    for symbol in (
        "_EmbeddedEntryState",
        "_ChipCallables",
        "add_chip_tensor_arg",
        "_submit_chip",
        "_resolve_chip_worker",
        "_reset_dfx_dispatch_state",
        "_dfx_rank_label",
        "_record_dispatch_program",
        "_decode_scalar",
        "_encode_scalar",
    ):
        assert callable(getattr(support, symbol, None)), f"Missing neutral support contract: {symbol}"
    callables = support._ChipCallables()
    chip = object()
    callables.add("chip", chip, {"host_tensor_indices": [1]})
    assert callables["chip"] is chip and callables.host_tensor_indices == {"chip": frozenset({1})}
    assert support._DISPATCH_PROGRAM_FILE == "dispatch_program.json"


_RUNTIME_ONLY_PROOF = r"""
import ctypes
import gc
import importlib.abc
import sys
import weakref
from types import MappingProxyType, SimpleNamespace

class RejectPyPTO(importlib.abc.MetaPathFinder):
    def __init__(self):
        self.attempts = []

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "pypto" or fullname.startswith("pypto."):
            self.attempts.append(fullname)
            raise ModuleNotFoundError("Forbidden runtime-only import: " + fullname)
        return None

assert not any(name == "pypto" or name.startswith("pypto.") for name in sys.modules)
guard = RejectPyPTO()
sys.meta_path.insert(0, guard)

from wire_proof_namespace import entry as generated
from wire_proof_namespace import _distributed_support as support
from simpler.buffer import TensorTransfer, create_host_shared_buffer, mint_owner_instance_id
from simpler.task_interface import CallConfig, DataType, TaskArgs, TensorArgType, scalar_to_uint64

entry = generated.node
layout = entry._pypto_embedded_layout
assert isinstance(layout, MappingProxyType) and layout["version"] == 1
try:
    layout["version"] = 2
    raise AssertionError("Entry layout was mutable")
except TypeError:
    pass

owner = mint_owner_instance_id()
buffers = [create_host_shared_buffer(32, owner, index + 1) for index in range(2)]
try:
    x, out = [buffer.tensor(shapes=(8,), dtype=DataType.FLOAT32) for buffer in buffers]
    incoming = TaskArgs()
    incoming.add_tensor(x, TensorArgType.INPUT)
    incoming.add_tensor(out, TensorArgType.OUTPUT_EXISTING)
    for value in (ctypes.c_int8(-7), True, ctypes.c_int64(-3), 1.25):
        incoming.add_scalar(value)

    calls = []
    def submit_next_level(callable_id, args, config, *, worker):
        calls.append((callable_id, args, config, worker))

    orch = SimpleNamespace(_worker=object(), submit_next_level=submit_next_level)
    config = CallConfig()
    try:
        entry(orch, incoming, config)
        raise AssertionError("Unbound state was accepted")
    except RuntimeError as error:
        assert "bound runtime state" in str(error)

    chip = object()
    callables = support._ChipCallables()
    chip_contract = {"host_tensor_indices": [0]} if sys.argv[1] == "host_control" else {}
    callables.add("chip", chip, chip_contract)
    state = support._EmbeddedEntryState(
        layout, callables=callables, sub_ids={}, world_size=2,
        prepared_tensors={}, domain_provider=None,
    )
    entry._pypto_embedded_state = state

    if sys.argv[1] == "wire":
        entry(orch, incoming, config)
        assert len(calls) == 1
        handle, outgoing, actual_config, worker = calls[0]
        assert handle is chip and actual_config is config and worker == 0
        assert outgoing.tensor_count() == 2 and outgoing.scalar_count() == 4
        assert outgoing.tensor(0) == x and outgoing.tensor(1) == out
        assert outgoing.tensor(0).buffer == incoming.tensor(0).buffer
        assert outgoing.tensor(1).buffer == incoming.tensor(1).buffer
        assert [outgoing.tag(i) for i in range(2)] == [TensorArgType.INPUT, TensorArgType.OUTPUT_EXISTING]
        assert support._decode_scalar(outgoing.scalar(0), "int8") == -7
        assert [outgoing.scalar(i) for i in range(1, 4)] == [
            scalar_to_uint64(True), scalar_to_uint64(-3), scalar_to_uint64(1.25),
        ]
        for dtype, value in (
            ("int8", ctypes.c_int8(-101)),
            ("int16", ctypes.c_int16(-12345)),
            ("int32", ctypes.c_int32(-123456789)),
            ("int64", ctypes.c_int64(-1234567890123)),
            ("bool", ctypes.c_bool(True)),
            ("index", ctypes.c_int64(-5)),
            ("fp32", ctypes.c_float(1.25)),
            ("fp64", ctypes.c_double(1.0000000000000002)),
        ):
            raw = scalar_to_uint64(value)
            decoded = support._decode_scalar(raw, dtype)
            assert decoded == value.value, (dtype, raw, decoded, value.value)
            assert scalar_to_uint64(support._encode_scalar(decoded, dtype)) == raw
        # FP64 is a neutral wire helper contract; current PyPTO has no legal
        # FP64 formal. Native Python float encoding is deliberately FP32.
        precise = 1.0000000000000002
        assert scalar_to_uint64(support._encode_scalar(precise, "fp64")) == scalar_to_uint64(
            ctypes.c_double(precise)
        )
        assert scalar_to_uint64(precise) != scalar_to_uint64(ctypes.c_double(precise))
        state.complete_invocation()
    elif sys.argv[1] == "host_control":
        entry(orch, incoming, config)
        assert len(calls) == 1
        handle, outgoing, actual_config, worker = calls[0]
        assert handle is chip and actual_config is config and worker == 0
        assert outgoing.tensor(0) == x and outgoing.tensor(1) == out
        # Keep the received HOST descriptor's real provenance and request no
        # staging for control slots. Ordinary output transfer stays unchanged.
        assert outgoing.tensor(0).buffer == incoming.tensor(0).buffer
        assert outgoing.transfer(0) == TensorTransfer.NONE
        default = TaskArgs()
        default.add_tensor(out, TensorArgType.OUTPUT_EXISTING)
        assert outgoing.transfer(1) == default.transfer(0)
        state.complete_invocation()
    else:
        bad_tensor_count = TaskArgs()
        bad_tensor_count.add_tensor(x, TensorArgType.INPUT)
        for value in (ctypes.c_int8(-7), True, ctypes.c_int64(-3), 1.25):
            bad_tensor_count.add_scalar(value)
        bad_scalar_count = TaskArgs()
        bad_scalar_count.add_tensor(x, TensorArgType.INPUT)
        bad_scalar_count.add_tensor(out, TensorArgType.OUTPUT_EXISTING)
        bad_scalar_count.add_scalar(1)
        for bad, pool in ((bad_tensor_count, "tensor"), (bad_scalar_count, "scalar")):
            try:
                state.begin_invocation(orch, bad, config)
                raise AssertionError("Wrong " + pool + " pool count was accepted")
            except ValueError as error:
                assert pool in str(error).lower() and "expected" in str(error).lower()

        frame = state.begin_invocation(orch, incoming, config)
        assert frame.tensors["x__ssa_v0"] == x and frame.tensors["out__ssa_v0"] == out
        assert frame.tensors["count__ssa_v0"] == -7
        assert frame.tensors["enabled__ssa_v0"] is True
        assert frame.tensors["limit__ssa_v0"] == -3 and frame.tensors["gain__ssa_v0"] == 1.25
        class Keepalive:
            pass
        retained = Keepalive()
        reference = weakref.ref(retained)
        frame.keepalive.append(retained)
        del retained, frame
        gc.collect()
        assert reference() is not None
        try:
            state.begin_invocation(orch, incoming, config)
            raise AssertionError("An incomplete invocation was overwritten")
        except RuntimeError as error:
            assert "complet" in str(error).lower()
        state.complete_invocation()
        gc.collect()
        assert reference() is None

        for _ in range(2):
            entry(orch, incoming, config)
            assert orch._pypto_chip_count == 2 and orch._pypto_commless_seq == 1
            assert orch._dfx_dispatch_idx == {}
            assert orch._dfx_chip_names == {id(chip): "chip"}
            state.complete_invocation()
        assert [call[3] for call in calls] == [0, 0]
    state.close()
    assert not guard.attempts
    assert not any(name == "pypto" or name.startswith("pypto.") for name in sys.modules)
    print("C3B_RUNTIME_ONLY_" + sys.argv[1].upper() + "_PASS")
finally:
    for buffer in buffers:
        buffer.close()
"""


@pytest.mark.parametrize("scenario", ["wire", "lifecycle", "host_control"])
def test_c3b_embedded_real_wire_without_pypto(
    scenario, mixed_embedded_source, canonical_support_path, tmp_path
):
    assert canonical_support_path.is_file(), "Missing canonical runtime-neutral support source"
    runtime_python = os.environ.get("L4_RUNTIME_ONLY_PYTHON")
    runtime_pythonpath = os.environ.get("L4_RUNTIME_ONLY_PYTHONPATH")
    if not runtime_python or not runtime_pythonpath:
        pytest.skip("Set L4_RUNTIME_ONLY_PYTHON and L4_RUNTIME_ONLY_PYTHONPATH for real Simpler CPU proof")
    package = tmp_path / "wire_proof_namespace"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "entry.py").write_text(mixed_embedded_source, encoding="utf-8")
    shutil.copyfile(canonical_support_path, package / "_distributed_support.py")
    env = os.environ | {
        "PYTHONPATH": os.pathsep.join((str(tmp_path), runtime_pythonpath)),
        "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
    }
    result = subprocess.run(
        [runtime_python, "-c", _RUNTIME_ONLY_PROOF, scenario],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, f"Runtime-only {scenario} proof failed:\n{result.stdout}\n{result.stderr}"
    assert f"C3B_RUNTIME_ONLY_{scenario.upper()}_PASS" in result.stdout


@pytest.fixture
def normalized_return_copy_program():
    program = pl.parse_program("""
@pl.program
class Program:
    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def node(self, x: pl.Tensor[[2, 2], pl.FP32], count: pl.Scalar[pl.INT32]) -> pl.Tensor[[2, 2], pl.FP32]:
        return x
    @pl.function(level=pl.Level.CLUSTER_0, role=pl.Role.Orchestrator)
    def cluster(self, x: pl.Tensor[[2, 2], pl.FP32], count: pl.Scalar[pl.INT32]):
        result = self.node(x, count, worker=0)
        return 0
""")
    program = passes.normalize_return_order()(passes.convert_to_ssa()(program))
    properties = passes.IRPropertySet()
    properties.insert(passes.IRProperty.ReturnParamsExplicit)
    assert passes.PropertyVerifierRegistry.verify(properties, program) == []
    compile(codegen.DistributedCodegen().generate(program), "return_cluster.py", "exec")
    return program


_RETURN_COPY_CPU_PROOF = r"""
import importlib.abc
import struct
import sys
import time
from types import SimpleNamespace

class RejectPyPTO(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "pypto" or fullname.startswith("pypto."):
            raise ModuleNotFoundError("Forbidden runtime-only import: " + fullname)
        return None

assert not any(name == "pypto" or name.startswith("pypto.") for name in sys.modules)
sys.meta_path.insert(0, RejectPyPTO())

import torch
from return_proof_namespace import entry as generated
from return_proof_namespace import cluster as caller
from return_proof_namespace import _distributed_support as support
from simpler.buffer import (
    AccessMode, BufferCapability, ImportContext, ImportRegistry, MappedArg, MappedArgs,
    create_host_shared_buffer, mint_owner_instance_id,
)
from simpler.comm_endpoints import HOST_CPU
from simpler.task_interface import CallConfig, DataType, TaskArgs, TensorArgType
from simpler.worker import Worker

strided = sys.argv[1] == "strided"
nbytes = 35 if strided else 32
owner = mint_owner_instance_id()
buffers = [create_host_shared_buffer(nbytes, owner, index + 1) for index in range(2)]
buffers.append(create_host_shared_buffer(1, owner, 3))
buffers.append(create_host_shared_buffer(nbytes, owner, 4, access=AccessMode.READ))
buffers.append(create_host_shared_buffer(nbytes, owner, 5, access=AccessMode.WRITE))
worker = None
try:
    buffers[0].shm.buf[:] = bytes(nbytes)
    buffers[1].shm.buf[:] = bytes([0xCD]) * nbytes
    buffers[2].shm.buf[0] = 0
    ready = buffers[2].tensor(shapes=(1,), dtype=DataType.UINT8)
    source = buffers[0].tensor(
        shapes=(2, 2), dtype=DataType.FLOAT32,
        strides=(4, 1) if strided else (2, 1), byte_offset=4 if strided else 0,
    )
    destination = buffers[1].tensor(
        shapes=(2, 2), dtype=DataType.FLOAT32,
        strides=(4, 1) if strided else (2, 1), byte_offset=8 if strided else 0,
    )

    def producer(args):
        # CPU SUB is a stand-in producer for the CHIP result. Dependency
        # inference and final copy execute through the actual runtime engine.
        # This real shared flag is opened only after the generated return
        # schedules its copy. An inline copy would therefore read stale zeros.
        deadline = time.monotonic() + 10
        while args[1].buffer[0] == 0:
            if time.monotonic() > deadline:
                raise RuntimeError("Return-copy graph construction never released the producer")
            time.sleep(0.001)
        arg = args[0]
        extent = 1 + sum((dim - 1) * stride for dim, stride in zip(arg.shapes, arg.strides))
        torch.frombuffer(arg.buffer, dtype=torch.float32, count=extent).as_strided(
            arg.shapes, arg.strides
        ).fill_(7.0)

    worker = Worker(level=3, num_sub_workers=2)
    producer_handle = worker.register(producer)
    copy_handle = worker.register(support._copy_return_tensor)
    state = support._EmbeddedEntryState(
        generated.node._pypto_embedded_layout, callables={}, sub_ids={}, world_size=1,
        prepared_tensors={}, domain_provider=None, return_copy_callable=copy_handle,
    )
    generated.node._pypto_embedded_state = state
    # Registered callable, state and named HOST storage all exist before fork.
    worker.init()

    routed = []
    packed = []
    node_handle = object()
    def io_arg(binding, tensor, direction):
        routed.append((tensor, direction))
        return tensor
    caller._io_arg = io_arg

    def orchestration(orch, args, config):
        producer_args = TaskArgs()
        producer_args.add_tensor(source, TensorArgType.INOUT)
        producer_args.add_tensor(ready, TensorArgType.INPUT)
        orch.submit_sub(producer_handle, producer_args)

        def submit_node(handle, task_args, actual_config, *, worker):
            assert handle is node_handle and actual_config is config and worker == 7
            packed.append(task_args)
            # This adapter bridges the future L4 transport only. Both emitted
            # bodies and their TaskArgs are real; the inner SUB DAG is native.
            generated.node(orch, task_args, config)

        caller.cluster(
            SimpleNamespace(submit_next_level=submit_node), args, config,
            tensors={"x__ssa_v0": source, "count__ssa_v0": 3, "result__ssa_v0": destination},
            callables={"node": [node_handle]}, nodes=[7], node_sizes=[1], node_bindings=[object()],
            _keep=[], _retain=False,
        )
        buffers[2].shm.buf[0] = 1

    worker.run(orchestration, args=None, config=CallConfig())
    state.complete_invocation()
    assert len(packed) == 1
    assert packed[0].tensor_count() == 2 and packed[0].scalar_count() == 1
    assert packed[0].tensor(0) == source and packed[0].tensor(1) == destination
    assert packed[0].scalar(0) == 3
    assert routed == [(source, TensorArgType.INPUT), (destination, TensorArgType.OUTPUT_EXISTING)]
    assert packed[0].tensor(0).buffer == source.buffer
    assert packed[0].tensor(1).buffer == destination.buffer
    # Source and destination views use different offsets. The final partial
    # element in a 35-byte backing must not make torch.frombuffer reject it.
    expected = bytearray([0xCD]) * nbytes
    positions = (2, 3, 6, 7) if strided else (0, 1, 2, 3)
    for index in positions:
        expected[index * 4 : index * 4 + 4] = struct.pack("<f", 7.0)
    assert bytes(buffers[1].shm.buf) == bytes(expected)
    worker.close()
    worker = None

    # Exercise validation through real mapped arguments and capability rights,
    # not invented convenience attributes or synthetic pointer-backed storage.
    registry = ImportRegistry(ImportContext(deployment=HOST_CPU))
    try:
        def mapped(tensor):
            return MappedArg(
                registry.materialize(tensor.buffer), tensor.byte_offset,
                tuple(tensor.shapes), tuple(tensor.strides), tensor.dtype,
            )

        mapped_source, mapped_destination = mapped(source), mapped(destination)
        wrong_dtype = buffers[1].tensor(
            shapes=destination.shapes, dtype=DataType.INT32,
            strides=destination.strides, byte_offset=destination.byte_offset,
        )
        readonly = mapped(buffers[3].tensor(shapes=(2, 2), dtype=DataType.FLOAT32))
        writeonly = mapped(buffers[4].tensor(shapes=(2, 2), dtype=DataType.FLOAT32))
        assert BufferCapability.DIRECT_STORE not in readonly.imported.capabilities
        assert BufferCapability.DIRECT_LOAD not in writeonly.imported.capabilities
        for arguments, diagnostic in (
            ((mapped_source, mapped(wrong_dtype)), ("dtype", "type")),
            ((mapped_source, readonly), ("store", "write")),
            ((writeonly, mapped_destination), ("load", "read")),
        ):
            try:
                support._copy_return_tensor(MappedArgs(list(arguments), ()))
                raise AssertionError("Invalid return-copy dtype or mapped rights were accepted")
            except ValueError as error:
                assert any(word in str(error).lower() for word in diagnostic), str(error)
        assert bytes(buffers[1].shm.buf) == bytes(expected)
    finally:
        registry.close()
    state.close()
    assert not any(name == "pypto" or name.startswith("pypto.") for name in sys.modules)
    print("C3B_RETURN_COPY_CPU_" + sys.argv[1].upper() + "_PASS")
finally:
    if worker is not None:
        worker.close()
    for buffer in buffers:
        buffer.close()
"""


@pytest.mark.parametrize("view", ["contiguous", "strided"])
def test_c3b_return_copy_uses_real_cpu_dag_and_mapped_views(
    view, normalized_return_copy_program, canonical_support_path, tmp_path, monkeypatch
):
    # This check follows normalized, verified native caller generation. Its
    # current missing callable is a feature RED, not an invalid IR fixture.
    name = "_c3b_return_copy_support_contract"
    spec = importlib.util.spec_from_file_location(name, canonical_support_path)
    assert spec is not None and spec.loader is not None
    support = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, support)
    spec.loader.exec_module(support)
    assert callable(getattr(support, "_copy_return_tensor", None)), (
        "Missing canonical return-copy SUB callable"
    )
    runtime_python = os.environ.get("L4_RUNTIME_ONLY_PYTHON")
    runtime_pythonpath = os.environ.get("L4_RUNTIME_ONLY_PYTHONPATH")
    if not runtime_python or not runtime_pythonpath:
        pytest.skip("Set L4_RUNTIME_ONLY_PYTHON and L4_RUNTIME_ONLY_PYTHONPATH for real Simpler CPU proof")
    package = tmp_path / "return_proof_namespace"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    source = codegen.DistributedCodegen()._generate_embedded(
        normalized_return_copy_program, entry_name="node"
    )
    caller = codegen.DistributedCodegen().generate(normalized_return_copy_program)
    (package / "entry.py").write_text(source, encoding="utf-8")
    (package / "cluster.py").write_text(caller, encoding="utf-8")
    shutil.copyfile(canonical_support_path, package / "_distributed_support.py")
    result = subprocess.run(
        [runtime_python, "-c", _RETURN_COPY_CPU_PROOF, view],
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
    assert result.returncode == 0, f"Real CPU return-copy proof failed:\n{result.stdout}\n{result.stderr}"
    assert f"C3B_RETURN_COPY_CPU_{view.upper()}_PASS" in result.stdout


@pytest.fixture
def inherited_domain_program():
    program = pl.parse_program("""
@pl.program
class Domains:
    @pl.function(level=pl.Level.CHIP, role=pl.Role.Orchestrator)
    def chip(self, x: pl.Tensor[[4], pl.FP32], count: pl.Scalar[pl.INT32],
             data: pld.DistributedTensor[[4], pl.FP32], out: pl.Out[pl.Tensor[[4], pl.FP32]],
             data_ctx: pld.CommCtx) -> pl.Tensor[[4], pl.FP32]:
        return out
    @pl.function(level=pl.Level.CHIP, role=pl.Role.Orchestrator)
    def ctx_only(self, x: pl.Tensor[[4], pl.FP32], out: pl.Out[pl.Tensor[[4], pl.FP32]],
                 explicit_ctx: pld.CommCtx):
        return 0
    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def node(self, x: pl.Tensor[[4], pl.FP32], count: pl.Scalar[pl.INT32],
             data: pld.DistributedTensor[[4], pl.FP32], out: pl.Out[pl.Tensor[[4], pl.FP32]],
             data_ctx: pld.CommCtx):
        out = self.chip(x, count, data, out, data_ctx, worker=3)
        self.ctx_only(x, out, data_ctx, worker=3)
        return 0
    @pl.function(level=pl.Level.CLUSTER_0, role=pl.Role.Orchestrator)
    def root(self, x: pl.Tensor[[4], pl.FP32], count: pl.Scalar[pl.INT32],
             out: pl.Out[pl.Tensor[[4], pl.FP32]]):
        buffer = pld.alloc_window_buffer(16)
        data = pld.window(buffer, [4], dtype=pl.FP32)
        original = pld.system.get_comm_ctx(data)
        self.node(x, count, data, out, original, worker=0)
        return 0
""")
    for transform in (
        passes.convert_to_ssa,
        passes.synthesize_allreduce_signals,
        passes.materialize_comm_domain_scopes,
        passes.lower_host_tensor_collectives,
        passes.derive_call_directions,
        passes.materialize_dist_tensor_ctx,
        passes.normalize_return_order,
    ):
        program = transform()(program)
    properties = passes.IRPropertySet()
    for prop in (
        passes.IRProperty.CommDomainScopesMaterialized,
        passes.IRProperty.DistTensorCtxMaterialized,
        passes.IRProperty.TypeChecked,
        passes.IRProperty.ReturnParamsExplicit,
    ):
        properties.insert(prop)
    assert passes.PropertyVerifierRegistry.verify(properties, program) == []
    compile(codegen.DistributedCodegen().generate(program), "inherited_cluster.py", "exec")
    return program


_INHERITED_DOMAIN_PROOF = r"""
import ctypes
import importlib.abc
import sys
from types import SimpleNamespace
from unittest import mock

class RejectPyPTO(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "pypto" or fullname.startswith("pypto."):
            raise ModuleNotFoundError("Forbidden runtime-only import: " + fullname)
        return None

assert not any(name == "pypto" or name.startswith("pypto.") for name in sys.modules)
sys.meta_path.insert(0, RejectPyPTO())

import torch
import simpler.buffer as buffer_api
from simpler.buffer import create_host_shared_buffer, mint_owner_instance_id
from simpler.task_interface import (
    CallConfig, ChipDomainContext, DataType, GlobalCommDomainView, GlobalDomainMember,
    TaskArgs, TensorArgType,
)
from simpler.worker import Worker
from domain_proof_namespace import entry, entry_b
from domain_proof_namespace import _distributed_support as support

scenario = sys.argv[1]
capacity = 2 if scenario == "wire" else 1
# Token coverage is the participating local subset, not the global domain ranks
# or implicit routing topology. Every emitted dispatch explicitly selects ID3.
tokens = tuple(
    tuple(torch.zeros((1,), dtype=torch.int32).share_memory_() for _ in (1, 3))
    for _ in range(capacity)
)
node = support._EmbeddedNodeState(local_worker_ids=(1, 3), prepared_domain_tokens=tokens)
owner = mint_owner_instance_id()
backings = [create_host_shared_buffer(16, owner, index + 1) for index in range(6)]
worker = None
try:
    x = backings[0].tensor(shapes=(4,), dtype=DataType.FLOAT32)
    out = backings[1].tensor(shapes=(4,), dtype=DataType.FLOAT32)
    window_name = entry.node._pypto_embedded_layout["window_slots"][0][4]
    members = tuple(
        GlobalDomainMember(
            node_worker_id=7 if rank in (4, 9) else 100 + rank,
            local_worker_id=1 if rank == 4 else 3 if rank == 9 else 0,
            global_device_rank=rank, domain_rank=rank,
        )
        for rank in range(12)
    )

    def domain_view(domain_id, generation):
        first = 2 if domain_id == 801 else 4
        contexts = {
            local_id: ChipDomainContext(
                name="received", domain_rank=rank, domain_size=12,
                device_ctx=domain_id * 100 + local_id, local_window_base=0,
                actual_window_size=16, buffers={window_name: backings[first + index]},
            )
            for index, (local_id, rank) in enumerate(((1, 4), (3, 9)))
        }
        view = GlobalCommDomainView(
            name="received", members=members, contexts=contexts,
            domain_id=domain_id, generation=generation, mapping_size=16,
        )
        # This is the recorded global-import/commit boundary. The actual class
        # enforces its real API; named leaf buffers are genuine HOST POSIX
        # descriptors. No physical VMM domain or CHIP execution is claimed.
        view._committed = True
        return view

    domains = {801: domain_view(801, 5), 802: domain_view(802, 5)}
    chip, ctx_only = object(), object()
    callables = support._ChipCallables()
    callables.add("chip", chip, {})
    callables.add("ctx_only", ctx_only, {})
    state_a = support._EmbeddedEntryState(
        entry.node._pypto_embedded_layout, callables=callables, sub_ids={}, world_size=4,
        prepared_tensors={}, domain_provider=None, node_state=node,
    )
    state_b = support._EmbeddedEntryState(
        entry_b.node._pypto_embedded_layout, callables=callables, sub_ids={}, world_size=4,
        prepared_tensors={}, domain_provider=None, node_state=node,
    )
    entry.node._pypto_embedded_state = state_a
    entry_b.node._pypto_embedded_state = state_b
    # CPU Worker supplies the real Torch-prefork converter and identities.
    # Its CHIP submission boundary is recorded rather than executed.
    worker = Worker(level=3, num_sub_workers=0)
    worker.init()
    calls, resolutions = [], []

    def resolve(domain_id):
        resolutions.append(domain_id)
        return domains[domain_id]

    def submit(handle, args, config, *, worker):
        assert worker == 3 and handle in (chip, ctx_only)
        calls.append((handle, args))

    orch = SimpleNamespace(_worker=worker, get_global_domain=resolve, submit_next_level=submit)

    def incoming(prefix=801, data=801):
        args = TaskArgs()
        args.add_tensor(x, TensorArgType.INPUT)
        args.add_tensor(out, TensorArgType.OUTPUT_EXISTING)
        args.add_scalar(ctypes.c_int32(9))
        args.add_scalar(prefix)
        args.add_scalar(data)
        return args

    def invoke(function, *, prefix=801):
        calls.clear()
        resolutions.clear()
        function(orch, incoming(prefix), CallConfig())
        assert [handle for handle, _ in calls] == [chip, ctx_only]
        custom, scalar_only = (args for _, args in calls)
        assert custom.tensor(0) == x and custom.tensor(2) == out
        expected_window = domains[801][3].buffers[window_name].tensor(shapes=(4,), dtype=DataType.FLOAT32)
        assert custom.tensor(1) == expected_window
        assert custom.tensor(1).buffer == expected_window.buffer
        assert [custom.scalar(i) for i in range(custom.scalar_count())] == [
            9, domains[prefix][3].device_ctx, domains[801][3].device_ctx,
        ]
        assert scalar_only.scalar_count() == 1 and scalar_only.scalar(0) == domains[prefix][3].device_ctx
        assert scalar_only.tensor(0) == x and scalar_only.tensor(1) == out
        assert scalar_only.tag(2) == TensorArgType.INOUT
        # First reference is DT-domain801, followed by explicit-ctx domain802.
        # The ctx-only call must join that exact second domain's token chain.
        assert custom.tensor(custom.tensor_count() - 1) == scalar_only.tensor(2)
        assert all(custom.tag(i) == TensorArgType.INOUT for i in range(3, custom.tensor_count()))
        assert set(resolutions) == {801, prefix} and len(resolutions) == len({801, prefix})
        return custom, scalar_only

    def forbidden(*args, **kwargs):
        raise AssertionError("Invocation allocated/shared storage after prefork preparation")

    with mock.patch.object(torch, "zeros", forbidden), mock.patch.object(torch, "empty", forbidden), \
         mock.patch.object(torch.Tensor, "share_memory_", forbidden), \
         mock.patch.object(buffer_api, "create_host_shared_buffer", forbidden):
        if scenario == "wire":
            custom, scalar_only = invoke(entry.node, prefix=802)
            assert custom.tensor_count() == 5 and scalar_only.tensor_count() == 3
            # Same WB hint in two received domains must still select domain801
            # for the DT and domain802 for the original, unrelated ctx prefix.
            assert custom.tensor(1).buffer != domains[802][3].buffers[window_name].to_descriptor()
            expected = [support.make_tensor_arg(worker, bundle[1]) for bundle in tokens]
            sent = [custom.tensor(3), custom.tensor(4)]
            assert all(any(actual == value for actual in sent) for value in expected)
            wrong_worker = [support.make_tensor_arg(worker, bundle[0]) for bundle in tokens]
            assert all(all(actual != value for actual in sent) for value in wrong_worker)
            state_a.complete_invocation()
            node.retire_domain(801, 5)
            node.retire_domain(802, 5)
        else:
            # Every failure is preflight-atomic: no CHIP, frame or stranded
            # lease may survive; a corrected invocation immediately retries.
            failures = ("unknown", "capacity", "uncommitted", "context", "slot")
            for failure in failures:
                calls.clear()
                missing = None
                prefix = 999 if failure == "unknown" else 802 if failure == "capacity" else 801
                if failure == "uncommitted":
                    domains[801]._committed = False
                elif failure == "context":
                    missing = domains[801].contexts.pop(3)
                elif failure == "slot":
                    missing = domains[801].contexts[3].buffers.pop(window_name)
                try:
                    entry.node(orch, incoming(prefix), CallConfig())
                    raise AssertionError("Invalid inherited-domain preflight was accepted: " + failure)
                except (ValueError, RuntimeError, KeyError):
                    pass
                finally:
                    domains[801]._committed = True
                    if failure == "context":
                        domains[801].contexts[3] = missing
                    elif failure == "slot":
                        domains[801].contexts[3].buffers[window_name] = missing
                assert not calls and state_a._frame is None
                valid, _ = invoke(entry.node)
                assert valid.tensor_count() == 4
                state_a.complete_invocation()
                node.retire_domain(801, 5)

            first, _ = invoke(entry.node)
            second, _ = invoke(entry_b.node)
            assert first.tensor(3) == second.tensor(3)
            for completed in (None, state_a):
                if completed is not None:
                    completed.complete_invocation()
                try:
                    node.retire_domain(801, 5)
                    raise AssertionError("A domain still leased by another entry frame was retired")
                except RuntimeError as error:
                    assert any(word in str(error).lower() for word in ("active", "drain", "complet"))
            state_b.complete_invocation()
            node.retire_domain(801, 5)
            # One bundle supports arbitrarily many sequential, explicitly
            # retired generations without growing a module/global cache.
            for generation in (6, 7, 8):
                domains[801] = domain_view(801, generation)
                current, _ = invoke(entry.node)
                assert current.tensor(3) == first.tensor(3)
                state_a.complete_invocation()
                node.retire_domain(801, generation)
    state_a.close()
    state_b.close()
    node.close()
    worker.close()
    worker = None
    assert not any(name == "pypto" or name.startswith("pypto.") for name in sys.modules)
    print("C3B_INHERITED_DOMAIN_" + scenario.upper() + "_PASS")
finally:
    if worker is not None:
        worker.close()
    for backing in backings:
        backing.close()
"""


@pytest.mark.parametrize("scenario", ["wire", "leases"])
def test_c3b_inherited_domain_real_wire_and_prefork_lease_contract(
    scenario, inherited_domain_program, canonical_support_path, tmp_path, monkeypatch
):
    name = "_c3b_node_state_contract"
    spec = importlib.util.spec_from_file_location(name, canonical_support_path)
    assert spec is not None and spec.loader is not None
    support = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, support)
    spec.loader.exec_module(support)
    assert callable(getattr(support, "_EmbeddedNodeState", None)), (
        "Missing canonical prefork node/domain token state"
    )
    runtime_python = os.environ.get("L4_RUNTIME_ONLY_PYTHON")
    runtime_pythonpath = os.environ.get("L4_RUNTIME_ONLY_PYTHONPATH")
    if not runtime_python or not runtime_pythonpath:
        pytest.skip("Set L4_RUNTIME_ONLY_PYTHON and L4_RUNTIME_ONLY_PYTHONPATH for real Simpler CPU proof")
    source = codegen.DistributedCodegen()._generate_embedded(inherited_domain_program, entry_name="node")
    package = tmp_path / "domain_proof_namespace"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    for module in ("entry.py", "entry_b.py"):
        (package / module).write_text(source, encoding="utf-8")
    shutil.copyfile(canonical_support_path, package / "_distributed_support.py")
    result = subprocess.run(
        [runtime_python, "-c", _INHERITED_DOMAIN_PROOF, scenario],
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
    assert result.returncode == 0, (
        f"Inherited-domain {scenario} proof failed:\n{result.stdout}\n{result.stderr}"
    )
    assert f"C3B_INHERITED_DOMAIN_{scenario.upper()}_PASS" in result.stdout


@pytest.fixture(params=["bare", "affine", "view", "reshape"])
def descriptor_program(request):
    mode = request.param
    dynamic = mode in {"bare", "affine"}
    dim = "M * 2" if mode == "affine" else "M"
    shape = f"[2, {dim}, 4]" if dynamic else "[2, 3, 4]"
    chip_shape = "[M, 4]" if dynamic else "[6, 4]"
    body = (
        "        viewed = pl.tensor.slice(x, [1, M, 4], [1, 0, 0], drop_dims=[0])\n"
        "        target = pl.tensor.slice(out, [1, M, 4], [1, 0, 0], drop_dims=[0])\n"
        if dynamic
        else f"        viewed = pl.tensor.{mode}(x, [6, 4])\n        target = pl.tensor.{mode}(out, [6, 4])\n"
    )
    program = pl.parse_program(
        ('M = pl.dynamic("M")\n' if dynamic else "") + "@pl.program\nclass Descriptors:\n"
        "    @pl.function(level=pl.Level.CHIP, role=pl.Role.Orchestrator)\n"
        f"    def chip(self, x: pl.Tensor[{chip_shape}, pl.FP32],\n"
        f"             out: pl.Out[pl.Tensor[{chip_shape}, pl.FP32]]):\n"
        "        return 0\n"
        "    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)\n"
        f"    def node(self, x: pl.Tensor[{shape}, pl.FP32],\n"
        f"             out: pl.Out[pl.Tensor[{shape}, pl.FP32]]):\n"
        + body
        + "        self.chip(viewed, target, worker=0)\n"
        "        return 0\n"
        "    @pl.function(level=pl.Level.CLUSTER_0, role=pl.Role.Orchestrator)\n"
        f"    def root(self, x: pl.Tensor[{shape}, pl.FP32],\n"
        f"             out: pl.Out[pl.Tensor[{shape}, pl.FP32]]):\n"
        "        self.node(x, out, worker=0)\n"
        "        return 0\n"
    )
    program = passes.normalize_return_order()(passes.convert_to_ssa()(program))
    properties = passes.IRPropertySet()
    for prop in (passes.IRProperty.TypeChecked, passes.IRProperty.ReturnParamsExplicit):
        properties.insert(prop)
    assert passes.PropertyVerifierRegistry.verify(properties, program) == []
    compile(codegen.DistributedCodegen().generate(program), "descriptor_cluster.py", "exec")
    return mode, program


_DESCRIPTOR_PROOF = r"""
import importlib.abc
import sys
from contextlib import ExitStack
from types import SimpleNamespace
from unittest import mock

class RejectPyPTO(importlib.abc.MetaPathFinder):
    def __init__(self):
        self.attempts = []

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "pypto" or fullname.startswith("pypto."):
            self.attempts.append(fullname)
            raise ModuleNotFoundError("Forbidden runtime-only import: " + fullname)
        return None

assert not any(name == "pypto" or name.startswith("pypto.") for name in sys.modules)
guard = RejectPyPTO()
sys.meta_path.insert(0, guard)

import torch
from descriptor_proof_namespace import entry as generated
from descriptor_proof_namespace import _distributed_support as support
from simpler.buffer import Tensor, create_host_shared_buffer, mint_owner_instance_id
from simpler.task_interface import CallConfig, DataType, TaskArgs, TensorArgType

mode = sys.argv[1]
dynamic = mode in ("bare", "affine")
owner = mint_owner_instance_id()
buffers = [create_host_shared_buffer(640, owner, index + 1) for index in range(2)]
try:
    extent = 6 if mode == "affine" else 3
    shape = (2, extent, 4)
    strides = ((40, 5, 1), (48, 6, 1)) if dynamic else ((12, 4, 1), (12, 4, 1))
    offsets = (4, 12)
    # Exercise both native constructor inputs. This is real HOST POSIX backing,
    # never a fabricated DEVICE descriptor or a new mapping of a raw pointer.
    x = Tensor(buffers[0], shapes=shape, dtype=DataType.FLOAT32,
               strides=strides[0], byte_offset=offsets[0])
    out = Tensor(buffers[1].to_descriptor(), shapes=shape, dtype=DataType.FLOAT32,
                 strides=strides[1], byte_offset=offsets[1])
    assert not hasattr(x, "shape") and not hasattr(type(x), "__getitem__")
    assert x.buffer == buffers[0].to_descriptor() and out.buffer == buffers[1].to_descriptor()
    try:
        Tensor(x.buffer, shapes=(2, 3, 4), dtype=DataType.FLOAT32,
               strides=(40, 5, 1), byte_offset=636)
        raise AssertionError("Native descriptor bounds were not checked")
    except ValueError:
        pass

    def arguments(source=x, target=out):
        args = TaskArgs()
        args.add_tensor(source, TensorArgType.INPUT)
        args.add_tensor(target, TensorArgType.OUTPUT_EXISTING)
        return args

    calls = []
    def submit_next_level(handle, args, config, *, worker):
        # This records the CHIP transport boundary with actual TaskArgs. It
        # does not claim execution of a CHIP worker or any physical device.
        calls.append((handle, args, config, worker))

    orch = SimpleNamespace(_worker=object(), submit_next_level=submit_next_level)
    config = CallConfig()
    chip = object()
    callables = support._ChipCallables()
    callables.add("chip", chip, {})
    state = support._EmbeddedEntryState(
        generated.node._pypto_embedded_layout, callables=callables, sub_ids={}, world_size=1,
        prepared_tensors={}, domain_provider=None,
    )
    generated.node._pypto_embedded_state = state
    incoming = arguments()

    def invoke(args):
        # Descriptor construction may allocate Python metadata, but storage and
        # shared backing must already exist before the first invocation.
        def forbidden(*args, **kwargs):
            raise AssertionError("Storage allocation after prefork preparation")
        with ExitStack() as stack:
            for name in ("zeros", "empty"):
                stack.enter_context(mock.patch.object(torch, name, side_effect=forbidden))
            stack.enter_context(mock.patch.object(torch.Tensor, "share_memory_", side_effect=forbidden))
            stack.enter_context(mock.patch("simpler.buffer.create_host_shared_buffer", side_effect=forbidden))
            generated.node(orch, args, config)

    invoke(incoming)
    assert len(calls) == 1
    handle, outgoing, actual_config, worker = calls[0]
    assert handle is chip and actual_config is config and worker == 0
    assert outgoing.tensor_count() == 2 and outgoing.scalar_count() == 0
    assert [outgoing.tag(i) for i in range(2)] == [TensorArgType.INPUT, TensorArgType.OUTPUT_EXISTING]
    expected_shape = (3, 4) if dynamic else (6, 4)
    expected_strides = ((5, 1), (6, 1)) if dynamic else ((4, 1), (4, 1))
    for index, original in enumerate((x, out)):
        descriptor = outgoing.tensor(index)
        assert tuple(descriptor.shapes) == expected_shape
        assert tuple(descriptor.strides) == expected_strides[index]
        assert descriptor.dtype == DataType.FLOAT32.value
        assert descriptor.byte_offset == offsets[index] + (strides[index][0] * 4 if dynamic else 0)
        # Equality covers canonical identity, address space, rights, backend and
        # business payload. A view must not forge a fresh FORK_SHM provenance.
        assert descriptor.buffer == original.buffer
        assert original == incoming.tensor(index)
    assert state._frame is not None and any(value is incoming for value in state._frame.keepalive)
    assert any(value is outgoing for value in state._frame.keepalive)
    state.complete_invocation()
    assert state._frame is None

    calls.clear()
    if dynamic:
        # A valid native descriptor with a wrong fixed axis is invalid for the
        # formal ABI; failure must not publish a frame or dispatch anything.
        bad = Tensor(out.buffer, shapes=(2, extent, 3), dtype=DataType.FLOAT32,
                     strides=strides[1], byte_offset=offsets[1])
        try:
            invoke(arguments(target=bad))
            raise AssertionError("Wrong static formal dimension was accepted")
        except ValueError as error:
            assert "shape" in str(error).lower() or "dimension" in str(error).lower()
        assert not calls and state._frame is None
    else:
        # The IR requests a valid logical reshape; the received backing is
        # noncontiguous, so shape-only reinterpretation would change its cells.
        bad = Tensor(x.buffer, shapes=shape, dtype=DataType.FLOAT32,
                     strides=(40, 5, 1), byte_offset=offsets[0])
        try:
            invoke(arguments(source=bad))
            raise AssertionError("Non-viewable descriptor reshape was accepted")
        except ValueError as error:
            message = str(error).lower()
            assert "contiguous" in message or "stride" in message or "view" in message
        assert not calls
        # Runtime completion/abort owns invocation cleanup after a body error.
        state.complete_invocation()
    invoke(incoming)
    assert len(calls) == 1
    state.complete_invocation()
    state.close()
    assert not guard.attempts
    assert not any(name == "pypto" or name.startswith("pypto.") for name in sys.modules)
    print("C3B_DESCRIPTOR_" + mode.upper() + "_PASS")
finally:
    for buffer in buffers:
        buffer.close()
"""


def test_c3b_descriptor_real_shapes_views_and_provenance_without_pypto(
    descriptor_program, canonical_support_path, tmp_path
):
    mode, program = descriptor_program
    source = codegen.DistributedCodegen()._generate_embedded(program, entry_name="node")
    compile(source, "descriptor_entry.py", "exec")
    runtime_python = os.environ.get("L4_RUNTIME_ONLY_PYTHON")
    runtime_pythonpath = os.environ.get("L4_RUNTIME_ONLY_PYTHONPATH")
    if not runtime_python or not runtime_pythonpath:
        pytest.skip("Set L4_RUNTIME_ONLY_PYTHON and L4_RUNTIME_ONLY_PYTHONPATH for real Simpler CPU proof")
    package = tmp_path / "descriptor_proof_namespace"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "entry.py").write_text(source, encoding="utf-8")
    shutil.copyfile(canonical_support_path, package / "_distributed_support.py")
    result = subprocess.run(
        [runtime_python, "-c", _DESCRIPTOR_PROOF, mode],
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
    assert result.returncode == 0, f"Descriptor {mode} proof failed:\n{result.stdout}\n{result.stderr}"
    assert f"C3B_DESCRIPTOR_{mode.upper()}_PASS" in result.stdout


@pytest.fixture(params=["geometry", "metadata_call", "metadata_submit", "ctx_tuple", "default_window"])
def dt_view_program(request):
    mode = request.param
    if mode == "default_window":
        program = pl.parse_program("""
@pl.program
class DefaultWindow:
    @pl.function(level=pl.Level.CHIP, role=pl.Role.Orchestrator)
    def chip(self, data: pld.DistributedTensor[[4], pl.FP32]):
        return 0
    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def node(self, data: pld.DistributedTensor[[4], pl.FP32]):
        selected = data
        self.chip(selected, worker=1)
        return 0
    @pl.function(level=pl.Level.CLUSTER_0, role=pl.Role.Orchestrator)
    def root(self):
        storage = pld.alloc_window_buffer(16)
        data = pld.window(storage, [4], dtype=pl.FP32)
        self.node(data, worker=0)
        return 0
""")
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910)
        try:
            program = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(program)
        finally:
            backend.reset_for_testing()
    elif mode == "ctx_tuple":
        program = pl.parse_program("""
@pl.program
class ContextTuple:
    @pl.function(level=pl.Level.CHIP, role=pl.Role.Orchestrator)
    def sink(self, data: pld.DistributedTensor[[4], pl.FP32], ctx: pld.CommCtx):
        return 0
    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def node(self, left: pld.DistributedTensor[[4], pl.FP32],
             right: pld.DistributedTensor[[4], pl.FP32], explicit: pld.CommCtx):
        packed = (left, right, explicit)
        alias = packed
        chosen_ctx = alias[2]
        self.sink(left, chosen_ctx, worker=1)
        return 0
    @pl.function(level=pl.Level.CLUSTER_0, role=pl.Role.Orchestrator)
    def root(self):
        left_storage = pld.alloc_window_buffer(16)
        left = pld.window(left_storage, [4], dtype=pl.FP32)
        right_storage = pld.alloc_window_buffer(16)
        right = pld.window(right_storage, [4], dtype=pl.FP32)
        explicit = pld.system.get_comm_ctx(right)
        self.node(left, right, explicit, worker=0)
        return 0
""")
        program = passes.convert_to_ssa()(program)
    elif mode == "geometry":
        program = pl.parse_program("""
@pl.program
class Geometry:
    @pl.function(level=pl.Level.CHIP, role=pl.Role.Orchestrator)
    def chip(self, left: pld.DistributedTensor[[2, 3], pl.FP32],
             right: pld.DistributedTensor[[2, 3], pl.FP32]):
        return 0
    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def node(self, left: pld.DistributedTensor[[2, 3], pl.FP32],
             right: pld.DistributedTensor[[2, 3], pl.FP32]):
        self.chip(left, right, worker=1)
        return 0
    @pl.function(level=pl.Level.CLUSTER_0, role=pl.Role.Orchestrator)
    def root(self):
        storage = pld.alloc_window_buffer(128)
        data = pld.window(storage, [2, 3], dtype=pl.FP32)
        self.node(data, data, worker=0)
        return 0
""")
        program = passes.convert_to_ssa()(program)
        root = program.functions[program.get_global_var("root")]
        base = root.body.stmts[0].var
        replacements = {}
        for name in ("node", "chip"):
            function = program.functions[program.get_global_var(name)]
            params = [
                ir.Var(
                    old.name_hint,
                    ir.DistributedTensorType(
                        [2, 3],
                        pl.FP32,
                        ir.MemRef(base, offset, 44, old.span),
                        ir.TensorView(stride=[8, 1], layout=ir.TensorLayout.ND),
                    ),
                    old.span,
                )
                for old, offset in zip(function.params, (36, 68))
            ]
            body = function.body
            if name == "node":
                previous = body.stmts[0].expr
                launch = ir.Call(
                    previous.op, params, previous.kwargs, previous.attrs, previous.type, previous.span
                )
                body = ir.SeqStmts([ir.EvalStmt(launch, previous.span), body.stmts[-1]], function.span)
            replacements[name] = ir.Function(
                name,
                list(zip(params, function.param_directions)),
                function.return_types,
                body,
                function.span,
                type=function.func_type,
                level=function.level,
                role=function.role,
                attrs=function.attrs,
            )
        program = ir.Program(
            [replacements.get(function.name, function) for function in program.functions.values()],
            program.name,
            program.span,
        )
    else:
        submit = mode == "metadata_submit"
        launch = (
            "        (returned_right, returned_left, repeated_right), task = "
            "pl.submit(self.stage, left, right, explicit, worker=1)\n"
            "        packed = (returned_right, returned_left, repeated_right)\n"
            if submit
            else "        packed = self.stage(left, right, explicit, worker=1)\n"
        )
        consume = (
            "        unused, next_task = pl.submit(self.sink, selected, ctx, worker=1, deps=[task])\n"
            if submit
            else "        self.sink(selected, ctx, worker=1)\n"
        )
        program = pl.parse_program(
            "@pl.program\nclass Metadata:\n"
            "    @pl.function(level=pl.Level.CHIP, role=pl.Role.Orchestrator)\n"
            "    def stage(self, left: pld.DistributedTensor[[4], pl.FP32],\n"
            "              right: pld.DistributedTensor[[4], pl.FP32], explicit: pld.CommCtx) -> "
            "tuple[pld.DistributedTensor[[4], pl.FP32], pld.DistributedTensor[[4], pl.FP32], "
            "pld.DistributedTensor[[4], pl.FP32]]:\n"
            "        return right, left, right\n"
            "    @pl.function(level=pl.Level.CHIP, role=pl.Role.Orchestrator)\n"
            "    def sink(self, data: pld.DistributedTensor[[4], pl.FP32], ctx: pld.CommCtx) -> "
            "pld.DistributedTensor[[4], pl.FP32]:\n"
            "        return data\n"
            "    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)\n"
            "    def node(self, left: pld.DistributedTensor[[4], pl.FP32],\n"
            "             right: pld.DistributedTensor[[4], pl.FP32], explicit: pld.CommCtx):\n"
            + launch
            + "        alias = packed\n        selected = alias[2]\n"
            "        ctx = pld.system.get_comm_ctx(selected)\n" + consume + "        return 0\n"
            "    @pl.function(level=pl.Level.CLUSTER_0, role=pl.Role.Orchestrator)\n"
            "    def root(self):\n"
            "        left_storage = pld.alloc_window_buffer(16)\n"
            "        left = pld.window(left_storage, [4], dtype=pl.FP32)\n"
            "        right_storage = pld.alloc_window_buffer(16)\n"
            "        right = pld.window(right_storage, [4], dtype=pl.FP32)\n"
            "        explicit = pld.system.get_comm_ctx(left)\n"
            "        self.node(left, right, explicit, worker=0)\n        return 0\n"
        )
        program = passes.convert_to_ssa()(program)
    transforms = (
        ()
        if mode == "default_window"
        else (
            passes.synthesize_allreduce_signals,
            passes.materialize_comm_domain_scopes,
            passes.lower_host_tensor_collectives,
            passes.derive_call_directions,
            passes.materialize_dist_tensor_ctx,
            passes.normalize_return_order,
        )
    )
    for transform in transforms:
        program = transform()(program)
    properties = passes.IRPropertySet()
    for prop in (
        passes.IRProperty.CommDomainScopesMaterialized,
        passes.IRProperty.DistTensorCtxMaterialized,
        passes.IRProperty.TypeChecked,
        passes.IRProperty.ReturnParamsExplicit,
    ):
        properties.insert(prop)
    assert passes.PropertyVerifierRegistry.verify(properties, program) == []
    ir.assert_structural_equal(program, pl.parse_program(ir.python_print(program, format=False)))
    ir.assert_structural_equal(program, ir.deserialize(ir.serialize(program)))
    compile(codegen.DistributedCodegen().generate(program), "dt_view_cluster.py", "exec")
    return mode, program


_DT_VIEW_PROOF = r"""
import importlib.abc
import sys
from contextlib import ExitStack
from types import SimpleNamespace
from unittest import mock

class RejectPyPTO(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "pypto" or fullname.startswith("pypto."):
            raise ModuleNotFoundError("Forbidden runtime-only import: " + fullname)
        return None

assert not any(name == "pypto" or name.startswith("pypto.") for name in sys.modules)
sys.meta_path.insert(0, RejectPyPTO())
import torch
from dt_view_proof_namespace import entry
from dt_view_proof_namespace import _distributed_support as support
from simpler.buffer import create_host_shared_buffer, mint_owner_instance_id
from simpler.task_interface import (
    CallConfig, ChipDomainContext, DataType, GlobalCommDomainView,
    GlobalDomainMember, TaskArgs, TaskHandle, TensorArgType,
)
from simpler.worker import Worker

mode = sys.argv[1]
geometry = mode == "geometry"
plain = mode == "default_window"
context_tuple = mode == "ctx_tuple"
capacity = 1 if geometry or plain else 2
tokens = tuple((torch.zeros((1,), dtype=torch.int32).share_memory_(),) for _ in range(capacity))
node = support._EmbeddedNodeState(local_worker_ids=(1,), prepared_domain_tokens=tokens)
owner = mint_owner_instance_id()
backings = [create_host_shared_buffer(128 if geometry else 16, owner, index + 1)
            for index in range(capacity)]
worker = None
try:
    rows = entry.node._pypto_embedded_layout["window_slots"]
    members = tuple(GlobalDomainMember(node_worker_id=7 if rank == 9 else 100 + rank,
                                      local_worker_id=1 if rank == 9 else 0,
                                      global_device_rank=rank, domain_rank=rank) for rank in range(12))
    domains = {}
    for index, backing in enumerate(backings):
        names = {row[4] for row in rows} if geometry or plain else {rows[index][4]}
        context = ChipDomainContext(name="received", domain_rank=9, domain_size=12,
                                    device_ctx=12001 + index, local_window_base=0,
                                    actual_window_size=128 if geometry else 16,
                                    buffers={name: backing for name in names})
        view = GlobalCommDomainView(name="received", members=members, contexts={1: context},
                                    domain_id=801 + index, generation=5,
                                    mapping_size=128 if geometry else 16)
        # Actual API/commit boundary with real HOST POSIX descriptors. There is
        # no physical VMM domain, CHIP worker or NPU in this CPU proof.
        view._committed = True
        domains[801 + index] = view

    def no_op(mapped_args):
        pass

    if mode == "metadata_submit":
        child = Worker(level=3, num_sub_workers=1)
        no_op_handle = child.register(no_op)
        def child_orchestration(orch, args, config):
            orch.submit_sub(no_op_handle, args)
        worker = Worker(level=4, num_sub_workers=0)
        next_handle = worker.register(child_orchestration)
        child_id = worker.add_worker(child)
    else:
        worker = Worker(level=3, num_sub_workers=2)
        no_op_handle = worker.register(no_op)
    names = ("chip",) if geometry or plain else ("sink",) if context_tuple else ("stage", "sink")
    handles = {name: object() for name in names}
    callables = support._ChipCallables()
    for name, handle in handles.items():
        callables.add(name, handle, {})
    state = support._EmbeddedEntryState(entry.node._pypto_embedded_layout, callables=callables,
                                       sub_ids={}, world_size=2, prepared_tensors={},
                                       domain_provider=None, node_state=node)
    entry.node._pypto_embedded_state = state
    worker.init()
    incoming = TaskArgs()
    received = ((801,) if plain else (801, 801) if geometry
                else (802, 801, 802) if context_tuple else (801, 801, 802))
    for domain_id in received:
        incoming.add_scalar(domain_id)
    calls = []

    def orchestration(orch, args, config):
        def submit(handle, task_args, actual_config, *, worker):
            assert actual_config is config and worker == 1
            calls.append((handle, task_args))
            if mode == "metadata_submit":
                # Actual CPU NEXT_LEVEL produces native TaskHandles. Logical
                # CHIP worker1 maps to the local L3 child; no device executes.
                result = orch.submit_next_level(next_handle, task_args, actual_config, worker=child_id)
                assert isinstance(result, TaskHandle)
                return result
            # Non-Submit source ignores this real CPU SUB's None return value.
            orch.submit_sub(no_op_handle, task_args)

        adapter = SimpleNamespace(_worker=orch._worker, submit_next_level=submit,
                                  get_global_domain=lambda domain_id: domains[domain_id])
        def forbidden(*args, **kwargs):
            raise AssertionError("First invocation allocated storage")
        with ExitStack() as stack:
            for name in ("zeros", "empty"):
                stack.enter_context(mock.patch.object(torch, name, side_effect=forbidden))
            stack.enter_context(mock.patch.object(torch.Tensor, "share_memory_", side_effect=forbidden))
            stack.enter_context(mock.patch("simpler.buffer.create_host_shared_buffer", side_effect=forbidden))
            entry.node(adapter, incoming, config)
        assert state._frame is not None
        assert any(value is incoming for value in state._frame.keepalive)
        assert all(any(value is args for value in state._frame.keepalive) for _, args in calls)

    worker.run(orchestration, args=None, config=CallConfig())
    if geometry:
        assert [handle for handle, _ in calls] == [handles["chip"]]
        args = calls[0][1]
        assert args.tensor_count() == 3 and args.scalar_count() == 2
        for index, offset in enumerate((36, 68)):
            actual = args.tensor(index)
            assert tuple(actual.shapes) == (2, 3) and tuple(actual.strides) == (8, 1)
            assert actual.byte_offset == offset and actual.dtype == DataType.FLOAT32.value
            assert actual.buffer == backings[0].to_descriptor()
        assert args.tensor(2) == support.make_tensor_arg(worker, tokens[0][0])
        assert args.tag(2) == TensorArgType.INOUT
        assert [args.scalar(i) for i in range(2)] == [12001, 12001]
    elif plain:
        assert [handle for handle, _ in calls] == [handles["chip"]]
        args = calls[0][1]
        assert args.tensor_count() == 2 and args.scalar_count() == 1
        actual = args.tensor(0)
        assert tuple(actual.shapes) == (4,) and tuple(actual.strides) == (1,)
        assert actual.byte_offset == 0 and actual.buffer == backings[0].to_descriptor()
        assert actual.dtype == DataType.FLOAT32.value and args.scalar(0) == 12001
        assert args.tensor(1) == support.make_tensor_arg(worker, tokens[0][0])
        assert args.tag(1) == TensorArgType.INOUT
    elif context_tuple:
        assert [handle for handle, _ in calls] == [handles["sink"]]
        args = calls[0][1]
        assert args.tensor_count() == 3 and args.scalar_count() == 2
        assert args.tensor(0).buffer == backings[0].to_descriptor()
        assert [args.scalar(i) for i in range(2)] == [12002, 12001]
        left = state._frame.domain_tokens[(801, 5)][1]
        right = state._frame.domain_tokens[(802, 5)][1]
        assert left is not right
        assert args.tensor(1) == support.make_tensor_arg(worker, left)
        assert args.tensor(2) == support.make_tensor_arg(worker, right)
        assert args.tag(1) == args.tag(2) == TensorArgType.INOUT
    else:
        assert [handle for handle, _ in calls] == [handles["stage"], handles["sink"]]
        stage, sink = [args for _, args in calls]
        assert stage.tensor_count() == 4 and sink.tensor_count() == 2
        assert stage.tensor(0).buffer == backings[0].to_descriptor()
        assert stage.tensor(1).buffer == sink.tensor(0).buffer == backings[1].to_descriptor()
        assert [stage.scalar(i) for i in range(stage.scalar_count())] == [12001, 12001, 12002]
        assert [sink.scalar(i) for i in range(sink.scalar_count())] == [12002, 12002]
        assert stage.tensor(3) == sink.tensor(1) == support.make_tensor_arg(worker, tokens[1][0])
        assert stage.tensor(2) == support.make_tensor_arg(worker, tokens[0][0])
        assert stage.tag(2) == stage.tag(3) == sink.tag(1) == TensorArgType.INOUT
    state.complete_invocation()
    assert state._frame is None
    state.close()
    node.retire_domain(801, 5)
    if not geometry and not plain:
        node.retire_domain(802, 5)
    node.close()
    assert not any(name == "pypto" or name.startswith("pypto.") for name in sys.modules)
    print("C3B_DT_VIEW_" + mode.upper() + "_PASS")
finally:
    if worker is not None:
        worker.close()
    for backing in backings:
        backing.close()
"""


def test_c3b_dt_real_geometry_and_metadata_context_wire(dt_view_program, canonical_support_path, tmp_path):
    mode, program = dt_view_program
    source = codegen.DistributedCodegen()._generate_embedded(program, entry_name="node")
    runtime_python = os.environ.get("L4_RUNTIME_ONLY_PYTHON")
    runtime_pythonpath = os.environ.get("L4_RUNTIME_ONLY_PYTHONPATH")
    if not runtime_python or not runtime_pythonpath:
        pytest.skip("Set L4_RUNTIME_ONLY_PYTHON and L4_RUNTIME_ONLY_PYTHONPATH for real Simpler CPU proof")
    package = tmp_path / "dt_view_proof_namespace"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "entry.py").write_text(source, encoding="utf-8")
    shutil.copyfile(canonical_support_path, package / "_distributed_support.py")
    result = subprocess.run(
        [runtime_python, "-c", _DT_VIEW_PROOF, mode],
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
    assert result.returncode == 0, f"DT {mode} proof failed:\n{result.stdout}\n{result.stderr}"
    assert f"C3B_DT_VIEW_{mode.upper()}_PASS" in result.stdout


def _c3b_control_flow_program(mode):
    bodies = {
        "scalar": """        count = seed
        for i in pl.range(2):
            self.scalar(count, worker=1)
            count = count + pl.const(1, pl.INT32)
        self.scalar(count, worker=1)
""",
        "zero": """        count = seed
        for i in pl.range(0):
            self.scalar(count, worker=1)
            count = count + pl.const(1, pl.INT32)
        self.scalar(count, worker=1)
""",
        "nested": """        count = seed
        for i in pl.range(2):
            inner = count
            for j in pl.range(2):
                self.scalar(inner, worker=1)
                inner = inner + pl.const(1, pl.INT32)
            count = inner
        self.scalar(count, worker=1)
""",
        "swap": """        for i, (a, da, p, ca, b, db, q, cb) in pl.range(
            2, init_values=(x, data, seed, explicit, y, data, other, explicit)):
            self.mixed(a, p, da, ca, b, q, db, cb, worker=1)
            a_out, da_out, p_out, ca_out, b_out, db_out, q_out, cb_out = pl.yield_(b, da, q, ca, a, db, p, cb)
        self.mixed(a_out, p_out, da_out, ca_out, b_out, q_out, db_out, cb_out, worker=1)
""",
        "metadata_for": """        for i, (carry,) in pl.range(2, init_values=(data,)):
            result = pl.yield_(carry)
        self.window(result, worker=1)
""",
        "if_mixed": """        if flag:
            chosen, chosen_window, count = pl.yield_(x, data, seed)
        else:
            chosen, chosen_window, count = pl.yield_(y, data, seed + pl.const(1, pl.INT32))
        self.observe(chosen, count, chosen_window, explicit, worker=1)
""",
        "if_metadata": """        if flag:
            chosen_window = pl.yield_(data)
        else:
            chosen_window = pl.yield_(data)
        self.window(chosen_window, worker=1)
""",
        "if_ctx": """        if flag:
            result = pl.yield_(explicit)
        else:
            result = pl.yield_(explicit)
        self.observe(x, seed, data, result, worker=1)
""",
        "ambiguous_ctx_for": """        for i, (ctx,) in pl.range(2, init_values=(explicit,)):
            self.observe(x, seed, data, ctx, worker=1)
            next_ctx = pld.system.get_comm_ctx(data)
            result = pl.yield_(next_ctx)
""",
        "ambiguous_ctx_if": """        next_ctx = pld.system.get_comm_ctx(data)
        if flag:
            result = pl.yield_(explicit)
        else:
            result = pl.yield_(next_ctx)
        self.observe(x, seed, data, result, worker=1)
""",
    }
    helper = (
        "    def scalar(self, count: pl.Scalar[pl.INT32]):\n"
        if mode in {"scalar", "zero", "nested"}
        else "    def mixed(self, a: pl.Tensor[[4], pl.FP32], p: pl.Scalar[pl.INT32],\n"
        "              da: pld.DistributedTensor[[4], pl.FP32], ca: pld.CommCtx,\n"
        "              b: pl.Tensor[[4], pl.FP32], q: pl.Scalar[pl.INT32],\n"
        "              db: pld.DistributedTensor[[4], pl.FP32], cb: pld.CommCtx):\n"
        if mode == "swap"
        else "    def window(self, data: pld.DistributedTensor[[4], pl.FP32]):\n"
        if mode in {"metadata_for", "if_metadata"}
        else "    def observe(self, x: pl.Tensor[[4], pl.FP32], count: pl.Scalar[pl.INT32],\n"
        "                data: pld.DistributedTensor[[4], pl.FP32], ctx: pld.CommCtx):\n"
    )
    program = pl.parse_program(
        "@pl.program\nclass ControlFlow:\n"
        "    @pl.function(level=pl.Level.CHIP, role=pl.Role.Orchestrator)\n" + helper + "        return 0\n"
        "    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)\n"
        "    def node(self, x: pl.Tensor[[4], pl.FP32], y: pl.Tensor[[4], pl.FP32],\n"
        "             seed: pl.Scalar[pl.INT32], other: pl.Scalar[pl.INT32], flag: pl.Scalar[pl.BOOL],\n"
        "             data: pld.DistributedTensor[[4], pl.FP32], explicit: pld.CommCtx):\n"
        + bodies[mode]
        + "        return 0\n"
        "    @pl.function(level=pl.Level.CLUSTER_0, role=pl.Role.Orchestrator)\n"
        "    def root(self, x: pl.Tensor[[4], pl.FP32], y: pl.Tensor[[4], pl.FP32],\n"
        "             seed: pl.Scalar[pl.INT32], other: pl.Scalar[pl.INT32], flag: pl.Scalar[pl.BOOL]):\n"
        "        storage = pld.alloc_window_buffer(16)\n"
        "        data = pld.window(storage, [4], dtype=pl.FP32)\n"
        "        explicit = pld.system.get_comm_ctx(data)\n"
        "        self.node(x, y, seed, other, flag, data, explicit, worker=0)\n"
        "        return 0\n"
    )
    for transform in (
        passes.convert_to_ssa,
        passes.synthesize_allreduce_signals,
        passes.materialize_comm_domain_scopes,
        passes.lower_host_tensor_collectives,
        passes.derive_call_directions,
        passes.materialize_dist_tensor_ctx,
        passes.normalize_return_order,
    ):
        program = transform()(program)
    properties = passes.IRPropertySet()
    for prop in (
        passes.IRProperty.CommDomainScopesMaterialized,
        passes.IRProperty.DistTensorCtxMaterialized,
        passes.IRProperty.TypeChecked,
        passes.IRProperty.ReturnParamsExplicit,
    ):
        properties.insert(prop)
    assert passes.PropertyVerifierRegistry.verify(properties, program) == []
    ir.assert_structural_equal(program, pl.parse_program(ir.python_print(program, format=False)))
    ir.assert_structural_equal(program, ir.deserialize(ir.serialize(program)))
    compile(codegen.DistributedCodegen().generate(program), "control_flow_cluster.py", "exec")
    return program


_CONTROL_FLOW_PROOF = r"""
import ctypes
import importlib.abc
import sys
from contextlib import ExitStack
from types import SimpleNamespace
from unittest import mock

class RejectPyPTO(importlib.abc.MetaPathFinder):
    def __init__(self):
        self.attempts = []
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "pypto" or fullname.startswith("pypto."):
            self.attempts.append(fullname)
            raise ModuleNotFoundError("Forbidden runtime-only import: " + fullname)
        return None

assert not any(name == "pypto" or name.startswith("pypto.") for name in sys.modules)
guard = RejectPyPTO()
sys.meta_path.insert(0, guard)
import torch
from control_flow_namespace import entry
from control_flow_namespace import _distributed_support as support
from simpler.buffer import create_host_shared_buffer, mint_owner_instance_id
from simpler.task_interface import (
    CallConfig, ChipDomainContext, DataType, GlobalCommDomainView,
    GlobalDomainMember, TaskArgs, TensorArgType,
)
from simpler.worker import Worker

mode = sys.argv[1]
token_bundles = tuple((torch.zeros((1,), dtype=torch.int32).share_memory_(),)
                      for _ in range(2 if mode in ("swap", "if_ctx") else 1))
tokens = token_bundles[0]
node = support._EmbeddedNodeState(local_worker_ids=(1,), prepared_domain_tokens=token_bundles)
owner = mint_owner_instance_id()
backings = [create_host_shared_buffer(16, owner, index + 1) for index in range(3)]
worker = None
try:
    x, y = [backing.tensor(shapes=(4,), dtype=DataType.FLOAT32) for backing in backings[:2]]
    assert x.buffer != y.buffer
    window = backings[2].tensor(shapes=(4,), dtype=DataType.FLOAT32)
    name = entry.node._pypto_embedded_layout["window_slots"][0][4]
    members = tuple(GlobalDomainMember(node_worker_id=7 if rank == 9 else 100 + rank,
                                      local_worker_id=1 if rank == 9 else 0,
                                      global_device_rank=rank, domain_rank=rank) for rank in range(12))
    context = ChipDomainContext(name="received", domain_rank=9, domain_size=12,
                                device_ctx=12001, local_window_base=0, actual_window_size=16,
                                buffers={name: backings[2]})
    view = GlobalCommDomainView(name="received", members=members, contexts={1: context},
                                domain_id=801, generation=5, mapping_size=16)
    # Recorded commit boundary, genuine HOST POSIX storage. No physical VMM or
    # CHIP execution is represented by this CPU control-flow/wire proof.
    view._committed = True
    domains = {801: view}
    if mode in ("swap", "if_ctx"):
        ctx_context = ChipDomainContext(name="explicit", domain_rank=9, domain_size=12,
                                       device_ctx=12002, local_window_base=0, actual_window_size=16,
                                       buffers={})
        ctx_view = GlobalCommDomainView(name="explicit", members=members, contexts={1: ctx_context},
                                        domain_id=802, generation=5, mapping_size=16)
        ctx_view._committed = True
        domains[802] = ctx_view
    def no_op(mapped_args):
        pass
    worker = Worker(level=3, num_sub_workers=2)
    no_op_handle = worker.register(no_op)
    handles = {name: object() for name in ("scalar", "mixed", "observe", "window")}
    callables = support._ChipCallables()
    for name, handle in handles.items():
        callables.add(name, handle, {})
    state = support._EmbeddedEntryState(entry.node._pypto_embedded_layout, callables=callables,
                                       sub_ids={}, world_size=2, prepared_tensors={},
                                       domain_provider=None, node_state=node)
    entry.node._pypto_embedded_state = state
    worker.init()

    for flag in ((True, False) if mode.startswith("if_") else (True,)):
        incoming = TaskArgs()
        incoming.add_tensor(x, TensorArgType.INPUT)
        incoming.add_tensor(y, TensorArgType.INPUT)
        for value in (ctypes.c_int32(5), ctypes.c_int32(9), flag,
                      802 if mode in ("swap", "if_ctx") else 801, 801):
            incoming.add_scalar(value)
        calls = []
        def orchestration(orch, args, config):
            def submit(handle, task_args, actual_config, *, worker):
                assert actual_config is config and worker == 1
                calls.append((handle, task_args))
                # Only the CHIP transport is adapted. Real mapped SUB tasks,
                # TaskArgs and canonical dependency draining execute on CPU.
                orch.submit_sub(no_op_handle, task_args)
            adapter = SimpleNamespace(_worker=orch._worker, submit_next_level=submit,
                                      get_global_domain=lambda domain_id: domains[domain_id])
            def forbidden(*args, **kwargs):
                raise AssertionError("Control-flow invocation allocated storage")
            with ExitStack() as stack:
                for name in ("zeros", "empty"):
                    stack.enter_context(mock.patch.object(torch, name, side_effect=forbidden))
                stack.enter_context(mock.patch.object(torch.Tensor, "share_memory_", side_effect=forbidden))
                stack.enter_context(mock.patch(
                    "simpler.buffer.create_host_shared_buffer", side_effect=forbidden))
                entry.node(adapter, incoming, config)
            assert state._frame is not None
            assert any(value is incoming for value in state._frame.keepalive)
            assert all(any(value is task_args for value in state._frame.keepalive) for _, task_args in calls)
        worker.run(orchestration, args=None, config=CallConfig())
        if mode in ("scalar", "zero", "nested"):
            expected = {"scalar": [5, 6, 7], "zero": [5], "nested": [5, 6, 7, 8, 9]}[mode]
            assert [handle for handle, _ in calls] == [handles["scalar"]] * len(expected)
            assert [support._decode_scalar(args.scalar(0), "int32") for _, args in calls] == expected
            assert all(args.tensor_count() == 0 and args.scalar_count() == 1 for _, args in calls)
        elif mode == "swap":
            assert [handle for handle, _ in calls] == [handles["mixed"]] * 3
            data_token = state._frame.domain_tokens[(801, 5)][1]
            ctx_token = state._frame.domain_tokens[(802, 5)][1]
            assert data_token is not ctx_token
            for (_, args), (left, p, right, q) in zip(calls, ((x, 5, y, 9), (y, 9, x, 5), (x, 5, y, 9))):
                assert args.tensor_count() == 6 and args.scalar_count() == 6
                assert args.tensor(0) == left and args.tensor(2) == right
                assert args.tensor(1) == args.tensor(3) == window
                assert args.tensor(4) == support.make_tensor_arg(worker, data_token)
                assert args.tensor(5) == support.make_tensor_arg(worker, ctx_token)
                assert args.tag(4) == args.tag(5) == TensorArgType.INOUT
                assert [support._decode_scalar(args.scalar(i), "int32") for i in (0, 2)] == [p, q]
                assert [args.scalar(i) for i in (1, 3, 4, 5)] == [12002, 12002, 12001, 12001]
        elif mode == "if_ctx":
            assert [handle for handle, _ in calls] == [handles["observe"]]
            args = calls[0][1]
            data_token = state._frame.domain_tokens[(801, 5)][1]
            ctx_token = state._frame.domain_tokens[(802, 5)][1]
            assert data_token is not ctx_token
            assert args.tensor_count() == 4 and args.scalar_count() == 3
            assert args.tensor(0) == x and args.tensor(1) == window
            assert args.tensor(2) == support.make_tensor_arg(worker, data_token)
            assert args.tensor(3) == support.make_tensor_arg(worker, ctx_token)
            assert args.tag(2) == args.tag(3) == TensorArgType.INOUT
            assert support._decode_scalar(args.scalar(0), "int32") == 5
            assert [args.scalar(i) for i in (1, 2)] == [12002, 12001]
        elif mode == "if_mixed":
            assert [handle for handle, _ in calls] == [handles["observe"]]
            args = calls[0][1]
            assert args.tensor_count() == 3 and args.scalar_count() == 3
            assert args.tensor(0) == (x if flag else y) and args.tensor(1) == window
            assert args.tensor(2) == support.make_tensor_arg(worker, tokens[0])
            assert support._decode_scalar(args.scalar(0), "int32") == (5 if flag else 6)
            assert [args.scalar(i) for i in (1, 2)] == [12001, 12001]
        else:
            assert [handle for handle, _ in calls] == [handles["window"]]
            args = calls[0][1]
            assert args.tensor_count() == 2 and args.scalar_count() == 1
            assert args.tensor(0) == window and args.tensor(1) == support.make_tensor_arg(worker, tokens[0])
            assert args.scalar(0) == 12001
        state.complete_invocation()
        assert state._frame is None
    state.close()
    node.retire_domain(801, 5)
    if mode in ("swap", "if_ctx"):
        node.retire_domain(802, 5)
    node.close()
    assert not guard.attempts
    assert not any(name == "pypto" or name.startswith("pypto.") for name in sys.modules)
    print("C3B_CONTROL_FLOW_" + mode.upper() + "_PASS")
finally:
    if worker is not None:
        worker.close()
    for backing in backings:
        backing.close()
"""


@pytest.mark.parametrize(
    "mode", ["scalar", "zero", "nested", "swap", "metadata_for", "if_mixed", "if_metadata", "if_ctx"]
)
def test_c3b_embedded_control_flow_real_wire_and_cpu_drain(mode, canonical_support_path, tmp_path):
    program = _c3b_control_flow_program(mode)
    source = codegen.DistributedCodegen()._generate_embedded(program, entry_name="node")
    compile(source, "embedded_control_flow.py", "exec")
    runtime_python = os.environ.get("L4_RUNTIME_ONLY_PYTHON")
    runtime_pythonpath = os.environ.get("L4_RUNTIME_ONLY_PYTHONPATH")
    if not runtime_python or not runtime_pythonpath:
        pytest.skip("Set L4_RUNTIME_ONLY_PYTHON and L4_RUNTIME_ONLY_PYTHONPATH for real Simpler CPU proof")
    package = tmp_path / "control_flow_namespace"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "entry.py").write_text(source, encoding="utf-8")
    shutil.copyfile(canonical_support_path, package / "_distributed_support.py")
    result = subprocess.run(
        [runtime_python, "-c", _CONTROL_FLOW_PROOF, mode],
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
    assert result.returncode == 0, f"Control-flow {mode} proof failed:\n{result.stdout}\n{result.stderr}"
    assert f"C3B_CONTROL_FLOW_{mode.upper()}_PASS" in result.stdout


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
