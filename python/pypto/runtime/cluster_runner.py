# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Synchronous CLUSTER_0 execution with parent compilation and rank-local I/O."""

import ctypes
import importlib.util
import json
import math
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from pypto.ir.compiled_program import _coerce_args
from pypto.jit._artifact_manifest import BuildKind

from ._artifact_sources import build_cluster_source_bundle
from ._prebuilt import chip_directories
from .distributed_runner import _make_call_config


def Worker(**kwargs: Any) -> Any:
    """Load the optional runtime only when preparing an execution session."""
    from simpler.worker import Worker as RuntimeWorker  # noqa: PLC0415

    return RuntimeWorker(**kwargs)


def _assemble_cluster_chips(compiled: Any) -> dict[str, tuple[Any, str, dict[str, Any]]]:
    """Compile the recursively declared CHIP closure exclusively on the parent."""
    artifact_runtime = vars(compiled).get("_artifact_runtime")
    if artifact_runtime is not None:
        return artifact_runtime.load()
    from .device_runner import _compile_and_assemble  # noqa: PLC0415

    return {
        path: _compile_and_assemble(directory, compiled.platform)
        for path, directory in chip_directories(compiled.output_dir, BuildKind.DISTRIBUTED).items()
    }


def _intervals(values: list[tuple[int, int]]) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    for start, end in sorted(values):
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        elif start < end:
            result.append((start, end))
    return result


@dataclass
class _Storage:
    """One owned storage range; strong references forbid address reuse mid-run."""

    storage: Any
    tensor: torch.Tensor
    worker: int
    handle: Any
    writes: list[tuple[int, int]] = field(default_factory=list)


class _InvocationIO:
    """Ordinary contiguous tensors staged through rank-local remote allocations.

    Offsets refer to the Torch storage base, preserving overlapping aliases on
    one rank. Cross-rank write/read or write/write overlaps require explicit
    forwarding and are rejected, including when both MPI ranks share a host.
    """

    def __init__(self, worker: Any) -> None:
        self.worker = worker
        self.storages: dict[tuple[int, int, int], _Storage] = {}
        self.accesses: list[tuple[int, int, int, bool]] = []

    def prepare(self, tensors: dict[str, Any], nodes: tuple[int, ...]) -> None:
        """Stage owned backings before entering the graph's first submission.

        Ordinary routing can depend on generated Python control flow. Stage the
        complete static tensor inventory on every node so graph construction
        performs descriptor lookup only, respecting Simpler's mailbox ordering.
        """
        for tensor in tensors.values():
            if not isinstance(tensor, torch.Tensor):
                continue
            if tensor.device.type != "cpu" or not tensor.is_contiguous():
                raise ValueError("Cluster ordinary I/O requires contiguous CPU tensors")
            storage = tensor.untyped_storage()
            base, size = storage.data_ptr(), storage.nbytes()
            if size <= 0:
                raise ValueError("Cluster ordinary I/O requires nonempty tensor storage")
            for node in nodes:
                key = (node, base, size)
                if key in self.storages:
                    continue
                handle = self.worker.remote_malloc(worker=node, nbytes=size)
                owned = _Storage(storage, tensor, node, handle)
                self.storages[key] = owned
                self.worker.remote_copy_to(handle, base, size)

    def argument(self, rank: int, tensor: torch.Tensor, tag: Any) -> Any:
        from simpler.task_interface import DataType, RemoteTensorRef, TensorArgType  # noqa: PLC0415

        if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu" or not tensor.is_contiguous():
            raise ValueError("Cluster ordinary I/O requires contiguous CPU torch.Tensor arguments")
        storage = tensor.untyped_storage()
        base, size = storage.data_ptr(), storage.nbytes()
        start = tensor.data_ptr() - base
        nbytes = tensor.numel() * tensor.element_size()
        end = start + nbytes
        if nbytes <= 0:
            raise ValueError("Cluster ordinary I/O requires nonempty tensor views")
        if start < 0 or end > size:
            raise ValueError("Cluster tensor view lies outside its owned storage range")
        reading = tag in (TensorArgType.INPUT, TensorArgType.INOUT)
        writing = tag in (TensorArgType.OUTPUT, TensorArgType.OUTPUT_EXISTING, TensorArgType.INOUT)
        if not reading and not writing:
            raise ValueError(f"Unsupported cluster tensor dependency tag {tag!r}")
        absolute_start, absolute_end = base + start, base + end
        for previous_rank, previous_start, previous_end, previous_write in self.accesses:
            if (
                previous_rank != rank
                and (writing or previous_write)
                and max(absolute_start, previous_start) < min(absolute_end, previous_end)
            ):
                raise NotImplementedError(
                    "Cross-node ordinary tensor forwarding requires explicit transfer and fences; "
                    "use independent outputs until global-domain execution is available"
                )
        self.accesses.append((rank, absolute_start, absolute_end, writing))
        key = (rank, base, size)
        owned = self.storages.get(key)
        if owned is None:
            raise NotImplementedError(
                "Generated cluster root introduced tensor storage after I/O preparation; "
                "all ordinary tensor storage must be declared in _alloc_intermediates"
            )
        if writing:
            owned.writes = _intervals([*owned.writes, (start, end)])
        dtype_name = str(tensor.dtype).removeprefix("torch.").upper()
        dtype_name = {"BFLOAT16": "BFLOAT16", "FLOAT": "FLOAT32", "DOUBLE": "FLOAT64"}.get(
            dtype_name, dtype_name
        )
        dtype = getattr(DataType, dtype_name, None)
        if dtype is None:
            raise ValueError(f"Unsupported cluster tensor dtype {tensor.dtype}")
        return RemoteTensorRef(
            owned.handle, offset=start, shape=tuple(tensor.shape), dtype=dtype, nbytes=nbytes
        )

    def publish(self) -> None:
        """Copy all ranks' completed outputs before releasing any allocation."""
        for owned in self.storages.values():
            for start, end in owned.writes:
                self.worker.remote_copy_from(
                    owned.handle,
                    owned.storage.data_ptr() + start,
                    end - start,
                    offset=start,
                )

    def release(self) -> None:
        # Keep every allocation journaled if a free fails; whole-tree close
        # remains the reclamation boundary and still owns every strong ref.
        for owned in self.storages.values():
            self.worker.remote_free(owned.handle)
        self.clear()

    def clear(self) -> None:
        self.storages.clear()
        self.accesses.clear()


def _io_arg(binding: tuple[_InvocationIO, int], tensor: torch.Tensor, tag: Any) -> Any:
    return binding[0].argument(binding[1], tensor, tag)


class ClusterWorker:
    """Prepared CLUSTER_0 session for synchronous ordinary Tensor/Scalar runs.

    All CHIP compilation happens on the parent. The installed rank source
    package imports only Simpler/Torch and is prepared before the inner fork.
    Sessions accept ordinary CPU tensors allocated after preparation.
    """

    def __init__(
        self, programs: list[Any], config: Any = None, *, startup_timeout_s: float | None = None
    ) -> None:
        from simpler.startup_artifacts import StartupArtifactSpec  # noqa: PLC0415
        from simpler.startup_preparation import NodeStartupHook  # noqa: PLC0415
        from simpler.worker import MpiL3GroupSpec, RemoteCallable  # noqa: PLC0415

        if len(programs) != 1:
            raise NotImplementedError("Cluster extra_compiled requires multi-program session support")
        compiled = programs[0]
        if compiled.entry_kind != "cluster":
            raise ValueError("ClusterWorker requires a CLUSTER_0 compiled program")
        compiled._require_artifact_ready()
        cc = compiled.cluster_config
        records = compiled.l3_entries["entries"]
        for record in records:
            layout = record["layout"]
            if layout["window_slots"] or any(row[3] == "comm_ctx" for row in layout["scalar_slots"]):
                raise NotImplementedError("Cluster global windows require the global-domain runner")
            required = compiled.output_dir / record["logical_path"] / "sub_workers/__required__.json"
            if required.exists() and json.loads(required.read_text(encoding="utf-8")):
                raise NotImplementedError("Cluster runtime-bound SUB callbacks are unsupported")
        timeout = cc.startup_timeout_s if startup_timeout_s is None else startup_timeout_s
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Cluster startup_timeout_s must be positive and finite")
        chips = _assemble_cluster_chips(compiled)
        if any(runtime != cc.runtime for _, runtime, _ in chips.values()):
            raise ValueError("Every cluster CHIP dependency must use ClusterConfig.runtime")
        digest, payload = build_cluster_source_bundle(compiled.output_dir)
        self._compiled = compiled
        self._closed = False
        self._lock = threading.RLock()
        self._active_io: _InvocationIO | None = None
        self._keep: list[Any] = []
        self._temporary = tempfile.TemporaryDirectory(prefix="pypto-cluster-")
        self._worker = None
        try:
            archive = Path(self._temporary.name) / "sources.tar.gz"
            archive.write_bytes(payload)
            prefix = "_pypto_cluster_" + digest
            self._worker = Worker(
                level=4,
                platform=compiled.platform,
                runtime=cc.runtime,
                startup_timeout_s=timeout,
                enable_sdma=any(bool(value[2].get("enable_sdma", False)) for value in chips.values()),
            )
            chip_digests = {
                path: self._worker.register(chip).digest.hex() for path, (chip, _, _) in chips.items()
            }
            parameters = {
                "module_prefix": prefix,
                "entries": records,
                "chip_digests": chip_digests,
                "chip_configs": {path: value[2] for path, value in chips.items()},
            }
            spec = MpiL3GroupSpec(
                hosts=tuple(node.host for node in cc.nodes),
                platform=compiled.platform,
                device_ids_by_rank=tuple(tuple(node.devices) for node in cc.nodes),
                runtime=cc.runtime,
                num_sub_workers_by_rank=(
                    max(cc.num_sub_workers, sum(record["minimum_sub_workers"] for record in records)),
                )
                * len(cc.nodes),
                comm_profile=cc.comm_profile,
                global_device_ranks_by_rank=tuple(cc.global_device_ranks()),
                mpirun_path=cc.mpirun_path,
                mpirun_args=tuple(cc.mpirun_args),
                python_executable=cc.python_executable,
                startup_artifacts=(StartupArtifactSpec(archive, digest, prefix),),
                startup_hooks_by_rank=tuple(
                    NodeStartupHook(prefix + ".orchestration.cluster_bootstrap:prepare_node", parameters)
                    for _ in cc.nodes
                ),
            )
            self._nodes = self._worker.add_mpirun_worker_group(spec)
            self._callables = {
                record["entry_name"]: [
                    self._worker.register(
                        RemoteCallable(prefix + "." + record["module"] + ":" + record["qualname"]),
                        workers=list(self._nodes),
                    )
                ]
                * len(self._nodes)
                for record in records
            }
            path = compiled.output_dir / "orchestration/cluster_orch.py"
            module_spec = importlib.util.spec_from_file_location(prefix + "_parent", path)
            if module_spec is None or module_spec.loader is None:
                raise RuntimeError(f"Cannot load generated cluster root {path}")
            module = importlib.util.module_from_spec(module_spec)
            module._io_arg = _io_arg
            # Artifact directories are immutable, including their inventory:
            # source execution must not create a loader's bytecode cache there.
            exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)
            self._entry = getattr(module, compiled.entry_name)
            self._allocate = getattr(module, "_alloc_intermediates", None)
            self._worker.init()
        except BaseException:
            self.close()
            raise

    def run(self, compiled: Any, *args: Any, config: Any = None) -> Any:
        """Execute the generated parent root and publish results after its fence."""
        if compiled is not self._compiled:
            raise ValueError("ClusterWorker.run requires its prepared compiled program")
        infos, outputs, returns = compiled._get_metadata()
        values, return_style = _coerce_args(args, infos, outputs, returns, caller_name="ClusterWorker")
        for info, value in zip(infos, values, strict=True):
            if info.shape is not None:
                if (
                    not isinstance(value, torch.Tensor)
                    or value.device.type != "cpu"
                    or not value.is_contiguous()
                ):
                    raise ValueError(f"Cluster parameter {info.name!r} requires a contiguous CPU tensor")
                if str(value.dtype).removeprefix("torch.") != {
                    "fp16": "float16",
                    "bf16": "bfloat16",
                    "fp32": "float32",
                }.get(str(info.dtype), str(info.dtype)):
                    raise ValueError(f"Cluster parameter {info.name!r} has wrong dtype: {value.dtype}")
                if len(value.shape) != len(info.shape) or any(
                    expected >= 0 and actual != expected
                    for actual, expected in zip(value.shape, info.shape, strict=True)
                ):
                    raise ValueError(f"Cluster parameter {info.name!r} has wrong shape: {tuple(value.shape)}")
        tensors = {
            info.name: value.value if isinstance(value, ctypes._SimpleCData) else value
            for info, value in zip(infos, values, strict=True)
        }
        with self._lock:
            if self._closed:
                raise RuntimeError("ClusterWorker is closed")
            if self._active_io is not None:
                raise RuntimeError("Previous cluster invocation requires a successful close before reuse")
            io = _InvocationIO(self._worker)
            self._active_io = io
            cc = compiled.cluster_config

            def orchestration(orch: Any, unused: Any, unused_config: Any) -> None:
                self._entry(
                    orch,
                    unused,
                    call_config,
                    tensors=tensors,
                    callables=self._callables,
                    nodes=list(self._nodes),
                    node_sizes=cc.node_device_counts(),
                    node_bindings=[(io, node) for node in self._nodes],
                    _keep=self._keep,
                    _retain=False,
                )

            released = False
            try:
                if self._allocate is not None:
                    self._allocate(tensors, nodes=list(self._nodes))
                io.prepare(tensors, self._nodes)
                call_config = _make_call_config(cc, config, dfx_base=compiled.output_dir / "dfx_outputs")
                self._worker.run(orchestration, config=call_config)
                io.publish()
                io.release()
                released = True
            except BaseException:
                # A native failure may leave accepted work unsettled. Closing
                # the hierarchy drains/reaps it before releasing metadata; no
                # partial output is copied after a failed DAG.
                self.close()
                raise
            finally:
                if self._closed or released:
                    # An unsuccessful tree close leaves acceptance uncertain;
                    # its submitted argument frames remain owned until retry.
                    self._keep.clear()
                    self._active_io = None
        if return_style:
            result = [values[index] for index in outputs]
            return result[0] if len(result) == 1 else tuple(result)
        return None

    def __call__(self, *args: Any, config: Any = None) -> Any:
        return self.run(self._compiled, *args, config=config)

    def submit(self, *args: Any, **kwargs: Any) -> Any:
        """Reject asynchronous admission before it can allocate or submit work."""
        raise NotImplementedError("ClusterWorker supports synchronous run only")

    def close(self) -> None:
        """Drain the complete worker tree before releasing invocation storage."""
        with self._lock:
            if self._closed:
                return
            if self._worker is not None:
                self._worker.close()
            if self._active_io is not None:
                self._active_io.clear()
                self._active_io = None
            self._keep.clear()
            self._temporary.cleanup()
            self._closed = True

    def __enter__(self) -> "ClusterWorker":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()
