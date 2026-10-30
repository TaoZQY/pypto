# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Runtime-only node preparation, copied under the installed artifact namespace.

The parent owns compilation and cold metadata validation. This module binds
native generated entries to local runtime handles before the inner worker forks.
It deliberately has no dependency on the compiler or the PyPTO package.
"""

import importlib
from types import MappingProxyType
from typing import Any

import torch

from ._distributed_support import _ChipCallables, _copy_return_tensor, _EmbeddedEntryState, _EmbeddedNodeState


def _freeze(value: Any) -> Any:
    """Restore immutable native layout records after their JSON transport."""
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, dict):
        return {key: _freeze(item) for key, item in value.items()}
    return value


def _tensor(dtype: str, shape: tuple[int, ...]) -> torch.Tensor:
    name = {"fp16": "float16", "bf16": "bfloat16", "fp32": "float32"}.get(dtype, dtype)
    return torch.zeros(shape, dtype=getattr(torch, name)).share_memory_()


class _NodeBindings:
    """Own prefork storage until all runtime children have drained and closed."""

    def __init__(self) -> None:
        self.entries: list[tuple[Any, _EmbeddedEntryState]] = []
        self.node_state: _EmbeddedNodeState | None = None

    def task_completed(self) -> None:
        """Called by the generic command loop after a successful DAG fence."""
        for _, state in self.entries:
            state.complete_invocation()

    def close(self) -> None:
        """The caller closes its inner worker before releasing these references."""
        for entry, state in self.entries:
            state.close()
            if getattr(entry, "_pypto_embedded_state", None) is state:
                del entry._pypto_embedded_state
        self.entries.clear()
        if self.node_state is not None:
            self.node_state.close()
            self.node_state = None


def _validated_entries(prefix: str, records: list[dict[str, Any]]) -> list[tuple[Any, Any, Any]]:
    # Validate every native/cold ABI before registering SUB workers or binding
    # state. The immutable literal remains authoritative on the receiving node.
    validated = []
    seen = set()
    for record in records:
        module = record["module"]
        qualname = record["qualname"]
        if (
            module != record["logical_path"].replace("/", ".") + ".orchestration.host_orch"
            or not isinstance(qualname, str)
            or not qualname.isidentifier()
            or module in seen
        ):
            raise ValueError("Cluster bootstrap has invalid or duplicate entry paths")
        seen.add(module)
        entry = getattr(importlib.import_module(prefix + "." + module), qualname)
        layout = getattr(entry, "_pypto_embedded_layout", None)
        if not isinstance(layout, MappingProxyType) or dict(layout) != _freeze(record["layout"]):
            raise ValueError(f"Cluster bootstrap native layout differs from cold entry {qualname!r}")
        if not isinstance(record["local_device_count"], int) or record["local_device_count"] <= 0:
            raise ValueError("Cluster bootstrap requires a positive local device count")
        validated.append((record, entry, layout))
    return validated


def prepare_node(context: Any) -> _NodeBindings:
    """Bind installed native HOST entry functions without importing PyPTO.

    Parameters carry one package's cold entry records, full logical CHIP paths,
    parent registration digests and tensor-transfer configuration. Actual CHIP
    handles come only from the runtime's process-local startup context.
    """
    parameters = context.parameters
    if not isinstance(parameters, dict) or set(parameters) != {
        "module_prefix",
        "entries",
        "chip_digests",
        "chip_configs",
    }:
        raise ValueError("Cluster bootstrap parameters require package, entries and CHIP bindings")
    prefix = parameters["module_prefix"]
    if not isinstance(prefix, str) or any(not part.isidentifier() for part in prefix.split(".")):
        raise ValueError("Cluster bootstrap module_prefix must use Python identifiers")
    records = parameters["entries"]
    if not isinstance(records, list) or not records:
        raise ValueError("Cluster bootstrap requires nonempty cold entry records")
    paths = {path for record in records for path in record["chip_paths"].values()}
    digests, configs = parameters["chip_digests"], parameters["chip_configs"]
    if (
        not isinstance(digests, dict)
        or not isinstance(configs, dict)
        or set(digests) != paths
        or set(configs) != paths
    ):
        raise ValueError("Cluster bootstrap CHIP bindings must preserve every full logical path")
    if any(digest not in context.chip_handles for digest in digests.values()):
        raise ValueError("Cluster bootstrap is missing a process-local CHIP handle")
    validated = _validated_entries(prefix, records)

    bindings = _NodeBindings()
    try:
        # Every entry can receive distinct physical domains. Their persisted
        # ctx upper bounds provide one finite pool shared by all entry states;
        # aliases acquire a single lease only after the actual views arrive.
        capacity = sum(
            row[1]
            for _, _, layout in validated
            for row in layout["prefork_requirements"]
            if row[0] == "inherited_domain_tokens"
        )
        if capacity:
            # R1 numbers local CHIP children densely, independently of their
            # physical device IDs and the domain's global ranks. This topology
            # already exists before init() forks the receiving worker's children.
            count = len(context.inner_worker._config.get("device_ids", ()))
            if not count or any(record["local_device_count"] != count for record, _, _ in validated):
                raise ValueError("Cluster inherited domains require the declared local CHIP topology")
            bindings.node_state = _EmbeddedNodeState(
                local_worker_ids=tuple(range(count)),
                prepared_domain_tokens=tuple(
                    tuple(_tensor("int32", (1,)) for _ in range(count)) for _ in range(capacity)
                ),
            )
        for record, entry, layout in validated:
            count = record["local_device_count"]
            prepared: dict[str, Any] = {}
            for row in layout["prefork_requirements"]:
                if row[0] == "tensor_storage":
                    prepared[row[1]] = _tensor(row[2], row[3])
                elif row[0] == "local_domain_tokens":
                    prepared[row[1]] = tuple(_tensor("int32", (1,)) for _ in range(count))
                elif row[0] not in {
                    "return_copy_subworker",
                    "inherited_domain_tokens",
                    "domain_dispatch",
                    "same_domain",
                }:
                    raise NotImplementedError(f"Cluster prefork preparation is unsupported: {row[0]}")
            callables = _ChipCallables()
            for name, path in record["chip_paths"].items():
                callables.add(name, context.chip_handles[digests[path]], configs[path])
            sub_ids = {}
            for name, path in record["sub_worker_paths"].items():
                target = importlib.import_module(prefix + "." + path[:-3].replace("/", "."))
                sub_ids[name] = context.inner_worker.register(getattr(target, name))
            copying = (
                context.inner_worker.register(_copy_return_tensor) if record["return_copy_callable"] else None
            )
            state = _EmbeddedEntryState(
                layout,
                callables=callables,
                sub_ids=sub_ids,
                world_size=count,
                prepared_tensors=prepared,
                domain_provider=None,
                return_copy_callable=copying,
                node_state=bindings.node_state,
            )
            entry._pypto_embedded_state = state
            bindings.entries.append((entry, state))
    except BaseException:
        bindings.close()
        raise
    return bindings
