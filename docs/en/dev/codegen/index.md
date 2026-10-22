# Code Generation

Lowering PyPTO IR to the two artifacts a compiled program consists of: device
kernels and the host-side orchestration that launches them.

Both generators follow the same design principle — a **strict 1-to-1 mapping** from
IR to generated code. Anything that needs a decision was already decided by a pass.

| Page | What it covers |
| ---- | -------------- |
| [PTO Codegen](00-pto_codegen.md) | Generating PTO-ISA dialect MLIR from PyPTO IR |
| [Orchestration Code Generation](01-orchestration_codegen.md) | Generating the host-side C++ that submits tasks to the runtime |
| [PTO Kernel Wrapper Generation](02-kernel_wrapper.md) | Connecting ptoas-generated kernels to runtime dispatch |

## Distributed hierarchy formatting

`DistributedCodegen` emits Python orchestration using the internal, stateless
`hierarchy::LevelHandler` (`include/pypto/hierarchy/level_handler.h`). Its seven
methods own domain allocator and membership formatting, context values, complete
ordinary host-tensor append statements, window materialization capability, entry
keyword parameters, and imports. The generator retains traversal,
allocation preparation, tensor/scalar grouping, ordering-token ownership, explicit
Submit dependencies, dynamic dimensions, and explicit CommCtx parameter forwarding.
It also selects the domain membership keyword and additional allocation arguments.

`GetLevelHandler(ir::Level)` registers HOST and CLUSTER_0 (also named POD) and returns `nullptr` for every
unsupported level; enum values are distinct from Linqu level numbers. At existing
generator emission sites, an explicit HOST-formatting fallback preserves legacy
output for entries at unregistered levels. This fallback does not authorize a
capability. Imports follow the selected highest-level entry. The standalone HOST
entry ABI and generated source remain unchanged.

| Boundary | HOST → CHIP | CLUSTER_0 → HOST |
| -------- | ----------- | ---------------- |
| Child count | `world_size` (local device count) | `num_workers = len(nodes)` |
| Ordinary tensor | `add_chip_tensor_arg(...)` | `_io_arg(node_bindings[index], tensor, direction)` |
| Window tensor | Materialized from the domain Buffer | Omitted from the wire tensor pool |
| Context scalar | Target child's `device_ctx` | Global `domain_id` |
| Domain members | `workers=[...]` | `members=[(nodes[n], d), ...]` in topology order |
| Submit | `_submit_chip(...)` | Per-node L3 handle with `worker=nodes[index]` |

CLUSTER dispatch uses the explicit `worker=` node index when supplied. Comm-less
calls without it select nodes round-robin by node count, resolving the logical
index once for tensor routing, the callable and the actual worker ID. This sequence
resets for each entry invocation and is independent of local device counts. Domain
buffers retain the existing physical alignment and logical byte counts. Global
allocation accepts `_domain_provider` and `retain_after_run=_retain`; CHIP ordering
tokens remain specific to HOST. All tensors precede scalars, preserving each
pool's formal order, explicit Submit dependencies, TaskId results and output aliases.
Implicit tensor return storage also uses the node routing binding with
`OUTPUT_EXISTING`; this is a packing contract, while the embedded HOST return ABI
is still pending. Same-level CLUSTER SubWorker dispatch is rejected explicitly.

CLUSTER `for` loops bind SSA carries before the loop, update them in parallel at
the backedge, and expose final values after the loop, including zero-trip loops.
Ordinary tensor carries use the tensor dictionary; window carries remain metadata.
Metadata-only loop bodies retain a `pass` statement so their Python suite is valid.
Its parent allocation helper has the explicit ABI
`_alloc_intermediates(tensors, *, nodes)`. It binds the node count independently of
the entry and resolves direct `num_workers()` shape queries and their SSA aliases
through the existing scalar-definition resolver. HOST loop emission and its
`_alloc_intermediates(tensors, world_size=1)` ABI remain unchanged.

This CLUSTER support currently exposes internal generated source only. Public L4
compilation, artifact execution and runners remain gated until recursive entry
metadata, embedded HOST entries and runner integration are complete. `_io_arg` is
bound by that future integration; codegen does not import a nonexistent runner.
L4 `world_size` and collective operations remain unsupported. Having a handler
does not authorize higher levels or calls that skip the adjacent hierarchy level.

## See Also

- [Passes](../passes/index.md) — everything that must hold before codegen runs.
- [PTO ISA reference](../../reference/index.md) — the instruction semantics the generated code uses.
- [PTOAS Op Status Matrix](../ptoas-op-status.md) — which PTOAS ops are available to emit.
