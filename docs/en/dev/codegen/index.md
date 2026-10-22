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
`OUTPUT_EXISTING`. Caller packing and embedded HOST returns share a return-position
plan: an actual returned Out/InOut aliases its supplied storage; every other
ordinary tensor return has a separate output slot. Same-level CLUSTER SubWorker
dispatch is rejected explicitly.

CLUSTER `for` loops bind SSA carries before the loop, update them in parallel at
the backedge, and expose final values after the loop, including zero-trip loops.
Ordinary tensor carries use the tensor dictionary; window carries remain metadata.
Metadata-only loop bodies retain a `pass` statement so their Python suite is valid.
Its parent allocation helper has the explicit ABI
`_alloc_intermediates(tensors, *, nodes)`. It binds the node count independently of
the entry and resolves direct `num_workers()` shape queries and their SSA aliases
through the existing scalar-definition resolver. HOST loop emission and its
`_alloc_intermediates(tensors, world_size=1)` ABI remain unchanged.

CLUSTER_0 supports compilation and parent-side artifact preparation with recursive
entry metadata and embedded HOST sources. L4 execution remains gated until runner
integration is complete. `_io_arg` is bound by that future integration; codegen
does not import a nonexistent runner.
L4 `world_size` and collective operations remain unsupported. Having a handler
does not authorize higher levels or calls that skip the adjacent hierarchy level.

## Private embedded HOST source API

The explicit internal call
`DistributedCodegen()._generate_embedded(program, entry_name="node")` emits the
selected HOST Orchestrator's real dispatch body as `def node(orch, args, config)`.
It shares the native emitter with ordinary `generate()`; merely including a HOST
function never changes the ordinary entry-selection rule or ABI. Omitting
`entry_name` requires exactly one HOST Orchestrator, even when the program also
contains a CLUSTER root. Unknown names, other levels and SubWorkers are rejected.

Pass the complete lowered Program. Native precondition verification still checks
the complete hierarchy, including CLUSTER scopes that own a selected HOST body's
window parameters; constructing a detached HOST fragment loses that ownership.

The actual entry requires `node._pypto_embedded_state` to be bound after module
import and before accepting a task. Reserved module aliases keep this lookup
valid when the entry's name matches an ABI parameter or Python builtin.
Its `begin_invocation(orch, args, config)` method supplies a frame with tensor
bindings and a keepalive collection. The
state also supplies registered CHIP callables, SubWorker IDs, local device count
and the domain provider. The frame must survive the Python body's return until
its submitted DAG completes.

`node._pypto_embedded_layout` is an immutable mapping with `version=1` and tuple
fields `tensor_slots`, `scalar_slots`, `window_slots`, `implicit_outputs` and
`prefork_requirements`. Tensor records are
`(formal_position, ssa_name, tensor_index, dtype, direction, shape)`; scalar records
are `(formal_position, ssa_name, scalar_index, dtype)`. The two wire pools have
independent dense indexes. Directions use `In`, `Out` and `InOut`; explicit output
descriptors remain caller supplied.

DistributedTensor formals consume no tensor slot. Their `window_slots` records
are `(formal_position, ssa_name, ctx_formal_position, ctx_scalar_index, window_name,
dtype, direction, shape)`. Pairing uses strict DT formal order and only the final
N CommCtx formals added by context materialization. Earlier CommCtx parameters
remain in their original scalar order with dtype `comm_ctx`; names do not choose
the pairs. Native WindowBuffer pointer identity chooses the generated binding.

All received CommCtx values are global domain IDs. Before dispatch, state resolves
each unique ID with `orch.get_global_domain`, checks its committed generation,
required local worker contexts and named window slots, and binds the real view.
Materialization uses `view[local_worker_id].buffers[name].tensor(...)`; forwarded
contexts use that worker's `device_ctx`, and builtin rank counts use its
`domain_size`. Local worker IDs, global ranks and domain sizes are independent.

Implicit output records are `(return_position, tensor_index, dtype, shape)` and
follow all ordinary tensor formals, before the independent scalar pool. A returned
input or internal result never replaces its input binding: the entry schedules
`_copy_return_tensor` as a registered CPU SubWorker with source `INPUT` and
destination `OUTPUT_EXISTING`. The native runtime infers dependencies on pending
CHIP/SUB producers from their canonical buffer identities. The copy consumes real
mapped views, validates matching dtype/shape, direct load/store rights, bounds,
alignment and destination strides, and preserves offsets and padding. Flat
multi-tensor returns retain per-position aliases and Submit's final TaskId.
Consuming scalar returns has an explicit unsupported-transport diagnostic; unused
scalar returns remain permitted.

The cached plan reads `ExplicitReturnedParamIndices` for the actual
`NormalizeReturnOrder`/`ReturnParamsExplicit` domain (InCore/AIC/AIV, Group/Spmd,
Graph). Those passes exclude hierarchy Opaque and Orchestration functions; only
those excluded functions use the existing general SSA return-lineage resolver.
When a function has no return annotation, the same cached plan derives one type
per terminal return operand, matching call-type inference without changing its
declared signature or flattening a packed tuple.

The relative `_distributed_support.py` closure owns tensor conversion, CHIP
dispatch/DFX and `_EmbeddedEntryState`. It imports no PyPTO/compiler modules.
Standalone generated HOST imports remain compatible through parent aliases and
a thin DeviceTensor provenance wrapper. Signed scalar widths, BOOL, INDEX and
FP32 decode from real uint64 slots. The neutral scalar helpers additionally
preserve FP64 via `ctypes.c_double`; PyPTO currently exposes no FP64 formal.

State checks both argument counts before binding, resets CHIP placement and DFX
per invocation, and refuses reuse until `complete_invocation()` is called after
the corresponding runtime DAG drains. `close()` retires state after worker
shutdown. These methods allocate Python metadata only; storage must be prepared
before the inner fork.

Entries with implicit outputs declare `("return_copy_subworker", 1)` in
`prefork_requirements`. Startup must provide at least one CPU SubWorker, register
the canonical copy callable before initialization, and bind its handle through
`return_copy_callable`. Constant internal storage is declared as
`("tensor_storage", ssa_name, dtype, shape)`. State construction checks every
required prepared value is either shared CPU Torch storage or a real HOST
descriptor with the declared dtype/shape. It allocates no tensor storage. Outgoing
CHIP and copy TaskArgs remain in the same invocation frame until explicit DAG
completion; returning from the Python entry does not retire them.

`_EmbeddedNodeState(local_worker_ids=..., prepared_domain_tokens=...)` supplies
fixed shared CPU int32[1] token bundles before the inner fork. Bind the same node
state to all entries through `node_state`. A canonical `(domain_id, generation)`
lease shares one worker-specific token across aliases, windows, explicit ctx-only
dispatches and entries. Both custom and builtin dispatch append deduplicated
INOUT tokens in argument-first-reference order, before every scalar. Builtin
window arguments must agree on a physical domain.

The `inherited_domain_tokens` requirement declares the maximum distinct incoming
ctx count; acquisition checks the actual deduplicated keys against prepared
capacity. `domain_dispatch` records required worker IDs (a dynamic selector
requires coverage of the full local topology), and `same_domain` records alias
or builtin agreement checks. All decoding, view/context/slot checks and capacity
preflight precede frame/lease publication. Failed preflight leaves the state
retryable. Completion releases active frame references; explicit
`retire_domain(id, generation)` returns a completed lease's slot for reassignment
and refuses while any entry frame still uses it. `close()` requires drained
leases and releases the bounded storage references.

HOST-owned domains retain their existing allocation provider/context manager.
Their separate `local_domain_tokens` requirement names prepared worker-indexed
tokens; it does not assume global ID/generation fields on a local handle. No
invocation path allocates or shares token storage. Startup, global import/commit
and physical-domain lifetime integration remain runner work. Already-lowered
inherited builtin source is supported here; public inherited HOST collective
lowering remains a separate optional extension. Existing local HOST public
collective lowering is unchanged.

Ordinary tensor formal shapes retain integer dimensions in layout v1. A dynamic
dimension is `("var", canonical_name)` or
`("mul", ("var", canonical_name), positive_integer)`; other expressions are
rejected. These immutable opcode tuples contain no IR objects or executable
strings. C3c must persist and strictly validate them. Before publishing a frame,
state checks received dtype, rank, fixed axes, exact multiplication divisibility
and consistent variable values across every formal. Generated dynamic bindings
read native descriptor `.shapes`.

Embedded ordinary tensor slices compose byte offsets and element strides over
the original BufferDescriptor, including unit-axis rank reduction. Contiguous,
product-preserving ND `tensor.view`/`tensor.reshape` preserve the same backing,
offset and dtype; noncontiguous reinterpretation, partial validity and padding
are rejected. Native Tensor construction checks descriptor bounds. No view
allocates or maps storage. Embedded `tensor.assemble` writes are unsupported and
rejected during generation. Already-lowered distributed-window formals preserve
full-valid ND element strides and constant byte offsets. A nonempty MemRef must
identify the exact owning WindowBuffer base. The sole normalization for a
function-local origin is a constant-zero, static dense full-valid ND view whose
overflow-checked logical storage bytes equal both the named slot size and the
MemRef size; other unknown origins are rejected. The offset is emitted once relative to the named Buffer's
slot origin, with no element-size multiplication or additional physical-slot
offset. Native Tensor construction retains descriptor provenance and checks
bounds. Public DT Slice and shape-view operations with unresolved slot geometry
give an early generation diagnostic. DT/ctx-only tuples, aliases and projections
remain IR metadata. CommCtx projections follow actual tuple definitions and
aliases back to the exact received formal or window query, with no new scalar
wire value. Their producer calls and Submit TaskId/dependencies
still execute; they create no Python tensor slots or returned ctx wire values.
Embedded `For` carries read their live SSA values, apply backedges in parallel
and expose initial values after zero iterations. Nested loops initialize from
the live outer carry. DT/ctx metadata keeps its original yield/phi positions
without Python tensor slots; metadata-only `For` and `If` suites remain valid.
Before dispatch/layout emission, CommCtx carries must prove the same exact
received formal or owned-window query origin at initialization and every
backedge. Context phis require that origin on both branches. Stable results
reuse that origin after the loop/branch; changing or unresolved origins receive
a generation diagnostic rather than a new scalar wire value.
Internal storage must have constant shapes and be prepared before
initialization; dynamic formal geometry does not authorize dynamic allocation.
Recursive source packaging is available for CLUSTER_0. Actual L4 execution
remains gated until node startup and I/O routing integration is complete.

## Recursive CLUSTER artifacts

CLUSTER_0 compilation emits `orchestration/cluster_orch.py` and one reachable
HOST subtree at `next_levels/<host>/`. Each HOST uses the unchanged private
native `(orch, args, config)` entry and owns its reachable CHIP sub-builds at
`next_levels/<host>/next_levels/<chip>/`, plus its HOST SubWorker sources.
Shared CHIP function names retain separate full logical registration paths;
unreachable or stale directory contents do not select additional children.
Ordinary HOST generation and schema-3 metadata retain their existing format.

Schema-4 `l3_entries` closure version 1 records each entry's relative module and
qualname, native layout version 1, independent dense tensor/scalar indices,
omitted distributed-window/final-CommCtx pairs, shapes/directions, prefork
storage/domain-token requirements, local device count and minimum SubWorkers.
A separate `view_geometry` version 1 stores window strides and byte offsets;
the native eight-field window rows are unchanged. Returned tensor copies name
the canonical helper callable explicitly. Cold restoration validates all fields
without executing source. Identity-only schema-4 records remain readable but
cannot become prepared artifacts without their versioned closure.

Parent-side CHIP compilation and prebuilt promotion use the declared full paths.
`runtime._artifact_sources.build_cluster_source_bundle` returns a SHA-256 digest
and deterministic gzip-compressed USTAR bytes containing only owned runtime sources, cold metadata
and a path/size/digest inventory. One canonical neutral helper lives under root
`orchestration/`; relative HOST package shims re-export it. The enclosing digest
namespace is supplied during installation and never enters source bytes. The bundle
is directly accepted by R1 `StartupArtifactSpec` read/install/activation APIs.
Before parsing source, the producer checks portable paths and bounds each file
to 64 MiB, the complete unpacked closure to 256 MiB and the member count to 4096
(including the inventory). Streaming gzip output, including its trailer, is
bounded to 64 MiB. Windows reserved components and case-folding path aliases
are rejected on all platforms. Installed
sources consume registered CHIP handles without PyPTO imports, compiler discovery,
`kernel_config.py` execution or `ArtifactRuntime.load`. The parent may still use
those compiler/assembly facilities. L4 execution remains guarded; CLUSTER_1 and
higher compilation is unsupported.

## See Also

- [Passes](../passes/index.md) — everything that must hold before codegen runs.
- [PTO ISA reference](../../reference/index.md) — the instruction semantics the generated code uses.
- [PTOAS Op Status Matrix](../ptoas-op-status.md) — which PTOAS ops are available to emit.
