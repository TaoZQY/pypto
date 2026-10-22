# MaterializeCommDomainScopes Pass

## Overview

`MaterializeCommDomainScopes` materializes local communication domains in HOST
and CLUSTER_0 orchestration functions. It assembles the host-side metadata that
the distributed runtime needs in order to size and
populate per-rank communication windows. It is the structural analogue of
[`InitMemRef`](35-init_memref.md): it traces an allocation through to its
consumption points, constructs a back-reference object, and threads it onto
the IR types so downstream codegen has O(1) access.

| Aspect | `MemRef` side | `WindowBuffer` side |
| ------ | ------------- | ------------------- |
| Allocation op | `tile.alloc(memory_space, size_in_bytes)` | `pld.tensor.alloc_window_buffer(size_in_bytes)` |
| Assignment LHS at parse time | `Var(PtrType)` | `Var(PtrType)` (same singleton) |
| Wrapper Var subclass | `MemRef` | `WindowBuffer` |
| Wrapper's SSA-edge type | `MemRefType` (singleton) | `WindowBufferType` (singleton) |
| Built by | `InitMemRef` | **`MaterializeCommDomainScopes`** (this pass) |
| Threaded back onto | `TensorType.memref_` | `DistributedTensorType.window_buffer_` |
| IR-level registry | `Program.functions_` (alloc stmts) | `CommDomainScopeStmt` in each local owner's body (one per inferred comm domain) |

## Position in the pipeline

```text
... -> ExpandManualPhaseFence -> SynthesizeAllReduceSignals -> MaterializeCommDomainScopes -> LowerHostTensorCollectives -> MaterializeDistTensorCtx -> Simplify (final)
```

The pass runs near the end of the default pipeline, immediately before
[`LowerHostTensorCollectives`](48-lower_host_tensor_collectives.md) and the final
`Simplify`. None of the intervening passes between `InlineFunctions` and here
touches the host_orch alloc/window/dispatch chain: host_orch is never
tile-lowered, and L2 (chip-level) orchestrations are never inlined into L3, so
the alloc / view / dispatch sites this pass needs are still discoverable. Running
late keeps the producing IR fully canonicalised before the descriptor analysis
kicks in, and any constant folding that the trailing `Simplify` does on the
collected sizes is applied uniformly.

## Algorithm

The pass handles local owners at `Level::HOST` and `Level::CLUSTER_0` with
`Role::Orchestrator`; canonical `FunctionType::Orchestration` declarations are
also accepted. Coverage is inferred only from adjacent dispatches:
`HOST -> CHIP` and `CLUSTER_0 -> HOST`. A CHIP `Graph` is not a HOST dispatch
target. HOST collective analysis and signal synthesis retain their HOST-only
contract.

1. **Collect local allocations.** Index each owner's allocation/window pairs,
   skipping already materialized scope bodies. Create one `WindowBuffer` per
   local allocation. A formal inherited from another owner never becomes a local
   slot, so a HOST function can use both its own domains and an inherited
   CLUSTER_0 domain.
2. **Index program identity.** Build a graph over values, tuple positions and
   declared return positions. Seed local windows and pre-existing identities.
   Cache each callee's trailing `CommCtx` count and explicit return-to-parameter
   map once. Bind actuals to formals for resolved `Call` and `Submit` expressions.
   Supplied `Out` operands participate; omitted runtime `Out` operands have no
   caller provenance and cannot consume a trailing context argument.
3. **Propagate identity once.** Follow SSA aliases, registry-declared aliasing
   views/writebacks, managed collective results, explicit returns, tuple
   construction/projection, and loop init/`IterArg`/yield/return edges. Branch
   merges require both inputs to resolve to the same object. An unknown input
   does not prove consensus. Arbitrary allocating tensor operations do not
   establish window identity. A value accepts one object; a second distinct
   object is an error even when names, shapes and dtypes match. Repeated callers
   using the same object are accepted without function cloning.
4. **Infer local coverage.** Resolve a dispatch argument's identity to the
   current owner's local allocation index. Inherited windows are not inserted
   into this index. The canonical placement attribute is `worker`; the parser
   normalizes the deprecated `device` spelling.

   | `worker=` expression | Coverage |
   | -------------------- | -------- |
   | Non-negative `ConstInt(N)` | `{N}` |
   | Induction variable of `[0, num_workers())`, step 1 | All children |
   | Induction variable of `[0, world_size())`, step 1, at HOST | All children |
   | Induction variable of a static unit-step range | That range |
   | Unsupported or unproven dynamic start/step | Error |

   Direct and SSA-hoisted child-count queries use the same memoized bound
   analysis. `world_size` remains illegal at CLUSTER_0. HOST collective signals
   and `all_to_all_v` count windows retain the existing paired-coverage rules.
5. **Group and rewrite.** Merge coverage per local allocation, group matching
   descriptors by canonical snapshot ID, and retain allocation order within each
   group. One final mutator rebuilds affected signatures, body and function-attr references,
   preserving each value's own shape, dtype, `MemRef` and tensor view while
   replacing only `window_buffer_`. It also updates declared returns, Call/
   Submit tuple result types, projections and loop types. Submit dependencies,
   attributes, launch fields, predicate and trailing `TASK_ID` are preserved.
   Wrap only real local allocations in nested `CommDomainScopeStmt` nodes;
   previously owned windows retain their existing objects.

The IR traversal and identity work queue are `O(N + E)`: each identity is
assigned once and each edge is processed once. Type relevance is memoized over
the shared type DAG; tuple lineage stores only DT-bearing positions, retaining
their original ABI indices. Ordinary scalar tuple fields are not expanded per
value. Changed tuple-root flags are computed once, then a bounded boolean queue
propagates mutation requirements through child-to-parent expression edges. This
also catches a scalar or context root containing a changed DT call. Function/root
use edges mark every affected function, including native functions sharing one
formal, body or attribute expression object. Functions without identity or
local-domain changes bypass mutation. Call/Submit result bindings store only
DT-bearing `(original ABI index, value)` entries, so a wide scalar-only return
list is not scanned at every call. The shared base mutator memoizes scalar-only
type trees using strong immutable `TypePtr` keys: these trees contain no embedded
expressions or subclass callbacks, including in Call/Submit result types.
Tensor shapes, views and other expression-bearing types still use the shared
remapper. Direct dispatch coverage uses
mutable interval maps (`O(D log D)` for `D` interval insertions), then freezes
once per allocation. Numeric ranges stay as intervals until a domain's sorted
device list is emitted. Allocations share canonical immutable coverage snapshots;
collective consumers update destination snapshots in source order, caching each
distinct snapshot-pair union. Grouping uses snapshot IDs, so identical large
sets are not copied or compared once per consumer or allocation.

Coverage retains a complexity exception: each uncached collective union costs
`O(I_left + I_right)` interval work plus ordered canonical-key comparisons;
comparisons can inspect interval prefixes. Stored snapshots consume the sum of
their interval counts, and emitted device lists cost their total member count.
Many distinct overlapping unions can still take quadratic work in IR size even
with a small final output. This residual bound requires explicit approval before
merge under the pass-complexity rule; no universal `O(N log N)` bound is claimed.
The pass does not rescan bodies to a fixed point, scan all formals per call, or
retrace alias chains per use. Existing scopes make repeated application idempotent.

## Diagnostics and output invariants

- Missing views and allocations without a consuming adjacent dispatch retain
  their existing diagnostics. Unsupported coverage expressions, ambiguous
  identity and non-null identities without an owning scope are rejected.
- Every materialized window and its supported aliases, callee formals, returns
  and projections share the exact same `WindowBuffer` object as the owning
  scope slot. Nullable standalone device formals can remain unbound.
- A scope owns only its function's local slots. Inherited slots are never
  copied into a child scope. Function iteration order cannot select a caller's
  identity or change the result.
- Assignments remain type-symmetric; preserving original shape/view metadata
  leaves genuine type mismatches visible to verification.
- HOST `allreduce` still has an explicit or synthesized signal after
  [`SynthesizeAllReduceSignals`](46-synthesize_allreduce_signals.md).
  The existing HOST `all_to_all_v` restriction inside `for`/`while` remains.

This is the bounded domain/identity stage. It does not add CLUSTER_0 collective
lowering, change context materialization, or enable the complete L4 `compile`
pipeline; those require their later lowering/runtime stages.

## Text round-trip

The ordinary authoring DSL continues to infer communication domains. Printed IR
records the materialized state using private `pl._dump` spellings:

- `comm_domain(name=..., devices=..., slots=[...], attrs={...})` preserves the
  scope boundary, device list, ordered slots, and attributes.
- `window_buffer("buf_window", base="buf", size=16, load_from_host=..., store_to_host=...)`
  defines a slot; `window_ref("buf_window")` refers to that same object from tensor types.
  The label is a text-format reference, not a device ID, slot index, or IR field.
  Labels derive from the printed base variable; suffixes distinguish different
  windows sharing that base. Earlier numeric IDs remain readable.
- `alloc_window_buffer(size, name=...)` preserves the allocation's logical name
  independently of an SSA-renamed result variable.
- `call(op, args, result_type, kwargs=..., attrs=...)` restores a Call directly,
  accepting any result type supported by the type parser. The printer uses it
  for distributed calls whose recorded type must be preserved, including an
  absent back-reference; inference can differ after a pass rewrites the operands.

The parser records window descriptors before parsing the body and resolves them
when their allocation and size expressions are available. Scope slots and tensor
back-references share one reconstructed `WindowBuffer`. These spellings are IR
text syntax, not operations that execute in a kernel.
The descriptor's `base` names the allocation variable defined in the body;
`window_buffer(...)` adds no allocation statement. Its `size` and the allocation
Call's size argument are separate recorded IR fields and are both retained.
Empty scopes print `pass`. Scalar attribute literals such as `-1` remain scalars;
`pl.const(-1, pl.INDEX)` records an IR expression instead.

For a self-contained program, `parse(python_print(program))` must be structurally
equal to `program`, with equal structural hashes. No lowering pass is rerun by
the parser or required from the caller. Statement fragments with references to
absent definitions are outside this contract. Legacy informational
`"window_buffer=<name>"` markers remain readable but do not encode a reference.

## Pass properties

| Field | Value |
| ----- | ----- |
| `required` | `{}` |
| `produced` | `{IRProperty::CommDomainScopesMaterialized}` |
| `invalidated` | `{}` |

## Reference

- Source: [src/ir/transforms/materialize_comm_domain_scopes_pass.cpp](../../../../src/ir/transforms/materialize_comm_domain_scopes_pass.cpp)
- Header: [include/pypto/ir/transforms/passes.h](../../../../include/pypto/ir/transforms/passes.h)
- Schema: [include/pypto/ir/program.h](../../../../include/pypto/ir/program.h)
  defines `WindowBuffer`;
  [include/pypto/ir/stmt.h](../../../../include/pypto/ir/stmt.h) defines
  `CommDomainScopeStmt`.
- DSL: [`pld.tensor.alloc_window_buffer`](../../../../python/pypto/language/distributed/op/tensor_ops.py),
  [`pld.tensor.window`](../../../../python/pypto/language/distributed/op/tensor_ops.py),
  [`pld.system.world_size`](../../../../python/pypto/language/distributed/op/system_ops.py).
