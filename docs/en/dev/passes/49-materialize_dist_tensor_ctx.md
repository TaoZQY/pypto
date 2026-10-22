# MaterializeDistTensorCtx Pass

Materializes one explicit `CommCtxType` parameter and argument for each
`DistributedTensorType` function parameter.

## Overview

Distributed tensors need a communication context at every dispatch boundary,
including the CLUSTER_0 (L4) → HOST (L3) → CHIP (L2) chain. Host orchestration
passes a per-rank `device_ctx`, L2 orchestration forwards it through task args,
and L1 PTO codegen uses it to lower `pld.system.rank`,
`pld.system.nranks`, `notify`, `wait`, `put`, and remote memory ops.

Older codegen paths synthesized those ctx values independently at several
sites. This pass makes the ctx flow explicit in IR instead:

1. For every function with `DistributedTensorType` parameters, append matching
   `CommCtxType` parameters at the tail of the signature, in distributed-tensor
   parameter order. The appended parameters are `ParamDirection::In`.
2. For every `Call` / `Submit` to such a function, append matching ctx args.
   If the distributed tensor arg is a caller parameter or an SSA alias of one,
   forward the caller's materialized ctx parameter. Return positions are
   matched to the callee's returned parameters, so mixed or reordered return
   tuples do not fall back to positional tail alignment. This holds for `Submit`
   too: its result positions are the callee's return positions (the trailing
   `Scalar[TASK_ID]` has no ctx). Builtin ops that bind a fresh SSA var to a
   DistributedTensor that already exists forward that value's ctx — both
   output-side writebacks, which the op declares itself via
   `set_output_reuses_input(idx)` (`tile.store` -> `args[2]`,
   `tensor.assemble` -> `args[0]`), and zero-copy buffer-aliasing views
   (`tensor.view`, `tile.slice`, `tensor.reshape`, ...), whose result type
   propagates `DistributedTensorType::window_buffer_` from `args[0]`.
   Tensor aliases carried through `ForStmt` / `WhileStmt` are tracked as well.
   In HOST or CLUSTER_0 orchestration, if the lineage cannot be resolved, bind
   `pld.system.get_comm_ctx(dist)` immediately before the call and pass that
   result. Chip orchestration and device functions must resolve an explicit
   context; an unresolved argument is diagnosed instead of synthesizing a
   device-side query.
3. In chip orchestration and device functions, replace every
   `pld.system.get_comm_ctx(dist)` with the resolved explicit `CommCtxType` SSA
   value. HOST and CLUSTER_0 orchestrators keep the op. HOST codegen resolves it
   from the window's per-rank runtime context; interpreting a CLUSTER_0 context
   at runtime belongs to subsequent lowering.
4. If call-site `arg_directions` are already resolved, append matching
   `ArgDirection::Scalar` entries so downstream codegen can keep treating ctx as
   ordinary scalar task payload.

`Submit` arguments retain their canonical layout: a caller-supplied prefix,
followed by the existing and newly appended `CommCtx` suffix. The only omitted
parameters form a contiguous `Out` gap immediately before that suffix. For
example, original parameters `[data, scratch: Out[Tensor], existing_ctx]` and
arguments `[data, existing_ctx]` become parameters
`[data, scratch, existing_ctx, data_ctx]` and arguments
`[data, existing_ctx, data_ctx]`. A supplied `Out` is real caller provenance;
an omitted `DistributedTensor` `Out` has no caller context and is diagnosed as
unsupported. This pass does not allocate a runtime context for that output.
The pass caches each original signature, arity, and trailing-context count
before extending any signature. Alias analysis binds original calls against
those facts; result lookup in recursively rewritten argument expressions uses
the extended signature and suffix. An argument count alone cannot distinguish
these stages when an `Out` gap is present.

Result provenance uses one original-position map per callee. A call returning
distributed tensors captures only the contexts of distinct returned DT
parameters, bounded by its supplied arguments, and tuple aliases share that
immutable capture. Scalar-only and empty returns allocate no result-context
capture. Direct nested projections resolve only their requested position;
repeating one DT parameter across a wide return tuple does not allocate or
resolve a context for every return field at every call site.

The pass also classifies immutable shared result-type DAGs once. Scalar/tuple
trees and Tensor/DistributedTensor metadata containing only integer literals,
without a MemRef, can skip the base type walk because this pass cannot rewrite
those leaves. The classification follows the canonical shape/view walker and
keeps window ownership opaque. It caches only this proof, not rewritten types:
MemRefs, dynamic shape/stride expressions, Vars, IterArgs, queries, and unknown
types retain the normal state-sensitive base traversal. Repeated uses of those
unsafe shared types may still repeat their metadata walk.

Expr-valued function attributes are rewritten by stored type, including
supported expression containers, so their calls, submits, and context queries
retain the updated ABI. Attributes have no statement insertion point: a call
there must resolve its context from existing formal/alias lineage, or receive a
clear unsupported-context diagnostic. Ordinary metadata, call kwargs, return
positions, argument directions, and Submit launch/dependency fields are
preserved. This pass does not register a CLUSTER_0 level handler or change the
HOST-only collective and signal lowering contracts.
Binary roundtrips preserve shared window references in function bodies and
metadata, including locally owned slots first defined by a body expression.

This pass does not add `CommCtxType` values to `IfStmt` return variables or
branch yields. DistributedTensor `if` lowering keeps the existing requirement
that both branches refer to the same backing/context; dynamic context merges
remain outside this change (issue #2027).

A loop carry is subject to the same one-context rule. The carry is seeded from
its init value before the body is walked, so a self-carry
(`data = self.comm(data)`) resolves; the value yielded back into the carry is
then checked against that seed, and rebinding the carry to a *different*
DistributedTensor inside the loop is diagnosed rather than silently taking the
init value's context. A yield whose lineage this pass cannot trace at all leaves
the seed in place.

The pass produces `IRProperty::DistTensorCtxMaterialized`: no
`pld.system.get_comm_ctx` survives outside HOST or CLUSTER_0 orchestration. The
pass enforces this for every function it rewrites, and the property verifier
checks it independently, including Expr-valued function attributes. This also
covers Programs the pass returns untouched because no function declares a
`DistributedTensorType` parameter. The property is listed in
`GetVerifiedProperties()`, so the pipeline checks it on the default verification
level rather than only when a test installs a `VerificationInstrument`.

It requires `IRProperty::ReturnParamsExplicit`: the return-position map comes
from `return_lineage::ExplicitReturnedParamIndices`, a pointer-identity read of
the `ReturnStmt` that is only meaningful once `NormalizeReturnOrder` has
canonicalized it.

The pass runs after `LowerHostTensorCollectives` and before the final
`Simplify`. At that point host window buffers have already been materialized by
`MaterializeCommDomainScopes`, host tensor collectives have been lowered, and
there is still time for the final simplify pass to clean up any forwarding
aliases.

## Why CommCtx Is Different From Dynamic Dims

Dynamic tensor dimensions can be recovered locally from tensor descriptors at
the wrapper boundary. A communication context cannot: it is real dataflow across
host -> orchestration -> task payload -> kernel signature. Keeping it in IR
prevents codegen sites from drifting out of sync.

## API

| C++ | Python | Level |
| --- | ------ | ----- |
| `pass::MaterializeDistTensorCtx()` | `passes.materialize_dist_tensor_ctx()` | Program-level |

```python
from pypto.pypto_core import passes

program = passes.materialize_dist_tensor_ctx()(program)
```

## Example

Before:

```python
def chip_orch(self, data: pld.DistributedTensor[[256], pl.FP32]):
    return self.kernel(data)

def host_orch(self):
    data = pld.window(buf, [256], dtype=pl.FP32)
    self.chip_orch(data, device=r)
```

After:

```python
def chip_orch(self, data, data_ctx: pld.CommCtx):
    return self.kernel(data, data_ctx)

def host_orch(self):
    data = pld.window(buf, [256], dtype=pl.FP32)
    data_ctx = pld.system.get_comm_ctx(data)
    self.chip_orch(data, data_ctx, device=r)
```

The kernel body does not need to change. Existing
`pld.system.get_comm_ctx(data)` uses in a device function are rewritten to the
explicit ctx parameter by this pass; HOST and CLUSTER_0 orchestration uses remain
runtime queries.
