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

## Distributed HOST formatting

`DistributedCodegen` emits Python orchestration using the internal, stateless
`hierarchy::LevelHandler` (`include/pypto/hierarchy/level_handler.h`). Its HOST
implementation owns domain allocator and membership formatting, local device
contexts, complete ordinary host-tensor append statements, window materialization
capability, entry keyword parameters, and imports. The generator retains traversal,
allocation preparation, tensor/scalar grouping, ordering-token ownership, explicit
Submit dependencies, dynamic dimensions, and explicit CommCtx parameter forwarding.

`GetLevelHandler(ir::Level)` registers HOST only and returns `nullptr` for every
unsupported level; enum values are distinct from Linqu level numbers. At existing
generator emission sites, an explicit HOST-formatting fallback preserves legacy
output for entries at unregistered levels. This fallback neither authorizes a
capability nor implements Cluster dispatch. Provider-acceptance classification
remains in the generator, and builtin collective emission remains HOST-specific.

## See Also

- [Passes](../passes/index.md) — everything that must hold before codegen runs.
- [PTO ISA reference](../../reference/index.md) — the instruction semantics the generated code uses.
- [PTOAS Op Status Matrix](../ptoas-op-status.md) — which PTOAS ops are available to emit.
