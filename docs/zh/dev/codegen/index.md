# 代码生成

把 PyPTO IR lower 成已编译程序的两类产物：设备侧 kernel，以及启动它们的主机侧编排代码。

两个生成器遵循同一条设计原则 —— 从 IR 到生成代码的**严格 1:1 映射**。任何需要做决策的
地方，都已由某个 pass 决定完毕。

| 页面 | 内容 |
| ---- | ---- |
| [PTO Codegen](00-pto_codegen.md) | 从 PyPTO IR 生成 PTO-ISA 方言的 MLIR |
| [编排代码生成](01-orchestration_codegen.md) | 生成向运行时提交任务的主机侧 C++ |
| [PTO 内核包装器生成](02-kernel_wrapper.md) | 将 ptoas 生成的内核连接到运行时派发 |

## 分布式 HOST 格式化

`DistributedCodegen` 通过内部无状态的层级处理器（`hierarchy::LevelHandler`，
位于 `include/pypto/hierarchy/level_handler.h`）生成 Python 编排代码。HOST 实现负责
域分配器与成员列表的格式化、本地设备上下文、完整的普通主机张量添加语句、窗口张量
物化能力、入口关键字参数及导入语句。生成器继续负责遍历、分配准备、张量与标量分组、
顺序令牌归属、显式 Submit 依赖、动态维度及显式 CommCtx 参数转发。

`GetLevelHandler(ir::Level)` 只注册 HOST，对所有不支持的层级返回 `nullptr`；枚举值
与 Linqu 层级编号不同。现有生成器发射位置显式回退到 HOST 格式化策略，以保留未注册
层级入口的历史输出。该回退不授予任何能力，也不实现 Cluster 派发。是否接受域提供器
仍由生成器分类判断，内置集合通信的发射仍限定于 HOST。

## 另请参阅

- [Passes](../passes/index.md) —— codegen 运行前必须成立的全部前提。
- [PTO ISA 参考](../../reference/index.md) —— 生成代码所使用的指令语义。
- [PTOAS 算子状态矩阵](../ptoas-op-status.md) —— 可供发射的 PTOAS 算子范围。
