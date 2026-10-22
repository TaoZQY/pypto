# 代码生成

把 PyPTO IR lower 成已编译程序的两类产物：设备侧 kernel，以及启动它们的主机侧编排代码。

两个生成器遵循同一条设计原则 —— 从 IR 到生成代码的**严格 1:1 映射**。任何需要做决策的
地方，都已由某个 pass 决定完毕。

| 页面 | 内容 |
| ---- | ---- |
| [PTO Codegen](00-pto_codegen.md) | 从 PyPTO IR 生成 PTO-ISA 方言的 MLIR |
| [编排代码生成](01-orchestration_codegen.md) | 生成向运行时提交任务的主机侧 C++ |
| [PTO 内核包装器生成](02-kernel_wrapper.md) | 将 ptoas 生成的内核连接到运行时派发 |

## 分布式层级格式化

`DistributedCodegen` 通过内部无状态的层级处理器（`hierarchy::LevelHandler`，
位于 `include/pypto/hierarchy/level_handler.h`）生成 Python 编排代码。其七个方法负责
域分配器与成员列表的格式化、上下文取值、完整的普通主机张量添加语句、窗口张量
物化能力、入口关键字参数及导入语句。生成器继续负责遍历、分配准备、张量与标量分组、
顺序令牌归属、显式 Submit 依赖、动态维度及显式 CommCtx 参数转发。
域成员的关键字与额外分配参数也由生成器选择。

`GetLevelHandler(ir::Level)` 注册 HOST 和 CLUSTER_0（别名 POD），对所有不支持的层级返回 `nullptr`；枚举值
与 Linqu 层级编号不同。现有生成器发射位置显式回退到 HOST 格式化策略，以保留未注册
层级入口的历史输出。该回退不授予任何能力。导入语句由所选的最高层入口决定；
独立 HOST 入口的 ABI 和生成源码保持不变。

| 调用边 | HOST → CHIP | CLUSTER_0 → HOST |
| ------ | ----------- | ---------------- |
| 子节点数量 | `world_size`（本地设备数） | `num_workers = len(nodes)` |
| 普通张量 | `add_chip_tensor_arg(...)` | `_io_arg(node_bindings[index], tensor, direction)` |
| 窗口张量 | 从域的 Buffer 物化 | 不占 wire tensor 槽位 |
| 上下文标量 | 目标子节点的 `device_ctx` | 全局 `domain_id` |
| 域成员 | `workers=[...]` | 按拓扑顺序展开 `members=[(nodes[n], d), ...]` |
| 下发 | `_submit_chip(...)` | 各节点的 L3 handle，指定 `worker=nodes[index]` |

CLUSTER 派发在指定 `worker=` 时使用该节点索引。无通信且省略它的调用按节点数轮询选择，
逻辑索引只解析一次，供张量路由、callable 与实际 worker ID 共同使用；每次入口调用都
重置该序列，且不使用本地设备数。域缓冲区保留现有的物理对齐与
逻辑字节数；全局域分配接受 `_domain_provider` 和 `retain_after_run=_retain`。
CHIP 顺序令牌仍只用于 HOST。所有张量先于标量添加，两组内部各保持形参顺序，
并保留显式 Submit 依赖、TaskId 结果和输出别名。
隐式张量返回存储也通过节点路由绑定，以 `OUTPUT_EXISTING` 打包；这只说明参数打包约定，
嵌入式 HOST 返回 ABI 仍待实现。同层 CLUSTER SubWorker 派发会被明确拒绝。

CLUSTER 的 `for` 循环在循环前绑定 SSA 携带值，在回边并行更新，并在循环后暴露最终值，
也覆盖零次迭代。普通张量携带值使用张量字典，窗口携带值仍只保留元数据。
仅包含元数据的循环体保留 `pass` 语句，以保证生成的 Python 语法有效。
其父进程分配辅助函数采用显式 ABI `_alloc_intermediates(tensors, *, nodes)`，
独立于入口绑定节点数，并通过现有标量定义解析器处理形状中的直接 `num_workers()` 查询
及其 SSA 别名。HOST 的循环发射与 `_alloc_intermediates(tensors, world_size=1)` ABI 保持不变。

CLUSTER 当前只支持内部生成源码。公共 L4 编译、产物执行和 runner 继续保留闸门，
待递归入口元数据、嵌入式 HOST 入口与 runner 集成完成后逐步开放。
`_io_arg` 由后续集成绑定；codegen 不导入尚不存在的 runner。
L4 的 `world_size` 和集合通信仍不支持。存在 handler 不代表放开更高层级或跨层跳级调用。

## 另请参阅

- [Passes](../passes/index.md) —— codegen 运行前必须成立的全部前提。
- [PTO ISA 参考](../../reference/index.md) —— 生成代码所使用的指令语义。
- [PTOAS 算子状态矩阵](../ptoas-op-status.md) —— 可供发射的 PTOAS 算子范围。
