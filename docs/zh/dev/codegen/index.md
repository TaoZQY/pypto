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
隐式张量返回存储也通过节点路由绑定，以 `OUTPUT_EXISTING` 打包。调用方打包与嵌入式
HOST 返回共用按返回位置记录的计划：实际返回的 Out/InOut 复用其调用方存储，其余每个
普通张量返回均占独立输出槽。同层 CLUSTER SubWorker 派发会被明确拒绝。

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

## 私有嵌入式 HOST 源码接口

显式内部调用
`DistributedCodegen()._generate_embedded(program, entry_name="node")`
将所选 HOST Orchestrator 的实际派发函数体生成为 `def node(orch, args, config)`。
该接口与普通 `generate()` 共用原生发射器；仅包含 HOST 函数不会改变普通入口选择规则或 ABI。
省略 `entry_name` 时必须恰好存在一个 HOST Orchestrator，即使程序还包含 CLUSTER 根入口。
未知名称、其他层级以及 SubWorker 均会被拒绝。

必须传入完整的已降低 Program。原生前提验证仍检查完整层级，包括拥有所选 HOST
窗口形参的 CLUSTER 作用域；截取独立 HOST 片段会丢失该所有权关系。

实际入口要求在模块导入后、接收任务前绑定 `node._pypto_embedded_state`。
该查找使用预留的模块别名，因此入口名与 ABI 形参或 Python 内建名称相同时仍然有效。
其 `begin_invocation(orch, args, config)` 方法提供包含张量绑定与 keepalive 集合的帧；
状态还提供已注册的 CHIP callable、SubWorker ID、本地设备数与域 provider。
该帧必须在 Python 函数体返回后继续存活，直至其提交的 DAG 完成。

`node._pypto_embedded_layout` 是不可变映射，包含 `version=1` 以及 tuple 字段
`tensor_slots`、`scalar_slots`、`window_slots`、`implicit_outputs` 和 `prefork_requirements`。
张量记录格式为 `(formal_position, ssa_name, tensor_index, dtype, direction, shape)`，
标量记录为 `(formal_position, ssa_name, scalar_index, dtype)`；两类 wire 索引各自连续。
方向使用 `In`、`Out`、`InOut`，显式输出描述符仍由调用方提供。

DistributedTensor 形参不占张量槽，`window_slots` 记录为
`(formal_position, ssa_name, ctx_formal_position, ctx_scalar_index, window_name, dtype, direction, shape)`。
配对只按严格 DT 形参顺序选择上下文物化追加的末尾 N 个 CommCtx；原有 CommCtx 以
`comm_ctx` 类型保留原标量顺序，不通过名称猜测配对。原生 WindowBuffer 指针身份决定生成绑定。

收到的全部 CommCtx 值均为全局域 ID。派发前通过 `orch.get_global_domain` 按唯一 ID 解析，
校验已提交代际、所需本地 worker 上下文和命名窗口槽，再绑定真实视图。
物化使用 `view[local_worker_id].buffers[name].tensor(...)`，下发上下文使用该 worker 的
`device_ctx`，builtin rank 数使用其 `domain_size`。本地 worker ID、全局 rank 和域大小相互独立。

隐式输出记录为 `(return_position, tensor_index, dtype, shape)`，接在所有普通张量形参后，
与标量池独立。返回输入或内部结果时不会替换原输入绑定：入口将 `_copy_return_tensor`
作为已注册 CPU SubWorker 提交，源标记为 `INPUT`，目标标记为 `OUTPUT_EXISTING`。
原生运行时通过规范缓冲区身份推断其与尚未完成的 CHIP/SUB 生产者之间的依赖。
复制直接使用真实映射视图，校验 dtype/形状一致、直接读写权限、边界、对齐及目标步长，
保留视图偏移和填充字节。扁平多张量返回逐位置保留别名，Submit 的 TaskId 仍位于末尾。
使用标量返回值会得到明确的不支持传输诊断；忽略标量返回仍被允许。

缓存计划在 `NormalizeReturnOrder`/`ReturnParamsExplicit` 实际覆盖的类型
（InCore/AIC/AIV、Group/Spmd、Graph）上读取 `ExplicitReturnedParamIndices`。
这两个阶段明确排除层级 Opaque 和 Orchestration 函数；只有这些未覆盖的函数才使用
现有通用 SSA 返回来源解析器。
函数未标注返回类型时，同一缓存计划按末尾返回语句的每个操作数推导一个类型，与调用
类型推断一致，不修改声明签名，也不展开打包的 tuple。

相对模块 `_distributed_support.py` 统一提供张量转换、CHIP 派发/DFX 与 `_EmbeddedEntryState`，
不导入 PyPTO 或编译器模块。父运行器通过别名及保留 DeviceTensor 来源检查的薄包装，
继续兼容独立 HOST 生成代码。带符号整数各位宽、BOOL、INDEX 与 FP32 按真实 uint64
槽位解码；中立标量辅助函数另通过 `ctypes.c_double` 保留 FP64 位模式，
但 PyPTO 当前尚无 FP64 形参类型。

状态在绑定前校验两类参数数量，每次调用重置 CHIP 选择与 DFX；只有对应运行时 DAG
排空后显式调用 `complete_invocation()`，才允许复用。worker 关闭后通过 `close()` 回收状态。
这些方法仅分配 Python 元数据，张量存储必须在内层 fork 前准备。

存在隐式输出的入口在 `prefork_requirements` 中声明 `("return_copy_subworker", 1)`。
启动方必须提供至少一个 CPU SubWorker，在初始化前注册规范复制 callable，并通过
`return_copy_callable` 绑定其句柄。常量内部存储声明为
`("tensor_storage", ssa_name, dtype, shape)`；状态构造时检查每个所需值为共享 CPU Torch
存储或真实 HOST 描述符，且 dtype/形状符合声明，不分配张量存储。下发的 CHIP 与复制
TaskArgs 均保留在同一调用帧，直到显式确认 DAG 完成；Python 入口返回不代表回收这些引用。

`_EmbeddedNodeState(local_worker_ids=..., prepared_domain_tokens=...)` 在内层 fork 前接收固定的
共享 CPU int32[1] 令牌束；所有入口通过 `node_state` 绑定同一节点状态。
规范键 `(domain_id, generation)` 的租约使别名、窗口、仅显式 ctx 的派发及不同入口共享
同一 worker 令牌。自定义和 builtin 派发都按参数首次引用顺序去重添加 INOUT 令牌，置于
所有标量前；builtin 窗口参数必须属于同一物理域。

`inherited_domain_tokens` 声明收到的 ctx 最大去重数，获取租约时按实际规范键与预备容量比较；
`domain_dispatch` 记录所需 worker ID（动态选择器要求覆盖整个本地拓扑），`same_domain`
记录别名或 builtin 的域一致性约束。解码、视图/上下文/槽检查及容量预检均先于帧和租约发布，
失败后可直接修正重试。完成操作释放活动帧引用；显式 `retire_domain(id, generation)` 归还
已完成租约的槽以供后续代际复用，只要任一入口帧仍持有租约就会拒绝回收。
`close()` 要求租约已排空，并释放有界存储引用。

HOST 自有域保留原分配 provider/上下文管理器；其独立 `local_domain_tokens` 要求声明预备的
按 worker 索引令牌，不假定本地域句柄具有全局 ID/代际字段。调用路径不分配或共享令牌存储。
实际启动、全局域导入/提交及物理域生命周期仍由后续 runner 集成完成。此处支持已降低的
继承域 builtin 源码；公共继承域 HOST 集合通信降低仍是独立可选扩展，本地 HOST 公共
集合通信降低保持不变。

布局 v1 的普通张量形参形状保留整数维度；动态维度表示为 `("var", canonical_name)`
或 `("mul", ("var", canonical_name), positive_integer)`，其他表达式给出拒绝诊断。
这些不可变操作码元组不含 IR 对象或可执行字符串，C3c 必须严格保存和校验。
发布调用帧之前，状态检查收到的 dtype、秩、固定轴、乘法整除关系及所有形参间同一变量
取值的一致性；生成的动态维度绑定读取原生描述符 `.shapes`。

嵌入式普通张量切片在原 BufferDescriptor 上组合字节偏移与元素步长，支持单位轴降秩。
连续、元素数不变的 ND `tensor.view`/`tensor.reshape` 保留同一后备、偏移和 dtype；
非连续重解释、部分有效形状和填充给出拒绝诊断。原生 Tensor 构造检查描述符边界，
视图不分配或映射存储。嵌入式 `tensor.assemble` 写入不受支持，在生成阶段拒绝。
已降低的分布式窗口形参保留完整有效 ND 元素步长和常量字节偏移；存在 MemRef 时，其
base 应是所属 WindowBuffer 的同一 base。函数局部来源仅允许一种规范化：常量零偏移、
静态稠密且完整有效的 ND 视图，经溢出检查的逻辑存储字节数同时等于槽大小和 MemRef
大小；其他无法证明来源的局部分配给出拒绝诊断。
偏移相对于命名 Buffer 的槽原点只应用一次，不乘元素大小，也不重复加入物理槽偏移。
原生 Tensor 构造保留描述符来源并检查边界。槽几何信息尚未确定的公共 DT Slice 和形状
视图在生成阶段给出诊断。仅含 DT/ctx 的元组、别名和投影保留为 IR 元数据；CommCtx
投影沿实际元组定义和别名追溯到精确的收到形参或窗口查询，不增加标量线值。其生产者
调用、Submit TaskId 和依赖仍然执行，不创建 Python 张量槽或返回 ctx 的线值。
嵌入式 `For` 携带值读取实时 SSA 值，并行更新回边，零次迭代后返回初始值；嵌套循环从
外层实时携带值初始化。DT/ctx 元数据保留 yield/phi 的原始位置，不创建 Python 张量槽，
仅含元数据的 `For` 和 `If` 语句体仍生成有效 Python。在生成派发和布局之前，CommCtx
携带值必须证明初始值与每条回边具有同一精确来源，即收到的同一形参或所属窗口查询；
上下文 phi 的两个分支也必须保持该来源。循环或分支之后的稳定结果复用该来源；变化或
无法解析的来源在生成阶段给出诊断，不增加标量线值。
内部存储必须具有常量形状，并在初始化前准备；
动态形参几何信息不授权动态存储分配。
递归打包与实际启动集成也尚未完成；公共 L4 编译与执行闸门继续保留。

## 另请参阅

- [Passes](../passes/index.md) —— codegen 运行前必须成立的全部前提。
- [PTO ISA 参考](../../reference/index.md) —— 生成代码所使用的指令语义。
- [PTOAS 算子状态矩阵](../ptoas-op-status.md) —— 可供发射的 PTOAS 算子范围。
