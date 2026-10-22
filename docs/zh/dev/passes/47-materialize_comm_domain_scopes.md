# MaterializeCommDomainScopes Pass

## 概览

`MaterializeCommDomainScopes` 物化 HOST 和 CLUSTER_0 orchestration 的本地通信域，组装出分布式 runtime
为分配 / 填充 per-rank 通信窗口所需要的 host 侧元数据。它与
[`InitMemRef`](35-init_memref.md) 在结构上完全同构：追溯一次分配到所有
消费点，构造反向引用对象，再把该对象挂到 IR 类型上，让下游 codegen 能 O(1)
访问。

| 比较项 | `MemRef` 一侧 | `WindowBuffer` 一侧 |
| ------ | ------------- | ------------------- |
| 分配 op | `tile.alloc(memory_space, size_in_bytes)` | `pld.tensor.alloc_window_buffer(size_in_bytes)` |
| Parse 时赋值语句 LHS | `Var(PtrType)` | `Var(PtrType)`（同一个 singleton） |
| 包装 Var 子类 | `MemRef` | `WindowBuffer` |
| 包装类的 SSA-edge 类型 | `MemRefType`（singleton） | `WindowBufferType`（singleton） |
| 构造者 | `InitMemRef` | **`MaterializeCommDomainScopes`**（本 pass） |
| 回填到 | `TensorType.memref_` | `DistributedTensorType.window_buffer_` |
| Program 级注册表 | `Program.functions_`（alloc 语句） | 各本地所有者函数体内的 `CommDomainScopeStmt`（每个推导出的通信域一个） |

## 流水线位置

```text
... -> ExpandManualPhaseFence -> SynthesizeAllReduceSignals -> MaterializeCommDomainScopes -> LowerHostTensorCollectives -> MaterializeDistTensorCtx -> Simplify（最终）
```

本 pass 跑在默认 pipeline 的末尾阶段，位于
[`LowerHostTensorCollectives`](48-lower_host_tensor_collectives.md) 和最后一次
`Simplify` 之前。从 `InlineFunctions` 到这里之间的所有 pass 都不会触碰
host_orch 的 alloc / window / dispatch 链：host_orch 本身不会被 tile lower，
L2（chip 级）orchestration 也永远不会被 inline 进 L3，所以本 pass 需要的
alloc / view / dispatch 点在此时仍然可见。放在较晚阶段还能让产生 IR 在描述符
分析之前先被充分规范化，最后的 `Simplify` 也能统一对收集到的 size 表达式做常量折叠。

## 算法

本 pass 支持 `Level::HOST` 和 `Level::CLUSTER_0` 上的
`Role::Orchestrator` 本地所有者，也接受规范化的
`FunctionType::Orchestration` 声明。覆盖范围只从相邻层级 dispatch 推导：
`HOST -> CHIP` 和 `CLUSTER_0 -> HOST`。CHIP `Graph` 不是 HOST dispatch
目标。HOST collective 分析和 signal 合成仍只作用于 HOST。

1. **收集本地分配。** 为每个所有者建立 allocation/window 索引，跳过已有物化
   scope 的函数体。每个本地分配构造一个 `WindowBuffer`。从其他所有者继承的
   形参永远不会成为本地 slot，因此 HOST 函数可以同时使用自己的通信域和继承的
   CLUSTER_0 通信域。
2. **建立程序级身份图。** 图节点覆盖值、tuple 元素和声明的返回位置，以本地
   window 和已有身份为起点。每个 callee 的尾部 `CommCtx` 数量和显式返回值到
   形参的映射只缓存一次。对已解析的 `Call` 和 `Submit` 建立实参到形参绑定。
   调用者提供的 `Out` 实参参与传播；省略并由 runtime 分配的 `Out` 没有调用者
   来源，也不会错误占用尾部 context 实参。
3. **一次传播身份。** 沿 SSA 别名、注册表声明的别名 view/writeback、托管
   collective 结果、显式返回、tuple 构造/投影，以及循环 init/`IterArg`/
   yield/return 边传播。分支合流要求两个输入都解析到同一个对象；未知输入不能
   证明一致。任意新分配 tensor 运算不会建立 window 身份。每个值只接受一个
   对象；第二个不同对象即使名称、shape 和 dtype 相同也报错。同一对象可被多个
   调用点重复使用，不复制函数。
4. **推导本地覆盖范围。** 通过 dispatch 实参的身份查询当前所有者的本地分配
   索引。继承的 window 不进入该索引。规范位置属性为 `worker`；parser 会把
   已弃用的 `device` 写法规范化。

   | `worker=` 表达式 | 覆盖范围 |
   | ---------------- | -------- |
   | 非负 `ConstInt(N)` | `{N}` |
   | `[0, num_workers())`、步长为 1 的归纳变量 | 所有子节点 |
   | HOST 上 `[0, world_size())`、步长为 1 的归纳变量 | 所有子节点 |
   | 静态单位步长 range 的归纳变量 | 该 range |
   | 不支持或无法证明的动态起点/步长 | 报错 |

   直接调用和 SSA 提升后的子节点数量查询使用相同的带缓存边界分析。
   CLUSTER_0 上仍禁止 `world_size`。HOST collective signal 和
   `all_to_all_v` count window 保留已有配对覆盖规则。
5. **分组并改写。** 合并各本地分配的覆盖范围，通过规范化快照 ID 查找分组，并保留
   组内分配顺序。最后只进行一次 mutator 改写，同步函数签名、函数体和函数属性引用；保留每个值
   自身的 shape、dtype、`MemRef` 和 tensor view，只替换 `window_buffer_`。
   同步声明的返回类型、Call/Submit tuple 结果、投影和循环类型。保留 Submit 的
   dependency、属性、launch 字段、predicate 和尾部 `TASK_ID`。
   仅为真实本地分配嵌套 `CommDomainScopeStmt`；已有窗口保留原对象。

IR 遍历和身份工作队列的复杂度为 `O(N + E)`：每个身份只赋值一次，每条边只处理
一次。类型相关性在共享类型 DAG 上记忆化；tuple 血缘只存储含 DT 的位置，并保留
原 ABI 索引。普通标量 tuple 字段不会按值重复展开。tuple 根的变化标志只计算一次，
随后用有界布尔队列沿子表达式到父表达式的边传播改写需求，因此也能发现包含已变化
DT 调用的标量或上下文根。函数/根使用边标记全部受影响函数，包括共享同一形参、
函数体或属性表达式对象的 native 函数。没有身份或本地 domain 变化的函数直接跳过
改写。Call/Submit 的结果绑定只保存含 DT 的 `(原 ABI 索引, 值)` 项，因此不会在
每个调用点扫描宽的纯标量返回列表。共享 base mutator 用持有对象的不可变 `TypePtr`
键记忆化纯标量类型树：这些树没有嵌入表达式或子类回调，包括 Call/Submit 结果类型。
Tensor 的 shape、view 等含表达式的类型仍使用共享 remapper。
直接 dispatch 的覆盖使用可变区间映射（`D` 次区间插入为 `O(D log D)`），
随后每个分配只冻结一次。数值范围保留为区间，直到输出 domain 的有序设备列表。
分配共享规范化的不可变覆盖快照；collective 消费者按源码顺序更新目标快照，缓存
每对不同快照的并集。分组使用快照 ID，因此不会按消费者或分配数反复复制或比较
相同的大集合。

覆盖计算仍有复杂度例外：每个未缓存的 collective 并集需要 `O(I_left + I_right)`
的区间处理以及有序规范化键比较；比较可能检查区间前缀。快照存储量为各快照区间数
之和，输出设备列表的成本为成员总数。许多不同且相交的并集即使最终输出很小，
仍可能相对 IR 大小产生二次工作量。按 pass-complexity 规则，该残余界限需要在
合并前明确批准；这里不声称普遍满足 `O(N log N)`。不会反复扫描函数体求不动点，
不会在每个调用点扫描全部形参，也不会为每次使用单独追溯别名链。已有 scope 保证
重复运行幂等。

## 诊断与输出不变量

- 缺少 view 和没有相邻 dispatch 消费者的分配保留已有诊断。不支持的覆盖表达式、
  有歧义的身份，以及没有所有者 scope 的非空身份会被拒绝。
- 每个物化 window 及其支持的别名、callee 形参、返回和投影都与所有者 scope slot
  共享完全相同的 `WindowBuffer` 对象。独立 device 函数的可空形参仍可保持未绑定。
- 一个 scope 只拥有其函数的本地 slot。继承的 slot 不会复制到子 scope。
  函数遍历顺序不会选择调用者身份，也不会改变结果。
- 赋值两侧类型保持对称；保留原 shape/view 元数据，使真实类型不匹配仍可由验证器发现。
- [`SynthesizeAllReduceSignals`](46-synthesize_allreduce_signals.md) 之后，HOST
  `allreduce` 仍带显式或合成 signal。HOST `all_to_all_v` 在 `for`/`while` 中的
  现有限制保持不变。

这是有明确边界的通信域与身份阶段，不新增 CLUSTER_0 collective lowering，不改变
context 物化，也不会启用完整 L4 `compile` 流水线；这些依赖后续 lowering/runtime 阶段。

## 文本 round-trip

用户编写程序时仍然自动推导通信域。打印后的 IR 使用私有 `pl._dump` 语法保存物化状态：

- `comm_domain(name=..., devices=..., slots=[...], attrs={...})` 保存作用域边界、
  设备列表、slot 顺序及属性。
- `window_buffer("buf_window", base="buf", size=16, load_from_host=..., store_to_host=...)` 定义窗口；
  tensor 类型通过 `window_ref("buf_window")` 引用同一个对象。
  名称是文本格式中的引用标识，不是设备 ID、slot 下标或 IR 字段。
  名称由打印后的 base 变量名生成；共享同一 base 的不同窗口通过后缀区分。旧的数字 ID 仍可读取。
- `alloc_window_buffer(size, name=...)` 保存分配的逻辑名称，不受 SSA 变量重命名影响。
- `call(op, args, result_type, kwargs=..., attrs=...)` 直接恢复 Call，接受类型解析器支持的任意结果类型。
  打印器用它保存分布式调用实际记录的类型，包括不存在反向引用的情况；
  pass 改写操作数后，重新推导的类型可能与原类型不同。

解析器先登记窗口描述，再解析 body，在分配和大小表达式可用时绑定引用。
作用域 slot 与 tensor 反向引用共享同一个恢复后的 `WindowBuffer`。
这些写法属于 IR 文本语法，不是在 kernel 中执行的操作。
描述中的 `base` 引用 body 内定义的分配变量；`window_buffer(...)` 不会增加分配语句。
它的 `size` 和分配 Call 的大小参数是 IR 分别保存的字段，因此都会保留。
空作用域打印为 `pass`。`-1` 等属性字面量保留为标量；
`pl.const(-1, pl.INDEX)` 则表示 IR 表达式。

对于自包含的完整程序，`parse(python_print(program))` 必须与原程序结构相等且结构哈希相等。
解析器和调用方均不需要重跑 lowering pass。引用缺失定义的单条语句片段不属于此契约。
旧的 `"window_buffer=<name>"` 提示字符串仍可读取，但不表达真实引用。

## Pass 属性

| 字段 | 取值 |
| ---- | ---- |
| `required` | `{}` |
| `produced` | `{IRProperty::CommDomainScopesMaterialized}` |
| `invalidated` | `{}` |

## 参考

- 实现：[src/ir/transforms/materialize_comm_domain_scopes_pass.cpp](../../../../src/ir/transforms/materialize_comm_domain_scopes_pass.cpp)
- 头文件：[include/pypto/ir/transforms/passes.h](../../../../include/pypto/ir/transforms/passes.h)
- Schema：[include/pypto/ir/program.h](../../../../include/pypto/ir/program.h)
  定义了 `WindowBuffer` 与 comm-domain scope。
- DSL：[`pld.tensor.alloc_window_buffer`](../../../../python/pypto/language/distributed/op/tensor_ops.py)、
  [`pld.tensor.window`](../../../../python/pypto/language/distributed/op/tensor_ops.py)、
  [`pld.system.world_size`](../../../../python/pypto/language/distributed/op/system_ops.py)。
