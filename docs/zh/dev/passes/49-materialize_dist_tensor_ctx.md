# MaterializeDistTensorCtx Pass

本 pass 为每个 `DistributedTensorType` 函数参数显式物化一个对应的
`CommCtxType` 参数和实参。

## 概览

DistributedTensor 的通信上下文需要沿完整调用链传递，包括 CLUSTER_0（L4）→
HOST（L3）→ CHIP（L2）。host orchestration
提供每个 rank 的 `device_ctx`，L2 orchestration 通过 task args 转发，L1 PTO
codegen 再用它降低 `pld.system.rank`、`pld.system.nranks`、`notify`、`wait`、
`put` 和 remote memory ops。

旧路径在多个 codegen 站点分别合成 ctx，容易漏站点或顺序漂移。本 pass 把这条
数据流放进 IR：

1. 对每个带 `DistributedTensorType` 参数的函数，按分布式 tensor 参数顺序在签名
   末尾追加 `CommCtxType` 参数，方向为 `ParamDirection::In`。
2. 对每个调用点追加对应 ctx 实参。若 distributed tensor 是调用者自己的参数或其
   SSA alias，则转发调用者已有的 ctx 参数。返回位置按被调函数实际返回的参数
   建立对应关系，因此混合类型或重排后的返回 tuple 不会退化为尾部位置猜测。
   `Submit` 同理：它的结果位置就是被调函数的返回位置（末尾的
   `Scalar[TASK_ID]` 没有对应 ctx）。把新 SSA 变量绑定到「已经存在的
   DistributedTensor」的 builtin op 会转发该值的 ctx，包括两类：输出侧写回
   （由 op 自己通过 `set_output_reuses_input(idx)` 声明，例如 `tile.store`
   -> `args[2]`、`tensor.assemble` -> `args[0]`），以及零拷贝 buffer-aliasing
   view（`tensor.view`、`tile.slice`、`tensor.reshape` 等），后者的结果类型
   直接从 `args[0]` 传播 `DistributedTensorType::window_buffer_`。
   `ForStmt` / `WhileStmt` 携带的 tensor alias 也会被追踪。对 HOST 或 CLUSTER_0
   orchestration，在无法解析 lineage 时会在调用前插入
   `pld.system.get_comm_ctx(dist)` 绑定并传递该结果。chip orchestration 和
   device 函数必须解析出显式 ctx；如果无法解析则直接报错，不会生成
   device-side 查询。
3. 在 chip orchestration 和 device 函数中，本 pass 会把所有
   `pld.system.get_comm_ctx(dist)` 替换为已经解析出的显式 `CommCtxType` SSA
   值。HOST 和 CLUSTER_0 orchestrator 保留该 op；HOST codegen 从每个 rank
   的运行时 window context 中解析，CLUSTER_0 context 的运行时解释由后续
   lowering 负责。
4. 若调用点已有 `arg_directions`，同步追加 `ArgDirection::Scalar`，让后续
   codegen 把 ctx 当作普通 scalar task payload 处理。

`Submit` 实参保留规范布局：调用者提供的前缀（prefix）后面跟已有及本次追加的
`CommCtx` 后缀（suffix）。仅允许省略紧接该后缀之前的连续 `Out` 参数段（gap）。
例如，原参数 `[data, scratch: Out[Tensor], existing_ctx]` 和实参
`[data, existing_ctx]` 会变成参数 `[data, scratch, existing_ctx, data_ctx]` 与
实参 `[data, existing_ctx, data_ctx]`。显式提供的 `Out` 具有真实调用者来源；
省略的 `DistributedTensor` `Out` 没有调用者 context，会给出不支持的明确诊断。
本 pass 不会为该输出分配运行时 context。在扩展任何签名之前，本 pass 会缓存
每个原始签名、参数数量和末尾 context 数量。别名分析根据这些事实绑定原始调用；
在递归改写后的实参表达式中解析返回值时，则使用扩展后的签名及后缀。存在
`Out` gap 时，不能仅靠实参数量区分这两个阶段。

返回值来源（result provenance）对每个被调函数只保存一份原始位置映射。返回
distributed tensor 的调用仅捕获实际返回的不同 DT 参数对应的 context，数量
受调用者提供的实参数量约束；tuple alias 共享这份不可变捕获。仅返回 scalar 或
空返回值时不分配结果 context 捕获。直接嵌套的投影只解析所请求的位置，因此
同一 DT 参数即使重复出现在宽返回 tuple 中，也不会让每个调用点都为所有返回
字段分配或解析 context。

本 pass 还会对不可变的共享结果类型 DAG 进行一次分类。Scalar/tuple 类型树，
以及不含 MemRef、且 metadata 表达式全部为整数常量的 Tensor/DistributedTensor，
可以跳过基础类型遍历，因为本 pass 无法改写这些叶节点。分类沿用规范的 shape/view
遍历，并保持 window 所属关系不透明。缓存仅保存这一证明，不保存改写后的类型：
MemRef、动态 shape/stride 表达式、Var、IterArg、查询和未知类型仍执行依赖当前状态
的基础遍历；这些未通过安全分类的共享类型被重复使用时，仍可能重复遍历 metadata。

函数属性中保存的表达式（Expr-valued attributes）按存储类型改写，包括受支持的
表达式容器，使其中的 Call、Submit 和 context 查询保持更新后的 ABI。属性没有
语句插入点：其中的调用必须能从已有形参或 alias lineage 解析 context，否则会
收到明确的不支持诊断。普通 metadata、调用 kwargs、返回位置、实参方向及 Submit
的启动和依赖字段均保留。本 pass 不注册 CLUSTER_0 level handler，也不改变仅适用于
HOST 的 collective 和 signal lowering 约定。
二进制往返会保留函数体和 metadata 中共享的 window 引用，包括首次由函数体
表达式定义的局部所属 slot。

本 pass 不会向 `IfStmt` 的 return variable 或分支 yield 添加 `CommCtxType`。
DistributedTensor 的 if lowering 继续保持 then/else 必须引用同一 backing/context
的现有约束；动态 context merge（issue #2027）不在本次修改范围内。

循环 carry 遵循同一条「单一 context」规则。carry 先用 init value 播种（这样
`data = self.comm(data)` 这类自携带才能解析），遍历完循环体后再用 yield 回来的
值校验该播种值：如果循环体内把 carry 重新绑定到**另一个** DistributedTensor，
会直接报错，而不是继续沿用 init value 的 context。若 yield 的 lineage 完全无法
追踪，则保留播种值。

本 pass 产出 `IRProperty::DistTensorCtxMaterialized`：HOST 或 CLUSTER_0
orchestration 之外不再存在 `pld.system.get_comm_ctx`。pass 对它改写过的每个函数
都保证了这一点，property verifier 再独立校验一遍，包括函数属性中保存的表达式。
这也覆盖了「程序里没有任何函数带
`DistributedTensorType` 参数、pass 原样返回」的情况。该属性已登记在
`GetVerifiedProperties()` 中，因此默认 verification level 下流水线就会校验它，
而不是只在测试手工安装 `VerificationInstrument` 时才生效。

本 pass 要求 `IRProperty::ReturnParamsExplicit`：返回位置映射来自
`return_lineage::ExplicitReturnedParamIndices`，它是对 `ReturnStmt` 的指针恒等
读取，只有在 `NormalizeReturnOrder` 规范化之后才有意义。

该 pass 位于 `LowerHostTensorCollectives` 之后、最终 `Simplify` 之前。此时
host window buffer 已由 `MaterializeCommDomainScopes` 填好，host tensor
collective 也已降低完成，同时后续仍有一次 simplify 可清理转发别名。

## 与 dynamic dim 的区别

dynamic dim 可以在 wrapper 边界从 tensor descriptor 本地恢复；CommCtx 不行。
CommCtx 是真实的跨层数据流，必须从 host 到 orchestration、task payload、kernel
signature 一路传递。把它放进 IR 可以避免多个 codegen 站点各自维护隐式规则。

## API

| C++ | Python | 级别 |
| --- | ------ | ---- |
| `pass::MaterializeDistTensorCtx()` | `passes.materialize_dist_tensor_ctx()` | Program-level |

```python
from pypto.pypto_core import passes

program = passes.materialize_dist_tensor_ctx()(program)
```

## 示例

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

kernel body 不需要修改。device 函数 body 中已有的
`pld.system.get_comm_ctx(data)` 会在本 pass 中改写为显式 ctx 参数；HOST 和
CLUSTER_0 orchestration 中的用法仍保留为运行时查询。
