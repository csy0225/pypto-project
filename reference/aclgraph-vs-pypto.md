# Step3p5 执行模型:pypto tile 融合 vs vllm-ascend aclgraph

> 跨仓 design note。回答一个反复出现的问题:
> **「decode 用 pypto kernel 替换掉 vllm-ascend 的 graph 调度后还兼容吗?上限谁更高?」**
>
> 关联:
> - `pypto-lib/models/step3p5/decode_fwd.py` — 顶层 `Step3p5DecodeFwd`(host_orch / chip_orch)
> - `pypto-lib/models/step3p5/decode_layer.py` — `select_decode_layer()` + 2 个 per-layer builder
> - `pypto/runtime/.../tensormap_and_ringbuffer/docs/RUNTIME_LOGIC.md` — simpler 运行时调度模型
> - 关联 Phase:[`../phases/20-vllm-backend-monkey-patch.md`](../archive/completed-phases/20-vllm-backend-monkey-patch.md)、[`../phases/22-perf-baseline.md`](../archive/completed-phases/22-perf-baseline.md)
> - 关联 arch:[`vllm-step3p5-mapping.md`](../design/vllm-pypto/03-vllm-op-mapping.md)(op-级映射)

---

## 0. TL;DR

1. **术语**:Ascend 上不是 CUDA graph,是 **ACL graph (aclgraph)** / **torchair graph mode**。
2. **兼容性**:pypto kernel **不能**作为 drop-in 塞进 vllm 捕获的图里——两者是**互斥**地解决同一个问题(decode 的 host launch 开销)。正确姿势是**用 pypto 融合取代那段 graph**,被 patch 的路径走 `enforce_eager`,MoE fallback 段可留在 aclgraph 里。
3. **粒度**:aclgraph 的节点 = 1 个 aclnn 算子(细);pypto 整网最终只有 **9 个粗粒度 program**(或融成 1 个),层内 attention/MoE/lm_head 全在编译期内联掉。
4. **调度**:单个 pypto program 内,host 只 upload 一次 + kick + sync,**AICPU 自调度,host 出 loop**。要让 host 在 48 层之间也出 loop,机制是**编译期把多层融成一个 orchestration**,而不是把多个 program 提交到共享队列。
5. **上限**:两者物理上限是**同一个 roofline**。aclgraph 只清「host 调度」一桶(和 pypto 共享的地板);pypto 额外清掉 kernel 边界 + 中间张量 HBM 往返 + 多引擎重叠 → **更逼近 roofline**,是 aclgraph 的超集。**收益 prefill 大、纯 decode 小**。
6. **大 batch**:收益曲线是**凸的**——离开小 batch 内存墙角后上升、中等 batch 见顶、极大 batch 因 compute-bound 又被压平;同时**大 batch 让 aclgraph 自身失去意义**(host 开销趋零)。

---

## 1. aclgraph 是什么、capture 什么

vllm 的图模式(`CUDAGraphMode`,Ascend 对应 aclgraph)有三档:

- **FULL**:把**整个 model forward = 全部 48 层的每一个 aclnn 算子**(matmul、rope、rmsnorm、attention、all_reduce…)录成一张可 replay 的图。节点是几百到上千个**细粒度 aclnn op**。
- **PIECEWISE**:在不可捕获的算子(典型是 varlen / 动态 attention)处把 fx graph **切段**,只 capture 段间部分,attention 留 eager。
- capture **按 batch-size 分桶**,每个桶一张图,运行时 pad 到最近桶。

**本质**:对一串**外部已定义好的 op 序列**做录制-回放,只消掉 host launch 开销,**不改变算子边界,不做跨算子计算融合,不动中间张量的 HBM 往返**。

---

## 2. pypto 整网怎么切分(四层)

切分**不是按 aclnn op**,而是四个层级:

1. **层边界(编译期静态展开)**:`Step3p5DecodeFwd` 顶层 `@pl.program`;48 层(45 主 + 3 MTP)在 **Python 编译期 for 循环**里用 `select_decode_layer(li)` 静态选层型,依据 `LAYER_TYPES`。层间通过被全局 reduce 过的 `hidden` 串接。**不是 runtime 循环**。
2. **InCore / Inline body = 融合单元**:因 §10(不能在 `@pl.program` body 里实例化另一个 `@pl.program`),每层 attention / dense MLP / MoE body 被**逐字内联**成 `@pl.function(InCore/Inline)`。整网 = 一个程序里大量内联体,**融合在编译期发生**。
3. **tile 级循环 → codegen task(真正派发单元)**:每个 body 内用 `pl.spmd / pl.range / pl.parallel / pl.pipeline` 切 block。这些才是 AICPU 派发给 AICore 的 task。一个 decode_layer ≈ 20 个 task,整网 ×48 层 → 几百个 task 的 DAG。
4. **硬件容量 + 通信边界 = 切点真正驱动**:chunk 大小由 L0/L1/UB 上限定死(如 `ROUTER_GATE_N_CHUNK=32` 因 `[K=256,N=32]FP32=32KB` 撞 L0B);`tp_all_reduce` 这类跨卡 collective 天然切开前后计算段。

### 粒度对比

| | aclgraph | pypto |
|---|---|---|
| 节点/任务粒度 | 1 个 aclnn op(细) | 一段融合 tile-loop(粗) |
| 切分依据 | 算子库边界 + 不可捕获点 | 层边界 + 硬件 buffer 容量 + TP/EP 通信点 |
| 何时确定 | runtime capture 录制既有序列 | **编译期**融合好,生成 task DAG |
| 谁来调度 | host launch + stream replay | 设备端 AICPU orchestrator |

---

## 3. step3p5 最终有几个 program

**decode 场景 = 9 个 `@pl.program`**:

| | program | 数量 |
|---|---|---|
| 顶层 | `Step3p5DecodeFwd`(host_orch + chip_orch) | 1 |
| 每层特化 | 见下 | 8 |

**8 种每层特化**(import 期建好的模块级单例):

- dense × 2:`full_dense`、`swa_dense`
- MoE × 6:`{full, swa}` × `{silu_silu, swiglu7_silu, swiglu7_swiglu16}`

要点:
- `select_decode_layer(li)` 对**同 kind 的层返回同一对象**(`prog_l4 is decode_layer_full_moe_silu_silu`)。48 层**复用**这 8 个。
- 源码层其实只有 **2 个参数化 builder**(`DecodeLayerDense` / `DecodeLayerMoE`),「8」是编译期按参数(full/swa、激活组合)展开的特化。
- `TpAttentionFull/Swa`、`EpTpMoE`、`TpRmsLmHead` 在代码里也有独立 `@pl.program` 定义(供 Phase 19 ST 单跑),但在整网里**被内联**,不增加 program 数。

> prefill 是平行的一套:`Step3p5PrefillFwd` + `PrefillLayerDense/MoE` + 内联的 `PrefillAttention*/PrefillMoE/QkvProjRope`。

---

## 4. 调度模型:host 何时出 loop

### 机制(已从 `RUNTIME_LOGIC.md` 确认)

单个 pypto program 内部,host 调度**确实被消掉**:

1. host 把**所有 kernel + orchestration `.so` 打包进一个 `ChipCallable` buffer**,`upload_chip_callable_buffer` **一次性 H2D**。
2. **AICPU Thread 3** `dlopen` 该 `.so`,调 `aicpu_orchestration_entry`(= pypto 的 `chip_orch`),里面一连串 `rt_submit_task(...)` 把 task 压进**按 resource-shape 分的队列**(`AIC_ONLY` / `AIV_X2` / `AIC_AIV_X2` …)。
3. **Threads 0-2(scheduler)** 拉 task、按依赖派发到 AIC/AIV,跑完 `rt_orchestration_done()`。
4. host 全程只做 **upload 一次 → kick → 等 `orchestrator_done_`**;层间、task 间不参与。

### 跨 program 怎么做到 host 出 loop

orchestration 加载是 `dlopen → run → dlclose` 的**单体生命周期,一个 `.so` 跑完才换下一个**。所以:

- ❌ **不是**「9 个 program 各自 chip_orch 并存到 AICPU,跨 9 个统一调度」——那样每换 program 就要 host 重新 kick = host 回 loop。
- ✅ **正解**:编译期把 8 个 per-layer body **全部内联进顶层那一个 `chip_orch`** → 生成**一个** orchestration `.so` → 48 层几百个 task 进**同一张 AICPU task graph** → 一次 kick、设备端统一调度、host 出 loop。**这正是 §10 逼出 body-copy 内联的根本原因**。

### 现状(Wave-3 未完成)

`decode_fwd.py` 的 host_orch 里 48 层 staging **目前是 placeholder**:

```python
# for Wave-3 the layer dispatch is staged outside the @pl.program
# (the per-layer programs each build their own host_orch ...)
self.chip_orch(...)   # 目前只跑最后的 RMSNorm + lm_head shard
```

即:今天每个 per-layer program 各自带 host_orch + chip_orch、**standalone 跑**(Phase 19 ST)。若现在按层跑(Phase 20 `per_layer=True` escape hatch),**host 在 48 层之间逐层回切 = 48 次 round-trip = 慢路径**。把 48 层融进一个 orchestration、host 彻底出 loop,是 **Phase 20 D1 + Wave-3 的目标,尚未 wire**。

---

## 5. 能不能全放一个 program?8 种特化会消失吗?

- **48 层放进一个 program:能**(Wave-3 / Phase 20 D1 目标)。
- **per-layer `@pl.program` 外壳会退化成内联 func**:对。融合后「可独立 dispatch 的 program 单元」消失,变成顶层里的 `@pl.function`(Inline/InCore)。
- **但 8 种特化不会塌成 1 个 func**,也**不是因为没整网跑通**。它们编码的是**编译期定死的真实结构差异**:

| 维度 | 差异 | 能 runtime 分支掉? |
|---|---|---|
| full vs swa | head 数 / window / RoPE 表 / tile shape 不同 | 不能,两套不同 kernel |
| dense vs MoE | MoE 多出 gate→dispatch→routed expert→combine 整条 a2a 链 | 不能,计算图不同 |
| 3 种 MoE 激活 | SWIGLU limit 是编译期常量 | 理论可参数化,目前 baked |

根因是 **tile codegen 要静态 shape**:L0/L1/UB 分配、循环展开、调度全在编译期 bake。full/swa、dense/MoE 不是「一个 kernel 的参数」,是**不同 kernel**。所以**即使融成一个 program,仍是 8 个不同的内联 code shape**,出现在各自层位(48 个 inline 点 → 8 个 shape)。

> 唯一收敛空间:3 种 MoE 激活若改 runtime scalar,6 个 MoE 特化**或许**能合并几个;full/swa、dense/MoE 的分裂跑不掉。

---

## 6. 上限对比(roofline)

**两种方式物理上限是同一个 roofline**(峰值算力 / 峰值 HBM 带宽,谁先撞谁)。区别是**能逼近多少**、代价多大。

把 decode 一步拆成四桶:

```
单步延迟 = host 调度 + device kernel 启动 + HBM 流量(权重+激活) + 纯计算
```

| 桶 | aclgraph 去掉? | pypto 去掉? |
|---|---|---|
| host 调度 | ✅ 录制回放 | ✅ 单 orchestration,host 出 loop |
| device kernel 启动 | ❌ 每 op 仍独立 kernel | ⚠️ 融合后 kernel 变少 |
| 中间张量 HBM 往返 | ❌ 边界由算子库定死 | ✅ 中间结果留 L0/L1/UB |
| 多引擎重叠(AIC/AIV/MTE) | 部分 | ✅ tile pipeline 显式重叠 |
| 纯计算 / 权重读(roofline) | 不可去 | 不可去 |

**关键**:

- **「消除 host 调度」是两家共享的地板,不是 pypto 的差异化**——这桶上打平。
- pypto 上限更高的部分 = aclgraph **被锁在算子库 kernel 边界**上够不到的 2/3/4 桶。pypto 是 aclgraph 的**超集**(也清 host 调度 + 再清这三桶)。
- 公平对手是「**aclgraph + 库级融合(torch.compile/inductor)**」,不是裸 per-op。pypto 的真正差异化在**跨库级融合边界**:整层融合、residual 全程留片上、TP collective 融进计算(simpler IPC comm)。

**regime 依赖**:

- **Decode(权重带宽 bound)**:权重读一遍是地板,激活小,第 3 桶融合收益被摊薄 → 两家向权重带宽 roofline 收敛,**差距小**。
- **Prefill(算力 bound)**:激活大 → 中间 HBM 往返是大头 → 融合留片上 + flash + gate/up/silu 融合是**大赢**,pypto 上限明显碾压。

---

## 7. 大 batch decode 的收益曲线(凸)

大 batch 本质:**权重读一次摊给 B 个 token**,算术强度 ∝ B → 越过 roofline 拐点从内存 bound 变 compute bound。

随 batch 变大:

| 桶 | 变化 | 影响 |
|---|---|---|
| host 调度 | 占比**缩小** | **aclgraph 本身价值缩水** |
| kernel 启动 | 占比缩小 | 两家「减启动」都贬值 |
| 中间张量 HBM | 激活 ∝ B,绝对量涨 | pypto 融合收益**涨** |
| 纯 GEMM | 主导上升 | 两家跑同样 cube,拉不开 |

**反直觉点**:大 batch 让 aclgraph 失去意义——host 开销趋零(它是小 batch / latency 场景的主场)。比较退化为「融合 vs 不融合」,pypto 按构造赢。

**decode 是混合体**:

1. **Attention(QK/softmax/PV over KV)**:每序列 KV 私有、不跨 batch 摊销 → 始终 KV 带宽 bound,总 KV 流量 ∝ B。**flash 式融合收益随 batch 线性涨**——pypto 大 batch 最硬的增量。
2. **投影 + MoE expert GEMM**:权重跨 batch 共享 → 大 batch 变 compute bound,GEMM 主导,融合只省 glue,**占比被压小**。

**曲线形状**:小 batch 差距小 → 中等 batch(过拐点附近)**相对收益见顶** → 极大 batch 撞 compute roofline 回落(且 SRAM 装不下大激活,留片上被 tiling 封顶)。

**step3p5 MoE 额外因素**:

- ✅ 大 batch → 每 expert token 增多,GEMM 利用率升、**padding 浪费降**。
- ⚠️ dispatch/combine 的 a2a 通信量 ∝ B,负载不均放大 → pypto option A(comm 融进 kernel、和计算重叠)价值**随 batch 上升**,这是 aclgraph 完全够不到的。

---

## 8. 对 step3p5 的工程含义

- **甜区**:中到大 batch 的 **prefill**,以及大 batch decode 的 **attention(flash 融合)+ MoE comm-compute 重叠**。纯投影 GEMM 段大 batch 下和「好融合算子」拉不太开。
- **集成策略**:被 patch 的 pypto decode 路径走 `enforce_eager`(关 graph mode),靠自身融合拿延迟;MoE fallback 段(mixed-mode)可留 aclgraph。
- **前置条件**:Phase 20 D1 + Wave-3 把 48 层融成一个 orchestration、host 出 loop——**这是上述所有收益的前提**。没融合、逐层跑反而是慢路径。

### 待验证(Phase 22 实测,勿纸面推)

1. 用 swimlane / PMU 实测四桶各自占比(host 调度 / kernel 启动 / HBM / 计算),分 decode、prefill、不同 batch。
2. 关 graph 后 step 间残留的 eager-Python 开销有多大。
3. attention flash 融合、MoE comm-compute 重叠的收益随 batch 的实际曲线。
4. 大 batch 下 SRAM 容量对「中间结果留片上」的封顶点。

---

## 9. 参考

- [`../phases/20-vllm-backend-monkey-patch.md`](../archive/completed-phases/20-vllm-backend-monkey-patch.md) — 整模型 monkey-patch(D1 whole-model patch、D2 comm option A、mixed-mode MoE)
- [`../phases/22-perf-baseline.md`](../archive/completed-phases/22-perf-baseline.md) — perf baseline + swimlane / PMU 采点
- [`vllm-step3p5-mapping.md`](../design/vllm-pypto/03-vllm-op-mapping.md) — `Step3p5Model` ↔ `decode_fwd` op-级映射
- `pypto-lib/docs/known-pypto-pitfalls.md` — kernel 硬限制(含 §10 no-nested-program、buffer 上限)(sub-repo)
- `pypto/runtime/.../tensormap_and_ringbuffer/docs/RUNTIME_LOGIC.md` — simpler 运行时:ChipCallable upload、AICPU Thread 0-3、`rt_submit_task` / resource-shape 队列(sub-repo)
- `pypto-lib/models/step3p5/decode_fwd.py` / `decode_layer.py` — 顶层 program + `select_decode_layer` + 2 builder(sub-repo)
