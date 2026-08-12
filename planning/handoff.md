# 接力上下文（Handoff）

> **只描述下一位 agent 现在要接的工作。最后更新：2026-08-12。**
> 当前状态以 [`../STATUS.md`](../STATUS.md) 为准；历史过程不要复制回本文件。

## 1. 当前判定

`fa58b5cffe41b30d3f8d94482230867ee34b9e84` 已完成源码集成，但当前性能验收
**NO-GO**：

```text
qkv_proj → qkv_split_qknorm_rope → attn_mix
```

- 整网精度：PASS；
- 整网 ITL：FAIL，candidate p50 `33.194 ms`，baseline center `31.846 ms`，
  回退 `+1.348 ms / +4.233%`；
- fresh 前五层 DFX：FAIL，strict `<46 us` 为 `39/40`；
- worst：rank7/L0 Full=`54.54 us`；
- inventory/dependency/legacy audit：PASS；
- immutable image：未构建。

2026-08-11 的五层 `40/40`、max `43.60 us` 是历史单次 capture，已被
2026-08-12 final-commit fresh run 的 39/40 supersede，不能继续作为当前准出结论。

## 2. 0162 源码状态

所有源码与脚本均只在 0162 创建或修改；本地项目仓仅允许文档变更。

```text
candidate worktree
  /mnt/persist/chensiyu/workspace/develop-worktrees/qkv-prerope-mix

branch
  perf/qkv-prerope-mix-20260811

parent/base
  f906526190dc2eca0d479f8e9fa9187ec6d31be9

final commit
  fa58b5cffe41b30d3f8d94482230867ee34b9e84

status
  origin/stepfun/develop, main checkout and candidate worktree aligned; clean
```

0162 路径：

```text
/mnt/persist/chensiyu/workspace/develop/pypto-lib
/mnt/persist/chensiyu/workspace/develop-worktrees/qkv-prerope-mix
```

固定验证镜像：

```text
manifest sha256:076af8a167405d5d0831e234cd16521c77d8bfdd173eff063d820802057c47f3
config   sha256:a9d111880883cea0b02e425fdfeaccc2b14bb1d1174c0b73488d8ee6d8004d39
```

镜像内 `pypto-lib` 仍为 `cb96747e`；候选是 read-only source overlay，runtime
无 overlay。

## 3. 已通过的正确性门

```text
unit                    362 passed, 7 skipped
whole compile           PASS
focused correctness     PASS
edge contexts           12/12 exact
Q publication           Full/SWA 12/12
heterogeneous contexts  [1,2816,2817] exact
five-layer L3/L4        exact, finite, TP spread=0
whole precision         exact, finite, token 14371
```

整网 A/B/A 三臂 hidden SHA256 均为：

```text
567b206bb03d89f84020e1dddd61098a8f79f32f81b8f4fcf56443113e27f03e
```

因此可以明确回答“整网精度通过”；不能据此回答“性能集成通过”。

## 4. 整网 ITL A/B/A

合同：

```text
A1/A2 baseline  f906526190dc2eca0d479f8e9fa9187ec6d31be9
B candidate     fa58b5cffe41b30d3f8d94482230867ee34b9e84
BS              1
context         65536
blocks          512
warmup/iters    10/100
devices         8–15
```

结果：

```text
A1 p50             31.787 ms
A2 p50             31.905 ms
baseline center    31.846 ms
half-range floor    0.059 ms
B p50              33.194 ms
delta              +1.348 ms / +4.233%
delta/floor         22.85x
precision           PASS
performance         REGRESSION_BEYOND_BRACKET
```

证据：

```text
/mnt/persist/chensiyu/workspace/perf-2026q3/
  attn-mix-device-gate-20260811/out/aba-bs1-ctx64k-20260812-102231/

ABA_RESULT.json
  sha256 065f67c889a5eb108c49770261ccadf4d8f2970882b657efafb205ee35d6510b
```

## 5. Fresh 前五层 DFX swimlane

运行时间为 2026-08-12 10:39:59–10:44:15（Asia/Hong_Kong），共生成 8 份
merged swimlane。严格口径：

```text
start  = earliest layer-local *_qkv_proj Worker View ts
finish = latest layer-local *_qkv_split_qknorm_rope Worker View ts+dur
gate   = max(8 ranks × 5 layers) < 46.000 us
```

结果：

| Layer | Min us | Max us | Pass |
|---|---:|---:|---:|
| L0 Full dense | 41.46 | **54.54** | 7/8 |
| L1 SWA dense | 38.90 | 43.14 | 8/8 |
| L2 SWA dense | 38.66 | 40.02 | 8/8 |
| L3 SWA MoE | 39.16 | 41.72 | 8/8 |
| L4 Full MoE | 39.38 | 41.50 | 8/8 |

```text
total       39/40
failed      rank7/L0 Full
span        54.54 us
over gate   8.54 us
```

只读诊断：

- QKV 10 个 Worker slice 的单片 kernel duration 与其它 rank 一致；
- fused task 实际 compute `6.44 us`，也与其它 rank 一致；
- rank7 在 QKV 发射窗口出现约 `12 us` AICPU scheduler dispatch stall，导致
  projection family span `44.68 us`；
- deps 与 lineage 完整，前驱已完成；
- 该 launch skew 是端到端 stage latency，不能从 strict gate 中剔除。

证据：

```text
/mnt/persist/chensiyu/workspace/perf-2026q3/
  qkv-prerope-postmerge-validation-20260811-r1/five_layer/

analysis_final/attention_gate_report.json
  sha256 0b5cbe2064663d179a509739e8c6ccd89777c839fcaca1023c4d1403c3a025a1

analysis_final/attention_gate_report.md
  sha256 f00149e36403e264018abd55fee4531672535a4b517dd43e5a052a78715c582e
```

外层 runner `rc=0`；candidate container `rc=1` 仍包含 canonical zero-route
missing-swim record 限制。独立 analyzer 本轮也返回 `rc=1`，原因是 39/40 性能门
失败；其 structural inventory/dependency/legacy 子门为 PASS。

## 6. 下一步

不要直接构建 `fa58b5cf` release image。所有代码和脚本仍只允许在 0162 完成。

优先隔离两个变体：

1. 恢复 SWA `head_gate_expand` 在 QKV projection 之前；
2. 保留 fused split/QKNorm/RoPE，但恢复独立 Q 与 KV projection task。

每个变体：

1. 先跑 candidate-only whole ITL 筛选；
2. 对最终候选跑同口径 A/B/A；
3. 重跑 unit/compile/focused exact/Q-publication；
4. 重跑 fresh 8-rank×5-layer strict `<46 us`；
5. 只有整网无回退且 40/40 后才考虑 immutable image。

如果 packed projection 持续造成整网回退，应明确 NO-GO，只保留 fused
split/QKNorm/RoPE。

## 8. 待续：tp-all-reduce barrier-mesh → ring（0162，2026-08-12）

```text
源 fa58b5c -> 分支 perf/tp-allreduce-ring-20260812 (pypto-lib a791071)
状态：ring 8 卡 compile OK；端到端 harness 受 pre-existing 镜像 pypto 配对 gate
     拦截（原始 barrier-mesh 同样失败），ITL 未在本镜像跑通。
```

任务：vllm-ascend 用 `hcclAllReduce`（HCCL V2 选 ring/tree，stream overlap）做 TP
all-reduce；pypto 现行 attention o_proj/shared-expert 末尾手写 **barrier-mesh**
（stage-in → 全局 barrier → 逐 chunk 全 mesh 读），跨卡 DMA ~4× 于 ring。对照
`/data/chensiyu/hw_project/hccl` 已实现算子，改造为同族 ring reduce-scatter +
all-gather（2(N-1) 步，每步独占信号 cell、非单调 Set(1)/Ge(1)，刻意避开当初
ring→barrier 的 codegen 507018 根因）。信号窗 `tp_size` → `2*(tp_size-1)+1`。

```text
files:  models/step3p5/attention_full.py, attention_swa.py
docs:   pypto-lib/docs/upstream-issues/step3p5-tp-allreduce-ring-refactor.md
        (0162)；local design/vllm-pypto/04-tp-allreduce-ring-refactor.md
```

- 8 卡 `_stage_two_layer_attn` harness 跑出 `compile OK in 1.4s`（ring 被编译器真
  实接受）；codegen-contract / chip_orch.cpp 编排编译两项 gate 与原始 baseline 同败
  （pre-existing，非本改造引入）。
- 待办：配平镜像上重跑 ITL；同模式扩展 moe.py (T/N_RANKS)、mtp_hidden_fwd.py、
  prefill_attention_*；collectives.py 的 @pl.jit.inline ring 同步改非单调 Set。
- 复现脚本在 0162 `/mnt/persist/chensiyu/workspace/ar_bench/`（仅 0162）。

## 7. 机器与约束

当前验证 session 已结束，cards 8–15 的容器与进程已清理；后续启动前仍须重新检查
锁、容器、fuser 与 NPU process，不能沿用旧空闲结论。

禁止事项：

- 本地创建/修改代码或测试脚本；
- 先在本地写脚本再复制到 0162；
- 用历史 40/40 覆盖 fresh 39/40；
- 用独立 analyzer 的 structural 子门覆盖 canonical fail-closed；
- 用局部五层变快推断整网 ITL 变快。
