# Phase 22 — Perf baseline + 调优

> **Component pin snapshot（doc 创建时，2026-06-22）**
>
> | 仓库 | 分支 | Pin | 备注 |
> |------|------|-----|------|
> | pypto-lib | `stepfun/develop` | `a6b5faa`（pre-commit） | 加这文件的 commit 之后会更新 |
> | pypto | `stepfun/develop` | `b00c8b23` | DFX env hooks：`PYPTO_DISTRIBUTED_DEP_GEN`, `PYPTO_DISTRIBUTED_L2_SWIMLANE` |
> | pto-isa | `stepfun/develop` | `e25732f0` | |
> | PTOAS | `stepfun/develop` | `da011a3d` | binary v0.45 |
> | simpler（submodule） | — | `a6e06406` | |

## 目标

产出 step3p5 通过 pypto kernel + vLLM 调度栈的**可发布** decode 性能数：

1. 单卡 token/s、TTFT、ITL 在标准 workload 矩阵下。
2. PMU + L2 swimlane 定位瓶颈。
3. 两轮优化（tile sizing / buffer reuse / scheduling）。
4. TP=8 多卡 scaling（gate 在 barrier all_reduce UB 修复 + MoE 507018 修复）。
5. 跟上游 vLLM Ascend backend / vLLM CUDA / native eager baseline 对比。

**Phase 22 入场条件**：Phase 21 通过（精度对齐证明）。精度没对齐就跑 perf
等于报无意义数。

## Scope

**In:**
- 标准 workload 矩阵的 benchmark 脚本 `bench_vllm_backend_perf.py`
  （input length / output length / batch size）。
- DFX trace 集成（`PYPTO_DISTRIBUTED_DEP_GEN=1`,
  `PYPTO_DISTRIBUTED_L2_SWIMLANE=1`）—— pypto `03136bf6` 加的 env hook。
- 按 kernel 段做瓶颈归因（matmul / attention / collective / lm_head）。
- Tile size + chunk 常量调优。
- TP=8 多卡 scaling（gated，见下）。
- 对比报告。

**Out:**
- Continuous batching 调优（vLLM scheduler 上游来的；本阶段不改）。
- 量化（只跑 bf16；本阶段不做 fp8 / int8）。
- KV cache 压缩 / sparsity。

## Phase 22 多卡部分前的硬 gate

| Gate | 来源 | 2026-06-22 状态 | doc |
|------|------|----------------|-----|
| Barrier all_reduce UB-friendly 重写 | `csy0225/pypto-lib wip/step3p5-barrier-allreduce-20260622` HEAD `b5bb6ee`（UB overflow 让 dense ST device 退化） | 未启动；需要重写 | `pypto-lib/docs/known-pypto-pitfalls.md` §7 |
| MoE 507018 device runtime 修复 | Phase 19 milestone | 未启动；需要 `P19_DISPATCH_LIMIT` dispatch-cut bisect 工具 | （无独立 doc；归 TASK-30） |

单卡 Phase 22 工作（mixed-mode MoE，Phase 20 默认）独立于这两个 gate
往前走。多卡段等。

## Benchmark 矩阵

| 维度 | 值 | 备注 |
|------|----|------|
| Input（prompt）长度 | 128 / 1024 / 4096 | short-Q&A / 中等上下文 / 长上下文 |
| Output（max_tokens） | 16 / 64 / 256 | short-reply / 对话轮 / 长生成 |
| Batch size | 1 / 4 / 16 | 单 / 小 / 满 tile-specialised batch |
| TP world size | 1 / 8 | 单卡 + 生产 |
| MoE mode | mixed / full-pypto | full-pypto 受 MoE 507018 修复 gate |

每个 (config) run 的 metric：
- **TTFT**（time-to-first-token，prefill 主导；如果 prefill 被 Phase 17
  卡，用合成 prefill —— 见下面 "Prefill workaround"）
- **ITL**（inter-token latency，decode-only，ms/token）
- **TPS**（throughput，token/s across batch）
- **NPU 利用率**（AICore%, AIV%, HBM bandwidth%）
- **Per-kernel wallclock breakdown**（从 L2 swimlane 拿）

## Prefill workaround（Phase 17 被卡）

Phase 17 prefill MoE L1 overflow（TASK-29）独立卡着。Phase 22
**decode-only** perf 跳过 prefill：

1. 用合成数据填 KV cache 到目标 input length。
2. 设初始 `seq_lens = input_length` per batch。
3. 跑 decode loop `max_tokens` 步。
4. 报 decode-only metric（TPS / ITL）；TTFT 标 N/A 或单独"合成 prefill"
   测一次。

这是标准做法，prefill 被卡时 decode 侧数就是用户关心的 serving steady-
state 吞吐。

## 交付物

```
pypto-lib/tests/step3p5/
└── bench_vllm_backend_perf.py    # 标准 workload runner + metrics

pypto-project/  (本仓)
└── archive/                # 暂时；perf 报告会落 archive
  或 pypto-lib/docs/step3p5/perf-reports/
    ├── single-card-baseline-<date>.md
    ├── single-card-opt-round1-<date>.md
    ├── single-card-opt-round2-<date>.md
    └── multi-card-tp8-<date>.md   # gated
```

报告 markdown 落在 pypto-project 还是 pypto-lib 待 Phase 22 启动时决定
（perf 数 + 趋势更像项目级，建议落 pypto-project archive 或新建
`reports/`）。

## 任务

| # | 任务 | 输出 | 估时 |
|---|------|------|------|
| 3.1 | `bench_vllm_backend_perf.py` —— 标准 workload runner；sweep 矩阵；输出 CSV + JSON metric | bench 脚本 | 2 d |
| 3.2 | 单卡 dense+mixed-MoE baseline run；首个 number 表 | `single-card-baseline-<date>.md` | 1 d |
| 3.3 | DFX trace 抓：开 env hook，跑 10 步 decode，dump swimlane + dep-graph | trace files | 1 d |
| 3.4 | 瓶颈归因 —— 按 kernel 分段 wallclock：gate_up matmul / fa_fused / out_proj / tp_all_reduce / dispatch / combine / lm_head；定位 top-3 | 分析报告 | 2 d |
| 3.5 | Optimization round 1 —— tile-size 调优（`MLP_OUT_CHUNK`, `OUT_PROJ_K_CHUNK`, `KV_OUT_CHUNK` 等）；每改一处测增量 | `single-card-opt-round1-<date>.md` | 1-2 w |
| 3.6 | Optimization round 2 —— 跨 kernel L2 reuse / PSC（pipeline schedule） / cube/vector 并行度平衡 | `single-card-opt-round2-<date>.md` | 1-2 w |
| 3.7 | （Gated）TP=8 多卡 baseline —— barrier all_reduce + MoE 507018 修完后，跑 8 卡 | `multi-card-tp8-<date>.md` | 2-3 w（gate 后） |
| 3.8 | 最终对比报告：vs 上游 vLLM（torch eager）+ vs Ascend 官方 vLLM backend 如可用 | 对比 doc | 1 w |

## 准出条件

**单卡 baseline（无 gate）**:

`single-card-baseline-<date>.md` 发布，含：

- TPS / ITL / TTFT-or-synthetic 在 `(prompt_len, output_len, batch)` 矩阵
- NPU 利用率快照
- Per-kernel wallclock pie chart
- 跟 vLLM Ascend backend 对比（如可用），否则跟上游 torch eager 对比

**Optimization rounds**:

Round 1 和 Round 2 报告每个都展示 **≥ X% speedup** 相比前一轮（X 在
round 1 baseline 出来后定；Ascend kernel 典型 first-round 收益 20-40%）。

**多卡 baseline（gated）**:

barrier all_reduce + MoE 507018 修完后：
`multi-card-tp8-<date>.md` 含 TP=8 数 / scaling efficiency / 同 matrix。

## 风险与缓解

| 风险 | 概率 | 缓解 |
|------|------|------|
| 多卡 gate（barrier all_reduce + MoE 507018）4-6 周才能解 | 高 | 单卡 Phase 22 工作（任务 3.1-3.6）独立推进 |
| Mixed-mode MoE 让单卡数不有代表性（21/45 层走 vLLM eager 不走 pypto） | 高 | 报告分两栏：整体 token/s **和** dense-only token/s；后者是 "pypto kernel" 数 |
| 优化增量被测量噪声淹掉（首次编译、page fault、IO） | 中 | 测前 warmup 16 步；频率能 pin 就 pin；报 median + p95 |
| L2 swimlane dump 太大不好作 artifact 上传 | 低 | 聚合到 kernel 段，只放 summary plot；原始 trace 归到 NVMe |
| TP=8 perf scaling 差 因为 allreduce comm overhead 大 | 中-高 | Phase 22 明确把 comm time 当单独 metric；为下一阶段 compute/comm overlap 提供输入 |

## Status

- 2026-06-22：设计已落（本 doc）。
- 任务全部未启动。gate Phase 21 PASS。
- 多卡段额外 gate barrier all_reduce UB 修复 + MoE 507018 修复。
- 完整 Phase 22 ~6-8 周（单卡 3-4 周 + 多卡 3-4 周 post-gate）。

## References

- [`20-vllm-backend-monkey-patch.md`](20-vllm-backend-monkey-patch.md) —— Phase 20（e2e）
- [`21-precision-validation.md`](21-precision-validation.md) —— Phase 21（精度 gate）
- `pypto-lib/docs/known-pypto-pitfalls.md` §7 —— `pl.range(constant)` UB
  overflow（barrier all_reduce gate）
- pypto DFX env hook：`pypto/python/pypto/runtime/distributed_runner.py:399-405`
- `tools/p15_trace/run_with_trace.py` —— 现成的单 rank trace runner
