# Phase 21 — 精度验证 harness（与上游 vLLM 对比）

> **Component pin snapshot（doc 创建时，2026-06-22）**
>
> | 仓库 | 分支 | Pin | 备注 |
> |------|------|-----|------|
> | pypto-lib | `stepfun/develop` | `a6b5faa`（pre-commit） | 加这文件的 commit 之后会更新 |
> | pypto | `stepfun/develop` | `b00c8b23` | |
> | pto-isa | `stepfun/develop` | `e25732f0` | |
> | PTOAS | `stepfun/develop` | `da011a3d` | binary v0.45 |
> | simpler（submodule） | — | `a6e06406` | |
> | vLLM（参考） | stepcast/develop | `0e0901376` | 对比 baseline |

## 目标

建一个精度验证 harness：用**同一个** prompt 跑 (a) 上游 vLLM
（`Step3p5Model.forward` torch eager）和 (b) pypto-backed vLLM
（先 `B.install()`），然后 3 个粒度对比。**Phase 21 PASS 是 Phase 22
perf 工作入场条件** —— 精度没对齐之前跑 perf 等于报无意义数。

## Scope

**In:**
- 双引擎 harness：同 prompt 同 seed 起两个 vLLM 引擎，一个不动，一个
  装 `pypto.step3p5.vllm_backend.install()`。
- 三层对比（L1 / L2 / L3，见下）。
- 容忍区间按 Phase 19 ST 约定。
- 覆盖矩阵：attention 类型（full / SWA）/ batch / prompt 类别 / decode
  步数。
- CI 可跑：`pytest tests/step3p5/test_vllm_backend_*.py` 返回绿或可
  actionable 的 diagnostic。

**Out:**
- 多卡精度测试（Phase 22 + 多卡 gate）。
- 全 pypto MoE 精度（Phase 22 + MoE 507018 修复）。
- 超出 exit criterion 之外的 numerical-debug 工具。

## 三层对比

| Tier | 对象 | 容忍 | 测什么 |
|------|------|------|--------|
| **L1 — per-layer** | 每个 `Step3p5DecoderLayer.forward` 后的 `hidden_states` `[B, H]`（or per-fused-block 在整模型 patch 时） | `ratio_allclose(atol=0.04, rtol=0.04, max_error_ratio=0.10)`（Phase 19 ST 约定） | kernel-级数值正确性；定位哪个层 / op drift |
| **L2 — per-token logits** | 采样前 logits `[B, VOCAB]` | cosine_similarity ≥ 0.999 **and** top-K (K=5) overlap ≥ 4/5 | 数值稳定 + lm_head + 最终 RMS gather 正确 |
| **L3 — per-token sampling** | 每步 decode 采到的 `token_id`（greedy, temperature=0） | 64 步 × 16 prompt top-1 match rate ≥ 95% | end-user 视角行为对齐 |

L1 信号最强但 noise 最大；L3 是我们对外的标准。L1 不过但 L3 过意味着
精度漂移但还在 argmax basin 里 —— 对某些 workload 可接受，对其他不行
（要标记）。

## 容忍区间的依据

L1 的数继承自 Phase 19 dense ST device run（bf16 路径 + FP32
accumulator）通过的容忍。Phase 21 **不**放宽这个 —— 如果 `decode_fwd`
端到端突破 `atol=0.04` 是真退化。

L2 cos 0.999 阈值是针对 head_gate ×1 旁路标定的：旁路 sigmoid gate 让
attention-out 比上游大 ~2×，但经过 RMS+lm_head 之后 cosine 仍然高。
Phase 15 单卡 run 经验上 cos ≈ 0.9995。

L3 95% 阈值匹配同一个模型的两个实现间，其中一个有 bf16 量化差时典型
greedy decode top-1 match。

## 覆盖矩阵

| 变量 | 值 | 为什么 |
|------|----|--------|
| Attention 类型 | full（24 层）/ SWA（21 层） | 都得过；SWA 窄 KV — 不同 paged path |
| Batch size | 1 / 4 / 16 | 抓 batch dim 广播 bug；pypto kernel 在 16 上 tile-specialized |
| Prompt 类别 | en-short / en-long / zh / code / math | 不同 vocab region 和序列长度 |
| Decode 步数 | 1 / 16 / 64 | 抓只在多次 KV-cache append 后才浮现的 bug |
| MoE 模式 | `mixed`（Phase 21 默认） / `dummy0`（sanity） | Phase 21 在 507018 修好前测不了全 pypto MoE |

完整矩阵 `2 × 3 × 5 × 3 × 2 = 180` test case。CI 子集（fast tier）
`1 × 1 × 5 × 16 × 1 = 80` run。

## 交付物

```
pypto-lib/tests/step3p5/
├── _vllm_precision_harness.py    # DualRunHarness + hook helpers
├── _vllm_test_prompts.py         # 16 fixed prompts (en/zh/code/math × short/long)
├── test_vllm_backend_per_layer.py    # L1
├── test_vllm_backend_per_token.py    # L2
└── test_vllm_backend_decode_n.py     # L3
```

加 `build_output/precision_reports/<timestamp>/` 下报告：

- `per_layer.json` —— 每个 (layer_idx, prompt_idx, step_idx) 的 max
  abs/rel diff，ratio_allclose pass/fail
- `per_token.json` —— 每个 (prompt_idx, step_idx) 的 cos / top-K overlap
- `per_decode.json` —— 每个 prompt 64 步的 top-1 match rate

## 任务

| # | 任务 | 输出 | 估时 |
|---|------|------|------|
| 2.1 | `_vllm_precision_harness.py:DualRunHarness` —— 同 prompt 同 seed 起两个引擎，forward hook 抓中间状态；支持 `per_layer=True` 模式用 Phase 20 escape hatch | reusable harness 类 | 4 d |
| 2.2 | `_vllm_test_prompts.py` —— 16 固定 prompt | 测试输入 | 0.5 d |
| 2.3 | L1 per-layer 测试 —— 每个 prompt+step，两边 hook 45 层输出，跑 `ratio_allclose`；fail 报具体层 | `test_vllm_backend_per_layer.py` | 3 d |
| 2.4 | L2 per-token logits 测试 —— 采样前 `[B, VOCAB]` cos + top-K overlap | `test_vllm_backend_per_token.py` | 2 d |
| 2.5 | L3 per-token sampling 测试 —— greedy decode `temperature=0`，top-1 match rate aggregate | `test_vllm_backend_decode_n.py` | 2 d |
| 2.6 | 覆盖矩阵 parametrize —— pytest fixture sweep `(att_type, batch, prompt_cat, n_steps)` | parametrized tests | 2 d |
| 2.7 | head_gate ×1 标定 —— patch 上游 vLLM `Step3p5Attention` 也走 ×1 旁路对齐 baseline，**或者**对 head_gate-affected 路径加宽 L1 tolerance；记录选择 | 标定 patch / tolerance note | 2 d |
| 2.8 | CI hookup —— 本地 CI 脚本 +（可选）GitHub Actions workflow | 绿 CI | 2 d |

## 准出条件

fast-tier 子集（80 run）：

```bash
pytest tests/step3p5/test_vllm_backend_per_layer.py -v   # 45 层 × 16 prompts → 100% layer-pass
pytest tests/step3p5/test_vllm_backend_per_token.py -v   # 16 prompts × 16 steps → cos≥0.999 + topK≥4/5 全过
pytest tests/step3p5/test_vllm_backend_decode_n.py -v    # 16 prompts × 64 steps → top-1 match rate ≥ 95%
```

三个都绿。JSON 报告归档供 trend 跟踪。

## 风险与缓解

| 风险 | 概率 | 缓解 |
|------|------|------|
| head_gate ×1 旁路下 L1 hidden_states 无法严格匹配 —— 语义差不可消，除非 TASK-L (cube-matmul 块对角 R) 落地 | 高 | 任务 2.7 —— patch vLLM 侧 gate 也 ×1 对齐 baseline，**或**把 head_gate-affected 层放到独立的较松 tolerance 段 |
| 两引擎 RNG state 漂（sampling、dropout init、weight init 顺序）让 L2/L3 noise 大 | 中 | 强制 `temperature=0`、固定 seed、deterministic init order；assert 推理时无 dropout |
| vLLM `FusedMoEBlock`（mixed-mode 走的）自己有量化/融合，跟数学参考不 byte-identical | 中 | MoE 层：Phase 21 把 vLLM 的融合输出当 reference（即 mixed-mode 下 MoE 层是 baseline，不是测试目标） |
| hook 45 层 / step 开销大，测试拖慢 | 低 | L1 只在子集 step 跑（1、8、64）；L2/L3 走 Phase 20 整模型路径 |
| KV cache layout 跨 run 漂，导致 attention 输出不同 | 低 | L1 时两边 pre-fill KV 同样合成数据；L3 让它自然漂，只看采样 token |

## Status

- 2026-06-22：设计已落（本 doc）。
- 任务 2.1-2.8 未启动。gate Phase 20 完成（要 `B.install()` 能跑）。
- Phase 20 落地后估时 ~3-4 周。

## References

- [`20-vllm-backend-monkey-patch.md`](20-vllm-backend-monkey-patch.md) —— Phase 20 前置
- [`22-perf-baseline.md`](22-perf-baseline.md) —— Phase 22 接力（被本
  phase 把关）
- Phase 19 dense ST tolerance：`tests/step3p5/test_decode_layer_full_dense_st.py`
- vLLM 参考：`<vllm_repo>/vllm/model_executor/models/step3p5.py`
