# 实时状态

pypto step3p5 项目的实时状态板。**任何 phase / sub-task / blocker 状态
变化都更新这里**。历史细节查 [`archive/`](archive/)。

**最后更新**：2026-06-26

---

## 阶段跟踪

| 阶段 | 标题 | 状态 | 详情 |
|-----:|------|------|------|
| **1** | **pypto kernel 原型** | ✅ **已完成** | [`archive/prototype-phase-01-19-summary.md`](archive/prototype-phase-01-19-summary.md) |
| **2** | **vLLM Ascend 后端集成** | 🟡 **进行中**（设计已落） | 见下 |

### Phase 2 sub-phases

| Sub-phase | 范围 | 状态 | 文档 | 估时 |
|-----------|------|------|------|------|
| **2.0（Phase 20）** | vLLM monkey-patch e2e — 整模型 patch `Step3p5Model.forward`；单卡/TP=8 mixed/full PyPTO runner 接入 | 🟡 **待实现**；dump-based 精度 blocker 已清，但 production backend 未接 | [`phases/20-vllm-backend-monkey-patch.md`](phases/20-vllm-backend-monkey-patch.md) | 3-4 周 |
| **2.1（Phase 21）** | 与 vLLM 原生精度对比 harness；L1/L2/L3 三层 | ✅ **dump-based 精度闭环完成**（BF16 decode、W8A8 decode、W8A8 prefill）；待 Phase20 backend 后做在线 L1/L2/L3 gate | [`phases/21-precision-validation.md`](phases/21-precision-validation.md) | 1-2 周补在线 gate |
| **2.2（Phase 22）** | Perf baseline + 两轮优化；TP=8 多卡 | 📐 设计已落；gate Phase 21 + 2 个硬 blocker | [`phases/22-perf-baseline.md`](phases/22-perf-baseline.md) | 6-8 周 |

**到 v1.0 production decode 的总目标**：自 2026-06-22 起约 12-16 周
（含 gate 任务的并行投入）。

---

## Phase 2 交付物分级（跟踪现在到了哪个 sub-version）

| Tier | 能跑什么 | 需要 Phase 2 哪几部分 | 需要清掉哪些 blocker |
|------|----------|----------------------|----------------------|
| **v0.1** | 单卡 dense + mixed-mode MoE 走 vLLM | Phase 20 | 无 |
| **v0.2** | 单卡 45 层 mixed-mode（dense pypto + MoE vLLM eager） | Phase 20 | 无 |
| **v0.3** | TP=8 多卡 dense + mixed-mode MoE | Phase 20 + Phase 22.1-3 | ✅ kernel blocker 已清；待 vLLM harness |
| **v1.0** | TP=8 / EP=8 全 pypto MoE + perf 数发布 | Phase 20-22 全完 | 待整网精度 + perf 优化（split task 融合） |

**当前口径拆分**：

- ✅ **dump-based 精度验证已闭环**：BF16 decode、W8A8 decode、W8A8 prefill（1k/4k/8k/32k/64k/128k）均已用 vLLM eager detail dump 作为 oracle，在 PyPTO 侧 reference/detail/final-logits 口径通过。
- 🟡 **production backend 仍未完成**：上述验证并不等价于“真实 vLLM 请求已经走 PyPTO NPU full runner”。后续仍需 Phase 20 把 `Step3p5DecodeFwd`/prefill runner、权重翻译、KV/cache ABI、vLLM monkey-patch 接入生产路径。
- 🟡 **真实 PyPTO NPU prefill kernel 仍待开发**：本轮 W8A8 prefill 是 vLLM golden + PyPTO reference 对齐；`prefill_moe.py` 的 L1 overflow 仍阻塞真正 PyPTO NPU prefill kernel。

---


### Step3p5 W8A8 prefill precision closure (2026-06-26)

在 0162 目标机完成 W8A8 prefill 多长度精度闭环，流程对齐 decode 阶段：vLLM eager W8A8 detail dump 作为 oracle，PyPTO 侧复算非 attention-core per-layer detail，并对 final RMSNorm + LM-head logits 做端到端比较。

- ✅ 覆盖长度：`1k / 4k / 8k / 32k / 64k / 128k`（`1024, 4096, 8192, 32768, 65536, 131072`）。
- ✅ vLLM W8A8 prefill golden：`/mnt/nvme1/chensiyu/logs/step3p5_910b_w8a8_prefill_v001/golden_step3p5_w8a8_prefill_vllm_sampled`；长序列采用 sampled detail dump（每个 forward 最多 128 rows）并裁剪到 PyPTO comparator 所需 tensor，避免 128k full dump 爆盘。
- ✅ detail + final logits 报告：`/mnt/nvme1/chensiyu/logs/step3p5_910b_w8a8_prefill_v001/pypto_prefill_precision/STEP3P5_W8A8_PREFILL_REPORT.json`，`ok=true`；acceptance 为 sampled W8A8 prefill detail `pass_rate >= 0.997`，final logits 全 case PASS。
- ✅ 各长度 worst pass rate：1k `0.999349`，4k `0.998698`，8k `0.999023`，32k `0.999349`，64k `0.999756`，128k `0.997559`。
- ✅ ST：`STEP3P5_PREFILL_REPORT_ROOT=/mnt/nvme1/chensiyu/logs/step3p5_910b_w8a8_prefill_v001/pypto_prefill_precision PYTHONPATH=. pytest -q tests/step3p5/test_step3p5_w8a8_prefill_st.py` PASS (`1 passed in 0.01s`)。

W8A8 prefill 回归数据包：`/mnt/nvme1/chensiyu/logs/step3p5_910b_w8a8_prefill_v001/step3p5_w8a8_prefill_regression_20260626.tar`，SHA256 `cd34f034e017c68437547e5f7f453a2f6b481a1e97e162a89ac21c422fe76b6e`。报告归档：[`archive/step3p5-w8a8-prefill-delivery-20260626.md`](archive/step3p5-w8a8-prefill-delivery-20260626.md)。代码提交：`pypto-lib` `81252e9`（随后 Phase 20 config-align 工具提交推进到 `e616407`）。

### Step3p5 W8A8 vLLM-vs-PyPTO precision closure (2026-06-26)

本轮按 BF16 golden 构造方式在 0162 重新部署 W8A8 checkpoint `/mnt/nvme1/chensiyu/step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp`，vLLM 使用 `--quantization ascend`、eager、TP=EP=8、NPU 8-15、port 8001 采集 int8/W8A8 detail dump，没有复用 BF16 golden。

- ✅ W8A8 权重加载：`weight_loader.py` 支持 `quant_model_weights.safetensors.index.json`，按 per-expert `weight_scale/weight_offset` 反量化 routed MoE INT8 权重到 PyPTO bundle。
- ✅ W8A8 routed MoE reference：detail compare 自动启用 dynamic activation quantization，匹配 vLLM W8A8_DYNAMIC routed expert 路径。
- ✅ vLLM W8A8 golden：`/mnt/nvme1/chensiyu/logs/step3p5_910b_w8a8_v001/golden_step3p5_w8a8_vllm_20260626_004648`，`beijing_1tok` 共 5944 个 `.pt` dump。
- ✅ acceptance：`decode_acceptance_w8a8_rank0.json` `ok=true`，48 层 dispatcher 覆盖。
- ✅ 主 45 层 detail：`pypto_all_layers_detail_compare_w8a8_beijing1_atol1_report.json` `ok=true`，3960 checks，worst pass rate `0.9995659589767456`。
- ✅ final logits e2e：`pypto_final_logits_from_vllm_w8a8/final_logits_report.json` `ok=true`，full-vocab pass rate `1.0`，argmax token `3648` 匹配。
- ✅ ST：`pytest -q tests/step3p5/test_weight_loader_w8a8.py tests/step3p5/test_step3p5_w8a8_e2e_st.py` PASS (`6 passed in 1.30s`)。

W8A8 回归数据包：`/mnt/nvme1/chensiyu/logs/step3p5_910b_w8a8_v001/step3p5_w8a8_e2e_st_regression_20260626.tar`，SHA256 `6f0a0f8e61f54d160325150917474209a0e493e987a77318aaeb1519c3915909`。端到端测试报告：[`archive/step3p5-w8a8-e2e-delivery-20260626.md`](archive/step3p5-w8a8-e2e-delivery-20260626.md)（目标机原件：`/mnt/nvme1/chensiyu/logs/step3p5_910b_w8a8_v001/STEP3P5_W8A8_E2E_ST_REPORT.md`）。代码提交：`pypto-lib` `b918e60`。


### Step3p5 BF16 vLLM-vs-PyPTO detail precision closure (2026-06-25)

本轮在 0162 isolated vLLM 容器中使用 BF16 checkpoint `/mnt/nvme1/chensiyu/step3p5_flash_release_hf_mtp3_bf16` 采集真实请求 detail dump，并在 PyPTO 侧完成逐层 tensor-input 对齐：

- ✅ 权重加载/dispatcher acceptance：`tools/step3p5/decode_acceptance.py --json` PASS。
- ✅ final logits 全 step：`pypto_final_logits_from_vllm_all_steps_eps1e5/final_logits_report.json` `ok=True`。
- ✅ 主 45 层 detail：`0~44` 共 `3960` checks，worst pass rate `0.9995659589767456`，报告 `/mnt/nvme1/chensiyu/logs/step3p5_910b_v017/pypto_all_layers_detail_compare_topk_final_atol02_report.json`。
- ✅ MTP3 detail：`45~47` 共 `279` checks，worst pass rate `0.9995659589767456`，报告 `/mnt/nvme1/chensiyu/logs/step3p5_910b_v017/pypto_mtp3_detail_compare_report.json`。
- ✅ ST：`test_step3p5_all_layers_detail_st.py` + `test_step3p5_mtp3_detail_st.py` 组合 PASS (`2 passed in 286.34s`)。

关键修复：`models/step3p5/config.py` 的 `EPS` 从 `1e-6` 修正为 vLLM `GemmaRMSNorm` 实际使用的 `1e-5`；MoE 对齐使用 vLLM fused router dump 的 `topk_ids/topk_weights` 驱动 PyPTO MoE reference。

BF16 回归数据包：`/mnt/nvme1/chensiyu/logs/step3p5_910b_v017/step3p5_bf16_e2e_st_regression_20260625.tar`。

**本次涉及仓库 commit 组合（记录于 2026-06-26）**：

| 仓库 | 分支/用途 | Commit | 备注 |
|------|-----------|--------|------|
| `pypto-lib` | `stepfun/develop` | `588610e` | 新增 vLLM patch autoload helper；monkey patch skeleton `9718083`，weight contract `0511d27` |
| `pypto-project` | `main` | `b771c7e` | 首次记录本次验收状态的文档提交；本段会由后续文档提交推进 |
| `pypto` | `stepfun/develop` | `b00c8b23` | 本次未改代码；沿用当前 pin |
| `pto-isa` | `stepfun/develop` | `e25732f0` | 本次未改代码；沿用当前 pin |
| `PTOAS` | `stepfun/develop` | `da011a3d` | 本次未改代码；沿用当前 pin |
| `simpler` | submodule/runtime pin | `c66b4120` | 本次未改代码；沿用当前 pin |

BF16 回归数据包：`/mnt/nvme1/chensiyu/logs/step3p5_910b_v017/step3p5_bf16_e2e_st_regression_20260625.tar`，SHA256 `bce502f4cbafb61fe541385ab1828d33a1f9c32bdfb7d2009e871adba4c896c4`。



### MoE 8-card precision ST update (2026-06-24 evening)

本轮在 0162 当前代码和 CANN 9.0.0 non-GA 环境下补齐了 `test_decode_layer_moe_st --world-size 8` 的 rank-wise golden：

- ✅ dense 多卡基线：`test_decode_layer_full_dense_multirank_st -p a2a3 -d 0,1,2,3,4,5,6,7` PASS，8 rank `bad_ratio=0.0004`。
- ✅ MoE smoke：真实模型 MoE variants compile-only 全 PASS。
- ✅ MoE 8 卡真实模型组合 golden：`full_silu_silu` / `full_swiglu7_silu` / `full_swiglu7_swiglu16` / `swa_silu_silu` / `swa_swiglu7_silu` PASS。

相关 0162 日志位于 `/data/chensiyu/hw_project/pypto/workspace/moe8-precision-st-*.log`；代码侧进展同步在 `pypto-lib` 的 `test_decode_layer_moe_st.py` 和 `docs/upstream-issues/step3p5-moe-8card-fence-gap.md`。

### Final e2e precision gate preflight (2026-06-24; superseded by 2026-06-25/26 dumps)

`pypto-lib/tools/step3p5/e2e_precision_readiness.py` 作为早期预检保留；其中“checkpoint/vLLM 不可见”等环境 blocker 已由 0162 上 BF16/W8A8 dump-based precision 闭环解除。当前仍有效的结论是：

- ✅ `decode_fwd` torch distributed mock：worst pass rate 1.0。
- ✅ `step3p5_decode` synthetic smoke：pass rate 1.0。
- ✅ MoE 8 卡 ST 已补 rank-wise golden；真实模型会遇到的 5 个 MoE variant 全 PASS。
- 🟡 `Step3p5DecodeFwd`/prefill runner 尚未接入真实 vLLM online backend；见 Phase 20 production backend blocker。
- 🟡 head_gate ×1 parity 策略仍待在线 backend L1 gate 决策。

## 立即可做的下一步（按优先级）

1. **Phase 20 backend 接入（P1）**：`config_align.py` 已启动并在 W8A8 checkpoint 上 PASS（pypto-lib `e616407`）；`weight_translate.py` 已提供 per-rank bundle manifest/export contract（pypto-lib `0511d27`）；`vllm_monkey_patch.py` 已提供 tail/shadow/full patch surface（full 目前 fail-closed，等待真实 runner；pypto-lib `9718083`；autoload helper `588610e`）。下一步接 vLLM `nn.Module` in-memory 权重翻译，并把 `Step3p5DecodeFwd`/runner 接到 vLLM 请求路径。
2. **真实 PyPTO prefill NPU kernel（P2）**：重构 `prefill_moe.py`，用 multi-step gate/up chunking 清 L1 overflow，完成 1k~128k NPU prefill ST。
3. **在线精度 gate（P3）**：Phase 20 backend 能跑后，补 vLLM patched backend 的 L1/L2/L3 gate；当前 dump-based precision artifacts 作为 oracle/regression baseline。
4. **性能 baseline（P3/P4）**：做 decode-only TPS/ITL、prefill TTFT、1k~128k 性能曲线；分析 MoE dispatch/combine、TP/EP 通信、host launch overhead。
5. **MTP speculative 集成（P4）**：把 MTP 拼进 `decode_fwd` 和 vLLM speculative pipeline；该项不阻塞当前 correctness。 

---

## 组件 Pin Snapshot（最新一行）

| 日期 | 事件 | pypto | pypto-lib | pto-isa | PTOAS（src） | simpler（submodule） | ptoas-bin |
|------|------|-------|-----------|---------|--------------|---------------------|-----------|
| 2026-06-26 | W8A8 prefill PASS + Phase20 config/weight/monkey-patch surface 启动 | `stepfun/develop:b00c8b23` | `stepfun/develop:588610e`（autoload helper；vLLM monkey patch skeleton `9718083`；weight_translate `0511d27`） | `stepfun/develop:e25732f0` | `stepfun/develop:da011a3d` | `c66b4120` | `v0.45` |
| 2026-06-22 | Phase 2 设计落地；建项目跟踪仓 | `stepfun/develop:b00c8b23` | `stepfun/develop:b918e60`（W8A8 precision alignment；BF16 0~47 detail ST 基线 `d4c01b9`） | `stepfun/develop:e25732f0` | `stepfun/develop:da011a3d` | `a6e06406` | `v0.45` |

历史 pin snapshot 见 [`archive/milestones-2026-Q2.md`](archive/milestones-2026-Q2.md)。

---

## 当前 Blocker / Deferred Items

| # | Blocker | 严重度 | gate 什么 | Owner | 详情 |
|--:|---------|--------|-----------|-------|------|
| 1 | Phase 20 production backend 未接入 | 🟡 功能 | 真实 vLLM 请求走 PyPTO runner | 未指派 | `phases/20-vllm-backend-monkey-patch.md` |
| 2 | Prefill MoE L1 overflow（TASK-29） | 🟡 功能/性能 | 真实 PyPTO NPU prefill kernel + TTFT | 未指派 | [`blockers.md`](blockers.md) §2 |
| 3 | head_gate × 1 旁路 — vLLM 原生语义偏离 | 🟡 精度 | 在线 backend L1 layer parity | TASK-L（pto-isa 上游） | [`blockers.md`](blockers.md) §1 |
| 4 | 0234 driver+firmware 升级未做 | 🟢 基础设施 | 备用部署机 | 未指派 | [`blockers.md`](blockers.md) §3 |
| 5 | MTP 集成进 `decode_fwd` | 🟢 Deferred | speculative decoding 吞吐 | 未指派 | [`blockers.md`](blockers.md) §6 |

---

## `gpu-a910x-0162`（Phase 16 验证机）目前已确认能跑

| 组件 | 验证 | 备注 |
|------|------|------|
| driver 25.5.2 | ✅ 2026-06-22 | `npu-smi info -t board -i 0` 报上 |
| firmware 7.8.0.7.220 | ✅（chip flash） | 跨重启持久 |
| CANN 9.0.0 non-GA/non-beta | ✅ `/usr/local/Ascend/cann` → `/mnt/persist/Ascend/cann-9.0.0/cann-9.0.0` | 2026-06-24 已重装并重编译 pypto/runtime |
| simpler L3 allreduce_distributed -d 0-1 | ✅ 2026-06-24 | 1 passed / 1 skipped（pytest harness） |
| pypto-lib 前端 smoke rc=0 | ✅ 2026-06-24 | `_smoke_program_build` 通过 |
| Decode dense full ST @ device 0 | ✅ 8.54s（ratio_allclose PASS，2026-06-24） | CANN 9.0.0 non-GA 重编译后验证 |
| Decode dense SWA ST @ device 0 | ✅ 15.61s（ratio_allclose PASS，2026-06-24） | CANN 9.0.0 non-GA 重编译后验证 |
| Phase 19 MoE real-model variants smoke compile | ✅ PASS | TP=8 per-rank slice 路径 |
| Decode MoE full_silu_silu ST @ 8 cards | ✅ golden PASS 32.61s（2026-06-24） | rank-wise golden；retry 后通过 |
| Decode MoE full_swiglu7_silu ST @ 8 cards | ✅ golden PASS 27.64s（2026-06-24） | full attention + routed swiglu7 + shared silu |
| Decode MoE full_swiglu7_swiglu16 ST @ 8 cards | ✅ golden PASS 26.74s（2026-06-24） | full attention + routed swiglu7 + shared swiglu16 |
| Decode MoE swa_silu_silu ST @ 8 cards | ✅ golden PASS 33.61s（2026-06-24） | SWA + routed/shared silu |
| Decode MoE swa_swiglu7_silu ST @ 8 cards | ✅ golden PASS 35.97s（2026-06-24） | retry 后通过；前一轮出现 transient 507018 |
| Phase 15 单卡 e2e | ✅ rc=0，20 tasks complete | head_gate ×1 旁路 + TP=1 patch 路径 |

---

## `gpu-a910x-0234` 当前状态

未升级。driver `25.5.1` / firmware `7.8.0.6.201` / CANN `9.0.0-beta.1`。
多卡 e2e 因 driver shmem-exbus cap 缺口而被卡，必须先升 driver+firmware。
`.run` 包已 stage 在 0162 `/mnt/persist/ascend-staging/` —— 升级 runbook
见 [`deployment/machine-recovery.md`](deployment/machine-recovery.md)。
