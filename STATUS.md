# 实时状态

pypto step3p5 项目的实时状态板。**任何 phase / sub-task / blocker 状态
变化都更新这里**。历史细节查 [`archive/`](archive/)。

**最后更新**：2026-06-25

---

## 阶段跟踪

| 阶段 | 标题 | 状态 | 详情 |
|-----:|------|------|------|
| **1** | **pypto kernel 原型** | ✅ **已完成** | [`archive/prototype-phase-01-19-summary.md`](archive/prototype-phase-01-19-summary.md) |
| **2** | **vLLM Ascend 后端集成** | 🟡 **进行中**（设计已落） | 见下 |

### Phase 2 sub-phases

| Sub-phase | 范围 | 状态 | 文档 | 估时 |
|-----------|------|------|------|------|
| **2.0（Phase 20）** | vLLM monkey-patch e2e — 整模型 patch `Step3p5Model.forward`；单卡 TP=1；mixed-mode MoE | 📐 设计已落；**任务 1.1-1.9 未启动** | [`phases/20-vllm-backend-monkey-patch.md`](phases/20-vllm-backend-monkey-patch.md) | 3-4 周 |
| **2.1（Phase 21）** | 与 vLLM 原生精度对比 harness；L1/L2/L3 三层 | 📐 设计已落；gate Phase 20 | [`phases/21-precision-validation.md`](phases/21-precision-validation.md) | 3-4 周 |
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

**当前**：Step3p5 BF16 decode 精度验证已完成 vLLM dump-based 整网闭环：主层 `0~44` + MTP3 `45~47` 全部逐层 tensor-input 对齐通过；final logits 全 step 对齐通过；权重加载/dispatcher/acceptance 通过。当前验证口径是 vLLM eager all-to-all 真实请求 detail dump 作为 oracle，PyPTO torch/reference 逐层复算非 attention-backend 内核边界并比较 `layer_out/logits`。若要宣称 PyPTO 自身 NPU full decode runner 从输入一路执行到 logits，还需后续接 `Step3p5DecodeFwd`/MTP runtime runner；但精度 blocker 已清.

---

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
| `pypto-lib` | `stepfun/develop` | `b918e60` | W8A8 precision alignment；BF16 gate 基线为 `d4c01b9` |
| `pypto-project` | `main` | `b771c7e` | 首次记录本次验收状态的文档提交；本段会由后续文档提交推进 |
| `pypto` | `stepfun/develop` | `b00c8b23` | 本次未改代码；沿用当前 pin |
| `pto-isa` | `stepfun/develop` | `e25732f0` | 本次未改代码；沿用当前 pin |
| `PTOAS` | `stepfun/develop` | `da011a3d` | 本次未改代码；沿用当前 pin |
| `simpler` | submodule/runtime pin | `c66b4120` | 本次未改代码；沿用当前 pin |

BF16 回归数据包：`/mnt/nvme1/chensiyu/logs/step3p5_910b_v017/step3p5_bf16_e2e_st_regression_20260625.tar`，SHA256 `bce502f4cbafb61fe541385ab1828d33a1f9c32bdfb7d2009e871adba4c896c4`。



### MoE 8-card precision ST update (2026-06-24 evening)

本轮在 0162 当前代码和 CANN 9.0.0 non-GA 环境下补齐了 `test_decode_layer_moe_st --world-size 8` 的 rank-wise golden：

- ✅ dense 多卡基线：`test_decode_layer_full_dense_multirank_st -p a2a3 -d 0,1,2,3,4,5,6,7` PASS，8 rank `bad_ratio=0.0004`。
- ✅ MoE smoke：6 variants compile-only 全 PASS。
- ✅ MoE 8 卡真实模型组合 golden：`full_silu_silu` / `full_swiglu7_silu` / `full_swiglu7_swiglu16` / `swa_silu_silu` / `swa_swiglu7_silu` PASS。
- ℹ️ `swa_swiglu7_swiglu16` runtime 完成但 validate 全 NaN；该组合不在真实模型 layer table 中，仅保留为 synthetic stress coverage，不计入 active blocker。

相关 0162 日志位于 `/data/chensiyu/hw_project/pypto/workspace/moe8-precision-st-*.log`；代码侧进展同步在 `pypto-lib` 的 `test_decode_layer_moe_st.py` 和 `docs/upstream-issues/step3p5-moe-8card-fence-gap.md`。

### Final e2e precision gate (2026-06-24)

新增可执行预检：`pypto-lib/tools/step3p5/e2e_precision_readiness.py`。当前结果：

- ✅ `decode_fwd` torch distributed mock：worst pass rate 1.0
- ✅ `step3p5_decode` synthetic smoke：pass rate 1.0
- ✅ MoE 8 卡 ST 已补 rank-wise golden；真实模型会遇到的 5 个 MoE variant 全 PASS
- ❌ checkpoint 不可见：`/mnt/chensiyu-jfs/.../step3p5_flash_release_hf_mtp3_bf16` 未挂载到 0162
- ❌ vLLM / stepcast 原生模型环境不可见
- ❌ `Step3p5DecodeFwd.host_orch` 仍未接 45 层 per-layer program，只跑 final RMS + LM head
- ❌ head_gate ×1 parity 策略未定

## 立即可做的下一步（按优先级）

1. **Phase 20.1**：`config_align.py` — 校验 vLLM `hf_config` 与 pypto `config.py` 常量。
2. **Phase 20.2**：`weight_translate.py` — vLLM `nn.Module` → pypto bundle dict。
3. **Phase 21 入场准备**：先跑整网 decode-only 端到端精度对齐（L1 hidden / L2 logits），确认 head_gate ×1 旁路的可接受策略。
4. **后续性能优化**：当前 MoE dispatch 采用 split task 保正确性；恢复/融合成非 split task 作为 Phase 22 perf 优化项，不阻塞精度 harness。

---

## 组件 Pin Snapshot（最新一行）

| 日期 | 事件 | pypto | pypto-lib | pto-isa | PTOAS（src） | simpler（submodule） | ptoas-bin |
|------|------|-------|-----------|---------|--------------|---------------------|-----------|
| 2026-06-22 | Phase 2 设计落地；建项目跟踪仓 | `stepfun/develop:b00c8b23` | `stepfun/develop:b918e60`（W8A8 precision alignment；BF16 0~47 detail ST 基线 `d4c01b9`） | `stepfun/develop:e25732f0` | `stepfun/develop:da011a3d` | `a6e06406` | `v0.45` |

历史 pin snapshot 见 [`archive/milestones-2026-Q2.md`](archive/milestones-2026-Q2.md)。

---

## 当前 Blocker / Deferred Items

| # | Blocker | 严重度 | gate 什么 | Owner | 详情 |
|--:|---------|--------|-----------|-------|------|
| 1 | head_gate × 1 旁路 — vLLM 原生语义偏离（sigmoid gate 用 identity 替代） | 🟡 精度 | Phase 21 L1 layer-级 parity | TASK-L（pto-isa 上游） | [`blockers.md`](blockers.md) §1 |
| 2 | Prefill MoE L1 overflow（TASK-29） | 🟢 Deferred | Phase 17 prefill e2e（Phase 22 decode-only 不需要） | 未指派 | [`blockers.md`](blockers.md) §2 |
| 3 | 0234 driver+firmware 升级未做 | 🟢 基础设施 | 备用部署机 | 未指派 | [`blockers.md`](blockers.md) §3 |

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
| Phase 19 MoE 6 variants smoke compile | ✅ 6/6 PASS | TP=8 per-rank slice 路径 |
| Decode MoE full_silu_silu ST @ 8 cards | ✅ golden PASS 32.61s（2026-06-24） | rank-wise golden；retry 后通过 |
| Decode MoE full_swiglu7_silu ST @ 8 cards | ✅ golden PASS 27.64s（2026-06-24） | full attention + routed swiglu7 + shared silu |
| Decode MoE full_swiglu7_swiglu16 ST @ 8 cards | ✅ golden PASS 26.74s（2026-06-24） | full attention + routed swiglu7 + shared swiglu16 |
| Decode MoE swa_silu_silu ST @ 8 cards | ✅ golden PASS 33.61s（2026-06-24） | SWA + routed/shared silu |
| Decode MoE swa_swiglu7_silu ST @ 8 cards | ✅ golden PASS 35.97s（2026-06-24） | retry 后通过；前一轮出现 transient 507018 |
| Decode MoE swa_swiglu7_swiglu16 ST @ 8 cards | ℹ️ synthetic-only FAIL（2026-06-24） | 真实模型不会遇到该组合；runtime 完成但 `next_hidden_out` 全 NaN |
| Phase 15 单卡 e2e | ✅ rc=0，20 tasks complete | head_gate ×1 旁路 + TP=1 patch 路径 |

---

## `gpu-a910x-0234` 当前状态

未升级。driver `25.5.1` / firmware `7.8.0.6.201` / CANN `9.0.0-beta.1`。
多卡 e2e 因 driver shmem-exbus cap 缺口而被卡，必须先升 driver+firmware。
`.run` 包已 stage 在 0162 `/mnt/persist/ascend-staging/` —— 升级 runbook
见 [`deployment/machine-recovery.md`](deployment/machine-recovery.md)。
