# 实时状态（STATUS）

> **只放当前真相**：当前 phase、组件 pin、活跃 blocker、机器状态。
> 每日流水在 [`archive/milestones-2026-Q2.md`](archive/milestones-2026-Q2.md)；
> 整体规划在 [`planning/roadmap.md`](planning/roadmap.md)；接力面在
> [`planning/handoff.md`](planning/handoff.md)。
> **最后更新：2026-07-28。**

## 两条线（项目结构）

本项目现聚焦**两条清晰的线**：

1. **Track A — pypto 本身开发**（kernel / 整网 / 精度）：代码在
   `workspace/{pypto, pypto-lib, pto-isa, PTOAS, pypto/runtime(simpler)}`，均已在 git 跟踪。
2. **Track B — vllm + pypto 接线**（集成，命名 **`vllm-pypto`**，原 `pypto-lib-live` worktree）：
  - pypto 侧集成 Python（hidden-only 程序 / holder / sidecar / backend / monkey-patch / CI）在
     `workspace/vllm-pypto`（pypto-lib worktree，`stepfun/develop`）；
   - vLLM 侧集成在 fork `vllm/`（`PYPTO_STEP3P5_TAIL_ONLY` 主网 tail-only +
     `PyPtoMetadataOnlyStep3p5DecoderLayer` + MTP-proposer 挂点 + MTP3 `hf_overrides` boot fix，
     commit `1b3e538c`）+ `vllm-ascend/` fork。

> **2026-07-28 集成现状快照**：唯一 release Main 仍为
> `models.step3p5.decode_fwd:whole_decode_step3p5`；retired unroll、rollback
> selector、自定义 Main module/name 参数和 `models/step3p5_opt` 均保持删除。
> `stepfun/develop@563fe62a` 已完成 C1/C2/C3/D1/D2/G1：V4-Flash-style
> shared EP window/epoch、expert-lane dispatch/combine、deferred norm/INT8
> producer、routed-expert W8A8 与 runtime active batch/token 均已收口。
>
> BS1 错误不是输入输出透传，也不是 gate/top-k：根因是 local experts 被压入动态
> prefix slab，BS1/BS2 会改变 expert 的物理基址，破坏“增加相同 row1 不得改变 row0”
> 的 batch-extension invariance。`b404a3c9` 恢复固定 expert physical lane bases 后，
> BS1/2/16 单步均为 `6127→303`、TP spread `0`，BS1 row0 hidden 与 BS2/BS16
> bit-identical；BS1 persistent 4-step 为 `6127→303→1207→19384→872`。
>
> 最终自包含镜像
> `hub.i.basemind.com/stepcast/vllm-pypto:step3p5-b404a3c9-ci-final-20260728`
> （0162 本地 image ID `sha256:06261920cced91dafc585cd5e63622a88f798ad5ef6aeeba6480433049d1544f`，
> 产品 HEAD `b404a3c9`，CI 三文件以工作树补丁包含后续 `563fe62a` 的 cleanup/skip-MTP
> 逻辑，**尚未推送 registry**）smoke PASS；canonical Main 8-step token
> `303,1207,19384,872,428,6127,4231,2636` 全 exact，hidden 全 finite、8 个
> active rank rows nonzero、TP spread `0`。镜像内 N=256 teacher-forced 回归为
> hidden finite `256/256`、TP spread `0`、token exact `241/256=94.14%`；因此
> raw vanilla 95% gate **不宣称通过**。MTP oracle 位于镜像外，本轮使用
> `--skip-mtp`，其缺失不作为 C/D/G 或 Main 失败。
>
> vLLM 侧 tail-only + MTP proposer 挂点仍在 `1b3e538c`；真实在线请求接管、
> KV bridge、动态 batch 映射与同代 MTP absolute gate 仍属于 Phase 20/28 后续。
> **push 状态（2026-07-28）**：GitHub `csy0225/pypto-lib` 的
> `stepfun/develop` 与 `perf/step3p5-bc-20260726` 均已推至 `563fe62a`；
> GitHub `csy0225/pypto-project:main` 已同步本轮 C/D/G 状态文档；GitLab
> `sys/stepcast/vllm:csy/pypto-tail-mtp-integration` 保持 `1b3e538c`。

## 阶段跟踪

| 阶段 | 标题 | 状态 | 详情 |
|-----:|------|------|------|
| **1** | pypto kernel 原型 | ✅ 已完成 | [`archive/prototype-phase-01-19-summary.md`](archive/prototype-phase-01-19-summary.md) |
| **2** | vLLM Ascend 后端集成 | 🟡 进行中 | 见下 |

### Phase 2 sub-phases

| Sub-phase | 范围 | 状态 | 文档 |
|-----------|------|------|------|
| **20** | vLLM monkey-patch e2e（整模型 patch `Step3p5Model.forward`） | 🟡 sidecar canonical Main wiring 已完成；独立 live front 接管仍待验证 | [`design/vllm-pypto/02-detailed-design.md`](design/vllm-pypto/02-detailed-design.md) |
| **21** | 与 vLLM 原生精度对比 harness（L1/L2/L3） | ✅ dump-based 闭环；在线 gate 待 Phase 20 | [`archive/completed-phases/21-precision-validation.md`](archive/completed-phases/21-precision-validation.md) |
| **22/26** | Perf baseline + 优化；TP=8 多卡 | 📐 设计已落；gate 见 roadmap | [`archive/completed-phases/22-perf-baseline.md`](archive/completed-phases/22-perf-baseline.md) |
| **27** | N=1 单 `@pl.program` whole-net standalone | ✅ canonical P42 20/20 `argmax=303`（2026-07-18 single-submit 合入三仓 `stepfun/develop`） | [`planning/phases/27-n1-whole-net-fusion.md`](planning/phases/27-n1-whole-net-fusion.md) |
| **28** | N=1 whole-net → vLLM live single-handoff | 🟡 C/D/G + BS1 + 自包含镜像 Main 已收口；live-8001 接管、同代 MTP absolute gate、3-way HBM 仍待完成 | [`planning/phases/28-n1-live-integration.md`](planning/phases/28-n1-live-integration.md) |

> 交付分级 / 到 v1.0 的规划见 [`planning/roadmap.md`](planning/roadmap.md)。
> **口径提醒**：dump-based 精度闭环 ≠ 真实 vLLM 请求已走 PyPTO NPU runner；
> production backend（Phase 20）仍未完成。
>
> **2026-07-23 主网 multi-decode 精度验证（device 0162, `stepfun/develop a632c42e`+CI `e66bda25`）**：
> 用 **live vanilla vLLM W8A8 oracle** 逐 token teacher-forced 对比，seed=6127 / N=128 →
> **ALIGNED=124/128=96.9%（≥95% L3 PASS）**；4 个 miss 全是 vanilla 自身 near/dead-tie
> （pypto 的选择 = vanilla fresh 查询 #1）。**即 pypto 整网 decode 与 vanilla 逐 token 对齐、
> 精度正常**。CI: `tests/step3p5/ci/LIVE_PRECISION_AB.md`。
> ⚠ **历史口径更正**：此前 session 里"multi-decode step-3 发散 / near-tie 未解决"的结论
> **作废**——根因是 harness 硬编码 `DEFAULT_ORACLE_TOKENS[2]=19384` 是过时/串位常量
> （one-shot `encode(text)` 边界串位），对相同 no-BOS 上下文 vanilla 自己也出 6127，pypto 无误。

## 组件 Pin Snapshot（最新）

| 日期 | 事件 | pypto | pypto-lib | pto-isa | PTOAS(src) | simpler | ptoas-bin |
|------|------|-------|-----------|---------|-----------|---------|-----------|
| 2026-07-28 | PERF-C4 TP all-reduce → reduce-scatter + **push** all-gather（`perf/step3p5-bc-20260726@cfbdcce8`，未推 develop）；根因=pull-after-remote-notify 跨方向握手无序（postmortems/13），修正 design/performance/03 §5；回归 whole-network CI PASS + 40 个 decode step token 全对且 `hidden_tp_spread` 全 0.0；ITL p50 −3.6%/−3.9% | `ca21ab5f` | `cfbdcce8` | `ecb6c303` | `fc8c6cae` | `216e7632` | v0.50 |
| 2026-07-28 | C/D/G + BS1 收口；`stepfun/develop@563fe62a`；固定 expert physical lanes；最终自包含镜像 smoke/Main 8-step PASS，N=256 hidden finite/TP spread=0、token exact 241/256 | `ca21ab5f` | `563fe62a` | `ecb6c303` | `fc8c6cae` | `216e7632` | v0.50 |
| 2026-07-24 | PERF-A1 逐层 DFX 接线（holder N1_DFX→swim/pmu + harness `--dfx`）；在镜像 20260724(cards 8-15) 采集 → routed-expert 占 90.7% / PMU cube_int8 88.6%（benchmark/2026-07-24-perlayer-dfx） | `ca21ab5f` | `bc5eecb1`(+DFX over `fd26b1be`) | `ecb6c303` | `fc8c6cae` | `216e7632` | v0.50 |
| 2026-07-24 | 合并 origin/main 到 stepfun/develop（全 FF，保留 fork ITL harness `7cb2a6b3`）+ IPC 权重 interior 指针 provenance 修复（解 `submit_next_level child_memory` 卡点）；镜像 `vllm-pypto:stepfun-develop-20260724`(ptoas v0.50) 冒烟 PASS + 整网 8 步 `6127→303→1207→6127` 与 live vanilla 逐 token 一致 | `ca21ab5f` | `fd26b1be` | `ecb6c303` | `fc8c6cae` | `216e7632` | v0.50 |
| 2026-07-26 | B2 loop-form Main canonical-only 收口；`stepfun/develop@53eb7212`；删除兼容 package/alias 后与清理前镜像 N=256 token/hidden `256/256` exact | `ca21ab5f` | `53eb7212` | `ecb6c303` | `fc8c6cae` | `216e7632` | v0.50 |
| 2026-07-23 | decode-ITL profiling harness(hidden-only via holder;64k ≈654ms/step,含 host glue;权威基线见 benchmark/ device-KV 590ms raw rt.run;均计算受限、近平坦)| `8af501fc` | `7cb2a6b3`(+ITL mode over `4c48215b`) | `ecb6c303` | `72ada0a1` | `36957c6b` | v0.45 |
| 2026-07-23 | simpler develop 回退到可编译 36957c6b（c7fdc574 Phase-24 import_ipc 半成品编译不过, 存 tag backup/stepfun-develop-c7fdc574-20260723）+ pypto develop gitlink 同步 | `8af501fc`(9ec303f6+gitlink→36957c6b) | `4c48215b` | `ecb6c303` | `72ada0a1` | `36957c6b`(develop 回退; 0162 验证过的 .so 就是它) | v0.45 |
| 2026-07-23 | 五仓 stepfun/develop 对齐验证过的 N=1 pin（pto-isa/PTOAS FF-push 到 fork stepfun/develop）+ 可复现 Docker 镜像 | `9ec303f6` | `4c48215b` | `ecb6c303`(FF `e25732f0`→,+111) | `72ada0a1`(FF `da011a3d`→,+307) | `c7fdc574` | v0.45 |
| 2026-07-18 | N=1 single-submit 合入三仓 `stepfun/develop` + 干净回归 20/20 | `9ec303f6` | `e1513d22` | `ecb6c303` | `72ada0a1` | `c7fdc574` | v0.45(见 stable SSOT) |
| 2026-07-17 | N=1 stable env freeze（SSOT `develop/N1/N1-STABLE-ENV-0162-20260717.md`） | `n1fusion-base:e277de9f` | `feat/whole-net-n1-fusion:0e7a0fdd` | `ecb6c303` | `72ada0a1` | `n1fusion-base:36957c6b` | v0.45 |
| 2026-06-22 | Phase 2 设计落地；建项目跟踪仓 | `stepfun/develop:b00c8b23` | `stepfun/develop:b918e60` | `e25732f0` | `da011a3d` | `a6e06406` | v0.45 |

> 完整 pin 历史见 [`archive/milestones-2026-Q2.md`](archive/milestones-2026-Q2.md)。

## 当前 Blocker / Deferred（摘要，详见 [`blockers.md`](blockers.md)）

| # | Blocker | 严重度 | gate 什么 | 详情 |
|--:|---------|--------|-----------|------|
| N1-S-0234 | 0234 同步 pypto-lib 后 whole-net stall（完整对象未确认） | 🔴 Active / 未独立复核 | 取得 SSH 后核对三仓/runtime/环境重跑 canonical | [`blockers.md`](blockers.md) |
| N1-L | Phase 28 live：per-layer KV + 3-way HBM + live token-exact A/B | 🔴 Active | live single-handoff | [`planning/phases/28-n1-live-integration.md`](planning/phases/28-n1-live-integration.md) |
| 1 | Phase 20 production backend 未接入 | 🟡 功能 | 真实 vLLM 请求走 PyPTO runner | [`design/vllm-pypto/`](design/vllm-pypto/) |
| 2 | Prefill MoE L1 overflow（TASK-29） | 🟡 功能/性能 | 真实 PyPTO NPU prefill kernel | [`blockers.md`](blockers.md) |
| 3 | head_gate 语义（历史 ×1 旁路已由 on-device gate 取代） | 🟡 精度 | 在线 backend L1 parity | [`postmortems/09-attention-multiposition-corruption.md`](postmortems/09-attention-multiposition-corruption.md) |
| 5 | MTP 集成进 decode | 🟢 Deferred | speculative 吞吐 | [`blockers.md`](blockers.md) |

> 已解 blocker 转为专项复盘：[`postmortems/`](postmortems/)（如 507899/507018、
> co-tenancy、tmov、gate_topk、gap-5、scheduler-timeout 等）。

## 机器状态

**`gpu-a910x-0162`（Phase 16 验证机，当前主力）**：driver 25.5.2 ✅ / firmware
7.8.0.7.220 ✅ / CANN 9.0.0-beta.1 ✅；simpler L3 allreduce、前端 smoke、dense/SWA/MoE
ST、N=1 canonical P42 20/20 均 PASS。2026-07-28 cards 8-15 完成最终自包含镜像
Main 8-step PASS 与 N=256 teacher-forced 回归（hidden finite `256/256`、TP spread `0`、
token exact `241/256`），作业退出后无残留主进程。唯一 stable 环境记录见
[`develop/N1/N1-STABLE-ENV-0162-20260717.md`](develop/N1/N1-STABLE-ENV-0162-20260717.md)。

**`gpu-a910x-0234`**：三剑合璧已齐（driver 25.5.2 / firmware 7.8.0.7.220 / CANN
9.0.0-beta.1）。2026-07-16 起 SSH `Permission denied`，不可达——既不能标 poisoned
也不能标已验证。恢复步骤见 [`deployment/machine-recovery.md`](deployment/machine-recovery.md)。
