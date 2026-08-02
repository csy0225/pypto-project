# 实时状态（STATUS）

> **只放当前真相**：当前 phase、组件 pin、活跃 blocker、机器状态。
> 每日流水在 [`archive/milestones-2026-Q2.md`](archive/milestones-2026-Q2.md)；
> 整体规划在 [`planning/roadmap.md`](planning/roadmap.md)；接力面在
> [`planning/handoff.md`](planning/handoff.md)。
> **最后更新：2026-08-02。**

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

> **2026-08-02 当前真相：Attention/Vec 源码已合入，clean canonical candidate
> 已构建并推送，但正式发布被 N=128 raw gate 阻塞。**
>
> 源码：
>
> ```text
> pypto-lib stepfun/develop
>   76d96bdbeac280f12ecf626b1bbd722b9278719e
>
> pypto stepfun/develop
>   defa97c526fec7e8f032dbbfcc39c820add02bf7
> ```
>
> 当前实现不固定 24 核：logical task 数按 active rows、每行真实 `seq_len` 与
> architecture profile grain 推导，runtime 再映射到 AIC/AIV wave。`5--10 us`
> 仅是搜索起点。Full 的 SV 已融合 segment-local recurrence，只保留必要的
> `full_online_softmax_reduce/finalize`；Full/SWA out-proj cast 默认均融合。
> dense RMS direct BF16 reread 与 dense down-proj cast fusion 保留；AR+residual、
> residual+RMS stats、RMS+projection、gate/up+SiLU 等无稳定收益方案不合入。
>
> clean canonical candidate：
>
> ```text
> hub.i.basemind.com/stepcast/vllm-pypto:
>   stepfun-develop-20260802-attn-final-canonical
>
> manifest:
>   sha256:64c573bcf64497da6df0d3d28d7de85dfddde8e2a2a1b70e8bd5123edd51cb9d
>
> config/image ID:
>   sha256:c7f612a2562e932908d2a0d9ffadd1a1bd155c70bff0e82c24be32ef6b9f79ea
> ```
>
> config/worktree/credential/canonical-only/CANN runtime/optimization symbol/PTOAS ldd
> audit 与 smoke 全 PASS；immutable 验证只挂载 driver(ro)、checkpoint(ro)、
> output(rw)，无宿主源码挂载。64K、bs=1、512 blocks、warmup=3、20 iters：
> min `49.213`、mean `50.568`、p50 **`50.563 ms`**、p99/max `52.537 ms`。
>
> DFX 本轮正确 LOW-WAIT reference 为 rank2：
>
> ```text
> 0162:/mnt/persist/chensiyu/workspace/attn-opt/out/
>   image_attn_final_canonical_20260802/itl64k/build_output/
>   WholeDecodeStep3p5_20260802_162729/dfx_outputs/rank2/d0/
>     critical_path_report.md
>     merged_swimlane_20260802_162823.json
> ```
>
> rank2 makespan `38.924 ms`，TP AR critical-path compute `2.049 ms`。rank5 的
> `344.553 ms` TP AR compute 主要吸收 collective 自旋等待，因此不能称为
> LOW-WAIT reference。
>
> **发布 blocker**：同一 fresh oracle 三轮均
> `121/128=94.53125% < 95%`；所有 hidden finite，run2/run3 有瞬态 TP spread。
> 禁止借用 v2 的历史 `123/128` 或无限重跑。当前镜像只能标记为
> **clean canonical candidate / release blocked**。详见
> [`benchmark/2026-08-02-step3p5-attention-final.md`](benchmark/2026-08-02-step3p5-attention-final.md)
> 和 [`blockers.md`](blockers.md)。
>
> **2026-07-29 集成现状快照**：唯一 release Main 仍为
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
> **PERF-H1（历史性能镜像，2026-07-29）**：`hub.i.basemind.com/stepcast/vllm-pypto:stepfun-develop-20260729-perf-h1`
> （registry digest `sha256:b4e8c8a457a5…`；pin = pypto `1f704616` / pypto-lib `4513007d` /
> pto-isa `ecb6c303` / PTOAS `fc8c6cae` / simpler `e2efebcb` / ptoas-bin `v0.50`）。
> 在下方 C4 发布镜像上把 retained CommDomain window 清零从 per-step host H2D reset 改为
> device `aclrtMemset`（`_CTRL_MEMSET` + 8 卡并行 `broadcast_control_all`，runtime-only，不动 kernel/数值）。
> 0162 回归：smoke PASS、整网 CI `ok=true`（Main token `303,1207,19384,872,428,6127,4231,2636`
> 全 exact + MTP single/batch16 `6178,410,303` exact、`hidden_tp_spread=0`）、N=256 H1 vs C4
> **token 256/256 exact**（step127/128/255 含）全步 finite（raw-hidden run-to-run 抖动 = C4 push
> all-reduce 归约顺序，H1a-vs-H1b 复跑证实非 H1 回归）。**ITL p50（`--num-blocks 512`）：1024
> `50.9` / 8192 `52.0` / 32768 `58.0` / 65536 `64.1` ms —— 较下方 C4 同工作点降 23–27%**；
> PMU/scope 与 C4 逐项一致（cube_int8 `46.35%`、ring heap 峰值 `79.9%`、`dropped=0`）。
> ⚠ 两点：MTP oracle-wiring 修复 `0f3650c7`(test-only) 为 mount 验证、**未烤进本镜像**；
> live N=128 vanilla-raw 精度门未跑（token 与 C4 一致 → 等价 `240/256`）。benchmark 见
> [`benchmark/2026-07-29-perf-h1-image-itl-dfx.md`](benchmark/2026-07-29-perf-h1-image-itl-dfx.md)。
>
> 上一发布 / N=256 等价基线镜像（PERF-C4）
> `hub.i.basemind.com/stepcast/vllm-pypto:stepfun-develop-20260729-allreduce-push`
> （digest `sha256:7924925f4b2816c5645910b90fd2a9fa9469baace2f48f7e0ee41a587bd5d6ba`，
> config `sha256:5402e07ba0d19b315935bfda1e9f6b445d1a3fdc9067c634a2ce302fd7f2a3dd`；
> 代码 pin = pypto `6933b1aa` / pypto-lib `cfbdcce8` / pto-isa `ecb6c303` /
> PTOAS `fc8c6cae` / simpler `8459d60f` / ptoas-bin `v0.50`）已推 registry，
> 含 PERF-C4 TP all-reduce reduce-scatter + push all-gather。0162 immutable-image
> 回归：5 pin 与 spec 逐字一致、五仓工作树全 clean、credential / canonical-only /
> allreduce-push audit 全 PASS、smoke PASS；整网 CI `rc=0`（198.3 s）6 项 check 全
> true，token `303,1207,19384,872,428,6127,4231,2636` 全 exact；`hidden_tp_spread`
> 在 ci/main + rep1/rep2/rep3 共 **32 步全 `0.0`**（PERF-C4 准出指标）；ITL p50
> `65.942 ms`(ctx=1024) / `66.455 ms`(ctx=4096)。N=256 teacher-forced raw vanilla
> 95% gate **仍不宣称通过**（0726 镜像为 `240/256=93.75%`）；MTP oracle 在镜像外，
> 本轮 `--skip-mtp`，其缺失不作为 Main 失败。
>
> 该镜像在 ctx=65536 的实测：ITL p50 **83.349 ms**（active_batch=1，`--num-blocks 512`；
> 1024→65536 只涨 18.8%），active_batch 扫描 bs≤8 可跑（bs=8 p50 145 ms ≈ 4.6× 吞吐）、
> **bs=16 撞 device HBM**。DFX 给出两条可用结论：① `tp_all_reduce` 在 64k 只占 span
> **1.84%**、routed expert busy 仅 **0.99%** —— **C 系与 D/F 系对 64k 单 token 延迟都已低 ROI**，
> 与 0724 那份 ctx≈1 采集的结论（通信 74% wall）相反；② ITL 曲线给出硬约束：context ×64
> 只涨 13.3 ms，即**随 context 变化的部分 ≤16%，≈70 ms（84%）是与 context 无关的固定 floor**，
> 而该 floor 的构成当前 DFX 回答不了（插桩开销占了 span 的 4/5）。
> ⚠ 不要把 DFX 里 attention 的 97.9% 当延迟占比 —— 插桩 span 是真实单步的 5.21×，
> attention 占 56% task 数因而被系统性放大。下一步是**同镜像 ctx=1024 vs 65536 的 DFX A/B**
> 相减，把 floor 拆开再决定动谁。ring heap 峰值 79.9% 是唯一偏紧的 runtime 资源。
>
> 详见 [`deployment/docker/README.md`](deployment/docker/README.md)、
> [`benchmark/2026-07-29-release-image-64k-dfx-itl.md`](benchmark/2026-07-29-release-image-64k-dfx-itl.md)、
> [`benchmark/2026-07-28-tp-allreduce-push.md`](benchmark/2026-07-28-tp-allreduce-push.md)、
> [`postmortems/13-tp-allreduce-pull-notify-race.md`](postmortems/13-tp-allreduce-pull-notify-race.md)。
>
> vLLM 侧 tail-only + MTP proposer 挂点仍在 `1b3e538c`；真实在线请求接管、
> KV bridge、动态 batch 映射与同代 MTP absolute gate 仍属于 Phase 20/28 后续。
> **历史 push 状态（2026-07-29）**：GitHub `csy0225/pypto-lib:stepfun/develop` = `cfbdcce8`、
> `csy0225/simpler:stepfun/develop` = `8459d60f`、`csy0225/pypto:stepfun/develop` = `6933b1aa`
> （runtime gitlink → simpler `8459d60f`）—— 三者即镜像 pin。之后各多一个纯测试提交
> （`pypto ce7fcb64` / `pypto-lib cc850ee5`），不改产品代码；
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
| 2026-08-02 | **Attention/Vec 源码收口 + clean canonical candidate**：workload-derived logical tasks；Full SV+segment recurrence；仅保留 reduce/finalize；Full/SWA out-proj cast fusion；dense RMS direct BF16 reread；dense down-cast fusion。镜像 manifest `sha256:64c573bc…`、config `sha256:c7f612a2…`，audit/smoke/64K ITL/DFX PASS，p50 `50.563 ms`。⚠ fresh-oracle N=128 三轮均 `121/128`，正式发布 BLOCKED | `defa97c5` | `76d96bdb` | `ecb6c303` | `fc8c6cae` | `e2efebcb` | v0.50 |
| 2026-07-29 | **PERF-H1 自包含镜像 build + 回归**：`vllm-pypto:stepfun-develop-20260729-perf-h1`（registry digest `sha256:b4e8c8a457a5…`）。在 C4 发布镜像上前进 pypto→`1f704616`（gitlink→simpler `e2efebcb`，device-memset 清零）、pypto-lib→`4513007d`（`cfbdcce8` + ITL `--active-batch` + **MTP CI oracle-dir 可配置化**，去掉镜像外 username 硬路径）。0162 回归：smoke PASS；整网 CI `ok=true`（Main 8 步 token `303,1207,19384,872,428,6127,4231,2636` exact + MTP single/batch16 token `6178,410,303` exact，`hidden_tp_spread=0`）；**N=256 H1 vs C4 发布镜像 token 256/256 exact**（含 step127/128/255），全步 finite（raw-hidden run-to-run 抖动 ~34–44 = C4 push all-reduce 归约顺序，非 H1 回归，经 H1a-vs-H1b 复跑证实）。**ITL p50（`--num-blocks 512`）：1024 `50.9` / 8192 `52.0` / 32768 `58.0` / 65536 `64.1` ms —— 较 C4 同工作点降 23–27%**。MTP CI 挂掉真因=`_run_mtp` 用本次 Main hidden 当输入却比 0718 配对 golden（喂配对输入 pass_rate=1.0），wiring 修复 `0f3650c7`（test-only，mount 验证未 rebuild）。见 [`benchmark/2026-07-29-perf-h1-image-itl-dfx.md`](benchmark/2026-07-29-perf-h1-image-itl-dfx.md) | `1f704616` | `4513007d` | `ecb6c303` | `fc8c6cae` | `e2efebcb` | v0.50 |
| 2026-07-29 | PERF-H1 host/device 分账 + retained window 清零改 device `aclrtMemset`。首次把 ITL 拆成 host vs device：85 ms 里只有 ~55 ms 是 device，`_reset_persistent_domains` 独占 21.5 ms（每步 248 次串行阻塞 mailbox 往返 + 244.7 MiB H2D）。改走 backend 给 fresh window 用的同一条 device memset（新增 `_CTRL_MEMSET`，`broadcast_control_all` 8 卡并行）：清零 `21.50→2.21 ms`、ITL p50 `85.02→65.55 ms`（−22.9%）、每步 H2D 归零；同镜像 A/B `main_hidden_only_report.json` 除 `run_sec` 外逐字段相同，单测 8/8。sim 平台仍走原 host 路径。⚠ live N=128 精度门未跑。证伪：`persistent=False` 更慢 3.25×（ITL 276.2 ms）。另立 H2（起跑阶梯 2.914 ms，v4-flash 同形状）/ H3（DFX 假长条，曾使 `tp_all_reduce` 被误判 74.1% wall）。见 [`benchmark/2026-07-29-host-window-memset.md`](benchmark/2026-07-29-host-window-memset.md) | `1f704616` | `cfbdcce8` | `ecb6c303` | `fc8c6cae` | `e2efebcb` | v0.50 |
| 2026-07-29 | PERF-C4 TP all-reduce → reduce-scatter + **push** all-gather 发布。根因=pull-after-remote-notify 跨方向握手无序（postmortems/13），修正 design/performance/03 §5。pypto-lib `cfbdcce8` 已推 `stepfun/develop`；simpler span-aware child provenance 入库为 `8459d60f`，并由 pypto `6933b1aa` 的 runtime gitlink 固定（postmortems/14）。镜像 `vllm-pypto:stepfun-develop-20260729-allreduce-push`（digest `sha256:7924925f…`）已推 registry：audit/smoke/整网 CI PASS，`hidden_tp_spread` 32 步全 `0.0`，ITL p50 65.942/66.455 ms | `6933b1aa` | `cfbdcce8` | `ecb6c303` | `fc8c6cae` | `8459d60f` | v0.50 |
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
| ATTN-CANONICAL-PRECISION | clean canonical candidate N=128 raw gate `121/128`，伴随瞬态 TP spread | 🔴 Active | 2026-08-02 candidate 正式发布 | [`blockers.md`](blockers.md) |
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

2026-08-02 attention-final immutable 验证只使用 cards `0–7`；cards `8–15` 上 PID
`2045390–2045397` 全程未操作。audit/smoke/64K ITL/DFX 完成后 cards `0–7` 无残留
进程。candidate 精度门仍为 `121/128`，因此机器验证完成不等价于镜像正式发布。

**`gpu-a910x-0234`**：三剑合璧已齐（driver 25.5.2 / firmware 7.8.0.7.220 / CANN
9.0.0-beta.1）。2026-07-16 起 SSH `Permission denied`，不可达——既不能标 poisoned
也不能标已验证。恢复步骤见 [`deployment/machine-recovery.md`](deployment/machine-recovery.md)。
