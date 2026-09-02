# 落地台账（LANDED）

> **这里只放「确定落地」的事实**：已发布镜像、已合入的源码 tip、以及它们各自通过了哪个门。
> 追加式，一次落地一行，**不写过程**。
>
> - 每次开发过程 → [`../archive/milestones-2026-Q2.md`](../archive/milestones-2026-Q2.md)
> - 长报告与原始数字 → [`../benchmark/`](../benchmark/)、0162 campaign 目录
> - 当前真相速览 → [`../STATUS.md`](../STATUS.md)
> - 未决 → [`../blockers.md`](../blockers.md)

## 「落地」的两级边界（本文件的核心区分）

| 级别 | 含义 | 可以拿它说什么 | **不能**拿它说什么 |
|---|---|---|---|
| **IMG** immutable-image released | 有 manifest digest，在 0162 上 **digest-only、无 source/runtime overlay** 验证过 | 这是生产可部署的形态 | 除非注明"完整"，否则不代表全矩阵准出 |
| **SRC** source-overlay GO | 代码已合入 `stepfun/develop`，在**固定 immutable 镜像**上通过只读 `/candidate` overlay 验证 | 实现 + 该门的结论成立 | ❌ 不代表镜像已包含它；❌ 不能当镜像级准出数据 |

**当前 canonical SRC pin 集（2026-09-02）**：
`pypto 655c7bda` / `pypto-lib a745ab659` / `pto-isa cd4a3d3f` /
`PTOAS(src) 307d0484` / `simpler 85a82c45` / `ptoas-bin v0.57`。
此前 r12 的 `14de90fd/e6c7d8ec` 仍作为历史 IMG pin 保留。
2026-08-24 之前表格中的“canonical/偏离”均按**当时**的旧 pin 集解释，不反向改写历史。

---

## 当前生产真相（2026-09-02）

- **最新 IMG** = whole-step host/graph/submit r12（manifest `ba42fd19…eb805d`）；
  final contract `1844/1844 PASS`，SHA256 `511a5459…87f3a`。
- **性能证据**是 r11 digest 上仅 overlay 两个 pypto runtime 文件的 A/B/A：
  ITL p50 `21.6805 → 21.115 ms`（`−2.608%`），graph build `−44.429%`，
  graph→first runner `−47.936%`；不是 r12 immutable 性能重采。
- **immutable r12 门**：Main H4 all/none 均 `126/128`，MTP BS1/BS16 token
  `[6178,410,303]` 且 hidden pass rate `1.0`，dep-only DFX hidden/token exact。
- **当前 SRC 已前进**：pypto `655c7bda`、pypto-lib `a745ab659` 已由远端 exact 复核；
  matched candidate H4=`all` A/B/A `21.617/20.516/21.257 ms`，reset 修复
  gain `0.921 ms / 4.296%`，extended correctness admission v2 PASS。
- r15 local tag `stepfun-upgrade-20260902-a745-k8-r15` manifest `19f51d37…a7f`
  与已测 r14b 字节相同，local audit PASS；registry 未 push，route v2/v1 publication 未闭合，
  因此不新增表 A IMG 行，也不改写 r12 release truth。
- `20.516 ms` 未刷新历史 a745 source-overlay `20.172 ms`；旧 r13
  `655+e6c7d8ec=21.562 ms` 也不是当前匹配栈。
- r11（manifest `401ead7d…a67b12`）保留为回退；其 r10/r11/r10 immutable A/B/A
  仅 `−0.0065 ms / −0.0299%`，结论是性能中性，不是性能收益。
- **H4 deployment contract 已落地**：三个 launcher 默认注入 `all`，`none` 可回退；
  r12 exact launcher 64K/1000 p50 `20.973 ms`。当前 `20.516 ms` 为另一份
  100-iter non-privileged matched ABA；跨历史 `−0.457 ms` 只作方向性检查，正式收益
  仍是同代 `0.921 ms / 4.296%`。r12/r15 image Config 均未 bake H4。

---

## 表 A —— IMG（immutable image released）

| 日期 | 镜像 / 落地了什么 | manifest digest | pypto | pypto-lib | 偏离 canonical pin | 准出范围 | 证据 |
|---|---|---|---|---|---|---|---|
| 2026-08-27 | `…:stepfun-upgrade-20260826-r12` —— prepared TaskArgs cache + whole-step graph-build/submit 优化 | `ba42fd19b3…eb805d`（config `b36f0cec3a…33f08f`） | `14de90fd` | `e6c7d8ec` | 其余 pin 与 r11 相同 | **最终 release-admitted，1844/1844 PASS**：publication/fresh digest、immutable smoke、Main H4 all/none `126/128`、MTP BS1/16、dep-only DFX、显式 0–7 卡且无 overlay；性能来自 r11 source-overlay A/B/A，不冒充 r12 immutable A/B/A；contract SHA `511a5459…87f3a` | [`2026-08-27-whole-step-host-graph-submit-r12-release.md`](../benchmark/2026-08-27-whole-step-host-graph-submit-r12-release.md) |
| 2026-08-26 | `…:stepfun-upgrade-20260826-r11` —— replicated-input local-owner MoE | `401ead7da4…a67b12`（config `35c42510a6…06876`） | `519b588a` | `e6c7d8ec` | 其余 pin 与 r10 相同 | **release-admitted，20/20 PASS**：H4 all/none 均 `126/128` 且 hidden/token parity；64K/1000 p50 `21.477 ms`；r10/r11/r10 immutable A/B/A `21.751/21.745/21.752 ms`，`−0.0065 ms`，只判性能中性；contract SHA `570bb04e…740af7` | [`2026-08-26-local-owner-moe-r11-release.md`](../benchmark/2026-08-26-local-owner-moe-r11-release.md) |
| 2026-08-25 | `…:stepfun-upgrade-20260825-r10` —— packed-NZ routed MoE + fused GMM1/SwiGLU/requant + adaptive down grid | `8510f30e1f…e9907b`（config `38ebba41d6…657b5f`） | `519b588a` | `fe641929` | 其余 pin 与 r9 相同 | **最终 release-admitted，71/71 PASS**：Main+MTP、N=128 `127/128`、H4 parity、ITL p50 `21.742 ms`、A/B/A `−3.241%`、六档 correctness `6/6`、outer DFX 8/8、develop sync；contract SHA `bcdd0b11…8dafca` | [`2026-08-25-moe-fusion-image-release.md`](../benchmark/2026-08-25-moe-fusion-image-release.md) |
| 2026-08-24 | `…:stepfun-upgrade-20260824-r9` —— 五仓全栈升级 + H4 resident constants | `b637f00c66…a71690f6`（config `f6c8f72e…e421dae`） | `519b588a` | `bf3ff440` | pto-isa `cd4a3d3f` / PTOAS `307d0484` / simpler `85a82c45` / ptoas-bin `v0.57` | **升级任务准出**：registry/fresh pull、precision `127/128`、Main+MTP liveness、L3/L4 exact、8/8 chip swimlane；64K p50 `22.253 ms` **仅在 `PYPTO_H4_RESIDENT=all`**，默认 `none=27.812 ms` | [`2026-08-24-upgrade-r9-release.md`](../benchmark/2026-08-24-upgrade-r9-release.md) |
| 2026-08-11 | `…:stepfun-develop-20260811-k8-selective` —— K8 persistent-window 选择性清零 | `076af8a167…c47f3` | `1c048a74` | `cb96747e` | — | **部分**：64K bs1 p50 `32.14 ms`（−5.02% vs pre-K8）、hidden byte-exact、N=128 `123/128`。batch16 / MTP / 六档 64K 未重跑 | [`2026-08-11-k8-selective-window-zeroing-image.md`](../benchmark/2026-08-11-k8-selective-window-zeroing-image.md) |
| 2026-08-06 | `…-20260806-attn-taskmajor-canonical` —— task-major Attention | `3eb694e045…25479` | `8e92b468` | `c9af5790` | — | **pre-fix evidence**（不含 SWA mask 修复 `63814d4a`）：64K p50 `39.612 ms` | [`2026-08-06-attention-taskmajor-canonical.md`](../benchmark/2026-08-06-attention-taskmajor-canonical.md) |
| 2026-08-06 | L0–L4 MoE formal 镜像（六档 focused A/B） | `cab8966816…9d88c` | `8e92b468` | `c9af5790` | — | **pre-fix evidence**：BS `1/2/4/7/8/16` × 3 轮 36/36，L3/L4 hash exact | seal `875804db…c531` |
| 2026-08-03 | `…-20260803-attn-final-wave5` —— source partial 改 self-target TPUT | `4acc77cdce…67b32` | `defa97c5` | `7099476b` | — | ✅ **历史完整 production-matrix release-qualified 基线**。当前直接回退是 r11；Wave5 保留作旧矩阵对账。64K p50 `49.796 ms`、N=128 三轮 `123/128` spread=0 | STATUS §2 |
| 2026-08-03 | Wave4 historical candidate（第三 completion wave） | `8125c678…` | `defa97c5` | `d7e1381b` | — | 已被 Wave5 取代（N=128 Run1 step2 spread=`2.0` 未过稳定性门） | STATUS §2 |
| 2026-08-02 | Attention/Vec 收口 historical candidate | `64c573bc…` | `defa97c5` | `76d96bdb` | — | 已被 Wave3/4 lifetime 修复取代（N=128 三轮 `121/128`） | STATUS §2 |
| 2026-07-29 | `…-20260729-perf-h1` —— retained-window 清零改 device memset | `b4e8c8a457a5…` | `1f704616` | `4513007d` | — | ITL p50 `50.9/52.0/58.0/64.1 ms`（ctx 1K/8K/32K/64K，较 C4 降 23–27%）；N=256 token `256/256` exact | [`2026-07-29-perf-h1-image-itl-dfx.md`](../benchmark/2026-07-29-perf-h1-image-itl-dfx.md) |
| 2026-07-29 | `…-20260729-allreduce-push` —— AR 改 reduce-scatter + **push** all-gather | `7924925f…` | `6933b1aa` | `cfbdcce8` | simpler `8459d60f` | audit/smoke/整网 CI PASS，`hidden_tp_spread` 32 步全 `0.0`，ITL p50 `65.9 ms` | [`13`](../postmortems/13-tp-allreduce-pull-notify-race.md)、[`14`](../postmortems/14-image-dirty-worktree-unreproducible-pins.md) |
| 2026-07-28 | C/D/G + BS1 收口自包含镜像（固定 expert physical lanes） | 未记录 | `ca21ab5f` | `563fe62a` | simpler `216e7632` | smoke / Main 8-step PASS；N=256 hidden finite、TP spread=0、token exact `241/256` | milestones 2026-07-28 |
| 2026-07-26 | canonical-only Step3p5 release（删兼容 package/alias） | 未记录 | `ca21ab5f` | `53eb7212` | simpler `216e7632` | 与清理前镜像 N=256 token/hidden `256/256` exact | milestones 2026-07-26 |
| 2026-07-24 | `vllm-pypto:stepfun-develop-20260724` —— IPC 权重 interior 指针 provenance 修复 | 未记录 | `ca21ab5f` | `fd26b1be` | simpler `216e7632` | 冒烟 PASS + 整网 8 步与 live vanilla 逐 token 一致 | milestones 2026-07-24 |

## 表 B —— SRC（代码已合入 `stepfun/develop`；可能已被后续 IMG 收录）

| 日期 | 落地了什么 | pypto | pypto-lib | 门结论 | 证据 |
|---|---|---|---|---|---|
| 2026-09-01 | local-owner persistent reset ABI 回退修复 + routed GMM latch participant candidate | `655c7bda` | `a745ab659` | 两 commit 已进入 `stepfun/develop`；matched reset A/B/A `21.617/20.516/21.257 ms`、gain `0.921 ms / 4.296%`，5-case extended gate PASS。r15 local image audit PASS，但 registry/route publication 未闭合；`20.516` 不是历史新低 | [`2026-09-01-k8-local-owner-reset-regression.md`](../benchmark/2026-09-01-k8-local-owner-reset-regression.md)、[`2026-09-02 对账`](../benchmark/2026-09-02-k8-historical-performance-reconciliation.md) |
| 2026-08-27 | prepared TaskArgs descriptor/signature cache 与 whole-step host/graph/submit 优化合入远端 `stepfun/develop` | `14de90fd` | `e6c7d8ec` | r11 digest 上 source-overlay A/B/A：ITL `21.6805→21.115 ms`（`−2.608%`）、graph build `−44.429%`、graph→first runner `−47.936%`，三臂 hidden/token exact；正式合同仍为 `serial-eight-rank`、`group_size=1`、无 group submit；随后 baked 入 r12 并过 immutable release gate | [`2026-08-27-whole-step-host-graph-submit-r12-release.md`](../benchmark/2026-08-27-whole-step-host-graph-submit-r12-release.md) |
| 2026-08-25 | packed-NZ MoE fusion 用 exact lease fast-forward 到远端 `stepfun/develop` | `519b588a` | `fe641929` | `bf3ff440 → fe641929` fast-forward，远端 SHA 复核；对应 r10 final contract `71/71 PASS` | [`2026-08-25-moe-fusion-image-release.md`](../benchmark/2026-08-25-moe-fusion-image-release.md) |
| 2026-08-24 | 五仓升级目标同步到远端 `stepfun/develop` | `519b588a` | `bf3ff440` | pto-isa `cd4a3d3f` / PTOAS `307d0484` / simpler `85a82c45`；五仓 `force-with-lease` 后远端 SHA 逐项复核。simpler 旧 tip 备份为 `backup/stepfun-develop-pre-upgrade-20260824-e2efebcb` | [`2026-08-24-upgrade-r9-release.md`](../benchmark/2026-08-24-upgrade-r9-release.md) §5 |
| 2026-08-15~19 | **MoE `moe-routed-packed-fusion` R5 —— 升级前的历史 source-overlay MoE 基线** | 未移动 | `decode_fwd.py` sha `67b73589…`（⚠ commit 未记录，见「台账缺口」） | ctx-64K BS1 p50 `27.757 ms`@ITERS=100 / `26.329 ms`@ITERS=1000；`hidden_sha256=567b206b…`、tail token `14371`；`ITERS=1000` 一轮长跑 0 liveness 事件 | [`16`](../postmortems/16-dispatch-fusion-orch-decouple.md)、`0162:…/moe-routed-packed-fusion-20260815/` |
| 2026-08-12 | TP all-reduce **small-message selector**（单行 8 KiB 走静态两波 one-shot mesh，其余走三波 fallback；ownership 与 transfer chunk 解耦） | `1c048a74` | **`69ad31e4`** | unit `365 passed, 7 skipped`；Main/MTP compile + 8 卡 rows `1/3/16` PASS；Whole A/B/A `31.065/29.912/30.999 ms` = **`−1.120 ms / −3.609%`**，三臂 precision PASS。`ABA_RESULT.json` sha `383caa23…` | [`03-tp-allreduce-algorithm-comparison.md`](../design/performance/03-tp-allreduce-algorithm-comparison.md) |
| 2026-08-12 | RMS→QKV critical prestage（I7：RMS producer 开 early resolve + 非关键 head-gate 隔离出 speculative fanout） | `1c048a74` | `e5e26f9f` | QKV Worker gap p50 `+4.77 → −1.78 µs`；raw-kernel residual p50/max `5.00/5.48 → 2.64/3.16 µs`；A/B/A `WITHIN_BASELINE_BRACKET` | [`2026-08-12-step3p5-rms-qkv-dispatch-gap.md`](../benchmark/2026-08-12-step3p5-rms-qkv-dispatch-gap.md) |
| 2026-08-10 | P1a gate 解耦（`gate_expert_fanout` 只存 raw FP32 logits，尾巴搬进 `gate_topk`） | 未移动 | `d13b2ca6` | bs1 `36.494 → 33.849 ms`（+7.25%）、bs8 `97.528 → 91.722 ms`（+5.95%）；两档三臂 hidden byte-exact | [`2026-08-10-step3p5-p1a-gate-decouple.md`](../benchmark/2026-08-10-step3p5-p1a-gate-decouple.md) |
| 2026-08-10 | MoE BS1 N256 | 未移动 | `a31977fb` | bs1 p50 `35.778 → 34.271 ms`（4.21%）；三臂 byte-exact，精度 replay `123/128`、TP spread=0 | [`2026-08-10-step3p5-moe-n256-final.md`](../benchmark/2026-08-10-step3p5-moe-n256-final.md) |
| 2026-08-06 | SWA tail-window mask 改显式 typed INT32 区间（`63814d4a`） | 未移动 | `63814d4a` | source-overlay N=128 `127/128 = 99.22%`、`hidden_tp_spread_max=0.0` ⇒ 过 `>=95%` **source-level** 门；镜像级未重跑 | summary sha `7f91dcdb…` |

> ⚠ **表 B 的性能数字一律是固定 IMG 上的 source-overlay 结果**，不得当作镜像级准出，也不得
> 与表 A 的 IMG 数字横比（编译期 `num_blocks` 容量不同：bs1 用 `512`、bs8 用 `4096`；
> bs16 ctx-64K 物理不可行 —— 16 GiB 单次 `rtMalloc` → `207001`）。

---

## 已否决，不要重试（NO-GO 台账）

| 方向 | 判定 | 为什么 | 出处 |
|---|---|---|---|
| 继续优化 `bind.args` | **NO-GO（当前 P0）** | r11 source-overlay A/B/A 为 `0.054449→0.054669 ms`，仅 `+0.000220 ms`、占候选 ITL `0.259%`，判定 `no_clear_change`；不再挤占 host/graph/submit 主路径预算 | [`2026-08-27-whole-step-host-graph-submit-r12-release.md`](../benchmark/2026-08-27-whole-step-host-graph-submit-r12-release.md) |
| routed GMM fixed-22-participant latch | **NO-GO** | H4 A/B/A 收益 `0.588 ms`，低于 required floor `0.616 ms`；不要恢复固定 22-worker 双 latch | [`2026-08-30-routed-gmm-active-worker-dual-latch.md`](../benchmark/2026-08-30-routed-gmm-active-worker-dual-latch.md) |
| MoE dispatch 域小算子融合（R6–R9 八个变体） | **NO-GO** | 概率性 liveness hang，且匹配曝光后也不快；orchestrator 从不在关键路径 ⇒ ROI 上界 0 | [`16`](../postmortems/16-dispatch-fusion-orch-decouple.md) |
| 删掉 `combine_scatter` 的 orchestration 阻塞读（静态 grid） | **NO-GO** | 那个读是承重的 run-ahead 流控阀，删掉即 `HEAP_RING_DEADLOCK` | [`16`](../postmortems/16-dispatch-fusion-orch-decouple.md) §3 |
| `a791071` TP all-reduce Ring 实验 | **结论撤回，不得合入** | 是 standalone-builder A/A，未命中 production 或 two-layer collective | STATUS §1 |
| K6b dynamic-valid-shape 产品路径 | 不恢复 | 被 `69ad31e4` selector 取代 | STATUS §5.3 |
| 多 `@pl.program` / per-layer 拆分 / Option-C | **NO-GO（架构裁定）** | N≥6 co-prepare 墙；生产只允许单 `@pl.program` | [`08`](../postmortems/08-multiprogram-coprepare-deadlock.md) |
| native W8A8 回退 BF16-dequant | 禁止 | 「明知临时的地基」，`CLAUDE.md` 铁律 6 | [`12`](../postmortems/12-integration-churn-meta.md) 根因 3 |
| 历史 standalone `gate_up+act` / `act+h_quant` DSL 融合、`tp_all_reduce` 降 ring step | NO-GO | 当时分别为 ROI 改判 / grid 维度冲突 / 前提未证实；不等于 2026-08-25 packed-NZ external 三合一 bundle | [`2026-08-10-step3p5-p1a-gate-decouple.md`](../benchmark/2026-08-10-step3p5-p1a-gate-decouple.md) |
| AR+residual、residual+RMS、RMS+projection 融合 | 已证伪或无稳定收益 | 不合入 | STATUS §3 |
| `down24` | NO-GO | 见 N256 报告 | [`2026-08-10-step3p5-moe-n256-final.md`](../benchmark/2026-08-10-step3p5-moe-n256-final.md) |
| 2026-08-05 R1 / R2 镜像 | R1 已撤销、R2 从未发布 | pypto-lib `91c7f46e` 已被多次 supersede，不得恢复 | [`2026-08-05-attention-canonical-r1-r2.md`](../benchmark/2026-08-05-attention-canonical-r1-r2.md) |

---

## 台账缺口（已知未记全，别当成"没有"）

1. **R5 的 git commit sha 没被记录**，本仓只有 `decode_fwd.py` 文件 sha `67b73589…`。
   下一次碰 MoE 生产路径时，从 0162 `…/moe-routed-packed-fusion-20260815/` 反查并补进表 B。
2. **2026-07-24 / 07-26 / 07-28 三个镜像的 manifest digest 未记录**，只有 tag。
   按 `CLAUDE.md` 判定顺序第 4 条，无 digest 的镜像不能作为发布依据。
3. r10 已按本次 packed-NZ MoE fusion 合同 `71/71` release-admitted；Wave5 的历史
   **完整 production matrix** 与本次合同范围不同，两者不得互相借证据。本次准出也不
   等价于 Phase 28 live serving 已完成。
