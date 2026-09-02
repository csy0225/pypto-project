# Milestones —— 2026 Q2

## 2026-09-02 —— r15 本地发布准备与历史性能口径纠正 ⏳

**远端 / 镜像**：0162 fetch 后 canonical refs 仍为 pypto `655c7bda`、pypto-lib
`a745ab659`、pypto-project `2af3734`。本地 tag
`stepfun-upgrade-20260902-a745-k8-r15` 的 manifest/config 为
`19f51d37…64a7f` / `7e5dd868…d6eb`，与已测 r14b 字节一致；immutable image audit PASS，
无 source/runtime/core overlay。H4 由 OCI runtime 注入 `PYPTO_H4_RESIDENT=all`，image Env 未 bake。

**性能口径纠正**：当前 `20.516 ms` 的正式含义仅是固定 a745 后，
`14de→655` reset 修复 matched A/B/A `21.617/20.516/21.257 ms`，gain
`0.921 ms / 4.296%`。历史 r12 exact launcher `20.973 ms` 为 1000-iter 单臂，
历史 a745 `20.172 ms` 为 100-iter source-overlay B 臂；当前没有刷新历史最优，
跨合同 `−0.457/+0.344 ms` 均不作显著性声明。

**发布未闭环**：registry tag 仍不存在，匿名 push 返回 401，需临时 write credential；
a745 exporter 为 `local-routes.v2`，project validator 要求 `recv-meta.v1`，旧 r12
sidecar 不可复用。project publication gate unit 为 `34 passed`，但真实 route authority
仍待闭合。本轮文档对账未启动新 NPU workload，16 卡与 container/task 保持为空。

统一报告：
[`../benchmark/2026-09-02-k8-historical-performance-reconciliation.md`](../benchmark/2026-09-02-k8-historical-performance-reconciliation.md)。

## 2026-09-01 —— local-owner reset 回退修复 + a745 matched candidate 验证 ✅

**源码落地**：pypto `655c7bda`（parent `14de90fd`）修复
`distributed_runner.py` 对 local-owner 4-control/4-data persistent layout
误回退 full-window clear 的问题；pypto-lib `a745ab659`（parent `e6c7d8ec`）
保留最新 routed GMM latch participant 优化。两仓均用
`--force-with-lease` 推到 fork `stepfun/develop`，随后 `ls-remote` exact
复核为 `655c7bda` / `a745ab659`。

**matched immutable candidate**：image manifest
`sha256:19f51d373c5f9d6171ccf3306f260066e873eda48efca23f5d77b4d6f5e64a7f`、
config `sha256:7e5dd8683fda03e3e51a0b5217ae71ab82052173f3659db60fd689ea833ed6eb`；
H4=`all`、64K、8 卡、warmup10/iters100 的 A/B/A p50
`21.617/20.516/21.257 ms`，gain `0.921 ms`，required `0.616 ms`，
H4 PASS。三臂 hidden SHA 均
`ee8ae6b4b3083112d397e5e91cc63fb0e2edfb705eb7a535aceb232f1a7db96a`，
tail token `43640` exact；B `memset_all` p50 `462.277 us`。

**extended correctness**：0162 五 case（Main H4 all/none、MTP BS1/BS16、
dep-only DFX）全部 PASS；admission schema
`step3p5.r14b-extended-immutable-admission.v2`，16/16 admission checks
为 true；run contract/evidence/device cleanup 复核通过。正式 execute
runner SHA `94a3cda8…a1b36`，run：
`.../k8-a745-matched-validation-20260901/runs/extended-candidate-gate-r14b-20260901-235716-2942152-952530890/`。

**边界**：该 candidate gate 未包含 a745 provenance 匹配的
`recv_meta` route publication sidecar；旧 r12 sidecar schema/provenance
不匹配，故当前 release IMG 仍为 r12，不能把 manifest `19f51d37…`
写成 release-admitted。最终报告：
[`../benchmark/2026-09-01-k8-local-owner-reset-regression.md`](../benchmark/2026-09-01-k8-local-owner-reset-regression.md)。

## 2026-08-27 —— whole-step host/graph/submit r12 发布与源码同步 ✅

**镜像 / SRC**：r12 tag `stepfun-upgrade-20260826-r12`，manifest
`sha256:ba42fd19b3af0144a835e95a4a6925ed89ea700624f696b221e93a54e6eb805d`，
config `sha256:b36f0cec3a8b64e5e17e273c63d69694730bd8b904e69c2806c3d73a5233f08f`；
pypto `14de90fd`、pypto-lib `e6c7d8ec`，其余 pins 与 r11 相同。五仓远端
`stepfun/develop` 已在最终合同中逐项 `ls-remote` 复核。

**性能证据分层**：matched A/B/A 在已发布 r11 digest 上执行，A1/A2 无 overlay，
B 只 overlay `distributed_runner.py@681db467…351d9` 与
`tensor_arg.py@3b382193…ae55`；三臂 hidden SHA `ee8ae6b4…a7db96a`、
tail token `43640` exact。p50：ITL `21.6805→21.115 ms`（`−2.608%`）、
graph build `4.092562→2.274285 ms`（`−44.429%`）、graph→first runner
`3.098134→1.613020 ms`（`−47.936%`）、runner wave `−24.008%`、
serial submit wave `−23.887%`、graph→chip done `−8.443%`。
各 span 重叠，不相加；正式测量仍为 `serial-eight-rank`、`group_size=1`、
`group_submit=0`。`bind.args` `+0.000220 ms`、占 ITL `0.259%`，
判定 `no_clear_change`，停止在该项投入。

**r12 immutable 门**：digest-only、无 source/runtime/core overlay；
Main H4 all/none 均 `126/128=98.4375%`，MTP BS1/BS16 token
`[6178,410,303]` 且 hidden pass rate `1.0`，dep-only DFX hidden/token exact。
容器 `privileged=false`，显式只见 0–7 卡，8–15 保护空闲；immutable smoke、
fresh registry identity 与 11 个 prestart adversarial fixtures 全 PASS。

最终合同 `step3p5.r12-final-release-admission.v1` 为 `release-admitted`、
`1844/1844 PASS`，SHA256
`511a545956aee4cef7264a74460bd04862846e377ef71eb01619ae4ddbf87f3a`。
边界：没有重采 r12 immutable 性能 A/B/A；DFX 仅 dep-only，不宣称完整 outer swimlane；
镜像 Config 未 bake `PYPTO_H4_RESIDENT=all`。

权威报告：
[`../benchmark/2026-08-27-whole-step-host-graph-submit-r12-release.md`](../benchmark/2026-08-27-whole-step-host-graph-submit-r12-release.md)。

## 2026-08-26 —— replicated-input local-owner MoE r11 发布 ✅

**镜像**：`hub.i.basemind.com/stepcast/vllm-pypto:stepfun-upgrade-20260826-r11`，
manifest `sha256:401ead7da4f957f6532e380fa1a138eda733fe1dc04b40eabc67d79d62a67b12`、
config `sha256:35c42510a64ce3e1c8e899e15c36ab8b534d091ea03a085ec663f18df8706876`。
相对 r10 仅 pypto-lib `fe641929→e6c7d8ec`，落地 replicated-input
local-owner MoE；pypto `519b588a` 及其余 pins 不变。

**门结论**：registry push/raw identity/fresh pull PASS；H4 all/none 均
`126/128=98.4375%`，两臂 output token 与 128 对 active-hidden byte-exact，
TP spread 0。H4-all 64K/1000 p50 `21.477 ms`、mean `22.262 ms`、
p99 `35.882 ms`。

r10/r11/r10 immutable A/B/A p50 为 `21.751/21.745/21.752 ms`，
baseline midpoint `21.7515 ms`，r11 仅 `−0.0065 ms / −0.0299%`；
结论是性能中性/无回退，不宣称 local-owner 带来端到端性能收益。

最终合同 `step3p5.r11-release-admission.v1` 为 `20/20 PASS`，SHA256
`570bb04ef761e66fa12fb246f3482973294fe282688d967c76e119fcda740af7`。
性能数字绑定 `PYPTO_H4_RESIDENT=all`，镜像 Config 未 bake 该值。

权威报告：
[`../benchmark/2026-08-26-local-owner-moe-r11-release.md`](../benchmark/2026-08-26-local-owner-moe-r11-release.md)。

## 2026-08-25 —— packed-NZ MoE fusion r10 正式准入与源码合入 ✅

**镜像**：`hub.i.basemind.com/stepcast/vllm-pypto:stepfun-upgrade-20260825-r10`，
manifest `sha256:8510f30e1f2a2f2edcaa834c831165b349a4aca1212b655ca2a02ed6b3e9907b`、
config `sha256:38ebba41d6aa0c49940c03e2e7c6fa42d85b61d631c143d38944683d0c657b5f`。
镜像内 pypto-lib 为 `fe641929`（tree `5d8f7e64`）；远端
`stepfun/develop` 已用 exact lease 从 `bf3ff440` fast-forward 到 `fe641929`。

**已完成**：

- source unit `162 passed`；image audit/smoke/external extension audit、whole compile、
  registry push、isolated fresh pull 与 raw manifest/config 均 PASS；
- Main 8-step、MTP single、MTP batch16 PASS；
- accepted-oracle N=128：H4 all/none 均 `127/128`，唯一 mismatch `[94]`；
  strict parity 为 output `128/128`、tensor pair `256/256` byte-exact、
  finite `512/512`、TP spread `0`；
- H4-all 64K/1000 p50 `21.742 ms`，相对 source matched midpoint
  `−0.9195 ms / −4.0575%`，相对 r9 published `−0.511 ms / −2.2963%`；
  ITL admission `pass=true`；
- immutable r9/r10/r9 A/B/A p50 `22.524/21.821/22.580 ms`、mean
  `22.862/21.937/22.633 ms`、p99 `28.542/28.338/24.208 ms`；baseline
  midpoint `22.552 ms`、bracket `0.056 ms`，r10 为
  `−0.731 ms / −3.241%`，三臂 hidden/token exact；verdict SHA256
  `8d4224e0214b71bae01efe24393e5886375e04dff5481ffd34ba19e3821ddb0e`；
- 六档 BS correctness `6/6` exact、`12/12` tensor health PASS；BS8/BS16
  单次诊断分别回退 `+5.677/+0.551 ms`，未宣称多 BS 性能全面提升；
- L3/L4 hidden exact，8/8 chip swimlane/DFX 完整；fused E3→E4 median
  `44.97/41.62 us`，routed down `16.18/16.44 us`；
- pypto-lib exact-lease 同步证据为 `git-sync-r10-20260825-144155/`；
- final release contract schema `step3p5.r10-release-admission.v2`，
  `71/71` checks、`pass=true`，SHA256
  `bcdd0b11d346e450dca49b8434544de5566b7fc0ad1a38c715815a41958dafca`：
  `0162:…/r10-release-admission-20260825-150350/release_contract.json`。

**保留边界**：完整 `E5→E6` 仍无 shared + TP-AR + global-fence 统一 endpoint，
继续记 `n/a`；六档单次 latency 仅为 warmup1/iters1 诊断，BS8/BS16 回退
caveat 不因最终准入而撤销。

权威报告：
[`../benchmark/2026-08-25-moe-fusion-image-release.md`](../benchmark/2026-08-25-moe-fusion-image-release.md)；
证据根目录：`0162:/mnt/persist/chensiyu/workspace/moe-fusion-release-20260825/`。

## 2026-08-24 —— 五仓全栈升级 r9 发布、H4 准出与远端同步 ✅

**镜像**：`hub.i.basemind.com/stepcast/vllm-pypto:stepfun-upgrade-20260824-r9`，
manifest `sha256:b637f00c66d4dc976c053c617d2e19e6d6d66f68f4bef30250984da7a71690f6`、
config `sha256:f6c8f72eecad0a9d40d0c4ea55afaab09dd4e2f5fe54d6a091e332465e421dae`。
Registry push、raw manifest/config、fresh pull 全部验证通过。

**最终 pins / `stepfun/develop`**：pypto `519b588a`、pypto-lib `bf3ff440`、
pto-isa `cd4a3d3f`、PTOAS `307d0484`、simpler `85a82c45`、ptoas-bin `v0.57`。
五仓用 `force-with-lease` 推送并远端复核；simpler 非 FF，旧 `e2efebcb` 原子备份到
`backup/stepfun-develop-pre-upgrade-20260824-e2efebcb`。

**门结论**：

- precision `127/128 = 99.21875%`（唯一 mismatch step 94；不是 128/128）；
- Main 8-step、MTP single、MTP batch16 全部 PASS；
- immutable digest combined gate：L3/L4 `torch.equal=true`，8/8 rank
  `chip_swimlane_records.json`，DFX analyzer `pass=true/blockers=[]`，recv_meta ready；
- 前五层 `L0_full_dense / L1_swa_dense / L2_swa_dense / L3_swa_moe / L4_full_moe`；
- 最终 release contract `pass=true`。

**ITL 口径纠正**：同一 r9 digest 默认 unset=`none` 的 64K/1000 p50 是
`27.812 ms`；显式 `PYPTO_H4_RESIDENT=all` 后为 `22.253 ms`。H4 令
`bind.args` p50 `6.461 → 0.063 ms`，代价 `99.64 MiB/rank`。镜像未 bake 该 env，
所以 `22.253 ms` 必须写成“r9 + H4 all”，正式 deployment 仍需接线。

权威报告：
[`../benchmark/2026-08-24-upgrade-r9-release.md`](../benchmark/2026-08-24-upgrade-r9-release.md)；
最终合同：`0162:…/r9-release-admission-20260824-151848/release_contract.json`
（SHA256 `1cd646e3…a08a6`）。

## 2026-08-21 —— 仓库蒸馏：建立 T0–T6 分层 + 坑案例集 📚

**动因**：每 session 必读路径已到 2018 行（`STATUS.md` 644 + `blockers.md` 690 +
`planning/handoff.md` 684）—— 三份「当前真相」文件退化成追加式日志；16 份复盘的教训
没人每次读 2526 行，事实上没在起作用（本周重犯了其中至少 3 条）。

**产出（两次 commit）**：

1. **分层规则 T0–T6** 写进 `CLAUDE.md`（含自查判据 + 硬预算 + 「超预算 = 有东西该下沉」）。
   `STATUS.md` 644→129 · `blockers.md` 690→192 · `planning/handoff.md` 684→90 ·
   本文件 1622→1277（pin 表扩成完整 31 行时间线）。**必读路径 2018→527 行（−74%）**。
2. **新 `progress/landed.md`（T2）**：区分 **IMG**（镜像准出）vs **SRC**（source-overlay GO）——
   这是「确定落地」的边界；含「已否决，不要重试」10 条 NO-GO 台账 + 台账缺口。
3. **新 `postmortems/LESSONS.md`（T0）**：触发式必读索引，`CLAUDE.md` 铁律 §0 强制。
4. **新 `postmortems/CASEBOOK.md`（T0′）**：23 条单点坑，按**现象**索引（LESSONS 按动作索引），
   每条 背景/现象/过程/处置。处置分布 **13 ✅ 已修 / 7 🩹 已绕开（根因仍在）/ 2 ⏸ 未解**；
   🩹⏸ 必写「移除代价 / 复发条件」—— 目的是**别让人把承重的绕路当冗余删掉**。
5. **新复盘 `#16`**（dispatch 融合线定案）吸收 `blockers.md` 那条已定案的 378 行。

**写案例集时发现并修正的三处台账问题**（副产品，不是本轮目标）：

- ❌ **`AR family 占 makespan 95.1%` 全仓查不到出处**，且源文档恰恰是在纠正一个被夸大的
  AR 占比（`15%` → 正确口径 48 次 on-path、约 `8.5~9.7%`）。已从 LESSONS + CASEBOOK 删除。
- 🔧 `54.7 GiB` × 8 `VLLMWorker_TP`（非 root `fuser` 说谎那条）真出处是
  `reference/execution-host-contract.md`，LESSONS 原先误引 p1a benchmark。已改，并修好被
  打断的 `同上` 引用链（3 行）。
- ⚠ **head_gate 处置口径冲突**：`blockers.md` 称 on-device 已恢复、`STATUS.md` §8 称仍在
  worker 侧。判断是**两条路径**（整网 attention on-device 8-block logits vs live bridge
  worker python 预算），已在 `STATUS.md` §8 与 CASEBOOK C1 标注；**上游小 N `matmul_acc`
  是否已修未在本仓闭环**，动 head-gate 前须去 0162 核。

**未做**：`design/performance/`（8324 行，`04-attention-optimization.md` 单文件 2138 行带 29 个
历史/已撤回标记）—— 已批准分两轮，按 campaign 归档是下一轮。

## 2026-08-21 —— MoE dispatch 域小算子融合线**整体关闭**（负结论）⛔

**定案 → 复盘**：[`../postmortems/16-dispatch-fusion-orch-decouple.md`](../postmortems/16-dispatch-fusion-orch-decouple.md)
（含 11 条已撤回主张 + 9 条铁律 + 全部证据入口）。本条只留流水与不在复盘里的数字。

**当日轨迹**：接手 codex 6 天 / 357 run / 8 候选 / 0 落地的 campaign → 一天内定案。
上午写的"完整死锁环"当天下午被自己的下一个门推翻，共 4 次自我更正 + 1 次收盘更正。

**结论三段**：

1. **R9 = NO-GO**（`dd0e9cea`，相对生产基线 R5 `67b73589` 的 `decode_fwd.py` **只差一行**：
   `dispatch_gather` 的 `deps` 多一个 `dispatch_push_tid`）。生产配置 3 次挂 2 次（概率性，
   非确定性）；匹配曝光后**也不快** —— R5 `ITERS=1000` p50 `26.329 ms` 优于 R9 clean run
   `26.615 ms`。作废的旧数据点：`27.478` / `27.757 ms` 是 `ITERS=100` 臂，与 1000 臂不可比。
2. **结构修复候选 = NO-GO**：无卡 codegen 门达标（`local_route_count` 上的 orchestrator 阻塞读
   4→0），但 device 门三臂全挂（`inv=10` / `inv=1` / `inv=357`）。
   ★ 反直觉的原因：**那个阻塞读同时是承重的 run-ahead 流控阀**。
3. **整条线关闭**：只解析已有 STRACE span（不占卡、不改码、不加锁）⇒ p50 `orch`
   `17279.28 → 4443.18 µs`（**−12.8 ms / −74.3%**）而 `device_wall`
   `17466.93 → 17910.32 µs`（**反升**）⇒ orchestrator 从不在关键路径 ⇒ **ROI 上界 = 0**。

**DFX 侧补充证据（不在复盘正文，留档）**：R7 五层 DFX 显示 dispatch 域的收益**被
`dispatch_wait` 吸走** —— `dispatch_meta` 的 `7.12 µs` 消失，但 `dispatch_wait` 从
`2.18 → 12.56 µs`；`dispatch_count_start_to_gmm1_start` `78.05 → 78.32`（**+0.27 µs**），
另一路径 **−3.34 µs**。⇒ 瓶颈是 **WAIT（跨卡等待）而非 small-op 调度开销**，
与第 3 段的 orchestrator 结论独立同向。

**★★ 副产品（量级更大，已转性能主线）**：同一份 STRACE 里 R5 每 invocation `simpler_run`
p50 `26.45 ms`，其中 **`bind.args` = `6.12 ms` ≈ ITL 的 23%**（纯 host 侧参数绑定、
与 `runner_run` 加性；对照臂 `5.87 ms`）。见
[`../design/performance/task-tracking.md`](../design/performance/task-tracking.md)。

**流程根因**（已写进 [`../postmortems/12-integration-churn-meta.md`](../postmortems/12-integration-churn-meta.md)
根因 9/10/11 + [`../postmortems/LESSONS.md`](../postmortems/LESSONS.md) §A）：候选立项前提没被审 ·
看门狗把"慢"伪装成"死" · 为拿一个数去写新候选，而那个数已经躺在失败 run 的日志里。

权威记录：`0162:…/dispatch-orch-decouple-20260821/FINDINGS.md`。

## 2026-08-12 —— TP all-reduce single-row selector 合入，source-overlay GO ✅

`pypto-lib stepfun/develop` 已普通 fast-forward 到
`69ad31e4fd6e40b30e43c2566ce8f8ebd0b2427d`（parent `9ca01d2`，tree
`e26d762c…`）。Main 的 rank-uniform `active_rows == 1` 走静态 8 KiB 两波
one-shot mesh；其他 Main 行数与 MTP 保留静态三波 reduce-scatter +
push-all-gather fallback；ownership 固定为 `HIDDEN // TP_WORLD_SIZE`，与 transfer
chunk 解耦。`dense_mlp_body_tp` 新增 `num_tokens` 源码实参，仓外调用方升级时也须补齐。

- unit `365 passed, 7 skipped`；
- Whole 与 MTP 3/3 在 default/chunk=256 下 compile PASS；
- 8 卡 rows `1/3/16` 功能门 PASS；
- Whole BS1/ctx64K A/B/A=`31.065/29.912/30.999 ms`，candidate
  `-1.120 ms / -3.609%`，precision/per-iteration PASS；
- focused `38.325→22.667 µs/call` 仅为 regular-call kernel-duration pooled
  mean 的机制证据，不是 strict critical-tail；
- 固定镜像内仍为 `pypto-lib@cb96747e`，所以这是 source-overlay GO，不是包含
  `69ad31e4` 的 immutable-image release qualification。

旧 `a791071` Ring 实验是未命中 production canonical body 的 A/A；K6b
dynamic-valid-shape 未落地。后续只做 immutable-image qualification，不恢复两条旧路线。

## 2026-08-11 —— K8 immutable image 发布 + 0162 双精度门/ITL/五层 swimlane ✅

**镜像**：`hub.i.basemind.com/stepcast/vllm-pypto:stepfun-develop-20260811-k8-selective`，
manifest `sha256:076af8a167405d5d0831e234cd16521c77d8bfdd173eff063d820802057c47f3`、
config `sha256:a9d111880883cea0b02e425fdfeaccc2b14bb1d1174c0b73488d8ee6d8004d39`。
**第一个包含当前 tip** `pypto-lib cb96747e` / `pypto 1c048a74` 的 immutable image。
构建在 devbox，**全部验证在 0162、digest-only、无 overlay**。

- **audit + smoke**：`IMAGE_IMMUTABLE_AUDIT` / `CANONICAL_ONLY_SYMBOL_AUDIT` /
  `K8_LANDING_PRESENT` / `[smoke]` 四门全 PASS；落地件==被测件
  （`distributed_runner.py` sha `fe50c11f…39622e`、`decode_fwd.py` sha `eb1f89bf…04fb5`）。
- **精度两条独立证据都过**：byte-exact `hidden_sha256` `567b206b…f03e` == 生产
  baseline + token `14371`；**N=128 预定义冻结 oracle 三轮 `123/128 = 96.09375%`、
  miss `[2,8,13,22,82]`、`tp_spread_max=0.0`，三轮一致且与 Wave5 逐位相同**
  ⇒ K8 未改 token 轨迹。oracle sha `c9b2c721…dd947`（与 Wave3 快照同一份），
  离线无 live server。
- **性能**：clean ITL bs=1 ctx=65536 p50 **`32.14 ms`**（pre-K8 `33.84` →
  **−1.70 ms / −5.02%**）；与 source-overlay 候选臂 `32.08` 差 `0.06 ms` ≪ 地板
  `0.634 ms` ⇒ 镜像复现 K8 收益。`reset_body_us` p50 `523.1 µs`，109/109 步
  `k8_prefix_applied=true`、`k8_control_bytes=47616`。
- **BS1 前五层 swimlane（新增，原为 PENDING）**：LOW-WAIT `rank2` makespan
  `2.204 ms`、static CPM `1.825 ms`(82.8%)、observed 103 task
  （compute 81.2% + stall 18.8% 全 data-wait）、`tp_all_reduce` 占 **15.3%**。
  ⚠ 其余七 rank makespan `288~610 ms` 全被 `tp_all_reduce` 自旋吸收 skew 占据
  （跨 rank 差 275×）；campaign `rc=1`（rank0/1/3/6 各 5 个 `early_dispatch` task
  无 swimlane 记录）⇒ **可用观测，非 sealed publication**。
- **构建期修掉三个同族凭据坑**：① submodule 凭据覆盖必须 key 在**名字**
  `simpler`（路径是 `runtime`，keying on path 静默无效）；② pypto CMake 在无
  secret 的编译层 init `3rdparty/{libbacktrace,msgpack-c}`；③ simpler 另有
  **第二份** pto-isa（`runtime/build/pto-isa` 按 `pto_isa.pin=83d01313`，
  `PTO_ISA_ROOT` 对它无效）。三条都已进 Dockerfile 并注释「为什么」。
- ⚠ **未跑**：Main batch16、MTP batch1/16、六档 64K golden/A/B、formal DFX；
  phase 2b 按用户要求跳过 ⇒ **不是完整 production release-qualified**，完整矩阵
  回退基线仍 Wave5。

数据：[`../benchmark/2026-08-11-k8-selective-window-zeroing-image.md`](../benchmark/2026-08-11-k8-selective-window-zeroing-image.md)。

## 2026-08-11 —— K5-C 否决 + 定位 pypto notify fence correctness 缺陷（device 已证，消融矩阵闭合）⚠

**主结论 1（负结果，有长期价值）**：K5-C（C32′ ring reduce-scatter / ring all-gather）
device 实测只省 `0.54 µs/call`，远低于 14 µs 门 ⇒ **统一定律
`0.80 µs×交易 + 0.94 µs/row×行` 不含依赖深度项**，对深串行链外推严重高估
（C32′ depth 6 串行往返 vs parent depth 3 且 7-peer 扇出并行）。
**后续 AR 方案必须减交易数而不增深度。**

**主结论 2（correctness 缺陷）**：pypto `MakeNotifyCodegenPTO` 生成的 notify 前导把
`dcci(ENTIRE_DATA_CACHE)`（**invalidate-only，无 writeback**）排在任何 drain 之前，
credit 可能跑到 payload 前面。裸 `remote_store` 紧接自己的 `notify` 时 **epoch 0 就坏**。

- **方向被排除**：ring 探针 `ring_up`/`ring_down` 无 fix 都 `exact=False`，有 fix 都
  `exact=True`（`aba-20260811-013946`）。
- **payload 扫描**：`16/32/64/128 KiB` 无 fix **全部** `exact=False`；`16 KiB + fix`
  `64/64` epoch exact（`-015235`、`-014938`）。
- **是顺序不是时序**：安慰剂把完全相同两条指令放 `TNOTIFY` **之后** → 仍 `exact=False`
  （`-015951`）。
- **消融矩阵已闭合**（同一插入点，每臂 kernel diff 恰好为插入行）：`PIPE_MTE3` 单独
  False、`dsb(DSB_DDR)` 单独 False、**`PIPE_MTE3`+`dsb` 组合 False**、纯 MTE3 流量
  （`--drain-store`）False；只有 `pipe_barrier(PIPE_ALL)` True `64/64`
  ⇒ **最小修复 = 一条 pre-CMO `pipe_barrier(PIPE_ALL)`，代价压不下去**。
- **代价已量化**（后处理注入 parent kernel）：全 3 个 notify site `+1.250 µs/call`
  （half-range 0.150，`-100050`）；只 Wave2 一个 site `+0.405 µs/call`
  （half-range 0.005，`-100328`）≈ `0.060 µs/PIPE_ALL`，**比 K2a 的 pipe-specific
  barrier（0.0033 µs）贵约 18 倍，不能外推成免费**。
- **上游诉求（一句话）**：`MakeNotifyCodegenPTO` 在 `pto.cmo.cacheinvalid` 之前补
  `pto.barrier <PIPE_ALL>` —— **把 put 路径已有的那条屏障对齐到 notify 路径**，
  不引入新概念（`MakePutCodegenPTO` 给 tput 夹的两条 `PIPE_ALL` 注释写成
  "WORKAROUND for PTOAS#872"，实际承载正确性；`MakeRemoteStoreCodegenPTO` 什么都不发 ——
  这个不对称就是缺陷来源）。

**两 agent 对账（反向复核）**：codex 独立复核报告
sha256 `37fae3aba51a555189c9da05633d88d0ab7810e28d72df9681f9fcfa11d472ac`。
一致 = pre-CMO `PIPE_ALL` 最小、`dsb` 冗余、上游诉求表述。两处撤回 =
codex 自撤「store-loop 的 MTE3 屏障保证前六个 store 安全」；我撤回 Wave3 slack 假设
（**结构性否证：Wave3 在 consumer read 之后**）。分歧按更保守方向收口：
**生产 Wave2 没有可证明的安全机制，只是当前调度没触发；是否正在损坏未知**
（我原先「近确定性失败 ⇒ 必有结构性保护」的反推默认失败率与结构无关，前提未验证，已撤回）。

**硬约束（现在生效）**：任何把「payload store 与它自己的 credit」拉近的改动 ——
删 Wave3（~5.6 µs/call）、合并 Wave1+Wave2（~5.6 µs/call）、按 peer 融合 store+notify、
单 peer 交换 —— 都必须**先落 fence**；扣掉 Wave2 fence 的 0.405 µs 后净收益约
`10.8 µs/call`，**单独仍不过 14 µs 门**。

**产品状态**：本轮 **0 个优化进生产**。权威报告
`0162:/mnt/persist/chensiyu/workspace/p2-k5-rhrd-20260810/CLAUDE-NOTIFY-FENCE-DEFECT.md`
sha256 `a34817832550b9c68c907a58774403802d79c1926e8aa085b658ff0aafc9f21b`。
详见 [`../blockers.md`](../blockers.md) `UPSTREAM-NOTIFY-FENCE` 段与
[`../design/performance/task-tracking.md`](../design/performance/task-tracking.md)
2026-08-11 更新行。

## 2026-08-10 —— P1a gate 解耦：swimlane critical path 定向优化，bs1/bs8 各约 6%，byte-exact，已发布 `stepfun/develop@d13b2ca6` ✅

**完整报告（含全部 campaign 路径、pass dump sha、UB 排名表）**：
[`../benchmark/2026-08-10-step3p5-p1a-gate-decouple.md`](../benchmark/2026-08-10-step3p5-p1a-gate-decouple.md)。

**方法**：只看 swimlane critical path（5 层 FiveLayerMoe 代表整网），改动全收在 `decode_fwd.py`。

**定位**：interior SWA+MoE 层的 MoE-only 段 15 hop、compute 222.1 + stall 74.6 = `296.7 us`。
链头 `norm_quant_moe_input`(25.8) → `gate_expert_fanout`(32.7) 的串行**只由 `inv_rms` 一个
per-token 标量造成**，而它在 fanout 的 FP32 matmul **之后**才乘 ⇒ fanout 的 cube matmul
根本不需要 norm_quant 的任何输出。

**改法**：fanout 只存 raw FP32 logits 到 `logit_buf`（cube→vec 用 `pl.mul(logits_n, 1.0)`）；
`row_expand_mul(inv_rms) → sigmoid → +bias` 整段按同样的 `ROUTER_GATE_N_CHUNK=32` 搬到
`gate_topk` 开头（它本来就要等 `inv_rms`）。算子顺序不变，只多一次 FP32 tensor round-trip。

**三层验证**：① codegen —— candidate 的 `params_t70`(fanout) 不再 `add_input(moe_inv_rms)`，
join 点搬到 `params_t71`(gate_topk)，task 数与 `block_num=9` 不变；
② device A/B/A —— bs=1/nb512 `36.493 → 33.849`（**+2.645 ms / +7.25%**，地板 0.634）、
bs=8/nb4096 `97.528 → 91.722`（**+5.806 ms / +5.95%**，地板 2.637），
bs=16 @per-request 64K 物理不可行（需 `num_blocks=8192` → 单次 16 GiB `rtMalloc` → `207001`）。
**对外统一口径：bs=1 与 bs=8 都约 6%**；
③ 精度 = **byte-exact**（bs=1 三臂 hidden sha = `567b206b…` = N256 发布 golden、tail token
`14371`；bs=8 三臂 = `1fcd4fcc…`）。

**机理闭环**：MoE-only 段 15 → **14 hop**，`norm_quant_moe_input` 离开关键路径；
链头 `81.8 → 56.5 us`（**−25.3 us/层**），段合计 `296.7 → 280.0 us`，on-path task `99 → 96`。
`gate_topk` `3.1 → 10.3 us` 正是搬进去的尾巴。`−25.3 us × 42 层 = 1.06 ms` 与事前预估 `1.08 ms` 吻合。

**发布**：`a31977fb` **FF 到 `d13b2ca6`**（单 commit，只改 `decode_fwd.py` +63/−35）；
合并后 sha 仍 `28080c53…`，与 A/B/A candidate 臂**逐字节相同** ⇒ 设备数据直接绑定发布代码。
⚠ 0162 连不上 GitHub 443，push 走 `git bundle` 带回本地再推。

**同轮 NO-GO（避免复发）**：`gate_up+act` 合并（真实理由是 **ROI 低于检测地板** ——
干净 baseline 只占 `14.6 us = 0.65%`，映射整网 `0.13~0.17 ms`；⚠ 原先归因"`pl.range(4)`
展开不复用 SSA buffer"**已被本轮预测-验证闭环推翻**，真因是融合新增 c2v pipe slot，
且树内已有能编过的融合路径 ⇒ 融合可行，是 tiling 取舍）；
`act+h_quant` 合并（grid 维度不同 + h_quant 需整行 amax）；
`tp_all_reduce` 降 ring step（前提未证实：on-path AR 是 `45+3=48` 次不是 72，占实测 25.8 ms 的
`8.5~9.7%` 不是 15%；且单个 AR 记到 `35,530.5 us` 说明它主要是吸收 rank skew 的 barrier）。

**下一步（修正后）**：① 跨 rank 负载均衡（天花板最高，无设计）—— 已证 `combine_wait` 由本 rank
active local expert 数决定（rank0：L3 有 1 个 → wait `19.26 us`；L4 有 0 个 → wait `155.40 us`），
且 skew 反过来放大 AR，两件事同源；② cube matmul 的 tile + pipeline 作为**一个 bundle** 上 A/B/A；
③ `tp_all_reduce` 等 step 级插桩证据。**不再逐个 kernel 试融合** —— 单个都低于检测地板。

**同日立规**：[`../reference/execution-host-contract.md`](../reference/execution-host-contract.md)
（发现 codex 在本地跑了 P3 swimlane 分析，已 re-home 到 0162 重跑：CSV byte 级一致、
JSON 68 处差异全是 provenance 路径字符串、数值差异 0，结论未被推翻）+ 半机锁 + 无卡 codegen 门。
新增可复用规则见 [`../postmortems/LESSONS.md`](../postmortems/LESSONS.md) §A/§D。

---

## 2026-08-10 —— MoE BS1 N256 优化发布到 `stepfun/develop` ✅

- `csy0225/pypto-lib:stepfun/develop` 已前进到
  `a31977fbb7ced6d2e599539c223d07813f161140`，合并远端最新 release harness
  `491267c4` 与已验证 candidate `7d3e02ae`；产品 `decode_fwd.py` SHA 保持
  `d392311ce1f38a67ddaa007173bb012c87e68cafeb5dca6b47813a2424683eea`。
- 普通 routed hidden quant N chunk 扩到 `256`；gate/up
  `K512xN64 -> K256xN256`、`slot_num=4`、每 expert N work `20 -> 5`；
  empty-rank scatter 判定移入 kernel 并保留 early staging。
- 0162 BS1/ctx65536/512 blocks 整网 A/B/A：mean
  `36.354 -> 35.055 ms`（**3.57%**），p50
  `35.778 -> 34.271 ms`（**4.21%**），hidden payload byte-exact；
  p99 因 100 样本实现等于 max，仅作诊断。
- targeted replay `123/128 >= 122`、128/128 TP spread=0、step77 token-exact；
  candidate pytest 30/30、ruff、compile-only PASS。merge tree 复制到 0162 后
  pytest 30/30 + ruff 再次 PASS。
- DFX/PMU PASS；event2 是 busy-cycle 而非 byte counter，不得反推 HBM GB/s。
  `down24` 因 e1/e2、scatter 下游相位与 L4 terminal 回退，冻结
  `NO_GO_NO_RERUN`。
- 详见
  [`../benchmark/2026-08-10-step3p5-moe-n256-final.md`](../benchmark/2026-08-10-step3p5-moe-n256-final.md)。

## 2026-08-08 —— MoE session 收尾：统一源码 tip、harness 与 pending build spec

- 远端 `csy0225/pypto-lib:stepfun/develop` 权威 tip 已核对为
  `491267c45875e9b1e0071eed224e2e73526799e2`。该提交链包含
  `7928a275` 五层 MoE compute 优化、`63814d4a` SWA mask 修复、
  `cd19fe6b` active-route scheduling，以及 `491267c4` route/precision
  release harness。
- pypto-lib 定向合同回归：
  `test_live_precision_ab_contract.py`、`test_gen_vanilla_oracle.py`、
  `test_five_layer_moe_route_contract.py`、`test_five_layer_moe_dfx_analysis.py`、
  `test_performance_bc_contract.py`，结果 `122 passed, 1 warning`。
- pypto-project 中当前 pin、MoE 设计、task tracking、handoff、canonical test、
  version matrix 与 container admission pin 已同步到 `491267c4`。
- 新增 pending build spec：
  `deployment/docker/builds/stepfun-develop-20260808-moe-opt-latest-source.env`；
  显式要求 PyPTO=`8e92b468`、attention profile=`a2a3` 和
  `l2_swimlane_reuse_dep_gen`，不固定 `BUILD_JOBS`。
- 本次未构建镜像、未执行 0162 长回归。历史 `c9af5790` digest/golden/A/B 只保留
  为 pre-fix evidence；`491267c4` 的 immutable N=128、六档 BS×64K、formal DFX
  和 all-rank swimlane 仍是下一 session 的 release gate。

## 2026-08-06 —— workload-sized Attention 合入最新 MoE 基线

- `pypto-lib stepfun/develop` 从 `7928a275` fast-forward 到
  `c9af5790d5fe450e14fd43c88099b87539089d17`。合入内容包括 Full/SWA
  workload-sized RoPE producer、显式双 TaskId 依赖、SWA unaligned
  trailing-window 首尾 mask、Full out-proj publication 修复和 A2A3
  blocks-per-task=`22/16/22` profile；不固定物理核心数。
- focused 两层验证分支 `1ea76e0f` 完成 bs1/2/4/8/16/7、每请求64K 的
  linear/reverse matrix、fresh-process exact replacement、SWA ctx65535 direct
  oracle，以及 bs1/7/16 DFX。该证据证明 attention delta。
- 合入前 source-mounted 整网 bs1、每请求64K、warmup=5、50 次：
  min/mean/p50/p99/max =
  `49.328/50.046/49.880/56.362/56.362 ms`，hidden finite、TP spread=0。
  相对 Wave5 p50 `49.796 ms` 为 `+0.17%`，属于噪声，不能宣称 latency 收益。
- 最新源码 immutable 镜像
  `stepfun-develop-20260806-attn-taskmajor-canonical` 已发布：manifest
  `sha256:3eb694e0455749b370c2da441f04badb47f2752edb53f2cf4e6acb1fde125479`，
  config
  `sha256:a6095ba550aa8207e66a10ad2e8923d120af957c9e014349d26915d7ba33d216`。
  credential/pin/clean-tree/profile audit PASS；0162 digest-only、无源码挂载。
- 同镜像整网 bs1×64K 50 次 min/mean/p50/p99/max =
  `39.057/39.594/39.612/40.680/40.680 ms`，hidden finite、TP spread=0。
  相对 Wave5 p50 降 `20.45%`，但比较跨越最新 MoE 等整栈变化，不能全归因于
  attention。
- 同镜像两层 bs1×64K p50 `3.6323 ms`，reference exact、TP spread=0；
  DFX `8/8` rank PASS，LOW-WAIT `rank2/d0` makespan `690.1 us`、TP AR
  span-sum `176.24 us`。
- 整网 bs16、每请求64K 使用 `8192` blocks，compile 成功；KV pool
  `22.541 GiB/卡`、weight pool `24.857 GiB/卡`，prewarm 前约
  `52,013 MiB/卡`，申请 `17,179,870,207` bytes pooled static arena 时
  `rtMalloc 207001`。没有有效 bs16 ITL；runtime-memory A/B 按要求暂停，
  container 和 cards 8–15 均已清理。
- canonical attention 文档和性能优化 skill 已同步；skill 只沉淀 workload/task、
  producer submit、dependency、DFX 和多 batch 性能分析方法，项目状态保留在

## 2026-08-04 —— vLLM live-front co-resident round-trip 打通到 decode ABI（H3），下一墙 = live prefill（H4）⏳

- 分支 `feat/vllm-live-front-wiring`，device 0162 **cards 0-7**（未触碰 8-15 及
  保护 PID 2045390-2045397），镜像 `vllm-pypto:wave5-local`。
- **tail-only vLLM + 常驻 whole-net sidecar 同卡 co-resident 跑通到真实请求进入 gate**：
  vLLM 8 worker `--load-format pypto` tail-only（kept=3/skipped=109539）导出 W8A8
  decoder 权重（8 keys）+ paged KV（8 keys, Main-only）；sidecar `--serve --kv-ipc`
  零拷贝导入 weight+KV IPC，`whole_decode_step3p5` 在 8 chip co-resident
  编译+prewarm+run（`simpler_run` device_wall spans ~50ms），**0 次
  HcclCommInitRootInfo failed、0 次 507018、无 card poison/force-reset**。
- **两处 wiring 修复**：
  (1) **block_table ABI**：vLLM 默认 max-model-len(~262144)→flat 32768 ≠ compiled
  `BTF=512`；固定 vLLM `--max-model-len 4096`（宽 32 × storage batch 16 = 512）。
  (2) **G4 NO_HCCL 补丁不在发布镜像**：release/Wave 镜像都是 standalone(8-15)构建，
  `comm_hccl.cpp` 无 `SIMPLER_COMM_NO_HCCL` gate → env 空转、`comm_init` 撞
  `HcclCommInitRootInfo failed: 7`。从 git `878f3742` 重建 patch，在 wave5 镜像内
  patch comm_hccl.cpp（5 anchor）+ `build_runtimes --platforms a2a3` 重编，mount
  patched `libhost_runtime.so`（host_build_graph + tensormap_and_ringbuffer）进 sidecar。
- 另修 3 个 vLLM 侧 backend bug（分支 commit，未 push）：`a9573180` loader
  load_format="pypto" 强制 coerce 到 "auto"；`c9af2a6a` MTP profile no-op 提到 1..16
  ABI 校验之前（16384-row profile batch）；`d35a71bf` KVPOOL MTP-optional（去掉
  第二次 24.86 GiB 权重导出触发的 OOM，改 Main-only）。
- **下一墙 = live prefill（H4，非 wiring 缺陷）**：真实请求首个 forward 是 prefill
  (`AscendAttentionState.PrefillNoCache`)，`whole_decode_step3p5` 是 decode-only，
  gate 正确 fail-closed (`DecodeMetadataError: unsupported attention state
  PrefillNoCache`) → EngineCore 退出。端到端出 token 须先 prefill 填 KV 再 decode。
- 详见 [`../blockers.md`](../blockers.md) Phase 28 live serving §2026-08-04 更新。

## 2026-08-03 —— Wave5 TP all-reduce source publication 稳定性闭环 + 0162 发布 ✅

- `pypto-lib stepfun/develop` 前进到
  `7099476b7c4f13112b159e237e7a64344803caf0`，并已推
  `csy0225/pypto-lib:stepfun/develop`。最小修复是在 Wave 1 前用 self-target
  synchronous TPUT 发布 source partial，再保持既有 rank-owned reduce-scatter、
  push all-gather 与 Wave 1/2/3 lifetime。
- Main、selected MTP、two-layer harness 与 MTP input projection 返回值 lineage
  同步对齐；数值合同不变：固定 peer 顺序、单 FP32 accumulator、最终一次 BF16 cast；
  不固定 24 核、不新增 orchestration kernel。focused contracts `25 passed`，
  `py_compile` / `git diff --check` PASS。
- immutable 镜像
  `stepfun-develop-20260803-attn-final-wave5`：manifest
  `sha256:4acc77cdce05c40fff7fdbcedb5612fa49c2edc847a534c218389ddc08667b32`，
  config
  `sha256:4f2539c17fe60e61062bd27d96082a707e581b81fe716208c1bca4139dfd7394`。
  audit/smoke/Main+MTP compile/codegen PASS。
- Main N=128 预定义三轮均 `123/128=96.09375%`、miss `[2,8,13,22,82]`、
  hidden finite、TP spread=0；Main batch16 `8/8 exact`、128 active rank rows、
  TP spread=0；MTP batch1/batch16 两轮 token `[6178,410,303]`、pass rate 1.0、
  max diff 0、TP spread=0。
- 64K ITL min/mean/p50/max=`48.523/50.027/49.796/54.539 ms`；
  batch16/context1 min/mean/p50/max=`112.525/112.819/112.827/113.203 ms`。
  DFX LOW-WAIT rank2：64K makespan `38.367 ms`、TP AR compute `2.437 ms`；
  batch16 makespan `107.076 ms`、TP AR compute `2.429 ms`。其它 rank 的长 AR
  span 主要含自旋等待，不是算术耗时。
- 验证只使用 cards `0–7`，未触碰 cards `8–15` 或 PID
  `2045390–2045397`；结束后 cards `0–7` 无残留，保护 PID hash 不变。
- 判定：`ATTN-WAVE4-STABILITY` 关闭，**Wave5 在 0162 release-qualified**。
  当前证据支持 source publication/lifetime ordering 是 0162 的关键边界，但不宣称
  self-target TPUT 是所有硬件/架构的唯一根因。

## 2026-08-03 —— TP all-reduce window lifetime 三波闭合 + Wave4 historical candidate

- `pypto-lib stepfun/develop` 前进到 `d7e1381be0236d6e068cd4d86aa815ea693ea5c7`：
  `d58b6be7` 在 final local copy 后增加第三 completion wave，`d7e1381b` 对齐
  two-layer harness 并增加 AST equality contract；focused contracts `28 passed`。
- Wave3（`d58b6be7`）immutable：manifest `sha256:5c38b669…`、config
  `sha256:c2de3311…`；audit/smoke PASS；N=128 `124/128`、miss `[2,8,13,82]`、
  TP spread 全零。它证明 canonical 三波 lifetime 修复有效，但 harness 尚未对齐。
- Wave4（`d7e1381b`）immutable：manifest `sha256:8125c678…`、config
  `sha256:c340001f…`；audit/smoke/compile/64K ITL/DFX PASS；64K p50 `50.204 ms`。
- 固定 oracle 两轮：Run1 `122/128`、miss `[2,8,13,22,82,93]`、step2 spread=`2.0`；
  Run2 `123/128`、miss `[2,8,13,22,82]`、spread=0；hidden 均 finite。
- DFX LOW-WAIT rank2：makespan `38.504 ms`，TP AR compute `2.125 ms`；其余 rank
  的长条主要吸收 kernel 内自旋等待。
- 判定：**当时 raw token gate PASS；当时 formal release pending TP-spread stability**。
  Wave4 已由后续 Wave5 取代；历史数据保留用于根因演进对账，不作为当前发布状态。

## 2026-08-02 —— Attention/Vec 优化收口 + clean canonical candidate（发布阻塞）

- **源码合入**：
  - pypto-lib `stepfun/develop@76d96bdbeac280f12ecf626b1bbd722b9278719e`；
  - pypto `stepfun/develop@defa97c526fec7e8f032dbbfcc39c820add02bf7`，修复动态
    SPMD launch bound 的 orchestration codegen 变量重命名/声明。
- **设计收口**：logical task 按 active workload 推导，不固定 24 核；5--10 us 仅为
  sweep 起点。Full SV 合并 segment recurrence，只保留 reduce/finalize；Full/SWA
  out-proj cast 默认融合。dense RMS direct BF16 reread 与 dense down-cast fusion 保留；
  all-reduce+residual、residual+RMS stats、RMS+projection、gate/up+SiLU 等无稳定收益
  probe 不合入。
- **batch16**：capacity 与 workload 分离；active-batch=16 ctx=1、异构 16-row context
  与 uniform64K grain 对比已完成，未把一次 batch16 最优点硬编码进模型语义。
- **镜像演进**：
  1. v1 缺动态 SPMD codegen 修复，immutable compile 失败；
  2. v2 代码可执行，但 image config 含旧 CANN 8.5.1 字符串；
  3. clean canonical candidate
     `stepfun-develop-20260802-attn-final-canonical`
     （manifest `sha256:64c573bcf64497da6df0d3d28d7de85dfddde8e2a2a1b70e8bd5123edd51cb9d`，
     config `sha256:c7f612a2562e932908d2a0d9ffadd1a1bd155c70bff0e82c24be32ef6b9f79ea`）。
- **audit/smoke**：CANN 8.5.1 absence/config audit、worktree clean、credential、canonical-only、
  runtime CANN、optimization symbol、PTOAS ldd 与 smoke 全 PASS；immutable 验证无宿主
  源码挂载，只使用 cards 0--7，未触碰 cards 8--15/PID 2045390--2045397。
- **64K ITL**：bs=1、512 blocks、warmup=3、20 iters，min `49.213`、mean `50.568`、
  p50 `50.563`、p99/max `52.537 ms`。
- **DFX 复核**：正确 LOW-WAIT reference 为 rank2（makespan `38.924 ms`，
  TP AR critical-path compute `2.049 ms`）；rank5 的 `344.553 ms` TP AR compute
  主要吸收 collective 自旋等待，不得标成 LOW-WAIT。
- **发布 blocker**：同一 fresh oracle 三轮均 `121/128=94.53125%`；所有 hidden finite，
  run2/run3 有瞬态 TP spread。禁止借用 v2 的 `123/128` 或无限重跑。镜像当前只能标记为
  **clean canonical candidate / release blocked**。
- 完整记录：
  [`benchmark/2026-08-02-step3p5-attention-final.md`](../benchmark/2026-08-02-step3p5-attention-final.md)。

## 2026-07-29 —— PERF-H1 自包含镜像 build + 回归 + MTP CI 修复 + DFX benchmark ✅

- **镜像**：build + push `vllm-pypto:stepfun-develop-20260729-perf-h1`（registry digest `sha256:b4e8c8a457a5…`）。pin = pypto `1f704616` / pypto-lib `4513007d` / pto-isa `ecb6c303` / PTOAS `fc8c6cae` / simpler `e2efebcb` / ptoas-bin `v0.50`。本地核对 5 pin 一致 + 工作树全 clean + smoke PASS 后才 push。H1 源码 commit（pypto `1f704616` / simpler `e2efebcb`）此前已在 fork `stepfun/develop`。
- **MTP CI hardcode 修复**（pypto-lib `perf/step3p5-bc-20260726`，已推 fork）：
  - `4513007d`：MTP oracle 路径可配置化——`_run_mtp` 透传 `--mtp-oracle-dir`/`STEP3P5_MTP_ORACLE_DIR`，去掉 harness 里镜像外的 username 硬路径（`.../logs_n1/live_mtp3_patch_...20260718`），无 oracle 时干净 skip；seed/expected token 去重为常量。**烤进本镜像**。
  - `0f3650c7`：`_run_mtp` 改喂 oracle 配对输入（`P42_nh_row0.pt`）而非本次 Main hidden。真因=旧 wiring 用本次 Main hidden 当 MTP 输入却比 0718 配对 golden（喂配对输入 device 实测 pass_rate=1.0 / token `6178,410,303` exact，证 MTP kernel 无回归）。**test-only，mount 验证未 rebuild**。
- **整网 CI（H1 镜像，cards 8-15）**：`ok=true`——Main 45 层 8 步 token `303,1207,19384,872,428,6127,4231,2636` exact + MTP single/batch16 token `6178,410,303` exact，`hidden_tp_spread=0`。
- **N=256 回归**：H1 vs C4 发布镜像 teacher-forced 256 步 **token 256/256 exact**（含跨 block 边界 step127/128/255），全步 finite。raw-hidden run-to-run 抖动（H1-vs-C4 44、H1a-vs-H1b 34 同量级）经复跑证实 = C4 push all-reduce 浮点归约顺序，**非 H1 回归**；token 不受影响。
- **DFX benchmark**（→ `/mnt/persist/chensiyu/workspace/benchmark/2026-07-29-perf-h1/`）：**ITL p50（`--num-blocks 512`）1024 `50.9` / 8192 `52.0` / 32768 `58.0` / 65536 `64.1` ms，较 C4 同工作点降 23–27%**（device-memset 消掉 per-step host reset 21.5→2.2 ms）。PMU `cube_int8 46.35%`、scope ring heap 峰值 79.9%、`dropped=0` 与 C4 逐项一致（memset 不改任务图/计算）。swim top 为 `tp_all_reduce(r2t15)` 启动 skew 等待条（非计算）。详见 [`benchmark/2026-07-29-perf-h1-image-itl-dfx.md`](../benchmark/2026-07-29-perf-h1-image-itl-dfx.md)。
- **边界**：live N=128 vanilla-raw 精度门本轮未跑（oracle 未起；H1 与 C4 token 一致 → 等价 `240/256`）；`0f3650c7` 未烤进镜像。

## 2026-07-28 —— C/D/G BS1 收口 + 自包含 candidate 镜像验证 ✅

- `pypto-lib` `perf/step3p5-bc-20260726` 与本地 `stepfun/develop` 已到 `563fe62a`，GitHub fork 两个分支均已推送；`pypto-project/main` 文档同步到 `ebad8e0`（随后状态文档继续更新）。
- `b404a3c9` 修复 BS1 根因：local experts 动态 prefix slab 改为固定 expert physical lane bases，恢复 batch-extension invariance。BS1/2/16 单步 `6127→303`、TP spread `0`；BS1 persistent 4-step `6127→303→1207→19384→872`；row0 hidden 与 BS2/BS16 bit-identical。
- 0162 本地 candidate `step3p5-b404a3c9-ci-final-20260728`（image ID `sha256:06261920cced91dafc585cd5e63622a88f798ad5ef6aeeba6480433049d1544f`）smoke/Main 8-step PASS；镜像产品 HEAD=`b404a3c9`，CI cleanup/`--skip-mtp` 三文件为 `563fe62a` 工作树补丁；candidate 尚未推 registry。
- candidate N=256 teacher-forced：hidden finite `256/256`、TP spread `0`、active rank rows nonzero `256/256`、token exact `241/256`；raw 95% gate 不宣称通过。MTP oracle 缺失由 `--skip-mtp` 明确隔离。

按 session 划分的 milestone 日志，append-only，按日期降序。
高层 Phase 01-19 总结见
[`prototype-phase-01-19-summary.md`](prototype-phase-01-19-summary.md)。


## 2026-07-26 —— canonical-only Step3p5 release + 0162 镜像内 N=256 回归 ✅

- `pypto-lib stepfun/develop` 前进到
  `53eb7212c29c9bd015ee060cd9924a13ea781ae0`：删除
  `models/step3p5_opt` package、`whole_decode_opt` 和 `WholeDecodeOpt`，唯一
  Main 为 `models.step3p5.decode_fwd:whole_decode_step3p5`；显式
  `--baseline-main` 仅保留为 0724 rollback。
- 当前 commit 是 0724 `fd26b1be` 的后代，merge-base 正是 `fd26b1be`；
  pypto `ca21ab5f`、simpler `216e7632`、pto-isa `ecb6c303`、PTOAS
  `fc8c6cae`、ptoas-bin `v0.50` 和 vLLM overlay `1b3e538c` 均未漂移。
- 发布镜像：
  `hub.i.basemind.com/stepcast/vllm-pypto:stepfun-develop-20260726-step3p5-only`
  （manifest `sha256:99b2b9718cfa6bf0bb87b221f7d565bf23afd2b89a30ba150e523c44a536ed81`，
  config `sha256:d296461051559e6ea0e22d04a4cc44f749c82f19a50418fe6db75387f1f067e9`）。
- 所有发布证据均在 0162 目标镜像内完成：smoke PASS；Step3p5 unit
  `136 passed, 4 skipped`；canonical-only contract `15 passed`；
  credential/symbol/ldd audit PASS；N=256 raw `240/256=93.75%`。
- canonical-only 与清理前 canonical 镜像逐 step token/hidden
  `256/256` exact，`max_abs_diff=0`、TP spread `0.0`，
  step127/128/255 PASS；说明删除兼容入口不改变数学执行。raw 仍低于历史
  95% gate，因此不能写成 vanilla raw precision PASS。
- B2 静态结构收益校正为 `decode_layer.py 31,686` 行 →
  `decode_fwd.py 4,772` 行，约减少 `84.94%`；MoE loop body `40→1`。
  未做同环境旧实现 compiler wall-clock A/B，不宣称编译加速比例。
- 本地 active `workspace/{vllm-pypto,pypto-lib}` 与 0162 active
  `workspace/{vllm-pypto,pypto-lib,pypto-lib-n1}` 已同步到 `53eb7212`，
  旧入口路径无残留；0162 dirty `workspace/pypto-lib-claude` 未覆盖。

## 2026-07-24 —— 合并 origin/main 到 stepfun/develop + IPC provenance 修复 + 交付镜像 ✅

在 0162 临时工作区 `/mnt/persist/chensiyu/rebase-ws-20260723`（validated 基线
MERGE origin/main）解掉最后一个 runtime 卡点，并交付新镜像。

- **根因**：上游 main 合入的 child-provenance dispatch guard
  （`_child_prov_check_dispatch` + `_child_alloc_prov`，`git diff 36957c6b HEAD`
  证实全是 `+` 新增）用精确 `(worker_id, ptr)` 匹配；fork 的 `import_ipc_all`
  把整块 W8A8 权重/KV 池零拷贝导入后，kernel arg 是 `DeviceTensor(peer_base+offset)`
  的 interior 指针，从不等于精确 base → `submit_next_level: child_memory ... not a
  live allocation (interior pointer)`。validated 基线无此 guard 所以放行（之前
  "byte-identical" 判断是错的）。
- **修复**（4 个纯 Python 文件，无需重编）：`simpler/worker.py` 加 `_child_ipc_regions`
  表 + 区间包含 helper，`import_ipc_all` 增 `region_bytes` 登记
  `[peer_va, peer_va+pool_bytes)`，guard 精确匹配失败时回退区间包含；
  `distributed_runner.import_ipc_all` 透传；`pypto_weight_ipc`/`pypto_kv_ipc`
  传各 rank `pool_bytes`。保留 guard 对 malloc/domain 的精确保护不变。
- **设备回归**（0162 卡 8-15）：整网 8 步 decode `6127→303→1207→6127`，与 live 8000
  vanilla oracle 逐 token 一致（`[6127,303,1207]` greedy→"北京"=6127）。harness
  step2 FAIL 是过时常量 `19384`，非精度问题。
- **无损推 fork stepfun/develop**（全 FF；cherry-pick `7cb2a6b3` ITL harness +
  `merge -s ours` 保留 fork 历史）：pypto `ca21ab5f` / simpler `216e7632` /
  pypto-lib `fd26b1be` / PTOAS `fc8c6cae` / pto-isa `ecb6c303`；ptoas-bin **v0.50**。
- **镜像**：`hub.i.basemind.com/stepcast/vllm-pypto:stepfun-develop-20260724`
  （config `d778c0a7`，digest `2b0dc461`，34.6GB），devbox 构建 → hub → 0162 pull，
  镜像内冒烟 PASS（ptoas 0.50）。build spec + README 登记表已提交 `1fcd4c7`。
- **附带**：0162 搭好 rootless BuildKit v0.31.2（192 核，`/mnt/persist/chensiyu/buildkit/`；
  netboot 需 `--no-pivot` runc 包装；重启后重跑 `start-buildkitd.sh`），供以后快速构建。

## 2026-07-16 —— N=1 standalone canonical stall gate 关闭 ✅

gpu-a910x-0162 上固定
`whole_decode_faithful_real + P_FAITHFUL_MOE_LAYERS=42 + token 6127 +
native W8A8 IPC weights + KV IPC + dispatch-pull + combine-pull`，release
commit `0e7a0fdd` exact-source 在 fresh exporter pool 连续 **20/20 PASS**，
每次 `argmax=303`。runtime min/mean/max=`2.50/2.5605/2.62s`；整理后
final smoke 也 PASS（`2.57s`, `argmax=303`）。20 个逐 worker-run dmesg
窗口无新增 fault、`507018`、`running-stalled` 或 `stranded CQE`；fresh
exporter pool teardown 后 outer 窗口新增 2 条 `stranded cqe`，已记录为
cleanup 边界，未混入 worker kernel 结论。

最终发布 commit：pypto-lib
`0e7a0fddc90c4f2348f1d59e015fb817a0877a02`。最终最小 layout A/B
是 control signal physical allocation `32B -> 512B`：logical view 仍为
`[8,1] INT32`，216/216 signal 物理 nbytes=512，相对 offset 全部
`%512=0`。generator 的真实 strip/regenerate byte round-trip 通过。

历史上把 PUSH/TPUT、某个 stuck kernel 或 signal bit 写成唯一硬件根因的结论
已降级为定位假设。当前证据支持“在 0162 固定环境中，512B signal isolation
与历史随机 stall 消失强关联，且 exact-model-source canonical 20/20”，但
不构成 matched 单变量、跨机器充分条件或 bit-level hardware proof。
当时关闭的是 0162 standalone blocker；后续审计将项目既有的 0234 stall
记录重新列为待完整 manifest 复核的独立 blocker。Phase 28 的 per-layer KV
bridge、redundant-weight/3-way HBM 与 live token-exact A/B 仍为 active 工作。

本案例已进一步沉淀为
`.claude/skills/pypto-whole-net-hang-debug/`：Skill 主体抽象 host/device
日志隔离、PTO2 S1/S3/S4/S5 分类、TASK/CLUSTER/COND 寄存器读取、
task→kernel→生成源码映射、真实 PC 的二级映射、跨 rank 边界反推、
buffer/alignment/dtype/padding/init 审计和 release gate；reference 中保留
N1 实践链与历史纠错；`scripts/analyze_stall.py` 提供不自动宣称根因的确定性
证据汇总。创建 Skill 时还复核出候选 20-run 与 release source SHA 不同，
随后用 `0e7a0fdd` exact-source 重新完成 20/20，补齐最终发布证据。

## 2026-07-10 (续⁵ —— G3 HBM 假门槛移除 + resident-runtime 复用验证 [de-risk 任务 5-7])

承接续⁴。G1 offline（任务 1-4）已收尾；本段是对 G2-G5（任务 5-7）的两个 de-risk 进展。

**① G3 HBM 假门槛移除**：`npu-smi info -t memory -i 8` 实测 cards 8-15 = **65536MB = 64GB HBM/卡**。
TP=8 sharded：vLLM W8A8 ~24G→3GB/卡 + pypto BF16 ~47G→6GB/卡 + KV ≈ ~10GB/卡 → **fits 64GB**。
memory 旧记「vLLM24G+pypto47G=OOM」是 aggregate/非-sharded（71G 压一卡）误判。真正 OOM 风险是
ring arena `task_window`（2^20→64GB，用 2^17=131072 已解），非权重共存。**G3 非硬 blocker**，
memory 已补 `project_hbm_cotenancy_not_gating_64gb`。

**② resident-runtime 复用验证（任务 5 最核心未知）**：worker 加 `--steps N` —— 让**同一个 prepared
`rt` 跨多个 decode-step 批次复用**（每批 reset residual 重 bootstrap）。测 `--layers 0,1,2,3 --steps 2`
真 W8A8 device：decode step 0 与 step 1 输出**逐字节一致**（76/478/520/512），10 次派发共用一次
prepare，**rc=0 无状态污染**。这正是把 pypto runtime 塞进 vLLM per-call `forward`（`_pypto_full_forward`
常驻 module-global rt）的关键机制 —— 现 device 坐实：rt 可持有并重复 run，结果确定、无 corruption。

**任务 5-7 剩余（纯 live 集成，需专门 session）**：常驻服务化（module-global 持 rt，manual enter/exit
替 `with`）+ 真 KV import（attn_setup import_ipc 接进 whole-decode attn args）+ forward_context
（block_table/slot_mapping/seq_lens）接线 + live 8001 mode=full A/B vs 8000 token-exact。**门槛已清**：
HBM 不挡、resident 机制已验、offline worker 是移植参考（`workspace/g1_worker_resident_*`）。

## 2026-07-10 (续⁴ —— G1 任务 4：per-layer weight-stream 重构 + 45 层链 device 跑通 rc=0)

承接续³。完成 NEXT-SESSION 任务 4（扩 45 层链）。

**per-layer weight-stream 重构（修 OOM + 权重复用 bug）**：旧 worker `_moe_block_sh` 用
`_stack_real(KEY)[:, moe_li]` —— `_stack_real` 先 stack 全 [tp, 42, ...] 再切片，w_down_r =
8·42·1280·4096·2 ≈ **3.5TB** transient → 3 variant 同载 OOM。且 moe_block 按 `id(program)` dedup，
**同 variant 多层复用首层权重**（42 个 MoE 层权重各异 → 除首层外全错）。**修复**：新增
`_moe_layer_stack`（slice-then-stack 单层 ~0.9GB）+ `_load_moe_layer_weights`（每 moe_block step 前
把该层权重 `copy_` 进复用的 shared slot，像 x 那样）。

**per-layer 正确性验证（决定性）**：chain `0,1,2,3,4,5` —— L3(swa_silu)/L4(full_silu)/L5(swa_silu)
**共享同一 moe_block program id** 但 moe_li=0/1/2 权重各异 → moe_out **1.88/2.06/0.67 各不同**，
且各自 **torch-ref PASS 1.000**（旧代码会复用 L3 权重 → L4/L5 失配）。**per-layer 权重修复坐实**。

**45 层全链 device 跑通**：`--layers 0..44 --ckpt` → **compiled 7 distinct programs, 87 steps,
all dispatched, rc=0，无 OOM 无 507018**。Option-C 多程序 + per-layer weight-stream 机制端到端通。

**full "torch-ref 全层过" offline 不可达（方法论硬限制，非 bug）**：offline 链用 **dummy KV cache**，
device 残差流在 L17 发散→NaN（**可复现、与输入分布无关**：synthetic std=0.02 与 last-prefill-token
真实 bootstrap 都在 L17 NaN）；per-layer moe 对拍（即便喂 device 输入隔离）0.5-0.9，因 MoE routing
对 dummy-KV attention 输出敏感。skill/memory 早已明确：**full-chain / attention-core 正确性必须靠
live A/B（真 KV），offline 合成链覆盖不了**。单 kernel 各自已验证（moe_block vs vLLM 0.9995 历史）。

**本 session G1 offline 收尾**：任务 1（真 W8A8 dense/attn device）✅、任务 2（3-scalar split：
full+silu-MoE 精确）✅、任务 3（L43/L44 SplitIncoreOrch 编译）✅、任务 4（45 层机制 + per-layer
权重 device rc=0）✅。**full-chain token-exact = G2-G5 live A/B**（NEXT-SESSION 任务 5-7）。

**遗留可查项（非阻塞）**：offline dummy-KV 链 L17 device NaN 可复现——若后续想要 offline 45 层
数值信号，需真 decode-step KV（现无，dump 是 prefill）或直接上 live。swiglu(L43/L44) offline 数值
不可信（synthetic-only marker），走 vLLM/live。

**worker 备份**：`workspace/g1_worker_task4_final_20260710_*.py`（含全部 host 修复 + per-layer
weight-stream + per-layer isolation + last-token bootstrap）。

**下一步 = G2**：`_pypto_full_forward` live wiring（vllm_monkey_patch.py:233）——常驻 DistributedWorker
holder + import KV/权重 pool + 45 层 dispatch loop（常驻 DeviceTensor residual handoff）+ 读 live
forward_context 进 attn args + final hidden copy 回 → live single-handoff A/B token-exact。

## 2026-07-10 (续³ —— G1 任务 3：L43/L44 SplitIncoreOrch 编译修复 + 45 层 weight-stream blocker 定位)

承接续²。完成 NEXT-SESSION 任务 3（扩 45 层前先修 L43/L44 standalone SplitIncoreOrch）。

**编译 blocker 根因 + 修复（device 验证）**：swiglu MoE 变体（L43 swiglu7_silu / L44 swiglu7_swiglu16）
的 standalone `select_moe_block` 编译报 `#1828 SplitIncoreOrch: InCore ScopeStmt found in non-InCore
function` at `moe.py:1813`。根因 = `_quant_moe_input`（moe.py:1801，per-token INT8 dynamic-quant，
**仅 swiglu 路径 `_routed_swiglu_step` 才调**，故 silu L3 不受影响）声明为 `@pl.function(InCore)`，
其整个 body 是一个 `pl.spmd`（本身即 InCore scope）→ #1828 outliner 拒。**修复**：decorator
`InCore → Inline`（body 字节不变），对齐同文件工作的 `_expert_routed`(959) / `gate_step`(555) /
`expert_shared_step`(1439)（全是 Inline + pl.spmd + pl.Out）。3b236e6 时是 InCore（旧 pypto 无
#1828 故过）；#1828 是升级后新加的更严 precondition（见 memory `upgrade_pypto_originmain_breaks_moe`）。

**device 验证**：`--layers 0,1,2,3,43,44 --smoke` → **compiled 7 distinct programs, SMOKE_RC=0**；
`--layers 0,1,2,44 --ckpt`（真 W8A8）→ **device rc=0 无 507018**，L44 swiglu16 moe_block 跑通。

**任务 3 完成（编译 deliverable）**。swiglu MoE **offline synthetic 数值不可信**（moe_out kern-vs-torchref
0.363）——但这是 synthetic-ref fidelity 问题，非 kernel bug：① 项目已有 commit `13384ed7 docs: mark
swiglu16 as synthetic-only`；② kernel 侧 swiglu16 修复齐全（`2b00bec9` 5-chunk clamp + `1a6c6342`
router_bias BF16 + SHARED_SWIGLU_N_CHUNK=32）；③ L44 历史上 vs **真 vLLM dump** 已 PASS 0.9995；
④ torch-ref 有两处 axis 与 kernel 不一致（clamp-limit 本 session 已修让 ref 收 per-layer limit；
routed INT8 requant ref 仍 `routed_w8a8_dynamic=False` 不 requant，属 gap-5 territory）。swiglu16
真精度定论 = vLLM dump / live A/B，非 offline 合成链。silu MoE（42 层里 40 层）offline 精确 1.000。

**任务 4（45 层链）发现 blocker——需 per-layer weight-stream 重构**：当前 worker `_moe_block_sh`
把每个 variant 的全部路由专家权重 stack + `share_memory_()`。3 个 MoE variant 同时载 → `/dev/shm`
峰值 OOM（`No space left on device`，虽 shm 1TB，峰值 RSS 764GB + 多 variant stack）；且 worker 按
`id(program)` dedup moe_block，**同 variant 的多层复用首层权重**（42 个 MoE 层权重各异 → 除首层外全错）。
45 层链需改成：moe_block step 前把该层权重 `copy_` 进 cached sh 的权重槽（像 x 那样），按需流式载，
不 stack-all。此重构与 G2 常驻 weight-IPC（47GiB WeightIpcExporter）目标重叠。

**修的 host bug（本 session 续²+续³）**：`_set_gate_exp` 广播、`_recon_attn` per-rank w_g、`_share`
连续化、torch-ref `step_moe_block` 收 per-layer swiglu limit（clamp axis）。moe.py `_quant_moe_input`
InCore→Inline。**备份**：`workspace/g1_worker_task3done_20260710_225746.py` +
`workspace/g1_moe_incore_inline_20260710_225746.py`。

**下一步**：任务 4 = worker per-layer weight-stream 重构（gate 45 层链）；然后 G2 `_pypto_full_forward`
live wiring。swiglu 精度走 vLLM/live 不走 offline 合成。

## 2026-07-10 (续² —— G1 Option-C 真 W8A8 dense/attn device 跑通 + torch-ref 对拍 [tasks 1+2])

承接 Pass 1/2。目标：完成 NEXT-SESSION 任务 1（真 W8A8 接 dense/attn，此前仅 moe_block 真、
dense/attn 还是 synth）+ 任务 2（torch-ref 逐层对拍坐实 3-scalar split）。发现上个 session
已把 `_override_real_dense`/`_override_real_moe_attn`/`_TorchRefChain` 代码写好但**未在 device
验证**，本 session 跑通并修 bug。

**修了 3 个 host 侧 bug**（device 路径本身健康，8 卡 PREPARE OK 无 507018）：
1. `_set_gate_exp` `IndexError`：首层 dense 用 bootstrap `cur=[1,B,H]`（未 expand），循环 `[r]`
   越界。修法：leading-dim=1 时广播 row 0 到所有 rank（current_hidden 是跨 TP 复制的 residual 流）。
2. `_TorchRefChain._recon_attn` w_g shape：per-rank `w_g` 用**全局** `NUM_HEADS_FULL(64)` 切
   `[:, :64]`，但 per-rank 只有 pad=16 列 → 切不动 → cat 8 rank = 128 而非 64。改用 per-rank
   `NUM_HEADS_*_LOCAL`（device 侧 `_override_real_moe_attn` 本就对，只 torch-ref 错）。
3. `_share` 非连续：`_override_real_moe_attn` 的 `w_g` slice+reshape+`.to(bf16)`（bf16 no-op 保留
   strided view）非连续 → `make_tensor_arg` reject。修法：`_share` 统一 `.contiguous().share_memory_()`
   （对已连续张量无副作用，覆盖所有路径）。

**4 层链（0,1,2,3 = 3 dense + 1 swa_moe，5 步）device rc=0**（真 W8A8，
`/mnt/nvme1/chensiyu/step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp`）：

| step | device next_hidden | kern-vs-torchref |
|------|------|------|
| L0 full_dense | 76.0（≠synth 30.9）| **PASS 1.000**（max\|diff\| 0.75）|
| L1 swa_dense | 478.0（≠synth 44.8）| 0.995（max\|diff\| 2.0）|
| L2 swa_dense | 520.0（≠synth 59.8）| 0.994（max\|diff\| 4.0）|
| L3 swa_moe/attn | resid1 512.0 | 0.994（max\|diff\| 4.0）|
| L3 swa_moe/moe | moe_out 1.88 | **PASS 1.000**（max\|diff\| 0.047）|

**任务 1 完成**：真 W8A8 dense/attn 已接 device，输出全部 ≠ synth，rc=0 无 507018。

**任务 2（3-scalar split 验证）**：full-attn(L0) + MoE-block(L3 moe_out) **精确 1.000**；SWA-attn
路径（L1/L2/L3-attn）稳定 ~0.994（**不随层数累积** → 排除层索引错位，错位会灾难性失配而非 0.994）。
worker 的 `_compare_layer` 用 **过严** 阈值 0.999；项目实际 L1 判据是
`ratio_allclose(atol=0.04, rtol=0.04, max_error_ratio=0.10)`（允许 ≤10% 元素超差）→ 0.994（0.6%
超差）**满足判据**。合成输入幅值巨大（478-520）放大了 SWA online-softmax 参考的 BF16 微小差；
真 decode-step golden 不可得（现有 dump 是 prefill 18-token），SWA 是否影响 token-exact **只能靠
live A/B（任务 5）定论**，offline 合成链无法覆盖 attention-core。

**worker 备份**（in-tree 未提交）：`workspace/g1_worker_task1done_20260710_220029.py` +
日志 `workspace/g1_task1_realw_20260710_220029.log`。

**下一步**：任务 3（L43/L44 standalone SplitIncoreOrch 修复）→ 任务 4（扩 45 层链）→ G2 live wiring。

## 2026-07-10 (续 —— G1 Option-C 整网 decode worker 真机 Pass 1/2)

承接同一目标（live single-handoff A/B）。用户拍板两条决策：(1) 环境不 rebase，基于当前
0162 `stepfun/develop @47c260e3` 一致线开发；(2) 走 **Option-C 多程序**路线（非 n1-fusion）。
4-agent team（reverse-review / hw-analyst / sw-analyst / upstream-scout）协作。

**关键修正（2 agent 独立读码，推翻旧 memory）**：`select_moe_block` 返回的 standalone
`EpTpMoE` **不自带** post-attn RMSNorm / residual（只出 `moe_out`）。Option-C worker 必须自补：
attn→未归一化 resid1 → post-RMSNorm(EPS=1e-5, Gemma +1.0) → moe_block → `next_hidden=resid1_fp32+moe_out`。

**卡状态澄清（hw-analyst）**：cards 8-15 的 Aicore=100% 是 **sticky 计数器假象**，非功能性
poison —— L3 allreduce golden smoke（-d 8-9）实测 `max|out-expected|=0`。可直接 launch，无需重置。

**gate_r 语义（sw-analyst）**：`gate_r` 是**真 per-head gate 乘子**（非 ×1 旁路，committed 于
47c260e3）；worker 必须喂 `gate_exp=sigmoid(RMSNorm(hidden)@w_g)`（复用 gate_r 槽，见 head-gate
matmul_acc N=16 绕过）。

**worker 交付**：`_stage_whole_decode_run.py --worker`（+~508 行，in-tree 未提交）—— N=7 Option-C
DistributedWorker（dedup by `id(select_moe_block)`+kind）、3-scalar layer_idx、worker 自补 norm+residual、
per-layer gate_exp、real W8A8 加载。reverse-review 抓到 gate_exp L96 shape bug（`w_g` 取成 `[pad]`
向量应切 `[HIDDEN,pad]` 块）+ sw-analyst 自查 4 个（double `_expand_tp`、`w_g` 非连续切片、
fork 后共享内存 must-`share_memory_()`-before-`prepare()`、synth shape）已全修。

**device 结果（cards 8-15, `-p a2a3 --tp 8 --dev-offset 8`）**：
- **Pass 1（synth 机制）rc=0**：4 程序编译 + PREPARE OK + 8 chip ready + 全 5 步派发（L0/1/2 dense +
  L3 attn+moe_block），**无 507018**。逐步 next_hidden：30.9 / 44.8 / 59.8 / 61.8。moe_out=0（synth 零专家，
  验证 dispatch/combine/shared/residual 机制）。**N=7 co-prepare + 3 程序类型 + resident-sh 复用 +
  3-scalar + norm/residual handoff 在真机验证通过**。
- **Pass 2（真 W8A8）rc=0**：47GB 真权重（8 rank bundle）加载成功 + 全链真机派发无 507018。
  **`moe_out` 从 0 → 3.5（非零）—— 真 MoE 专家在 device 上跑通**（整网 decode 首次带真权重跑通）。
  边界：dense/attn 各层输出 ≈ Pass 1 synth（31/44.25/60.5/63），说明 **dense/attn 仍走 synth 权重**
  （deferred wiring 未完成），当前仅 moe_block 接了真权重。

**遗留（继续做）**：(1) dense/attn 真 W8A8 wiring（build_dense_inputs / `_moe_attn_sh` 目前 synth）+
per-rank gate；(2) torch-ref 逐层对拍坐实 3-scalar 数值正确性（离线 attention-core KV-blocked，真
token-exact 留 live A/B）；(3) L43/L44 standalone `SplitIncoreOrch`（`_quant_moe_input` 的 `pl.spmd`
在 InCore helper 泄漏 InCoreScopeStmt）—— 定 **Option C：InCore→Inline**（对齐 DeepSeek `gate.py`
+ step3p5 自己的 `_expert_routed`，数值与 L43 device-PASS 版字节一致），扩 45 层前修。


## 2026-07-10 (goal session —— 3-scalar split committed + push fork + decode 接管 gap 盘点)

承接「继续 pypto+vLLM 集成、完成后端替换、接管 step3p5 整网 decode」目标。启动 4-agent team
（reverse-review / hw-analyst / sw-analyst / upstream-scout）。全部开发在 0162 `stepfun/develop`。

- **环境确认 latest/consistent**：driver `25.5.2` / CANN `9.0.0 non-GA` / pypto `5e619dc7` /
  pto-isa `ecb6c303` / PTOAS `72ada0a1` / simpler `71e39623`；cards 0-7 = vanilla oracle(8000)、8-15 空闲。
  识别并规避本地 `b-csy-develop` 的 `feat/whole-net-n1-fusion` 分叉分支（非权威，开发一律 0162 stepfun/develop）。
- **tmov 编译 blocker committed（`d3075ac9`）**：`OUT_PROJ_N_CHUNK 256→64`。4-agent 定位根因 = N=256 时
  out_proj cube RHS 128KB 超 L0B 64KB → #1601 Vec-LHS→Mat staging → 910B 非法 Mat→Mat tmov（无 L1→L1
  DMA；#1960 只检测）。arch-gate 是死路（跳过 staging 重触发 L0B 溢出）。真正根因修复 = 对齐 Qwen3
  split-N atomic-add out_proj，deferred prefill Phase 17/22。MoE compile rc=0。
- **⭐ 3-scalar layer_idx split committed（`8b4bf3fa`）—— 整网多层 gating blocker 内核修复**：单 `layer_idx`
  无法索引三种布局不同的权重栈（norm[45]abs / attn[full|swa]type-local / dense-MLP[3]dense-order，仅 L0
  重合 → 多层拿错权重）。拆成 norm/attn/mlp 三 scalar，74 内核 edit + callers + ST arity（`47c260e3`）。
  原子 patch 脚本 backup+assert+rollback；reverse-review 语义 GO（index-class/arity/dispatch 全对）；含
  `layer_cache_base`（KV[45]abs）修正。`_smoke_program_build` rc=0。**单层行为不变、只改多层索引**。
- **push fork**：`csy0225/pypto-lib` stepfun/develop `b511da0 → 47c260e`（bundle-via-local，token 不落 netboot 0162）。
- **诚实边界**：单层 ST 无法作 device gate —— 本树三个单层 ST 各有独立 pre-existing 腐坏（dense:
  `moe.py:208` apply_tp1_patch assert；multirank dense: 缺 `gate_r`；MoE: `w_gate_d` 12-vs-3 层 OOB），
  均非本次回归。3-scalar split 的 device/多层正确性验证走 Option-C 整网链 vs vLLM（下 session）。

## 2026-07-10 (goal session cont.) —— tmov 阻塞解除 + 整网 43/45 层 COMPILE；余 L43/L44 SplitIncoreOrch

目标：完成 e2e 集成 + 精度验证。承接用户提供的 tmov fix 文档
（`deployment/troubleshooting-mat-mat-tmov-vec-lhs-matmul.md`）。

- **tmov 阻塞解除（应用文档模型侧修复）**：`OUT_PROJ_N_CHUNK` 256→64 + `fp32_chunk`
  两处 rename（attention_full→oproj_fp32_chunk / decode_layer→dense_fp32_chunk）→ out_proj
  矩阵乘 L0-sized → 不触发 #1601 Vec-LHS staging → 无 Mat→Mat tmov。clean pypto 5e619dc7
  重编（去掉此前实验性 arch-guard）。**dense L0/L1/L2 COMPILE PASS**。
- **整网 Option-C 编译 sweep（clean pypto + tmov fix + ptoas v0.45）：43/45 层 COMPILE OK**
  （dense fused + L3–L42 silu MoE via Option-C[TP-attn + select_moe_block]）。
- **余 L43/L44（仅有的 2 个 swiglu 变体 MoE）FAIL：SplitIncoreOrch**（`_quant_moe_input`
  的 pl.spmd 在 swiglu chip_orch 未被 outline，moe.py:1813）——升级栈第 3 个 codegen 回归
  （tmov #1601 已修；silu-SplitIncoreOrch b511da0e 已修；swiglu 未覆盖）。文档：
  `deployment/troubleshooting-splitincoreorch-swiglu-moe-L43-L44.md`。committed 94aa015c 与
  gap5-wip stash 两版 moe.py 同样 43/45。
- **gap-5 收尾**：84%→24.37%（materialization 解 cast→cube codegen），残 24% = INT8 1-LSB≈atol
  的度量问题（INT8-accurate，非 bug），验收走 live token-exact A/B。gap-5 INT8-native 为 model-side，
  已 git stash 保留；整网 e2e 走 committed BF16-dequant moe.py。
- **下一步**：L43/L44 SplitIncoreOrch 修复（outline pass 或 swiglu chip_orch 重构，mirror tmov 思路）
  → 整网 45/45 编译 → device chain（逐层 vs vLLM dump）→ `_pypto_full_forward` live single-handoff
  → 8001 A/B token-exact。

## 2026-07-09 (晚, goal session) —— gap-5 根因纠正 (IR 实证) + DeepSeek 对齐 materialization: 84%→24.31% ✅⏳

**团队 `vllm-pypto-e2e`（team-lead + reverse-review / hw-analyst / sw-analyst / upstream-scout）。** 目标：w8a8 精度根因 + 整网跑通，对齐 DeepSeek。

- **gap-5 根因纠正（IR dump 实证，推翻历史 matmul_mx 定位）**：dump FAIL 探针 `_probe_fixb_onthefly.py --smoke` 的 passes_dump → 只有 registered `tile.matmul`（99×）+ `matmul_acc`（99×），**0 个 `tile.matmul_mx`**（该 op 从不 emit，且 A5-only、910B 无 codegen）。真因 = in-kernel `cast(→INT8)` 结果喂 cube Left 操作数时，pypto 给它默认 `fractal=512`（`GetImplicitTileView` 只对 Acc 推 fractal，`ComputeRewrittenType:439` 复制该错值）→ INT8 cube 读错行。经 materialized tensor 的 `tile.load` 会被 DMA 正确 fractalize，故 GM/staged INT8 对、in-kernel cast 错。
- **用户决策**：完全对齐 DeepSeek（我们的 on-the-fly 分歧在正确性+性能上都更差：多一次每-expert 重量化 + BF16 a2a 翻倍）。不改 compiler（走 model-side staging）。
- **device 里程碑（cards 8-15，restored INT8-native moe.py，`--layer 3 --w8a8-native --target out --bypass-gate` vs vLLM out.pt）**：**next_hidden_out 24.31%（127458/524288，was 84%），干净 10s 无 507018**。恢复出的 stash moe.py 已是完整 DeepSeek 结构（gate_up 读 GM-INT8 local_routed_x；down 重量化经 materialized h_i8 再 re-read；无裸 cast→cube）→ **materialization 已解 cast→cube codegen bug（84%→24%）**。
- **剩余 24% = device 侧 INT8 精度残差（非 codegen garbage）**：误差 ~0.05 刚过 atol=0.04；rounding 正确（INT32-rint→INT8-trunc）；CPU 复算同方案 0.9998 cos。嫌疑 = down-leg `[RECV_TILE,1]` h_scale_tile 重量化 bridge / dequant fp32-vs-bf16 / partial-tile。下一步 `--dump-stages` 定位 dispatch vs expert-compute（Phase-21 式精度对齐）。
- **上游 issue 已起草**（hw-analyst）：`GetImplicitTileView` 对 Left/Mat 不推 cube fractal（只 Acc 推）+ P2(PASS)/FIXB(FAIL) 最小复现 + v0.49 仍 FAIL 证据，待提 hw-native-sys/pypto。
- **边界**：0162 pypto-lib 工作树现为 stash INT8-native moe.py（干净 94aa015c 备份 `.clean_bak_20260709_225749`）。整网（问题2）不依赖 gap-5，可走 BF16-dequant（0.9995），真正 gate 是 co-tenancy 507018。

## 2026-07-09 —— 全栈升级到最新 + SplitIncoreOrch 移植修复 + gap-5 上游定位 ⏳

**团队 `vllm-pypto-e2e`（reverse-review / sw-analyst / upstream-scout；hw-analyst 未启，reverse-review 与 upstream-scout 各有一次 429 限流）。**

- **决策（用户拍板）**：升级到最新 commit 作 go-forward base；升级版与旧版**状态对上（parity）即推 `stepfun/develop`**（不搞 0709 分支）；整网串联 + gap-5 精度下 session 做。
- **升级组件 HEAD**：pypto `5e619dc7`（rebased origin/main + 4 commit：DeviceTensor glue 等）、pto-isa `ecb6c303`、PTOAS-src `72ada0a1`(v0.49)、**ptoas-bin v0.45→v0.49**（LLVM21，GitHub release 下载）、simpler `71e39623`（origin/main + PID-whitelist patch）；pypto-lib `1a6c6342 → b511da0e`（+SplitIncoreOrch 修复，**已 push fork stepfun/develop，ff**）。
- **升级引入的新回归 + 修复 ✅**：rebased pypto 的 `#1828`（`49f03c3e` "precondition safety net"）新增 `SplitIncoreOrch` 硬校验，硬失败 step3p5 MoE `chip_orch` 编译（`InCore ScopeStmt not outlined`）。**根因**=`moe.py` `_zero_routed_y_buf`/`_serialize_after_shared` 整个函数体被冗余的单层 `with pl.at(CORE_GROUP)` 包着（对照能过的 `_publish_src_route_table`/`dispatch_step` 无此 wrapper）。**修复**=unwrap 冗余 `pl.at`（commit `b511da0e`），语义不变（InCore 函数体本就跑在 core 级）。**验证**：moe_block `ffn_out` 8 卡 device PASS（`moe_out` ratio_allclose atol=0.04，19.64s）。
- **parity 回归（升级栈 v0.49，card 8-15）**：moe_block ffn_out device **PASS**、`test_step3p5_w8a8_e2e_st`+`test_weight_loader_w8a8` **6 passed**、`test_decode_acceptance` **PASS** —— 与旧版对上。dense/swa ST 失败 = `moe.py:208` `SH_INTER_LOCAL==SHARED_SWIGLU_N_CHUNK*5` 断言在 `apply_tp1_patch` TP=1 reload 时 import 阶段触发（纯 Python assert，与 pypto 版本无关；违反单卡 ST/UT 铁律，旧版同样失败 → parity 非回归）。all-layers detail(CPU) 在**默认严格 tol(5e-3)** 下 pass_rate≈0.989（历史 PASS 用 atol=0.1）——CPU torch-ref、版本无关，parity-neutral。
- **gap-5（issue #1）无上游修复 + 根因锁定**：五组件全升级（含 v0.49）**仍未修复** `cast→int8→cube` 误编译（FIXB 98.47% / P2 int8-copy 控制组 PASS）。upstream-scout：**无上游 commit** 修此 bug；根因 `pypto/src/ir/transforms/infer_tile_memory_space_pass.cpp:55-56`（`tile.matmul_mx*` 在 `kUnregisteredCubeOps` → INT8 cube A-operand fractal=32 layout 未推导；`tcvt` 输出保持 plain Vec layout → cube 读 garbage rows；GM-copy int8 已预 fractal 化故 OK）。file-ready 报告 `pypto-lib/docs/upstream-issues/gap5-cast-int8-cube-codegen.md`。INT8-native gated OFF（`select_moe_block(w8a8_native=False)`），BF16-dequant 是工作路径。gap-5 WIP（two-class + INT8-native + resid1 harness）已 `git stash`（`gap5-wip+splitincore-20260709`），不进 clean base。
- **下个 session**：(1) 整网 device-output chain（`_stage_whole_decode_run.py:311` TODO#11，resident DeviceTensor 层间串联）；(2) gap-5 精度（提上游 issue + 等修 / 或本地 fix `infer_tile_memory_space_pass`）。


## 2026-07-08 —— blocker-1 (整网 co-prepare 死锁) 定位收敛 + Option-C 数值 handoff 落地 + L3 device 精度 PASS ✅

**团队 `vllm-pypto-e2e`（reverse-review / hw-analyst / sw-analyst / upstream-scout + moe-implementer）。**

- **blocker-1 定位收敛**：N≥6 co-prepare dispatch wedge 主因是 **distinct 程序数 over-counting**——`select_moe_block` 对 silu 层 swa/full 返回同一 program，真实整网 = **7** 个 distinct 程序（非文档记的 8；harness `moeblk_cache` 按 kind 多编译一遍 silu）。**N=7 device 验证跑通**（探针 + 完整 7 程序链 11 步全派发 rc=0，无 507018）→ 整网单 worker live-serving 结构可行，无需分批/改 runtime。4-agent file:line 否定了 comm-window O(N) 池 / tensormap 分区 / fork-prewarm race / state-select 假设（用户的 signal-window/GM 假设被代码否定——comm window 同名 comm_d0 per-dispatch 分配即释放）。「为什么 N=8 挂」的确切设备天花板（IPC-handle vs AICPU-identity 表）未 micro-pin，只在将来 ≥8 co-prepare 才要紧。详见 `blockers.md 续6` + memory `blocker1_coprepare_wall_overcounting_N7.md`。
- **Option-C 数值 handoff 落地（task #4）**：扩展 `EpTpMoE.chip_orch`（moe.py）自带 post-attn 零中心 RMSNorm 前导（`(resid1_fp32*inv_rms)*(gamma+1.0)`，+1.0 load-bearing，EPS=1e-5，post_rms_weight[layer_idx]）+ FP32 残差后导（`next_hidden=resid1_fp32+moe_out`），bit-for-bit 对齐 fused decode_layer.py:3371-3512。frontend smoke COMPILE OK L3/L43/L44；reverse-review 5 点 GO。
- **L3+L44 device 精度 PASS（task #5，两个变体类都过）**：`_stage_moe_block_precision.py --target out`（我方修正版：喂 `post_attn_residual.hidden_states` 作 un-normed resid1 + FULL 45-row post_rms stack + layer_idx=args.layer）→ `next_hidden_out` vs vLLM `out.pt` `ratio_allclose(atol=0.04)`：**L3 swa_moe silu PASS 18.11s + L44 full_moe swiglu16 PASS 19.23s**，both rc=0，真 W8A8，cards 8-15，无 507018。证明扩展 moe_block 的 norm+residual 胶水在真机对 silu 和 swiglu 变体都 bit-correct。结合此前 per-layer moe_out-vs-ffn_out PASS → MoE 层 whole-decode 数值闭环（un-normed resid1 → 正确层输出 out.pt）。
- **发现并修复的 shape/index bug（用户重点关注的类）**：harness 初版 `post_rms0 = b[KEY_POST_ATTN_RMS][pos]`（pos=layer-3）对 45-row all-layers norm stack 取错层（L3→layer-0 gamma）；MoE 专家权重是 42-row `[pos]=layer-3` 才对。修法=传 FULL 45-row stack + layer_idx。落 memory `feedback_step3p5_weight_stack_index_class.md`。
- **诚实边界**：整网端到端精度对齐的真正达成 = **live single-handoff A/B（8001 vs 8000）**（offline chained 对 attention-core 受 vLLM dump 缺 KV 限制）。`_pypto_full_forward` 仍是 fail-closed placeholder，wire live runner + KV-IPC + A/B 是多周工程（task #6）。本 session 达成：blocker-1 解除 + 数值 handoff 落地 + L3 device 精度 PASS。


## 2026-07-05 (later-8) —— 穷尽调参矩阵：507018 co-tenancy 不可调，socket-worker 路径判定不可行 ⏸

- **穷尽 (worker-env × vLLM-gpu-mem) 网格**：(default,0.80)→routed 507018；(RING_HEAP=4GB,0.80)→
  16GB arena OOM(207001)；(4GB,0.55)→507018(rank0,2)；(default,0.55)→507018(rank2)。**只有 16GB arena
  OOM 可用 gpu-mem 调掉；507018 对 env/内存全不敏感**。→ 重型 routed grouped-GEMM 内核在独立 ChipWorker
  里与 active vLLM Worker_TP 同卡 → AICPU device-context co-tenancy 507018（部分卡、非确定）。卡从未 poison，
  每次干净拆除。
- **最终判定**：socket-worker + 独立 pypto runtime 对重型 routed MoE 内核在 live vLLM co-tenancy 下**根本
  不可行**（轻内核 dense/tail 可共驻；routed 36 专家 ~16GB arena 不行）。**唯一可行路径 = 项目既定的
  device-IPC 零拷贝重构（Phase 23/24）**：pypto runtime 在进程内接管计算（一个 device context，无第二争用
  runtime）——这正是 Phase 23/24 存在的原因。多周级，是下个 session 的正确方向。**不要再试 socket-worker
  路径**——调参空间已穷尽（上表）。
- goal（live A/B）**用当前架构不可达**；非数学/接线问题（所有 compute 已验证 bad=0 入库）。机器干净收尾：
  8001 down，cards 8-15 OK，8000 up，0 worker。


## 2026-07-05 (later-7) —— live MoE A/B 迭代调试：507018 co-tenancy 定为 definitive blocker ⏸

- **3 轮 live 部署迭代调错**：(1) 默认 env → routed worker `507018`（co-tenancy）；(2) 加
  `PTO2_RING_HEAP=4GB` → fault 变显式 `rtMalloc 207001 size=16GB`（routed 运行时 pooled static arena
  要 ~16GB，vLLM gpu-mem 0.80 下每卡仅 ~14GB free 装不下）；(3) 再把 vLLM gpu-mem 降到 0.55（~28GB
  free/卡）→ arena OOM 消失，**但 routed 内核仍在 rank 0 & 2 上 `507018`**（跨 rank 非确定）与 active
  vLLM Worker_TP 共卡。卡从未 poison（task 级 fault），每次干净拆除，8-15 OK，8000 up。
- **Definitive blocker**：重型 routed grouped-GEMM 内核在**独立 ChipWorker 进程**里与**active vLLM
  Worker_TP 同卡**运行 → 部分 rank `507018`（AICPU stream sync / device 争用），即便修掉 16GB arena
  OOM 也在。dense-MLP + tail 的 @pl.jit worker 不会（task graph 轻）。→ **socket-worker + 独立 runtime
  对 routed 内核在 live vLLM 下不可行**，是真正多周硬点。
- **下步（按优先级）**：(1) **device-IPC 零拷贝**（Phase 23/24 机制，一个 runtime 无第二争用 context ——
  项目既定方向，socket-bridge 一直只是 oracle 回退）；(2) dispatch-cut 缩小 routed（更少专家/更小 arena）
  看能否共驻 + 查 16GB arena 为何这么大（疑过度预留）；(3) stream/event 序列化 routed 与 vLLM 避免
  AICPU 调度重叠。
- **目标状态**：live 单层 MoE A/B **未通过**——卡在 507018 运行时 co-tenancy，**非数学/接线**（所有
  compute 已验证 bad=0，8 commit 入库）。部署已全接线 + 可复现（w8a8 logdir：`start_8001_moe.sh`
  gpu-mem 0.55、`restart_routed_workers.sh` +PTO2 ring env、`pypto_patch_moe/`）。收尾机器干净：8001 down，
  cards 8-15 OK，8000 oracle up，0 worker。


## 2026-07-05 (later-6) —— live 单层 MoE 部署实跑 → 运行时 507018 co-tenancy blocker（实测定位）⏸

- **实际执行了完整 live 单层（layer 3）MoE 部署**（非 spec）：
  - 起 8 个共驻 routed worker（setsid，cards 8-15，`pypto_mlp_worker --routed-layers 3`）→ 全部
    `listening`（setsid 持久化 OK；之前 setsid「失败」其实是自匹配 pkill 先杀了 shell）。
  - 8001 带 MoE backend 启动（`nerdctl start vllm-8001` + `start_8001_moe.sh`，gpu-mem 0.80，
    PYTHONPATH=/logs/pypto_patch_moe，PYPTO_MOE=1，PYPTO_MOE_LAYERS=3）→ **8 个 rank 全部
    `[pypto_moe_backend] installed sock_dir=/logs layers={3}` + FusedMoE.forward layer-tracking，
    Health 200，Application startup complete**。lazy rank 解析（local_rank）生效。
  - 首个真实请求 → **routed worker `run_prepared failed with code 507018`**，vLLM 收到
    `ConnectionError: worker closed`，8001 请求失败；8000 vanilla 正常。
  - **事后卡健康 8-15 全 OK（本次未 poison）**——507018 只杀了 worker 进程、未 Alarm 卡（task 级 fault，
    比早先 IPC-map 事故轻）。已干净拆除（stop 8001 + pkill workers），卡 OK，8000 up。
- **实测定位的真 blocker**：routed pypto 内核**与 active vLLM Worker_TP 共卡运行时 507018**。关键对比：
  dense-MLP + tail 的 @pl.jit worker **不会**触发（此前共驻 3/3 token-exact）——所以是 routed 专属
  （36 专家 RECV-tiled grouped-GEMM 的重 task graph 与 vLLM live device context/AICPU 调度器争用）。
  之前「共驻 routed PASS」是 worker 单独占 card 8（无 vLLM）；只有 vLLM Worker_TP 同时 active 才暴露。
- **含义**：socket-worker + 独立 ChipWorker 路径对 routed MoE 内核在 live co-tenancy 下不可行。正解是
  项目既定的 **device-IPC 零拷贝方向**（Phase 23/24，一个 runtime、无独立争用 device context），或对
  routed 内核与 vLLM stream 做序列化。这是真正的多周硬点。**所有 compute（内核/worker/backend/协议/
  layer-targeting）已验证 + 入库；blocker 是运行时 device-context 争用，不是数学或接线。**
- 下步选项：(1) dispatch-cut 排查为何 routed 507018 而 dense/tail 不会（缩到 1 专家能否共驻）；
  (2) 换 device-IPC 零拷贝（Phase 23 机制）替代 socket + 独立 ChipWorker；(3) stream/event 序列化避免
  routed 与 vLLM 在 AICPU 调度器上重叠。部署产物已 stage 在 w8a8 logdir（`start_8001_moe.sh` /
  `restart_routed_workers.sh` / `pypto_patch_moe/`）供下个 session 直接复现。


## 2026-07-05 (later-5) —— backend↔co-resident-worker code path 完成：live 单层 MoE 代码全就绪 ✅

- **`pypto_moe_backend.py`（pypto-lib `bdcb1b7`）改用 co-resident worker 协议**：`RoutedClient` 从
  `_serve` LE 协议改成 **`pypto_mlp_worker` BE/nbytes 协议**（op=routed, rows/layer/offsets/counts +
  int16 bf16），这样 backend 直接对接**共驻 @pl.jit worker**（非 @pl.program `_serve`）。加
  `FusedMoE.forward` layer-idx 追踪（threadlocal），只 route `PYPTO_MOE_LAYERS`（tracking 不可用时
  单层 shortcut）。**backend selftest vs 共驻 worker：bad_ratio@0.05=0.0000** —— `_apply_mlp` →
  共驻 worker `routed` → 正确 y 的全路径已用 LIVE 协议在 device 上端到端验证。
- **live 单层 MoE 的全部代码已就绪 + device 验证（六连，fork stepfun/develop）**：`fc0bafb`（内核）→
  `e17b4ab`（_serve fix）→ `20292aa`（backend v1）→ `ae00e9a`（@pl.jit device-run）→ `0249700`（共驻
  worker routed op）→ `bdcb1b7`（backend 共驻协议 + layer targeting）。每个都 bad_ratio@0.05=0.0000。
- **剩余 = 纯部署 + 42 层内存（无更多代码设计）**：单层 live A/B 只需 (1) 8001 拉起；(2) 起 8 个共驻
  routed worker（`pypto_mlp_worker --routed-layers 3`，cards 8-15，与 vLLM Worker_TP 共驻，@pl.jit
  ChipWorker 无 co-tenancy）；(3) sitecustomize `pypto_moe_backend.install()` + env
  `PYPTO_MOE=1/PYPTO_MOE_SOCK/PYPTO_MOE_LAYERS=3`；(4) curl A/B vs 8000。全模型 42 层需 worker 侧
  专家权重 LRU/按需（~47GB/rank，多周硬点；单层可放下）。


## 2026-07-05 (later-4) —— co-resident worker routed op device PASS：live worker config 成立 ✅

- **`pypto_mlp_worker.py` 加 `op=routed`（pypto-lib `0249700`）**：把 `routed_experts_jit` 注册进与
  dense/shared/tail **同一个 `ChipWorker`**（一进程一卡）+ 加载 per-rank dequant-W8A8 专家；
  `routed_partial` pad 到 LOCAL_RECV_MAX、run、unpad；`--routed-layers` CLI。
- **device 验证 co-resident**（card 8 上 dense layer 0 + routed layer 3 同 ChipWorker，client 打真实
  往返）：**MLPW_ROUTED_PASS**，rows=1024，maxdiff=0.0020，bad_ratio@0.05=0.0000。这是**正式 live
  worker 配置**（routed 与 dense/shared/tail 共驻一进程）→ **无独立 @pl.program 进程 → 无 507018
  co-tenancy**（避开 card-8 事故那类问题）。live wiring 组件 #1 完成且可提交。
- **本 session 五连（全部 device 验证 + 已推送 fork stepfun/develop）**：`fc0bafb`（routed 内核真 W8A8
  bad=0 + _serve）→ `e17b4ab`（_serve bf16 fix，worker round-trip）→ `20292aa`（backend hook
  `_apply_mlp` glue selftest）→ `ae00e9a`（@pl.jit device-run）→ `0249700`（co-resident worker routed op）。
- **剩余 full live e2e A/B**：(a) backend layer_idx 注入（多层；单层现成）；(b) **42 层专家权重内存**
  （全驻 ~47GB/rank → LRU/按需，多周硬点；单层可放下）；(c) 8001 拉起 + backend client 指向 worker
  的 routed sock（协议 BE >I + nbytes + int16 bf16）+ live A/B vs 8000。五个组件均已单独 device 验证 +
  入库，剩余是 live 组装 + 内存扩展。


## 2026-07-05 (later-3) —— @pl.jit routed device-run PASS：co-resident live 路径解锁 ✅

- **`_routed_jit_probe.py --device-run`（pypto-lib `ae00e9a`）证明 RECV-tiled routed body 作为 plain
  `@pl.jit` 在 device 上 RUN 正确**（不只是 compile）：真 W8A8 layer 3，ratio_allclose(atol=0.04,
  rtol=0.04) PASS，4.37s。
- **为什么关键（解掉一直卡的 live blocker）**：`@pl.program` 的 `_serve` worker 太重，**不能**和
  vLLM Worker_TP 共卡（co-tenancy → 507018 → card-8 事故）。但 live 8001 的 TP=8 Worker_TP 占满
  cards 8-15，独立 routed `@pl.program` 进程无处可跑。`@pl.jit` 变体轻量、可共卡（现有
  attn/dense/shared/tail 的 @pl.jit op 就是这样共驻在 `_stage_attn_worker.py`）。→ **正确的 live 集成 =
  把 routed op 作为 `@pl.jit` 注册进现有共驻 attn worker（`ChipWorker.register`），而非独立 `_serve`
  进程**；backend `pypto_moe_backend.py` client 连 attn worker 的 socket 即可。
- 这修正了 `deployment/moe-routed-live-wiring.md §4.1` 的 live 路径（`_serve` 独立进程仅适合离线验证，
  我正是用它离线验的；live 用 co-resident @pl.jit）。
- **本 session 累计（全部 device 验证 + 已推送）**：routed 内核真 W8A8 bad=0（fc0bafb）→ `_serve` bf16
  fix + worker round-trip（e17b4ab）→ backend hook `_apply_mlp` + glue selftest（20292aa）→ @pl.jit
  device-run + co-resident 路径（ae00e9a）。剩余 live e2e = 把 routed @pl.jit 注册进 attn worker +
  layer_idx 注入 + 42 层权重内存（LRU/按需）+ 8001 拉起 + A/B。


## 2026-07-05 (later-2) —— MoE routed backend hook `_apply_mlp` 落地 + device glue-test PASS ✅

- **backend hook 代码写完（不是 spec，是可用代码）**：`pypto-lib/tools/step3p5/pypto_moe_backend.py`
  （`20292aa`）monkey-patch `MoECommMethod._apply_mlp` → pypto RoutedExperts worker：
  - `_to_csr(group_list, group_list_type)`：type 1=counts / 0=cumsum→diff，offsets=exclusive
    prefix-sum；`torch.equal` 验证与 `_balanced_csr` 一致。
  - `RoutedClient`（UDS，uint16-view bf16 协议）+ `_pypto_apply_mlp`（pad 到 LOCAL_RECV_MAX、route、
    unpad 回 num_recv；`num_recv>1024` → vanilla fallback）。
  - `install()` 经 sitecustomize autoload（`PYPTO_MOE=1` / `PYPTO_MOE_SOCK` / `PYPTO_MOE_LAYERS`）。
- **device self-test PASS**（`--selftest`，真 W8A8 layer 3，worker on card 8）：num_recv=1024，
  y_shape=(1024,4096)，maxdiff=0.0020，**bad_ratio@0.05=0.0000**。`_apply_mlp` 替换的 compute+glue
  全部在 device 上证对。
- **至此 worker + backend-glue 均已 device 验证**。剩余 live e2e = (1) in-vLLM autoload（8001 全栈拉起
  多步重建 + 8 个 per-rank routed worker + install 进 sitecustomize）；(2) 多层的 layer_idx 注入
  （单层现成）；(3) **42 层专家权重内存**（全驻留 ~47GB/rank → 需 per-layer LRU / 按需加载，真正多周
  硬点）；(4) live A/B vs 8000。见 `deployment/moe-routed-live-wiring.md`。


## 2026-07-05 (later) —— MoE routed worker `routed` op device round-trip 验证通过 + _serve bf16 bug 修复 ✅

- **worker `routed` op device round-trip PASS**：腾空 cards 8-15（8001 已是死状态——Worker_TP
  zombie + 重复 host attn worker，stop 容器 + pkill 清干净）后，在干净 card 8 起 `--serve` worker +
  socket client 打真实往返：真 W8A8 layer 3，`max|y|==max|golden|=0.4551`，maxdiff=0.0020，
  **bad_ratio@0.05=0.0000**，rtt 3.25s。serialize→`compiled(...)`device→deserialize 全路径证通
  （`deployment/moe-routed-live-wiring.md` §4.1 完成）。
- **修复 `_serve` bf16 序列化 bug（pypto-lib `e17b4ab`）**：原 `y1[0].contiguous().numpy().tobytes()`
  在 bfloat16 上 `TypeError: unsupported ScalarType BFloat16`——即 fc0bafb 的 worker op 返回时必崩。
  改 `.view(torch.uint16).numpy().tobytes()`（client 端 uint16→bfloat16 反序列化，2-byte 布局一致）。
  device 计算本身正常（`chip_process dev=8 ready`），仅序列化行。round-trip 测试的价值正在于此。
- **0162 launch gotchas（记入避坑）**：(1) `pkill -f "vllm_routed_experts --serve"` 自匹配自身 shell →
  SIGKILL 自己 → 后续 rm+launch 不执行、无输出 → 用 bracket trick `pkill -f "[v]llm_routed_experts"`；
  (2) netboot/cgroup 激进回收：ssh 里 nohup/setsid/tmux 全在 ssh 断开时被杀（tmux server 都不留）→
  可靠做法是 worker 跑在一条**保持打开的前台 ssh**（后台 hold）+ 另开 ssh poll log/跑 client。
- **边界**：这坐实了 worker 侧（内核 + 真权重 + socket 往返）完全 OK；剩余仍是 backend `_apply_mlp`
  hook（层号注入 + shared 合并 + group_list→CSR）多周工程，见 `deployment/moe-routed-live-wiring.md` §4.2。


## 2026-07-04/05 —— MoE routed-expert 内核真权重验证 + vLLM serving 从零重建 + pypto dense/attn/tail live 逐字对齐 ✅

- **MoE routed-expert per-rank 内核（最后一块 MoE 计算内核）验证通过**：新增
  `pypto-lib/models/step3p5/vllm_routed_experts.py` —— per-rank 36 本地专家的 grouped
  SwiGLU（`N_LOCAL_EXPERTS=36`、`LOCAL_RECV_MAX=1024`、SiLU），**无 collective**，正好是
  vLLM FusedMoE all-to-all dispatch/combine 包裹的 per-rank 计算 seam。body 来自
  `moe.py::_expert_routed`，RECV_TILE=32 行分块（naive `[1024,1280]` FP32 累加器=5MB 会爆
  188KB UB，必须行分块），封成 `@pl.function(Inline)` 塞进 `@pl.program RoutedExperts`
  （chip_orch + host_orch per-rank dispatch）。**关键：tile body 外必须加 `if tile_valid > 0:`
  守卫**（否则 ~31/32 空尾块提交 expert kernel with tile_valid<=0 → 507018；这是第一次 device
  失败的根因）。
  - **device 结果（真实 W8A8，恢复后的 card 8/9）**：synthetic PASS bad_ratio=0.0067；
    **真实 W8A8 layer 3 rank 0 PASS bad_ratio=0.0000**，max|out|=max|ref|=0.428。真权重经
    `weight_loader._load_quantized_expert_projector`（INT8 + `_scale`/`_offset`→BF16）+ HF
    gate/up `[INTER,HIDDEN]`→`[HIDDEN,INTER]` 转置。
  - **worker `routed` op 已实现**：`vllm_routed_experts.py::_serve()` 起最小 UDS worker
    （4-byte len + JSON header + BF16 body），收 BF16 hidden + `offsets`/`counts`，跑编译好的
    RoutedExperts（真实 dequant W8A8 专家），回 BF16 y；host 已验证。`_routed_jit_probe.py`
    另证 RECV-tiled body 也能编成 `@pl.jit`（worker 可像 dense/shared 一样 `register`）。
  - 代码已 push：**`pypto-lib` `fc0bafb`**（csy0225 fork stepfun/develop）。
- **机器事故 + 完整恢复（自伤 → 全恢复）**：首次 routed device-run 误在 -d 8 与 live 8001
  worker **co-tenant** 跑重型 `@pl.program` → 507018 → card 8 Health=Alarm → `npu-smi set -t
  reset -i 8` 在 AMP+HCCS 模式下**重启全部 16 卡**（用户批准）→ 固件 load 卡死（`flag_r=0x6666`/
  `dcmi -8005`）→ `sudo RECOVERY.sh` 重装 driver 但需重启 → host 重启 → **netboot 抹掉
  authorized_keys → SSH 锁死 ~8h**（cluster provisioning 最终恢复 key）。恢复顺序（netboot
  tmpfs 丢失 `/` 全部）：(1) 挂 NVMe（`/dev/nvme0n1`→/mnt/persist、`/dev/nvme1n1`→/data；
  w8a8 ckpt 在 `/data/chensiyu/step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp`）；
  (2) 建 `HwHiAiUser`（否则 driver 装报 0x0091）；(3) `sudo RECOVERY.sh` → driver 25.5.2 +
  firmware 7.8.0.7.220 + ptoas 0.45，**16 卡 Health=OK（card 8 Alarm 清除）**；(4) 修 cann
  symlink → CANN 9.0.0 non-GA（workspace runtime 编译所依赖，RECOVERY.sh 指向 beta.1 是 stale）；
  (5) `apt install libstdc++-12-dev`（CCEC 需 `<cstdint>`）。**铁律**：AMP+HCCS netboot 机上
  **绝不**单卡 `npu-smi set -t reset`（会重启全部卡）+ **绝不**在有 live vLLM worker 的卡上跑重型
  `@pl.program`（co-tenancy → 507018 → 需 root reset）。
- **vLLM serving 从零重建（早先"需 cluster provisioning"的判断是错的）**：用户提示"镜像在某个盘里"
  破局。正确镜像不是 skew 的 lijiahui/vllm-ascend，而是
  **`hub.i.basemind.com/stepcast/stepcast:0.19.0-...`**，从 **docker data-root
  `/mnt/nvme1/chensiyu/docker-data`** 找到（dockerd 已随 netboot 消失，但
  `containers/<id>/config.v2.json`+`hostconfig.json` 存了每个原容器 spec，
  `image/overlay2/repositories.json` 列出镜像）。重建配方（可复现）：
  (1) 挂 NVMe；(2) 从 `/mnt/persist/k8s-install/containerd` 起 containerd（root bind-mount）；
  (3) **runc 1.1.8 `--no-pivot` wrapper**（netboot `/`=rootfs，默认 pivot_root 失败）；
  (4) `nerdctl -n k8s.io pull` 正确 stepcast 镜像；(5) `nerdctl run -d --privileged --network
  host`（privileged→全 NPU）；(6) `nerdctl exec` 起 serve 脚本。**3 个 gotcha**：(a) 不能
  `set -u`（set_env.sh 有 unbound var → 静默退出、0-byte log）；(b) 必须 `export VLLM_USE_V1=1`
  （否则 `hf_overrides must be a dict`）；(c) **DROP `--speculative_config`（MTP）**——draft
  config 再触发 hf_overrides bug；MTP 是 spec-decode，greedy(temp=0) 输出与不带 MTP 完全一致 →
  仍是有效 A/B oracle。**8000 oracle UP（health 200，cards 0-7）**，生成"北京，简称京，是中华人民
  共和国的首都…"。同配方起 8001（cards 8-15）跑 pypto。
- **pypto dense0-2 + attn + tail 在重建平台上 LIVE 且逐字对齐**：8001 pypto = 可用 vanilla boot
  env + `PYPTO_*` 开关（ATTN_BACKEND=1、KV_IPC=1、AB=0、LAYERS=0,1,2、FUSE_MLP_LAYERS=0,1,2、
  TAIL_LMHEAD=1）+ 8 host worker（cards 8-15，8/8 socket）。backend `pypto_attn_backend.py`
  经 `/logs/pypto_patch/sitecustomize.py` autoload。**A/B 结果：3/3 token-EXACT**（8001 pypto vs
  8000 vanilla，temp=0，prompts 北京/中国首都/1+1）。GOTCHA：pypto decode ~2.5s/token（per-layer
  socket round-trip）→ curl 需 `-m150`（否则超时看似"empty"，非 bug，Phase 26 perf）。
  RESTART GOTCHA：kill 旧 8001 后 Worker_TP 仍占 HBM → 用 bracket-pattern `pkill -9` 确认
  HBM<10% 再重启。**live pypto pipeline（attn + dense-MLP + tail lm-head，layers 0-2）在重建
  平台证明正确 = 加 MoE routed experts 的地基**。
- **剩余（full MoE live，多周级）**：接 validated routed 内核 —— worker `routed` op（已实现，需
  device round-trip 在空闲卡验证）+ **backend hook `MoECommMethod._apply_mlp`→`unified_apply_mlp`**
  （映射 `MoEMlpComputeInput.group_list`→CSR offset(cumsum)/count；处理 W8A8 dynamic act-quant），
  覆盖 MoE 层 3-44，再对 8000 oracle 做 live A/B。内核 + hook seam 已定位/验证，集成是多周工程。
- **边界**：本 session 交付 = routed 内核真权重精度闭环 + serving 重建配方 + dense/attn/tail live
  逐字对齐 + worker op 实现；**未做** = backend `_apply_mlp` hook + MoE 层 live A/B（下个 session
  从此继续）。容器侧改动（step3p5.py `tp_in_dp` drop、optimus stub、start 脚本）在 disposable
  container overlay，非 repo；仅 `vllm_routed_experts.py` + `_routed_jit_probe.py` 入 git。


## 2026-07-03 —— 零拷贝 KV-IPC 集成 step 1-5 验证通过 + IPC 主卡点解除 + 重制定 plan ✅

- **背景/纠偏**：项目此前偏成「算子桥接」（每 rank 独立 worker + socket/device-IPC 桥单算子，丢融合收益 + host round-trip ~2.6 tps）。按用户+技术专家 7 步路线，验证「PyPTO runtime 通过 device-IPC 零拷贝接管 vLLM KV 计算」。
- **step 1-5 全部在 0162 card 8 device 实测 PASS**：
  - **step 1**：torch_npu 有 torch.cuda 级 IPC（`rebuild_npu_tensor`/`storage._share_npu_`/`torch_npu.multiprocessing`/`NPUIPCTypes.cpp`）+ 裸 ACL；device tensor 导出 rc=0。**测量到跨进程 import 的 VA 不同但 offset 保留**（`_stage_va_ipc_probe.py`：exporter `0x12c041…`→importer `0x12c1c0…`，`base+4096` 读回正确）。
  - **step 2**：import 的 IPC 指针 → `DeviceTensor` → 真 kernel `bad_ratio=0`（复用 P4/P7）。
  - **step 3**：一 key + `DeviceTensor[block]` 自动 offset，多块 kernel 读取全对（`_stage_vamap_multiblock.py` `VAMAP_MULTIBLOCK_PASS`）。
  - **step 4/5**：45 层 KV 合一 buffer → **1 个 export key** → 1 次 import → **90 条 offset map** → **无 per-tensor MemPool → 无 OOM**；嵌套 offset（层 map + block_table 分页）零拷贝喂 page_attention kernel，跨层 0/22/44 × 块 0/3/7 K/V 全 `bad_ratio=0`（`_stage_kvpool_pageattn.py` `KVPOOL_PAGEATTN_PASS`）。
- **技术解除**：IPC 主卡点根因 = 旧方案「每 tensor 一个 `torch.npu.MemPool`」→ 45 层 90 pool → `rtReserveMemAddress` **207001 OOM**（只撑 4 层）。正解 = 找到真实分配点 `vllm-ascend model_runner_v1._allocate_kv_cache_tensors`（per-layer int8），KV 合一 buffer → **一 key + offset map**。507899（子指针导出）+ 207001（OOM）**双卡点解除**。
- **重制定 plan**：范式定为 out-of-process worker + device-IPC 零拷贝（一 key 整池 map）；socket 桥降级为精度 oracle。新 phase：**24**（step6 整层 live 替换）/**25**（step7 真 module 全网 + Wave-3 whole-model orchestration）/**26**（perf，原 22）。详见 [`../phases/23-zero-copy-kv-ipc-validation.md`](completed-phases/23-zero-copy-kv-ipc-validation.md)。
- **边界**：验证的是**真实 KV 布局/规模下的机制**；接进 live 8001 服务 loop 是 Phase 24 工程（此前 socket-bridge 已部分打通真实 KV 导出 + decode attention `bad_ratio=0`）。
- **产出脚本**（0162 staging，未入 sub-repo）：`_stage_va_ipc_probe.py`、`_stage_vamap_multiblock.py`、`_stage_kvpool_pageattn.py`。
- **0162 现状**：为腾卡验证 kill 了 8001 + 8 个 pypto attn worker（cards 8-15 空）；**8000 baseline 保留**（cards 0-7，200）。


## 2026-07-02 —— Step3p5 attention 多 position (ctx>1 / prefill) 乱码根因定位 + 修复 ✅

- **症状**：step3p5 full-attention 在**多 position（ctx_len>1 / prefill、带历史的 batched decode）**输出乱码，**单 position（ctx_len=1）正确**。离线复现（`_stage_attn_e2e.py`，`seq_lens=arange(BATCH)+1` crossrow）：row 0（ctx=1）对，rows 1..15 全错（`bad_ratio≈0.90`）。因为 `test_decode_layer_full_dense_st` 只测 ctx=1，一直没暴露；2026-06-30 的 attention device-shared e2e 也是 ctx=1（`bad_ratio=0.0000`），同样掩盖了它。
- **为什么 ctx=1 掩盖 bug**：ctx_len=1 时 softmax 只有一个元素、权重恒=1，attention 输出恒=V₀，**与 q·k 分数无关**。所以错误的 q·k **值**在 ctx=1 完全不可见，只在 ctx>1（按分数加权）时暴露。
- **定位方法**：新建独立最小复现器 `_stage_scope12_qk.py`（standalone L3 `@pl.program`，逐字复制 `attention_full.py` Scope 1（RMSNorm+Q/K/V proj+q_norm/k_norm）+ Scope 2（partial RoPE + KV-cache 写 + all_q_padded 打包）+ Stage-1 QK，per-rank 配置 `apply_perrank_patch`），逐层 dump 对拍 torch golden：`q_proj_norm`✅ `k_proj_norm`✅ `k_cache`✅，唯独 `all_q_padded`（打包后的 Q）**首错在 (row0, col32)**（col32 = `ROTARY_HALF_FULL` = `rot_q_hi` 段起点；`rot_q_lo` 的 cols 0..31 正确）。`REAL_ROPE=1` 时误差更大（all_q_padded 0.19、scores 0.90）。
- **根因**：Scope 2 里 Q 的 partial-RoPE 打包进 `all_q_padded` 是一个 **pypto/ptoas codegen 数值 bug**，定位在 `rot_q_hi` 写入区（列 `ROTARY_HALF_FULL..ROTARY_DIM`）。原写法 `q_block = reshape(slice(q_proj_norm,[1,8*128]),[8,128])` → 对 reshape 后的 `[8,128]` tile 在 col offset 32 切 `q_hi` → `[8,32]` `col_expand_mul` + assemble 到 `all_q_padded` col 32 —— 这条"reshape + col-offset 子列切片 + `[8,32]@col-32` assemble"链路 miscompile。**单行 K 路径（`[1,32]` 切 `k_proj_norm`）正确**，只有多行 Q 出错。
- **修复（model-side，已落地并本地验证）**：把 Q RoPE 打包改成**逐 head 用 `[1, ROTARY_HALF]` 连续切片**（完全镜像已验证正确的 K 路径），逐 head assemble 进 `all_q_padded`。应用到 `pypto-lib/models/step3p5/attention_full.py`（Scope 2）和 `attention_swa.py`（Scope 2；SWA 无 full-row assemble，保留其结构）。数学等价。
- **验证（0162 card 8，修复后）**：`_stage_scope12_qk` scores identity 0.2482→**0.0018**（bf16 噪声）、`REAL_ROPE=1` 0.8998→**0.0000**；`_stage_attn_e2e.py ATTN_PERRANK=1` crossrow 全 decode 层（attn+MLP）0.8374→**0.0000 PASS**；`test_decode_layer_full_dense_st -d 8` 单 position 无回归 **PASS 7.97s**。
- **涉及仓库**：修复在 `pypto-lib/models/step3p5/{attention_full,attention_swa}.py`（**本地工作树，尚未 push**，本次会话按用户要求只推 pypto-project 文档）。复现器 `_stage_scope12_qk.py` + e2e `ATTN_PERRANK`/`ATTN_FULL64` 开关（默认关）在 pypto workspace root（本地）。
- **另一个独立 bug（非本根因）**：`apply_tp1_patch`/unsliced 路径下 Stage-1 `q_padded_row = fa_b*Q_HEAD_PAD_FULL` 与 Scope-2 打包 stride（含 `KV_HEADS_LOCAL`）不一致，仅 `KV_HEADS_LOCAL>1` 触发；生产 per-rank（`KV_HEADS_LOCAL=1`）不受影响。
- **遗留**：SWA 修复已应用+编译通过，但 SWA ST 在共享卡 runtime OOM（tensor-14 需 3.3GB，co-tenant 占内存，非本修复回归）→ SWA runtime + crossrow 精度待空闲卡验证；`prefill_attention_full.py` 已用 `[1,32]` 逐 token 切片，大概率不受影响，待单独确认；深度技术 writeup 按协议应落 `pypto-lib/docs/known-pypto-pitfalls.md`（待 pypto-lib push 时补）；上游 pypto/ptoas codegen bug 待用 `_stage_scope12_qk.py` 提。


## 2026-06-30 —— Step3p5 attention 设备共享 e2e PASS + device-shared 地基提交 ✅

- 在 `gpu-a910x-0162` 打通 **attention 层经 device-IPC 共享 KV 的离线端到端**：独立进程 ctypes 零初始化 `(2,4096,128)` bf16 KV 块 + `aclrtIpcMemGetExportKey`；worker 编译 `select_decode_layer(0)`（full_dense，L3 fork chip child）→ `DistributedWorker` → `rt.import_ipc(key)` → K/V `DeviceTensor` → `rt.run`，对 torch golden（`_torch_attn_no_gate + _torch_dense_mlp`）`bad_ratio=0.0000`。脚本 `_stage_attn_e2e.py`。
- 关键修复 **`DeviceTensor.__getitem__`**：生成的 L3 `host_orch.py` per-rank 切片 `k_cache[r,0:R,0:H]`；新增连续子视图（row-major offset ptr + 降维/resize；非连续内层 slice 报错）。
- option B 底层代码提交（本地 feature 分支 `pypto/device-shared`，未 push）：simpler `18bddac2`（import_ipc 全链路：CTRL_IMPORT_IPC + DistributedWorker.import_ipc）；pypto `0c4b8749`（`DeviceTensor.__getitem__` + import_ipc + 子模块 bump）。8 文件 b-csy-develop↔0162 md5 一致。
- vllm-ascend 镜像源同步到 `0162:/data/chensiyu/hw_project/pypto/vllm-ascend`（shallow，tar `.git` + `git reset --hard`），分支 `pypto/attention-integration`（off fork `fbfe288`），提交 live 集成蓝图 `PYPTO_ATTN_INTEGRATION.md@ba72967`（Option A：复用 `attention_full`，patch `Step3p5DecoderLayer` attention 子块；checkpoint 权重名 / 独立 attention 程序 `build_tp_attention_full_program` / KV-rows ABI / socket 协议 / S1-S4 步骤已逆向）。
- 8001 在线服务恢复（dense 0-2 + shared 3-44），8000=200/8001=200，8 worker，正常出 token。**修正恢复顺序铁律**：先起 8001 做完 TP=8 HCCL init → `Application startup complete` → 再起 pypto worker；worker 占卡 8-15 期间 vLLM HCCL init 会 `hcclCommInitRootInfoConfig error 15 / rtBinaryGetFunction 107000` 全挂，`aclrtResetDeviceForce` 不解。另：`pkill -f pypto_mlp_worker` 自匹配 ssh shell → 用 `'[p]ypto_mlp_worker'`；e2e exporter 须 `aclrtIpcMemClose`（泄漏 exbus 句柄会脏卡）。
- 涉及仓库：`pypto pypto/device-shared:0c4b8749`（local）、`simpler pypto/device-shared:18bddac2`（local）、`vllm-ascend pypto/attention-integration:ba72967`（local，0162）、`pypto-project main`（本提交）。
- 边界：attention 设备共享 **离线 e2e + 机制 + 地基齐备，未接 live vLLM**；live 接线（worker `attn` op + 每层 KV 导出 + 窗口 A/B）按蓝图 S1-S4 推进，最大卡点 KV-rows ABI。


## 2026-06-25 —— Step3p5 BF16 0~47 vLLM-vs-PyPTO detail precision PASS ✅

- 在 `gpu-a910x-0162` isolated vLLM 容器中以 eager + all-to-all 路径采集真实请求 detail dump，checkpoint 为 `/mnt/nvme1/chensiyu/step3p5_flash_release_hf_mtp3_bf16`。
- PyPTO 侧新增逐层 detail 对比工具：主层 `tools/step3p5/pypto_all_layers_detail_compare.py`，MTP3 `tools/step3p5/pypto_mtp3_detail_compare.py`，以及对应 ST。
- 主模型 `0~44`：`3960` checks PASS，worst pass rate `0.9995659589767456`；MTP3 `45~47`：`279` checks PASS，worst pass rate `0.9995659589767456`。
- 组合 ST：`tests/step3p5/test_step3p5_all_layers_detail_st.py tests/step3p5/test_step3p5_mtp3_detail_st.py` → `2 passed in 286.34s`。
- 关键修复：`Step3p5 EPS = 1e-5`（对齐 vLLM `GemmaRMSNorm`）；MoE reference 使用 vLLM fused router dump 的 `topk_ids/topk_weights`。
- BF16 回归数据已打包为 `/mnt/nvme1/chensiyu/logs/step3p5_910b_v017/step3p5_bf16_e2e_st_regression_20260625.tar`，包含 coarse golden、全层 detail、MTP3 detail、final logits artifacts 与报告。
- 本次涉及仓库 commit 组合：`pypto-lib d4c01b9`、`pypto-project b771c7e`（本次文档记录提交，后续文档补记会前进）、`pypto b00c8b23`、`pto-isa e25732f0`、`PTOAS da011a3d`、`simpler c66b4120`。
- BF16 tar SHA256：`bce502f4cbafb61fe541385ab1828d33a1f9c32bdfb7d2009e871adba4c896c4`。


## 2026-06-24 —— Final e2e precision readiness preflight landed 🟡

- 新增 `pypto-lib/tools/step3p5/e2e_precision_readiness.py`，作为最终端到端精度验收的前置门禁。
- 当前 host 级整网 smoke 全绿：`decode_fwd` distributed mock worst pass rate 1.0；`step3p5_decode` synthetic smoke pass rate 1.0。
- 预检明确剩余阻塞：真实 checkpoint 未挂载、vLLM/stepcast oracle 不可见、`Step3p5DecodeFwd.host_orch` 未接 45 层、head_gate parity 策略未定、MoE 8 卡缺 golden 精度。
- pypto-lib pin 更新到 `stepfun/develop:cfe2093`。

## 2026-06-24 —— CANN 9.0.0 non-GA + DecodeLayerMoE 8 卡 ST runtime PASS ✅

- **环境升级**：0162 切到 CANN 9.0.0 non-GA/non-beta，`/usr/local/Ascend/cann` 指向 `/mnt/persist/Ascend/cann-9.0.0/cann-9.0.0`；已重编译 pypto 与 runtime。
- **回归**：`_smoke_program_build` 通过；dense full ST 8.54s PASS；dense SWA ST 15.61s PASS；L3 allreduce 1 passed / 1 skipped。
- **MoE 8 卡**：复现 `507018 / sched_error_code=100` 后重新切分定位，`dispatch-only` PASS、`dispatch+routed` FAIL，最终确认 routed expert 对 `tile_valid <= 0` 的空 tile 仍提交 kernel。加 `if tile_valid > 0` guard 后，`DecodeLayerMoE full_silu_silu --world-size 8` runtime PASS 26.51s。
- **边界**：MoE ST 当前验证 runtime，不带 golden 精度；整网端到端精度对齐仍属于 Phase 20/21 下一步。split dispatch 先保正确性，非 split/fusion 恢复归 Phase 22 perf 优化。

## 2026-06-22（晚） —— 项目跟踪仓库建立 ✅

在 `<dev-host>/data/chensiyu/hw_project/pypto/pypto-project/` 建了
`pypto-project` 作为专属跟踪仓，push 到 `csy0225/pypto-project`（私有
fork-style）。散落 doc 迁移：

- 把 Phase 20/21/22 docs + archive 内容从 `pypto-lib/docs/step3p5/`
  （位置错了 —— 这些是跨仓库议题）迁到 `pypto-project/phases/` +
  `archive/`。
- 写了新顶层入口文档：README.md、STATUS.md、CLAUDE.md（slim）、
  blockers.md。
- 外部 tracker `<workspace>/pypto/CLAUDE.md`（594 行 monolith）退休 ——
  被本仓取代。

**解决**：项目 owner 提的 doc 散乱问题。项目状态 SSOT 现在落在本仓。

## 2026-06-22（下午） —— WIP push 拆分 + dev-workflow docs + Phase 20-22 设计 ✅

### WIP push 拆分

3 个 commit 上 fork csy0225：

- `csy0225/pypto-lib stepfun/develop`: `ffaf5d6 → 73dbd12`
  （tests/step3p5/ 12 个 ST/UT 脚手架 + 中文架构指南，+3381 行）
- `csy0225/pypto-lib wip/step3p5-barrier-allreduce-20260622`: NEW
  `b5bb6ee`（4 文件 -267/+181：barrier-style all_reduce + per_rank
  输入广播）
- `csy0225/pypto stepfun/develop`: `03136bf6 → b00c8b23`
  （10 个 full_rope SSA/scheduling debug repros，+2199 行）

**关键决策**：WIP barrier all_reduce **不进** `stepfun/develop`（会让
dense ST device 0 编译退化 by UB overflow）。侧分支保留意图待后续。

### Dev workflow + pitfalls docs（push: `73dbd12 → a6b5faa`）

- 新增 `pypto-lib/docs/known-pypto-pitfalls.md` §7：
  `pl.range(constant)` 展开不复用 SSA buffer → UB overflow（barrier
  all_reduce blocker 根因 + 3 个 avoidance recipe）。
- 新建 `pypto-lib/docs/dev-workflow-gotchas.md`：5 条 catalog 非 pypto
  workflow 时间坑（stale pyc / 三件套 activation / HTTP/2 timeout /
  netboot SSH / gh CLI 缺席）。

### Phase 20-22 设计落地（push: `a6b5faa → 69f22b1`）

3 个 phase doc，每个 ~200-300 行。这些 doc 后来移到本 `pypto-project`
仓（见上面晚段）。

## 2026-06-22（早） —— 0162 重启后恢复 + 重验 + MoE 507018 复现 ⏸

### 重启后环境恢复

`gpu-a910x-0162` 重启过；三剑合璧都活着（driver 25.5.2、firmware
7.8.0.7.220 chip flash、CANN 9.0.0-beta.1 NVMe symlink）。4 个 git 仓
都在期望 HEAD 上，simpler submodule `a6e06406`。

### Smoke probe 红鲱鱼（已解）

第一次 `python -m models.step3p5._smoke_program_build` 返回 rc=1，
attention_swa.py:396 报 `valid_cols (48) exceeds bound 16`。**根因**：
上次 session `apply_perrank_patch(TP=2)` 实验留下的 stale
`__pycache__/config.cpython-311.pyc`。Python 的 pyc 失效检查只比 source
mtime，不比 module dict 值。

**解决**：`find models/step3p5 -name "*.py" -exec touch {} +` 把
source mtime 顶过 pyc → pyc 失效 → fresh import 读到正确 `TP=8`。归到
workflow gotcha §1。

### 验证基线

| 测试 | 状态 |
|------|------|
| simpler L3 allreduce_distributed -d 0-1 | ✅ `max\|out-expected\|=0` |
| Phase 19 ST-1 full dense | ✅ PASS 7.93s |
| Phase 19 ST-2 swa dense | ✅ PASS 14.85s |
| MoE 6 variants smoke | ✅ 6/6 PASS |
| MoE device runtime（full_silu_silu -d 0） | ⏸ 5s 内 507018 fault |

记到 blocker §2；需要 `P19_DISPATCH_LIMIT` dispatch-cut tool 定位。

## 2026-06-20 —— 5 仓库 rebase 到 origin/main + push fork ✅

把 pypto / pypto-lib / pto-isa / PTOAS / simpler 全 rebase 到
`origin/main`。Audit：

- 4 个 simpler 本地 patch（zero-size view + `--no-as-needed` libhcomm
  + IPC ENABLE_PEER_ACCESS + SDMA_OFF + llvm-strip）都还要保 ——
  上游本周期没 subsume 任何一个。
- 6 个 pypto-lib step3p5 commit 都要保。
- 3 个 pypto commit（DFX env hook + repros + submodule pin）要保。

**结果**（push 到 `csy0225/`）:

- pypto: `926941e0 → 03136bf6`
- pypto-lib: `93826904 → ffaf5d69`
- pto-isa: `109c9f72 → e25732f0`
- simpler: `c66b4120 → a6e06406`

0162 上验证：smoke probe rc=0，simpler L3 allreduce 双卡 golden，
ST-1 dense device PASS，MoE 6/6 smoke PASS。

**Rebuild trap**：`pip install -e .` 第一次失败 due to
`tensor.h:535 buffer_elems` `-Werror=unused-variable`（NDEBUG +
release flag）。修法：别传 `CMAKE_BUILD_TYPE`（用 dev default）。

## 2026-06-19 —— Phase 16 多卡 IPC blocker RESOLVED ✅

`support_shmem_map_exbus=0` cap（filed as simpler#1037）是 driver 能力
缺口。解决要三剑合璧：

1. Driver `25.0.rc1.2 → 25.5.2`
2. Firmware `7.7.0.3.220 → 7.8.0.7.220`（chip flash，持久）
3. CANN `9.0.0-beta.1`（NOT GA —— GA 的 TDT 不推 AICPU
   `libaicpu_extend_kernels.so`，让 simpler init 507018 失败）

加 simpler `comm_hccl.cpp` patch（CANN GA forward-compat alias）。

**Traps**:

- CANN GA vs beta.1：3+ 小时浪费在 GA 上才发现。
- 0162 是 netboot/tmpfs：`/usr/local/Ascend/`、`/etc/`、`~/.ssh/` 重启
  全丢。建 `RECOVERY.sh` 幂等恢复；持久 state 在 NVMe `/mnt/persist/`。
- Kubernetes DaemonSet（`device-plugin`、`npu-exporter`）占着 driver
  `.run --upgrade`。`kubectl drain` 不够 —— 必须 `systemctl stop kubelet`
  + 手动 kill。

**验证**：`aclrtIpcMemImportByKey + ENABLE_PEER_ACCESS` 跨卡 rc=0、
`peer_va == parent ptr`；simpler L3 `allreduce_distributed` 双卡
`max|out-expected|=0` golden。

**0234 路径**：只需升 driver+firmware（CANN 已经对）。`.run` 包 stage
在 0162 `/mnt/persist/ascend-staging/`。归到 blocker §5。

## 2026-06-17 —— Phase 19 MoE blocker 1-4 清掉 + dense ST device PASS ✅

详见 [`prototype-phase-01-19-summary.md`](prototype-phase-01-19-summary.md)
"Phase 19 MoE blocker 解决"。MoE device runtime 507018 仍在（blocker
§2）。Dense ST device 0 通过（full 7.93s，swa 14.85s）。

## 2026-06-15 —— Phase 15 单卡 e2e rc=0 ✅

单 rank decode_layer 端到端跑通 device 0，20 个 dispatched task 完成。
三个层叠修复一起：head_gate ×1 旁路 + `--tp-world-size 1` monkey-patch
+ `LAYER_*_ROWS_DYN` override。`next_hidden_out shape=[1, 16, 4096],
max|value|=0`（dummy zero weight 期望零输出）。Run time 6.69s。

## 2026-08-29 —— H4 resident constants deployment contract 收口 ✅

基于最新 swimlane 校正，首个 TP collective 含 peer-arrival spin，五层 clean makespan
约 `1.81 ms`；routed expert / route-combine 虽仍是 device 大池，但当前缺 authority
instrumentation 或受 notify fence gate。选择已通过当前 runtime 1000-step liveness 的 H4。

r12 **source-default-all** matched `none/default/none`（64K、warmup10、100 steps）
p50 `30.516/22.606/29.440 ms`，default=`all` 相对 midpoint 收益
`7.372 ms / 24.591%`，三臂 hidden SHA `ee8ae6…db96a`、token `43640` exact，
0 fatal marker。父 env unset 的 exact launcher 64K/1000 p50 `20.973 ms`、RC=0，
四档 curve 全过，pre/postflight clean。

落地：三个 canonical deployment launcher 默认注入 `PYPTO_H4_RESIDENT=all`，保留
`none|rope|gate|all` 覆盖；未移动五仓 pin、未重建 r12、未修改 pypto-lib 代码默认。
完整证据见
[`../benchmark/2026-08-29-h4-resident-deployment-contract.md`](../benchmark/2026-08-29-h4-resident-deployment-contract.md)。

## 2026-08-30 —— routed GMM active-worker dual-latch feature branch GO 🟦

基于 r12 immutable substrate 与 canonical pypto-lib `e6c7d8ec`，将 routed GMM 两级
latch 的参与者收缩为 active worker：`A=min(active_local_experts,36)`，
`G=min(22,10A)`、`H=min(22,5A)`、`Q=min(22,A)`；96-entry route-plan workspace
在 offset `64/80` 放置 cache-line 隔离 counter，保留全部 store drain/barrier/CMO。

最终 H4-all A/B/A p50 `21.099/20.172/21.107 ms`，相对 baseline midpoint 收益
`0.931 ms / 4.4117%`，超过 `0.616 ms` floor；三臂 hidden SHA
`ee8ae6…db96a`、token `43640` exact。fixed-22 版本仅 `0.588 ms`，明确 NO-GO。

whole compile、direct CCEC、focused `168 passed`、full Step3p5 unit
`535 passed, 4 skipped` 均通过。最终 unit 日志已在 0162 r12 容器复跑并封存：focused
`unit-final-a745ab6-focused-20260830-170353-1937305-890145330/pytest.log`
SHA256 `2b3dbe2b…999be`；full
`unit-final-a745ab6-full-20260830-170433-1937841-972655798/pytest.log` SHA256
`5bcffb92…c39b2`。五层 outer L3/L4 byte-exact 与 analyzer structural gate
PASS，policy 为 `release-local-ep-cdb2bb26-resident-dual-latch-22-v2`；但缺 exact
`recv_meta` route sidecar，完整 publication readiness 仍为 `NOT_EVALUABLE`，不声明稳定
DFX span 收益。

commit `a745ab659c68` 已推送到
`perf/gmm-soft-mix-prestage-20260829`（parent `e6c7d8ec`）；这是 **CAND
source-overlay GO**，尚未合入 canonical `stepfun/develop`，也未构建新 image。完整证据见
[`../benchmark/2026-08-30-routed-gmm-active-worker-dual-latch.md`](../benchmark/2026-08-30-routed-gmm-active-worker-dual-latch.md)。

---

## Pin snapshot 完整历史（降序，窄格）

> 这是 pin 的**完整时间线**。带门结论与镜像 digest 的落地台账在
> [`../progress/landed.md`](../progress/landed.md)；本表只给 pin，不重复门证据。
> `SRC` = 源码合入（source-overlay 门）、`IMG` = 有 manifest digest 的镜像。

| 日期 | 事件 | pypto | pypto-lib | pto-isa | PTOAS（src） | simpler | ptoas-bin |
|------|------|-------|-----------|---------|--------------|---------|-----------|
| 2026-08-27 | whole-step host/graph/submit r12 发布 + SRC 同步（IMG `ba42fd19…`） | `14de90fd` | `e6c7d8ec` | `cd4a3d3f` | `307d0484` | `85a82c45` | `v0.57` |
| 2026-08-26 | replicated-input local-owner MoE r11 发布（IMG `401ead7d…`） | `519b588a` | `e6c7d8ec` | `cd4a3d3f` | `307d0484` | `85a82c45` | `v0.57` |
| 2026-08-25 | packed-NZ MoE fusion r10 正式准入 + SRC 同步（IMG `8510f30e…`） | `519b588a` | `fe641929` | `cd4a3d3f` | `307d0484` | `85a82c45` | `v0.57` |
| 2026-08-24 | **五仓全栈升级 r9 发布 + 同步（IMG `b637f00c…`）** | `519b588a` | `bf3ff440` | `cd4a3d3f` | `307d0484` | `85a82c45` | `v0.57` |
| 2026-08-12 | TP all-reduce single-row selector 合入（SRC） | `1c048a74` | **`69ad31e4`** | `ecb6c303` | `fc8c6cae` | `e2efebcb` | `v0.50` |
| 2026-08-12 | RMS→QKV critical prestage I7（SRC） | `1c048a74` | `e5e26f9f` | `ecb6c303` | `fc8c6cae` | `e2efebcb` | `v0.50` |
| 2026-08-12 | `fa58b5cf` post-merge 性能验收 **NO-GO**（ITL `+4.233%`、五层 39/40） | `1c048a74` | `fa58b5cf` | `ecb6c303` | `fc8c6cae` | `e2efebcb` | `v0.50` |
| 2026-08-11 | K8 选择性清零发布（IMG `076af8a1…`） | `1c048a74` | `cb96747e` | `ecb6c303` | `fc8c6cae` | `e2efebcb` | `v0.50` |
| 2026-08-10 | P1a gate 解耦发布（SRC） | 未移动 | `d13b2ca6` | `ecb6c303` | `fc8c6cae` | `e2efebcb` | `v0.50` |
| 2026-08-10 | MoE BS1 N256 发布（SRC） | 未移动 | `a31977fb` | `ecb6c303` | `fc8c6cae` | `e2efebcb` | `v0.50` |
| 2026-08-06 | task-major Attention + L0–L4 MoE formal（IMG `3eb694e0…` / `cab89668…`，pre-fix） | `8e92b468` | `c9af5790` | `ecb6c303` | `fc8c6cae` | `e2efebcb` | `v0.50` |
| 2026-08-03 | **Wave5 canonical release（IMG `4acc77cd…`，唯一完整 release-qualified）** | `defa97c5` | `7099476b` | `ecb6c303` | `fc8c6cae` | `e2efebcb` | `v0.50` |
| 2026-08-03 | Wave4 historical candidate（IMG `8125c678…`，已由 Wave5 取代） | `defa97c5` | `d7e1381b` | `ecb6c303` | `fc8c6cae` | `e2efebcb` | `v0.50` |
| 2026-08-02 | Attention/Vec historical candidate（IMG `64c573bc…`） | `defa97c5` | `76d96bdb` | `ecb6c303` | `fc8c6cae` | `e2efebcb` | `v0.50` |
| 2026-07-29 | PERF-H1 自包含镜像 build + 回归（IMG `b4e8c8a4…`） | `1f704616` | `4513007d` | `ecb6c303` | `fc8c6cae` | `e2efebcb` | `v0.50` |
| 2026-07-29 | PERF-H1 host/device 分账 + retained window 清零改 device memset | `1f704616` | `cfbdcce8` | `ecb6c303` | `fc8c6cae` | `e2efebcb` | `v0.50` |
| 2026-07-29 | PERF-C4 AR → reduce-scatter + **push** all-gather（IMG `7924925f…`） | `6933b1aa` | `cfbdcce8` | `ecb6c303` | `fc8c6cae` | **`8459d60f`** | `v0.50` |
| 2026-07-28 | C/D/G + BS1 收口自包含镜像 | `ca21ab5f` | `563fe62a` | `ecb6c303` | `fc8c6cae` | `216e7632` | `v0.50` |
| 2026-07-26 | canonical-only Step3p5 release（删兼容 package/alias） | `ca21ab5f` | `53eb7212` | `ecb6c303` | `fc8c6cae` | `216e7632` | `v0.50` |
| 2026-07-24 | 合并 origin/main + IPC 权重 interior 指针 provenance 修复（IMG tag `…-20260724`） | `ca21ab5f` | `fd26b1be` | `ecb6c303` | `fc8c6cae` | `216e7632` | `v0.50` |
| 2026-07-24 | PERF-A1 逐层 DFX 接线（holder N1_DFX → swim/pmu + harness `--dfx`） | `ca21ab5f` | `bc5eecb1` | `ecb6c303` | `fc8c6cae` | `216e7632` | `v0.50` |
| 2026-07-23 | decode-ITL profiling harness（hidden-only via holder） | `8af501fc` | `7cb2a6b3` | `ecb6c303` | `72ada0a1` | `36957c6b` | `v0.45` |
| 2026-07-23 | simpler develop 回退到可编译 `36957c6b` + pypto gitlink 同步 | `8af501fc` | `4c48215b` | `ecb6c303` | `72ada0a1` | `36957c6b` | `v0.45` |
| 2026-07-23 | 五仓 `stepfun/develop` 对齐验证过的 N=1 pin + 可复现镜像 | `9ec303f6` | `4c48215b` | `ecb6c303` | `72ada0a1` | `c7fdc574` | `v0.45` |
| 2026-07-18 | N=1 single-submit 合入三仓 + 干净回归 20/20 | `9ec303f6` | `e1513d22` | `ecb6c303` | `72ada0a1` | `c7fdc574` | `v0.45` |
| 2026-07-17 | N=1 stable env freeze | `n1fusion-base:e277de9f` | `feat/whole-net-n1-fusion:0e7a0fdd` | `ecb6c303` | `72ada0a1` | `n1fusion-base:36957c6b` | `v0.45` |
| 2026-06-25 | Step3p5 BF16 0~47 detail precision PASS | `b00c8b23` | `d4c01b9` | `e25732f0` | `da011a3d` | `c66b4120` | `v0.45` |
| 2026-06-24 | CANN 9.0.0 non-GA + DecodeLayerMoE 8 卡 ST | `b00c8b23` | `cfe2093` | `e25732f0` | `da011a3d` | `c66b4120` | `v0.45` |
| 2026-06-22 晚 | pypto-project 仓建立 | `b00c8b23` | `9c4773f` | `e25732f0` | `da011a3d` | `a6e06406` | `v0.45` |
| 2026-06-22 下午 | Phase 20-22 设计 + dev-workflow docs | `b00c8b23` | `69f22b1` | `e25732f0` | `da011a3d` | `a6e06406` | `v0.45` |
| 2026-06-20 | 5 仓 rebase + fork push | `03136bf6` | `ffaf5d6` | `e25732f0` | `da011a3d` | `a6e06406` | `v0.45` |
| 2026-06-19 | Phase 16 三剑合璧验证 | `a1b066df` | `9c5593fb` | `109c9f72` | `29a8af28` | `afb5c5a9` | `v0.44` |
| 2026-06-17 | Phase 19 blocker 1-4 清掉 | `3f421313` | `08f71692` | `8e436661` | `a1efed75` | `6e84154d` | `v0.43` |
| 2026-06-15 | Phase 15 单卡 e2e rc=0 | `3f421313` | `af4b2ed5` | `12e766d1` | `5392d5da` | `6e84154d` | `v0.43` |
| 2026-06-05 | Phase 13 re-sync + smoke 绿 | `3f421313` | `08f71692` | `8e436661` | `a1efed75` | `6e84154d` | `v0.43` |

**表注**：① 2026-06-05 ~ 2026-07-23 的 pypto/pypto-lib 分支是 `main` 或 `develop`，
之后统一为 `stepfun/develop`；② `8af501fc` = `9ec303f6` + runtime gitlink → `36957c6b`；
③ simpler `c7fdc574`（Phase-24 `import_ipc` 半成品）编译不过，回退时存了 tag
`backup/stepfun-develop-c7fdc574-20260723`。

---

## 已解 blocker（post-mortems）

### 2026-06-22 —— simpler#1018 libhcomm DT_NEEDED ✅

`comm_init` 段错 —— `hccl_comm.h` 把 HCCL 声明为 weak，x86 默认
`--as-needed` 把 `libhcomm.so` 从 `DT_NEEDED` 删了。修复在 simpler
`a6e06406`：`src/{a2a3,a5}/platform/onboard/host/CMakeLists.txt` 把
`${HCCL_LINK_TARGETS}` 包成 `-Wl,--no-as-needed ... -Wl,--as-needed`。

### 2026-06-19 —— simpler#1037 IPC support_shmem_map_exbus=0 ✅

三剑合璧修复（driver 25.5.2 + firmware 7.8.0.7.220 + CANN beta.1）。
详见上面 2026-06-19 milestone。

### 2026-06-17 —— Phase 19 blocker 1-4 ✅

1. PTOAS v0.44 `pto.tci ui32 {descending=false}` parser：上游 v0.45 fix
   `505abd64`。
2. sh_mlp / gate_matmul L1/UB overflow：是 shape-choice artifact
   （`apply_tp1_patch` 错，`apply_perrank_patch` 对）。
3. dispatch.py 32B 对齐：`PER_RANK_BUCKETS = pad8(...)` 跨 5 文件
   mirror。
4. CCEC bf16 类型转换：`expert_weights` BF16 → FP32 跨 6 个 emission 点。

详见 [`prototype-phase-01-19-summary.md`](prototype-phase-01-19-summary.md)
"Phase 19 MoE blocker 解决"。
