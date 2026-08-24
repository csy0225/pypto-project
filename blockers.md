# 活跃 Blocker

**T4 = 只放 open。** 每条 ≤25 行（**含多个独立缺口的 🔴 条目 ≤35 行**），
只写：症状 / 根因一句 / 解除条件 / 链接。
长论证、消融矩阵、campaign 编号一律留在 [`design/performance/`](design/performance/README.md)、
[`benchmark/`](benchmark/) 或 0162，本文件只给指针。

**协议**：blocker **定案或解决**即从本文件删掉 → 到 [`postmortems/`](postmortems/README.md) 建一篇
六段复盘（模板 [`postmortems/TEMPLATE.md`](postmortems/TEMPLATE.md)）+ 更新
[`STATUS.md`](STATUS.md) §8 摘要。**已解 / 已定案的东西不留在本文件** —— 包括"负结论"。

**最后检视：2026-08-24。**

---

## 🔴 ACTIVE — R9-H4-DEPLOY-CONTRACT：发布性能依赖未内置的运行时 env

**症状**：同一 r9 manifest `b637f00c…`、同一 64K/1000 口径，默认启动（unset=`none`）
p50 `27.812 ms`；显式 `PYPTO_H4_RESIDENT=all` 后 p50 `22.253 ms`。后者达到发布目标，
但镜像 Config Env 没有 bake 该变量，代码默认值也是 `none`。

**根因**：H4 把 4 个 RoPE 表 + 4 个 gate-R 常量一次上传并常驻（`99.64 MiB/rank`），
使 `bind.args` p50 `6.461 → 0.063 ms`；它是**部署运行合同**，不是 digest 自带属性。当前只在
ITL/precision/swimlane gate 脚本显式注入 `all`，尚未在正式 serving launcher / manifest 中找到配置。

**解除条件**：① 正式 launcher / manifest 显式设置 `PYPTO_H4_RESIDENT=all`，并在 exact deployment
上重跑 startup contract + 64K ITL；或 ② 修改代码/镜像默认值为 `all`，重新构建、发布并跑完整门。
在此之前，文档必须写“r9 + H4 all = 22.253 ms”，不得写“镜像默认 = 22.253 ms”。

**证据**：[`benchmark/2026-08-24-upgrade-r9-release.md`](benchmark/2026-08-24-upgrade-r9-release.md) §2；
[`postmortems/18`](postmortems/18-upgrade-itl-fixed-cost-runtime-contract.md)。

---

## 🔴 ACTIVE — UPSTREAM-NOTIFY-FENCE：notify 的 cache-invalidate 排在 payload drain 之前

**症状**：`pld.tile.remote_store(...)` 紧接 `pld.system.notify(...)` 时，接收方读到的 payload
部分或全部丢失（受损区恰为 `0`），受损 rank 随时序变化。

**根因（device 已证）**：pypto `MakeNotifyCodegenPTO`（`src/backend/common/pto_ops_distributed.cpp`）
把 `dcci(ENTIRE_DATA_CACHE)`（invalidate-only、无 writeback）排在 payload `TSTORE` 排空之前；
现成的 `pipe_barrier(PIPE_MTE3)` 在 invalidate **之后**，无用。不对称来源：`MakePutCodegenPTO`
给 tput 夹了 `pipe_barrier(PIPE_ALL)`，`MakeRemoteStoreCodegenPTO` 什么都不发。

**最小修复 = 一条 `pipe_barrier(PIPE_ALL)` 插在 `cacheinvalid` 之前**。消融矩阵已闭合
（`PIPE_MTE3` 单独 / `dsb` 单独 / 两者组合 / 纯 MTE3 流量 / 安慰剂后置 —— 全部 `exact=False`；
`PIPE_ALL` 单独 `exact=True` 64/64）⇒ **不能用更便宜的屏障替代**，故代价也压不低：
3 个 site 共 `+1.250 µs/call`、只 Wave2 单点 `+0.405 µs/call`（比 K2a pipe-specific barrier
贵约 18×，**不可外推成免费**）。

**生产暴露面（对账口径，别说成"正在损坏"也别说成"是安全的"）**：`A1_parent` 的 Wave2/Wave3
notify 前导与被证伪形状**逐字节相同**，Wave2 前面正是 `remote_store`；四个候选保护机制
（纯 MTE3 流量 / MTE3 级屏障 / store-loop 自带屏障 / rank 到达 skew）全部已否证
⇒ **生产 Wave2 没有可证明的安全机制，只是当前调度没触发它；是否正在损坏未知。**

**硬约束（现在就生效）**：任何把"payload store 与它自己的 credit"拉近的改动（合并波次、
按 peer 融合 store+notify、单 peer 交换）都进入探针的近确定性失败区间，**必须先落 fence**。
⚠ 删 Wave3 **不**属此类，但**已被生产整网否决**（byte-exact 却 ITL `+1.72 ms/step` ——
`critical_tail` 对"删同步点"结构性失明）；合并 Wave1+Wave2 同属"删同步点"，即使落了 fence
也必须先过整网 ITL 才能计收益，不得按 bench `5.6 µs/call` 记账。

**解除条件**：① 上游在 `pto.cmo.cacheinvalid` 之前补 `pto.barrier <PIPE_ALL>`（= 把 put 路径
已有的那条对齐到 notify 路径，不引入新概念）；② 或在产品 AR 里显式插等效屏障并过 A/B/A + 精度门。

**详情**：[`design/performance/06-upstream-asks.md`](design/performance/06-upstream-asks.md)、
[`design/performance/task-tracking.md`](design/performance/task-tracking.md) 2026-08-11 K5-C 行
（含完整消融矩阵、全部 campaign 编号、两份权威报告 sha256）。

---

## 🔴 ACTIVE — Phase 28 live serving：live prefill + per-layer paged-KV + 3-way HBM

**当前边界**：per-layer KV bridge **已接线并可跑 multi-step**（`a632c42e` hidden-only 集成），
**多步精度 = pypto 与 vanilla vLLM 逐 token 一致**（历史"near-tie/未完全正常"表述作废：harness
的 `DEFAULT_ORACLE_TOKENS[2]=19384` 是过时常量，vanilla 自己 step2 也输出 `6127`）。
co-resident round-trip 已在 0162 打通到 decode ABI（0 次 HCCL 失败、0 次 507018、无 poison）。

**三个仍 open 的缺口**：

1. **live prefill（live-front 的第 4 墙，下一堵）**：真实请求首个 forward 是 prefill
   （`AscendAttentionState.PrefillNoCache`），而 `whole_decode_step3p5` 是 decode-only；
   gate 在 `extract_pypto_decode_plan` 正确 fail-closed 后 EngineCore 退出。
   要端到端出 token 必须先 wire prefill program/bridge 或 prefill KV-fill 路径。
   ⚠ 这条历史上被记作"H4"，与性能线的 `H4`（host `bind.args`）**不是同一件事**，勿混。
2. **live paged-KV / dynamic batch**：需从真实 vLLM paged KV pool 导入 per-layer BF16 slice。
   ⚠ 已知 ABI 约束：vLLM launcher 必须固定 `--max-model-len 4096`，否则 block table 宽 2048
   → flat `32768` ≠ compiled `BLOCK_TABLE_FLAT_DYN=512`。
3. **HBM 两个独立口径，不可混写根因**：① live 3-way（vLLM W8A8 常驻权重 + exporter whole-net
   INT8 IPC 权重 + runtime working set 同存）→ `207001`，需消除重复权重；② standalone
   bs16×每请求 64K（KV pool `22.541` + weight pool `24.857 GiB/卡`，prewarm 前 ~`52,013 MiB/卡`，
   再要 ~16 GiB static arena）→ `207001`，**无 live 重复权重**。

**另需**：独立 live-front A/B + 同代 Main→MTP absolute token/hidden oracle 闭环。
**未推送**：`feat/vllm-live-front-wiring` 3 个 fix（`a9573180`/`c9af2a6a`/`d35a71bf`）待授权 push；
NO_HCCL patch 只在镜像内重建（从 `878f3742`），未 bake 进发布镜像。

**解除条件**：live prefill 接线 + device 验证 → live paged-KV/dynamic batch → 解决重复权重与
live HBM 预算 → 独立 live-front A/B + MTP absolute gate。详见
[`planning/phases/28-live-integration.md`](planning/phases/28-live-integration.md)、
[`design/vllm-pypto/02-detailed-design.md`](design/vllm-pypto/02-detailed-design.md)。

---

## 🔴 ACTIVE — N1-S-0234：0234 同步 pypto-lib 后 whole-net stall

**症状**：0234 上同步 pypto-lib 后整网 stall。**完整停摆对象未确认**，本条**从未独立复核**。

**根因**：未知。0234 的 driver `25.5.1` / firmware `7.8.0.6.201` 仍低于 Phase 16 三剑合璧
要求（CANN 已是 beta.1，**不要动**），所以也不能排除环境。

**解除条件**：取得 0234 SSH → 核对五仓 pin / runtime / 环境三件套 → 按
[`reference/canonical-test.md`](reference/canonical-test.md) 重跑 canonical，把停摆对象
按 [`postmortems/LESSONS.md`](postmortems/LESSONS.md) §A「下死锁结论前先抬超时」解码清楚。

---

## 🟡 ACTIVE — DEPLOY-REPRO：历史镜像的 dirty 工作树回溯未完成（主路径已解）

**症状**：按记录 pin 全新构建的镜像跑整网 CI 报
`device pointer 0x… is not a live allocation on worker 0 (… or an interior pointer)`，
而同样 pin 的 0728 candidate 镜像能跑。**pin 相同 ≠ 内容相同。**

**根因**：0728 三个 candidate 都带一份**未提交**的 simpler 补丁（span-aware child provenance）。
完整分析见 [`postmortems/14`](postmortems/14-image-dirty-worktree-unreproducible-pins.md)。

**已解**：补丁逐字节入库为 `csy0225/simpler@8459d60f`，由 `csy0225/pypto@6933b1aa` 的 runtime
submodule gitlink 固定；`img_regress.sh` 增加 `IMAGE_WORKTREE_CLEAN_AUDIT`（五仓逐个查 dirty）。

**仍 open**：
1. 本地 `workspace/pypto/runtime` 还有一版**更新的** WIP（给 `exact_malloc_live` 加 `malloc_size`
   上界，+119 行单测），**设备上从未验证**、未入库 —— 前进 pin 前必须独立验证，否则重复制造同类问题。
2. 0728 三个 candidate 的既有结论（C/D/G 的 BS1/2/16、N=256、Main 8-step）都建立在未入库补丁上，
   需在 `8459d60f` 基线上复核后才能作可复现证据引用。
3. 其余四仓是否也曾以 dirty 工作树参与历史镜像构建，尚未回溯。

**解除条件**：①②③ 全部完成。

---

## 🟡 Phase 20 production backend 未接入（功能）

**症状**：BF16/W8A8 decode 与 W8A8 prefill 的结论来自 vLLM eager detail dump + PyPTO
reference/detail/final-logits replay —— 证明数值路径可对齐，**不是** production backend。

**根因**：`Step3p5DecodeFwd`/prefill runner、`Step3p5Model.forward` monkey-patch、runtime
weight bundle 注入、KV/block_table/slot_mapping ABI 尚未接成在线请求路径。

**解除条件**：① `config_align.py` 校验 vLLM `hf_config` vs PyPTO constants；
② `weight_translate.py` 支持 vLLM module → PyPTO bundle；③ runner 接入 vLLM 请求路径
（decode-only 能返回 token）；④ 在线 L1/L2/L3 精度 gate 通过。
详见 [`design/vllm-pypto/`](design/vllm-pypto/README.md)。**Owner**：未指派。

---

## 🟡 Prefill MoE L1 overflow（TASK-29）

**症状**：`models/step3p5/prefill_moe.py` 编译时 `moe_gate_up` L1 buffer overflow（~5 MB > 限），
prefill MoE 层编译不过。

**根因**：prefill 跑在宽 SEQ 维（如 4096 vs decode `BATCH=16`），decode UB 装得下的 MoE 结构
到 prefill 爆 L1。

**解除条件**：重设计 `prefill_moe`，加 multi-step `gate_up` chunking（~1–2 周）。
**decode-only 绕路**：合成数据预填 KV 到目标 length，跳 prefill 测 decode-only TPS/ITL
（见 [`archive/completed-phases/22-perf-baseline.md`](archive/completed-phases/22-perf-baseline.md)）。
**Owner**：未指派。

---

## 🟡 head_gate 剩余：L1 A/B 暴露的整网 MoE NaN

**已解部分**：`matmul_acc N=16` codegen bug 已修，on-device head-gate 已在 `attention_full/swa`
Scope 1.f 恢复（`gate_r` 承载 layer-independent block-diag R）——
见 [`postmortems/09`](postmortems/09-attention-multiposition-corruption.md)。

**剩余**：L1 ctx=1 A/B（tid 6127 → 期望 303）曾 `logits=nan`；bisect 定位 NaN 在 42 层
INT8 W8A8 routed-MoE（单层即复现），**不是 attention** —— 属 gap-5 territory，
见 [`postmortems/10`](postmortems/10-gap5-attention-quant-scope.md)。

**解除条件**：per-op MoE dump 定位首个 NaN 算子 → 修 → 重跑 L1。**Owner**：TASK-L 上游。

---

## 🔴 Final e2e precision prerequisites

**gate**：最终验收"端到端精度正确且无阻塞"。预检
`python tools/step3p5/e2e_precision_readiness.py --batch 2`（host smoke 全绿）。

**剩余前置**：① 真实权重目录挂载；② vLLM/stepcast 原生 Step3p5 代码可见；
③ live backend 接入（见上面 Phase 20 与 Phase 28 两条）；④ head_gate vLLM parity 策略。

**解除条件**：真实权重 + vLLM oracle 可见 + `decode_fwd` 接线完成 +
8 rank logits shard concat 对齐 vLLM top-k。

---

## 🟢 Deferred — MTP 集成进 decode

3 个 MTP 层有 kernel（`models/step3p5/mtp.py`）但没拼进 decode。收益是 speculative decoding
吞吐倍率，**不在当前关键路径**。gate 在 perf baseline 之后 2–4 周。**Owner**：未指派。

---

## 怎么加新 blocker

1. 按严重度插入（🔴 Critical / 🟡 功能 / 🟢 Deferred）。
2. 写 **症状 / 根因一句 / 解除条件 / 链接**，**上限 25 行**（多缺口 🔴 条目 35 行）。
   超了就把论证下沉到 `design/` 或 `benchmark/`，这里只留指针。
3. 在 [`STATUS.md`](STATUS.md) §8 摘要表加一行。
4. **定案或解决 → 删本节 + 建 [`postmortems/`](postmortems/README.md) 复盘**（负结论也一样），
   并把可复用的教训提炼进 [`postmortems/LESSONS.md`](postmortems/LESSONS.md)。
