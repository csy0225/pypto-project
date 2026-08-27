# 由 swimlane 推导的下一批优化（2026-08-21）

> **2026-08-27 结果更新（优先于下方立项时快照）**：H4/H5 后，H6 以
> pypto `14de90fd` 落地并进入 r12。r11 source-overlay A/B/A 的 ITL
> `21.6805→21.115 ms`（`−2.608%`）、graph build `−44.429%`、
> graph→first runner `−47.936%`、serial rank submit envelope `−23.887%`；
> 正式门仍是 serial 8-rank independent submit，不是 native group-submit。
> r12 immutable correctness/security/final contract `1844/1844` PASS，但未重采
> r12 immutable 性能 A/B/A。`bind.args` 已降至候选 ITL 的 `0.259%` 且
> `no_clear_change`，继续优化明确 NO-GO。镜像仍未 bake H4 env。证据见
> [`../../benchmark/2026-08-27-whole-step-host-graph-submit-r12-release.md`](../../benchmark/2026-08-27-whole-step-host-graph-submit-r12-release.md)。
> 下文保留 2026-08-21 的立项推导与当时尚未验证的假设，不能覆盖该结果。
>
> **触发**：R6-R9 dispatch 域小算子融合线整体关闭后，重新用 swimlane 定方向。
> **方法论基准**：上游 `pypto:docs/zh/user/performance/`（`origin/main@5b15048e8`，
> 8 篇：`index / 00-swimlane / 01-task-granularity / 02-runtime-overhead /
> 03-dependencies / 04-incore / 05-memory / 06-host`）。
> **本文只做排序与否决，不改代码。** 每条候选都给「ROI 上界 vs A/B/A 检测地板」的对账。

---

## 0. 结论（按该做的顺序）

| # | 候选 | ROI 上界 | vs 地板 `0.616 ms` | 状态 |
|---|------|---------:|---:|------|
| **1** | **H4 step-invariant constants resident** | 实测 `5.559 ms` | **9.0×** | ✅ r9 落地；deployment env 待接 |
| **2** | **观测性修复**：恢复 8/8 rank chip swimlane + outer admission | 非性能项，是 3/4 的前置 | — | ✅ r9 收口 |
| **3** | **H6 prepared TaskArgs cache / graph-submit 收口** | 实测 ITL `0.5655 ms`；graph build `1.8183 ms` | ITL `0.92×`，紧 bracket 过门 | ✅ r12 落地 |
| **4** | 继续优化 `bind.args` | 当前仅 `0.0547 ms` / ITL `0.259%` | 低于地板一个数量级 | ⛔ NO-GO |
| **5** | K10 去掉剩余一次阻塞 host control round | `0.45–0.53 ms` | `0.73–0.86×` ⚠ 需紧 bracket | ⬜ 已登记 |
| **6** | MoE 阶段预算（`E0→E7` 269 → <200 µs） | `2.90–3.02 ms` | `4.7–4.9×` | ⛔ gated（见 §4） |

**一句话**：2026-08-21 的“host 优先”判断已兑现，但机制分成两步：H4 回收静态大参数
H2D/bind，H6 回收 TaskArgs signature、graph build 与 rank submit envelope。当前
`bind.args` 只剩 `0.259%`，不再是主路径；下方“23%”仅是立项时代的历史输入。

---

## 1. swimlane 读数（唯一可分析 rank）

来源：[`../../benchmark/2026-08-11-k8-selective-window-zeroing-image.md`](../../benchmark/2026-08-11-k8-selective-window-zeroing-image.md) §5.2
（L0–L4 focused、BS1、`context_len=65536`、cards 0-7、插桩 run）。

```text
rank2/d0: tasks 150 | happens-before edges 229
makespan               2.204 ms
static CPM path        1.825 ms (82.8%) over 87 tasks
observed critical path 103 tasks
  compute 1.788 ms (81.2%)
  stall   0.415 ms (18.8%)   —— 全部 data-wait，front-gap 0.000 ms
tiling check exact (110182 ticks)
```

**必须先定 LOW-WAIT rank**：同一次 8 卡采集 makespan 跨 rank 差 **275×**
（rank2 `2.204 ms` vs rank5 `609.764 ms`），其余七个 rank 的 `tp_all_reduce`
占 99.4%+ —— 那是自旋吸收 rank skew，不是算力。

### 1.1 三个数字各自对应方法论的哪一章

| 读数 | 方法论归属 | 含义 |
|------|-----------|------|
| `front-gap = 0.000 ms` | `01-task-granularity` + `02-runtime-overhead` | **关键路径上没有取件延迟**。方法论的取件代价约 `0.8 µs/切换`，这里是 0 |
| `stall 0.415 ms` 全 data-wait | `03-dependencies` | 空闲是依赖图规定的（"没有就绪活"），按方法论**不算调度开销** |
| `compute 1.788 ms (81.2%)` | `04-incore` | 唯一大头是 kernel 本体 |
| `static CPM 1.825 ms (82.8%)` | 图结构下界 | 完美调度也只能回收 `2.204 − 1.825 = 0.379 ms (17.2%)` |

`00-swimlane.md:142-143` 把真假问题分开的两条定义正好适用：
「**没有就绪活**的空闲核**不算**开销」「**有就绪但未派发**的活时才算」。
本 run 属前者。

---

## 2. 这份 swimlane 证伪了什么（否决清单）

### 2.1 可复用的否决判据（本轮新得）

> **若关键路径 `front-gap ≈ 0` 且 stall 100% 是 data-wait，则
> `01-task-granularity` + `02-runtime-overhead` 两章在关键路径上的 ROI 上界 = 0。**

这条判据**本可在 6 天 / 357 个 run 之前就否决 dispatch 融合线** —— 它和
[`07-hardware-scheduler-performance.md`](07-hardware-scheduler-performance.md) §9
的 `orch` vs `device_wall` 判据（`orch 17279 → 4443 µs` 而 `device_wall 17467 → 17910 µs`）
是**同一结论的两个独立证据**，只是后者晚了 6 天且花了整机锁。

### 2.2 据此明确不做

| 不做 | 理由 |
|------|------|
| dispatch 域小算子融合（R6-R9 及后继） | 已关闭；front-gap=0 是其结构原因 |
| **「任务图瘦身」以省派发为动机** | `1086-hop 关键路径` 本身不是成本，派发才是，而派发 = 0 |
| 在关键路径上继续加 `allow_early_resolve` | I7 已把 RMS→QKV residual 压到 `2.64 µs`；front-gap=0 说明这条线已到底 |
| 为省派发次数而改 SPMD / mix kernel | 同上。mix kernel 若为省 GM 往返（`04-incore`）另算，但动机不能写"省一次派发" |
| dense 前缀（L0–L2）单 family 微调 | 见 §3：3 层 / 48 层，且单 family ≤ 4.4%，落在地板下 |

⚠ **一个必须区分的例外**：`878/1542 个 block_num=1 task`（[`README.md:141`](README.md)）
**不是**派发问题，是**占用率**问题 —— 48 AIV 只用 1 个会让**条本身变宽**，属
`04-incore` / `01` 的「太粗」分支，**不被 front-gap=0 否决**。但该杠杆已有一个负样本：
C5（给 AR 补并行扇出）设备实测 NO-GO。

---

## 3. 5 层 profile 不能直接当整网权重（重要修正）

按 kernel family 占 makespan 比例，Top-15 合计 `1.032 ms = 46.8%`，
**除 `tp_all_reduce` 15.3% 外没有任何 family 超过 4.4%**：

| family | compute ms | % makespan |
|---|---:|---:|
| `tp_all_reduce` | 0.336 | **15.3%** |
| `swa_chip_orch_dense_gate_up_matmul_tp` | 0.098 | 4.4% |
| `swa_q_proj` | 0.089 | 4.0% |
| `swa_out_proj_matmul` | 0.072 | 3.3% |
| …（其余 11 项 ≤ 2.3%，尾部各 < 1.2%） | | |

**但层混比不同，不能按份额直接外推整网**：

| | L0–L4 focused | 真实模型（Main 45 + MTP 3） |
|---|---|---|
| dense 层 | 3 / 5 = **60%** | 3 / 48 = **6.3%** |
| MoE 层 | 2 / 5 = **40%** | 42 / 48 = **87.5%** |

**推论（两条方向相反，都要记）**：

1. **dense 前缀（L0 full_dense + L1/L2 swa_dense）在 5 层图里被放大约 9×。**
   但反过来说，5 层图**恰好包含了真实模型全部 3 个 dense 层** ⇒ 那些 dense family
   的**绝对**贡献已接近整网真值，是一次性前缀成本，**不随层数放大** ⇒ 更不值得投。
2. **MoE family 在 5 层图里被缩小约 21×。** `expert_gate_up + expert_down +
   combine_wait` 2 层合计 `0.156 ms`，×21 ≈ **`3.3 ms`** 整网量级 ⇒ **device 侧唯一
   还够大的池子在 MoE**，与 vLLM 阶段对齐报告独立吻合（见 §4）。

⇒ **不要因为 5 层 profile "很平" 就宣布 device 侧没得做**；要按整网层混比重新加权。

---

## 4. device 侧还够大的只剩 MoE 阶段预算，但已 gated

来源：[`../../benchmark/2026-08-12-vllm-ascend-decode-moe-trace-gap.md`](../../benchmark/2026-08-12-vllm-ascend-decode-moe-trace-gap.md) §13.5 / §14.1。

| 阶段 | 当前 | 目标 | 整网 42 层 ROI |
|---|---:|---:|---:|
| 完整 MoE `E0→E7` | `269 µs` | `<200 µs` | **`2.90–3.02 ms`** |
| 其中 GMM1+act+requant | `77.3 µs` | `≤52 µs` | `1.06 ms` |
| 其中 EP return/restore | `46.6 µs` | `≤30 µs` | `0.70 ms` |

**为什么现在不能动**（三条，全部来自现有记录）：

1. **§13.5 执行顺序第 1 步未做** —— 「先落地 E0–E7 全 rank instrumentation 和
   authority harness」。在此之前只能写"有描述性局部收益"。
2. **最高优先级那项需要上游 compiler primitive**，且本地已 **0-for-4**：
   两个 GMM1 fusion 候选整网 A/B/A NO-GO；R4 静态 NO-GO（16B FP32 Vec row 不满足
   32B 合同）；R8 三次固定镜像 compile 失败。
3. **`tp_all_reduce` 虽是单项最大（15.3%）但模型侧已饱和**：算法已到下界
   （`224 KB/卡 = 2(P−1)/P × N`）；K11 已落地 `−1.120 ms`；K2a / C5 / K5-C 实测 NO-GO；
   K2b 需上游 pypto 补丁（`pld.system.notify` 无 fence 参数）；且
   `UPSTREAM-NOTIFY-FENCE` 这条 correctness blocker gate 住"一切把 payload store
   与自己 credit 拉近"的 AR 优化。

---

## 5. 为什么第一顺位是 host（Amdahl 已经翻转）

方法论 [`index.md:52`](https://github.com/hw-native-sys/pypto/blob/main/docs/zh/user/performance/index.md)
与 `06-host.md:7` 都把 host 排在**最前**：
「**这件事要先查，不是最后查**」「如果时间在 host 上，00–05 页里没有任何东西能动它」。
本项目是**最后**才（且是作为副产品）查到的。

### 5.1 数字

| span | p50 | 占 ITL |
|---|---:|---:|
| `simpler_run` | `26.45 ms` | ≈ ITL p50 `26.329 ms` ✅ 对得上 |
| **`bind.args`** | **`6.12 ms`** | **≈ 23%** |
| `runner_run` | ≈ `20.3 ms` | — |

`simpler_run ≈ bind + runner_run`（**加性串行**）。对照臂 `5.87 ms` 同量级 ⇒ 不是偶发。

### 5.2 重心已经翻转过一次，且现在翻回来了

[`README.md`](README.md) 在 H1 落地后写过：
「H1 之前 host 24.9 ms / device 59.9 ms（host 29%）。H1 落地后 host 压到 5.7 ms，
**device 执行首次成为主导项** ⇒ 后续重心从 L3 移到 cross-chip 与 L2/L1。」

**该结论现已过期。** 此后 device 侧被连续优化（ITL `65 → 26.3 ms`，H1 / K8 / K11 /
MoE 线累计），host 侧没动 ⇒ host 份额从 `3.49/65 = 5.4%` 涨回 **23%**。
这是教科书式 Amdahl 漂移：**优化了分母，分子就升上来了。**

### 5.3 ★ 强怀疑：`bind.args` 就是 H2

[`task-tracking.md:313`](task-tracking.md) 的 H2 根因写得很具体：

> 根因 = 生成 `host_orch.py` 把 per-rank 体（**53 常量 slice + 38 `pl.reshape` +
> ~92 `make_tensor_arg`/`add_tensor` + 1 `_submit_chip`**）包在
> `for r__idx_v0 in range(0, world_size, 1)`

`make_tensor_arg` / `add_tensor` **就是参数绑定**。每 step host 要做
`8 rank × (53 + 38 + 92) = 1464` 次 Python 级操作，纯 host CPU、与字节数无关。
**这与 `bind.args = 6.12 ms` 的形状高度一致。**

两者此前从未被联系起来，因为是两个不同探针在两个不同时期测的：
H2 = `submit` 阶段 `3.49 ms`（2026-07-29，bs=16 / ITL 65 ms era）；
`bind.args = 6.12 ms`（2026-08-21，BS1 / ITL 26.3 ms era）。

⇒ **若成立，则"量级最大的新线索"其实早有一份现成设计**，且 H2 的红队复核已经
把收益口径修正对了：「抹平阶梯只值 ~0.4 ms（关键路径是最后一个 rank），
**真收益来自减少 host 工作本身**」。

---

## 6. H4 的实施顺序（每步都有独立可判的产出）

> **结果**：最终没有走“手改生成 `host_orch.py` hoist”路线，而是通过 holder 把
> 4 个 RoPE 表 + 4 个 gate-R 常量一次上传并重绑为 device-resident tensor。
> 下表是立项时的历史分解；正式落地与门以本文顶部 2026-08-27 更新为准。

> 纪律：**先穷举现有 artifact，再考虑占卡**（见
> [`../../postmortems/12-integration-churn-meta.md`](../../postmortems/12-integration-churn-meta.md) 根因 11：
> "为了拿一个数去写新候选，而那个数已经躺在失败 run 的日志里"）。

| 步 | 动作 | 占卡 | 判据 |
|---|------|------|------|
| **H4a** | 从现有 STRACE 归因 `6.12 ms`：per-arg 次数主导 还是 字节/H2D 主导 | **不占卡** | 若与 `active_batch`/`context_len` 无关 ⇒ per-arg 次数主导 ⇒ H2 假设成立 |
| **H4b** | 判定 `bind.args ≡ H2`：核对 `host_orch.py` 的 `make_tensor_arg`/`add_tensor` 调用数 × rank 数 与 span 的关系 | 不占卡（读生成源码） | 数量级对得上即立 H2 为落地件 |
| **H4c** | **cheap gate**：手改生成的 `host_orch.py`，把 loop-invariant 的 53 slice + 38 reshape + 92 make_tensor_arg hoist 出 rank 循环 / 跨 step 缓存 | 半机锁冒烟 | `bind.args` 下降 + `hidden_sha256` byte-exact |
| **H4d** | 若 bind 压不下去：改为 **overlap** —— step N+1 的 bind 与 step N 的 device 执行重叠 | 半机锁 | 从关键路径移除，`simpler_run` 不再 ≈ bind + runner_run |
| **H4e** | 整网 A/B/A + 精度门 | **整机锁** | 见 §7 |

**为什么 H4 风险低**：纯 host 侧 Python，**不碰 device 语义、不碰跨卡同步、不动
`@pl.program` 结构**（铁律 6 得以保持），预期 byte-exact。

**上游边界**：H2 已判定「**v4-flash `decode_fwd.py:774` 完全同形状** ⇒ 属 pypto codegen
通用改进而非 step3p5 缺陷」。⇒ H4c 的手改是**验证**，正式落地可能要走上游 codegen
（可并入 [`06-upstream-asks.md`](06-upstream-asks.md)）。先用手改拿到数，再决定提 issue 的措辞。

---

## 7. 验收口径（不新增，沿用既有）

- **精度**：多步 decode 逐 token vs vanilla（N=128 ≥95%）为准出；单 token
  `argmax==303` 只作首 token 冒烟。byte-exact `hidden_sha256 = 567b206b…` +
  tail token `14371` 作为 host-only 改动的强判据（host 改动不应改变任何比特）。
- **性能**：整机锁三臂 A/B/A，`delta / floor > 1` 且方向一致才记账。
  近期整网 bracket 地板 `0.616 / 0.634 ms`；K8 曾拿到 `0.0195 ms` 紧 bracket
  ⇒ **K10 的 `0.45–0.53 ms` 只有在紧 bracket 下才可判**，必须先看两个 A 臂 half-range。
- **liveness**：概率性失败意味着"跑通一次"不构成门 ⇒ 拉长单轮曝光
  （`--itl-iters 1000`）而非重复整轮。

---

## 8. 明确未确立（不要当结论引用）

1. 历史假设 `bind.args ≡ H2` 没有成为最终落地机制；r9 实测确认主要可回收项是
   8 个静态大参数的每步 H2D/bind，最终用 resident constants 解决。
2. §3 的 `×21` 整网外推是**层混比量级估计**，不是 release 统计量；插桩 run 的绝对
   时间按纪律不可当干净延迟。
3. `front-gap` 是分析器自有术语；本文按"派发侧分量"解读。即使换解读，
   "stall 100% 是 data-wait" 单独也支撑同一结论。
4. §5.2 的 Amdahl 漂移用了两个不同 era / 不同 bs 的 host 数（`3.49` vs `6.12 ms`），
   **不是配对比较**；只用于说明"份额升高"这个方向，不用于记账。
5. r12 没有 immutable 性能 A/B/A；H6 的性能百分比只绑定 r11 digest 上两文件
   source-overlay。commit 标题中的 `parallelize rank submit` 也不等于正式门证明并行/
   native group-submit。

---

## 9. 证据入口

| 内容 | 位置 |
|------|------|
| swimlane rank2 五件 sha256 | [`05-moe-optimization.md`](05-moe-optimization.md) §Candidate merged swimlane |
| swimlane 全文 + `rc=1` 契约失败 | [`../../benchmark/2026-08-11-k8-selective-window-zeroing-image.md`](../../benchmark/2026-08-11-k8-selective-window-zeroing-image.md) §5 |
| `bind.args` / `orch` vs `device_wall` | [`07-hardware-scheduler-performance.md`](07-hardware-scheduler-performance.md) §9；`0162:…/dispatch-orch-decouple-20260821/{FINDINGS.md, analysis-bin/orch_span_stats.py}` |
| H6 whole-step A/B/A 与 r12 准入 | [`../../benchmark/2026-08-27-whole-step-host-graph-submit-r12-release.md`](../../benchmark/2026-08-27-whole-step-host-graph-submit-r12-release.md) |
| MoE 阶段预算 | [`../../benchmark/2026-08-12-vllm-ascend-decode-moe-trace-gap.md`](../../benchmark/2026-08-12-vllm-ascend-decode-moe-trace-gap.md) §13.5 / §14.1 |
| H2 根因 | [`task-tracking.md`](task-tracking.md) H2 行 |
| 上游方法论 | `pypto:docs/zh/user/performance/`（`origin/main@5b15048e8`） |
