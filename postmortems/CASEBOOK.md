# 踩坑案例集（CASEBOOK）

> **按现象查的单点坑档案。** 每条写清 背景 / 现象 / 过程 / 处置（解决还是绕开）。
> 最后更新：2026-08-21。**真正的约束是每条 ≤14 行 + 索引一行一条**（全文 ≤400 行）——
> 一条写到像复盘那么长，说明它该独立成 `NN-*.md`。

## 这一页和别的页什么关系

| 页 | 索引轴 | 什么时候看 | 粒度 |
|---|---|---|---|
| [`LESSONS.md`](LESSONS.md) | **动作**（你正要做 X） | **开工前**扫一遍（铁律 §0 强制） | 一行 |
| **本页 CASEBOOK** | **现象**（你看到了 X） | **撞上了**再查 | 一条 ≤14 行 |
| [`NN-*.md`](README.md) 16 份复盘 | **error signature / 一类根因** | 要完整证据链、消融矩阵、弯路 | 一篇 |

**只收本仓该管的坑**：项目级 / 环境级 / 运维级 / **测量与诊断方法论级**，以及
**仍在生效的绕路台账**。kernel 硬限制与 dev-workflow 坑归 sub-repo（见文末路由表），
本页只留指针 —— 同一个坑不在两个仓维护。

**处置图例**：✅ 已修（根因消除）· 🩹 已绕开（**根因仍在**，绕路是承重的）· ⏸ 未解。
🩹 和 ⏸ 必须写「移除代价 / 复发条件」—— 不写就会有人把承重的绕路当冗余删掉。

---

## 0. 按现象查

| # | 你看到的现象 | 处置 |
|---|---|---|
| [A1](#a1) | `fuser /dev/davinciN` 报卡空闲，但卡上其实有 8 个 worker 各占 54.7 GiB | ✅ |
| [A2](#a2) | swimlane 里单个 all-reduce 记到 `35,530.5 µs`，而 payload 只有 128 KB | ✅ 认知 |
| [A3](#a3) | `orch` p50 降 74%，`device_wall` 反而 `+443 µs` | ✅ 定案 |
| [A4](#a4) | kernel 从 `38.325` 降到 `22.667 µs/call`，但整网 ITL 没动 | ✅ 纪律 |
| [A5](#a5) | A/B/A 结论永远是「统计不可区分」 | ✅ 纪律 |
| [A6](#a6) | 8 卡里 7 个 rank 报一样的错、1 个不一样 | ❌ 已撤回 |
| [A7](#a7) | 整网挂住，看门狗报 timeout，看起来是死锁 | ✅ 判别法 |
| [A8](#a8) | 概率性 liveness 缺陷「跑了 5 轮都没复现」 | ✅ 口径 |
| [A9](#a9) | 149 个 AIV 函数 Vec 用量加起来 24.8× 超 UB 限额，却编译通过 | ✅ 认知 |
| [A10](#a10) | 同一组 pin 重新构建的镜像跑不通，老镜像能跑 | ✅ 主路径 |
| [B1](#b1) | `ptoas compilation failed`，但错误正文是空的 | ✅ 判 flake |
| [B2](#b2) | 按 runtime 打印的 env 名去设，完全不起作用 | 🩹 |
| [B3](#b3) | 设了 `ASCEND_PROCESS_LOG_PATH`，目录建了、文件是空的 | ✅ |
| [B4](#b4) | agent 侧：Bash 工具报 `ENOSPC`，清了 `/tmp` 也没用 | ✅ |
| [C1](#c1) | head-gate `gate_logits` ~20× 偏小 → 整层乱码 | 🩹 |
| [C2](#c2) | full-attn ctx>1 乱码，`rot_q_hi` 列 32..63 损坏 | 🩹 |
| [C3](#c3) | INT8-native routed MoE `bad_ratio≈0.9847`、`max diff ~254`、**无 device fault** | 🩹 |
| [C4](#c4) | 运行时 `errcode 0x800 "UB not aligned"` → `507018`，编译却过了 | 🩹 |
| [C5](#c5) | `SplitIncoreOrch: InCore ScopeStmt found in non-InCore function` | 🩹 |
| [C6](#c6) | `remote_store` 紧接 `notify`，接收方 payload 部分为 `0` | ⏸ |
| [C7](#c7) | 单卡 ST 报 L1/UB overflow（`sh_mlp` 1.66 MB > 512 KB） | ✅ 认知 |
| [C8](#c8) | `fatal error: error in backend: not support bf16 type cast` | 🩹 |
| [C9](#c9) | canonical structural analyzer 恒 `FAIL_CLOSED` | ⏸ |

---

## A. 测量与诊断 —— 会直接生产错误结论的一类

<a id="a1"></a>
### A1. 非 root `fuser` 报卡空闲 ✅

- **背景**：占卡前判断「卡是空的」。
- **现象**：`fuser /dev/davinciN` 无输出 ⇒ 判为 free。实际 8 个 `VLLMWorker_TP` 各占 `54.7 GiB`。
- **过程**：非 root 看不到别人的 fd，`fuser` 静默返回空 —— **空输出与「无人占用」不可区分**。
- **处置**：✅ 改成 `sudo -n fuser` + `npu-smi info -t proc-mem` **双查、fail-closed**
  （任一不可用即视为占用）。每次作业前重查，**不沿用旧 session 的空闲结论**。
- **出处**：[`../reference/execution-host-contract.md`](../reference/execution-host-contract.md)（`vllm-oracle-0724-n256` 容器实测）

<a id="a2"></a>
### A2. collective 的 spin-wait 被记成 kernel compute ✅

- **背景**：看 swimlane 找「谁最慢」。
- **现象**：单个 all-reduce kernel 记到 `35,530.5 µs`，而 payload 只有 128 KB。
- **过程**：按字面读成「搬运慢」，于是去优化 transfer chunk / 降 ring step —— 方向全错。
  同一轮里 AR 的 makespan 占比也曾被记高（`15%`），核对后正确口径是 **48 次 on-path、约 `8.5~9.7%`**。
- **处置**：✅ 认知修正：**AR 主要是吸收 rank skew 的 barrier**，spin-wait 计入 kernel duration。
  它等的是最慢的 rank，降 step 救不了。要么消 skew，要么别把 AR 当搬运优化。
- **出处**：[`../benchmark/2026-08-10-step3p5-p1a-gate-decouple.md`](../benchmark/2026-08-10-step3p5-p1a-gate-decouple.md)、[`../archive/milestones-2026-Q2.md`](../archive/milestones-2026-Q2.md)

<a id="a3"></a>
### A3. orchestrator 从不在关键路径 ✅

- **背景**：想通过融合小算子 / 减 task 数来降 orchestration 开销。
- **现象**：改动让 `orch` p50 从 `17279.28` 降到 `4443.18 µs`（**−74.3%**），
  而 `device_wall` 反而从 `17466.93` 涨到 `17910.32 µs`（**+443.4**）。
- **过程**：6 天 / 357 run 才拿到这张对照表。事后看，**开工前从现有日志读一次
  `orch` vs `device_wall` 的 p50 就能否决整条线**。
- **处置**：✅ 定案为**可量化的否决门**：若 `orch <= device_wall`，orchestrator 不在关键路径，
  整类 orchestration 级改动 **ROI 上界 = 0**。同类判据：若关键路径 `front-gap = 0.000 ms`
  且 stall **100% 为 data-wait**，则 task-granularity 与 runtime-overhead 两类方法论上界也是 0。
- **出处**：[`16`](16-dispatch-fusion-orch-decouple.md) §3、[`../design/performance/09-swimlane-derived-next-optimizations.md`](../design/performance/09-swimlane-derived-next-optimizations.md)

<a id="a4"></a>
### A4. pooled mean 冒充 critical-tail ✅

- **背景**：想用便宜的 focused 测量代替整网 A/B/A 记收益。
- **现象**：focused regular-call kernel duration pooled mean `38.325 → 22.667 µs/call`，
  听起来是 41% 收益；整网 ITL 没有相应变化。
- **过程**：pooled mean 把所有 call 平均，**关键路径只由 critical tail 决定**；
  两个口径不可互换，更不能相加。
- **处置**：✅ 纪律：pooled mean 只能用来判「这个 kernel 本体有没有变快」，
  **不得**写成 strict critical-tail 或完整源码 A/B/A 的结论。
- **出处**：[`../STATUS.md`](../STATUS.md) §3、[`../planning/handoff.md`](../planning/handoff.md)「禁止」段

<a id="a5"></a>
### A5. 单点收益低于 A/B 检测地板 ✅

- **背景**：逐个 kernel 试融合。
- **现象**：A/B/A 结论总是「统计不可区分」。
- **过程**：bs=1 整网 bracket 的检测地板是 `0.616 / 0.634 ms`（K8 曾拿到 `0.0195 ms` 紧 bracket）。
  单个 small-op 融合的上界几乎都在地板以下 —— 测不出来不是没做对，是**问题本身不可判**。
- **处置**：✅ 纪律：立项前先把上界对上地板。低于地板的候选要么凑成 bundle，
  要么换更大的项，**不要单独上 A/B/A**。
- **出处**：[`../STATUS.md`](../STATUS.md) §3、[`16`](16-dispatch-fusion-orch-decouple.md)

<a id="a6"></a>
### A6. 「少数派报告 = 元凶」❌ 已撤回

- **背景**：8 卡故障，`sched_error_code` 分布是「7 个一致 + 1 个例外」。
- **现象**：曾据此判定那个例外 rank 是元凶，并按它去找根因。
- **过程**：**错**。该分布只说明**哪个子系统停了**（orchestrator vs scheduler），
  不说明**哪个 rank 有责任**。按 AICPU pid 计数
  （`grep -ho "AICPU([0-9]*," log | sort | uniq -c`）后发现 8 个 orchestrator **全部**阻塞。
- **处置**：✅ 已在 [`12`](12-integration-churn-meta.md) §3 根因 8 撤回，§6 的旧表述已同步改正。
  少数派报告只读到「子系统」这一层，责任判定必须数 distinct pid。
- **出处**：[`12`](12-integration-churn-meta.md) 根因 8、[`16`](16-dispatch-fusion-orch-decouple.md) §3

<a id="a7"></a>
### A7. 看门狗把「慢」伪装成「死」✅

- **背景**：整网挂住，runtime 报 `SCHEDULER_TIMEOUT`（`sched=100`）。
- **现象**：看起来是死锁，于是直接去找环。
- **过程**：看门狗只知道「超过阈值没进展」，**慢和死在它眼里一样**。
- **处置**：✅ 判别法：先抬 `SIMPLER_SCHEDULER_TIMEOUT_MS`（env-only、零改码）。
  抬了就过 ⇒ 是慢；抬到很大仍挂 ⇒ 才是死。**下死锁结论之前必须先做这一步。**
  ⚠ 配套见 [B2](#b2)：runtime 提示的 env 名不可信。
- **出处**：[`12`](12-integration-churn-meta.md) 根因 10、[`16`](16-dispatch-fusion-orch-decouple.md) §6.5

<a id="a8"></a>
### A8. 概率性缺陷的曝光口径 ✅

- **背景**：验证一个概率性 liveness 缺陷是否已修。
- **现象**：「跑了 5 轮都没复现」被当作已修。
- **过程**：比较单位错了。真正的单位是 **invocation-until-failure**，不是「跑了几轮」。
- **处置**：✅ 口径：用**一轮长跑**（`ITERS` ×10，墙钟只多约 25 s，曝光 ×10），
  而不是「多跑几轮」。另：想拿一个数就再跑一轮 device 之前，**先穷举现有 artifact** ——
  失败的 run 在挂之前已经跑了几百个 invocation，稳态 span 早就够统计。
- **出处**：[`16`](16-dispatch-fusion-orch-decouple.md) §5 / §6.9

<a id="a9"></a>
### A9. UB 限额是 per-kernel-per-core，不是聚合 ✅

- **背景**：估算「这两个 kernel 能不能融合」。
- **现象**：149 个 AIV 函数的 Vec 用量加起来是限额（`188416 B`）的 `24.8×`，却编译通过。
- **过程**：误以为 `188416 B` 是聚合预算，于是得出「早就超了」的错误结论。
  实际每个函数的 offset 都从 0 重新开始。
- **处置**：✅ 认知修正，且**反向推论才是真约束**：kernel **不能**把中间结果留在 UB
  给下一个 kernel 用 ⇒ 「融合」必须**同时装下两份 staging**，这才是融合的真实门槛。
- **出处**：[`../benchmark/2026-08-10-step3p5-p1a-gate-decouple.md`](../benchmark/2026-08-10-step3p5-p1a-gate-decouple.md)（`4676512 B = 24.8×`）

<a id="a10"></a>
### A10. pin 相同 ≠ 内容相同 ✅（主路径）

- **背景**：按记录的 pin 重新构建镜像，复现历史结论。
- **现象**：新镜像整网 CI 报
  `device pointer 0x… is not a live allocation on worker 0`，而同 pin 的老 candidate 镜像能跑。
- **过程**：老镜像里带了一份**未提交**的补丁（span-aware child provenance），
  只活在镜像层里 —— 记录的 pin 复现不出验证环境。
- **处置**：✅ 补丁逐字节入库（`csy0225/simpler@8459d60f`），`img_regress.sh` 增加
  `IMAGE_WORKTREE_CLEAN_AUDIT` 五仓逐个查 dirty。**仍 open** 三项见 blockers DEPLOY-REPRO。
  连带铁律：**source-overlay 数据不得写成 immutable-image 结果**。
- **出处**：[`14`](14-image-dirty-worktree-unreproducible-pins.md)、[`../progress/landed.md`](../progress/landed.md)

---

## B. 环境 / 工具链 / 运行环境

<a id="b1"></a>
### B1. `ptoas compilation failed` 但错误正文为空 ✅

- **背景**：campaign 中途某次 codegen 失败。
- **现象**：`ptoas compilation failed`，**错误正文是空的**（没有任何 diagnostic）。
- **过程**：容易当成「我刚改的东西编不过」，于是开始 rotate knob（换 chunk / 换 flag），
  把一次瞬时崩溃变成一串假结论。
- **处置**：✅ 判 flake 的流程：**先不要动任何 knob** —— 在同 image / 同 NB / 同串行 codegen 下，
  对 parent 与 candidate 各跑一次 compile-only。两边都 OK ⇒ 判 ptoas 瞬时崩溃，
  **整个 campaign 作废重跑**。
- **出处**：[`../benchmark/2026-08-10-step3p5-p1a-gate-decouple.md`](../benchmark/2026-08-10-step3p5-p1a-gate-decouple.md)

<a id="b2"></a>
### B2. runtime 打印的 env 名不存在 🩹

- **背景**：想按 [A7](#a7) 抬 scheduler 超时。
- **现象**：runtime 在错误信息里点名 `PTO2_SCHEDULER_TIMEOUT_MS`（`error_names.h:172`），
  设了完全不起作用。
- **过程**：那个名字**在代码里不存在**；真正读的是 `SIMPLER_SCHEDULER_TIMEOUT_MS`
  （`runtime_timeout_config.h:25`）。另一个近名 `PTO2_TENSOR_DATA_TIMEOUT_MS` 是
  compile-time constexpr，**根本不读 env**。
- **处置**：🩹 绕开：**一律 grep `getenv` 核对**再设，不信错误信息里的 env 名。
- **移除代价 / 复发条件**：上游不修就永久有效；任何新 env knob 都要重做这一步核对。
- **出处**：[`16`](16-dispatch-fusion-orch-decouple.md) §6.5

<a id="b3"></a>
### B3. 「目录建了、文件是空的」—— FATAL 被日志级别门掉 ✅

- **背景**：抓 orchestrator 侧 FATAL（想知道它在等哪个 producer）。
- **现象**：设了 `ASCEND_PROCESS_LOG_PATH`，目录如期建出来，**里面的文件是空的**。
- **过程**：点名 producer 的那条字符串在 `CheckLogLevel(AICPU, DLOG_ERROR)` 处被门掉，
  只设路径不改级别拿不到任何内容。
- **处置**：✅ 必须同时 `ASCEND_GLOBAL_LOG_LEVEL=3`。
- **出处**：[`16`](16-dispatch-fusion-orch-decouple.md) §6.7

<a id="b4"></a>
### B4. agent 侧：Bash 沙箱自带文件系统写满 ✅

- **背景**：agent 连续执行命令时 Bash 工具突然全部失败。
- **现象**：`ENOSPC`；清 `/tmp`、查 inode 都没用 —— 主机磁盘其实很空。
- **过程**：误诊成「`/tmp` inode 耗尽」，花了时间在主机上找空间。真因是
  **Bash 工具跑在沙箱里、沙箱有自己的文件系统**，满的是沙箱不是主机。
- **处置**：✅ 所有 Bash 调用改为 `dangerouslyDisableSandbox: true`。连带：子 agent 预算耗尽
  会报 `402 Budget pool quota has been exhausted`，此时对抗式 review 由主 agent 自己承担
  （需用户授权），**不要静默跳过 review**。
- **出处**：本仓 session 记录（2026-08-21）

---

## C. 仍在生效的绕路 —— 移除前必读

> 这一节是**技术债台账**。每条的绕路都是承重的：删掉它，对应的 bug 会立刻回来。

<a id="c1"></a>
### C1. head-gate `matmul_acc` N=16 丢 K 累加 🩹

- **背景**：head-gate = `sigmoid(input_layernorm(hidden) @ w_g)` per head（**NORMED hidden**），输出 N=16。
- **现象**：`gate_logits` ~20× 偏小、逐 head 比例不一 → `sigmoid≈0.35` 不压制 → 「热」rank 的
  o_proj partial 爆 ~40× → 整层乱码（in-process A/B `bad_ratio≈0.97`）。
  **离线 e2e `bad_ratio=0.0000` 但 live 乱码** —— 这个反差曾把排查带到 KV 来源上去。
- **过程**：同一累加循环、大 N 的 q_proj 正确，**唯独小 N=16 出错**；输入逐位验证正确。
- **处置**：🩹 live bridge 路径把 gate 搬到 worker 端 python 预算 `gate_exp`，复用现有 `gate_r`
  slot（`BATCH == NH_PAD == 16`，`[16,1024]` 无签名/形状变化）。放回 on-device 小 N matmul ⇒ 乱码回来。
- **⚠ 两条路径处置不同**：**整网 attention** 据 [`../blockers.md`](../blockers.md) 与
  `design/performance/task-tracking.md` I7（on-device 8-block logits）已恢复 on-device；
  上游小 N `matmul_acc` 是否已修 **未在本仓闭环** —— 改之前先去 0162 核你动的是哪条路径。
- **出处**：[`09`](09-attention-multiposition-corruption.md) §3.2/§4.2、[`11`](11-8001-bridge-live-ops.md) 症状 5

<a id="c2"></a>
### C2. Q-side RoPE pack codegen 数值错 🩹

- **背景**：full-attn Scope 2 把 Q 的 RoPE 结果打包进 `all_q_padded`。
- **现象**：ctx>1 / prefill 乱码，`bad_ratio≈0.90`；损坏区恰为 `rot_q_hi` 的列 32..63。
  **单 position（ctx=1）ST 是 PASS 的** —— 值错被 ctx=1 的 softmax 掩盖了。
- **过程**：model-side 逐 head 连续切片重写后 crossrow `0.8374 → 0.0000`。
- **处置**：🩹 model-side 改切片结构（**不改数学**）。上游未修。
- **移除代价 / 复发条件**：恢复原打包写法即复现。另：若未来 `KV_HEADS_LOCAL>1`
  （unsliced / `apply_tp1_patch` 路径）会触发另一个独立 bug；生产 per-rank 不受影响。
- **连带纪律**：attention 的 ST **必须含 crossrow / multi-position**（`seq_lens=arange(BATCH)+1`），
  单 position ST 不构成 attention 门。
- **出处**：[`09`](09-attention-multiposition-corruption.md) §4.1/§4.3

<a id="c3"></a>
### C3. gap-5：in-kernel `cast(→INT8)` 喂 cube A-operand 误编译 🩹

- **背景**：routed MoE 从 BF16-dequant 切到真 W8A8 INT8-native（HBM 从 OOM 边缘降下来）。
- **现象**：`bad_ratio≈0.9847`、`max diff ~254`、**无 device fault**（最难查的组合：
  没有任何 runtime 信号，只有数值全错）。
- **过程**：曾归因 `tile.matmul_mx`；IR dump 实证推翻 —— 该 op **一次都没 emit**。
  真因是 in-kernel `cast(→INT8)` 的结果喂 cube Left 操作数时 fractal 推导错
  （`infer_tile_memory_space_pass` 未推 INT8 cube fractal），经 materialized tensor 的
  `tile.load` 会被 DMA 正确 fractalize，所以 **GM/staged INT8 对、in-kernel cast 错**。
- **处置**：🩹 model-side 把 quant 切到 dispatch 侧（不改 compiler）。上游 IR-level fix 未落。
- **移除代价 / 复发条件**：任何把 quant 放回 in-kernel cast 的改动都会回到 ~98% wrong 且无 fault。
  控制组做法：同 shape INT8 从 GM 读入（预 fractal 化）→ PASS 即确认是这个 bug。
- **出处**：[`10`](10-gap5-attention-quant-scope.md)

<a id="c4"></a>
### C4. `[N,1]` intra-UB VEC tile 运行时对齐 fault 🩹

- **背景**：head-gate 里 `pl.row_expand_mul(left[N,K], right[N,1])`。
- **现象**：编译通过，运行时 AIV `errcode 0x800 "UB not aligned"` → `507018`。
- **过程**：`[N,1]` FP32 的行字节 = 4 B，违反 Vec tile 行字节 32-B 对齐规则；
  `pl.full([1,1])` 会被 verifier 拦，**但 `pl.slice` 漏检** ⇒ 编译期放过、运行期炸。
- **处置**：🩹 用 `pl.row_sum/row_max` reduction 或 `pl.reshape` 构造 `[N,1]`，**不要用 slice**。
- **移除代价 / 复发条件**：上游在静态期补上 slice 的对齐检查前，这条一直有效。
- **完整条目在 sub-repo**：`pypto-lib/docs/known-pypto-pitfalls.md` §1 / §2（本页只留指针）
- **出处**：[`../reference/pypto-programming-api.md`](../reference/pypto-programming-api.md)

<a id="c5"></a>
### C5. `SplitIncoreOrch` precondition 失败 🩹

- **背景**：升级 pypto 后新增了 `SplitIncoreOrch` 硬校验。
- **现象**：`SplitIncoreOrch: InCore ScopeStmt found in non-InCore function (should have been outlined)`
  —— step3p5 MoE `chip_orch` 编译硬失败。
- **过程**：定位到两个 InCore helper（`_zero_routed_y_buf` / `_serialize_after_shared`）
  整个函数体被**冗余的单层 `with pl.at(CORE_GROUP)`** 包着；对照能过的
  `_publish_src_route_table` / `dispatch_step` 没有这层 wrapper。
- **处置**：🩹 unwrap 那层冗余 `pl.at`。**根因 pass 未最终锁定**（状态 🟡 缓解）。
- **移除代价 / 复发条件**：再包回 `pl.at` 即复现；升级 pypto 后需重验。
- **连带方法论**：两个相似层「一 PASS 一 FAIL」时，**逐 pass dump IR diff**
  定位「从 PASS 变回 FAIL」的那个 pass，比凭表层结构猜命中率高得多。
- **出处**：[`05`](05-splitincoreorch-swiglu-l43-l44.md)

<a id="c6"></a>
### C6. notify 的 cache-invalidate 排在 payload drain 之前 ⏸

- **背景**：`pld.tile.remote_store(...)` 紧接 `pld.system.notify(...)`。
- **现象**：接收方读到的 payload 部分或全部丢失，**受损区恰为 `0`**，受损 rank 随时序变化。
- **过程**：device 已证根因 —— `MakeNotifyCodegenPTO` 把 `dcci(ENTIRE_DATA_CACHE)`
  （invalidate-only、无 writeback）排在 payload `TSTORE` 排空之前；现成的
  `pipe_barrier(PIPE_MTE3)` 在 invalidate **之后**，无用。消融矩阵已闭合：
  只有 `PIPE_ALL` 单独 `exact=True` 64/64，**更便宜的屏障全部不行**。
- **处置**：⏸ **未修，也没有绕路** —— 它现在是一条**硬约束**：任何把「payload store 与
  它自己的 credit」拉近的改动（合并波次、按 peer 融合 store+notify、单 peer 交换）
  都进入探针的近确定性失败区间，**必须先落 fence**。生产 Wave2 **没有可证明的安全机制**，
  只是当前调度没触发它 —— 别说成「正在损坏」，也别说成「是安全的」。
- **出处**：[`../blockers.md`](../blockers.md) UPSTREAM-NOTIFY-FENCE、[`../design/performance/06-upstream-asks.md`](../design/performance/06-upstream-asks.md)

<a id="c7"></a>
### C7. 单卡 ST 用 unslice 全量 shape ✅（认知）

- **背景**：写单卡 ST/UT 验证 per-rank 正确性。
- **现象**：`sh_mlp` 1.66 MB > 512 KB、`gate_matmul` 753 KB > 512 KB 等 L1/UB overflow。
- **过程**：把「单卡」理解成「8 卡的全量装一卡」，用了 `apply_tp1_patch()` 把
  `*_LOCAL` 折回 unsliced 全量。kernel chunking 是按 TP=8 per-rank 设计的
  （`_CHUNK = INTER_S_LOCAL` 这类「chunk 跟 slice 走」的常量会跟着放大 8×），必然爆。
  **这个 overflow 不是单卡的阻塞，是 shape 选错了。**
- **处置**：✅ 单卡 ST/UT 一律 `apply_perrank_patch()`，保 TP=8 per-rank slice 宽度；
  collectives 在单 rank 下退化为 identity 由 codegen 自动消除。
  `apply_tp1_patch()` 只适合 Phase 15 e2e 路径。
- **出处**：`CLAUDE.md` 铁律 1

<a id="c8"></a>
### C8. ccec 后端不支持 bf16 C-style cast 🩹

- **背景**：gate 输出 `expert_weights` 原本是 BF16，combine 侧再 cast 成 FP32。
- **现象**：`fatal error: error in backend: not support bf16 type cast`
  （生成码里的 `(bfloat16_t) float` / `(float) bfloat16_t`）。
- **过程**：gate 内部 `weights_pad` 本来就是 FP32，BF16 中转纯属丢精度。
- **处置**：🩹 把 `expert_weights` 全链路改成 FP32（5+ 个文件同步），
  并删掉各 mirror 内联点的 element-level scalar bf16 cast。等价且更准。
- **移除代价 / 复发条件**：任何新增的 element-level bf16 scalar cast 都会再撞。
- **出处**：上层 `CLAUDE.md` 2026-06-17 记录

<a id="c9"></a>
### C9. canonical structural analyzer 恒 `FAIL_CLOSED` ⏸

- **背景**：canonical 验收里的 structural analyzer。
- **现象**：恒 `FAIL_CLOSED` —— 零本地 routed-token 的 early-dispatch task **缺 AICore swim record**。
- **过程**：既有限制，**不是**某轮回归引入的。后果被低估过：8 卡里只有 rank2 可分析，
  **一切 device 侧 cross-rank 结论都以它为前置**。
- **处置**：⏸ 未解。补它 = 性能线的 `H5`（P1）。在解决之前**继续 fail-closed**，
  不得用 host 独立检查覆盖 canonical structural 的 fail-closed 结论。
- **出处**：[`../STATUS.md`](../STATUS.md) §5 / §6

---

## 不在本页的坑去哪找

| 坑类 | 唯一落点 |
|---|---|
| kernel / codegen 硬限制（`[N,1]` tile、32-B 行对齐、`pl.dynamic` 丢 stride、幻 `int32_t` 参数、AICPU 不能 `fprintf`、裸 `for`、`pl.range(常量)` unroll 爆 UB） | sub-repo `pypto-lib/docs/known-pypto-pitfalls.md` |
| dev workflow（stale `.pyc`、三件套激活、HTTP/2 push 超时、netboot 丢 SSH key、`gh` 不可用） | sub-repo `pypto-lib/docs/dev-workflow-gotchas.md` |
| 部署失败（`507899`、simpler init `507018`、driver upgrade device busy、`-Werror` buffer_elems、netboot 丢 cmake / libstdc++-12-dev、containerd 重建） | [`../deployment/machine-recovery.md`](../deployment/machine-recovery.md)「常见部署失败」 |
| 三件套版本绑定为什么是硬的 | [`../deployment/phase16-three-pillars.md`](../deployment/phase16-three-pillars.md) |
| 一个 error signature 的完整证据链 / 消融矩阵 / 走过的弯路 | [`README.md`](README.md) 16 份复盘 |
| 已被否决、不要重试的方向 | [`../progress/landed.md`](../progress/landed.md)「已否决，不要重试」 |

## 怎么加一条

1. 先问**它有家了吗** —— 命中上面路由表就写到那里，不要在本页复制。
2. 选段（A 测量诊断 / B 环境工具链 / C 仍在生效的绕路），按现有条目格式写，**≤14 行**。
3. 处置必须打 ✅ / 🩹 / ⏸；🩹 和 ⏸ **必须**写「移除代价 / 复发条件」。
4. 到 [§0 按现象查](#0-按现象查) 加一行（现象写成**你会看到什么**，不是根因）。
5. 如果这条坑值得**开工前就记住**，再去 [`LESSONS.md`](LESSONS.md) 加一行触发式索引。
