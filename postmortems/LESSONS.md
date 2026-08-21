# 必读教训索引（开工前先读这一页）

从本目录 16 份复盘的「§6 如何避免」提炼。**触发式**：先扫左列，命中就读中列，要细节再点右列。
不是规则罗列 —— 罗列不会被应用。

> 这一页是 `CLAUDE.md` **铁律 §0** 的强制必读项。它只收**已确立**的结论；
> 复盘里标「未确立 / 已撤回」的一律不进这里。
> 新增一行的条件：某条教训在**两次以上**开发里都能救回时间，或它是一个**可量化的否决门**。

---

## A. 流程 / 方法论（最贵的教训都在这一段）

| 你正要做 | 先记住 | 出处 |
|---|---|---|
| 声明一个 root cause | **先跑证伪实验**（dispatch-cut bisect / fill-batch / golden 对拍 / VA 范围比对）。假设写进 doc 标「假设」，事实标「隔离证明」 | [12](12-integration-churn-meta.md) 根因 2、[10](10-gap5-attention-quant-scope.md) |
| 第 3 次复现同一个 error | **停手**，别换姿势重试。强制回答三问：① runtime 自己说了什么（`sub_class`/`stuck_*`）？② 我改的层和它指的层是同一层吗？③ 我这次跑绿的门能看见这个 bug 吗？答不上先补诊断 | [12](12-integration-churn-meta.md) 根因 6 |
| 说某个东西「ready」 | 只认 **live-token-exact-device**。compile / offline / synthetic / 单卡 / 单配置一律标 `provisional`。声明 bar 与真 bar 的每个 gap 都是未来推翻点 | [12](12-integration-churn-meta.md) 根因 1 |
| 用一个便宜门验证整网 bug | 先问「**这个门能看见这个 bug 吗**」。整网 liveness 门必须跑**生产 ctx / block 数**，小配置绿是假绿 | [12](12-integration-churn-meta.md) 根因 7 |
| 接手别人的候选 | 先跑**三列**特征矩阵 `grep -c <特征> candidate/ 生产基线/ develop/`，再对**生产基线**（不是 develop）直接 `diff`。漏掉基线那一列会把单点候选误判成「捆了两个」 | [12](12-integration-churn-meta.md) 根因 8 |
| 删一个「看似多余的同步 / 冗余构造」 | **先审它的立项理由** —— 它可能是承重的流控阀。删掉稀有的概率性死锁，可能换来确定性死锁 | [12](12-integration-churn-meta.md) 根因 9、[16](16-dispatch-fusion-orch-decouple.md) §3 |
| 下「死锁」结论 | 看门狗会把**慢**伪装成**死**。`S1` 先抬 `SIMPLER_SCHEDULER_TIMEOUT_MS`（env-only、零改码）分「慢 vs 死」。⚠ runtime 提示的 env 名不可信，一律 grep `getenv` 核对 | [12](12-integration-churn-meta.md) 根因 10、[16](16-dispatch-fusion-orch-decouple.md) §6.5 |
| 「为了拿一个数」再跑一轮 device | **先穷举现有 artifact**。失败 run 在挂前跑了几百个 invocation，稳态 span 早就够统计 | [12](12-integration-churn-meta.md) 根因 11、[16](16-dispatch-fusion-orch-decouple.md) §5 |
| 改任何 orchestration 级构造 | **先从现有日志读 `orch` 与 `device_wall` 的 p50**。若 `orch <= device_wall`，orchestrator 不在关键路径，**整类改动 ROI 上界 = 0**。这是可量化的否决门 | [16](16-dispatch-fusion-orch-decouple.md) §3 ★★ |
| 立项"减少 task 粒度 / 降 runtime 开销"类优化 | 先看关键路径的 **`front-gap`** 与 stall 构成。若 `front-gap = 0.000 ms` 且 stall **100% 为 data-wait** ⇒ task-granularity 与 runtime-overhead 两类方法论的 ROI 上界 = 0。**这一条本可在 dispatch 融合线的 6 天 / 357 run 之前就否决它** | [`../design/performance/09-swimlane-derived-next-optimizations.md`](../design/performance/09-swimlane-derived-next-optimizations.md) ★★ |
| 追概率性 liveness 缺陷 | 用「**一轮长跑**」（`ITERS` ×10，墙钟只多 ~25 s，曝光 ×10），不要「多跑几轮」。比较单位是 **invocation-until-failure**，不是「跑了几轮」 | [16](16-dispatch-fusion-orch-decouple.md) §5/§6.9 |
| 读 8 卡故障的少数派报告 | `sched_error_code` 分布「多数一致 + 一个例外」只说明**哪个子系统停了**（orch vs scheduler），**不说明哪个 rank 有责任**。责任要用 `grep -ho "AICPU([0-9]*," \| sort \| uniq -c` 数 distinct pid | [12](12-integration-churn-meta.md) 根因 8、[16](16-dispatch-fusion-orch-decouple.md) §3 |
| 在整网上黑盒二分「机制」 | 先按**调用点**二分（让两个实现并存，一次只切一类站点），把「哪一类站点」从「什么机制」里分离出来。45 层整网上按机制二分会连续证伪十几个假设 | [13](13-tp-allreduce-pull-notify-race.md) §5 |
| 判定一个 race 已修 | **单次绿不算通过**，必须重复采样。同理「每步都非零」也不能当「系统性错误、非 race」的证据 | [13](13-tp-allreduce-pull-notify-race.md) §5 |
| 把 step3p5 写得和 DeepSeek 不一样 | 先问「**为什么 DeepSeek 不爆**」。每个 divergence（addr-align / padding / shape / dtype / layout / static-vs-dynamic）都是潜在重复 bug；必须不一样就写清为什么 + 验证口径 | [12](12-integration-churn-meta.md) 根因 4、[10](10-gap5-attention-quant-scope.md) |
| 在「明知临时」的地基上继续堆集成 | 停。先消地基（例：别在 BF16-dequant 上堆 live 集成，直接上 INT8-native） | [12](12-integration-churn-meta.md) 根因 3 |
| 申请机器锁 | 只有 **A/B/A 计时门**要整机锁；compile / liveness / 精度门一律半机。错配会把分钟级任务变成要等两半都空 | [12](12-integration-churn-meta.md) |
| 占卡前判断"卡是空的" | **非 root `fuser /dev/davinciN` 不可信**（曾在 8 个 `VLLMWorker_TP` 各占 54.7 GiB 时报 free）。必须 `sudo -n fuser` + `npu-smi info -t proc-mem` **双查、fail-closed** | [`../benchmark/2026-08-10-step3p5-p1a-gate-decouple.md`](../benchmark/2026-08-10-step3p5-p1a-gate-decouple.md) |
| 逐个 kernel 试融合以求性能 | **先把单点收益对上检测地板**（bs=1 是 `0.634 ms`）。单个 small-op 融合几乎都低于地板 ⇒ 上 A/B/A 只会得到"统计不可区分"。要凑成 bundle 或换更大的项 | 同上 + [16](16-dispatch-fusion-orch-decouple.md) |
| 遇到 `ptoas compilation failed` 且**错误正文为空** | 可能是 ptoas 瞬时崩溃。**不要 rotate 任何 knob** —— 在同 image / 同 NB / 同串行 codegen 下对 parent 与 candidate 各跑一次 compile-only；两边都 OK 就判 flake，整个 campaign 作废重跑 | 同上 |
| 看 swimlane 里某个 collective 花了几十 ms | 那不是搬运时间。**collective spin-wait 会被记成 kernel compute**（曾见单个 AR `35,530.5 µs`、AR family 占 makespan `95.1%`，而 payload 只有 128 KB）⇒ AR 主要是吸收 rank skew 的 barrier，降 step 救不了它该等的最慢 rank | 同上 |
| 换底座（分支 / 机器 / 版本组合） | ready 状态**不自动迁移**，必须 re-derive | [12](12-integration-churn-meta.md) 根因 5 |
| 发布一个镜像 / 记一组 pin | **pin 相同 ≠ 内容相同**。发布审计必须查工作树是否 dirty，否则记录的 pin 复现不出验证环境 | [14](14-image-dirty-worktree-unreproducible-pins.md) |
| 遇到「镜像能跑但 pin 跑不通」 | **第一步导出 diff 存档**（补丁只活在镜像层里，镜像一删即失），不要先去改构建 | [14](14-image-dirty-worktree-unreproducible-pins.md) §5 |
| 怀疑失败是自己刚改的东西引入的 | 先做**单变量对照**（把改动前的文件挂到新环境上）——先证明「与我无关」，再找真因 | [14](14-image-dirty-worktree-unreproducible-pins.md) §5 |
| 引用本地分支 / 旧 phase / 旧 benchmark 作当前事实 | 不行。当前状态只认带日期的 [`../STATUS.md`](../STATUS.md) + GitHub 远端 ref | `CLAUDE.md` 铁律 9 |

---

## B. 整网 / 分布式（硬约束，违反就是 hang）

| 你正要做 | 先记住 | 出处 |
|---|---|---|
| 拆分 / 新增 `@pl.program` | **生产整网只允许单个 `@pl.program`**。N≥6 撞 co-prepare 墙（`SCOPE_DEADLOCK` + `sched=100` + `TaskMapSize=0`）。N=1 跑不通就修 collective handshake，不换路径 | [08](08-multiprogram-coprepare-deadlock.md) |
| 调 `task_window` | 生产值 **65536**。盲目调大（2^20 → 64 GB arena OOM）、用默认 16384（N≥6 SCOPE_DEADLOCK）都不行。`PREPARE OK` 后仍挂 ⇒ 不是 ring sizing | [08](08-multiprogram-coprepare-deadlock.md) |
| 写 / 改逐层 comm window | 每层 window 必须 **distinct**（`_L{pos}` 前缀），层间不复用 SSA；死的旧 alloc 必须删。生成 `host_orch.py` 后先 grep 同名 SSA，比再跑一次 device 便宜得多 | [07](07-whole-net-scheduler-timeout.md) |
| 摆 signal buffer | **512 B 物理隔离**（`COMM_CONTROL_SIGNAL_BYTES=512`），不与 payload 共 L2 cache line | [07](07-whole-net-scheduler-timeout.md) |
| 改 `tp_all_reduce` | 保留两波完成 barrier（stage-in → notify(Ge 1) → accumulate → completion(Ge 2)）；固定 peer order + 单 FP32 accumulator + 最后一次 BF16 cast；**必须写成 `x = f(x)`**，丢返回值靠副作用是硬约束违规 | [07](07-whole-net-scheduler-timeout.md)、[15](15-tp-allreduce-source-publication-lifetime.md)、[13](13-tp-allreduce-pull-notify-race.md) |
| 设计任何跨 rank 协议 | 显式画全 `producer → payload publication → notify → wait → consumer → reuse`。**data publication 与 control publication 各需同方向同步 primitive**；source / result / final-read 的 lifetime 各自建 gate，不能只靠一个 completion wave | [15](15-tp-allreduce-source-publication-lifetime.md)、[13](13-tp-allreduce-pull-notify-race.md) |
| 让层间共享中间张量 | **write-once per layer**（`h_moe_L{pos}` / `h_mid`），不跨层 stash / WAW | [07](07-whole-net-scheduler-timeout.md) |
| 用动态 `pl.spmd` grid（阻塞标量读） | 不得指向**体内含跨卡 wait** 的 producer —— 那会把上游跨卡停滞放大成全 rank orchestrator 冻结。**也不得直接删**，必须换成只依赖本地 device 进度的节流。`predicate=` 不阻塞，host 标量不阻塞 | [16](16-dispatch-fusion-orch-decouple.md) §3 |
| 看到 `orch_done=1` 而基线没有 | 你删掉了一个节流阀，run-ahead 已无界，注定撞 ring —— 不必等 device 门跑完 | [16](16-dispatch-fusion-orch-decouple.md) §6.4 |
| 遇到 `HEAP_RING_DEADLOCK` | **先算容量账**：单次 invocation 的 task 数 vs ring slot 数。单次填不满 ⇒ 不是 run-ahead 深度问题，是累积 / 回收问题（读 `scope_end()` / `on_task_release` / `release_producer`） | [16](16-dispatch-fusion-orch-decouple.md) §6.6 |
| 调 grid-stride 的 grid 值 | 与**正确性无关**：`pl.range(worker, N, grid)` 在 worker 取遍 `[0, grid)` 时并集恰为 `[0, N)`，任何 `>=1` 都全覆盖。它只是并行度 / 节流旋钮，别写成正确性问题 | [16](16-dispatch-fusion-orch-decouple.md) §6.8 |
| 写 top-k | 必须 **format1 渐进 `block_len`** 链排满整宽；禁止 format2 归并未完全排序的半块。先对照 DeepSeek v3_2 working gate | [06](06-gate-topk-deadlock.md) |
| 抓 orchestrator 侧 FATAL | 必须 `ASCEND_GLOBAL_LOG_LEVEL=3`，否则点名 producer 的字符串被 `CheckLogLevel(AICPU, DLOG_ERROR)` 门掉（只设 `ASCEND_PROCESS_LOG_PATH` 会「建了目录、文件是空的」） | [16](16-dispatch-fusion-orch-decouple.md) §6.7 |
| 排查 `507018` / `orch_error=8` | 先跑 dispatch-cut bisect（K=1 / K=2 / K=42）缩到具体层再下结论。K=1 clean + K=2 确定性挂 ⇒ 几乎一定是跨层 alias 或 lifetime，不是容量 / 概率 / poison | [07](07-whole-net-scheduler-timeout.md) |
| 清 device fault | **禁止 `npu-smi set -t reset`**（AMP+HCCS netboot 机会重启全 16 卡 → SSH-key 抹除 → 锁死）；**禁止 `-9` 强杀 device 上的进程**（无 finalize → card poison）。等 finalize 的 `aclrtResetDeviceForce` 清 | [06](06-gate-topk-deadlock.md) |

---

## C. 部署 / 运维

| 你正要做 | 先记住 | 出处 |
|---|---|---|
| 部署多卡 | **Phase 16 三剑合璧**：driver 25.5.2 + firmware 7.8.0.7.220 + CANN 9.0.0-beta.1（**NOT GA**）。三件缺一都不行，各自解决正交失败模式 | [01](01-multirank-ipc-507899-507018.md)、[`../deployment/phase16-three-pillars.md`](../deployment/phase16-three-pillars.md) |
| 升级 CANN | **先备份 beta.1**，防集群自动化脚本覆盖成 GA。升 driver 前 `systemctl stop kubelet` + `stop bip-agent`（DaemonSets 由 containerd 拉起，`kubectl drain` 抓不掉） | [01](01-multirank-ipc-507899-507018.md) |
| 升级到新 origin pin | **审计本地 patch 被吞 / 丢了哪些**（曾丢过 SDMA-OFF）。只做 compile 验证不够，device 路径（单卡 `hello_world` + `allreduce -d 0-7`）必须重验 | [01](01-multirank-ipc-507899-507018.md) |
| 遇到 `507899` / `207006` | **先跑裸 ACL probe**。通 ⇒ 白名单路径问题（[02](02-0234-l3-ipc-pid-validation.md)）；不通 ⇒ 三件套问题（[01](01-multirank-ipc-507899-507018.md)）。别一律归因三件套 | [02](02-0234-l3-ipc-pid-validation.md) |
| 在容器化 forked worker 里导出 IPC window | 统一 `ACL_RT_IPC_MEM_EXPORT_FLAG_DISABLE_PID_VALIDATION` **并跳过** `aclrtIpcMemSetImportPid`（两种模式不可混用）；import 侧 `ENABLE_PEER_ACCESS` | [02](02-0234-l3-ipc-pid-validation.md) |
| 让 pypto 与 vLLM 同卡 co-tenancy | worker 必须 `export SIMPLER_COMM_NO_HCCL=1`；启动顺序**先 vLLM（等 health=200）再 worker**。顺序反了必挂 | [03](03-hccl-cotenancy.md)、[11](11-8001-bridge-live-ops.md) |
| 做 device-IPC export | export 后必须 `aclrtIpcMemClose`（在 **import 侧** teardown，别在 export 侧裸指针上调，会 segfault），或复用长生命周期 key。`aclrtResetDeviceForce` **清不掉** exbus 泄漏 | [11](11-8001-bridge-live-ops.md) |
| 导出 KV cache 给 IPC | 用 K/V **各自独立**的 `torch.npu.MemPool`（非共享、非裸 `aclrtMalloc`），保证 `data_ptr == 块基址`。`aclrtIpcMemGetExportKey rc=507899` ⇒ dptr 是 sub-pointer | [11](11-8001-bridge-live-ops.md) |
| 用 code reading 论证「上游有没有 escape hatch」 | 以 0162 `stepfun/develop`（authoritative）为准；本地 feature 分支可能落后。stale branch 的 code reading 不是事实 | [03](03-hccl-cotenancy.md) |

---

## D. codegen / kernel

| 你正要做 | 先记住 | 出处 |
|---|---|---|
| 给 attention kernel 写 ST | **单 position ST 不构成 attention 门**，必须含 crossrow / multi-position（`seq_lens=arange(BATCH)+1`）。ST PASS 但 live/prefill 乱码 ⇒ q·k 值错被 ctx=1 softmax 掩盖 | [09](09-attention-multiposition-corruption.md)、[11](11-8001-bridge-live-ops.md) |
| 用小输出宽度的 cube matmul | **N=16 这类小 N 要单独 probe**：`matmul` + `matmul_acc` 累加循环可能丢 K 累加，与 N≥128 行为不同。输出系统性偏小 + 逐 column 比例不一 ⇒ dump 出来对 python FP32 ref 比 ratio | [09](09-attention-multiposition-corruption.md) |
| 在 kernel 里 `pl.cast(→INT8)` 喂 cube A-operand | **是 suspect**（`infer_tile_memory_space_pass` 未推 INT8 cube fractal）。~98% wrong + `max diff ~254` + **no device fault** 就是它。控制组：同 shape INT8 从 GM 读入（预 fractal 化）→ PASS 即确认。`0× matmul_mx` 不代表没问题 | [10](10-gap5-attention-quant-scope.md) |
| 算 head-gate | 口径 = `sigmoid(input_layernorm(hidden) @ w_g)` per head —— **NORMED hidden，不是 raw**。上游修好前 gate 保持 worker 侧预算，别放回 on-device 小 N matmul | [09](09-attention-multiposition-corruption.md)、[11](11-8001-bridge-live-ops.md) |
| 升级栈后跑 codegen 回归 | **逐变体 compile-gate**（silu / swiglu7 / swiglu16 / swiglu7_swiglu16 各自跑），「silu 过」不等于「MoE 全过」 | [05](05-splitincoreorch-swiglu-l43-l44.md) |
| 面对两个相似层一 PASS 一 FAIL | **逐 pass dump IR diff**，定位「从 PASS 状态变回 FAIL 状态」的那个 pass，再读该 pass 源码。凭表层结构猜改动点命中率低 | [05](05-splitincoreorch-swiglu-l43-l44.md) |
| 撞 `pto.tmov ... supported tmov address-space pair` | Vec-LHS 矩阵乘 staging × GM-pipe 路径。**优先 model-side reshape / 缩 chunk**，别回退 tiling pass（会撞 L0B 溢出） | [04](04-tmov-vec-lhs-matmul.md) |
| 改完模型再跑 | `find <pypto-lib>/models/step3p5 -name "*.py" -exec touch {} +` 清 stale pyc，否则被旧字节码（含被 monkey-patch 污染的模块全局量）误导 | [04](04-tmov-vec-lhs-matmul.md)、`CLAUDE.md` 铁律 3 |
| 修一个 MoE kernel | whole-net **inlined copy 与 standalone kernel 是 decoupled 的**（inlined MoE 有 11+ OWN copies）。任何 kernel 修复必须同步到 inlined copy，或从 generator rebuild | [10](10-gap5-attention-quant-scope.md)、[05](05-splitincoreorch-swiglu-l43-l44.md) |
| dump 一个 create+assemble 张量 | 直接 assemble 崩 `fuse_create_assemble_to_slice`；用 `matmul(tensor, eye)` 打断 lineage 后再 dump | [09](09-attention-multiposition-corruption.md) |
| 用 model-side workaround 绕上游 bug | **保留原 buggy 写法的最小复现器** —— 提上游 issue 时必须附 | [09](09-attention-multiposition-corruption.md) |
| 写单卡 ST/UT | 保 **TP=8 per-rank slice 宽度**（`apply_perrank_patch()`），不用 `apply_tp1_patch()`。unslice 只适合 Phase 15 e2e，chunk-follow-slice 的 kernel 会爆 L1/UB | `CLAUDE.md` 铁律 1 |
| 估算 kernel 的 UB 预算 | **UB（`188416 B`）是 per-kernel-per-core**，不是聚合约束 —— 149 个 AIV 函数 Vec 总和 `24.8×` 限额仍编译通过，每个函数 offset 从 0 重新开始。反过来说 kernel **不能**把中间结果留在 UB 给下一个 kernel 用 ⇒ "融合"必须同时装下两份 staging | [`../benchmark/2026-08-10-step3p5-p1a-gate-decouple.md`](../benchmark/2026-08-10-step3p5-p1a-gate-decouple.md) |

---

## 相关

- 复盘全文索引：[`README.md`](README.md)
- 当前活跃阻塞：[`../blockers.md`](../blockers.md)
- 落地台账（哪些是**确定落地**的）：[`../progress/landed.md`](../progress/landed.md)
- 开发流水（每 session 干了什么）：[`../archive/milestones-2026-Q2.md`](../archive/milestones-2026-Q2.md)
- sub-repo 侧的坑（kernel 限制 / dev workflow）：`pypto-lib/docs/known-pypto-pitfalls.md`、`pypto-lib/docs/dev-workflow-gotchas.md`
