# 专项：retained CommDomain window 清零从 host 搬零改为 device memset

| 字段 | 值 |
|------|----|
| **子任务** | PERF-H1（新增 Track H — host 侧 per-step 开销） |
| **镜像** | `hub.i.basemind.com/stepcast/vllm-pypto:stepfun-develop-20260729-allreduce-push-v3` |
| **卡 / 工作点** | cards 8–15，`--num-blocks 512`，ctx=65536，batch=16，`--itl-iters 20 --itl-warmup 3` |
| **代码** | simpler `e2efebcb`（over `8459d60f`）+ pypto `1f704616`（over `ce7fcb64`，含 gitlink bump） |
| **相关** | [`../design/performance/02-detailed-design.md` PERF-H1](../design/performance/02-detailed-design.md)、[`postmortems/14`](../postmortems/14-image-dirty-worktree-unreproducible-pins.md) |

> **口径**：所有数字来自**同一个已发布镜像**。改动以「只含本改动 hunk 的 patch 挂进容器 `patch -p1`」方式生效，
> 保证被测对象 = **image + delta**，不是本地工作树（`postmortems/14` 的教训）。dry-run 干净（仅 offset，无 fuzz）；
> patch 对镜像是**纯新增**：worker.py `+120/-0`，distributed_runner.py `+12/-0`。

---

## 1. 结论

`_reset_persistent_domains` 每步用 host 零缓冲把整个 retained window 覆盖一遍，**占 ITL 的 25%**。
改为 device 侧 `aclrtMemset`（backend 给 fresh window 清零本来就用它）后：

| 指标 | 改前 | 改后 | 变化 |
|---|---|---|---|
| window 清零 | **21.50 ms** | **2.21 ms** | **−89.7%** |
| ITL p50 @64k/bs16 | 85.02 ms | **65.55 ms** | **−22.9%** |
| 每步 H2D | 244.7 MiB | **0** | 探针实测 `copy_to=0 bytes=0` |
| 每步 mailbox 往返 | 248（串行阻塞） | **1 次广播**（8 卡并行） | |

---

## 2. per-step host 阶段拆分

探针（纯 monkey-patch，不改镜像/仓库）包住 `_reset_persistent_domains` / `_finish_persistent_request` /
`_dispatch_prepared`，warm 口径跳过前 3 步。工件 `0162:/mnt/persist/chensiyu/workspace/benchmark/2026-07-29-{clearcost,memset}`。

| mode | total | clear | drain | submit | ITL p50 |
|---|---|---|---|---|---|
| A baseline | 84.85 | **21.50** | 59.95 | 3.39 | 85.02 |
| B chunk 1→32 MiB（语义不变，仅减往返） | 78.48 | 14.94 | 60.20 | 3.35 | 78.67 |
| C 只清 signal buffer（**语义改变**，仅作上限参考） | 63.76 | 0.87 | 59.50 | 63.93 |
| **E device memset（本改动）** | **65.37** | **2.21** | 59.56 | 3.47 | **65.55** |

三点读数：

- **握手 vs PCIe 分离**：248 → 8 次往返省 6.56 ms ⇒ 单次 mailbox 往返 ≈ **27 µs**；余下 14.94 ms 是
  244.7 MiB 真实 H2D ⇒ 有效带宽 ≈ **16.4 GB/s**。
- **B/C 都不是终态**：B 保留全部 H2D；C 跳过 9 个数据 buffer 属语义改变，须过精度门。E 语义等价且优于 C 之外只差 1.6 ms。
- **`drain` 四个 mode 稳定在 59.5–60.2 ms**，与 swimlane 实测 device span 55.3 ms 相符（差 ~4.6 ms 为 drain/cleanup）
  ⇒ 本改动**没有触碰 device 执行**。

预算闭合：`59.56 + 2.21 + 3.47 = 65.24 ≈ total 65.37`。
baseline 85.02 vs 无探针 83.5 ⇒ 探针自身 ~1.5 ms，四个 mode 同等承担，delta 干净。

---

## 3. 窗口构成：99.85% 的搬运是无谓的

生成 `host_orch.py` 的单一 domain `comm_d0`，per-rank `window_size = 32,063,232 B = 30.58 MiB`，
按 `_PERSISTENT_ZERO_CHUNK_BYTES = 1 MiB` 分 31 段 × 8 卡 = 248 次。

| 类别 | buffer | 字节 |
|---|---|---|
| **信号 / 计数**（`Ge` 阈值必须从 0 重启） | `dense_attn_signal` 1536 · `dense_mlp_signal` 1536 · `moe_attn_signal` 21504 · `moe_sh_signal` 21504 · `moe_meta_arrived` 512 · `moe_data_arrived` 512 · `moe_combine_arrived` 512 | **47,616** |
| 数据（生产者每步全写） | `moe_recv_x` 18,874,368 · `moe_attn_tmp` 5,505,024 · `moe_sh_tmp` 5,505,024 · `moe_routed_y_buf` 1,048,576 · `dense_attn_tmp`/`dense_mlp_tmp` 各 393,216 · `moe_recv_aux`/`moe_recv_route` 各 147,456 · `moe_recv_meta` 1,280 | **32,015,616** |

本改动**不利用**这个区分（仍清整个窗口），所以语义与改前完全一致；上表只用于解释成本量级，
并说明 mode C 那条「只清 signal」的路为什么被 E 取代。

---

## 4. 功能等价证据（同镜像 A/B 跑 `run_whole_network_ci`）

工件 `0162:/mnt/persist/chensiyu/workspace/benchmark/2026-07-29-ci-ab`。

`main_hidden_8step` 两边均 `rc=0 passed=True`；`main_hidden_only_report.json` **除 `run_sec` 外逐字段相同**：

```text
tokens              [303, 1207, 19384, 872, 428, 6127, 4231, 2636]   两边相同
token_exact         全 True                                          两边相同
hidden_finite       全 True                                          两边相同
hidden_tp_spread    max 0.0                                          两边相同
hidden_row0_abs_max [664, 272, 368, 318, 352, 404, 266, 350]         元素级相同
run_sec  baseline   [1.6325, 0.0675, 0.0672, 0.0667, 0.0667, 0.0671, 0.0667, 0.0654]
run_sec  memset     [1.6370, 0.0488, 0.0480, 0.0476, 0.0480, 0.0485, 0.0472, 0.0480]
```

warm per-step `run_sec` **0.0667 → 0.0478 s（−28.6%，约 19 ms/step）**——与 clear 省下的 19.3 ms 吻合，
且**与 ctx 无关**（窗口大小不随 ctx 变），构成第二个独立工作点的印证。

⚠ CI **整体** rc=1，但**两边同样失败**在 `mtp_hidden_single`：缺一个未挂进容器的 host fixture
（`workspace/logs_n1/live_mtp3_patch_ci4_inline_runtime_20260718_220645/dumps/single/mtp3_hidden.pt`）。
环境缺失，与本改动无关；**MTP gate 在该容器配置下无法执行**。

单测：`tests/ut/py/test_worker/test_memset_all.py` 8 例，镜像内（需先 `pip install pytest`）**8 passed**。

---

## 5. 被证伪的假设（勿重复走）

| # | 假设 | 判定 | 依据 |
|---|---|---|---|
| 1 | 3 波 barrier 共用 signal cell + 每步清零 ⇒ `notify` 可能被抹 ⇒ 挂死 | **不成立** | 清零虽 per-request（`distributed_runner.py:1296`），但 `run_control_command` 是忙等自旋（`worker_manager.cpp:539`），host **同步清完 8 卡**才 `entry_fn` 下发。且生成代码只有一个 `allocate_domain`，6 个信号窗口全在 `comm_d0` 内，确实被清到 |
| 2 | step3p5 关掉 `persistent=True` 即可（不动公共仓） | **实测否决** | `persistent=False` ITL **276.2 ms**（慢 3.25×）。每步 `alloc_domain`+`release_domain` churn ≈ **212.7 ms**，约为它能省下的 21.5 ms 的 10 倍。`alloc_domain` = `aclrtMalloc` + Fabric/VMM-or-IPC 跨卡映射 + 8 rank 握手 + 最后才 `aclrtMemset` |
| 3 | 每步有 `8 × 1542 = 12336` 次串行 `_submit_chip` | **不成立** | B2 收成单 `whole_chip_orch` 后，生成 `host_orch.py` 仅 218 行 / **1 个** `_submit_chip`；1542 个 task 由片上 AICPU orchestrator 展开，host 每步只提交 **8** 次 |
| 4 | 运行时有 multi-step / resident / replay 接口可用 | **不存在** | mailbox 状态与控制子命令里都没有「再跑一次」；要做需新增 opcode + chip 侧循环 + dispatch-N-drain-once |

---

## 6. 与 `2026-07-29-release-image-64k-dfx-itl.md` 的分母对账（重要）

那份 DFX 报告与本文测的是**同一个量**，只是分母不同 —— 不是矛盾，但引用时必须说清用哪个：

| | 那份报告 | 本文 |
|---|---|---|
| `tp_all_reduce` wall union | **7.989 ms** | **~8 ms** ← 同一个数 |
| 分母 | instrumented span **435.205 ms** | device 执行 **55.3 ms** |
| 得到的占比 | **1.84%** | **14.4%** |

`435.205 − 379.9 = 55.3` —— 那份报告的 span 里**含 r2t15 那一个 task 的 379.9 ms 假等待**。
它自己也记了「instrumented span 是真实单步的 **5.21×**」并归因于插桩总体；
本文进一步定位到 **5.21× 几乎全部来自 r2t15 这一个 task**（PERF-H3）。

**推论：那份报告里所有「% span」都被同一个 5.21× 系数系统性压低。**
- 谈**绝对量**（`tp_all_reduce` 7.989 ms、routed expert busy 10.533 ms）两边一致，其「C/D/F 系对 64k
  低 ROI」的结论**不受影响**，可以放心引用。
- 谈**占比**要用 device 执行做分母，否则会低估 5.21 倍。
- 这也是 H3 值得修的第二个理由：修完后 span ≈ device 执行，两套百分比自动收敛。

---

## 7. 残留与后续（Track H 其余项）

改完后 ITL 65 ms 的构成变为 **drain 59.15（91%）+ clear 2.20 + submit 3.49**，device 执行终于成为主导项。

| 项 | 量级 | 状态 |
|---|---|---|
| **H2** per-rank 视图重建（= 跨卡起跑阶梯的病根） | submit 3.49 ms；起跑阶梯 clean run 实测 **2.914 ms**（每 rank 等距 +0.412 ms） | 已量化，未修 |
| **H3** DFX run 的第一 barrier 假长条（115–764 ms） | 仅 DFX 模式；clean run 算术上界 5.7 ms，不可能有 380 ms | 已定位方向（child 侧 collector 开销），未修 |

`clear` 剩下的 2.21 ms 若还要压，只能走 mode C 那条「只清 signal buffer」（391 KiB，0.87 ms），
但那是**语义改变**，必须先过 N=128 精度门。当前不做。
