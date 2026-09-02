# 上游诉求（pypto / PTOAS）

> 本文件是 P2 性能专项对上游的**可直接提 issue 的诉求清单**。每条给：诉求 /
> 为什么（我们撞到的具体形态）/ 证据在哪 / 解锁多少 / 是否有本地绕路。
> 完整实验记录在 [`task-tracking.md`](task-tracking.md)，本文件只做面向上游的收口，
> **不重复证据细节**。
>
> **最后更新**：2026-09-02。
>
> **当前产品边界**：canonical SRC 为 pypto `655c7bda` / pypto-lib `a745ab659`；
> release-admitted IMG 仍为 r12（`14de90fd/e6c7d8ec`）。产品仍继承 `69ad31e4` 的静态
> single-row selector + 静态三波 fallback，不依赖下列 K6b dynamic-valid-shape 能力。
> H6 已落地，H7 仅修复 local-owner reset 回退；`bind.args` 不再是优化项。
> 第 2/3 条仍是
> 通用上游能力诉求，其 `µs/call` 数字只绑定历史 K6b 模型，不得写成当前产品待解锁收益。

排序按「correctness 优先，其余按解锁收益」：

| # | 诉求 | 仓 | 类型 | 解锁 | 有本地绕路？ |
|---|------|----|----|------|------------|
| 1 | `MakeNotifyCodegenPTO` 在 `cacheinvalid` 前补 `pto.barrier <PIPE_ALL>` | pypto | **CORRECTNESS** | 消掉一个不可证明的安全条件 + 解锁删/合并波次类优化 | ❌ 无（模型侧无 PIPE_ALL 原语） |
| 2 | `pld.tile.remote_load` 加 `valid_shape=` kwarg | pypto | perf / generic capability | 历史 K6b 模型 `3.27 µs/call` | ❌ |
| 3 | `pto.comm.tput` 接受 dynamic dst shape | PTOAS | perf / generic capability | 历史 K6b 模型 `3.75 µs/call` | ❌ |
| 4 | `pld.system.notify` 加 fence/release 参数 | pypto | perf | `2.39 µs/call`（K2b） | ❌ |
| 5 | `pld.alloc_window_buffer` 声明「是否需要每请求清零」 | pypto | robustness | 0（收益已用绕路拿到）；防回归 | ✅ 有绕路，但脆 |
| 6 | host 作用域 `pl.read` 发出未定义的 `tensor.read` | pypto | bug | — | ✅ 改写调用点 |

---

## 1. notify 的 cache-invalidate 排在 payload drain 之前（CORRECTNESS）

**诉求**：`src/backend/common/pto_ops_distributed.cpp` 的 `MakeNotifyCodegenPTO` 在
`pto.cmo.cacheinvalid` **之前**补一条 `pto.barrier <PIPE_ALL>`。

**这不是引入新概念** —— 同文件的 `MakePutCodegenPTO` 已经给 `tput` 夹了两条
`pipe_barrier(PIPE_ALL)`（注释写成 "WORKAROUND for PTOAS#872"）。诉求可表述为
**「把 put 路径已有的那条屏障对齐到 notify 路径」**。根因就是这个不对称：
`MakeRemoteStoreCodegenPTO` 不发任何屏障，而 notify 的前导把 invalidate 排在
任何 drain 之前：

```c
TSTORE(peer_window, tile);                 // payload
dcci((__gm__ void*)0, ENTIRE_DATA_CACHE);  // invalidate-only，无 writeback
pipe_barrier(PIPE_MTE3); dsb(DSB_DDR); pipe_barrier(PIPE_MTE2);
pto::comm::TNOTIFY(peer_signal, ...);      // credit
```

现成的 `pipe_barrier(PIPE_MTE3)` 排在 invalidate **之后**，对此毫无作用。

**证据（device 已证，消融矩阵已闭合）**：`pipe_barrier(PIPE_MTE3)` 单独、
`dsb(DSB_DDR)` 单独、两者组合、纯 MTE3 流量 —— **全部 `exact=False`**；
`pipe_barrier(PIPE_ALL)` 单独 **`exact=True` 64/64**。安慰剂对照（同样两条指令
放到 `TNOTIFY` **之后**）仍 `exact=False` ⇒ **原因是顺序，不是时序**。
每臂相对 baseline 的 kernel diff 恰好一行。详见 [`blockers.md`](../../blockers.md)
的 UPSTREAM-NOTIFY-FENCE 节 + [`task-tracking.md`](task-tracking.md) 2026-08-11 行。

**代价已量化**（生产形状，后处理注入生成后的 parent kernel）：3 个 notify site
全补 = `+1.250 µs/call`；只 Wave2 一个 site = `+0.405 µs/call`。约 `0.417 µs/site`。
**比 K2a 的 pipe-specific barrier（`0.0033 µs`）贵约 18 倍，不要按 K2a 外推成免费。**

**为什么不能只靠产品侧绕过**：`pipe_barrier(PIPE_ALL)` 只由 C++ backend 发出，
`pld`/`pl` **没有**对应的 intra-core full-pipe barrier 原语（`pl.system.fence` 降为
`pto.fence.barrier_all <gm>`，属 `dsb` 家族，消融已证不够；`pld.tensor.barrier` 是
**跨 rank** barrier，不是这里要的东西）。⇒ 干净修复只能在上游。

**生产暴露面（口径要准）**：静态三波 fallback 的 Wave2 notify 前导与被证伪的
形状逐字节相同，且该 Wave2 前面正是 `remote_store`。Main single-row 两波路径的
数据面不同，不能把这条暴露面直接套到当前 BS1 快路径。已否证四个候选保护机制
（纯 MTE3 流量 / MTE3 级屏障 / store-loop 自带屏障 / rank 到达 skew）。对 fallback
结论按更保守方向收口：**没有可证明的安全机制，只是当前调度没触发；是否正在损坏未知。**
不要写成「生产正在损坏」，也不要写成「生产是安全的」。

---

## 2. `pld.tile.remote_load` 缺 `valid_shape=` kwarg

**诉求**：让 `pld.tile.remote_load` 像 `pld.tile.remote_store` 一样接受运行期
valid shape，使 DMA extent 取自 valid shape 而不是静态 `shape=` 参数。

**我们撞到的形态**：K6b 要按 runtime `active_tokens` 裁剪 AR 流量。逐 op 读生成
MLIR 确认：`remote_store` 的 `tstore` 已经动态（`?x512`）、本地 `pl.load`/`pl.store`
也跟着动态，**但 `remote_load` 仍是静态 16 行**（`tload ins(16x512)`，extent 取自
`shape=` 而非 valid_shape）。

**历史 K6b 估算**：AR 单次调用总列数 15360 中 `remote_load` 占 `23.3%` ⇒
`3.27 µs/call`。该 dynamic-valid-shape 分支未落地，已被 `69ad31e4` 的静态
single-row selector supersede；这里保留的是通用能力与历史量级，不是当前产品 blocker。

**证据**：`0162:.../p2-k6b-runtime-validshape-20260810/K6B-CODEGEN-GATE.md`
sha256 `03baffd3…`；生成件 `.../ptoas/tp_all_reduce.pto`。

---

## 3. PTOAS `pto.comm.tput` 硬拒 dynamic dst shape

**诉求**：`pto.comm.tput` 接受 dynamic dst partition view（现在要求 dst 为
positive static shape）。

**我们撞到的形态**：pypto 侧**已经正确**了 —— 补齐 `shape=` + `dst_offsets` +
`src_offsets`（三者必须同时给）后 pypto emit 出
`pto.comm.tput(... !pto.partition_tensor_view<?x4096xbf16> ...)`；卡在 PTOAS：

```
tp_all_reduce.pto:51:3: 'pto.comm.tput' op expects dst to have a positive static shape
```

⇒ **阻塞在 PTOAS 而非 pypto**。原始日志
`0162:.../p2-k6b-runtime-validshape-20260810/codegen-gate-20260810-235230/host.log:1`。

**历史 K6b 估算**：TPUT 占总列数 `26.7%` ⇒ `3.75 µs/call`；与第 2 条一起
曾预测 K6b 从 `7.02` 到 `14.04 µs/call`。K6b 未进入产品，最终
`69ad31e4` 不依赖 dynamic destination shape。

---

## 4. `pld.system.notify` 缺 fence / release 参数

**诉求**：让 `notify` 能声明它需要的 release 语义（例如 `fence="release"` /
`fence="none"`），而不是固定发一套 `dcci(ENTIRE_DATA_CACHE)` + 三条屏障。

**两个动机，方向相反但同一个 API 口子**：
- **正确性**（第 1 条）：默认需要更强的序（`PIPE_ALL` 在 invalidate 之前）。
- **性能**（K2b）：一次 AR 内多个 notify 各自发一遍 **whole-cache** invalidate，
  publisher 侧的 release fence 可以 hoist 到波次外只做一次。解锁 `2.39 µs/call`。

⇒ 正确的上游形态是**把 fence 语义参数化**，而不是把默认调强或调弱。

**状态**：K2b 尚缺 A1/B/A2 bracket（本仓任务 #18），代价/收益数字仍是 bench 口径。

---

## 5. `pld.alloc_window_buffer` 无法声明「是否需要每请求清零」

**诉求**：`pld.alloc_window_buffer` 增加语义标注（例如
`needs_per_request_zero=True/False`），让 runtime 能从**声明的语义**推出每请求要清零
的范围，而不是清整个 window。

**我们撞到的形态**：`_reset_persistent_domains` 每步把**整个** retained window
（32,063,232 B）清零，但只有 `47,616 B`（7 个 control counter / signal stack）真的需要；
另外 9 个 data buffer 每步先写后读、无跨步残留依赖。整网实测这个多余清零值
**`1.75 ms/step`（`5.18%` ITL）**。

**已有绕路（收益已经拿到，所以本条不是 blocker）**：simpler 的 carve 是
「严格顺序、无 padding」，所以只要模型把 7 个 control buffer **声明在最前面**，它们
就构成唯一连续前缀 `[0, 47616)`，一次 `memset_all` 即可 —— **完全不动 pypto 与
simpler**。四臂 A/B/A 实测 `−1.7505 ms/step`，byte-exact。

**但这条绕路很脆，所以诉求仍然成立**：正确性依赖「control 全在 data 之前」这个
**没有任何地方强制**的性质。今天谁往前面插一个 data buffer，清零范围就会静默清错。
我们在产品侧加了 fail-closed 检查（control 出现在任何 data 之后即 raise），但那是
产品在替上游守一个上游本该表达的语义。⇒ 建议把这个语义放进 alloc API。

**顺带一条机制数据供上游参考**：多做一次阻塞 `broadcast_control_all` **很贵**（0-vs-5
的加性对照给出每次约 `472 µs` reset wall / `920 µs` ITL），而**字节节省约 1:1 传到 ITL**
（整块 memset 去掉 32 MiB：wall `−1720.6 µs` / ITL `−1750.5 µs`）。
**这两个数只能定性用**：所谓 `1.95×` 的放大率随统计量而变（同一实验按 p50 是 `1.95×`、
按 mean 是 `1.67×`；字节路径 `1.017–1.066×`），且 broadcast 次数 1/2/3 未测、
ITL 与 reset 的分位数不是逐迭代配对的 ⇒ **它不是一个已辨识的固定开销系数，不得做定量外推**。
定性结论足够指导 API 设计：「把清零拆成多段」是**反向**优化（我们试过，`+1.886 ms`）；
正确方向是**一次 broadcast、更少字节**。如果上游要支持 per-buffer 语义，**必须让 runtime
能把结果合并成尽量少的 broadcast**，否则 API 变好但性能变差。

---

## 6. host 作用域的 `pl.read` 发出未定义的 `tensor.read`

**诉求**：修 host 作用域 `pl.read` 的降级（现在会发出一个未定义的 `tensor.read`），
或在前端明确拒绝。

**我们撞到的形态**：在 host orchestration 里读一个标量做控制流时命中。**有绕路**
（改写调用点），所以只是 bug 报告，不阻塞。

---

## 附：与上游无关但同类的产品侧约束（记此以免重复踩）

这些是我们自己代码的约束，**不是上游诉求**，但和上面几条常一起出现：

- dynamic valid shape 贯穿整个 reduction 会触发 **loop-phi dominance bug**；历史
  K6b 能通过部分 codegen 的结构是 own/remote load 与 reduce loop 保持静态，只在
  publish 处 `set_validshape(...)`、只对 final-copy 的 load/store 用 dynamic valid
  shape。该结构的 dynamic publish 仍位于 notify-fence seam；现有运行未复现错误，
  但没有独立 rank-skew/zero-gap/多 epoch safety proof，且 K11 不采用此路径。
- `MaterializeCommDomainScopes` 拒绝「分配了 window buffer 但没有对应
  `pld.tensor.window` 材化」的 IR。
- pypto host DSL 不接受 list comprehension（`Unsupported expression type: ListComp`）。
- notify/wait 的值要显式 `pl.cast(1, pl.INT32)`。
- window 数量变化会打乱位置参数映射。

更细的 dev-workflow 坑记在 sub-repo `pypto-lib/docs/known-pypto-pitfalls.md` 与
`dev-workflow-gotchas.md`，**不在本仓**。
