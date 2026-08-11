# 2026-08-11 · K8 选择性清零（源码级 A/B/A）

> 本文是 K8 的**源码级**唯一性能记录：整网 A/B/A 三/四臂 bracket、机制常数、
> 以及三个负结果。**镜像级**复现记在
> [`2026-08-11-k8-selective-window-zeroing-image.md`](2026-08-11-k8-selective-window-zeroing-image.md)。
> 两者不得混写：本文的臂是 source-overlay，镜像那份是 digest-only。

工作点统一为 0162 / dev0-7 / bs=1 / ctx=65536 / `blocks=512` / warmup=10 / iters=100 /
seed_token 6127，`p50` 单位 ms、`reset_body`/`memset_all` 单位 µs。

---

## 1. 改动内容

每步 `_reset_persistent_domains` 原本清整个 persistent window `32,063,232 B`，
而真正必需归零的只有 7 个 control counter 共 **`47,616 B`（0.1485%）**。

| 侧 | 文件 | 改动 |
|---|---|---|
| 模型 | `models/step3p5/decode_fwd.py` | +11/−7：把 7 个 control buffer 提到 16 个 alloc 的**最前面**，使其构成**唯一连续前缀** `[0, 47616)`。alloc 数量不变 |
| runtime | `pypto/python/pypto/runtime/distributed_runner.py` | +174/−22：只 `memset_all` 该前缀；带 **16-buffer 指纹 fail-closed 回退全清**；**只对 WholeDecode 生效**；含 reset 仪表（`PYPTO_PERSISTENT_RESET_TRACE` / `reset_body_us` / `memset_all_us`） |

被清的 7 个 buffer 字节和精确等于 `47,616`：

```text
dense_attn_signal_stack_buf__ssa_v0     1536
dense_mlp_signal_stack_buf__ssa_v0      1536
moe_attn_signal_stack_buf__ssa_v0      21504
moe_meta_arrived_stack_buf__ssa_v0       512
moe_data_arrived_stack_buf__ssa_v0       512
moe_sh_signal_stack_buf__ssa_v0        21504
moe_combine_arrived_stack_buf__ssa_v0    512
```

**为什么「连续前缀」是关键**：carve 是严格顺序、无 padding，所以让模型把 7 个
control buffer 声明在最前面后，它们天然构成一段连续区间 ⇒ **一次** `memset_all`
即可，**完全不需要改 simpler 的 ctrl 协议**（原以为要扩
`_encode_memset_payload` / `_handle_ctrl_memset` 支持每 device 多段）。
这一步是整个 K8 能兑现的唯一原因，见 §4。

---

## 2. 最终判定：v3 确认臂（落地件 == 被测件）

campaign `0162:/mnt/persist/chensiyu/workspace/k8-selective-20260811/v3-20260811-144622`，
三臂 `rc=0`，`K8_V3_RESULT.json` sha256
`7bb0226326cfe77f9af2d2789673f81da99ab12b651a5c7495ba8e876561a045`。

| 臂 | ITL p50 | `reset_body` | `memset_all` |
|---|---:|---:|---:|
| `A1_parent` | 33.842 | 2260.3 | 2239.7 |
| **`B_reorder_prefix_v3`** | **32.077** | **518.3** | **476.4** |
| `A2_parent` | 33.803 | 2246.1 | 2222.1 |

```text
floor = |A1 - A2| / 2 = 0.0195 ms   （本轮最小）
delta = -1.7455 ms = -5.16% = 89.5x floor
bias-corrected delta = -1.565 ms    （按 §5 的中间臂偏置 -0.18 ms 校正）
reset_body:  2253.2(avg A) -> 518.3  = -1734.9 us
```

**两道门都 PASS**：

- `PRECISION_GATE=PASS` —— 三臂 `hidden_sha256` 全 `567b206b…`、`token=14371`、
  `all_finite`、`matches_production_baseline_sha`；
- `PREFIX_APPLIED_GATE=PASS` —— B 臂 trace `k8_prefix_applied=true` +
  `k8_control_bytes=47616`，两个 A 臂为 `None`。**这道门专门用来挡「静默回退到
  全窗清零会长得像没收益」**，没有它，一次回退就会被误读成「优化无效」。

**落地件锁定（就是被测的这两份）**：

```text
src_k8reorder/models/step3p5/decode_fwd.py
  eb1f89bf7add419f2382836c1eab9a1c4b1f63f738923d47e771e4159f104fb5
runtime/distributed_runner_prefix_v3.py
  fe50c11fb76ec77789636de05e7376711c731d2b00db5033f0564c07a739622e
驱动 bin/run_k8_v3_aba.sh                 c36bb537…
离线门 bin/claude_check_prefix_v3_layout.py 9f81ee67…
```

这两个 sha 与产品仓落地后、以及 immutable image 内的 sha **逐字节一致**，
所以「落地件 == 被测件 == 发布件」三处闭合。

---

## 3. 独立第二 bracket：v1 四臂（隔离「重排本身」）

campaign `.../k8-selective-20260811/prefix-20260811-135232`，四臂 `rc=0`，
`K8_PREFIX_RESULT.json` sha256
`f3b81a91513597f32331e4972e6d8e2d44224dbe4e33e5dfc0dffdfb92300751`。

四臂设计的要点是 **B 臂只重排、清零一字不动**，用来回答「收益是不是重排本身带来的」：

| 臂 | 源码 | runtime | ITL p50 | `reset_body` |
|---|---|---|---:|---:|
| `A1_parent` | `src_parent` | baseline | 33.776 | 2254.1 |
| `B_reorder_fullclear` | `src_k8reorder` | **baseline** | 33.879 | 2247.8 |
| `C_reorder_prefix` | `src_k8reorder` | `distributed_runner_prefix.py` | **32.052** | **530.4** |
| `A2_parent` | `src_parent` | baseline | 33.829 | 2245.1 |

```text
floor = 0.0265 ms
B  = +0.0765 ms (2.9x floor)  => 重排本身中性
                                 （落在历次 A 臂 33.776-33.933 的自然散布内）
C  = -1.7505 ms raw / -1.570 ms bias-corrected / 66x floor
C vs B（重排 held constant，纯前缀效应）= -1.827 ms
memset_all: 2233.0 / 2223.4 / 2221.9 / 496.9  => C 砍掉 1720.6 us
```

C 臂 trace 自证 `k8_control_bytes=47616`、`k8_control_range_count=1`、
`k8_control_ranges=[[0,47616]]` ⇒ 确实是**一次** broadcast、`47,616 B`。

四臂 `hidden_sha256` 全 `567b206b…`、`token=14371`、`all_finite`
⇒ **不清那 30 MiB data buffer，一个 bit 都没变**，与 window 审计（每个 data buffer
每步先写后读、无跨步残留依赖）实测一致。

**两个 bracket 一致**：v1 `−1.7505` vs v3 `−1.7455`，**差 `0.005 ms` = 0.26× floor**
⇒ v3 硬化没有性能代价，效应可复现。

v1 runtime sha `de1301234f533655818bd3cb6ef32c6df1eeecc3b34072c0db314823bf0338e9`
（仅作 bracket 证据，**不是**落地件；落地件见 §2）。

---

## 4. 机制常数：一次阻塞 broadcast = `472.3 µs` wall / `920.4 µs` ITL

campaign `.../k8-selective-20260811/bcast-20260811-133*`，三臂 `rc=0`。
探针设计：**保留全量清零一字不动**，其后追加 5 次对**刚被覆盖过**的 range 的阻塞
memset —— 幂等、零语义变化，是纯 broadcast-count 探针。arm runtime sha `515064e5…`
（diff 16 行）。

```text
A1 34.021 / A2 33.811  (floor 0.105) / B_extra5bcast 38.518  => ITL +4.602 ms
reset_body 2240.7 / 2250.2 / 4606.9                          => reset +2361 us
三臂 sha 全 567b206b…、token=14371 => 精度 PASS（幂等改动本应如此，用作 harness 自证）
```

⇒ **每次额外 broadcast = `472.3 µs` reset wall / `920.4 µs` ITL，放大 `1.95×`**。
即阻塞式 `broadcast_control_all` 在 ITL 上的代价约为其 host wall 的两倍：它不只占用
host 时间，**还打断了本可掩盖 reset 的 host/device 重叠**。这是本轮最有用的机制常数，
且与字节数无关（这 5 次每次只清 0.5–21 KiB）。

### 4.1 两个 regime 必须分开记账（对早期外推的修正）

字节传输时间在 ITL 上的放大是 **`1750.5 / 1720.6 = 1.02×`**，**不是 `1.95×`**。
`1.95×` 只适用于「额外 broadcast 的固定开销」。早期用 `1.95×` 外推字节部分
（`−3403 µs`）是错的，那个 `+687 µs` 残差因此**不是需要解释的物理量，而是模型
外推错误的产物**。用两因子模型重算 selective 得 `+5×920.4 − 1720.6 = +2881 µs`
vs 实测 `+1886 µs`，**残差换了符号** ⇒ **两因子模型只对到量级，不要再拿它做定量外推**。

### 4.2 与 codex ballast 的独立交叉验证

codex 的双 memset 探针给「整块清零 ≈ `1943 µs` wall」，减去其自带的一次固定开销
`472 µs` ⇒ 字节部分 `≈1471 µs`；本臂实测字节部分 `1720.6 µs`。同量级、差约 17%，
**不同机制 / 不同卡 / 不同臂序**，互为独立验证。

---

## 5. null control：中间臂偏置 = `−0.18 ms`（更快）

campaign `.../k8-selective-20260811/null-20260811-131548`，三臂 `rc=0`。
`B_null` 用的就是 baseline runtime + `src_parent`，与两个 A 臂**逐字节相同**，
所以任何差异只能是位置与噪声。

```text
A1 33.889 / A2 33.922 / B_null 33.725
floor = 0.0165 ms      B_null delta = -0.1805 ms (-0.53%) = 10.9x floor
reset_body 2260.9 / 2256.5 / 2247.8   (delta -10.9 us)
三臂 sha 全 567b206b…
```

**为什么必须做这一步**：K9 的 `B +1.72 ms` 与 K8-selective 的 `B +1.89 ms` 都出现在
中间臂，而两个改动完全无关（一个改 kernel Wave3、一个改 host memset 范围），
量级巧合到不排除 harness 混淆就不能下结论。**判定：混淆不存在**，且中间臂偏置是
`−0.18 ms`（更快）⇒ 两个结论不仅成立，按偏置校正后**真实效应还要再大 0.18 ms**。

副产物：`floor` 在这台机上可小到 `0.0165 ms`，说明 100-iter p50 重复性很好；
`B_null` 的 `−0.18 ms` 是 10.9× floor，本身是**可测的系统效应而非噪声**，
记为此后所有中间臂读数的已知偏置。

---

## 6. 三个负结果（都有长期价值）

### 6.1 第一版 selective（6 段 / 6 次 broadcast）= NO-GO

campaign `.../k8-selective-20260811/aba-20260811-125724`，runtime sha `fcb71f83…`。
**模型源码三臂完全相同，只有 runtime 文件不同** —— 最干净的 matched-source。
设备 trace 自证 `k8_control_bytes=47616`、`k8_control_range_count=6`。

```text
memset_all 2214.5 / 2226.0 / 2055.2      => reset 只省 155.4 us
reset_body 2236.3 / 2247.4 / 2086.4
A1 33.776 / A2 33.933 (floor 0.0785) / B 35.740  => ITL +1.8855 ms (24x floor)
```

比全量少 **673 倍字节**，`memset_all` 却只从 `2214.5 → 2055.2 µs`。解模型：设每次
broadcast 固定开销 `f`、32 MiB 数据成本 `c`，则 baseline `= f + c ≈ 2220`、
selective `= 6f + ε ≈ 2055` ⇒ **`f ≈ 342 µs`、`c ≈ 1878 µs`** ⇒ 字节省下的
`≈1878 µs` 几乎被 5 次额外 broadcast（`5×342 ≈ 1710 µs`）吃光。

⇒ **结论是「实现方式错了，不是想法错了」。** 正确实现必须是**一次** broadcast，
而 §1 的 control-prefix 重排让它不需要动 simpler 就能做到。**这条负结果直接
决定了最终实现形态**，不是弯路。

### 6.2 天花板探针失败 —— 语义无效的臂不能界定性能上界

为判断「reset 异步化」值不值得开，我想量出 reset 路径的**全部**成本，做法是加一个
**完全不 reset** 的臂。该臂语义无效（跨步残留会破坏 control 语义），因此它的读数
**不能**作为「reset 全部成本」的上界。

⇒ 纪律：**性能上界只能由语义等价的臂给出。** 这也直接约束 K10 的表述口径：
**不得**说「4.4 ms 异步化 reset」，**不得**引用这个被否证的 no-reset 探针。

### 6.3 v1 落地件不能原样进产品（codex landing review 抓到）

v1 对 WholeDecode 的 16-buffer 布局做全局假设，会影响非 WholeDecode 的 program。
v3 硬化为：**只对 WholeDecode 生效** + 指纹不匹配/名字变体 → **回退全清** +
新增确认臂。§3 已证硬化的性能代价为 `0.26× floor`（即无代价）。

### 6.4 一个实现坑（记录给后续）

v1 第一次上设备 `rc=1`：`NameError: worker_ranges` —— 删了绑定但后面 trace dict 还在
读它，`compile()` 看不出来。已加 AST 名字解析预检
`claude_check_undefined_names.py`（带正对照自测：对已知坏版本必须 flag、对 baseline
与修好版必须 pass），接进 driver 在**取锁前**跑。driver 的 `set -e` 正确中止并
释放了锁。

---

## 7. 结论与后继

| 项 | 值 |
|---|---|
| 源码级 ITL 收益 | **`33.84 → 32.08 ms` = −1.7455 ms / −5.16%（89.5× floor）** |
| `reset_body` | `2253 → 518 µs`（−1734.9 µs） |
| 精度 | byte-exact，`hidden_sha256 567b206b…` + `token 14371`，三/四臂全一致 |
| 清零量 | `32,063,232 B → 47,616 B`（0.1485%），一次 broadcast |
| 两个独立 bracket | v1 `−1.7505` / v3 `−1.7455`，差 `0.26× floor` |
| 镜像级复现 | p50 `32.14 ms`，见 [image benchmark](2026-08-11-k8-selective-window-zeroing-image.md) |

**后继 K10**：reset 路径仍有**一次阻塞** host↔device control 往返，上界
**`0.45–0.53 ms/step`**（由 §4 的 `472.3 µs` wall / `920.4 µs` ITL 常数界定，
**不是**由 §6.2 那个无效探针界定）。实施顺序：① device 侧 zero prologue →
② request epoch/generation → ③ async / 双缓冲。评估只认整网 A/B/A + byte-exact。

所有 campaign 原件在 `0162:/mnt/persist/chensiyu/workspace/k8-selective-20260811/`。
