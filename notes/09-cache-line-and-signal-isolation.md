# 09 Cache Line 与 Signal 隔离笔记

> 目标：解释 `single cache line` / `cache-line isolation` 是什么，以及为什么在 PyPTO 跨 rank 通信、atomic notify/wait、N1 stall 定位中需要关注它。本文是学习与调试笔记；架构约束仍以 `pypto_top_level_documents/pypto-runtime-arch-docs/02-logical-view/04-memory.md` 等权威文档为准。

---

## 1. cache line 是什么？

**cache line（缓存行）** 是硬件缓存、互联、内存一致性管理时使用的最小内存块。

程序里看起来是按 byte / int32 / tensor element 访问：

```text
flag[0] = 1       # 只写 4B
```

但硬件通常不是只搬这 4B，而是把它所在的一整块 cache line 一起纳入缓存、一致性或互联事务：

```text
cache line size = 64B 时：
访问 0x1004，硬件实际管理 0x1000 ~ 0x103f

cache line size = 512B 时：
访问 0x1104，硬件实际管理 0x1000 ~ 0x11ff
```

在 N1/A2A3 调试语境中，已知需要重点关注的是 **512B L2 cache line / GM-NoC burst 粒度**。因此跨 rank control-plane signal 不能只按“8 个 int32 = 32B”来理解，还要看它们是否和相邻 buffer 落在同一个 512B 硬件管理单元里。

---

## 2. single cache line 有两种常见含义

### 2.1 多个变量落在同一条 cache line 里

例如 cache line 是 512B：

```text
0x1000 ~ 0x11ff 是同一条 cache line
```

如果两个变量地址是：

```text
count_done[0] = 0x1010
count_done[1] = 0x1014
```

它们是两个不同的 int32，但在硬件眼里共享同一个 cache line。

### 2.2 某个热点对象独占一条 cache line

调试 atomic / signal 时更常说的是这个意思：

```text
count_done 独占 0x1000 ~ 0x11ff
combine_done 独占 0x1200 ~ 0x13ff
```

也就是说，一个热点 signal / counter / atomic flag 周围不要放别的热点变量或大数据 payload。这个做法通常叫：

```text
cache-line isolation
cache-line padding
single-writer-per-line
独占 cache line
```

---

## 3. 为什么 signal 要考虑 cache-line isolation？

跨 rank 通信中的 signal 通常不是普通数据，而是 control plane：

```text
rank A notify / AtomicAdd / Set
rank B wait / TWAIT / Ge
```

典型对象包括：

```text
count_done
data_done
combine_done
attn_sig
mlp_sig
sh_sig
```

如果这些 signal 和相邻 payload 共用同一条 cache line，例如：

```text
pub_counts_buf_L0      offset 0xe00e0, size 10240
count_done_buf_L0      offset 0xe28e0, size 32
recv_x_buf_L0          offset 0xe2900, size ...
```

那么从程序逻辑看：

```text
count_done 是 32B signal
recv_x 是大 payload
```

但从硬件粒度看，它们可能共享同一个 512B line：

```text
|---------------- 512B cache line ----------------|
              pub_counts tail | count_done | recv_x head
```

这会带来几个风险。

### 3.1 false sharing

不同 rank / worker 写的是不同地址，但如果地址落在同一 cache line，硬件一致性或内存引擎仍可能以整条 line 为单位仲裁。

结果是：

- 不同 flag 的 atomic 更新互相串行化；
- wait/notify 变慢或抖动变大；
- control-plane signal 被 data-plane 大块 load/store 干扰；
- 某些平台上同线 atomic hotspot 会显著放大长尾延迟。

### 3.2 control plane 与 data plane 互相污染

通信协议里通常希望：

```text
payload 写完 / 可见
-> signal notify
-> peer wait 通过
-> peer remote_load payload
```

如果 signal 与 payload 共享硬件 cache line，那么 signal 这个小控制对象可能被 payload 的大块读写牵连，导致定位 stall 时现象非常混乱：

```text
看起来卡在 TWAIT / signal helper
实际可能是 signal 所在线被相邻 payload 或同线 atomic 干扰
```

这不一定是最终根因，但它是跨 rank kernel stall 的高优先级候选风险。

---

## 4. 只把 signal buffer 扩到 512B 一定够吗？

不一定。

例如：

```text
cache line = 512B
signal base = 0x1080
signal size = 512B
```

这个 signal 覆盖：

```text
0x1080 ~ 0x127f
```

它实际跨了两条 cache line：

```text
0x1000 ~ 0x11ff
0x1200 ~ 0x13ff
```

所以它仍可能和前一个 buffer 共享第一条 line，和后一个 buffer 共享第二条 line。

真正理想的“独占一条 512B cache line”至少需要：

```text
base % 512 == 0
size >= 512
```

也就是：

```text
signal base = 0x1000
signal size = 512B
覆盖 0x1000 ~ 0x11ff
```

因此调试时要同时检查：

1. 逻辑 shape 是否仍是 `[n_ranks, 1] INT32`；
2. 物理 allocation 是否至少有一个 cache line；
3. base offset 是否按目标 cache line 粒度对齐；
4. signal 前后是否还与 payload 共线；
5. 不要无脑给所有 data buffer padding，优先隔离 control-plane signal。

---

## 5. 在 PyPTO / N1 通信里的落地口径

以 N1 whole-net 的 pull + pull 通信为例，常见信号包括：

```text
# dispatch pull
count_done_sig
data_done_sig

# combine pull
combine_done_sig

# dense / shared path
attn_signal_window
mlp_signal_window
sh_signal_window
```

逻辑上这些 signal 可以继续保持：

```python
pld.window(signal_buf, [n_ranks, 1], dtype=pl.INT32)
```

因为协议只需要 `n_ranks` 个 int32 cell。

但物理分配上，不应该简单等同于：

```python
signal_buf = pld.alloc_window_buffer(n_ranks * 4)  # 8 ranks -> 32B
```

更合理的调试方向是将 control signal 作为 control-plane hotspot 隔离，例如按平台 cache line 粒度申请独占空间：

```python
COMM_SIGNAL_BYTES = 512  # A2/A3 调试假设；最终应来自平台 descriptor/架构约束
signal_buf = pld.alloc_window_buffer(COMM_SIGNAL_BYTES)
```

注意：这只是“物理容量隔离”的一部分；如果 allocator 不保证 512B 对齐，还需要从生成的 `host_orch.py` / `CommBufferSpec` / 实际 offset 继续确认：

```text
signal offset % 512 == 0
signal nbytes >= 512
signal 与前后 payload 不共享 512B line
```

---

## 6. 调试 checklist

定位跨 rank stall 时，建议对每个 signal buffer 做下面的表：

```text
buffer name
logical shape / dtype
physical nbytes
physical offset
offset % 512
previous buffer / next buffer
producer kernel / consumer kernel
notify op: Set or AtomicAdd
wait condition: Ge or Eq
是否跨 rank 多 writer
是否和 payload 共 cache line
```

如果出现：

```text
nbytes = 32
或 offset % 512 != 0
或 signal 前后紧贴大 payload
```

则应把它标为 control-plane false-sharing 风险。

但结论要谨慎：

```text
signal cache-line 共用 = 候选风险 / 可疑点
不是自动等于 stall 根因
```

真正确认需要单变量 A/B：

1. 不改 W8A8 数学；
2. 不改 dispatch/combine 协议语义；
3. 不改 logical shape；
4. 只改 signal 物理隔离；
5. fresh build + canonical 测试；
6. 如果仍 stall，再继续看 barrier ordering、rank mapping、buffer lifetime。

---

## 7. 一句话总结

**cache line 是硬件管理内存可见性、缓存和传输的基本块。single cache line 隔离的作用，是让热点 signal / atomic flag 不和其他变量共享这个硬件管理单元，从而降低 false sharing、atomic 串行化以及 control-plane/data-plane 干扰。**

在 N1 stall 定位中，`count_done` / `combine_done` 等 signal 只有 32B 且可能贴着大 payload，是值得验证的候选风险；但必须通过严格单变量实验确认，不能把 padding 假设直接写成已证明根因。

---

## 8. 0162 最终 device A/B 记录（2026-07-16）

本节记录当前 release 的最小布局变量和验证边界。它覆盖的是
`whole_decode_faithful_real` 的 standalone canonical P42，不是 live serving
的 token-exact 结论。

固定组合：

```text
machine = gpu-a910x-0162
devices = 8..15
P_FAITHFUL_MOE_LAYERS = 42
weights = native W8A8 IPC
KV = IPC
dispatch = fixed-slot pull
combine = pull
token = 6127
golden argmax = 303
```

最终最小 layout A/B 是 control signal 的 physical allocation：

```text
logical view: [8,1] INT32 = 32B
historical physical allocation: 32B
release physical allocation: 512B
COMM_CONTROL_SIGNAL_BYTES = 512
signal count = 216
216/216 physical nbytes = 512
all relative offsets % 512 = 0
window size = 766525440B, % 512 = 0
```

同一最终源码与 fresh exporter pool 连续运行 20 次：

```text
20/20 PASS
each argmax = 303
runtime min/mean/max = 2.53 / 2.5685 / 2.62s
```

最终整理后的 smoke 也通过（`2.57s`, `argmax=303`）。20-run 和 final
smoke 的 dmesg 时间窗均没有新增 `507018`、`running-stalled`、
`stranded CQE`、devmm/page fault、illegal VA/instruction 或 DMA/UB fault。

结论必须保持谨慎：512B signal isolation 相对历史随机 stall 是**强因果
证据**，但不是 bit-level hardware proof；不能据此单独声称已经证明某个
signal bit、某个 TPUT 或某个 stuck kernel 是唯一根因。完整测试入口见
[`../N1-CANONICAL-TEST.md`](../N1-CANONICAL-TEST.md)。
