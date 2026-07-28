# 专项：TP all-reduce 改 reduce-scatter 后 8 卡结果不一致（`hidden_tp_spread != 0`），根因 = pull-form `store → notify → peer remote_load` 跨方向握手无序

| 字段 | 值 |
|------|----|
| **子系统** | whole-net / codegen（`models/step3p5/decode_fwd.py::tp_all_reduce`） |
| **error signature** | `hidden_tp_spread` 非零（2~58），token 仍可能全对；间歇性（部分 step 恰好 0.0） |
| **首次出现** | 2026-07-28 |
| **状态** | ✅ 已解（all-gather 改 push） |
| **相关 skill / doc** | [`design/performance/03-tp-allreduce-algorithm-comparison.md`](../design/performance/03-tp-allreduce-algorithm-comparison.md) §5、[`.claude/skills/pypto-dev-constraints`](../.claude/skills/pypto-dev-constraints/SKILL.md) §0.4/§4.2 |

## 1. 背景（Background）

按 [`03-tp-allreduce-algorithm-comparison.md`](../design/performance/03-tp-allreduce-algorithm-comparison.md) 的实测结论，把 canonical Main
（`models.step3p5.decode_fwd:whole_decode_step3p5`）的 `tp_all_reduce` 从
onephase full-mesh 换成 twophase_par（reduce-scatter + all-gather + 并行扇出），
以降低每卡远程读次数（56 → 14）与远程字节（896 KB → 224 KB）。

环境：0162，镜像 `vllm-pypto:step3p5-b404a3c9-ci-final-20260728`，cards 8–15，
`whole_decode_step3p5` 45 层、每层 2 处 all-reduce，共 90 个调用点。

## 2. 现象（Symptom）

编译干净，8-step token 与 canonical 基线完全一致，但 8 卡的 `next_hidden` 不再一致。

```
# baseline（onephase mesh）
tokens [303, 1207, 19384, 872, 428, 6127, 4231, 2636]
spread [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# twophase（pull-form all-gather）
tokens [303, 1207, 19384, 872, 428, 6127, 4231, 2636]      # 仍全对
spread [58.5, 4.125, 2.0, 2.59375, 4.0, 3.125, 2.6875, 2.50390625]
```

两个易误判的观测特征：

- **token 全对不代表通过**。token 只从 rank0 采样，rank 间不一致不会体现在 token 上；
  真正的回归信号是 `hidden_tp_spread`。
- **按 512 列分 shard 统计，8 个 shard 全部跨 rank 不一致，rank0 之外每个 rank 都差**。
  这是 45 层 × 90 次 all-reduce 逐层放大的结果，**输出端的 spread 无法定位首次发散点**。

只切 attention（45 个调用点）时才漏出关键线索——出现了两个恰好 `0.0` 的 step：

```
spread [24.0, 0.0, 2.15625, 2.0, 5.078125, 3.25, 2.1875, 0.0]
```

即**这是 race，不是系统性错误**（全量 90 点时每步几乎必中，所以看起来像系统性的）。

## 3. 根因（Root Cause）

**在当前 pypto/simpler runtime 下，`本地 store → 远端 notify → peer 远端 remote_load`
这条握手不保证顺序。** payload 落在我自己的 HBM，notify 落在对方的 HBM，是两个方向；
本地的 `V→MTE3` fence（codegen 确认存在）只能排空我自己的流水，**无法保证"对方看到
notify 时我的数据已可被对方读到"**。

为什么只有 twophase 暴露：

| 版本 | 每次调用中**带数据依赖**的 pull barrier |
|---|---|
| onephase mesh | **1 个**（wave1 后 pull staged 数据；wave2 是纯 completion，之后不再读远端） |
| twophase(pull AG) | **2 个**（wave1 后 pull staged；wave2 后 pull 别人归约好的 shard） |

暴露翻倍，从"侥幸不中"变成"几乎必中"。这与本设计文档 §5 判 ring 不可用的是**同一个上游缺口**
（ring 每次调用 14 个依赖握手，所以必挂），只是 §5 当时把它记成了"缺 `pipe_barrier(PIPE_ALL)`"。

**决定性实验**（同一接线、同一镜像，只改 all-gather 的方向）：

| all-gather 形式 | 8 步 `tp_spread` |
|---|---|
| pull（`remote_load` 别人的窗口） | `24.0, 0.0, 2.16, 2.0, 5.08, 3.25, 2.19, 0.0` |
| **push（`remote_store` 进别人的窗口）** | **全 `0.0`**（并经 3 次独立重复 24 步复现） |

## 4. 如何解决（Fix）

all-gather 由 pull 改 push，对齐 V4-Flash 的 dispatch/combine 写法
（`origin/main:models/deepseek/v4-flash/moe.py`，注释原文 "payload-arrival notify
folded into the push"）：算完自己的 shard 后直接 `pld.tile.remote_store` 推进每个 peer
的窗口，再 notify；**barrier 之后完全没有远端访问**，全部本地 `pl.load`。

```python
# Phase 3 末尾（owner == my_rank 分支内）
reduced_tile = pl.cast(acc, target_type=pl.BF16)
pl.store(reduced_tile, [0, base], local)          # 自己那段直接落 local
for dst in pl.range(group_size):                  # PUSH，与 notify 同方向
    if dst != my_rank:
        pld.tile.remote_store(reduced_tile, tmp_window, dst, [0, base])
# wave 2 (notify + wait Ge 2)
# Phase 5：纯本地读
peer_shard = pl.load(tmp_window, [0, off], [BATCH, shard])
pl.store(peer_shard, [0, off], local)
```

改动文件：`models/step3p5/decode_fwd.py`（`tp_all_reduce`）、`models/step3p5/dense_mlp.py`
（返回值赋值，见 §5）。

**数值等价**：每个 shard 仍按 canonical peer order `0..P-1` 用单个 FP32 累加器求和、
每元素只做一次 BF16 cast/store；且一个 shard 只由一个 rank 归约后广播，rank 相关的
rounding 在结构上被消除。

**适用边界 / 残留风险**：wave1 的 stage-in **仍是 pull**，同类弱序依赖还在，但
**暴露量与改前 mesh 完全相同（各 1 个）**，因此不构成回归。要彻底消除需把 stage-in 也
改 push；不新增窗口的话（rank r 把它对 shard s 的贡献推到 peer s 窗口第 `r*shard` 列，
8 来源 × 512 列 = HIDDEN，刚好装得下）会因"窗口列语义在两阶段不同 + 我推 reduced shard
会覆盖 peer 尚未消费的 Phase-1 数据"而需要第 4 个 wave；要避免就得引入第二个窗口，属
host ABI 改动。**记为后续独立项，不在本次改动内。**

## 5. 走过的弯路（Detours / What We Got Wrong）

**方法论层面的两个主错**（比单个技术假设更值得记）：

1. **在 45 层整网上按"机制"黑盒二分**，每轮 8–10 分钟，连续证伪 12 个假设。正确顺序应是
   **先按调用点二分**（让 mesh / twophase 两个 collective 并存，一次只切一类站点），
   把"哪一类站点"从"什么机制"里分离出来——正是这一步才漏出那两个 `0.0`。
2. **单次绿当成通过**。这是 race，必须重复采样。同理 pull 版早期"每步都非零"被我当成
   "系统性错误、非 race"的证据，方向就此偏掉。

**被逐个证伪的假设**（留档，勿重走）：

| # | 假设 | 证伪方式 |
|---|---|---|
| 1 | `pl.parallel` fan-out 缺跨核 join | 全 `pl.range` 变体同样发散 |
| 2 | 运行期标量列偏移（`my_rank*shard`）寻址错 | OLD mesh 也用运行期 `k0` 做列偏移且完全正确 |
| 3 | 90 个调用点槽位冲突/越界 | 逐一枚举：槽位唯一，`alloc_window_buffer` 字节与 max offset+extent 吻合 |
| 4 | barrier 阈值算错 | codegen 确认 3 个 wave `expected=1/2/3` 正确发射 |
| 5 | `local` 与 `tmp_window` 被 alias | 两版 `memory_after_AllocateMemoryAddr` 逐字节相同 |
| 6 | store→notify 缺 pipe 排空（局部 fence 缺失） | `V→MTE3 set_flag/wait_flag` 实际存在；且 read-back fence 被 DCE，那轮无效有解释 |
| 7 | stacked window 切片丢父 stride | bench 复现 PASS；codegen 证 origin 经 `start_offset` 保留、stride 相同 |
| 8 | 90 次调用/多槽位复用 | bench `--layers 42` PASS |
| 9 | 原地写回入参（in-place） | bench PASS（一度出现的 `max_diff=588` 是 golden 快照时机写错，mesh 对照组同样 588 已自证） |
| 10 | 跨函数传递窗口切片 | bench `--indirect`（外层切片 → 中间 Orchestration → InCore）PASS |
| 11 | producer 输出按列分片（非全列有效 partial） | 三类 producer 全部全 4096 列有效；`sh_tp_chunk` 是**零引用死常量** |
| 12 | `tp_all_reduce` 返回值被丢弃 | 修完仍发散（但**这本身是既存硬约束违规**，见下） |

**#12 虽未修好 bug，但确实是既存缺陷并已一并修掉**：`dense_mlp.py:240` 与
`decode_fwd.py` 两处 MoE shared-expert 调用点原本写成 `self.tp_all_reduce(...)`
（丢弃返回值），依赖"函数体内 `pl.store` 进普通 `pl.Tensor` 入参会原地改到调用方"的副作用。
`local` 不是 `Out`/`InOut`，只有 `x = f(x)` 才强制 must-alias。违反
dev-constraints §1.1「`Out`/`InOut`/返回值的物理 alias 和 ownership 必须明确」。
已全部改为赋值形式。

**bench（`allreduce_bench.py::_build_twophase_par`）的 golden PASS 不可外推**：
它是单次孤立 all-reduce，没有 45 层其它 kernel 制造的时序压力，因此对本 race 完全不敏感。
这正是 dev-constraints §4.2「probe 不能反向规定架构」与 §0.4「证据不能越级」的实例——
设计文档 §5 的"换 twophase_par"建议是基于这个 probe 得出的，落到整网即失效。
本次为二分专门写的 `ar_wn_bisect.py`（把 stacked 切片 / in-place / 42 次复用 / 跨函数传递
做成开关）在**四个条件全开时依然 PASS**，进一步印证了这一点。

**两个 review agent 各有一处误判**（复核 agent 结论时的提醒）：codegen agent 把两个不同
build 的特征混在一起，得出"Phase 3 从未 staging 的槽位读 own_tile"（与实际文件不符）；
source agent 关于 `pl.range(constant)` 不展开（只有 `pl.unroll` 展开）是对的、纠正了我的
错误认知，但由此推出的结论被 OLD 版同样用运行期 `k0` 却完全正确所推翻。
两者都没直接命中，但把搜索空间收敛到了调用约定与通信方向这一层。
