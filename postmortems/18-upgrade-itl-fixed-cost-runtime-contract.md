# 专项：升级 ITL 固定 host 开销与 H4 运行合同混写

| 字段 | 值 |
|------|----|
| **子系统** | whole-net / performance / deployment |
| **error signature** | 同一工作点出现 `47.993 / 27.812 / 22.253 ms` 三种 ITL |
| **首次出现** | 2026-08-23 |
| **状态** | ✅ 已解（2026-08-29 deployment launcher 接线） |
| **相关证据** | [`../benchmark/2026-08-29-h4-resident-deployment-contract.md`](../benchmark/2026-08-29-h4-resident-deployment-contract.md) |

## 1. 背景（Background）

全栈升级的性能目标是 0162、BS1、ctx 64K、1000 iterations 的 ITL 对齐约
`26.5 ms`。早期候选 `cafbc4d9…`、最终 r9 默认运行与最终 r9 H4 运行先后给出了三个
看似互相矛盾的数字。

## 2. 现象（Symptom）

| 对象 | 运行合同 | 64K/1000 p50 |
|---|---|---:|
| 早期升级候选 `cafbc4d9…` | 默认 | `47.993 ms` |
| r9 `b637f00c…` | unset = `none` | `27.812 ms` |
| 同一 r9 `b637f00c…` | `PYPTO_H4_RESIDENT=all` | `22.253 ms` |

最容易犯的错是把后两项都简称成“digest-bound r9 性能”，从而把差异归咎于机器负载；
或反过来把早期 `node.graph_build` 与后来的 H4 `bind.args` 当成同一个开销。

## 3. 根因（Root Cause）

这是两个独立固定 host 开销，必须分账：

1. 早期 `cafbc4d9…` 每步出现 `node.graph_build` p50 `8.326 ms`；该候选已被最终 pins
   取代，r9 默认值回到 `27.812 ms`。从旧候选到最终 r9 的精确 resolving commit
   未做单变量 bisect，因此文档只能说“最终 pins 已消除发布阻塞”，不能伪造 commit 归因。
2. r9 默认仍把 4 个 RoPE 表和 4 个 gate-R 常量作为 host tensor 每步 staging。
   `PYPTO_H4_RESIDENT=all` 把 8 个静态参数一次上传并重绑，令 `bind.args`
   p50 `6.461 → 0.063 ms`；`runner_run` 只从 `19.130 → 18.959 ms`。

四档 context 的 H4 收益都在 `5.36–5.76 ms`，所以主要差异不是随机负载。

## 4. 如何解决（Fix）

- 最终 r9 pins 取代早期候选，Main/MTP/precision/swimlane 全部门闭环；
- 发布性能合同显式设置 `PYPTO_H4_RESIDENT=all`，64K/1000 p50 `22.253 ms`；
- H4 `all/none` 在两套 oracle 上的输出 token 序列各自完全一致；
- 最终 release contract 明确记录 `h4_resident=all`，不再只写镜像 tag/digest。
- 2026-08-29 三个 canonical deployment launcher 默认显式注入 `all`，并保留
  `PYPTO_H4_RESIDENT=none` 回退；r12 matched A/B/A 收益 `7.372 ms / 24.591%`；
- 父 env unset 的 exact launcher 64K/1000 p50 `20.973 ms`、RC=0，完成 clean teardown。

H4 每 rank 额外占用约 `99.64 MiB` device memory。resident RoPE 不允许原地修改 host
副本；若常量生命周期变化，必须重建 holder 并重新做正确性门。

## 5. 走过的弯路（Detours / What We Got Wrong）

- ❌ 把 `27.812 → 22.253 ms` 初步描述成复测波动。runner diff 只有一行 H4 env，
  且 `bind.args` span 精确解释了约 `5.5 ms`。
- ❌ 只绑定 digest，不记录 effective env。digest 能证明代码/文件身份，不能证明运行合同。
- ❌ 把 H4 说成“修 node graph”。H4 消除的是静态参数每步 bind/H2D；旧候选的
  `node.graph_build` 是另一层开销。

## 6. 如何避免（Prevention）

- 每份性能报告至少绑定：manifest/config、完整命令、effective env、checkpoint、
  context/batch/blocks/warmup/iters 与锁/设备范围。
- 同 digest 数字不一致时，先 `diff` runner 和 env，再查负载；不要先猜硬件波动。
- 用环境变量开启的优化必须同时回答：代码默认是什么、镜像是否 bake、正式 launcher
  是否注入。三者没闭环时，只能写“image + env”的性能。
- launcher 默认、image Config 与代码默认必须继续分层记录；本次只关闭 deployment
  contract，不把 r12 描述成 bake 了 H4。
