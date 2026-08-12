# Step3p5 TP all-reduce：small-message selector

> 日期：2026-08-12
> 最终源码：`pypto-lib stepfun/develop@69ad31e4fd6e40b30e43c2566ce8f8ebd0b2427d`
> parent：`9ca01d243e534949287fa769e5be35031ebc4be7`
> tree：`e26d762cb8c4abd49a1546e7db2beddeb6480e14`

## 0. 最终结论

最终产品方案不是固定 Ring，也不是 runtime dynamic-valid-shape：

- Main 的 `active_rows == 1` 走静态 `1×4096`、两波 one-shot mesh；
- Main 多行和 MTP 走已验证的静态三波 reduce-scatter + push-all-gather；
- reduce-scatter ownership 固定为 `HIDDEN / TP = 512`，与可调的 transfer
  chunk 解耦；
- 保持固定 rank 顺序、单个 FP32 accumulator、最终一次 BF16 cast；
- `a791071` 的 attention-inline Ring 与 `b4d45b3` 的 K6b dynamic-valid-shape
  均已 supersede，不得恢复为产品路径。

本轮只迁移 HCCL“按消息规模选择算法”的思路。没有复制或调用 HCCL executor，
也不能从 vLLM 的 `HcclAllReduce` 调用断言运行时一定选择 one-shot。

## 1. 为什么旧方案退出

### 1.1 `a791071` Ring：实验未命中 production

`perf/tp-allreduce-ring-20260812@a791071` 只改了
`attention_full.py` / `attention_swa.py` 内的 standalone body。实际 Whole、
five-layer 与 exact two-layer mirror 调用的是
`WholeDecodeStep3p5.tp_all_reduce`；因此当时的 compile/运行对照没有切换
production canonical body，本质是 A/A。

即使手写 Ring 可编译，也不能据此推断可 dispatch、正确或更快。历史 ring 还受
多轮串行依赖、payload-before-credit fence 和设备 codegen 约束，不再作为落地方向。

### 1.2 K6b dynamic-valid-shape：静态 shape 与 publication 门未闭合

`perf/tp-allreduce-canonical-opt-20260812@b4d45b3` 尝试只在 publish/final-copy
使用 runtime valid shape。该方案能生成动态 partition view，但：

- self-target TPUT 的 destination 必须是正的静态 shape；
- `remote_load` 没有独立 `valid_shape=` 合同；
- dynamic publish 位于已知 notify-fence seam；现有设备运行未复现错误，但没有
  针对该 seam 的独立 rank-skew/zero-gap/多 epoch safety proof；
- compile/codegen 通过不能替代多 epoch、跨 rank 与 Whole 精度门。

所以 K6b 只保留为历史 focused 证据，不进入产品。

## 2. 最终 selector

### 2.1 单行：两波 one-shot mesh

Main 从所有 TP rank 一致的 `active_rows` 选择单行路径：

```text
payload = 1 × 4096 × BF16 = 8192 B

static 1×4096 self-TPUT
  → Wave 1 publication notify/wait
  → 按 rank 0..7 remote-load 完整行
  → 单 FP32 accumulator，最终一次 BF16 cast
  → Wave 2 completion notify/wait
```

Wave 1 保证所有 source partial 已发布；Wave 2 保证所有 peer 读取结束后才结束
window 生命周期。固定 peer 顺序保持数值合同。

`active_rows` 必须在所有 TP rank 上一致，否则不同 rank 会进入不同协议并死锁。
当前 holder 把同一个 `valid_tokens` 写给所有 TP owner，因此合同成立；未来调用点
必须继续守住该约束。

### 2.2 多行与 MTP：静态三波 fallback

`active_rows != 1` 的 Main 保留：

```text
static full-capacity self-TPUT
  → Wave 1
  → static reduce-scatter
  → push all-gather
  → Wave 2
  → static final local copy
  → Wave 3
```

MTP 共享 all-reduce ABI，但三个调用传入静态 `BATCH`，明确走这条 fallback。
本轮不宣称 MTP 命中或受益于 single-row selector。

### 2.3 ownership 与 transfer grain

```text
TP_ALL_REDUCE_OWNED_CHUNK = HIDDEN // TP_WORLD_SIZE = 512
TP_ALL_REDUCE_CHUNK       = transfer/staging grain
```

前者决定每个 rank 负责的 reduce-scatter hidden 区间；后者只控制 self-TPUT 与
final-copy 的搬运粒度。两者不可混用。否则 `chunk=256` 会漏归约半个 hidden，
更大的 chunk 也可能越过 ownership 边界。

## 3. 调用范围

最终提交将 rank-uniform selector 参数贯穿：

- Full/SWA attention o_proj；
- dense MLP；
- Whole 中 shared-expert 路径；
- exact two-layer mirror 与 single-layer probe；
- MTP shared ABI（但静态选择 fallback）。

two-layer `tp_all_reduce` body 与 canonical AST 精确一致，防止 harness 再次测到
与 production 不同的算法。

这次贯穿同时改变了 `dense_mlp_body_tp` 的源码调用 ABI：在
`mlp_layer_idx` 后新增 `num_tokens: pl.Scalar[pl.INT32]`。仓内 Main 调用传运行时
`num_tokens`，MTP 调用传静态 `BATCH`；任何仓外直接调用该函数或
`pl.inline(dense_mlp_body_tp._func)` 的代码，升级到 `69ad31e` 时也必须在同一位置
补该实参，否则不能按旧参数表继续调用。

## 4. 正确性与生命周期边界

- peer 累加顺序固定为 rank `0..7`；
- FP32 中间累加，最终只 cast 一次 BF16；
- signal slot 结束值两波为 2、三波为 3；两种路径不能复用未清零的同一 slot；
- 当前 attention/dense、各 layer 使用独立 slot，请求间由 host drain/reset；
- 本方案依赖 self-TPUT 的同步 publication 与 completion wave，并不等于修复
  上游 notify fence 缺口；
- 任何未来将 payload store 与自己的 credit 拉近、合并波次或改为 peer-fused
  store+notify，仍须先解决 `UPSTREAM-NOTIFY-FENCE`。

## 5. 最终验证

### 5.1 静态、单测与编译

| 验证项 | 结果 |
|---|---|
| canonical/two-layer AST | `FINAL_STATIC_SELECTOR_CONTRACT_PASS` |
| unit | `365 passed, 7 skipped` |
| targeted ruff / diff-check | PASS |
| Whole compile | default chunk、chunk=256 均 PASS |
| MTP compile | 3/3 programs；default、chunk=256 均 PASS |
| 8-card device matrix | rows `1/3/16` finite、TP spread=0、审计 PASS |

device matrix 未持性能锁，只作为功能与协议正确性证据。

### 5.2 Whole BS1 / ctx64K A/B/A

固定镜像上的 source-overlay A/B/A：

| arm | source | p50 |
|---|---|---:|
| A1 | `9ca01d2` static three-wave baseline | `31.065 ms` |
| B | final-tree-equivalent smallmesh | `29.912 ms` |
| A2 | `9ca01d2` static three-wave baseline | `30.999 ms` |

```text
baseline center       31.032 ms
candidate delta       -1.120 ms / -3.609%
performance verdict   IMPROVEMENT_BEYOND_BRACKET
precision gate        PASS
per-iteration gate    PASS
```

B 臂 `b67afe77` 与 landing `69ad31e` 的 tree 均为
`e26d762cb8c4abd49a1546e7db2beddeb6480e14`。三臂 hidden SHA256 全等：

```text
567b206bb03d89f84020e1dddd61098a8f79f32f81b8f4fcf56443113e27f03e
```

三臂 finite、TP spread=0、tail token `14371` exact。

### 5.3 focused DFX 的使用边界

历史 focused K6b-vs-smallmesh regular-call kernel-duration pooled mean 为
`38.325 → 22.667 µs/call`（`-40.9%`，8 ranks × 7 calls）。该口径排除首次同步
污染与 local-setup 大值，不是 strict critical-tail；它只用于解释协议机制，不代表
该 focused campaign 的完整 source tree 就是最终 `69ad31e`。

five-layer 只声明 L3/L4 exact、finite、TP spread=0，并提取了 regular-call
kernel-duration pooled mean；
不覆盖既有 zero-token canonical structural fail-closed。

## 6. 权威产物与发布边界

```text
/mnt/persist/chensiyu/workspace/perf-2026q3/
  tp-allreduce-hccl-smallmesh-validation-20260812/final-static-fallback/

whole-aba/out/final-aba-bs1-ctx64k-20260812-174433/ABA_RESULT.json
  sha256 383caa23124c7da42d676ef642bc8b488344349564fd4131efa560c6b5ea3757
```

验证镜像：

```text
manifest sha256:076af8a167405d5d0831e234cd16521c77d8bfdd173eff063d820802057c47f3
config   sha256:a9d111880883cea0b02e425fdfeaccc2b14bb1d1174c0b73488d8ee6d8004d39
```

镜像内 `pypto-lib` 仍为 `cb96747e`。所有最终数据均为该 immutable image 上的
source overlay；尚未构建包含 `69ad31e` 的 immutable image，不能写成 production
release qualification。

## 7. 后续

1. 构建固定 `pypto-lib@69ad31e` 的 immutable candidate image；
2. 在该镜像上重跑 Whole A/B/A、Main N=128、多 batch、MTP 与 canonical
   structural analyzer；
3. 保留 single-row / multi-row selector、ownership/chunk 解耦和 exact mirror
   合同测试；
4. 在 immutable gate 完成前，不再启动 Ring 或 dynamic-valid-shape 产品化。
