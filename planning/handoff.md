# 接力上下文（Handoff）

> **只描述下一位 agent 现在要接的工作。最后更新：2026-08-12。**
> 当前状态以 [`../STATUS.md`](../STATUS.md) 为准。

## 1. 当前判定

TP all-reduce 单行优化已完成源码落地和 source-overlay 最终门：

```text
pypto-lib stepfun/develop
  HEAD    69ad31e4fd6e40b30e43c2566ce8f8ebd0b2427d
  parent  9ca01d243e534949287fa769e5be35031ebc4be7
  tree    e26d762cb8c4abd49a1546e7db2beddeb6480e14
```

- GitHub remote 与 0162 指定 checkout 对齐、clean；
- Main BS1 使用静态 `1×4096` 两波 one-shot mesh；
- Main 多行与 MTP 使用静态三波 fallback；
- Whole A/B/A：`31.065 / 29.912 / 30.999 ms`；
- candidate：`-1.120 ms / -3.609%`；
- precision/per-iteration gate：PASS；
- immutable image：**未构建**。

## 2. 源码位置与最终实现

0162 指定 checkout：

```text
/mnt/persist/chensiyu/workspace/develop/pypto-lib
branch  stepfun/develop
HEAD    69ad31e4fd6e40b30e43c2566ce8f8ebd0b2427d
origin  69ad31e4fd6e40b30e43c2566ce8f8ebd0b2427d
status  clean
```

最终 selector：

```text
Main active_rows == 1
  static 1×4096 self-TPUT
  -> Wave 1 publication
  -> fixed rank-order full-row remote loads
  -> one FP32 accumulator / one final BF16 cast
  -> Wave 2 completion

Main active_rows != 1
  static three-wave reduce-scatter + push-all-gather fallback

MTP
  shared ABI, three calls pass static BATCH, always static fallback
```

ownership 必须保持：

```text
TP_ALL_REDUCE_OWNED_CHUNK = HIDDEN // TP_WORLD_SIZE = 512
```

它与 `TP_ALL_REDUCE_CHUNK` 的 staging/final-copy transfer grain 解耦。

源码兼容性提醒：`dense_mlp_body_tp` 在 `mlp_layer_idx` 后新增了
`num_tokens: pl.Scalar[pl.INT32]`。仓内 Main 已传运行时 `num_tokens`，MTP 已传
静态 `BATCH`。仓外的直接调用方以及
`pl.inline(dense_mlp_body_tp._func)` 调用方升级时必须同步补该位置实参；这是源码
调用 ABI 变化，不能沿用旧参数表。

## 3. 最终验证

```text
canonical/two-layer AST       FINAL_STATIC_SELECTOR_CONTRACT_PASS
unit                           365 passed, 7 skipped
ruff / diff-check              PASS
Whole compile default          PASS
Whole compile chunk=256        PASS
MTP 3/3 default                PASS
MTP 3/3 chunk=256              PASS
8-card rows 1/3/16             PASS
```

rows `1/3/16` 覆盖 single-row smallmesh 与两档 multi-row fallback；该 device
matrix 未持性能锁，只是功能证据。

Whole BS1 / ctx64K / 512 blocks / warmup 10 / measured 100：

```text
A1 9ca static fallback p50     31.065 ms
B  final-tree smallmesh p50    29.912 ms
A2 9ca static fallback p50     30.999 ms
baseline center                31.032 ms
candidate delta                -1.120 ms / -3.609%
performance                    IMPROVEMENT_BEYOND_BRACKET
precision/per-iteration        PASS / PASS
```

B 臂 `b67afe77` 与最终 landing `69ad31e` 的 Git tree 相同。三臂 hidden SHA：

```text
567b206bb03d89f84020e1dddd61098a8f79f32f81b8f4fcf56443113e27f03e
```

三臂 finite、TP spread=0、tail token `14371` exact。

five-layer 只声明 L3/L4 exact、finite、TP spread=0，并提取了 regular-call
kernel-duration pooled mean；
既有 zero-token canonical structural analyzer 仍 fail-closed。

## 4. 权威产物

```text
/mnt/persist/chensiyu/workspace/perf-2026q3/
  tp-allreduce-hccl-smallmesh-validation-20260812/final-static-fallback/

whole-aba/out/final-aba-bs1-ctx64k-20260812-174433/ABA_RESULT.json
  sha256 383caa23124c7da42d676ef642bc8b488344349564fd4131efa560c6b5ea3757
```

固定验证镜像：

```text
manifest sha256:076af8a167405d5d0831e234cd16521c77d8bfdd173eff063d820802057c47f3
config   sha256:a9d111880883cea0b02e425fdfeaccc2b14bb1d1174c0b73488d8ee6d8004d39
```

镜像内 `pypto-lib` 仍为 `cb96747e`；`69ad31e` 是 read-only source overlay，
runtime 无 overlay。

## 5. 已退休的分支

### `a791071` attention-inline Ring

该实验没有命中 production canonical `WholeDecodeStep3p5.tp_all_reduce`，实质为
A/A；compile OK 不能推断 device correctness 或性能。不得继续扩展或恢复。

### `b4d45b3` K6b dynamic-valid-shape

动态 publish/final-copy 虽能过部分 codegen，但 self-TPUT/remote-load 仍受静态
shape 约束。dynamic publish 位于已知 notify-fence seam；现有设备运行未复现错误，
但没有针对该 seam 的独立 rank-skew/zero-gap/多 epoch safety proof。该分支只保留为
focused 历史证据，不得写成“可进产品、不必上卡”。

## 6. 正确性硬约束

1. `active_rows` 必须在所有 TP rank 上一致，否则 selector 分叉会死锁；
2. 固定 peer 顺序、单 FP32 accumulator、一次 BF16 cast 不得改变；
3. 两波与三波不能复用未清零的同一 signal slot；
4. exact two-layer mirror 必须继续与 canonical body AST 一致；
5. 本实现不等于修复 notify fence；未来合并波次或把 payload store 与自己的
   credit 拉近，仍受 `UPSTREAM-NOTIFY-FENCE` 阻塞；
6. 仓外 `dense_mlp_body_tp` 调用点必须传新增的 `num_tokens` 实参。

## 7. 下一步

1. 基于 `pypto-lib@69ad31e` 构建 immutable candidate image；
2. 固定 manifest/config 和所有组件 pin；
3. 在新镜像上重跑 Whole A/B/A、Main N=128、多 batch、MTP、canonical
   structural analyzer；
4. source-overlay 与 image gate 分账；新镜像闭环前不得写
   production/release-qualified；
5. 不再启动 Ring 或 dynamic-valid-shape 产品化，除非出现新的独立证据。

## 8. 机器与操作约束

后续启动前重新检查锁、container、`fuser` 与 NPU process，不能沿用旧 session
的空闲结论。

禁止事项：

- 在本地项目仓创建或修改 pypto-lib 产品代码；
- 用未持锁的 device matrix 作为性能数据；
- 把 focused regular-call kernel-duration pooled mean
  `38.325 → 22.667 µs/call` 当作 strict critical-tail 或最终完整源码 A/B/A；
- 用 host 独立检查覆盖 canonical structural fail-closed；
- 把 source-overlay 数据写成 immutable-image 结果。
