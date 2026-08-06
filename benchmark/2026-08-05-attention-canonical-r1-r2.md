# Attention canonical R1/R2 immutable 验证记录（2026-08-05）

> **HISTORICAL / SUPERSEDED（2026-08-06）**：本文只保留 R1 失败和 R2
> 未发布的过程证据，不是当前任务或恢复指引。R2 已被 pypto-lib `c9af5790` 的
> latest-source canonical image 取代，**不得恢复**。当前 manifest、ITL、DFX 与
> swimlane 见
> [`2026-08-06-attention-taskmajor-canonical.md`](2026-08-06-attention-taskmajor-canonical.md)。

## 结论

- **R1 已废弃，不是 canonical 交付镜像。** 两层 bs1/context=65536 的 50 次
  timing、数值一致性均完成，但 immutable DFX 在创建 `RunConfig` 时失败，
  因而没有生成可交付的 swimlane。
- **R2 从未发布，现已 supersede。** 当时固定了 PyPTO 修复和串行构建参数，
  但构建被停止，没有 manifest/config digest 或 ITL/DFX 结论。
- 不允许通过挂载宿主源码或 runtime overlay 绕过 R1 缺失接口；也不得继续构建
  R2。当前验证对象是 2026-08-06 `c9af5790` 镜像。

## R1：失败证据

镜像：

```text
tag:
hub.i.basemind.com/stepcast/vllm-pypto:
stepfun-develop-20260805-attn-final-canonical

manifest:
sha256:fb613c2d5a74592f248c6d923e3ada6582edbe40349ada530017e622ca735b23

config:
sha256:95bf9657adc09650fc85c23544756169519f85c145b42b14641bfc41e6c173e2
```

源码 pin：

```text
pypto      defa97c526fec7e8f032dbbfcc39c820add02bf7
pypto-lib  91c7f46ee949045e2fce807276412b48d8121763
```

bs1/context=65536、前两层 attention 的 50 次 timing：

| 指标 | 结果 |
|---|---:|
| min | `3.6147 ms` |
| p50 | `3.7181 ms` |
| mean | `4.0087 ms` |
| p99 | `9.9181 ms` |

正确性观测：

```text
output unique_count = 1
reference exact     = true
TP spread          = 0
```

随后 DFX 失败：

```text
TypeError: RunConfig.__init__() got an unexpected keyword argument
'l2_swimlane_reuse_dep_gen'
```

原因是 launcher 请求了 prepared swimlane 的 dep-gen reuse 接口，而 R1 镜像内
PyPTO pin 尚未包含该接口。失败目录仅用于保留失败证据：

```text
/mnt/persist/chensiyu/workspace/attn-opt/out/
image_attn_final_canonical_20260805_91c7f46e/
bs1_ctx65536_two_layer_dfx/
```

该目录**不包含最终可交付的 `l2_swimlane_records.json`**，上面的两层 timing
也不能替代同一最终镜像的整网 ITL。

## R2：历史修复规格与最终未发布状态

正式修复：

```text
pypto branch  stepfun/develop
pypto commit  8e92b46808f9f7c09b6431ad4691503f09c12ee5
test          88 passed, 3 skipped
```

R2 构建规格：

```text
deployment/docker/builds/
stepfun-develop-20260805-attn-final-canonical-r2.env
```

关键 pin/约束：

```text
PYPTO_COMMIT=8e92b46808f9f7c09b6431ad4691503f09c12ee5
PYPTO_LIB_COMMIT=91c7f46ee949045e2fce807276412b48d8121763
ATTN_TASK_PROFILE=a2a3
BUILD_JOBS=1
```

R2 构建在串行 PyPTO C++ editable build 阶段按用户要求停止。停止后已检查本机和
0162，均无残留 build/compile 进程。该历史对象的最终状态是：

```text
R1: REVOKED
R2: NEVER PUBLISHED / UNVERIFIED / SUPERSEDED
```

不得为 R2 填写或借用 R1/Wave5 的 manifest、ITL 或 swimlane 数据。

## 当时计划的准出顺序（已取消，禁止执行）

以下列表只解释 2026-08-05 为什么 R2 当时还不能交付；2026-08-06 后不得把它
当作待办。实际 gate 已由新镜像按
[`2026-08-06-attention-taskmajor-canonical.md`](2026-08-06-attention-taskmajor-canonical.md)
执行。

1. 当时计划保持 `BUILD_JOBS=1` 完成 R2 构建，并执行 credential、pin、import 和
   `RunConfig` 接口审计。
2. push 后记录 R2 manifest/config digest；0162 只允许 digest-only 启动，不挂载
   宿主源码。
3. 显式清除八个 attention 单参数 override，只保留命名的 `a2a3` profile。
4. 先采 bs1/context=65536 前两层 attention DFX，返回具体 rank 的
   `l2_swimlane_records.json`。
5. 再在同一 R2 digest 上采整网 50 次 ITL，报告
   min/mean/p50/p99/max 和正确性结果。
