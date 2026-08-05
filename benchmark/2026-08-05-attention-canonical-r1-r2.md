# Attention canonical R1/R2 immutable 验证记录（2026-08-05）

## 结论

- **R1 已废弃，不是 canonical 交付镜像。** 两层 bs1/context=65536 的 50 次
  timing、数值一致性均完成，但 immutable DFX 在创建 `RunConfig` 时失败，
  因而没有生成可交付的 swimlane。
- **R2 尚未构建完成。** R2 已固定正式 PyPTO 修复和串行构建参数；按用户要求，
  构建已暂停，当前没有 R2 manifest/config digest，也没有 R2 ITL/DFX 结论。
- 不允许通过挂载宿主源码或 runtime overlay 绕过 R1 缺失接口。恢复验证后必须使用
  R2 digest-only 镜像。

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

## R2：修复、规格与暂停状态

正式修复：

```text
pypto branch  release/attn-final-dfx-20260805
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
0162，均无残留 build/compile 进程。因此当前状态必须写作：

```text
R1: REVOKED
R2: BUILD PAUSED / UNPUBLISHED / UNVERIFIED
```

不得为 R2 填写或借用 R1/Wave5 的 manifest、ITL 或 swimlane 数据。

## 恢复后的准出顺序

1. 保持 `BUILD_JOBS=1` 完成 R2 构建，并执行 credential、pin、import 和
   `RunConfig` 接口审计。
2. push 后记录 R2 manifest/config digest；0162 只允许 digest-only 启动，不挂载
   宿主源码。
3. 显式清除八个 attention 单参数 override，只保留命名的 `a2a3` profile。
4. 先采 bs1/context=65536 前两层 attention DFX，返回具体 rank 的
   `l2_swimlane_records.json`。
5. 再在同一 R2 digest 上采整网 50 次 ITL，报告
   min/mean/p50/p99/max 和正确性结果。

