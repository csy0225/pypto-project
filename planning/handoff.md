# 接力上下文（Handoff）

> **只描述下一位 agent 现在要接的工作。最后更新：2026-08-05。**
> 当前状态以 [`../STATUS.md`](../STATUS.md) 为准；历史过程不要复制回本文件。

## 1. 当前结论

- 用户要求暂停 build；本机和 0162 均无 R2 build/compile 残留进程。
- 当前源码已经推送：

```text
csy0225/pypto-lib:stepfun/develop
91c7f46ee949045e2fce807276412b48d8121763

csy0225/pypto:stepfun/develop
8e92b46808f9f7c09b6431ad4691503f09c12ee5
```

- `pypto-project/main` 已记录 R1 失败、R2 暂停和 DFX 修复。
- 本地曾有一个错误指向旧 N1 代码线的 `pypto stepfun/develop` 引用；已删除并按
  GitHub 远端重建到 `8e92b468`。任何 agent 都不得再从本地旧分支名推断当前 tip。

## 2. 镜像状态

### R1：废弃

```text
tag:      stepfun-develop-20260805-attn-final-canonical
manifest: sha256:fb613c2d5a74592f248c6d923e3ada6582edbe40349ada530017e622ca735b23
config:   sha256:95bf9657adc09650fc85c23544756169519f85c145b42b14641bfc41e6c173e2
status:   REVOKED
```

R1 两层 bs1/context=65536 的 50 次 timing 与数值检查完成，但 DFX 因镜像缺
`l2_swimlane_reuse_dep_gen` 失败，没有最终 `l2_swimlane_records.json`。
禁止源码挂载或 runtime overlay 绕过。

### R2：暂停

```text
spec:
/tmp/attn-final-canonical/deployment/docker/builds/
stepfun-develop-20260805-attn-final-canonical-r2.env

pins:
pypto      8e92b46808f9f7c09b6431ad4691503f09c12ee5
pypto-lib  91c7f46ee949045e2fce807276412b48d8121763
profile    a2a3
jobs       1

status: BUILD PAUSED / UNPUBLISHED / UNVERIFIED
```

当前没有 R2 digest、swimlane 或整网 ITL；不得借用 R1/Wave5 数据。

## 3. 用户恢复后立即执行的任务

1. 串行继续 R2 build，保持 `BUILD_JOBS=1`。
2. 本地执行 credential、pin、import、RunConfig 和 worktree-clean 审计。
3. push 并记录 R2 manifest/config digest。
4. 更新 0162 launcher 为 R2 digest；禁止源码挂载，显式清除八个 attention override。
5. 先跑 bs1/context=65536 前两层 DFX。
6. 再跑同一 digest 的整网 50 次 ITL。

交付必须包含：

```text
具体 rank 的 l2_swimlane_records.json 路径
ITL min / mean / p50 / p99 / max
数值一致性和 TP spread
镜像 manifest/config digest
```

## 4. 已有路径

```text
R2 build log:
/data/chensiyu/hw_project/pypto/workspace/attn-opt/build_logs/
stepfun-develop-20260805-attn-final-canonical-r2.build.log

两层 launcher（恢复后必须先替换为 R2 digest）:
/mnt/persist/chensiyu/workspace/attn-opt/
final_canonical_bs1_64k_two_layer_dfx.sh

整网 launcher（恢复后必须先替换为 R2 digest）:
/mnt/persist/chensiyu/workspace/attn-opt/
final_canonical_bs1_64k_whole_net_itl.sh

两层 reference:
/mnt/persist/chensiyu/workspace/attn-opt/out/
attn_a2a3_profile_64k_bs1_0_7_50x_20260805/outputs/context_65536.pt
```

## 5. 不得使用的旧信息

- 历史 N1 pin、stable-env 和旧 phase handoff：仅为历史证据。
- 本地分支名或 dirty worktree：不是 SSOT。
- R1 timing：不是 R2 整网 ITL。
- Wave5 swimlane/ITL：不能充当 R2 结果。
- 没有 digest 的 tag：不能作为 canonical 交付。
