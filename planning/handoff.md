# 接力上下文（Handoff）

> **这是 ephemeral 接力文档**——给“接着干”的 agent 一页纸当前工作面。
> durable 规划看 [`roadmap.md`](roadmap.md)，实时状态看
> [`../STATUS.md`](../STATUS.md)，B2 详细证据看
> [`../develop/OPT-B2-NEXT-SESSION-HANDOFF-20260725.md`](../develop/OPT-B2-NEXT-SESSION-HANDOFF-20260725.md)。
> **最后更新：2026-07-26。** 更新时直接改写本文，不追加流水。

## 1. 当前 active release

```text
machine:    gpu-a910x-0162
devices:    8..15（vanilla oracle 使用 0..7）
checkpoint: /data/chensiyu/step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp

pypto-lib / vllm-pypto:
  53eb7212c29c9bd015ee060cd9924a13ea781ae0
  branch stepfun/develop
pypto:   ca21ab5fcfd8203165928428302d273c377db5c6
simpler: 216e7632267ae815c484cdeba7991c87fabf3086
pto-isa: ecb6c303f797749f811a494742c3c08156aacabb
PTOAS:   fc8c6caee561914b4fb991dfc8427bb63194269e
ptoas:   0.50
```

当前 Main 默认入口：

```text
models.step3p5.decode_fwd:whole_decode_step3p5
```

- hidden-only harness 和 production sidecar 不传 Main 参数时默认使用
  canonical loop-form Main；
- `models/step3p5_opt` package、`whole_decode_opt` 和 `WholeDecodeOpt` 已删除；
- 只有显式 `--baseline-main` 才回退到 0724 unroll baseline；
- `--layer-module` 与 `--layer-name` 必须成对使用，且不能与
  `--baseline-main` 同时使用。

active pypto-lib checkout 只保留：

```text
workspace/pypto-lib  -> detached 53eb7212
workspace/pypto-lib-n1 -> detached 53eb7212（历史 dirty 已 stash）
workspace/vllm-pypto -> stepfun/develop @ 53eb7212
```

不要把 pypto-lib 内容覆盖到 `workspace/pypto`、`workspace/pto-isa` 或
`workspace/PTOAS`。已清理的 experiment checkout 不得重新当作 source of truth。

0162 其余 active toolchain checkout 也已与最终镜像统一：

```text
workspace/pypto          -> detached ca21ab5f
workspace/pypto/runtime  -> detached 216e7632
workspace/pto-isa        -> ecb6c303
workspace/PTOAS          -> detached fc8c6cae
```

历史 dirty 内容保存在 git stash；runtime 的旧 build backup 移到
`/tmp/pypto_workspace_dirty_backup_20260726/0162/`，未直接删除。

## 2. 已关闭的 B2 gate

固定 0724 镜像（历史 baseline 对照）：

```text
hub.i.basemind.com/stepcast/vllm-pypto:stepfun-develop-20260724
digest sha256:2b0dc4612796a34bea6720ccb4bf8fa3af4ea406cdd0f12add34586ca860d7e0
```

256-step 结果必须按两条 gate 分开写：

| gate | 结果 | 结论 |
|------|------|------|
| vanilla raw alignment | canonical `240/256=93.75%`；baseline `240/256=93.75%` | 历史 `>=95%` raw gate **未通过** |
| canonical replacement equivalence（历史 baseline 对照） | token `256/256` exact；hidden `256/256` exact；`max_abs_diff=0`；TP spread `0.0` | **PASS** |
| canonical-only 清理前后 | token `256/256` exact；hidden `256/256` bit-exact；`max_abs_diff=0`；step127/128/255 PASS | **PASS** |

raw miss steps：

```text
2, 20, 49, 52, 57, 62, 125, 131,
151, 153, 161, 162, 187, 221, 231, 252
```

历史 baseline 和清理前 canonical 均完全复现相同 raw 结果，所以该差异不是
loop-form replacement 或兼容入口清理引入的 regression。
不能把 `93.75%` 写成 vanilla precision PASS；也不能把 replacement PASS
扩写成 production Main+MTP serving 已完全平替。

设备回归 artifact：

```text
0162:/tmp/canonical_only_image_verify_20260726/smoke.log
0162:/tmp/canonical_only_image_verify_20260726/all_unit.log
0162:/tmp/canonical_only_image_verify_20260726/contracts.log
0162:/tmp/canonical_only_image_verify_20260726/audit.log
0162:/tmp/canonical_only_image_verify_20260726/n256_compare.log
0162:/tmp/canonical_only_n256_20260726/
0162:/tmp/canonical_only_n256_20260726.launcher.log
```

`53eb7212` 将同一已验证 loop-form 实现正式迁入
`models/step3p5/decode_fwd.py`；已删除 `step3p5_opt` package 和 opt aliases；与清理前
镜像产物逐 step bit-exact，因此没有修改数学实现。

## 3. 当前镜像发布

已发布并在 0162 复验的目标镜像：

```text
hub.i.basemind.com/stepcast/vllm-pypto:stepfun-develop-20260726-step3p5-only
digest: sha256:99b2b9718cfa6bf0bb87b221f7d565bf23afd2b89a30ba150e523c44a536ed81
config: sha256:d296461051559e6ea0e22d04a4cc44f749c82f19a50418fe6db75387f1f067e9
spec: deployment/docker/builds/stepfun-develop-20260726-step3p5-only.env
```

源码/镜像/设备侧已完成：

1. 六个源码 pin 与上节一致；
2. 不传 Main 参数时 holder 实际 program 为 `whole_decode_step3p5`；
3. canonical-only N=256 raw `240/256`，与清理前 canonical 产物 token/hidden
   `256/256` bit-exact、`max_abs_diff=0`、TP spread `0`；
4. step127、step128、step255 通过；
5. `--baseline-main` 成功编译为
   `WholeDecodeFaithfulRealSingleChipHiddenOnly` 并完成 3-step device smoke；
   step2 仅因已知 stale hardcoded oracle 预期退出。
6. 镜像内 `ptoas 0.50`、`/workspace/pypto-smoke.sh`、
   Git credential audit 和 canonical-only symbol audit 全部 PASS；
7. 新镜像默认 8-step device smoke 实际打印
   `program=whole_decode_step3p5`；hidden 全 finite、TP spread `0.0`，
   仅已知 stale-oracle step2 不匹配，其余 `7/8` exact。artifact：
   `0162:/tmp/canonical_only_image_verify_20260726/`。
8. 新镜像完整 N=256 canonical regression：
   raw `240/256=93.75%`，16 个 miss step 与此前对照完全一致；
   与既有 canonical artifact token/hidden `256/256` exact，
   `max_abs_diff=0`、TP spread `0.0`，step127/128/255 PASS。artifact：
   `0162:/tmp/canonical_only_n256_20260726/`。

历史 `opt-b2@sha256:0b22fcef…` 是安全的 rename 前对照镜像；更早的
`sha256:285514c1…` 因 `.git/config` 留有 credential-bearing clone URL
已废弃，禁止部署。

操作命令与判读见：

- [`../deployment/docker/README.md`](../deployment/docker/README.md)
- [`../.claude/skills/pypto-image-verify/SKILL.md`](../.claude/skills/pypto-image-verify/SKILL.md)

## 4. 尚未关闭的边界

以下仍是独立工作，不与 B2 replacement regression 混写：

1. 独立 live vLLM front 接管；
2. current Main→MTP 的同代 absolute token/hidden oracle；
3. live per-layer paged KV bridge 与动态 batch 映射；
4. vLLM、exporter、sidecar 的 3-way HBM 收口；
5. C1 单 window / `moe_epoch` 通信优化。

## 5. 操作约束与残留

- 不操作 `/mnt/persist/chensiyu/perf-opt-ws`；
- 设备回归默认使用 8–15，启动前确认无其他 agent 的 device process；
- 禁止 `kill -9`，禁止 `npu-smi reset`；
- 清理前必须先停 exporter/worker 并确认设备进程退出；
- `workspace/_ws_archive_20260723` 仍有 root-owned 历史归档残留，不是 active
  source checkout，也未参与构建/运行；未获提权删除批准前不要强删；
- 旧 dirty checkout 已备份到 `/tmp/pypto_workspace_dirty_backup_20260726`。
