# 2026-09-01 · K8 local-owner persistent-reset 回退修复

> **状态：SRC 已推送；candidate IMG 的 correctness gate 已通过；**
> **正式 IMG release admission 仍待 `recv_meta` publication gate。**
> 本报告记录此前 local-owner 变更造成的 persistent-window reset 回退，
> 以及在最新匹配栈上的修复验证。性能数字严格区分旧 `e6c7d8ec`
> candidate 与新 `a745ab659` candidate；历史绝对值对账见
> [`2026-09-02-k8-historical-performance-reconciliation.md`](2026-09-02-k8-historical-performance-reconciliation.md)。

## 1. 结论先行

之前看到的 `21.562 ms` 并不是最新匹配栈的结果。该运行使用：

```text
pypto     655c7bda7b0a0b495a3387b2570ea68c4a857a40
pypto-lib e6c7d8ec34a05c3051ccf0dd169639f40f041a57
```

本次候选臂使用最新匹配栈：

```text
pypto     655c7bda7b0a0b495a3387b2570ea68c4a857a40
pypto-lib a745ab659c68afca01de37870e29ccb9648d7c87
```

在同一台 0162、H4=`all`、64K context、8 卡、`warmup=10`、
`iters=100` 的 immutable A/B/A 中，A1/A2 是 `r14a`
(`14de90fd+a745ab659`) 控制臂，B 是 `r14b`
(`655c7bda+a745ab659`) 候选臂：

| Arm | 栈 | ITL p50 |
|---|---|---:|
| A1 | r14a baseline (`14de+a745`) | `21.617 ms` |
| B | r14b candidate (`655+a745`) | **`20.516 ms`** |
| A2 | r14a baseline (`14de+a745`) | `21.257 ms` |

基线 midpoint 为 `21.437 ms`，half-range 为 `0.180 ms`；candidate
组合候选收益为 `0.921 ms`，门槛为 `max(0.616, 0.180)=0.616 ms`，且
`B < A1`、`B < A2`，因此 H4 **PASS**。
所以日志里出现的 `21 ms` 是 A1/A2 控制臂，不是最新 commit 的候选臂；
最新 `655c7bda+a745ab659` 的实际候选 p50 是 **20.516 ms**。

### 1.1 历史绝对值边界

`20.516 ms` 只属于本轮固定 `pypto-lib@a745ab659`、比较
`pypto@14de90fd → 655c7bda` 的 reset-ABI matched A/B/A。它不是历史
routed-GMM `20.172 ms` 的原合同复测，也不是新的历史最优：

| 历史点 | 形态 | iterations | 与当前关系 |
|---|---|---:|---|
| r12 exact launcher `20.973 ms` | `14de+e6c7` immutable，privileged | 1000 | 当前低 `0.457 ms`，但非同合同且低于 `0.616 ms` floor，只作方向性检查 |
| routed-GMM B 臂 `20.172 ms` | `14de+a745` source overlay | 100 | 当前高 `0.344 ms`，不能声明刷新历史最优 |

正式可声明的收益仍是本轮同代 A/B/A 的 `0.921 ms / 4.296%`。旧 a745 优化的 9 个文件已逐 SHA 复核未丢失；跨 campaign 的绝对值偏移不能拿来推算 `19 ms` 结果。

证据目录：

```text
/mnt/persist/chensiyu/workspace/perf-2026q3/
  k8-a745-matched-validation-20260901/
  runs/h4-k8-a745-immutable-20260901-211101-2847484-669318413/
```

关键文件 SHA256：

```text
run_contract.json  bc7062255b9ef953cb27953e1509ac9c5f3ee24cb81d84fcd240dcf2377499fa
h4_admission.json  5e425f012e1e29012f7b1ff3a178e873470f8719d1ce1eeb22a653439e53ef51
evidence.sha256    8d30d4a8dc097a2e245df90ab3feab36eb067278e5f099266a35775773b79207
```

## 2. 回退根因

`pypto` 旧的 K8 selective-reset 判断只认识 legacy EP 的
7-control/9-data buffer 集合，并将 control prefix 固定为 `47,616 B`。
local-owner layout 改为 4-control/4-data 后，buffer 集合不再匹配，
因此每个 persistent domain 回退到完整 `11,842,560 B` window 清零。

这不是计算 kernel 变慢，而是 reset path 在 host/device critical path
上重新支付了大段 memset 成本。

## 3. 修复内容

提交：

```text
655c7bda7 perf(runtime): pin local-owner persistent reset ABI
```

变更文件：

```text
python/pypto/runtime/distributed_runner.py
tests/ut/runtime/test_distributed_worker.py
```

修复将 layout 识别改为显式 profile：

| Layout | control buffers | data buffers | control prefix | full window |
|---|---:|---:|---:|---:|
| legacy EP | 7 | 9 | `47,616 B` | legacy ABI |
| local-owner | 4 | 4 | **`46,080 B`** | **`11,842,560 B`** |

local-owner profile 同时 pin buffer 顺序、每个 buffer 的 byte size 和
完整 window 大小；顺序或大小变化会 fail closed。未知 layout 继续完整
window clear，不会误清错误范围。

在 A/B/A trace 中，persistent reset 的 `memset_all` p50 为：

```text
A1  1059.529 us
B    462.277 us
A2  1068.354 us
```

三臂 hidden SHA 一致：

```text
ee8ae6b4b3083112d397e5e91cc63fb0e2edfb705eb7a535aceb232f1a7db96a
```

tail token 均为 `43640`。

## 4. 验证

PyPTO worktree 为 `655c7bda`，测试结果：

```text
focused K8 tests             11 passed
tests/ut/runtime/test_distributed_worker.py  209 passed
whole compile                454/454
git diff --check              PASS
```

最新 `pypto-lib` commit `a745ab659` 的关键变化是减少 routed GMM latch
参与者（`ROUTED_FUSED_GRID_WORKERS: 24 -> 22`），并同步 local-route
soft-sync ABI。本轮 A1/B/A2 都固定使用 a745，因此 `0.921 ms` 隔离的是
`pypto@655c7bda` reset 修复，不是对 a745 收益的再次测量。

## 5. Extended correctness gate 边界

修复后的 matched candidate extended runner 完成 5 个 workload case，包含：

- Main precision H4 `all` / `none`；
- MTP liveness BS1 / BS16；
- dep-only DFX H4 `all`。

最终 run：

```text
/mnt/persist/chensiyu/workspace/perf-2026q3/
  k8-a745-matched-validation-20260901/
  runs/extended-candidate-gate-r14b-20260901-235716-2942152-952530890/
```

```text
runner execute SHA      94a3cda8fc52f57f7b00acf9be30cf6add722c2b3df19221c3e1ca87fc0a1b36
extended admission SHA   c2d3324a34c1e72fc76e01c7c61b82e4f4cfeffed76c3848ba0131b95351c761
run_contract SHA         c1234cec667f8734746f25a25dadf00e5584dce9ecaf1e38c8e9adb3b6618904
evidence.sha256 SHA      d5968c8077bfefb823b3eeec408231e29689de8ff82fde882689aa42ac6896e1
```

`extended_gate_admission.json` 为
`step3p5.r14b-extended-immutable-admission.v2`，16/16 admission checks
为 `true`；5 个 case、prestart/revalidation、device visibility、source
identity、H4/MTP parity、DFX dep/hidden/token cross-check 均通过。
`run_contract.json` 为 v2、`status=pass`、`completed=true`；
在 0162 上复核 `sudo sha256sum -c evidence.sha256`、container/task
为空、设备 `0–15` process-free，均通过。

该 extended gate 的 scope 不含 `recv_meta` route publication sidecar。
旧 r12 sidecar（`step3p5.five-layer-moe-local-routes.v2`，仅 count-only）与当前
a745 要求的 provenance-matched route-identity publication contract 不匹配，不能复用。
因此 candidate IMG 不能写成正式 release-admitted。

## 6. 落地边界

- `pypto stepfun/develop` 已由 `14de90fd` fast-forward 到
  `655c7bda7b0a0b495a3387b2570ea68c4a857a40`。
- `pypto-lib stepfun/develop` 已由 `e6c7d8ec` fast-forward 到
  `a745ab659c68afca01de37870e29ccb9648d7c87`。
- 两次 push 均使用刚复核的
  `--force-with-lease`，push 后 `ls-remote` exact。
- candidate manifest `sha256:19f51d373c5f9d6171ccf3306f260066e873eda48efca23f5d77b4d6f5e64a7f`
  已做 immutable audit/H4/extended correctness，但仍不是正式发布 IMG。
- 2026-09-02 本地 r15 tag `stepfun-upgrade-20260902-a745-k8-r15` 与上述
  manifest/config 完全相同；无需再次重建，下一步是 registry push/raw/fresh pull、
  route publication/`recv_meta`、同合同 long gate 与完整 release contract。
