# 2026-08-24 · 全栈升级 r9 最终发布与 H4 性能口径

> 本文是 [`2026-08-23-upgrade-image-release-gates.md`](2026-08-23-upgrade-image-release-gates.md)
> 中受阻候选的后续收口。所有最终结论都绑定同一 immutable manifest；不借用旧镜像或
> source-overlay 数据。
>
> **2026-08-25 状态更新**：packed-NZ MoE fusion r10 已完成 immutable
> A/B/A、六档 correctness、outer DFX、`stepfun/develop` exact-lease 同步和最终
> release contract，现为当前 release-admitted 基线。本文保留为 r9 回退记录。r10 见
> [`2026-08-25-moe-fusion-image-release.md`](2026-08-25-moe-fusion-image-release.md)。
>
> r10 A1/B/A2 p50 为 `22.524/21.821/22.580 ms`、mean
> `22.862/21.937/22.633 ms`、p99 `28.542/28.338/24.208 ms`；相对 baseline
> midpoint `22.552 ms` 为 `−0.731 ms / −3.241%`，bracket `0.056 ms`，
> 三臂 hidden/token exact，ABA verdict SHA256
> `8d4224e0214b71bae01efe24393e5886375e04dff5481ffd34ba19e3821ddb0e`。
> pypto-lib `stepfun/develop` 已由 `bf3ff440` exact-lease FF 到 `fe641929`
> （`git-sync-r10-20260825-144155/`）。最终合同
> `r10-release-admission-20260825-150350/release_contract.json` 为
> `step3p5.r10-release-admission.v2`、`71/71` PASS，SHA256
> `bcdd0b11d346e450dca49b8434544de5566b7fc0ad1a38c715815a41958dafca`。
> 完整 `E5→E6` 仍为 `n/a`，BS8/BS16 单次回退 caveat 仍保留。

## 1. 最终发布身份

| 字段 | 值 |
|---|---|
| 机器 | `gpu-a910x-0162.host.platform.shaipower.com` |
| Tag | `hub.i.basemind.com/stepcast/vllm-pypto:stepfun-upgrade-20260824-r9` |
| Manifest | `sha256:b637f00c66d4dc976c053c617d2e19e6d6d66f68f4bef30250984da7a71690f6` |
| Config | `sha256:f6c8f72eecad0a9d40d0c4ea55afaab09dd4e2f5fe54d6a091e332465e421dae` |
| pypto | `519b588a7a6461cac0e443e853accf29479c1d15` |
| pypto-lib | `bf3ff4400082f74b35fbdb5b3e0f5f4bf51ce373` |
| pto-isa | `cd4a3d3f7a1a27fcfe536f617e9bca3008929664` |
| PTOAS source / binary | `307d0484a9e7d5e36f01b253d2bebe4d2f45fe81` / `v0.57` |
| simpler | `85a82c454074c069315ed6485033c3c2b136e562` |
| vLLM patch | `1b3e538c35999e62b6d24e0651b3a85b7d16c826` |
| `decode_fwd.py` sha256 | `671a5df8a07e09303c398871fd1772f306b2998ea3e8168048588de6cc3fa323` |

Registry 发布证据：

```text
0162:/mnt/persist/chensiyu/workspace/upgrade-20260821/
  publication-r9-20260824-145744/
```

`push_rc=0`、fresh pull `rc=0`，远端 raw manifest、config 与 fresh-pull
identity 均逐项匹配。`registry_verdict.json` 的 `pass=true`。

## 2. ITL：同一 digest 的两个运行合同

两次运行的镜像、checkpoint、cards `0-7`、`a2a3`、`active_batch=1`、
`num_blocks=512` 完全相同；long 均为 ctx `65536`、warmup `10`、iters `1000`。
唯一 runner 差异是第二次显式注入：

```bash
PYPTO_H4_RESIDENT=all
```

| 运行 | H4 模式 | p50 | mean | p99 | 证据 |
|---|---|---:|---:|---:|---|
| r9 默认行为 | unset = `none` | `27.812 ms` | `28.164 ms` | `32.583 ms` | `itl-20260824-134457/` |
| r9 发布性能合同 | `all` | **`22.253 ms`** | `22.426 ms` | `27.206 ms` | `itl-20260824-135721/` |

重复性旁证：

- 更早一轮同 digest/config 的默认路径为 p50 `27.970 ms`
  （`itl-20260824-132235/`），与 `27.812 ms` 同簇；
- 更早一轮 H4 all 为 p50 `22.352 ms`（`itl-20260824-114115/`），其原始合同只写
  tag，但运行日志确认 `resident constants (all)`；最终发布仍只采用上表严格
  digest-bound 的 `22.253 ms`。

四点曲线的 p50：

| context | 默认 `none` | `all` | Δ |
|---:|---:|---:|---:|
| 1024 | `27.139 ms` | `21.715 ms` | `-5.424 ms` |
| 8192 | `27.748 ms` | `22.355 ms` | `-5.393 ms` |
| 32768 | `27.728 ms` | `22.369 ms` | `-5.359 ms` |
| 65536 | `28.171 ms` | `22.409 ms` | `-5.762 ms` |

### 2.1 为什么不是负载波动

四档 context 都稳定下降 `5.36–5.76 ms`。同一 long run 的 STRACE p50：

| span | 默认 `none` | `all` |
|---|---:|---:|
| `bind.args` | `6.461 ms` | `0.063 ms` |
| `bind` | `6.479 ms` | `0.075 ms` |
| `runner_run` | `19.130 ms` | `18.959 ms` |
| 报告 ITL | `27.812 ms` | `22.253 ms` |

`runner_run` 主体只差 `0.171 ms`，主要收益与 `bind.args` 消失逐项对应。
两份 artifact 没保存 CPU 频率、温度和 loadavg 快照，因此不能排除亚毫秒级环境抖动；
但它不能解释跨全部 context 稳定约 `5.5 ms` 的阶跃。

### 2.2 H4 做了什么

`pypto-lib@bf3ff440` 的 `whole_decode_holder.py` 定义：

- `none`：默认，不做常量驻留；
- `rope`：常驻 4 个 RoPE 表；
- `gate`：常驻 4 个 block-diag gate-R；
- `all`：8 个静态参数全部常驻。

`all` 在 holder 初始化时把这些参数一次上传并重绑为 device-resident
`StackedDeviceTensor`，后续 decode step 不再重复 H2D staging。日志确认：

```text
[holder] resident constants (all): 8 args, 99.64 MiB/rank
```

代价是每 rank 约 `99.64 MiB` 额外 device memory，以及初始化阶段的一次上传。

### 2.3 必须保留的部署边界

**r9 镜像 Config Env 没有内置 `PYPTO_H4_RESIDENT`，代码默认值是 `none`。**
因此：

- `22.253 ms` 是 **r9 + `PYPTO_H4_RESIDENT=all` 部署合同**的性能；
- 直接启动镜像且不传环境变量，会回到约 `27.8 ms`；
- 正式 launcher / manifest 必须显式配置 `PYPTO_H4_RESIDENT=all`，否则不能宣称
  开箱即得 `22.253 ms`。

## 3. 精度与 liveness

Precision admission：

```text
0162:/mnt/persist/chensiyu/workspace/upgrade-20260821/
  precision-r9-h4-admission-20260824-144951/admission.json
```

- accepted oracle：`127/128 = 99.21875%`，门限 `95%`；
- 唯一 mismatch：step `94`；
- oracle SHA256：`eb561cf8eb1de49cccefe2cbda91071d3fed8fa163fa84a3461c68eb70a241c2`；
- accepted 与 alternate 两组 oracle 中，`all` / `none` 的输出 token 序列各自完全一致；
- `all_none_output_parity=true`，未观察到 H4 引入的数值差异。

Full liveness：

```text
0162:/mnt/persist/chensiyu/workspace/upgrade-20260821/
  liveness-r9-full-20260824-140550/whole_network_report.json
```

Main 8-step、MTP single、MTP batch16 全部 `rc=0`、`passed=true`。

## 4. 前五层 hidden / swimlane / DFX

最终使用 immutable digest 并挂载 golden 的 combined focused gate：

```text
0162:/mnt/persist/chensiyu/workspace/upgrade-20260821/
  outer-swimlane-r9-h4-20260824-151416/
```

- hidden L3 `torch.equal=true`，SHA256
  `5aca3716156b190ece14780bc32316e23423ab0c4f9525ba50b4730f108ee8b9`；
- hidden L4 `torch.equal=true`，SHA256
  `0308be3197bfe1921215c2082146946a625350896949f63572b52145afe400a4`；
- 8/8 rank 均有 `chip_swimlane_records.json`、`deps.json`、`name_map.json`、
  `critical_path_report.md` 和 merged swimlane；
- DFX analyzer `pass=true`、`blockers=[]`；
- 上述 `pass=true` 是 analyzer 的结构检查结果；其原始报告仍按职责边界保留
  `PENDING_EXTERNAL_GATE`、`publication_allowed=false`。L3/L4 hidden 由 outer
  admission 消费，只有下文最终 `release_contract.json` 才给出整项 `pass=true`；
- recv_meta sidecar ready，SHA256
  `7dc01bf4e73a5a6ef47acd4ff6f21623bae39e33a798037b9767fa076e5bf994`。
- LOW-WAIT reference 为 `rank2/d0`：makespan `1.867 ms`、static CPM
  `1.544 ms`、data-wait/stall `0.323 ms`。

配套 route sidecar：
`r9-formal-route-sidecar-20260824-123143/route-runtime/route_artifact_validation.json`
为 `passed=true`、`hidden_bit_exact=true`；L3/L4 global routes 均为 `64`，
每 source/layer local count 均为 `8`，`padding_zero/local_count_exact=true`。

前五层：

```text
L0_full_dense
L1_swa_dense
L2_swa_dense
L3_swa_moe
L4_full_moe
```

r9 正式 raw schema 是 `chip_swimlane_records.json`；不得复制或改名伪造旧的
`l2_swimlane_records.json`。

## 5. 五仓同步

`git-sync-r9-20260824-150512/verdict.json` 为 `pass=true`。远端
`refs/heads/stepfun/develop` 已复核：

| 仓库 | 最终 SHA |
|---|---|
| pto-isa | `cd4a3d3f7a1a27fcfe536f617e9bca3008929664` |
| PTOAS | `307d0484a9e7d5e36f01b253d2bebe4d2f45fe81` |
| simpler | `85a82c454074c069315ed6485033c3c2b136e562` |
| pypto | `519b588a7a6461cac0e443e853accf29479c1d15` |
| pypto-lib | `bf3ff4400082f74b35fbdb5b3e0f5f4bf51ce373` |

`simpler` 是非 fast-forward 更新，旧 tip 已原子备份为：

```text
refs/heads/backup/stepfun-develop-pre-upgrade-20260824-e2efebcb
```

## 6. 最终准入

```text
0162:/mnt/persist/chensiyu/workspace/upgrade-20260821/
  r9-release-admission-20260824-151848/release_contract.json
```

合同 SHA256：

```text
1cd646e31cd6ce4dd0f3817219c297690b5ab1d355ab47c71eaafe489b2a08a6
```

`pass=true`。本次升级任务的镜像发布、precision、liveness、前五层 hidden/swimlane/DFX
及五仓同步均已闭环；性能结论必须始终连同
`PYPTO_H4_RESIDENT=all` 运行合同一起引用。
