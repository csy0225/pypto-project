# 2026-08-27 · whole-step host/graph/submit 优化与 r12 发布

> **状态：release-admitted，最终合同 1844/1844 PASS。**
> 本报告严格区分三层证据：pypto 源码提交（SRC）、r11 digest 上的
> source-overlay A/B/A 性能（PERF）、r12 immutable image 的发布与
> correctness/security gate（IMG）。本轮性能数字不是 r12 immutable
> A/B/A；r12 最终合同封存并校验了先行的 r11 source-overlay admission。

## 1. 证据分层与结论

| 层 | 对象 | 本轮可声明 |
|---|---|---|
| SRC | pypto `14de90fd` | TaskArgs cache 路径优化已提交并出现在远端 `stepfun/develop` |
| PERF | published r11 digest + candidate 两文件 overlay | whole-step host/graph/submit p50 收益成立 |
| IMG | r12 manifest `ba42fd19…eb805d` | registry、digest-only smoke、immutable correctness/security、设备合同与最终准入成立 |

必须保留的边界：

1. A/B/A 三臂都运行在 r11 manifest
   `sha256:401ead7da4f957f6532e380fa1a138eda733fe1dc04b40eabc67d79d62a67b12`
   上；B 仅 source/runtime overlay 两个 Python 文件。
2. 没有采集 r12 immutable image 的 matched 性能 A/B/A。
3. A/B/A 契约仍是 `serial-eight-rank`、`group_size=1`、每次 invocation
   8 次 submit，且 `group_submit_calls=0`。因此只能声明 rank submit
   envelope 缩短，不能写成 native group submit 已通过设备门。
4. Graph、submit、bind、runner span 相互重叠，各百分比不能相加。
5. 性能运行显式使用 `PYPTO_H4_RESIDENT=all`；r12 image Config 没有 bake
   `PYPTO_H4_RESIDENT`。

## 2. r12 镜像与 pins

| 字段 | 值 |
|---|---|
| 机器 | `gpu-a910x-0162.host.platform.shaipower.com` |
| Tag | `hub.i.basemind.com/stepcast/vllm-pypto:stepfun-upgrade-20260826-r12` |
| Digest ref | `hub.i.basemind.com/stepcast/vllm-pypto@sha256:ba42fd19b3af0144a835e95a4a6925ed89ea700624f696b221e93a54e6eb805d` |
| Manifest | `sha256:ba42fd19b3af0144a835e95a4a6925ed89ea700624f696b221e93a54e6eb805d` |
| Config | `sha256:b36f0cec3a8b64e5e17e273c63d69694730bd8b904e69c2806c3d73a5233f08f` |
| Base | published r11 digest `sha256:401ead7da4f957f6532e380fa1a138eda733fe1dc04b40eabc67d79d62a67b12` |
| pypto | `14de90fd74b3c0716f94b9d4eafdd004d4eaed73` |
| pypto-lib | `e6c7d8ec34a05c3051ccf0dd169639f40f041a57` |
| pto-isa | `cd4a3d3f7a1a27fcfe536f617e9bca3008929664` |
| PTOAS source / binary | `307d0484a9e7d5e36f01b253d2bebe4d2f45fe81` / `v0.57` |
| simpler | `85a82c454074c069315ed6485033c3c2b136e562` |
| vLLM patch | `1b3e538c35999e62b6d24e0651b3a85b7d16c826` |

最终合同中的 GitHub `ls-remote` 证据确认五仓
`refs/heads/stepfun/develop` 与以上 SHA 精确一致，均为 `verified=true`。

## 3. 源码实现

pypto 提交：

```text
14de90fd74b3c0716f94b9d4eafdd004d4eaed73
perf(runtime): cache task args and parallelize rank submit
parent 519b588a7a6461cac0e443e853accf29479c1d15
```

提交修改 5 个文件，`1001 insertions / 40 deletions`：

```text
python/pypto/runtime/distributed_runner.py
python/pypto/runtime/tensor_arg.py
tests/ut/codegen/distributed/test_host_orch_distributed.py
tests/ut/runtime/test_distributed_worker.py
tests/ut/runtime/test_tensor_arg.py
```

实现边界：

- `distributed_runner.py` 为 prepared dispatch 建立稳定的 argument
  descriptor key，并按 frame slot / generation 发布 signature token；
- `tensor_arg.py` 使用已经过 public argument validation 的 token 复用昂贵的
  TaskArgs descriptor signature；
- cache metadata 与 TaskArgs entry 分离、容量有界，异常时 fail-open 回到
  live signature 计算；
- `free()` 与 submit validation/token publication/同步 graph construction
  共享生命周期互斥，避免 cache hit 提交已失效 Buffer 对应的 stale TaskArgs；
- UT 覆盖 descriptor 变化、缓存容量、失效与并发生命周期等路径。

提交标题包含 `parallelize rank submit`，但本次正式设备 admission 只证明
serial 8-rank submit envelope 变短；没有用 group-submit 合同替代逐 rank submit。

## 4. r11 source-overlay A/B/A 方法

证据目录：

```text
/mnt/persist/chensiyu/workspace/perf-2026q3/
  whole-step-host-graph-submit-20260826/runs/
  source-overlay-aba-cache-only-r11-20260827-001739-52171-467801141/
```

运行合同：

```text
arm order          A1 -> B -> A2
image              r11 manifest sha256:401ead7d...a67b12
devices            0,1,2,3,4,5,6,7
protected idle     8,9,10,11,12,13,14,15
active batch       1
context            65536
num_blocks         512
H4                 all
warmup / measured  10 / 100
fresh              container + build directory + nonce per arm
```

Arm 定义：

| Arm | Overlay | runtime source |
|---|---|---|
| A1 | 无 | r11 baked `distributed_runner.py@ece27a83…9513`、`tensor_arg.py@699568f8…1d91` |
| B | 两文件 | `distributed_runner.py@681db467…351d9`、`tensor_arg.py@3b382193…ae55` |
| A2 | 无 | 与 A1 相同 |

三臂共同保持：

- r11 manifest/config exact；
- pypto-lib、`decode_fwd.py` 和 native `pypto_core` 不变；
- 无 core overlay、无 pypto-lib overlay；
- explicit device cgroup + `ASCEND_RT_VISIBLE_DEVICES=0..7`；
- `privileged=false`；
- 每臂 container rc=0。

正式 submit contract：

```text
required_submit_mode        serial-eight-rank
required_group_size         1
required_submits_per_call   8
required_worker_ids         0,1,2,3,4,5,6,7
observed invocations        110 per arm
group_submit_calls          0
```

## 5. Whole-step 性能结果

以下均为 p50，baseline 是 A1/A2 midpoint：

| Metric | A1 | B | A2 | Baseline midpoint | B - midpoint | Delta |
|---|---:|---:|---:|---:|---:|---:|
| ITL | `21.761000` | `21.115000` | `21.600000` | `21.680500` | `-0.565500 ms` | `-2.608%` |
| `node.graph_build` | `4.159378` | `2.274285` | `4.025746` | `4.092562` | `-1.818277 ms` | `-44.429%` |
| graph → first runner | `3.162133` | `1.613020` | `3.034136` | `3.098134` | `-1.485114 ms` | `-47.936%` |
| 8-rank runner start wave | `1.187589` | `0.891732` | `1.159328` | `1.173459` | `-0.281727 ms` | `-24.008%` |
| rank submit envelope | `1.217525` | `0.916042` | `1.189524` | `1.203524` | `-0.287482 ms` | `-23.887%` |
| graph → last runner | `4.328152` | `2.513347` | `4.194828` | `4.261490` | `-1.748143 ms` | `-41.022%` |
| graph → last chip completion | `20.931999` | `19.110885` | `20.814599` | `20.873299` | `-1.762414 ms` | `-8.443%` |
| `bind.args` pooled/rank | `0.054451` | `0.054669` | `0.054447` | `0.054449` | `+0.000220 ms` | `+0.404%` |

候选分布：

| Metric | Samples | p10 | p50 | mean | p90 | p99 |
|---|---:|---:|---:|---:|---:|---:|
| graph build | 100 | `2.236109` | `2.274285` | `2.333970` | `2.398126` | `3.021119` |
| graph → first runner | 100 | `1.579446` | `1.613020` | `1.675413` | `1.748444` | `2.426890` |
| runner start wave | 100 | `0.875549` | `0.891732` | `0.896376` | `0.922040` | `0.975770` |
| rank submit envelope | 100 | `0.903884` | `0.916042` | `0.921865` | `0.938029` | `1.001963` |
| `bind.args` | 800 | `0.053007` | `0.054669` | `0.055043` | `0.057024` | `0.064739` |
| graph → chip completion | 100 | `19.067650` | `19.110885` | `19.171709` | `19.242617` | `19.915162` |

`bind.args` 的 B p50 仅占 B ITL p50 的 `0.259%`，绝对变化
`+0.000220 ms`，admission 判定为 `no_clear_change`。它不是下一步优化对象。

A/B/A correctness：

```text
hidden sha256  ee8ae6b4b3083112d397e5e91cc63fb0e2edfb705eb7a535aceb232f1a7db96a
tail token     43640
shape          [8,16,4096]
dtype          torch.bfloat16
finite         true
TP spread      0
token exact    true
```

关键 SHA：

```text
run_contract.json          98d452750b09b603a2516a1cc249441f22ac195fda9dd7b8600d6bc713bdcf63
whole_step_metrics.json    dd1a555b16d23bf04a2c3e5ba02115d5492e7f238f1b12047d354f0d689a0055
whole_step_admission.json  3af53853c9ce1cafa095a600a0e7587aaa3530a8a90b5109eea56a2950f00989
aba_admission.json         110632f22c6cfbe1d30005c7fa964e22590e19f94a250e02d683bfc44e17172b
evidence.sha256            7583ea5ccea9cd3a69018ba0d3d1201b6188fdd5d65b817d20f82d463ad0f613
```

## 6. r12 可复现构建

证据根：

```text
/mnt/persist/chensiyu/workspace/moe-fusion-release-20260826-r12/
```

Dockerfile 从 published r11 digest 派生，移除并重新物化 pinned source trees，
再重建 runtime，避免继承 base 中的可变源码状态。

原始 spec 的第二行注释沿用了“published r9 image”旧措辞，但不可执行注释不作为
base 身份依据；机器可执行的 `BASE=` 明确是 r11 manifest
`sha256:401ead7d…a67b12`，最终合同也按该 digest 校验。仓内 spec 为保持与 0162
原件逐字一致，不反向修改这条历史注释。

| 工件 | SHA256 |
|---|---|
| `image-build/builds/stepfun-upgrade-20260826-r12.env` | `94e018c068bd4f4b2fb65071cd47596f3d35cd15dcbccec80326055335f93f27` |
| `image-build/src-pins.shas` | `7dcc2d59f23e8b52b94e1a4e29b923406a3e3f6b6cacf5f617c6155b67572717` |
| `image-build/build-r12-20260826.log` | `a20eed39fed9af5a302894955eb570e47dc13ea6b577185d1d2df9e921636abb` |

Build spec 还固定：

```text
PTOAS_BIN_VER                       v0.57
PTOAS_BIN_SHA256                    2183e4cf00fd019403825290233c32b84c1b9904474ec4614ab976dac143aaae
ATTN_TASK_PROFILE                   a2a3
REQUIRE_L2_SWIMLANE_REUSE_DEP_GEN   1
BUILD_JOBS                          24
```

## 7. Registry 发布

成功 publication：

```text
/mnt/persist/chensiyu/workspace/moe-fusion-release-20260826-r12/
  publication-r12-20260826-193648-authretry/
```

发布门：

- push rc=0；
- tag 与 digest 的 raw manifest/config identity exact；
- 隔离 namespace `pypto-r12-fresh-20260826-193648` fresh pull PASS；
- digest-only image audit、source-bake audit 与 smoke PASS；
- non-privileged digest-only device inventory、image audit 与 smoke PASS；
- fresh compute device 为 `/dev/davinci0`，control nodes 为
  `/dev/davinci_manager`、`/dev/hisi_hdc`、`/dev/devmm_svm`；
- explicit device cgroup + visible-device env，major/minor exact，
  `privileged=false`；
- publication 后设备空闲；
- registry credential 仅通过 stdin 写入 tmpfs 最小单-registry config，
  完成后移除。

关键 SHA：

```text
run_contract.txt              14fbe9d4f5adec52c2e8b456075acce73b9dca94cf824868d34d85dd4a43084a
fresh_identity_verdict.json   87886127bb26fab67b872bd4d0e77b3598d2482886fa433f9cce4759818f2061
evidence.sha256               5cb9363692d84c50be3a65a86c7f68623e2283e6a79b9d13492094df6ef8f270
```

## 8. Immutable correctness 与 security gate

最终 gate：

```text
/mnt/persist/chensiyu/workspace/perf-2026q3/
  whole-step-host-graph-submit-20260826/runs/
  immutable-release-gate-r12-20260827-055402-243696-524853898/
```

Runner：

```text
/mnt/persist/chensiyu/workspace/perf-2026q3/
  whole-step-host-graph-submit-20260826/run_immutable_release_gate_r12.sh
sha256 9c9cb8bc2ea9af565bf04841a4669494d41ee2dbb80d794aba2f0f0eeb3234d8
```

Case matrix：

| Case | 结果 |
|---|---|
| immutable smoke | baked image audit + smoke PASS |
| Main precision H4 all | `126/128 = 98.4375%`，128 active-hidden finite，TP spread 0 |
| Main precision H4 none | `126/128 = 98.4375%`，与 H4 all token/active-hidden parity PASS |
| MTP BS1 | tokens `[6178,410,303]`，hidden pass `[1,1,1]`，max abs diff `[0,0,0]` |
| MTP BS16 | tokens `[6178,410,303]`，hidden pass `[1,1,1]`，max abs diff `[0,0,0]` |
| Main dep-only DFX | hidden/token exact，tail `43640`，hidden SHA `ee8ae6b4…a7db96a` |

Dep-only DFX 的单次 `2521.351 ms` 是 warmup=0、iters=1、带 dep capture
的 correctness/DFX 运行，不是生产 ITL 性能数据。

设备与容器合同：

```text
requested / visible NPU   0,1,2,3,4,5,6,7
protected idle NPU        8,9,10,11,12,13,14,15
device mode               explicit-device-cgroup+ASCEND_RT_VISIBLE_DEVICES
control nodes             davinci_manager,hisi_hdc,devmm_svm
major/minor               exact
privileged                false
seccomp                   default
cap_add                   []
apparmor                  unconfined
network_mode              host
ipc_mode                  host
source/runtime/core       no overlay
```

Prestart attestation lifecycle：

```text
create
-> inspect
-> validate
-> recapture
-> revalidate
-> start by validated container ID
-> cleanup by validated container ID
```

Validator 覆盖并通过 11 个 adversarial fixtures：

```text
name_swap_prebaseline
name_swap_recapture
extra_bind
wildcard_device
apparmor_inspect_drift
apparmor_oci_drift
stale_verdict_replaced_artifact
state_null_expected_id_present
state_null_task_query_rc_nonzero
task_appears_before_start
stale_task_verdict_replaced_raw
```

同时在 baseline、recapture、final-prestart 三个阶段验证 containerd task absence。

关键 SHA：

```text
run_contract.json                 214946fc5048b74ec62889b98378dcd4c25ef97d3b1e620cd8c0b034eb91daf8
immutable_release_admission.json  0b4b44f99197244b67d434a8c750433f89da9b13ae3368658500a82642b569ec
evidence.sha256                   4200022109290aaa812a99e73eaef0f8b941226216e4ce6edc2274aa2ea2a9c6
```

## 9. Cleanup

最终 cleanup snapshot：

- 16/16 NPU `Health=OK`；
- 16/16 NPU process memory 为空；
- `/dev/davinci0..15` 的 `fuser` 全空；
- containerd `default` 与 fresh-pull namespace 无 container/task；
- 无 runc task、无 containerd shim；
- 未执行 generic process kill 或 NPU reset。

这些结论描述最终合同生成时的封存快照；后续作业仍须重新执行
`sudo -n fuser` 与 `npu-smi info -t proc-mem`，不能沿用历史空闲状态。

## 10. 最终 release contract

```text
/mnt/persist/chensiyu/workspace/moe-fusion-release-20260826-r12/
  release-admission-r12-20260826-224620/release_contract.json
```

```text
schema   step3p5.r12-final-release-admission.v1
status   release-admitted
pass     true
checks   1844/1844
errors   []
sha256   511a545956aee4cef7264a74460bd04862846e377ef71eb01619ae4ddbf87f3a
```

合同封存：

- 五仓远端 `stepfun/develop` exact SHA；
- r12 local/tag/digest/raw registry identity；
- publication 与 fresh-pull evidence；
- immutable r12 correctness/security gate；
- prior admitted r11 source-overlay cache-only A/B/A；
- 16 卡 cleanup snapshot；
- 各 evidence manifest 与关键工件 SHA。

合同创建时间为 `2026-08-26T22:46:26Z`，对应 0162 / 香港时区
**2026-08-27 06:46:26 +08:00**。因此本文日期采用 2026-08-27；
合同字段 `campaign_date_utc` 仍为 `2026-08-26`。

## 11. 准确结论

本轮可以声明：

> pypto `14de90fd` 的 whole-step host/graph/submit candidate 在 published
> r11 digest 上通过 source-overlay A/B/A，ITL p50 改善 `2.608%`，
> graph build 改善 `44.429%`，graph→first-runner 改善 `47.936%`，
> serial 8-rank submit envelope 改善 `23.887%`；相同源码已烧入 r12，
> r12 immutable image 完成 registry、correctness、security、设备隔离
> 和 1844/1844 final release admission。

本轮不能声明：

- 这些性能百分比来自 r12 immutable-image A/B/A；
- native group submit 已通过正式设备门；
- 各重叠 span 的收益可以相加；
- image Config 已 bake `PYPTO_H4_RESIDENT=all`；
- dep-only DFX 单次耗时是生产 ITL。
