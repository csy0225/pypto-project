# H4 resident constants 部署合同收口（2026-08-29）

## 1. 结论

基于 0162 最新 whole-step 与 swimlane 证据，当前最高收益且可直接落地的项目不是继续
拆 task、调 bind 或重复 routed-expert 扫描，而是把已发布的 **H4 resident constants**
接入正式 deployment launcher：

- `deployment/docker/run_itl_gate.sh`
- `deployment/docker/run_precision_gate.sh`
- `deployment/docker/run_swimlane_gate.sh`

三个 launcher 在父进程未设置 `PYPTO_H4_RESIDENT` 时默认显式注入 `all`；仍允许
`PYPTO_H4_RESIDENT=none|rope|gate|all` 覆盖，其中 `none` 是明确回退路径。pypto-lib
代码默认仍为 `none`，r12 image Config Env 也未 bake 此变量，因此本次落地边界是
**deployment contract**，不是新镜像或库级全局默认变更。

## 2. 当前基线

```text
host:      gpu-a910x-0162.host.platform.shaipower.com
image:     hub.i.basemind.com/stepcast/vllm-pypto@sha256:ba42fd19b3af0144a835e95a4a6925ed89ea700624f696b221e93a54e6eb805d
config:    sha256:b36f0cec3a8b64e5e17e273c63d69694730bd8b904e69c2806c3d73a5233f08f
pypto:     14de90fd74b3c0716f94b9d4eafdd004d4eaed73
pypto-lib: e6c7d8ec34a05c3051ccf0dd169639f40f041a57
workload:  BS1, ctx=65536, num_blocks=512, TP=8
```

r12 final release contract 仍为 `1844/1844 PASS`。本次没有重建 image，也没有移动五仓
pin；只改变项目仓中的 deployment launcher 默认运行合同。

## 3. 为什么选 H4

### 3.1 最新 swimlane 校正

最新 raw capture：

```text
0162:/mnt/persist/chensiyu/workspace/perf-2026q3/
  p1-attn-fcj-runtime-e2e-r4-20260829-015039/
  runs/capture-20260829-020636-1010265-247858051
```

首个 `tp_all_reduce_residual` 含 peer-arrival spin，不能把数百毫秒条带当成 collective
service time。按未污染 rank 与最小 collective service 修正后：

- 五层 clean makespan 约 `1.81 ms`；
- 45 层外推 device 段约 `17.60 ms`；
- routed expert 约 `5.115 ms`（stall `1.646 ms`）；
- route/combine 约 `3.472 ms`（stall `0.956 ms`）；
- TP collective 约 `2.861 ms`（stall `0.515 ms`）。

这些仍是后续 device 优化池，但 routed/route 需要新的 authority instrumentation，TP
collective 又被 `UPSTREAM-NOTIFY-FENCE` correctness blocker 限制，不能作为本轮低风险落地。

### 3.2 候选排序与已否决项

| 候选 | 当前证据 / 上界 | 判定 |
|---|---:|---|
| H4 resident constants 部署 | 历史 `5.559 ms`；本轮 matched `7.372 ms` | **GO** |
| H6 TaskArgs cache | 已在 `pypto@14de90fd` 与 r12，收益 `0.5655 ms` | 不重复移植 |
| 继续优化 `bind.args` | 候选仅 `0.054669 ms`，ITL 占比 `0.259%` | NO-GO |
| K10 host control round | 上界 `0.45–0.53 ms` | 低于 `0.616 ms` 地板 |
| FCJ attention finalize | matched mean `−0.028590 ms`，95% CI 跨 0 | NO-EFFECT |
| dispatch/task 粒度与既有 AR/MoE 变体 | 已有明确 NO-GO / correctness gate | 不重试 |

H4 已具备 r12 `all/none` 128-step parity、MTP/DFX 门，以及当前 runtime 的显式
`all` 1000-step liveness，因此本轮只需补齐 source-default sensitivity 与 unset launcher
等价性。

## 4. Source-default-all matched `none / default / none` A/B/A

证据目录：

```text
0162:/mnt/persist/chensiyu/workspace/perf-2026q3/h4-default-all-20260829/
  evidence/none-default-none-20260829-102612-1360791-303996280
```

合同：r12 digest、cards 8–15、fresh container/nonce/build、warmup 10、100 measured
steps；A1/A2 显式 `none`，B 不传 env，候选 source 的 unset 默认为 `all`。

> 口径说明：本节的 B 臂使用临时 **pypto-lib source overlay** 将代码默认改为
> `all`，用于隔离 H4 的运行时收益；它不是当前 launcher 代码本身的证据。最终
> canonical launcher 在父环境 unset 下的实际注入行为由下一节 exact deployment gate
> 独立验证，且发布的 pypto-lib 代码默认仍为 `none`。

| arm | H4 | p50 | mean | hidden / token |
|---|---|---:|---:|---|
| A1 | `none` | `30.516 ms` | `30.881 ms` | exact |
| B | env unset → `all` | `22.606 ms` | `22.872 ms` | exact |
| A2 | `none` | `29.440 ms` | `29.498 ms` | exact |

结果：

- baseline midpoint `29.978 ms`；
- baseline bracket `1.076 ms`，half-range `0.538 ms`；
- gain `7.372 ms / 24.591%`；
- `7.372 ms > max(0.616, 0.538) ms`；
- B 同时低于 A1/A2；
- 三臂 hidden SHA 均为 `ee8ae6b…db96a`，tail token 均为 `43640`，TP spread `0`；
- `TENSOR_WAIT_TIMEOUT / HEAP_RING_DEADLOCK / Fatal teardown / Segmentation fault` 为 0；
- postflight container、device process 与锁均干净。

`ABA_RESULT.json` SHA256：
`ba1962fb4129756293d1cbad29bc683fb73e3b349192eea802d05c19ff33c2d3`。

## 5. Exact deployment launcher 长门

父 shell 明确 `unset PYPTO_H4_RESIDENT`，直接运行修改后的
`deployment/docker/run_itl_gate.sh`。launcher 自己记录并向容器注入 `all`。

证据：

```text
meta:
  0162:/mnt/persist/chensiyu/workspace/perf-2026q3/h4-deploy-contract-20260829/
    exact-launcher-20260829-104651-1389455
output:
  0162:/mnt/persist/chensiyu/workspace/upgrade-20260821/itl-20260829-104704
```

- 64K，warmup 10，1000 steps：p50 `20.973 ms`，mean `21.089 ms`，p99
  `24.369 ms`，RC=0；
- context curve p50：1K `20.139`、8K `20.698`、32K `20.827`、64K
  `20.821 ms`；
- 两个 container log 都有
  `[holder] resident constants (all): 8 args, 99.64 MiB/rank`；
- 无 fatal marker；pre/postflight 16 卡 process-free、container-free。

`exact_launcher_admission.json` SHA256：
`cd6c962de3883d072006e6fc860a9dab088d9e0bfb9e33ff842dcd1cfdbc14d5`。

## 6. 落地边界与回退

- 默认路径：launcher 注入 `PYPTO_H4_RESIDENT=all`；
- 回退：启动前设置 `PYPTO_H4_RESIDENT=none`；
- 额外 HBM：约 `99.64 MiB/rank`；
- image Config 与 pypto-lib 默认保持不变，因此绕过这些 launcher 的调用方仍必须显式设置
  env；不能据此宣称 image 自身 bake 了 H4；
- 本结果不关闭 Phase 28 live prefill/paged-KV/3-way HBM，也不解除
  `UPSTREAM-NOTIFY-FENCE`。
