# Routed GMM active-worker dual-latch 验证（2026-08-30）

> **2026-09-02 后续边界**：`a745ab659` 已进入 canonical `stepfun/develop`，
> 但本文 `20.172 ms` 是 r12 source-overlay B 臂，不是当前 immutable image 的
> 原合同复测。当前 `655+a745` 的 `20.516 ms` 属另一份 reset-ABI matched A/B/A，
> 两者不可直接写成前后回归。见
> [`2026-09-02-k8-historical-performance-reconciliation.md`](2026-09-02-k8-historical-performance-reconciliation.md)。

## 1. 结论与证据等级

以 r12 immutable image 为固定 substrate、仅 overlay candidate source，routed packed-NZ
GMM 的 active-worker dual-latch 候选通过了 whole compile、H4-all 整网 A/B/A、五层结构
DFX 与 L3/L4 byte-exact 门：

- A1/B/A2 p50 为 `21.099 / 20.172 / 21.107 ms`；
- baseline midpoint `21.103 ms`，收益 `0.931 ms / 4.4117%`；
- required floor `0.616 ms`，判定 **PASS**；
- 三臂 hidden SHA256 均为
  `ee8ae6b4b3083112d397e5e91cc63fb0e2edfb705eb7a535aceb232f1a7db96a`，
  tail token 均为 `43640`，TP spread `0`。

证据必须分层：

| 层级 | 当前状态 | 可以声明 | 不能声明 |
|---|---|---|---|
| **CAND** | `perf/gmm-soft-mix-prestage-20260829@a745ab6` 已推送 | feature branch 与 r12 source-overlay 门成立 | 已合入 canonical `stepfun/develop` |
| **SRC** | canonical 仍为 `e6c7d8ec` | 当前生产源码 pin 未变化 | `a745ab6` 已成为 canonical SRC |
| **IMG** | r12 manifest 仍为 `ba42fd19…eb805d` | 固定 immutable substrate 身份成立 | r12 已包含候选或已有新 release image |

因此当前判定是 **feature-branch source-overlay GO**。下一门是 canonical merge、补齐
exact `recv_meta` route sidecar、重跑完整 DFX publication gate，再构建并验证新 immutable
image。本文不更新 r12 digest，也不把候选写成 image 级结果。

## 2. Source 与运行合同

```text
host:              gpu-a910x-0162.host.platform.shaipower.com
image manifest:    sha256:ba42fd19b3af0144a835e95a4a6925ed89ea700624f696b221e93a54e6eb805d
image config:      sha256:b36f0cec3a8b64e5e17e273c63d69694730bd8b904e69c2806c3d73a5233f08f
pypto:             14de90fd74b3c0716f94b9d4eafdd004d4eaed73
canonical base:    pypto-lib e6c7d8ec34a05c3051ccf0dd169639f40f041a57
candidate branch:  perf/gmm-soft-mix-prestage-20260829
candidate commit:  a745ab659c68afca01de37870e29ccb9648d7c87
candidate parent:  e6c7d8ec34a05c3051ccf0dd169639f40f041a57
candidate diff:    4cc1e10b7949d57893fbf4dc19e6170ffce771dacc92eb1abf31e2e95c2e7c5e
decode_fwd SHA256: cdb2bb26ddc0ca773bcddd0629bfc7bdfa5c426a334e26dde4364aacd867f348
workload:          BS1, ctx=65536, num_blocks=512, TP=8, PYPTO_H4_RESIDENT=all
```

冻结源位于：

```text
0162:/mnt/persist/chensiyu/workspace/perf-2026q3/
  r12-gmm-softsync-validation-20260829/
  sources/seal-final-active-worker-dual-latch-v8
```

`freeze_contract.json` SHA256：
`73183861dbace2e792a12a22863068db34633c27df8fcd52d3b55ac3ca241850`。
冻结时以 `e6c7d8ec + candidate diff` 表示候选；已推送 commit `a745ab6` 的 parent 与
diff SHA256 均逐项一致。2026-08-30 在 0162 使用 GitHub credential、HTTP/1.1
重新执行 exact ref 查询：

```text
git ls-remote https://github.com/csy0225/pypto-lib.git \
  refs/heads/perf/gmm-soft-mix-prestage-20260829
a745ab659c68afca01de37870e29ccb9648d7c87  refs/heads/perf/gmm-soft-mix-prestage-20260829
```

因此“已推送”只绑定该 feature ref；canonical `stepfun/develop` 仍为 `e6c7d8ec`。

## 3. 实现与协议

### 3.1 Active-worker 参与者

设：

```text
A = min(active_local_experts, 36)
G = min(22, 10A)
H = min(22, 5A)
Q = min(22, A)
```

- 只有 AIC block `b < G` publish gate/up completion；
- 只有 lane-0 AIV block `b < H` await gate/up，并 publish hidden completion；
- 只有 lane-0 AIV block `b < Q` await hidden completion；
- `A=0` 由外层 predicate 跳过 soft latch；边界合同覆盖 `A=0..36`，重点包含
  `0,1,2,3,4,5,21,22,36`。

这不是删同步。三个 external routed GMM kernel 的 store drain、`pipe_barrier`、`dcci`
与 `dsb` 均保留；改变的是参与 latch 的 worker 集合，避免无 active-expert 工作的 resident
core 做 atomic publication 或轮询。

### 3.2 Workspace 与 ABI

- local expert count 继续使用 40-entry INT32 physical ABI；
- compact route plan 扩为 96-entry INT32 workspace；语义 plan 仍为 `0:38`；
- gate/up-ready 与 hidden-ready counter 位于 offset `64`、`80`，与 metadata 及彼此
  cache-line 隔离；producer 清零完整 workspace；
- Python orchestration 对 workspace 使用 `pl.InOut`，三个 routed GMM external kernel
  的 workspace 参数保持 InOut lineage；
- fused grid 固定为 `22 AIC / 44 AIV`，BS1 routed down 为 `23 AIC / 46 AIV`。

## 4. Compile 与静态合同

| 门 | 结果 |
|---|---|
| focused contracts | `168 passed` |
| full Step3p5 unit | `535 passed, 4 skipped` |
| whole compile | **PASS**，`lowered_contract_ok=true` |
| scoped Ruff / diff check | **PASS** |
| direct CCEC | AIC `.text=1392 B`；AIV `.text=5428 B`；SwiGLU7 AIV `.text=5596 B` |

最终 0162 容器复跑（candidate commit/diff 已写入 manifest，`rc=0`）：

```text
focused: /mnt/persist/chensiyu/workspace/perf-2026q3/r12-gmm-softsync-validation-20260829/
  unit-final-a745ab6-focused-20260830-170353-1937305-890145330
  pytest.log SHA256    2b3dbe2b67e271bed3e025c8aa9ba4f34440dcd34031250f98e6d64eb13999be
  manifest.txt SHA256  ff456ee4f634613aee31abc589505699ea060b3689ab4929bd7dc3555789a028
  result               168 passed, 1 warning
full: /mnt/persist/chensiyu/workspace/perf-2026q3/r12-gmm-softsync-validation-20260829/
  unit-final-a745ab6-full-20260830-170433-1937841-972655798
  pytest.log SHA256    5bcffb927f4b6ce1d45efc2b3053d3132a7d1ce54de464aa84f7e717f18c39b2
  manifest.txt SHA256  59a5a4ab958676999d54c85d6544eaadbb86739d18073910456dd0dbdfd7789a
  result               535 passed, 4 skipped, 3 warnings
```

whole compile：

```text
0162:.../r12-gmm-softsync-validation-20260829/
  runs/whole-compile-20260829-230237-1790530-244254697
```

`compile_report.json` SHA256：
`ccfe643deb7a1d09e0809a62f6a19ea106426e6c39356c1fc643d58679607976`。

direct CCEC：

```text
0162:/mnt/persist/chensiyu/workspace/perf-2026q3/
  compile-gmm-dual-latch-ccec-r12-v5-20260829-MEmpE7
```

## 5. 整网 H4-all A/B/A

最终 v8：

```text
0162:.../r12-gmm-softsync-validation-20260829/
  runs/h4-all-aba-20260829-230334-1791325-441605187
```

| arm | source | p50 | hidden / token |
|---|---|---:|---|
| A1 | `e6c7d8ec` baseline | `21.099 ms` | exact |
| B | active-worker dual-latch | `20.172 ms` | exact |
| A2 | `e6c7d8ec` baseline | `21.107 ms` | exact |

`aba_admission.json` SHA256：
`a7d5842b145b0f85727890d4eaaab89b5d513c58828ddf533b3fcda18e7d6680`。

### 5.1 已否决的 fixed-22 参与者

前一版让固定 22 个 resident participant 进入 latch：

```text
runs/h4-all-aba-20260829-203409-1647239-998273119
gain:          0.588 ms
required:      0.616 ms
admission:     FAIL
```

该方向低于检测地板，明确 **NO-GO，不要恢复**。最终 v8 的收益来自让 latch participant
随 active local expert 数收缩，而不是仅把同步拆成两个固定 22-worker counter。

## 6. 五层 DFX 与边界

```text
0162:.../r12-gmm-softsync-validation-20260829/
  runs/five-layer-bs1-dfx-20260829-231928-1820586-810930323
```

- outer admission **PASS**：L3/L4 baseline 与 candidate payload byte-exact、finite；
- analyzer gate **PASS**：`blockers=[]`，source identity PASS；
- policy：`release-local-ep-cdb2bb26-resident-dual-latch-22-v2`；
- mixed-grid gate：fused `22 AIC / 44 AIV`，BS1 down `23 AIC / 46 AIV`；
- outer `admission.json` SHA256：
  `206bb9e371a7c6ad759c36572791b34450e5820d3d818c896d0d4965bb4cfa3d`；
- candidate analyzer report SHA256：
  `d50a193b3c5f0e10f65d85ea04898abada2c79c53ffb7ab3c7fdcab92ffcf84b`。

完整 release readiness 仍为 **`NOT_EVALUABLE`**：本轮 DFX 没有 exact `recv_meta`
token-to-expert / per-expert count sidecar，`publication_allowed=false`。dependency task 和
physical execution slice 只能证明执行形状，不能冒充 route histogram。reference-rank span
受采样与 peer-arrival spin 影响，本轮不声明稳定 DFX span 收益。

仅按 execution-nonempty physical slices 做协议参与者估算，两层 atomics 由固定参与者的
`528` 降至 `224`（约 `-57.6%`），poll participant 由 `528` 降至 `96`
（约 `-81.8%`）。这组数只解释机制，不是实际 atomic/poll 指令计数，也不是 route histogram。

### 6.1 Validation harness 修正（与优化结论分账）

fatal-marker scanner 原先用裸 `507018` regex，曾误命中时间戳
`251929535070183` 的数字子串。最终 harness 改为数字边界匹配，并增加 shell regression。
这只消除 false fatal，不构成优化正确性或设备稳定性的额外证据。

## 7. 下一门

1. **已完成**：`a745ab6` 已以 exact lease 合入 pypto-lib `stepfun/develop`，远端 SHA exact；
2. **已完成部分**：successor local image 已构建并通过 digest-only audit/H4/extended gate；
3. **仍 open**：生成 provenance-matched route identity/`recv_meta` publication，解决
   `local-routes.v2` exporter 与 `recv-meta.v1` validator 的合同差异；
4. **仍 open**：registry push/raw/fresh pull、历史性能同合同 long gate 与最终 release contract。
