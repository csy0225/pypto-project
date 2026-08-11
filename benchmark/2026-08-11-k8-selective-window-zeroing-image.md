# 2026-08-11 · K8 选择性清零 immutable image 发布回归

> 本文是 K8（persistent window 选择性清零）**镜像层级**的唯一性能/精度记录。
> K8 的**源码层级** A/B/A 三臂 bracket 记在
> [`../design/performance/task-tracking.md`](../design/performance/task-tracking.md)
> 2026-08-11 行；本文只记「同一份改动烧进 immutable image 后，在 0162 digest-only
> 复现了什么」。两者不得混写。

---

## 1. 发布镜像

```text
tag:
hub.i.basemind.com/stepcast/vllm-pypto:stepfun-develop-20260811-k8-selective
manifest: sha256:076af8a167405d5d0831e234cd16521c77d8bfdd173eff063d820802057c47f3
config:   sha256:a9d111880883cea0b02e425fdfeaccc2b14bb1d1174c0b73488d8ee6d8004d39
spec:     deployment/docker/builds/stepfun-develop-20260811-k8-selective.env
```

| 组件 | pin |
|---|---|
| pypto-lib | `cb96747eb21f5f4932d6a24eddaa69c85d095ef6` |
| pypto | `1c048a744d5f63a8bce1ddb45dac8d1b7f458bb0` |
| simpler | `e2efebcbd190302609c0775d2984f409f5f42c76` |
| pto-isa | `ecb6c303f797749f811a494742c3c08156aacabb` |
| PTOAS | `fc8c6caee561914b4fb991dfc8427bb63194269e` |
| ptoas-bin | `v0.50` |
| vLLM overlay | `1b3e538c35999e62b6d24e0651b3a85b7d16c826` |
| base | `hub.i.basemind.com/stepcast/vllm-pypto@sha256:3d6392588fe9fb6ce4f5852100667d24f09d70f262dbd0ebe6c45b380f49573a` |

`ATTN_TASK_PROFILE=a2a3`、`REQUIRE_L2_SWIMLANE_REUSE_DEP_GEN=1`。

### 构建位置（不是可选项，是环境约束）

构建在 **devbox**（有 docker + BuildKit secret）。0162 **结构上不能构建**：
buildkitd inactive、github 与 build proxy 均不可达（只有 hub 可达）。
**全部验证在 0162**，digest-only，无源码 / runtime overlay。

### 构建期修掉的三个凭据坑（同族，都是「凭据覆盖没生效 → 静默退化成匿名 GitHub → 走 build proxy 401」）

1. **submodule 凭据覆盖 key 用了 path 而非 name**：pypto 的 submodule **名叫
   `simpler`**、**路径是 `runtime`**。原 Dockerfile 写 `submodule.runtime.url`，
   git 静默忽略 → 匿名 clone `hw-native-sys/simpler` → 401。改成
   `submodule.simpler.url`。（`.gitmodules` 在 `defa97c5`/`8e92b468`/`1c048a74`
   三个 commit 上完全相同，所以这个 key 一直是错的，只是以前匿名访问还通。）
2. **3rdparty submodule 在无 secret 的编译层被 CMake 初始化**：pypto 的
   `cmake/libbacktrace.cmake:23` 会对 `3rdparty/libbacktrace` + `3rdparty/msgpack-c`
   跑 `git submodule update --init`，而那一层没挂 secret → 401 →
   `CMake configuration failed`。改为在带凭据的 clone 层用一次性
   `-c url.<token>@github.com/.insteadOf=...` 预置（token 只作用于这一条命令）。
3. **simpler 自己还有第二份 pto-isa**：`build_runtimes.py` 调
   `ensure_pto_isa_root()`，要求 `runtime/build/pto-isa` 是**干净且 HEAD ==
   `runtime/pto_isa.pin`（`83d01313`）** 的 checkout —— 这与
   `PTO_ISA_COMMIT=ecb6c303` 是**两份不同的 pto-isa**，且 `PTO_ISA_ROOT` 对它无效。
   上游 `pto_isa.py:_ensure_locked` 明确支持「预置的 pristine checkout 直接复用、
   不走网络」，故在 clone 层按 pin 预置并断言 `HEAD == pin` + `include/` 存在 +
   worktree 干净。

> 三条都已落进 `deployment/docker/Dockerfile` 并带注释说明**为什么**这么写。
> ①③ 同时也是 sub-repo 该记的 dev-workflow gotcha（落点
> `pypto-lib/docs/dev-workflow-gotchas.md`，不落本仓）。

---

## 2. 镜像内 audit + smoke（phase 1）

cards 无关，`nerdctl run` digest-only。四个 gate 全 PASS：

```text
IMAGE_IMMUTABLE_AUDIT=PASS
CANONICAL_ONLY_SYMBOL_AUDIT=PASS
K8_LANDING_PRESENT=PASS
[smoke] PASS
PHASE1_AUDIT_SMOKE=PASS
```

- 五仓 pin 逐一 `rev-parse HEAD` 命中且 worktree clean；
- git credential scrub PASS（新增扫描 `runtime/build/pto-isa`）；
- `attention profile: a2a3`；
- `prepared swimlane reuse capability: {available: True, constructed: True, required: '1'}`；
- `ptoas 0.50`；
- runtime `libsimpler_aicpu_dispatcher.so` 在位。

### 落地件 == 被测件

```text
models/step3p5/decode_fwd.py
  eb1f89bf7add419f2382836c1eab9a1c4b1f63f738923d47e771e4159f104fb5
pypto/python/pypto/runtime/distributed_runner.py
  fe50c11fb76ec77789636de05e7376711c731d2b00db5033f0564c07a739622e
```

后者与 `task-tracking.md` 记录的 K8 落地件权威 sha **完全一致**。

---

## 3. 精度：两条独立证据

### 3.1 byte-exact hidden（cards 0-7）

```json
{"hidden_sha256": "567b206bb03d89f84020e1dddd61098a8f79f32f81b8f4fcf56443113e27f03e",
 "matches_production_baseline_sha": true,
 "candidate_token": 14371, "expected_token": 14371, "token_exact": true,
 "hidden_finite": true, "shape": [8, 16, 4096], "dtype": "torch.bfloat16"}
```

**这条证明的是**：镜像与生产 baseline 逐字节一致。对 K8 这种语义保持型改动，
这就是精度准出。

**这条不证明的是**：它不是 128 个不同 step 的逐 token 精度 —— ITL harness 是
在固定 ctx 上把**同一个 step 重复 110 次**（10 warmup + 100 计时），只落一份
末次 payload。所以「100 次每次都校验过」是错的说法。

### 3.2 N=128 逐 token（cards 8-15，预定义冻结 oracle，三轮）

```text
round1: 123/128 = 96.09375%  miss=[2,8,13,22,82]  tp_spread_max=0.0  finite=True  PASS
round2: 123/128 = 96.09375%  miss=[2,8,13,22,82]  tp_spread_max=0.0  finite=True  PASS
round3: 123/128 = 96.09375%  miss=[2,8,13,22,82]  tp_spread_max=0.0  finite=True  PASS
N128_GATE=PASS
```

门限：per-position top-1 ≥ 95% 且 `tp_spread_max == 0`。**三轮完全一致，且与
Wave5 immutable 那轮逐位相同**（同样 `123/128`、同样 miss 位置、同样 spread=0）
⇒ K8 没有改动 token 轨迹。这与 §3.1 的 byte-exact 互为独立印证。

oracle 口径（**离线，无 live server**）：

```text
/mnt/persist/chensiyu/workspace/attn-opt/out/fresh_vanilla_oracle_20260731/oracle_ids.json
SHA256 c9b2c72121880e9c605ae70d1cf85c0d4fc8815b180598bc76f7e293551dd947
  SEED=6127  N=128  (frozen vanilla vLLM W8A8 greedy capture)
```

该 sha 与 Wave3 快照 `wave3_ab_n128_20260803_053242/oracle_ids.snapshot.json`
**相同**，即各 Wave 门用的同一份权威 oracle；生成它的 vanilla server 用的
checkpoint 就是本次被测的 `step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp`
（见该目录 `start_oracle.sh:14`），checkpoint 身份一致。

---

## 4. 性能：clean ITL（非插桩）

cards 0-7，ctx=65536、`active_batch=1`、`num_blocks=512`、`warmup=10`、`iters=100`、
`seed_token=6127`、`storage_batch_capacity=16`、`codegen_max_workers=1`。

| 指标 | ms |
|---|---:|
| min | 31.766 |
| **p50** | **32.14** |
| mean | 32.467 |
| p99 | 37.644 |
| max | 37.644 |

| 对照 | p50 | Δ |
|---|---:|---:|
| pre-K8（parent 臂） | 33.84 | — |
| **本镜像** | **32.14** | **−1.70 ms / −5.02%** |
| K8 source-overlay 候选臂 | 32.08 | `+0.06 ms` |

`+0.06 ms` 远小于 bs=1 检测地板 `parent_half_range = 0.634 ms` ⇒ 镜像与
source-overlay 候选臂**统计不可区分**，即镜像**复现**了 K8 收益。

> ⚠ bs=1 用 `blocks=512`、bs=8 用 `blocks=4096`，编译期容量不同，**绝对值不可横比**。

### K8 runtime 生效证据（reset trace）

109 条记录，109/109：

```json
{"k8_prefix_applied": true, "k8_control_bytes": 47616,
 "k8_control_range_count": 1, "k8_full_window_bytes": 32063232}
```

| 指标 | min | p50 | mean | p99 | max |
|---|---:|---:|---:|---:|---:|
| `reset_body_us` | 485.3 | **523.1** | 675.3 | 1930.8 | 5943.1 |
| `memset_all_us` | 443.8 | 481.2 | 632.5 | 1891.3 | 5901.0 |

`523.1 µs` 与 K8 A/B/A 实测 `518 µs` 一致（pre-K8 为 `2253 µs`）。

被清的 7 个 control buffer 正是 window 最前 7 个，字节和精确等于 `47,616`：

```text
dense_attn_signal_stack_buf__ssa_v0     1536
dense_mlp_signal_stack_buf__ssa_v0      1536
moe_attn_signal_stack_buf__ssa_v0      21504
moe_meta_arrived_stack_buf__ssa_v0       512
moe_data_arrived_stack_buf__ssa_v0       512
moe_sh_signal_stack_buf__ssa_v0        21504
moe_combine_arrived_stack_buf__ssa_v0    512
                                    = 47616 / 32063232 = 0.1485%
```

---

## 5. BS1 前五层 swimlane（L0–L4 focused，插桩）

工作点：cards 0-7、`active_batch=1`、`context_len=65536`、`num_blocks=512`、
`warmup=3`、`iters=20`。**插桩 run，绝对时间按纪律不可当干净延迟、不可乘层数比
反推整网**；只用于算**占比**。干净绝对延迟只认 §4。

### 5.1 先定分析 rank（LOW-WAIT），否则结论全错

同一次 8 卡采集的 makespan 跨 rank 差 **275×**：

| rank | makespan ms | 其中 `tp_all_reduce` ms | 占比 |
|---|---:|---:|---:|
| **rank2** | **2.204** | 0.336 | 15.3% |
| rank0 | 288.252 | 286.412 | 99.4% |
| rank1 | 398.750 | 396.921 | 99.5% |
| rank3 | 436.450 | 434.606 | 99.6% |
| rank7 | 444.583 | 442.748 | 99.6% |
| rank4 | 591.379 | 589.528 | 99.7% |
| rank6 | 606.033 | 604.180 | 99.7% |
| rank5 | 609.764 | 607.932 | 99.7% |

其余七个 rank 的 `tp_all_reduce` 是**自旋吸收 rank skew，不是算力**。这与 skill
记录的「跨 rank 2.210 ms ~ 555.892 ms」同形。**只有 rank2 可用于分析。**

### 5.2 rank2 关键路径

```text
tasks: 150 | happens-before edges: 229
makespan:               2.204 ms
static CPM path:        1.825 ms (82.8%) over 87 tasks
observed critical path: 103 tasks
  compute: 1.788 ms (81.2%)
  stall:   0.415 ms (18.8%)  —— 全部 data-wait，front-gap 0.000 ms
tiling check: compute+stall = 110182 ticks vs makespan 110182 ticks (exact)
```

按 kernel family 占 makespan 比例（Top）：

| kernel family | compute ms | % makespan | stall ms | # on path |
|---|---:|---:|---:|---:|
| `tp_all_reduce` | 0.336 | **15.3%** | 0.059 | 8 |
| `swa_chip_orch_dense_gate_up_matmul_tp` | 0.098 | 4.4% | 0.006 | 2 |
| `swa_q_proj` | 0.089 | 4.0% | 0.007 | 2 |
| `swa_out_proj_matmul` | 0.072 | 3.3% | 0.006 | 2 |
| `swa_moe_chip_orch_swa_q_proj` | 0.052 | 2.3% | 0.004 | 1 |
| `dense_gate_up_matmul_tp` | 0.049 | 2.2% | 0.003 | 1 |
| `swa_moe_chip_orch_expert_down` | 0.048 | 2.2% | 0.008 | 1 |
| `swa_moe_chip_orch_expert_gate_up` | 0.041 | 1.9% | 0.007 | 1 |
| `expert_gate_up` | 0.040 | 1.8% | 0.007 | 1 |
| `swa_chip_orch_dense_down_matmul_tp` | 0.037 | 1.7% | 0.019 | 2 |
| `swa_moe_chip_orch_swa_out_proj_matmul` | 0.037 | 1.7% | 0.003 | 1 |
| `swa_rmsnorm_zc` | 0.036 | 1.6% | 0.005 | 2 |
| `swa_chip_orch_dense_post_rmsnorm_zc` | 0.036 | 1.6% | 0.013 | 2 |
| `full_sv_matmul` | 0.034 | 1.5% | 0.007 | 1 |
| `combine_wait` | 0.027 | 1.2% | 0.001 | 1 |

### 5.3 本轮 campaign `rc=1`：如实记为「可用观测」而非 sealed publication

分析器 `analyze_five_layer_moe_dfx.py:1052` 的 fail-closed 结构契约在
**rank0/1/3/6** 拒收：每个 rank 各有 **5 个 `early_dispatch=true` 的 task** 出现在
`deps.json` 却在 swimlane 记录里缺失（rank0 为
`8589934741/743/744/745/747`，`block_num` 8/23/23/23/23）。

rank2 自身 150 task 无缺失、tiling check 精确，故 §5.2 的占比是**可用观测**；
但**整轮不是 sealed publication**。转正需先解决 `early_dispatch` task 的
swimlane 记录缺失（怀疑是记录窗口开启早于 early-dispatch，待查）。

因此 [`../design/performance/05-moe-optimization.md`](../design/performance/05-moe-optimization.md)
的 "Candidate merged swimlane" 段从 `PENDING` 改为「已采集、rank2 可用、
cross-rank 契约未过」，**不写成 final**。

---

## 6. 权威证据路径（全部在 0162）

### 镜像门（phase 1 + ITL）

```text
/mnt/persist/chensiyu/workspace/k8-image-release-20260811/
├── phase1/                                  # 独立 audit+smoke
├── gate-20260811-190328/
│   ├── phase1/
│   ├── itl-dev07/                           # cards 0-7，本文 §3.1 + §4
│   │   ├── precision_summary.json
│   │   ├── reset_trace.jsonl                # 109 records，§4 K8 生效证据
│   │   ├── runtime/itl_report.json          # §4 ITL 表
│   │   ├── decode_fwd.container.sha256
│   │   ├── distributed_runner.container.sha256
│   │   └── run_contract.json
│   └── itl-dev815/                          # phase 2b，按用户要求跳过（未完成）
└── bin/                                     # 采集件
    ├── audit_smoke.sh        1054357a579d2d1efde3baba6895aa687c628c1f4a3f318a97dae3f2b39fa34d
    ├── container_itl.sh      d9e786849207a0856522c1228acdfc977e2da73e740e40e80f75349d9ec9735d
    ├── run_itl.sh            35263e8b23aba10d960820d733aaa27a9c6504bba78242d7a80be7702fc0a5d0
    └── release_gate.sh       3d2baa8f809b0108fcdb02e9575273d0279216fc017d612dc7599174c2c62cc7
```

### N=128 精度门

```text
/mnt/persist/chensiyu/workspace/k8-image-release-20260811/n128-20260811-192620/
  container.rc = 0      19:26:21 -> 19:44:46
  N128_SUMMARY.txt                   c705ebb650ddfafd813df2aba31753fd813754edadec2f6d173e94ae7c196ee1
  run_contract.json                  382b7adcbd51dbd26b57792e0795172b0dba2b1975081193d090865d495a42e0
  round1/main_hidden_only_report.json c4addc56b5a6d86c88414af1bcf16b8afa9d86913691a6dd28e7dfb79edc8350
  round2/main_hidden_only_report.json 0418d081f52721094a092c98c442e12ebae20b116e946c8de966048a867318d7
  round3/main_hidden_only_report.json 2e156041bd5246b948d17c1fe052890bdbd613a595632af7f29c4e812123f6ef
采集件:
  bin/container_n128.sh  82e06e2800f53df17c72ba8259782630db970834bb8fca6e290a24cf675e0757
  bin/run_n128.sh        73ab0ebb877ed599eded7bf2e64d1a0715677f20ee13972da08052f0e2a3cac3
```

### 五层 swimlane

```text
/mnt/persist/chensiyu/workspace/k8-image-release-20260811/swim-20260811-192117/
  container.rc = 1   (§5.3：cross-rank 契约未过；rank2 可用)
  run_contract.json  c68b1d1213b1132f5ecfd645e54a7afbabe549d29ce3e53cf743e9117cb4a1ff
  runtime/build_output/FiveLayerMoe_20260811_112405/dfx_outputs/
    rank{0..7}/d0/{merged_swimlane_*.json,critical_path_report.md,
                   CPM_observed.json,CPM_static.json,deps.json,
                   l2_swimlane_records.json,name_map.json}
LOW-WAIT rank2 五件 sha256:
  merged_swimlane_20260811_112527.json e7c3ee771754f82f2cab9acba1a20c0b232efee0ea82db4bb8d5e0c6269e85ad
  critical_path_report.md              0cb08fa97fce923c033c9ed993dbaa26649f5c75b9627619970b31f959b0056e
  CPM_observed.json                    352b6eefb076720e2a3f48d2402a786b06f26048269be380a58aa37aa54a2d02
  CPM_static.json                      ef5a43a74619a644cb0ed25c0b7037ef09f0c25ecbac9e8e095d61cd5a7861eb
  deps.json                            f1254388db7e8ba633b2f460bdf9f7b354afb0bb51e5b35600af098cc79b4720
采集件:
  bin/container_swimlane.sh c2a08062d80d8df2d7552253a2b39bbcf21569e6b9a01bed00613b0b731cd971
  bin/run_swimlane.sh       c9109e517438d5a4da9210af1ae39a12b5b6607556018a748d7e160701f0c10a
```

### 卡与锁

- ITL（§3.1/§4）在 **cards 0-7**，持 `0162-full-machine-perf.lock` 串行；
- N=128（§3.2）在 **cards 8-15**，持 `0162-cards8-15.lock`；codex 19:23 实查
  dev8-15 空闲后放行；
- swimlane（§5）在 **cards 0-7**，持 `0162-cards0-7.lock`，运行时段 19:24–19:25，
  **与 N=128（19:26 起）无重叠**；
- 全部结束后三把锁均已释放。

---

## 7. 本轮**未做**的项（不得由本文数据代替）

| 项 | 状态 |
|---|---|
| phase 2b（cards 8-15 的 64K ITL 半机） | 按用户要求**跳过**；启动 68 s（仍在权重导出、未进 device 执行）时干净停容器，NPU 8-15 全部 "No running processes"、无残留进程 |
| Main batch16 | 未跑 |
| MTP batch1 / batch16 | 未跑 |
| 六档（BS 1/2/4/7/8/16）每请求独立 64K golden/A/B | 未跑 |
| formal matched-source DFX + route-aware reanalysis | 未跑 |
| 五层 swimlane cross-rank 契约 | **未过**（§5.3） |

⇒ 本镜像**不是**完整 production release-qualified；完整矩阵的回退基线仍是 Wave5
（`sha256:4acc77cd…`）。
