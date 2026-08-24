# 2026-08-23 · 全栈升级候选镜像三门实测（精度 / ITL / 前五层 swimlane）

> **历史受阻候选，已由 2026-08-24 r9 最终发布取代。** 当前结论、最终 pins、
> registry digest、H4 `all/none` 性能口径及前五层正式 admission 见
> [`2026-08-24-upgrade-r9-release.md`](2026-08-24-upgrade-r9-release.md)。

| 字段 | 值 |
|------|----|
| **镜像** | `hub.i.basemind.com/stepcast/vllm-pypto:stepfun-upgrade-20260822`，manifest `sha256:cafbc4d9399b14da3fa75370f0097d6032a41bc69ee7b3b5838745b9e62b509c`、config `sha256:2907eab75dfd889d5d5fce07e3c01d52e4317809fa5a7ddeed0b15ec3a2085f6` |
| **pin** | pypto `143ea205` / pypto-lib `26977738` / pto-isa `cd4a3d3f` / PTOAS-src `307d0484` / simpler `85a82c45` / ptoas-bin `v0.57` |
| **spec** | [`../deployment/docker/builds/stepfun-upgrade-20260822.env`](../deployment/docker/builds/stepfun-upgrade-20260822.env) |
| **机器** | gpu-a910x-0162 |
| **前置门** | `IMAGE_VERIFY=PASS`（镜像内、digest 定身）、`[smoke] PASS`、整网 liveness `SINGLE_CHIP_HIDDEN_CI=PASS` rc=0 201.376s `tokens_exact` |
| **口径** | **三门全部在镜像内跑，无 source overlay、无 runtime overlay**。这是本次升级第一次拿到镜像级（而非 source-overlay）的精度证据 |

> 升级动机不是性能，是解锁 **H4**（resident step-invariant args）：H4 撞非确定性
> `orch_error_code=8 TENSOR_WAIT_TIMEOUT`，根因 simpler 上游 #1902 无法 backport
> （`buffer.h deleted in HEAD`），整栈升级是唯一路径。H4 本身仍未复测（见文末）。

---

## 1. 精度准出 —— PASS（127/128 = 99.2%，门限 95%）

**这是唯一算精度准出的口径**：多步 decode 逐 token vs vanilla。单 token
`argmax==303` 只是 liveness 冒烟。

两阶段，**都在被测镜像内**：

| 阶段 | 在哪 | 做什么 |
|---|---|---|
| A | cards 0-7，port 8000 | vanilla vLLM W8A8（torch_npu 路径）逐 token 喂显式 id，产出 N=128 oracle |
| B | cards 0-7（A 停容器后串行） | pypto 整网 holder teacher-forced 复放同一 id 序列，逐位比 top-1 |

**A 阶段是真 vanilla**：`vllm/model_executor/models/step3p5.py` 只在
`PYPTO_STEP3P5_TAIL_ONLY` 为真时才走 pypto，默认 `'0'`；脚本显式 `unset` 它，
所以继承来的值不可能把 oracle 悄悄变成候选自己。

```text
LIVE_AB_ALIGNED=127/128 (99.2%), threshold=95.0%
RESULT=MAIN_HIDDEN_ONLY_TEACHER_FORCED_MATCH_127_of_128
rows 128   miss [20]
PRECISION_GATE_PASS
```

| 参照 | ALIGNED |
|---|---|
| **本次（升级候选镜像）** | **127/128 = 99.2%** |
| r5（2026-08-20） | 127/128 |
| 2026-07-23 | 124/128 = 96.9% |
| Wave5 | 123/128 = 96.09%，miss `[2,8,13,22,82]` |

### 1.1 唯一的 miss 落在 step 20，而那正是 oracle 自己分叉的位置

新 oracle 与 2026-08-20 冻结序列的一致前缀 = **20**
（`ORACLE_AGREEMENT_PREFIX=20`）—— 也就是 vLLM **自己**在不同 build 之间就会在
position 20 翻面。pypto 唯一的 miss 恰好也在 20。**结论：position 20 是 logits
近似平局，不是 pypto 缺陷。**

⚠ 顺带纠正一个容易犯的错：**不能拿"新 oracle 是否逐位等于旧冻结 oracle"当检查**。
128 步 greedy 自续写是混沌的，两个不同 build 的 oracle 一定共享一段前缀然后永久分叉
（2026-07-27 vs 2026-08-20 就在 0..48 一致、其余 128 位里差 75 个）。正确的读法是
**一致前缀长度**：< 8 说明 checkpoint 或 serving 配置错了；20 / 49 这种是正常分叉。

### 1.2 checkpoint 身份

两阶段各自独立 re-hash 全部 51 个实体（49 shard，约 206 GB，每次约 2m51s）后
`compare-checkpoint` 通过：

```text
identity_sha256      afaca2f91f3566de113bb8e9816476a882bf3d8351ff87a60e4c117afdc3795f
logical_id           step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp
weight_shard_count   49        weight_tensor_count  109542
manifest             r5-live-oracle-20260820-…/checkpoint_identity.json
```

与 2026-08-20 r5 的 manifest 逐位一致 ⇒ **权重没变，精度差异只可能来自代码**。

### 1.3 复现

```bash
# 0162，持 0162-cards0-7.lock
bash deployment/docker/run_precision_gate.sh cafbc4d9399b 128 6127
```

artifacts：`0162:/mnt/persist/chensiyu/workspace/upgrade-20260821/precision-20260823-175406/`
（`oracle.txt`、`oracle_ids.json` sha `3c03d0ef07e4363d…`、`oracle_prefix.txt`、
`pypto.log`、两份 `*_checkpoint_identity.json`、`verdict.txt`、`run_contract.txt`）。
墙钟 17:54:06 → 18:08:39 = 14m33s。

---

## 2. ITL —— 采到了，但是**回退**（47.99 ms vs 基线 32.14 / 26.33）

未插桩，cards 0-7，`active_batch=1`、`num_blocks=512`、`batch_capacity=16`、
无任何 override（`batch_capacity` 默认就是 16，`codegen_max_workers` 只影响编译期），
**与 K8 镜像那次的工作点一致**。

### 2.1 数字

| run | ctx | iters | warmup | p50 | mean | p99 | min | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| long | 65536 | 1000 | 10 | **47.993** | 48.647 | 53.985 | 46.967 | 201.029 |
| curve | 1024 | 20 | 3 | 48.293 | 52.509 | 134.99 | 47.404 | 134.99 |
| curve | 8192 | 20 | 3 | 48.213 | 48.251 | 48.91 | 47.758 | 48.91 |
| curve | 32768 | 20 | 3 | 47.886 | 59.239 | 268.1 | 47.644 | 268.1 |
| curve | 65536 | 20 | 3 | 47.963 | 49.377 | 61.095 | 47.516 | 61.095 |

| 对照 | ctx-64K BS1 p50 | Δ vs 本次 |
|---|---:|---:|
| **本候选镜像** | **47.993** | — |
| K8 镜像 `076af8a1`（上一个镜像） | 32.14 | **+15.85 ms / +49.3%** |
| R5 source-overlay（升级前的历史 MoE 基线） | 26.329 | +21.66 ms / +82.3% |
| Wave5 镜像（完整准出回退基线） | 49.796 | −1.80 ms |

**context 曲线完全平**：1024 → 65536 只从 `48.293` 走到 `47.963`（`−0.33 ms`，噪声内）。
对照 2026-07-29 那条曲线是 `+13.3 ms / +18.8%`。attention 的工作量必然随 context 线性增长，
所以「平」只能意味着**一个与 context 无关的固定开销现在比 64K 全部 attention 还大**。
这与 2026-07-29 §3.1 描述的旧 `654 ms` floor 是同一种形状 —— **固定 floor 回来了**。

### 2.2 归因：不是 device，也不是 `bind.args`，是**新增的 parent 侧 `node.*` 层**

升级把 host span 树从 `simpler_run.*` 改名成 `chip.run.*`，两边一一对应。三份日志用同一
脚本聚合（steady state `inv >= 20` 的 p50，单位 ms）：

| span | K8 镜像 32.14 | R5 src 26.33 | **本候选 47.99** |
|---|---:|---:|---:|
| 每 rank run 总计 | 29.261 | 24.122 | **26.974** |
| `…runner_run` | 22.764 | 18.285 | 21.058 |
| `…device_wall` | 22.153 | 17.623 | 20.948 |
| `…device_wall.orch` | 21.686 | 17.386 | 20.680 |
| `…bind` | 6.888 | 6.287 | 5.872 |
| `…bind.args` | 6.862 | 6.267 | **5.843** |
| **`node.graph_build`** | **—** | **—** | **8.326** |

三条结论：

1. **device 侧没有回退**：每 rank `chip.run` `26.974` ms **比 K8 的 `29.261` 还好**，
   只比 R5 差 `2.85 ms`。`device_wall.orch` 同理。
2. **`bind.args` 没有回退**：`5.843` ms 是三者里**最低**的（H4 立项时记录的是 `6.12 ms`）。
3. **`node.graph_build` 只在本候选存在**，两个基线里**没有对应 span**，且 991 个样本 /
   991 步 ⇒ **每步都跑一次，每次 8.326 ms**。其余 `node.*`
   （`submit` / `dispatch` / `frame_submit` / `complete` / `post_fence_retirement`）同样是新增。

算总账：`47.99 − 26.97`（最慢 rank 的 run）≈ `21 ms` 落在 run 之外；K8 那边同一算法是
`32.14 − 29.26 ≈ 2.9 ms`。**+16 ms 全部来自新增的 parent 侧 node 层**，其中 `8.33 ms`
已被 span 直接量到。

⚠ 曾怀疑「新 base 默认打开了 host STRACE，日志写放大了延迟」—— **证伪**：旧 base
`e2efebcb` 的 `profiling_config.h` 同样是 `#define SIMPLER_HOST_STRACE 1`，两个基线的
`container.log` 里各有 8325 / 8532 条 `[STRACE]`。之前 grep 不到只是因为**span 名改过**。

> 这条走的正是 LESSONS「为了拿一个数再跑一轮 device → 先穷举现有 artifact」：
> 上面整张对照表**没有多跑任何一次 device**，全部来自三份已存在的日志。

`node.graph_build` 每步重建，而 `chip.run.bind.prebuilt` 只要 `0.003 ms`
—— 说明 prebuilt 快路径存在但 node 层没用上。最可疑的是上游 pypto #2273 的
address-free wire ABI 把 descriptor 改成**每次 dispatch 由 `arg.buffer.tensor()` 导出**。
**H4（resident step-invariant args）针对的正是这一项开销**。

### 2.3 复现

```bash
bash deployment/docker/run_itl_gate.sh cafbc4d9399b
```

artifacts：`0162:/mnt/persist/chensiyu/workspace/upgrade-20260821/itl-20260823-183225/`
（`long/itl_report.json`、`curve/itl_report.json`、`long.log` 163656 行含全部 STRACE span）。

---

## 3. BS1 前五层 swimlane —— FAIL，被 UPGRADE-IPC-PROV 挡在门外

```text
TypeError: Parameter 'input_rms__ssa_v0' shard 0: a raw-pointer DeviceTensor
cannot be dispatched by DistributedWorker; use this same
DistributedWorker.alloc_tensor() to create it.
container rc=1        l2_swimlane_records.json: 一个都没产出
```

**和整网那次是同一个错**，只是我的修复只覆盖了 `whole_decode_holder.py`。还有三处
holder 仍在造裸指针 `DeviceTensor`：

| 文件:行 | 形态 | 修法 |
|---|---|---|
| `five_layer_moe_holder.py:320` | `device_tensor(key)[start:stop]` | `device_tensor_slice(key, start, stop)`，与 `whole_decode_holder.py:328` 同一处改动 |
| `five_layer_moe_route_holder.py:377` | 同上 | 同上 |
| `mtp_layer_holder.py:320` | `DeviceTensor(source.data_ptr, shape, dtype)` reshape 整个 key，把 buffer 丢了 | 走 `reshape`（已修好会保留 buffer）或 `_imported_tensor` |

⚠ **顺带暴露一个更大的盲区**：`mtp_layer_holder.py` 也在名单里，而整网 liveness 门是带
`--skip-mtp` 跑过的 ⇒ **MTP 整网路径在新 base 上根本没被验证过**。

所以本次拿不到 swimlane 占比，无法回答「device 侧关键路径是否被升级改变」。
—— 不过 §2.2 的 span 对照已经从另一个方向回答了：device 侧没有回退。

artifacts：`0162:…/upgrade-20260821/swimlane-20260823-190756/`（`container.log`、`container.rc=1`）。

---

## 4. 三门结论

| 门 | 结果 |
|---|---|
| 精度准出 N=128 | **PASS** `127/128 = 99.2%`（门限 95%） |
| ITL | **回退** `47.99 ms` vs K8 镜像 `32.14`（`+49.3%`），归因 = 新增 parent 侧 `node.graph_build` 8.33 ms/步 |
| 前五层 swimlane | **FAIL** —— 三个 holder 未修完 UPGRADE-IPC-PROV |

⇒ **不推 `stepfun/develop`**。精度过了，但性能回退 + 观测性门跑不起来 + MTP 未验证，
三条都不满足「没有问题」。

---

## 5. 仍未跑（所以这不是完整 production release-qualified）

- **H4 复测**（`PYPTO_H4_RESIDENT=all`，一轮长跑 `ITERS`×10）—— 整次升级的目的本身，
  且 §2.2 表明它现在的目标（每步 arg/graph 工作）比升级前更值钱
- MTP 整网（`--skip-mtp` 去掉后重跑）
- Main batch16 / MTP batch1+16、六档独立 64K golden-A/B、formal matched-source DFX
- 完整矩阵回退基线仍是 Wave5
