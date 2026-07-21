# 专项：EpTpMoE 8 卡 real-W8A8 gate_topk 死锁（mrgsort format2-on-unsorted）

| 字段 | 值 |
|------|----|
| **子系统** | whole-net / MoE gate |
| **error signature** | `507018` + `orch_error_code=0 sched_error_code=100 runtime_status=-100`（SCHEDULER_TIMEOUT）；stuck kernel `gate_topk`（func_id 2） |
| **首次出现** | 2026-07-06 |
| **状态** | ✅ 已解（gate 死锁段；routed 精度段另案 ⏸） |
| **相关 skill / doc** | [`../reference/moe-constraints.md`](../reference/moe-constraints.md)、`pypto-lib/models/step3p5/moe.py::_gate`、DeepSeek `models/deepseek/v3_2/deepseek_v3_2_decode_front.py` |

## 1. 背景（Background）

2026-07-06 会话。目标：standalone `EpTpMoE` MoE-block 在 8 卡真实 W8A8 上跑通 + 精度对齐 vLLM `ffn_out`（decode，layer 3，T=16）。gate / dispatch / experts / combine 全在 pypto 上（vLLM 只做调度 / KV）。

- 验证机：`gpu-a910x-0162`，cards 8-15，CANN 9.0.0 non-GA。
- 精度 harness：`pypto-lib/_stage_moe_block_precision.py`。
- 代码：`csy0225/pypto-lib` 分支 `wip/moe-gate-fix-20260706`（commit `956aede`）。

## 2. 现象（Symptom）

8 卡 real-W8A8 跑 `EpTpMoE`，报：

```
507018
orch_error_code=0 sched_error_code=100 runtime_status=-100   # SCHEDULER_TIMEOUT
```

V0 设备停机快照：所有 core idle，`completed=2/7 running=2`，卡死 task
`state=RUNNING kernels=[aic:-1 aiv0:2 aiv1:-1]`。用该 build 的
`next_levels/chip_orch/kernel_config.py` 映射 **func_id 2 = `gate_topk`**。
即 `gate_matmul`(0,1) 完成后卡在第 3 个 kernel `gate_topk`，combine 从未执行到。

## 3. 根因（Root Cause）

定位方法（按 simpler Device-Error-Codes wiki）：

1. wiki 指明 `sched_error_code=100` 要看 sub_class + `stuck_task_id`，需 V0 设备日志。
2. 开 V0：`logging.getLogger("simpler").setLevel(15)`（15==V0，必须在 `worker.init()` 前）；`ASCEND_PROCESS_LOG_PATH=<预建目录>`。
3. 拿到停机快照（见 §2），func_id 映射锁定 `gate_topk`。

对照 DeepSeek v3_2 working gate（同 pto-isa 栈）：step3p5 `_gate`（`moe.py`）top-k 用了：

```python
srt = pl.sort32(row, idx_init)                 # 32-run
srt = pl.mrgsort(srt, block_len=64)            # 64-run
srt = pl.mrgsort(srt[:, 0:512], srt[:, 512:1024])   # <- format2 两路归并
```

第 3 步是 **format2 两路 mrgsort**，要求两个入参各自是**单个完全有序**的序列；但 `block_len=64` 之后每个 512 半块只是 8 段 64-run（**未完全排序**）。把未排序序列当有序去归并 → 结果错误 + **归并状态机在分散分数上不终止 → `gate_topk` kernel 挂死**。

数据相关性证据：`post_attn_norm` 分散分数挂死；`post_attn_residual` 饱和 / 大量重复分数反而能过 —— 与"ties 越多越挂"直觉相反，证伪了 ties 假设。

DeepSeek v3_2（`models/deepseek/v3_2/deepseek_v3_2_decode_front.py`）用 **format1 渐进链** `mrgsort(64)->256->1024->4096`（每级 4 路，`block_len` = 输出 run 长度），完整排序。

分界证据：上游 topK NaN 修复 `pto-isa dda4b6b3` 已在 HEAD（非本问题）；dtype（fp32 分数 + uint32 idx）与 DeepSeek 一致（非 dtype 问题）。

## 4. 如何解决（Fix）

修复 `moe.py::_gate`（canonical，无 flag / 绕过）：

```python
srt = pl.sort32(row, idx_init)
srt = pl.mrgsort(srt, block_len=64)   # 64-run
srt = pl.mrgsort(srt, block_len=256)  # 256-run -> 一趟排满 1024 packed 宽度
```

**结果**：canonical gate 8 卡跑通 ~20s，**无 507018**，topk 与 vLLM 一致。

**适用边界**：本修复只解决 gate_topk 死锁。routed 路径精度仍 ⏸ 未解（见 §5 末段）。

**同类排查提示**：任何用 `sort32 + mrgsort` 做 top-k 的地方，`mrgsort` 必须用 format1 渐进 `block_len` 链排满整宽，**不要**用 format2 去归并未完全排序的半块。

## 5. 走过的弯路（Detours / What We Got Wrong）

- ❌ 假设 ties（大量重复分数）导致挂死 → 证伪：`post_attn_residual` 饱和 / 大量重复分数反而**能过**，`post_attn_norm` 分散分数才挂；挂死与 ties 多寡反向。
- ❌ 假设 dtype（fp32 分数 + uint32 idx）与 DeepSeek 不一致是根因 → 证伪：逐位核对一致。
- ❌ 假设上游 topK NaN bug（`pto-isa dda4b6b3` 之前）未修 → 证伪：HEAD 已含该修复。
- ❌ routed 精度方向上多个假设被证伪（用 0.12% torch 参考逐项排除）：
  - act-quant（BF16 torch 参考就匹配 `ffn_out` 0.12% → 不需要 int8 act-quant）；
  - gate（topk 正确）、权重（反量化验证 ~0.01 正常）；
  - `moe_parts` dump（**不可靠**：shared+routed != ffn，ch4094 差 567；`routed_output[token0]` 全零 → 是 per-rank 部分量，别用它做参考）；
  - dispatch 读偏移（`ep_all_to_all` pull READ 偏移改为 `read_offsets[peer]=Σ_{d<my_rank}Σ_e pub_counts[peer*N+d,e]` → **no-op**）；
  - combine push 原语（`remote_store` → DeepSeek 式 `pld.tensor.put` → **no-op**）；
  - 逐行审计 dispatch pack / publish-idx / re-pack、`_expert_routed` 写 `local_routed_y`、combine push / gather 的行序索引，**全部一致**。

routed 精度结论：bug 在 `_expert_routed` 的分块 grouped-GEMM（`pl.parallel` + RECV_TILE=32 分块 + `pl.spmd` + `valid_shape` 部分-tile 处理）的数值 / 调度细节，或该 spmd 的设备级 hazard —— **读代码定位不出**。下一步（确定路径）：逐级设备 dump —— 把 `local_routed_x`（dispatch 后）→ `local_routed_y`（combine 前每 expert 输出）→ `routed_y_buf`（combine 后）导出为 `EpTpMoE.chip_orch` / `host_orch` 调试输出，逐级对 torch 参考比对；哪级发散即 bug 所在（grouped-GEMM vs gather）。

## 6. 如何避免（Prevention）

- **铁律**：任何 `sort32 + mrgsort` top-k 实现必须用 format1 渐进 `block_len` 链排满整宽；禁止 format2 归并未完全排序的半块。新 kernel 写 top-k 时先对照 DeepSeek v3_2 working gate。
- **早期识别信号**：`507018` + `sched_error_code=100` + stuck kernel `gate_topk`（V0 快照 func_id 映射）→ 立刻查 `mrgsort` 调用形态。
- **V0 日志获取**：`logging.getLogger("simpler").setLevel(15)` 必须在 `worker.init()` 前；`ASCEND_PROCESS_LOG_PATH` 预建目录。停机快照在 `$ASCEND_PROCESS_LOG_PATH/debug/device-*/device-*.log` 的 `[STALL ...] TASK ... state=RUNNING kernels=[aic aiv0 aiv1]`；kernel id → 该 build 的 `chip_orch/kernel_config.py` func_id。
- **运维要点（本会话踩坑）**：
  - 禁止 `npu-smi set -t reset`（AMP+HCCS netboot 机会重启全部 16 卡 → SSH-key 抹除 → 锁死）。设备 fault 由进程 finalize 的 `aclrtResetDeviceForce` 清。
  - 勿 `-9` 强杀在 device 上的进程（无 finalize → card poison → 下一次 507018）；等 finalize 跑完 force-reset 再重启。
  - 8 卡 harness 每次要加载 8×~47GB W8A8 bundle（~5 分钟），单轮很慢。
- **shared expert 路径已验证数值正确**（本会话）：device shared 输出（含 ring→barrier-mesh `tp_all_reduce` 移植，mirror `decode_layer.py` DenseFull / `test_l3_allreduce.py`）对 torch shared **PASS**（`--zero-routed --torch-golden`）；建 torch BF16 参考 `shared = Σ_r SwiGLU(x@wg_s[r])@wd_s[r]` 对 vLLM `ffn_out` 差 0.12% = ground truth。后续改 MoE 别回归 shared。
- 相关约束落点：`pypto-lib/docs/known-pypto-pitfalls.md`、本仓 [`../reference/moe-constraints.md`](../reference/moe-constraints.md)。
