# Troubleshooting: EpTpMoE 单块 8 卡真实 W8A8 bring-up（gate_topk 死锁 + routed 精度）

> 2026-07-06 会话。目标：standalone `EpTpMoE` MoE-block 在 8 卡真实 W8A8
> 上跑通 + 精度对齐 vLLM `ffn_out`（decode，layer 3，T=16）。gate/dispatch/
> experts/combine 全在 pypto 上（vLLM 只做调度/KV）。
>
> 验证机：`gpu-a910x-0162`，cards 8-15，CANN 9.0.0 non-GA。
> 精度 harness：`pypto-lib/_stage_moe_block_precision.py`。
> 代码：`csy0225/pypto-lib` 分支 `wip/moe-gate-fix-20260706`（commit 956aede）。

---

## 1. gate_topk 8 卡死锁 —— ✅ 已解决（真修复，非绕过）

**症状**：8 卡 real-W8A8 跑 `EpTpMoE`，`507018` + `orch_error_code=0
sched_error_code=100 runtime_status=-100`（SCHEDULER_TIMEOUT）。

**定位方法（按 simpler Device-Error-Codes wiki）**：
1. wiki 指明 `sched_error_code=100` 要看 sub_class + `stuck_task_id`，需 V0 设备日志。
2. 开 V0（`logging.getLogger("simpler").setLevel(15)`，15==V0，必须在 `worker.init()` 前；
   `ASCEND_PROCESS_LOG_PATH=<预建目录>`）。
3. 设备停机快照：所有 core idle，`completed=2/7 running=2`，卡死 task
   `state=RUNNING kernels=[aic:-1 aiv0:2 aiv1:-1]`；用该 build 的
   `next_levels/chip_orch/kernel_config.py` 映射 **func_id 2 = `gate_topk`**。
   即 gate_matmul(0,1) 完成后卡在第 3 个 kernel `gate_topk`，combine 从未执行到。

**根因（对照 DeepSeek v3_2 working gate，同 pto-isa 栈）**：
step3p5 `_gate`（`moe.py`）top-k 用了：

```python
srt = pl.sort32(row, idx_init)                 # 32-run
srt = pl.mrgsort(srt, block_len=64)            # 64-run
srt = pl.mrgsort(srt[:, 0:512], srt[:, 512:1024])   # <- format2 两路归并
```

第 3 步是 **format2 两路 mrgsort**，要求两个入参各自是**单个完全有序**的
序列；但 `block_len=64` 之后每个 512 半块只是 8 段 64-run（**未完全排序**）。
把未排序序列当有序去归并 -> 结果错误 + **归并状态机在分散分数上不终止 ->
`gate_topk` kernel 挂死**。（数据相关：`post_attn_norm` 分散分数挂死；
`post_attn_residual` 饱和/大量重复分数反而能过 —— 与"ties 越多越挂"直觉相反，
证伪了 ties 假设。）

DeepSeek v3_2（`models/deepseek/v3_2/deepseek_v3_2_decode_front.py`）用
**format1 渐进链** `mrgsort(64)->256->1024->4096`（每级 4 路，`block_len` = 输出
run 长度），完整排序。

**修复**（`moe.py::_gate`，canonical，无 flag/绕过）：

```python
srt = pl.sort32(row, idx_init)
srt = pl.mrgsort(srt, block_len=64)   # 64-run
srt = pl.mrgsort(srt, block_len=256)  # 256-run -> 一趟排满 1024 packed 宽度
```

**结果**：canonical gate 8 卡跑通 ~20s，**无 507018**，topk 与 vLLM 一致。
上游 topK NaN 修复 `pto-isa dda4b6b3` 已在 HEAD（非本问题）；dtype
（fp32 分数 + uint32 idx）与 DeepSeek 一致（非 dtype 问题）。

**同类排查提示**：任何用 `sort32 + mrgsort` 做 top-k 的地方，`mrgsort`
必须用 format1 渐进 `block_len` 链排满整宽，**不要**用 format2 去归并未
完全排序的半块。

---

## 2. shared expert 路径 —— ✅ 验证数值正确

建 torch BF16 参考（`shared = Σ_r SwiGLU(x@wg_s[r])@wd_s[r]`，对 vLLM
`ffn_out` 差 **0.12%** = ground truth）。device shared 输出（含 ring->
**barrier-mesh** `tp_all_reduce` 移植，mirror `decode_layer.py` DenseFull /
`test_l3_allreduce.py`）对 torch shared **PASS**（`--zero-routed --torch-golden`）。

---

## 3. routed 路径精度 —— ⏸ 未解决（已精确隔离）

**症状**：`--zero-shared --torch-golden`：device routed vs torch routed
**FAIL 41.8%**，小值**符号翻转**（[4] dev -0.041 vs ref +0.038 等）。全量
`moe_out` vs `ffn_out` 也 ~41%。

**已排除**（用 0.12% torch 参考）：
- act-quant（BF16 torch 参考就匹配 `ffn_out` 0.12% -> 不需要 int8 act-quant）；
- gate（topk 正确）、权重（反量化验证 ~0.01 正常）；
- `moe_parts` dump（**不可靠**：shared+routed != ffn，ch4094 差 567；
  routed_output[token0] 全零 -> 是 per-rank 部分量，别用它做参考）；
- dispatch 读偏移（`ep_all_to_all` pull READ 偏移改为
  `read_offsets[peer]=Σ_{d<my_rank}Σ_e pub_counts[peer*N+d,e]` -> **no-op**）；
- combine push 原语（`remote_store`->DeepSeek 式 `pld.tensor.put` -> **no-op**）。
- 逐行审计 dispatch pack/publish-idx/re-pack、`_expert_routed` 写
  `local_routed_y`、combine push/gather 的行序索引，**全部一致**。

**结论**：bug 在 `_expert_routed` 的分块 grouped-GEMM（`pl.parallel`
+ RECV_TILE=32 分块 + `pl.spmd` + `valid_shape` 部分-tile 处理）的数值/
调度细节，或该 spmd 的设备级 hazard —— **读代码定位不出**。

**下一步（确定路径）**：逐级设备 dump —— 把 `local_routed_x`（dispatch 后）
-> `local_routed_y`（combine 前每 expert 输出）-> `routed_y_buf`（combine 后）
导出为 `EpTpMoE.chip_orch`/`host_orch` 调试输出，逐级对 torch 参考比对。
哪级发散即 bug 所在（grouped-GEMM vs gather）。中等改动，建议在低上下文时
干净执行，避免破坏已验证的 gate/shared 修复。

---

## 运维要点（本会话踩坑）
- **禁止 `npu-smi set -t reset`**（AMP+HCCS netboot 机会重启全部 16 卡 -> SSH-key
  抹除 -> 锁死）。设备 fault 由进程 finalize 的 `aclrtResetDeviceForce` 清。
- **勿 `-9` 强杀在 device 上的进程**（无 finalize -> card poison -> 下一次 507018）；
  等 finalize 跑完 force-reset 再重启。
- 8 卡 harness 每次要加载 8x~47GB W8A8 bundle（~5 分钟），单轮很慢。
- V0 停机快照在 `$ASCEND_PROCESS_LOG_PATH/debug/device-*/device-*.log`
  的 `[STALL ...] TASK ... state=RUNNING kernels=[aic aiv0 aiv1]`；kernel id ->
  该 build 的 `chip_orch/kernel_config.py` func_id。
