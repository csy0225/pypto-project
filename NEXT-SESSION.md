# 下一阶段启动提示词 —— G2-G5 live wiring（tasks 5-7）集中攻克

> 新 session 直接把下面 code block 作为第一条消息粘贴即可。自包含，不依赖记忆。
> 生成于 2026-07-11，承接本 session G1 offline 全线打通（tasks 1-4 ✅）。
> 详细里程碑见 `archive/milestones-2026-Q2.md` 2026-07-10 (续²~续⁵)。

## G1 offline 已完成（tasks 1-4，别重做，已 device 验证 + 独立复核 + push）

- **task 1**：真 W8A8 接 dense/attn。offline worker `_stage_whole_decode_run.py` 4 层链
  cards 8-15 真 W8A8 **device rc=0 无 507018**，输出全 ≠ synth。修 3 host bug（`_set_gate_exp`
  广播 / `_recon_attn` per-rank w_g / `_share` 连续化）。
- **task 2**：torch-ref 逐层对拍。full-attn(L0)+MoE-block(L3) **精确 1.000**，坐实 3-scalar
  layer_idx split。SWA-attn 稳定 0.994（满足 max_error_ratio=0.10，非索引错位）。
- **task 3**：L43/L44 SplitIncoreOrch 编译修复。根因 `_quant_moe_input`（moe.py:1801，仅 swiglu
  路径调）`@pl.function(InCore)` 的 pl.spmd body 触发 #1828 → 改 **InCore→Inline**（body 字节不变，
  对齐 `_expert_routed`）。compile SMOKE_RC=0 + device rc=0。
- **task 4**：per-layer weight-stream 重构 + 45 层链。修 `_stack_real` 的 3.5TB mega-stack OOM
  （改 `_moe_layer_stack` slice-then-stack）+ 同 variant 多层复用首层权重 bug（`_load_moe_layer_weights`
  每 step copy）。**45 层全链 device rc=0**（7 programs, 87 steps）；per-layer 权重坐实（L3/L4/L5
  各异 moe_out 1.000）。
- **独立复核**：reverse-review 对 4 处改动全 GO（D 数值安全）；sw-analyst 独立发现同一 per-layer
  权重 bug + 确认 norm[45]abs / MoE[42]pos=li-3 / dense[3] 索引分离 + EPS=1e-5 全对。

## 本 session 额外 de-risk（两堵墙已清 / 已定位）

- **HBM 非门槛**（旧 memory「24G+47G=OOM」是误判）：npu-smi 实测 cards 8-15 = **64GB/卡**。
  TP=8 sharded：vLLM W8A8 ~3GB/卡 + pypto BF16 ~6GB/卡 + KV ≈ ~10GB/卡 ≪ 64GB。**G3 HBM 不挡路**。
- **resident-runtime 复用已验证**：worker 加 `--steps N`，同一 prepared `rt` 跨 decode-step 批次复用，
  输出逐字节一致、rc=0 无状态污染。这是 `_pypto_full_forward` 常驻的核心机制，device 坐实。

## 两堵仍未解的真实墙（tasks 5-7 的核心攻坚点，别绕过）

1. **真 KV import**：offline 链用 dummy KV → 45 层链 L17 device NaN（可复现、输入无关）。full-chain /
   attention-core 正确性 **只能对着跑起来的 8001 vLLM 验证**（attn_metadata / paged-KV device buffer
   只在 serving 时存在）。可复用积木：phase 24 的零拷贝 KV-IPC 已在 per-op 路径 token-exact
   （`project_phase24_25_zero_copy_kv_handoff`）；attn_setup import_ipc 全 45 层 token-exact。
2. **co-tenancy（G4）✅ DISPATCH-RESOLVED（2026-07-11）**：原判「pypto whole-decode worker 与 vLLM
   同卡两个 chip_process owner = 未解架构 blocker」**已解**。真实症状不是 507018 而是
   `comm_hccl.cpp:301 HcclCommInitRootInfo failed: 7`（两个 HCCL communicator 同 8 卡不共存）。
   **根因 + 修法**：simpler 的 HCCL control comm 是 vestigial（只用 GetRootInfo/CommInitRootInfo/
   Barrier/Destroy，无 AllReduce/Send/Recv；唯一消费者 `comm_barrier` 在 dispatch 路径无调用者；
   数据面 + domain 建立已走 `file_barrier`+IPC peer-access）。**env-gated 修复 `SIMPLER_COMM_NO_HCCL=1`**：
   comm_init 跳过 HcclGetRootInfo/HcclCommInitRootInfo（保留 run_token 文件 + file_barrier），
   relax `hccl_comm==nullptr` 检查，no-op comm_barrier。默认(flag 未设)=原 HCCL 路径不变（安全）。
   patch 在 a2a3 `comm_hccl.cpp`（5 anchors，备份 `.bak_nohccl`），已重编 a2a3 runtime。
   **device 验证**：idle vLLM 8001 + whole-decode worker(`SIMPLER_COMM_NO_HCCL=1`) 同卡 8-15 →
   PREPARE OK、all steps dispatched、rc=0、无 HcclCommInitRootInfo failure、8001 health=200。
   → **task 5（co-tenancy）前提已成立**。剩：real-weight torch-ref 数值确认（合成跑出 L3 MoE nan，
   须排除 barrier-ordering race vs 合成数据 blowup）+ commit simpler patch + standalone regression。

## 可复用积木（别从零写）

- offline worker `_stage_whole_decode_run.py`（in-tree 未提交，本 session 收尾版）= task 5 的移植参考：
  build 7 programs + `with c0.prepare() as rt` + 45 层 dispatch loop + per-layer weight-stream +
  resident `--steps`。备份见文件尾「worker 备份」。
- per-op live 服务 `tools/step3p5/pypto_mlp_worker.py::_MlpService`（socket 服务，mlp/shared/routed/tail
  partial，复用 vLLM attention+KV）—— 现有 live 路径是**逐 op**，task 5 的 whole-net 是另一套架构。
- `tools/step3p5/pypto_weight_ipc.py::WeightIpcExporter`（47GiB 权重 IPC）。
- monkey-patch seam：`tools/step3p5/vllm_monkey_patch.py:233 _pypto_full_forward`（当前 fail-closed
  stub，mode=full；default 是 tail 不影响 serving）。

---

```
继续 pypto + vLLM 集成，集中攻克 tasks 5-7（G2-G5 live wiring），目标：live single-handoff A/B
—— 8001(pypto)跑完整 45 层 step3p5 decode，token-exact vs 8000(vanilla)。G1 offline（tasks 1-4）
上个 session 已 device 验证完成（见 pypto-project STATUS.md + archive/milestones 续²~续⁵），别重做。

## 运行环境（权威）
- 全部在 0162：`ssh 0162`，repo `/data/chensiyu/hw_project/pypto/workspace/pypto-lib`，
  分支 `stepfun/develop`。**不 rebase**，**不用 b-csy-develop / feat/whole-net-n1-fusion**。
- fresh shell 三件套：`source /usr/local/Ascend/cann/set_env.sh`（非-GA symlink，不是 beta.1）
  `&& source /data/chensiyu/hw_project/pypto/workspace/activate.sh
  && export PTO_ISA_ROOT=/data/chensiyu/hw_project/pypto/workspace/pto-isa`。
- device 跑必设：`PTO2_RING_HEAP=4294967296 PTO2_RING_TASK_WINDOW=131072 PTO2_RING_DEP_POOL=131072`。
- 8000 oracle 在 cards 0-7（别碰）；pypto 用 cards 8-15，worker `--tp 8 --dev-offset 8 -p a2a3`
  （必须 -p a2a3；默认 a2a3sim 要 g++-15 没装）。cards 8-15 Aicore=100% 是 sticky 假象，非 poison。
  launch 前 `pgrep -af "[c]hip_process"` 确认空 + `rm -f /dev/shm/torch_*`。
- **禁** `-9` 强杀 / `npu-smi set -t reset`（netboot 机重启锁死全卡）。
- 起 4-agent team（reverse-review/hw-analyst/sw-analyst/upstream-scout），但 lead 直接跑 device，
  盯紧增量别让 agent 空转。动手前读 skill `pypto-project/.claude/skills/pypto-dev-constraints/SKILL.md`。

## 硬约束
- 架构走 Option-C 多程序（TP-attn program + select_moe_block moe_block），不走 n1-fusion。
- debug 四板斧：DeepSeek/Qwen（models/deepseek/v4）→ 上游是否已修 → 我们 kernel → dtype；
  507018/507899 先查 simpler wiki。精度 oracle = vLLM eager dump（synthetic 会 stale），
  真 token-exact 靠 live A/B。
- 同步协议：进展改 STATUS.md/archive/milestones，commit + push
  （`git -c http.version=HTTP/1.1 push`，PAT 在 /data/chensiyu/secrets/github.env，屏蔽 token；
  注意 .git/objects/a8 root-owned 偶发挡 commit，retry 换 hash 可过，或让用户 sudo chown 修）。

## 按顺序攻克（tasks 5-7）
0. 先起 team + 读 skill + ssh 0162 + 拉最新 offline worker 备份核对现状（见「worker 备份」）。
1. ✅ **【前置 gate】co-tenancy（G4）已解（2026-07-11）**：`SIMPLER_COMM_NO_HCCL=1`（重编 a2a3 runtime，
   simpler commit 0162-local `878f3742`）→ whole-decode worker 与 idle vLLM 8001 同卡 rc=0、8001 200。
   完整 runbook + 根因：[`deployment/cotenancy-simpler-no-hccl.md`](deployment/cotenancy-simpler-no-hccl.md)。
   运行时机制全 device de-risk：NO_HCCL + resident rt 跨 step 复用（`--steps 2` co-resident rc=0）+
   real-weight L0 torch-ref 1.000。**下一 session 从第 2 步（G2 code）开始，别重测 co-tenancy。**
   → **具体实现蓝图见 [`phases/20-vllm-backend-monkey-patch.md`](phases/20-vllm-backend-monkey-patch.md)
   §G2/G3/G5 live single-handoff 实现蓝图**（whole-decode=单 DistributedWorker 非 per-op 8-ChipWorker；
   TP hidden 层间 replicated → 单 handoff rank-0 drive+broadcast；G3 每 rank 一条 KV-IPC；5 步实现顺序）。
2. ✅ **建常驻 whole-decode 服务（sidecar）— 部分完成 + device 验证（2026-07-11）**：`_stage_whole_decode_run.py`
   加 `--serve`/`--serve-sock` + `_WholeDecodeServer`（AF_UNIX），把 build+prepare 后的 resident rt 包成
   socket 服务：收 full hidden `[BATCH,HIDDEN]` bf16（经 resid1_host 喂 layer-0）→ 45 层 decode（复用
   prepared rt）→ 回 next hidden。**device 验证（cards 8-15, SIMPLER_COMM_NO_HCCL=1, 2 请求）**：
   PREPARE OK → serve request 0 → serve request 1（reusing prepared rt）→ client round-trip
   `[16,4096]→[16,4096]` ×2、rc=0、clean exit。smoke rc=0。备份 `workspace/g2_worker_sidecar_20260711_022638.py`
   （in-tree `M _stage_whole_decode_run.py`，未 push fork）。**剩**：`decode(kv_args, forward_context)`
   透传（= G3 真 KV，现 dummy-KV → nan 是预期 synthetic blowup，非 mechanism）；封装成独立常驻进程 holder。
3. 接真 KV（G3）：把 offline 的 dummy k_cache/v_cache/seq_lens/block_table/slot_mapping 换成 vLLM
   forward_context 的真值 —— 复用 phase 24 零拷贝 KV-IPC（`project_phase24_25_zero_copy_kv_handoff`
   + phase 23 doc + attn_setup import_ipc，per-op 已 token-exact）。forked chip 的 IPC import 必须
   在 child 进程 context 内（每 rank 一条：sidecar chip-r import vLLM rank-r KV）。sidecar 协议加
   kv_args/attn_args 随 hidden 一起收。
4. 接 `_pypto_full_forward`（`tools/step3p5/vllm_monkey_patch.py:233`，现 fail-closed）：
   lazily 建常驻服务 client（rank-0 drive + `tensor_model_parallel_broadcast` 给其余 7 rank）+ 45 层
   dispatch loop（常驻 DeviceTensor residual handoff）+ 读 live forward_context 进 attn args +
   final hidden copy 回。整模型 patch（不 per-layer），enforce_eager。
5. G5 live A/B：8001 起 mode=full（`PYPTO_STEP3P5_PATCH_MODE=full`），恢复 8001 顺序 = 先起
   8001 等 HCCL init 完再起 pypto sidecar（`SIMPLER_COMM_NO_HCCL=1`）。跑 3-prompt A/B vs 8000
   vanilla，要 token-exact。swiglu(L43/L44) 精度也在此定论（offline 合成不可信）。

## 关键契约（读码已证，别信旧 memory）
- HBM 非门槛：64GB/卡，TP=8 sharded vLLM+pypto ≈10GB/卡 fits（旧「24G+47G=OOM」是 aggregate 误判）。
- select_moe_block 返回的 EpTpMoE 不自带 norm/residual，只出 moe_out → worker 自补 post-RMSNorm
  (EPS=1e-5,Gemma +1.0) + next_hidden=resid1_fp32+moe_out。
- select_moe_block 按 silu 去重（L3-42 silu 共享一 program），但每层权重各异 → 每 moe step 前必须
  copy 该层权重进 shared slot（本 session task 4 已实现 `_load_moe_layer_weights`）。
- weight_loader：norm[45]按绝对 layer_idx，MoE[42]按 pos=layer-3，dense[3]，不混。
- gate_r 是真 per-head gate 乘子（sigmoid(RMSNorm(hidden)@w_g)），非 ×1 旁路。

先起 team + 读 skill + ssh 0162 核对 worker/8001 现状，从任务 1（co-tenancy gate）开始。
```

## worker 备份位置（0162，本 session 收尾版，别丢）
- 工作树：`/data/chensiyu/hw_project/pypto/workspace/pypto-lib/_stage_whole_decode_run.py`
  （含全部 host 修复 + per-layer weight-stream + per-layer isolation + last-token bootstrap + `--steps`）
- 持久备份（NFS `workspace/`）：`g1_worker_task4_final_20260710_*.py`、`g1_worker_resident_20260710_*.py`
- moe.py InCore→Inline 修复备份：`workspace/g1_moe_incore_inline_20260710_*.py`

## pypto-project pin
- 最新 `b615d46`（STATUS.md + archive/milestones 续²~续⁵ + memory HBM 修正）。
