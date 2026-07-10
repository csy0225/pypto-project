# NEXT SESSION — N=1 整网融合 · live vLLM-IPC 集成 kickoff prompt

> 直接把下面代码块粘贴为 `/goal` 执行。自包含。

```
继续 step3p5 N=1 整网融合。上阶段(2026-07-11)已完成 Task#2 增量②:真实 per-layer 权重结构 + full+swa
完整路由的新 program `whole_decode_faithful_real` 编译 + 8 卡 device dispatch 干净。本阶段集中攻克
**live vLLM-IPC 集成**——这是解锁「真实权重执行 + 逐层对齐 vLLM + live A/B」的唯一路径,核心 blocker 是
HCCL 同卡共存。

## 起点(已完成,勿重做)
分支 pypto-lib `feat/whole-net-n1-fusion`(csy0225 fork)。**用这个分支。**
- HEAD `d9b7dc6`。关键 commit:`be10ba0`(增量② real-per-layer builder) + `1a68c47`(Task#1 per-layer
  comm 窗口)+ `7b9693b`(增量① dense-prefix 索引)。pypto-project main `5c7e811`(phase27 + 本 doc)。
- ✅ `whole_decode_faithful_real`(`WholeDecodeFaithfulReal`,`decode_layer.py` 末尾):真实 per-layer 权重,
  norm[45] kernel-index 绝对 L;attn/dense/MoE 全 host-slice 单层;11 个 full-MoE 层走 `full_attn_only_orch`,
  31 个 swa 走 `swa_attn_only_orch`;`chip_orch` 用 `norm_layer_idx`。生成器 `tools/step3p5/_gen_faithful_real.py`
  (如需再生:改生成器不要手改 42 个展开块;`.bak.pregen_real` 是基线备份)。
- ✅ 验证:smoke rc=0、compile rc=0、8 卡 DISPATCH_CLEAN 285s(dummy,88 pass-block,无 507018/stall)。
  harness `tests/step3p5/_stage_whole_faithful_real_device.py`(`--compile-only` / `-d 0..7`)。
- ⚠️ 探针结论(已固化 memory `n1_real_per_layer_builder_device_clean`):`pl.inline` 接受实参 leading dim
  比标注小 → host-slice 单层可行。
- ⛔ **作废**:`tests/step3p5/_stage_whole_faithful_real_weights.py`(自 load checkpoint,架构错)。真实权重
  **走 vLLM-IPC,pypto 不自读 checkpoint**——见本阶段目标。

## 环境铁律(0234,保持不变)
- CANN 保持 beta.1:`/usr/local/Ascend/cann-9.0.0-beta.1`(新装 cann-9.0.0 HCCL ABI 坏,勿用)。
- simpler 必须 SDMA-OFF:`SIMPLER_ENABLE_PTO_SDMA_WORKSPACE=OFF`。重编启 device 前先
  `allreduce_distributed -p a2a3 -d 0-7 --mode twophase` 复验 = `max|out-expected|=0`。
- 换 pod/CANN 后 clean 重编:`build_runtimes --platforms a2a3`(`--clone-protocol` 已废弃)。
- 三件套(每 fresh shell):`source /usr/local/Ascend/cann/set_env.sh && source $WS/activate.sh &&
  export PTO_ISA_ROOT=$WS/pto-isa`($WS=/data/chensiyu/hw_project/pypto/workspace)。
- 开发/编译/device 都在 0234(tmux `pypto-ascend-0`,window 0 = ssh root@0234;新开 window 是本地
  b-csy-develop,别在那跑 device)。8 卡跑设 `PTO2_RING_HEAP=4294967296 PTO2_RING_TASK_WINDOW=131072
  PTO2_RING_DEP_POOL=131072`。
- git push 用 PAT `/data/chensiyu/secrets/github.env` + `-c http.version=HTTP/1.1`,输出屏蔽 token。
  pypto-project 的 `.git/objects` 有 0234 root 建的对象 → doc commit 必须在 window 0 以 root 提
  (先 `git config --global --add safe.directory <pypto-project>`)。
- launch device 前 `pgrep -af '[p]ython -m tests'` 确认无残留 + `npu-smi info` HBM 空(AICore % 是 idle 噪声)。

## 本阶段目标(按依赖顺序;全部 gate 在第 1 步)

### ① 【核心 blocker】HCCL 同卡共存 —— 让 whole-decode worker 与 vLLM 共存于同 8 卡
device-proven(memory `hccl_single_world_cotency_blocks_forked_worker` / `project_g4_cotenancy_hccl_conflict`):
forked whole-decode worker 第一次 `allocate_domain → _ensure_comm_base` 挂 `comm_hccl.cpp:301
HcclCommInitRootInfo failed:7`——一卡只能一个 HCCL world,vLLM 已占。standalone 单跑没问题。
- **关键洞察**:N=1 program 的 collectives(`tp_all_reduce`/`ep_all_to_all`)本身走 **shmem-IPC**
  (`pld.system.notify/wait` + `pld.tile.remote_load`),**不是 HCCL**。HCCL 只来自 simpler 运行时的
  **domain bootstrap**(rootinfo 握手,`worker.py:_ensure_comm_base` → `comm_hccl.cpp:299-301`)。
- **要攻的问题**:能否让 domain 分配的 rootinfo 握手**不走 HCCL**(改用 IPC/文件/socket 交换 handle),
  或**复用 vLLM 已建的 comm**。两条上游路线:
  - (a) IPC-only control comm:simpler 运行时把 `HcclCommInitRootInfo` 换成 IPC-based rootinfo 交换
    (kernel 侧已全 IPC,只差 bootstrap)。无先例、新代码,但可能最小改动。
  - (b) in-process 复用 vLLM comm:pypto worker 与 vLLM 同进程,复用其 HCCL comm(Phase 22 opt A,
    "simpler runtime 较大改造")。
  - **先做**:精读 `simpler/.../worker.py::_ensure_comm_base`(~L3573-3631)+ `comm_hccl.cpp:290-310`,
    判断 (a) 是否可行(rootinfo 是否只在 init 握手、后续全 IPC);写最小 PoC:两进程(mock vLLM 占 HCCL
    + whole-decode worker)在同卡 allocate_domain 成功。gate 在此 PoC。

### ② vLLM-IPC 权重 handoff(47GiB 单 key)—— vLLM load,pypto IPC import
真实权重**不由 pypto 读 checkpoint**:vLLM 进程 load 权重进 HBM(per-rank TP/EP 已切),pypto whole-decode
program 零拷贝 IPC import(镜像零拷贝 KV handoff,memory `project_task6_live_wiring_plan` /
`zero-copy-ipc-integration-route`)。45 层权重合一 buffer → 1 key → 每 rank VA-map → `DeviceTensor(base+offset)`。
- forked chip 的 IPC import 必须在 child 进程 context 内(父 import 的 ptr 在 child 读 0)。
- `whole_decode_faithful_real` 的 host_orch 权重入参改成 resident DeviceTensor(IPC-imported),而非 host 张量。

### ③ wire `_pypto_full_forward` + live A/B(= Task#4 逐层对齐 的真正落点)
- `tools/step3p5/vllm_monkey_patch.py:233` 的 `_pypto_full_forward` 仍是 D3 fail-closed stub。整模型
  monkey-patch(非 per-layer)+ `enforce_eager`(pypto kernel 与 aclgraph 互斥)。
- **live A/B**:8001(pypto 整网 `whole_decode_faithful_real`)vs 8000(vanilla)token-exact。这 **就是** Task#4
  的逐层/端到端数值对齐——offline 逐层对不了(vLLM dump 是 18-tok prefill,decode kernel BATCH=16 单 token,
  shape 不兼容;需 live 或另生成 decode-step golden)。
- 逐层 debug 口径:`per_layer=True` 出 per-layer hidden 做 L1 ratio_allclose(atol=0.04);final logits argmax 一致。

## 先读(对齐状态)
memory:`n1_real_per_layer_builder_device_clean`、`hccl_single_world_cotency_blocks_forked_worker`、
`project_g4_cotenancy_hccl_conflict`、`project_hbm_cotenancy_not_gating_64gb`、`project_task6_live_wiring_plan`、
`vllm_golden_dumps_are_prefill_not_decode`、`n1_multicard_ipc_fix_sdma_workspace_off`;
`pypto-project/phases/27-n1-whole-net-fusion.md`(§Step-2 增量②)、`phases/22-*`(in-process/zero-copy IPC);
严格遵守 `pypto-project/.claude/skills/pypto-dev-constraints/SKILL.md`。

## debug/开发约束
遇问题先看 DeepSeek/Qwen + 历史文档避免重复造轮子;多看地址对齐/padding/shape/dtype/layout。HCCL 共存是上游
runtime 问题,改 simpler 前先写最小 PoC 定位 rootinfo 握手是否可 IPC 化,别盲改。每完成关键节点更新 phase27 + push
(两个 push 同 session 做:代码到 fork + pypto-project pin/doc)。
```
