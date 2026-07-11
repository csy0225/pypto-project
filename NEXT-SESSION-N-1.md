# NEXT SESSION — N=1 整网 live 集成 · 收尾 Task② real-weight IPC → 真 KV → live A/B

> 直接把下面 code block 当第一条消息粘贴。自包含。生成于 2026-07-11（续）。
> **本阶段核心已 device 验证：N=1 `whole_decode_faithful_real` 真实 W8A8 权重经借用的 `import_ipc` 零拷贝导入 device 打通**
> （8 exporter export 47.46GiB 池 + `import_ipc_all` 返回 8 peer_bases + compile OK + 42 args）。
> 只剩把 `rt.run` 的 arg-marshalling 契约走完 → finite logits，再接真 KV + socket → live A/B。

---

## ⭐ 路线（用户拍板，勿偏离）
- **继续 N=1 program `whole_decode_faithful_real`**（单 @pl.program 全 45 层）；**只借用** 0162 G-series 的 `import_ipc` feature，**不照搬其 Option-C 多程序架构**。
- **N=1 的关键优势**：`whole_decode_faithful_real` 已**编过全 45 层（含 swa-MoE）** → **绕开 G-series 卡多个 session 的 swa_moe const-fold 级联**（那条卡 30/45 层）。
- **N=1 唯一不 OOM 的权重路 = import_ipc**：单 program 一次喂 45 层权重，无法逐层 host stream；self-load 8× stack = 752GB OOM（exit137，已验证死路）。

## ✅ 已完成（device 验证，勿重做）
- **机器**：**0234 经 tmux `pypto-ascend-0:0`** 进（裸 `ssh 0234` DNS 不解析）。0234 = **8 卡 0-7 空闲、781GB RAM、真 W8A8 ckpt 挂载、与编辑机 b-csy-develop 共享 NFS（编辑即时可见）**。0234 是 N=1 standalone dev 机。0162 = live-A/B 机（co-tenancy runtime + vLLM 8000/8001 容器），但 0-7/8-15 都被 vLLM 占（8001 util0.5=43.8GB/卡）→ 47GB 池无处放 = **gap-5 HBM gate**。
- **import_ipc 运行时（纯 Python，无需重编 C++）已打进 NFS runtime**（`pypto/runtime/python/simpler/worker.py` `_CTRL_IMPORT_IPC=16` + `pypto/python/pypto/runtime/distributed_runner.py` `import_ipc_all`；备份 `.bak_preimportipc_20260711_221737`）。0234/b-csy 共享 NFS → 已生效。**重装 runtime 后要 re-apply。**
- **分支 pypto-lib `feat/whole-net-n1-fusion` HEAD `a1891c8`**（用这个）。新增：`import_weights_all`+`build_stacked_weight`（`pypto_weight_ipc.py`，`b2cf225`）、harness `tests/step3p5/_stage_whole_faithful_real_ipc.py`（`6d09e47`）、norm FP32 cast（`a1891c8`）。
- **N=1 real-weight import_ipc DEVICE-VALIDATED（0234 cards 0-7）**：8 exporter export 真实 47.46GiB W8A8 池（`pool_base=0x12c1c0000000`）+ `import_ipc_all` 返回 8 real-weight peer_bases + `whole_decode_faithful_real` compile OK + built 42 args。**两个 `rt.run` arg 契约错已修**：(1) host 张量必须 prepare() 前 `.share_memory_()`；(2) norm 权重 exporter 侧 cast FP32。
- co-tenancy `SIMPLER_COMM_NO_HCCL=1`（0162 runtime）已 device 验证（bypass HcclCommInitRootInfo:7）。

## 环境铁律
- 三件套（每 fresh shell / 每 tmux 命令）：`source /usr/local/Ascend/cann/set_env.sh && source WS/activate.sh && export PTO_ISA_ROOT=WS/pto-isa`（WS=/data/chensiyu/hw_project/pypto/workspace）。`activate.sh` 不设 PYTHONPATH → 另设 `export PYTHONPATH=WS/pypto/python:WS/pypto-lib`。
- 8 卡跑：`PTO2_RING_HEAP=4294967296 PTO2_RING_TASK_WINDOW=131072 PTO2_RING_DEP_POOL=131072`。
- **0234 device 跑用 tmux**：`tmux send-keys -t pypto-ascend-0:0 '<cmd> > NFS_log 2>&1 &' Enter`，输出 tee 到 NFS log（`WS/logs_n1/`）从 b-csy-develop 直接 Read。b-csy-develop **无 python**（编辑用，device 用 0234）。
- launch 前 `pgrep -af '[c]hip_process'` 空 + `rm -f /dev/shm/torch_*`；**禁 `-9` 强杀 device 进程 / `npu-smi set -t reset`**（netboot 机锁死全卡）。
- push：编辑在 b-csy-develop NFS，PAT `/data/chensiyu/secrets/github.env` + `-c http.version=HTTP/1.1`，屏蔽 token。跨仓 push 同步 pypto-project STATUS/phase27（同 session）。
- 动手前读 skill `pypto-project/.claude/skills/pypto-dev-constraints/SKILL.md` + memory `n1_live_integration_consolidated_on_0162`。

---

```
继续 step3p5 N=1 整网 live 集成，收尾 Task②→③。全部读上面 ⭐/✅/铁律。分支 pypto-lib
feat/whole-net-n1-fusion（a1891c8），N=1 program whole_decode_faithful_real，只借 import_ipc。
0234 经 tmux pypto-ascend-0:0（8 卡 0-7 空闲）做 standalone；0162 做 co-resident live（gap-5 gate）。

## 攻坚顺序（每步 device 可验证）
1. **run #3 —— real-weight IPC harness 出 finite logits**（norm FP32 fix 已入 a1891c8）：
   0234 tmux: cd WS/pypto-lib && 三件套 env + PYTHONPATH + PTO2_RING +
   pgrep [c]hip_process 空 + rm /dev/shm/torch_* →
   nohup python -m tests.step3p5._stage_whole_faithful_real_ipc -p a2a3 -d 0,1,2,3,4,5,6,7 \
     --out /tmp/n1_weight_ipc > WS/logs_n1/0234_ipc_run3.log 2>&1 &
   期望 RESULT=REAL_WEIGHT_IPC_RUN_CLEAN（dummy KV → logits 非真值，但证 N=1 real-weight import_ipc
   全链 dispatch 通 = Task② device 落地）。若再撞 arg dtype/shape，逐个修（对照 host_orch 签名：norm/
   final_norm/moe_gate_w/router_bias=FP32，其余=BF16；gate_r=zeros；final_norm per-rank [1,HIDDEN]）。
   注意：8 exporter 并发 load 峰值 ~110GB/rank（dequant），781GB 够；若换 RAM 小的机器要串行化。
2. **接真 KV（G3，同 import_ipc_all）**：mirror pypto_kv_ipc.KvIpcMap/build_stacked_kv；harness/worker
   加 --kv-ipc-dir；KV key/map 从 8001 容器 /logs 导（`nerdctl exec vllm-8001 cat pypto_kvpool.key.rankR`）。
   ⚠ 真 KV 需 8001 活着（KV buffer 在 serving 时才存在）→ 回到 co-resident（gap-5 HBM）。standalone dummy
   KV 只证 dispatch 非数值。
3. **socket serve + live A/B**：复用 G5a backend（`/data/chensiyu/logs/step3p5_910b_w8a8_v001/pypto_patch/
   pypto_whole_decode_backend.py` + `WS/g5_start_8001_full_*.sh` + start_sidecar），把 whole_decode_faithful_real
   换进 sidecar（socket 收 [BATCH,HIDDEN] hidden → 45 层 → 回 next hidden）；8001(pypto mode=full,
   PYPTO_WHOLE_DECODE=1) vs 8000(vanilla) 3-prompt token-exact。
4. **gap-5 HBM（co-resident live 硬 gate）**：47GB BF16 池 + vLLM(24-47GB) > 64GB。选 (a) in-kernel W8A8
   dequant（whole_decode 矩阵吃 INT8 + 片上 dequant，保 ~24GB 共享 vLLM 权重，见 memory
   project_task6_live_wiring_plan gap-5）；或 (b) standalone-only 数值验证（对 decode-step golden）。

## 可复用积木（文件位置）
- N=1 program：`models/step3p5/decode_layer.py::whole_decode_faithful_real`（生成器 tools/step3p5/_gen_faithful_real.py）。
- weight IPC：`tools/step3p5/pypto_weight_ipc.py`（WeightIpcExporter.export_from_checkpoint / import_weights_all /
  build_stacked_weight / WeightIpcMap）。harness `tests/step3p5/_stage_whole_faithful_real_ipc.py`。
- KV IPC 模板：`tools/step3p5/pypto_kv_ipc.py`（KvIpcMap/build_stacked_kv）+ G-series worker
  `_stage_whole_decode_run.py --worker --kv-ipc-dir`（0162 stepfun/develop 工作树，参考不照搬）。
- import_ipc 运行时：NFS `pypto/runtime/python/simpler/worker.py`(_CTRL_IMPORT_IPC=16) + `distributed_runner.py`(import_ipc_all)。
- live backend：0162 容器 `pypto_whole_decode_backend.py` + start 脚本 `WS/g5_*_20260711_030800.*`。
- host_orch arg 顺序：见 `tests/step3p5/_stage_whole_faithful_real_device.py:104-157`（dummy）与 `_ipc.py`（IPC）。

可起 team（reverse-review/hw-analyst/sw-analyst/upstream-scout），但 lead 直接跑 device，盯增量别让 agent 空转。
每完成关键节点更新 pypto-project phase27 + STATUS + push（代码 fork + pypto-project 同 session）。
```
