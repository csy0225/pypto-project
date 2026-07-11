# 下个 session 集中攻关 —— G5b 最后一关：pypto+vLLM 端到端 token-exact（数值对齐）

> 新 session 直接把下面 code block 当第一条消息粘贴。自包含。生成于 2026-07-12（续⁵）。
> **上个 session 攻破 G5b 全部结构性 blocker**：const-fold 证伪、socket 真 metadata 协议、import_ipc 真 KV 零拷贝、
> **co-tenancy crash 彻底解决（file-based broadcast）** → 整条 live 45 层 single-handoff **HTTP 200 稳定跑通**。
> **唯一剩余 = 数值正确性**（生成 token 错误）。见 STATUS.md 顶部 2026-07-12(续⁵) + 下方历史。

```
继续 pypto+vLLM 集成，攻最后一关：让 8001(pypto mode=full) 对 8000(vanilla) **token-exact**（整网端到端
精度对齐）。全部在 0162（ssh 0162），repo /data/chensiyu/hw_project/pypto/workspace/pypto-lib 分支
stepfun/develop。动手前读 skill pypto-project/.claude/skills/pypto-dev-constraints/SKILL.md +
STATUS.md 顶部 2026-07-12(续⁵) + memory g5b_import_ipc_facade_missing / g5b_kv_bridge_not_pure_reshape /
vllm_golden_dumps_are_prefill_not_decode。

## ✅ 别重做（上个 session 全部 device 验证完成）
- **攻坚 1 const-fold**：证伪，非 blocker。canonical TP=8 全 COMPILE OK（复现器 `_probe_alllayers_compile.py`）。
- **攻坚 2 socket 真 metadata 协议**：self-describing length-prefixed 协议（`_wd_pack_fields`/`_wd_unpack_fields`）
  实装三处（sidecar `_stage_whole_decode_run.py` `_WholeDecodeServer.recv_step`+`_feed_meta`、in-tree
  `tools/step3p5/vllm_monkey_patch.py`、容器后端 `/logs/pypto_patch/pypto_whole_decode_backend.py` 已部署）。
  随 hidden 发 seq_lens/block_table(→BATCH×32 flat)/slot_mapping + 首请求静态 rope(full[4096,64]+swa[4096,128])。
- **import_ipc（真 KV 零拷贝）**：纯 Python `_CTRL_IMPORT_IPC=16`，已 device 验证 import 8 真 KV 池。
- **⭐ co-tenancy crash 彻底解决（file-based broadcast）**：一攻 `HcclBroadcast err9` + 二攻 `507018` 同源 =
  co-tenancy device 争用（rank-0 跑 sidecar 时 vLLM rank1-7 的 HcclBroadcast kernel 同卡自旋）。修法 = 容器后端
  `_pypto_full_forward` 把 `tp_group.broadcast` 换成 file-based broadcast（rank-0 写 /logs，rank1-7 CPU-poll）。
  **已部署 + device 验证：全 45 层 sidecar → prompt HTTP 200 完成、无 crash/507018/err9**。offline `--steps 4`
  证 rt-reuse 无 507018（纯 co-tenancy）。→ **整条 live 45 层 single-handoff 基础设施稳定跑通，别再碰 crash。**

## ⭐ 唯一剩余 = 数值正确性（本 session 主攻）
现象：8001(pypto) 生成 token 错误（text=""，finite 但不对）；8000(vanilla) 同 prompt "北京是" 出连贯
"中国的首都，也是世界上人口"。**注意**：上个 session 的 `_client_wd_sweep.py` 只用**合成随机 rope** 证「active
行无 nan」，**不是正确性**。真 rope/KV/权重下数值错。

**⚠ 关键量化口径（token-exact 前必读）**：当前集成用的是 **W8A8 checkpoint 但走 BF16-dequant 路径** ——
sidecar `_load_real_weights` 把 W8A8 权重在 loader 里**反量化成 BF16** 再喂 kernel（`select_moe_block` 无
`w8a8_native` 参数、torch-ref `routed_w8a8_dynamic=False`）。而 vLLM 8000/8001 是 **INT8-native W8A8** 计算。
→ 两侧存在固有 INT8↔BF16 数值差（历史 dump 对比 ~0.9995，很接近但非 bit-identical），**可能影响个别 token 的
greedy 选择 → 未必能真 token-exact**。策略二选一：(A) 接受 BF16-dequant，判据放宽到 top-1 ≥95% / cos≥0.999
（Phase 21 口径），不强求逐 token 全等；(B) 打通 gap-5 的 **INT8-native in-kernel** 路径（memory
`gap5_int8_cube_fractal_32_partial_tile` 等，尚未 device-work）做真 bit-level 对齐。建议先按 (A) 收尾拿到
「数值基本对齐 + 连贯生成」，(B) 作为后续精度攻坚。

## 攻坚顺序（每步 device 可验证）
1. **拿 decode-step golden**：现有 vLLM dump 是 18-tok prefill（memory `vllm_golden_dumps_are_prefill_not_decode`），
   BATCH=16 decode kernel 吃不下。需在 8000 或独立 vLLM eager 跑一个 decode-step dump（1 tok，batch≤16，每层
   hidden/attn_out/moe_out），或直接 live 单层对拍。
2. **单层 rope 核对（最可疑）**：`_wd_rope_from_emb`（容器后端 + in-tree，镜像 per-op `pypto_attn_backend
   ._rope_tables_from_attn`，per-op path token-exact）—— 但 whole-decode 是**不同 kernel**（`attention_full/swa`），
   核对它消费 rope 的 layout/convention 是否与 per-op 一致（cos=cat[cos_half,cos_half] tile、rotary_dim full=64/
   swa=128、pos 索引）。单层 probe：sidecar --layers 0 + 真 rope（从 vLLM cos_sin_cache 抽）+ 真 KV + 真 metadata，
   对 vLLM L0 attn 输出 active 行对拍。
3. **真 KV 读核对**：consolidated pool per-layer offset（`build_stacked_kv` 用 map 的 L{i}.K/V offset）+ block_table
   物理块 id → k_cache[pbid*128] 是否对齐 vLLM 该层 KV 布局（obstacle 3）。
4. **per-layer 权重流核对**：`_load_moe_layer_weights` 每 step copy 是否喂对该层 W8A8（norm[45]abs / MoE[42]pos=li-3
   / dense[3]，router_bias BF16-round，EPS=1e-5，swiglu_limit L43=7/L44=16）。
5. **修好后 live A/B**：3-prompt 对 8000 token-exact。swiglu(L43/L44) 精度在此定论。
6. **perf（后置，非门槛）**：现 ~120s/token（每步 MoE 权重 copy）→ 常驻权重（G2 weight-IPC 重叠）。

## live A/B runbook（基础设施已就绪）
(a) 停 8001：`sudo nerdctl --namespace k8s.io exec vllm-8001 bash -lc "pkill -9 -f '[V]LLM::EngineCore'"`；
    **清 card8-15 zombie**：`npu-smi info|grep VLLMWorker_TP|sed 's/|/ /g'|awk '$1>=8&&$1<=15{print $3}'|xargs -r sudo kill -9`
    （**别碰 cards 0-7 = 8000 oracle**）；`rm -f /mnt/nvme1/chensiyu/logs/step3p5_910b_w8a8_v001/{pypto_whole_decode.sock,pypto_wd_bcast.*}`。
(b) 起 8001：`sudo nerdctl --namespace k8s.io exec -d vllm-8001 bash -lc 'bash /logs/start_8001_full.sh > /logs/vllm_8001_x.log 2>&1'`
    （已含 HCCL_*_TIMEOUT=3600 + PYPTO_WHOLE_DECODE=1 + PYPTO_KVPOOL=1 + file-bcast 后端）→ 等 health=200 + KV keys 落 /logs。
(c) 起真权重全 45 层 sidecar（host）：`SIMPLER_COMM_NO_HCCL=1 WD_RING_HEAP=1073741824 PTO2_RING_TASK_WINDOW=131072
    PTO2_RING_DEP_POOL=131072 python _stage_whole_decode_run.py --worker --serve --serve-sock
    /mnt/nvme1/chensiyu/logs/step3p5_910b_w8a8_v001/pypto_whole_decode.sock --tp 8 --dev-offset 8 -p a2a3
    --ckpt /mnt/nvme1/chensiyu/step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp
    --kv-ipc-dir /mnt/nvme1/chensiyu/logs/step3p5_910b_w8a8_v001 --layers 0,1,...,44`（nohup 后台 + 监控 log 到
    `serve: listening`）。容器 /logs = host /mnt/nvme1/chensiyu/logs/step3p5_910b_w8a8_v001。
(d) A/B：curl 8001 vs 8000 同 prompt greedy(temperature=0) 比 text（pypto ~120s/token，用小 max_tokens）。

## 铁律 / hazard
- 三件套：`source /usr/local/Ascend/cann/set_env.sh && source WS/activate.sh && export PTO_ISA_ROOT=WS/pto-isa`。
- **停 sidecar 用 SIGTERM**（非 -9）；但其 teardown 507018 → force_reset(8) **会 nuke 同卡 co-resident 8001**
  → 停 sidecar 后必须 restart 8001 + 清 card8-15 zombie VLLMWorker。禁 `npu-smi set -t reset`。
- 8000 oracle 在 cards 0-7 别碰；launch 前 `pgrep [c]hip_process` 空 + card8-15 无 zombie。
- oracle = vLLM eager dump / live 8000（synthetic 会 stale）。debug 四板斧：DeepSeek → 上游 → 自写 kernel → dtype。
- push：pypto-lib 在 0162 fork SSH 无 key、PAT 在本地 box → 沿用 NFS 备份 `workspace/g5b_*`（或让用户 push）；
  pypto-project 文档在本地 box（PAT `/data/chensiyu/secrets/github.env`，HTTP/1.1）。

## 可复用积木（0162）
sidecar `_stage_whole_decode_run.py`；容器后端 `pypto_whole_decode_backend.py`（file-bcast 版已部署，备份 .bak-g5b
+ NFS `workspace/g5b_container_backend_filebcast_*`）；协议/提取/隔离复现器 `_test_wd_protocol.py`/
`_test_container_backend.py`/`_client_wd_sweep.py`/`_client_wd_row0.py`；start_8001_full.sh（HCCL timeout + file-bcast）。

先读 skill/STATUS/memory + ssh 0162 核对（8001/8000 健康、cards 干净），从「攻坚顺序 1（decode-step golden）」
或直接单层 rope 对拍开始。可起 team 但 lead 直接跑 device，盯增量别让 agent 空转。
```

---

# （以下为历史参考，上个 session 已完成的结构性工作）

---

## ⭐⭐ 通用可复用特性：device-IPC 零拷贝导入 `import_ipc`（KV + weight 都用；凡需把 vLLM 已驻显存零拷贝喂进 pypto worker 的 session 都要它）

**背景**：pypto runtime C++ `Orchestrator` **没有** `import_ipc` facade（`DistributedWorker.import_ipc`
是孤儿 wrapper，调会 `AttributeError: 'Orchestrator' object has no attribute 'import_ipc'`；worker_bind.h 只 bind
malloc/copy_to/remote_*；current + pre-upgrade tag 都没有）。所以 `pypto_kv_ipc.py`/`pypto_weight_ipc.py` 里
`rt.import_ipc(...)` 的写法**在当前栈上不能用**（aspirational 未验证）。

**本 session 解法（纯 Python，无需重编 C++，已 device 验证）**：chip child 经 **Python loop**
（`pypto/runtime/python/simpler/worker.py::_run_chip_main_loop`）消费 control op，`broadcast_control_all` 对
Python 可见 → 新增 control-op **`_CTRL_IMPORT_IPC = 16`**：
- `simpler/worker.py`：child `_handle_ctrl_import_ipc(buf, device_id)`（读 staged payload
  `<H rlen> reply_name <I count> count*(<I dev, 256B key)`，按 device_id 取本 rank key，ctypes
  `aclrtIpcMemImportByKey(&va, key, 0x1=ENABLE_PEER_ACCESS)` —— 导出侧用 `0x1=DISABLE_PID_VALIDATION` 故**无需
  SetImportPid**，VA 经 **reply-shm** 回传因 `ControlResult` 无 value 字段）+ dispatch elif + host
  `Worker.import_ipc_all(device_key_map:{dev:256B key})->{dev:va}`（建 reply-shm+payload→broadcast→读 reply-shm）。
- `pypto/python/pypto/runtime/distributed_runner.py`：`DistributedWorker.import_ipc_all` 委托 `self._w`(simpler Worker)。
- **用法**：`vas = dworker.import_ipc_all({dev_offset+r: key_bytes_r})` → 每 rank peer VA → `DeviceTensor(peer_base=va,...)`
  零拷贝喂 rt.run。**weight IPC 同理**（`pypto_weight_ipc.py` 的 47GiB 权重驻留把 `rt.import_ipc`→`import_ipc_all` 即可）。
- **device 实证**：dense L0 import vLLM KV（`peer_bases=[0x12c1c0000000 ×8]`，与 phase-16 probe2 VA 一致）+ attention
  读真 KV → `next_hidden 30.875` 非 nan、rc=0、co-resident live 8001。
- ⚠ **runtime 改动未提交**（0162 工作树）：`simpler/worker.py` + `distributed_runner.py`；备份
  `workspace/g5b_simpler_worker_import_ipc_20260711_144753.py` / `g5b_distributed_runner_import_ipc_*` /
  `g5b_worker_batchimport_*`。**重装/重编 runtime 后要 re-apply**；建议 commit 进 csy0225/**pypto**(非 pypto-lib) stepfun/develop。

---

```
继续 pypto+vLLM 集成，收尾 G5b：让 8001(pypto mode=full) 对 8000(vanilla) token-exact。全部在 0162
（ssh 0162），repo /data/chensiyu/hw_project/pypto/workspace/pypto-lib 分支 stepfun/develop。动手前读 skill
pypto-project/.claude/skills/pypto-dev-constraints/SKILL.md + memory g5b_import_ipc_facade_missing +
g5b_kv_bridge_not_pure_reshape + 本文件顶部 ⭐⭐「可复用特性 import_ipc」。

## ✅ 别重做（本 session device 验证）
- **import_ipc 已解**（纯 Python `_CTRL_IMPORT_IPC=16`，无需重编 C++）—— 详见顶部 ⭐⭐ 段。零拷贝导入 vLLM 显存
  （KV 或 weight）用 `dworker.import_ipc_all({dev:key})->{dev:va}`。改动未提交，重编 runtime 后要 re-apply。
- **KV bridge 不是纯 reshape**（纠正旧 memory）：MAX_SEQ_DEFAULT=nb×bs 撑爆 flash-attn scratch(667GB)。正确 =
  `KV_CACHE_ROWS_DYN=651904/652032`（=per-K nbytes/2/128，随 boot 变，worker 从 map 自动算）+ 逐层 feed
  `layer_cache_base=0`（attention_full.py:220 / attention_swa.py:228 已改 0）。dense L0 已 device 验证。
- **co-tenancy HBM**：8001 util 0.5 占 47GB/卡（util 0.3 起不来 "No available memory"）；sidecar 必设
  `WD_RING_HEAP=1073741824`(1GB) 才不 OOM（默认 4GB ring 撞 2.57GB static arena 分配失败）。

## 攻坚顺序（每步 device 可验证）
1. ✅ **DONE 2026-07-11(续²) — swa_moe const-fold 证伪**（不是 blocker）：当前工作树 canonical TP=8
   编译 clean（attn_full/attn_swa/full_dense/swa_dense/moe_block(swa L3/full L4/L43/L44) 全 COMPILE OK，
   含 `--kv-ipc-dir` override）。原复现是 `--smoke` 默认 `--tp 1` 走 `apply_tp1_patch`（unslice 违反铁律）
   撞 `moe.py:208` parity assert，非 const-fold。复现器 `_probe_alllayers_compile.py`；`--smoke --worker --tp 8` clean。
2. ✅ **DONE 2026-07-11(续²) — socket 带真 metadata**：sidecar 换 **self-describing length-prefixed 协议**
   （`_wd_pack_fields`/`_wd_unpack_fields`；`<I hlen>`+JSON+blobs）。三处同步：sidecar
   `_stage_whole_decode_run.py`（`_WholeDecodeServer.recv_step` + decode-loop `_feed_meta` 每 step copy
   seq_lens/block_table/slot_mapping 进各 attn sh + rope 按 full/swa 分流，首请求发静态 rope）、in-tree
   `tools/step3p5/vllm_monkey_patch.py`（`_WholeDecodeClient.decode(hidden, meta_fields)` +
   `_pypto_full_forward` 提取 forward_context + prefill→collective fallback）、容器后端
   `/logs/pypto_patch/pypto_whole_decode_backend.py`（自包含，**已部署**，备份 .bak-g5b）。提取镜像 per-op
   `pypto_attn_backend`。**验证**：offline round-trip + 容器后端 E2E PASS；**device**：sidecar co-resident
   live 8001（NO_HCCL, WD_RING_HEAP=1GB）import 8 真 KV 池 + 新协议喂 metadata → L0 full-attn active rows
   non-nan(27.6)。NFS 备份 `workspace/g5b_*_20260711_231307`。复现器 `_test_wd_protocol.py` /
   `_test_container_backend.py` / `_client_wd_metadata.py`。
3. **单层 paged-index 数值对拍**（下一步）：对 vLLM eager decode dump（须 decode-step golden，非 18-tok
   prefill dump）验证 worker `[num_slots,128]` flat + block_table/slot_mapping 索引 == vLLM
   `[nb,bs,1,128]` flatten `block_id*128+slot`。需真 metadata（step 2 协议已通）+ 真 rope（用 vLLM
   `cos_sin_cache`，容器后端已提取）。plumbing 已 device 证（step 2 non-nan）；此步定 attention 数值。
4. **live A/B**（终局）：exact runbook —
   (a) 停 8001：`sudo nerdctl --namespace k8s.io exec vllm-8001 bash -lc "pkill -9 -f '[V]LLM::EngineCore'"`
       （容器内 pkill，不碰 8000）；`rm -f /tmp/pypto_whole_decode.sock`；`pgrep -af [c]hip_process` 空。
   (b) 起 8001 mode=full+KVPOOL：`bash /logs/start_8001_full.sh`（须含 `PYPTO_WHOLE_DECODE=1` + `PYPTO_KVPOOL=1`
       + `PYPTO_WHOLE_DECODE_SOCK=/logs/...sock`）→ 等 health=200（collective fallback 让 profiling 存活）。
   (c) 拷 fresh KV key/map 到 host dir：`nerdctl exec vllm-8001 cat /logs/pypto_kvpool.{key,map}.rankR`
       → host `/tmp/g5b_kvtest2/`（fresh boot=新 VA，旧 key 失效）。
   (d) 起真权重全 45 层 sidecar：`SIMPLER_COMM_NO_HCCL=1 WD_RING_HEAP=1073741824 PTO2_RING_TASK_WINDOW=131072
       PTO2_RING_DEP_POOL=131072 python _stage_whole_decode_run.py --worker --serve --serve-sock <同(b)sock>
       --tp 8 --dev-offset 8 -p a2a3 --ckpt <W8A8> --kv-ipc-dir /tmp/g5b_kvtest2 --layers 0,1,...,44`
       （sock 必须在容器可见路径，用 /logs；SIGTERM 停，非 -9）。
   (e) 送 3-prompt 对 8000 token-exact。swiglu(L43/L44) 精度在此定论。
   ⚠ 真权重全 45 层 sidecar 首跑可能撞 MoE/swa runtime（507018）—— 若卡，先 `--layers 0`（dense）跑通再
   逐段加层 bisect。

**2026-07-11(续³) live A/B 首攻结果（重要，先读）**：runbook (a)-(d) 全跑通 —— 8001 restart mode=full
（新后端）+ 真权重全 45 层 sidecar：**7 programs/45 层/87 steps + PREPARE OK + import 8 真 KV 池 + serve
listening，无 OOM、无 prepare/dispatch 507018**；送 prompt → decode step 进 sidecar → **87 steps 全 dispatched
跑完**（整条 live single-handoff 机制全通）。**但两个精确遗留 gate token-exact**：
- **遗留 A（攻坚 3，数值）**：sidecar torch-ref 从 L0 nan，但 max|abs| 含 padded 行（decode 1 active seq，
  rows1-15 ctx=0→softmax 0/0=nan，与 isolated 测试一致）→ **active 行正确性未确认**。下步：只取 active 行
  （row 0）单层对拍 vLLM decode dump，查 block_table→consolidated-pool-offset 的 KV 读是否正确（早先 isolated
  synthetic-metadata + 14:12 pool 的 active 行 non-nan 27.6，证 KV-read 机制本身通；此步查 live 真 metadata）。
- **遗留 B（攻坚 4，co-tenancy 稳定性）**：第 2 请求 8001 EngineCore 崩在 vLLM `c10d ProcessGroupHCCL::
  broadcast`（`_pypto_full_forward` 的 `tp_group.broadcast(next_hidden,src=0)`）—— co-resident pypto sidecar
  device stream 与 vLLM HCCL broadcast 同卡时序冲突（sidecar teardown 507018 → force_reset card 8）。下步：
  sidecar 每 step 后 device 完全 sync/idle 再返回（让 vLLM broadcast 时 pypto 卡 idle），或改 handoff 时序。
  停机：sidecar SIGTERM（clean，force_reset 自清卡）；8001 crash 后 restart 回 vanilla-fallback serving。

**2026-07-12(续⁴) live A/B 二攻结果**：遗留 A（数值）**基本排除** —— 真权重+真 KV+合成 metadata 单层
sweep（ctx=10/300/4090、block=0/5000/10、多 block）active 行(row0) **全 FINITE**（`_client_wd_sweep.py`）；
「Lx nan」是 padded 行。遗留 B 拆成两个 crash mode：
- **① HCCL broadcast timeout ✅ FIXED**：`HcclBroadcast error 9` = rank-0 在 sidecar 跑 45 层（每步 MoE 权重
  copy 拖慢）而 rank1-7 阻塞 broadcast 超 120s。修法：`/logs/start_8001_full.sh` 加
  `HCCL_CONNECT_TIMEOUT/EXEC_TIMEOUT/EVENT_TIMEOUT=3600`（备份 .bak-g5b）→ broadcast 不再 crash。
- **② sidecar 间歇 507018（当前真遗留，= 下步主攻）**：timeout 修好后**全 45 层 forward 至少完整跑完一次
  （log 到 L44）**，但后续 forward 命中 `507018`（chip dev=8 run failed）→ 关 socket → vLLM 500。非单一确定
  层（跑完 45 层才 fault）。**下步**：(a) `--layers` 逐段缩小 + 多 forward（`--steps N`）复现，bisect 是哪
  段/第几个 forward 触发；(b) 查多 forward 复用 prepared rt 的资源累积（`WD_RING_HEAP`/ring 是否耗尽）；
  (c) 若确认是每步 MoE 权重 copy 相关，则做常驻权重（perf，与 G2 weight-IPC 重叠）顺带解。fix + 复现器在 0162。

**⚠ 恢复 gotcha（本 session 踩到）**：sidecar 507018 crash 后 vLLM EngineCore 死但 **8 个 `VLLMWorker_TP`
僵尸进程残留 cards 8-15（各 ~43GB）**，导致下次 8001 boot 报 `Engine core initialization failed`（WorkerProc
init 抢不到卡）。解法：`npu-smi info | grep VLLMWorker_TP | sed 's/|/ /g' | awk '$1>=8&&$1<=15{print $3}'`
拿 PID → `sudo kill -9`（**只杀 card 8-15 的，别碰 cards 0-7 的 327xxx = 8000 oracle**），再 restart 8001。
`nerdctl exec vllm-8001 pkill -9 -f VLLMWorker_TP` 会 self-match（exec 自己的 bash 命令行含该字符串）→ 137，
用显式 PID kill 更稳。禁 `npu-smi set -t reset`（netboot 重启全 16 卡）。

**2026-07-12(续⁵) ⭐ co-tenancy crash 彻底解决 — 剩纯数值**：offline `--steps 4`（无 vLLM）全 clean →
rt-reuse 排除 → 一攻 HcclBroadcast err9 + 二攻 507018 **同源 = co-tenancy device 争用**（vLLM `HcclBroadcast`
kernel 在同卡自旋等 rank-0 → 与 sidecar 争用）。**修复**：容器后端 `_pypto_full_forward` 把 `tp_group.broadcast`
换 **file-based broadcast**（rank-0 写 /logs，rank1-7 CPU-poll，无 device collective；已部署，备份 .bak-g5b）。
**device 验证**：8001 file-bcast 后端 + 全 45 层 sidecar → prompt **HTTP 200 完成 4 tokens 无 crash/507018**
（~120s/token）。**剩 = 攻坚 3 纯数值**：生成 token 错（text=""；早先 sweep 用合成随机 rope 只证无 nan 非正确）。
**下步**：单层 paged-index 数值对拍 vLLM decode dump，核 (1) `_wd_rope_from_emb` rope 是否匹配 step3p5
whole-decode kernel（per-op backend 的 `_rope_tables_from_attn` token-exact，但 whole-decode 是不同 kernel）、
(2) 真 KV 读（consolidated pool per-layer offset）、(3) per-layer 权重流。perf（每步权重 copy → 常驻）后置。
**⚠ hazard**：停 sidecar 后 force_reset nuke 同卡 8001 → 必 restart 8001 + 清 card8-15 zombie。

## 环境 / 铁律
- 三件套：`source /usr/local/Ascend/cann/set_env.sh && source WS/activate.sh && export PTO_ISA_ROOT=WS/pto-isa`。
- device 跑 sidecar：`SIMPLER_COMM_NO_HCCL=1 WD_RING_HEAP=1073741824 PTO2_RING_TASK_WINDOW=131072
  PTO2_RING_DEP_POOL=131072 python _stage_whole_decode_run.py --worker --tp 8 --dev-offset 8 -p a2a3
  --kv-ipc-dir <dir> ...`（cards 8-15；必须 `--worker` 才走 import_ipc 路径）。
- KV key/map 在容器 /logs（`pypto_kvpool.{key,map}.rank0..7`），`nerdctl exec vllm-8001 cat` 拷到 host dir
  （8001 必须活着）。8000 oracle cards 0-7 别碰；launch 前 `pgrep [c]hip_process` 空 + `rm /dev/shm/torch_*`。
- 停 8001：`nerdctl exec vllm-8001 bash -lc "pkill -9 -f '[V]LLM::EngineCore'"`；sidecar SIGTERM(非-9)；禁 `npu-smi set -t reset`。
  起 8001 util0.5：`bash /logs/start_8001_full.sh`。push：HTTP/1.1 + PAT /data/chensiyu/secrets/github.env（本仓在本地 box）。

## 机器状态（session 结束时）
- 8001 mode=full(util0.5) 仍在 cards 8-15、KV export 活着（容器 /logs + host /tmp/g5b_kvtest）。8000 oracle cards 0-7。
- attention_{full,swa}.py 含 G5b `layer_cache_base=0`（swa 已恢复 documented-blocker 原状，无 partial 改）；
  worker/importer/runtime 改动在 0162 工作树 + 备份 g5b_*_20260711_144753。

先读 skill/memory + ssh 0162 核对（8001 活否、/tmp/g5b_kvtest 在否），从「攻坚顺序 1」(swa 级联) 开始。
可起 team 但 lead 直接跑 device，盯增量别让 agent 空转。
```

---

# （已被上面取代 / 历史）G5b：把 vLLM 真 KV（bf16 纯 reshape）接进 whole-decode sidecar → token-exact live A/B

> 新 session 直接把下面 code block 当第一条消息粘贴。自包含。
> 生成于 2026-07-11。承接：G4 co-tenancy ✅、G2 sidecar+wiring ✅、G5a live plumbing ✅（8001 mode=full 经
> sidecar 出 token，device 验证）、G3 KV export ✅ + importer 已写；**只剩 G5b：真 KV 接进去出 token-exact**。
> 关键 device-verified 结论 + 两个已推翻的错误判断见 memory `g5b_kv_is_bf16_not_int8` + 本仓
> `phases/20-vllm-backend-monkey-patch.md` §"FINAL device-verified 定论"。下面「历史 tasks 1-5」是背景参考。

```
继续 pypto+vLLM 集成，集中攻关 G5b：把 vLLM 的真 KV 接进 pypto whole-decode sidecar，跑完整 45 层，
8001(pypto mode=full) 对 8000(vanilla) token-exact。全部在 0162（ssh 0162），repo
/data/chensiyu/hw_project/pypto/workspace/pypto-lib 分支 stepfun/develop。动手前读 skill
pypto-project/.claude/skills/pypto-dev-constraints/SKILL.md + memory g5b_kv_is_bf16_not_int8。

## 别重做（已 device 验证，见 pypto-project STATUS/phases/20）
- G4 co-tenancy 已解：simpler a2a3 comm_hccl.cpp env-gate `SIMPLER_COMM_NO_HCCL=1`（commit 878f3742，
  已重编 a2a3 runtime）→ pypto worker 与 vLLM 同卡共存 rc=0。sidecar 进程必设此 env。
- G2 sidecar：`_stage_whole_decode_run.py --serve --serve-sock <sock>` 常驻 rt + socket 服务
  （收 full hidden [BATCH=16,HIDDEN=4096] bf16 → 45 层 decode → 回 next hidden）。
- G5a live plumbing 通：容器自包含后端 /logs/pypto_patch/pypto_whole_decode_backend.py（PYPTO_WHOLE_DECODE=1
  autoload，install Step3p5Model.forward→sidecar，collective fallback 保 startup profiling）→ 8001 mode=full
  送 prompt 出 token（pypto forward #1，8 rank，0 fallback）。现 token 是 garbage（dummy KV + 4 层）。
- G3 KV export：/logs/pypto_patch/pypto_kvpool_backend.py（PYPTO_KVPOOL=1）→ 每 rank 一 key + offset map
  （pypto_kvpool.key.rankR + pypto_kvpool_map.json.rankR，45 层，L{i}.K/L{i}.V offset）。importer 已写：
  pypto-lib tools/step3p5/pypto_kv_ipc.py（KvIpcMap + build_stacked_kv，已改 bf16 默认）。

## ⭐ 核心 device-verified 事实（别再走弯路）
vLLM step3p5 W8A8 的 **KV cache 是 bf16、1 KV head/rank**（attn::Attention.kv_cache.dtype=bfloat16，
_k/v_scale=1.0 identity，calculate_kv_scales=False，num_kv_heads=1，total=8/tp=8/replicas=1），**与 worker
（bf16, KV_HEADS_LOCAL=1）完全对齐**。W8A8 只量化 weights（moe w8a8_dynamic），NOT KV。
→ **G5b 的 KV bridge = 纯 layout reshape，无 int8/dequant/scale/TP 重设计**：
vLLM KV `[2, nb, bs, 1, 128]` bf16 → 拆 kv[0]/kv[1] + drop 单 head + flatten → worker `[1, nb×bs, 128]` bf16。
⚠ 两个已推翻的错误判断（别重犯）：(1) KVPOOL 的 torch.zeros(total,int8) 只是字节容器，别据此判 KV 是 int8；
(2) 别拿某次 boot 的 "GPU KV cache: N tokens"（随 gpu-mem-util 变）当 nb×bs。用 map 的 per-K nbytes/(128×2) 算 nb×bs。

## 实现步骤（每步 device 可验证）
1. **改 KvIpcMap 产 worker 布局 bf16 DeviceTensor**：per-rank `import_ipc(worker_id=r)` → 每层 K/V
   `DeviceTensor(peer_base + L{i}.{K,V}.offset, [1, nb×bs, 128], bf16)`（nb×bs = nbytes/(128*2)）。
   build_stacked_kv → per-layer (k,v) StackedDeviceTensor(worker_ids=range(8))。
2. **worker 加 --kv-ipc-dir**：build 前设 config.MAX_SEQ_DEFAULT = nb×bs（重编 k_cache 形状）；prepare 后
   建 8 个 KvIpcMap；decode step 里 `sh["k_cache"]=stacked_k[layer]; sh["v_cache"]=stacked_v[layer]` 替 dummy。
   （_ordered_args 按 param name 取 sh；per-op attn_setup 已证 rt.run 收 DeviceTensor。）
3. **socket 协议扩 length-prefixed**：client（_pypto_full_forward / whole_decode_backend）随 hidden 发
   forward_context 的 block_table/slot_mapping/seq_lens；sidecar 每 step copy 进 sh。
4. **先 offline 验证**：sidecar --kv-ipc-dir 用一次导出的 key/map（8001 需在跑，KV buffer 活着）跑 dense L0，
   对 worker torch-ref / 或直接看 attention 出 non-nan（真 KV 进来）。
5. **复核 paged 索引等价**：worker 现用 [MAX_SEQ,128] flat + block_table/slot_mapping；vLLM [nb,bs,1,128]
   flatten nb×bs 后 block_id*bs+pos 索引应等价 —— device 验证一层 attention 数值对 vLLM dump。
6. **live A/B**：8001 mode=full（PYPTO_WHOLE_DECODE=1 + sidecar --kv-ipc-dir + PYPTO_KVPOOL=1 导出 KV，
   顺序：先起 8001 profiling 走 fallback → ready → 起 sidecar SIMPLER_COMM_NO_HCCL=1 → 送 prompt）→ 3-prompt
   对 8000 token-exact。swiglu(L43/L44) 精度也在此定论。

## 环境 / 铁律
- 三件套：`source /usr/local/Ascend/cann/set_env.sh && source WS/activate.sh && export PTO_ISA_ROOT=WS/pto-isa`。
- device：`PTO2_RING_HEAP=4294967296 PTO2_RING_TASK_WINDOW=131072 PTO2_RING_DEP_POOL=131072`；sidecar `-p a2a3`
  `--tp 8 --dev-offset 8`（cards 8-15）；sidecar 进程 `SIMPLER_COMM_NO_HCCL=1`。
- 8000 oracle 在 cards 0-7（别碰）。socket 必须在 /logs（容器 /tmp 不共享）。launch 前 `pgrep [c]hip_process` 空。
- 停 8001：`sudo nerdctl --namespace k8s.io exec vllm-8001 bash -lc "pkill -9 -f '[V]LLM::EngineCore'"`
  （不只 vllm serve，否则孤儿 EngineCore 抢卡）；sidecar 用 SIGTERM（非 -9）；禁 `npu-smi set -t reset`。
- 起 8001：`sudo nerdctl --namespace k8s.io exec -d vllm-8001 bash -lc "bash /logs/start_8001_full.sh ..."`。
- push：HTTP/1.1 + PAT /data/chensiyu/secrets/github.env（本仓在本地 box，非 0162）；同步协议见 CLAUDE.md。
- hazard：pypto AICore timeout → aclrtResetDeviceForce 卡级会 nuke vLLM；live 前确保 vLLM stream idle + pypto 不 timeout。

## 可复用积木（备份都在 workspace/g5*、g3*、g5b*）
KvIpcMap+build_stacked_kv（bf16 已改）、pypto_kvpool_backend（KV export）、pypto_whole_decode_backend
（mode=full + collective fallback）、pypto_kvscale_backend（探 attn.kv_cache.dtype 的 probe）、
start_8001_full.sh / start_sidecar.sh。sidecar worker = _stage_whole_decode_run.py（in-tree）。

先起 team（reverse-review/hw-analyst/sw-analyst/upstream-scout）+ 读 skill/memory + ssh 0162 核对现状，
从「实现步骤 1」开始。lead 直接跑 device，盯增量别让 agent 空转。
```

---

# （历史参考）G2-G5 live wiring（tasks 5-7）—— 本 session 已推进到 G5b（上面是聚焦提示词）


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
4. ✅ **接 `_pypto_full_forward` — 代码已写 + socket 客户端 unit-tested（2026-07-11）**：
   `tools/step3p5/vllm_monkey_patch.py` 加 `_WholeDecodeClient`（AF_UNIX，pad live `[T,HIDDEN]`→
   `[BATCH,HIDDEN]` 送、recv、slice 回）+ `_whole_decode_client()` singleton（env `PYPTO_WHOLE_DECODE_SOCK`，
   sidecar 缺失时 fail-closed 明确报错）；`_pypto_full_forward` body 实装：embed（vLLM）→ rank-0 drive
   sidecar → `get_tp_group().broadcast(src=0)` → 返回 post-45-layer / pre-final-norm hidden。**KV/attn 是
   sidecar 内部事（G3），此 body 对 G3 前向兼容、无需改**。**验证**：AST_OK + IMPORT_OK + socket 客户端
   pad/send/recv/unpad round-trip unit-test PASS（echo bit-exact）。备份
   `workspace/g2_vllm_monkey_patch_wiring_20260711_023659.py`。**剩（G5 时验证）**：live embed/broadcast
   API + 真数值（须 G3 真 KV + 8001 mode=full 实跑）。
5. 🟡 **G5a live plumbing DONE（2026-07-11 device 验证）/ G5b token-exact 剩**：8001 起 mode=full
   （`/logs/start_8001_full.sh`：`PYPTO_WHOLE_DECODE=1` + 自包含容器后端 autoload）→ **health=200**
   （collective fallback 让 startup profiling 全 rank 回退存活）→ 起 sidecar（`/tmp/start_sidecar.sh`，
   `SIMPLER_COMM_NO_HCCL=1 --serve`，co-resident running 8001）→ 送 prompt → **HTTP 200 出 token**，
   8001 log `[pypto whole-decode] pypto forward #1 hidden(2,4096)->(2,4096)`（8 rank，0 fallback），
   sidecar log 收到 live hidden 跑 decode。→ **embed + tp broadcast + socket 路由 + live co-tenancy 全
   device 验证**。**G5b 剩（gate 在 G3 真 KV + 全 45 层）**：现 sidecar 用 dummy-KV + 4 层 → token
   garbage（nan）；接 G3 真 KV + `--layers 0..44` → 3-prompt A/B vs 8000 token-exact。
   ⚠ 停机顺序：`pkill -9 -f "[V]LLM::EngineCore"`（不只 vllm serve，否则孤儿 EngineCore 抢卡）→
   sidecar 用 SIGTERM（非 -9，clean chip finalize）；启动前 `rm -f` sock 让 profiling 走 fallback。

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
