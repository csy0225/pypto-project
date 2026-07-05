# Milestones —— 2026 Q2

按 session 划分的 milestone 日志，append-only，按日期降序。
高层 Phase 01-19 总结见
[`prototype-phase-01-19-summary.md`](prototype-phase-01-19-summary.md)。



## 2026-07-05 (later-8) —— 穷尽调参矩阵：507018 co-tenancy 不可调，socket-worker 路径判定不可行 ⏸

- **穷尽 (worker-env × vLLM-gpu-mem) 网格**：(default,0.80)→routed 507018；(RING_HEAP=4GB,0.80)→
  16GB arena OOM(207001)；(4GB,0.55)→507018(rank0,2)；(default,0.55)→507018(rank2)。**只有 16GB arena
  OOM 可用 gpu-mem 调掉；507018 对 env/内存全不敏感**。→ 重型 routed grouped-GEMM 内核在独立 ChipWorker
  里与 active vLLM Worker_TP 同卡 → AICPU device-context co-tenancy 507018（部分卡、非确定）。卡从未 poison，
  每次干净拆除。
- **最终判定**：socket-worker + 独立 pypto runtime 对重型 routed MoE 内核在 live vLLM co-tenancy 下**根本
  不可行**（轻内核 dense/tail 可共驻；routed 36 专家 ~16GB arena 不行）。**唯一可行路径 = 项目既定的
  device-IPC 零拷贝重构（Phase 23/24）**：pypto runtime 在进程内接管计算（一个 device context，无第二争用
  runtime）——这正是 Phase 23/24 存在的原因。多周级，是下个 session 的正确方向。**不要再试 socket-worker
  路径**——调参空间已穷尽（上表）。
- goal（live A/B）**用当前架构不可达**；非数学/接线问题（所有 compute 已验证 bad=0 入库）。机器干净收尾：
  8001 down，cards 8-15 OK，8000 up，0 worker。



## 2026-07-05 (later-7) —— live MoE A/B 迭代调试：507018 co-tenancy 定为 definitive blocker ⏸

- **3 轮 live 部署迭代调错**：(1) 默认 env → routed worker `507018`（co-tenancy）；(2) 加
  `PTO2_RING_HEAP=4GB` → fault 变显式 `rtMalloc 207001 size=16GB`（routed 运行时 pooled static arena
  要 ~16GB，vLLM gpu-mem 0.80 下每卡仅 ~14GB free 装不下）；(3) 再把 vLLM gpu-mem 降到 0.55（~28GB
  free/卡）→ arena OOM 消失，**但 routed 内核仍在 rank 0 & 2 上 `507018`**（跨 rank 非确定）与 active
  vLLM Worker_TP 共卡。卡从未 poison（task 级 fault），每次干净拆除，8-15 OK，8000 up。
- **Definitive blocker**：重型 routed grouped-GEMM 内核在**独立 ChipWorker 进程**里与**active vLLM
  Worker_TP 同卡**运行 → 部分 rank `507018`（AICPU stream sync / device 争用），即便修掉 16GB arena
  OOM 也在。dense-MLP + tail 的 @pl.jit worker 不会（task graph 轻）。→ **socket-worker + 独立 runtime
  对 routed 内核在 live vLLM 下不可行**，是真正多周硬点。
- **下步（按优先级）**：(1) **device-IPC 零拷贝**（Phase 23/24 机制，一个 runtime 无第二争用 context ——
  项目既定方向，socket-bridge 一直只是 oracle 回退）；(2) dispatch-cut 缩小 routed（更少专家/更小 arena）
  看能否共驻 + 查 16GB arena 为何这么大（疑过度预留）；(3) stream/event 序列化 routed 与 vLLM 避免
  AICPU 调度重叠。
- **目标状态**：live 单层 MoE A/B **未通过**——卡在 507018 运行时 co-tenancy，**非数学/接线**（所有
  compute 已验证 bad=0，8 commit 入库）。部署已全接线 + 可复现（w8a8 logdir：`start_8001_moe.sh`
  gpu-mem 0.55、`restart_routed_workers.sh` +PTO2 ring env、`pypto_patch_moe/`）。收尾机器干净：8001 down，
  cards 8-15 OK，8000 oracle up，0 worker。



## 2026-07-05 (later-6) —— live 单层 MoE 部署实跑 → 运行时 507018 co-tenancy blocker（实测定位）⏸

- **实际执行了完整 live 单层（layer 3）MoE 部署**（非 spec）：
  - 起 8 个共驻 routed worker（setsid，cards 8-15，`pypto_mlp_worker --routed-layers 3`）→ 全部
    `listening`（setsid 持久化 OK；之前 setsid「失败」其实是自匹配 pkill 先杀了 shell）。
  - 8001 带 MoE backend 启动（`nerdctl start vllm-8001` + `start_8001_moe.sh`，gpu-mem 0.80，
    PYTHONPATH=/logs/pypto_patch_moe，PYPTO_MOE=1，PYPTO_MOE_LAYERS=3）→ **8 个 rank 全部
    `[pypto_moe_backend] installed sock_dir=/logs layers={3}` + FusedMoE.forward layer-tracking，
    Health 200，Application startup complete**。lazy rank 解析（local_rank）生效。
  - 首个真实请求 → **routed worker `run_prepared failed with code 507018`**，vLLM 收到
    `ConnectionError: worker closed`，8001 请求失败；8000 vanilla 正常。
  - **事后卡健康 8-15 全 OK（本次未 poison）**——507018 只杀了 worker 进程、未 Alarm 卡（task 级 fault，
    比早先 IPC-map 事故轻）。已干净拆除（stop 8001 + pkill workers），卡 OK，8000 up。
- **实测定位的真 blocker**：routed pypto 内核**与 active vLLM Worker_TP 共卡运行时 507018**。关键对比：
  dense-MLP + tail 的 @pl.jit worker **不会**触发（此前共驻 3/3 token-exact）——所以是 routed 专属
  （36 专家 RECV-tiled grouped-GEMM 的重 task graph 与 vLLM live device context/AICPU 调度器争用）。
  之前「共驻 routed PASS」是 worker 单独占 card 8（无 vLLM）；只有 vLLM Worker_TP 同时 active 才暴露。
- **含义**：socket-worker + 独立 ChipWorker 路径对 routed MoE 内核在 live co-tenancy 下不可行。正解是
  项目既定的 **device-IPC 零拷贝方向**（Phase 23/24，一个 runtime、无独立争用 device context），或对
  routed 内核与 vLLM stream 做序列化。这是真正的多周硬点。**所有 compute（内核/worker/backend/协议/
  layer-targeting）已验证 + 入库；blocker 是运行时 device-context 争用，不是数学或接线。**
- 下步选项：(1) dispatch-cut 排查为何 routed 507018 而 dense/tail 不会（缩到 1 专家能否共驻）；
  (2) 换 device-IPC 零拷贝（Phase 23 机制）替代 socket + 独立 ChipWorker；(3) stream/event 序列化避免
  routed 与 vLLM 在 AICPU 调度器上重叠。部署产物已 stage 在 w8a8 logdir（`start_8001_moe.sh` /
  `restart_routed_workers.sh` / `pypto_patch_moe/`）供下个 session 直接复现。



## 2026-07-05 (later-5) —— backend↔co-resident-worker code path 完成：live 单层 MoE 代码全就绪 ✅

- **`pypto_moe_backend.py`（pypto-lib `bdcb1b7`）改用 co-resident worker 协议**：`RoutedClient` 从
  `_serve` LE 协议改成 **`pypto_mlp_worker` BE/nbytes 协议**（op=routed, rows/layer/offsets/counts +
  int16 bf16），这样 backend 直接对接**共驻 @pl.jit worker**（非 @pl.program `_serve`）。加
  `FusedMoE.forward` layer-idx 追踪（threadlocal），只 route `PYPTO_MOE_LAYERS`（tracking 不可用时
  单层 shortcut）。**backend selftest vs 共驻 worker：bad_ratio@0.05=0.0000** —— `_apply_mlp` →
  共驻 worker `routed` → 正确 y 的全路径已用 LIVE 协议在 device 上端到端验证。
- **live 单层 MoE 的全部代码已就绪 + device 验证（六连，fork stepfun/develop）**：`fc0bafb`（内核）→
  `e17b4ab`（_serve fix）→ `20292aa`（backend v1）→ `ae00e9a`（@pl.jit device-run）→ `0249700`（共驻
  worker routed op）→ `bdcb1b7`（backend 共驻协议 + layer targeting）。每个都 bad_ratio@0.05=0.0000。
- **剩余 = 纯部署 + 42 层内存（无更多代码设计）**：单层 live A/B 只需 (1) 8001 拉起；(2) 起 8 个共驻
  routed worker（`pypto_mlp_worker --routed-layers 3`，cards 8-15，与 vLLM Worker_TP 共驻，@pl.jit
  ChipWorker 无 co-tenancy）；(3) sitecustomize `pypto_moe_backend.install()` + env
  `PYPTO_MOE=1/PYPTO_MOE_SOCK/PYPTO_MOE_LAYERS=3`；(4) curl A/B vs 8000。全模型 42 层需 worker 侧
  专家权重 LRU/按需（~47GB/rank，多周硬点；单层可放下）。



## 2026-07-05 (later-4) —— co-resident worker routed op device PASS：live worker config 成立 ✅

- **`pypto_mlp_worker.py` 加 `op=routed`（pypto-lib `0249700`）**：把 `routed_experts_jit` 注册进与
  dense/shared/tail **同一个 `ChipWorker`**（一进程一卡）+ 加载 per-rank dequant-W8A8 专家；
  `routed_partial` pad 到 LOCAL_RECV_MAX、run、unpad；`--routed-layers` CLI。
- **device 验证 co-resident**（card 8 上 dense layer 0 + routed layer 3 同 ChipWorker，client 打真实
  往返）：**MLPW_ROUTED_PASS**，rows=1024，maxdiff=0.0020，bad_ratio@0.05=0.0000。这是**正式 live
  worker 配置**（routed 与 dense/shared/tail 共驻一进程）→ **无独立 @pl.program 进程 → 无 507018
  co-tenancy**（避开 card-8 事故那类问题）。live wiring 组件 #1 完成且可提交。
- **本 session 五连（全部 device 验证 + 已推送 fork stepfun/develop）**：`fc0bafb`（routed 内核真 W8A8
  bad=0 + _serve）→ `e17b4ab`（_serve bf16 fix，worker round-trip）→ `20292aa`（backend hook
  `_apply_mlp` glue selftest）→ `ae00e9a`（@pl.jit device-run）→ `0249700`（co-resident worker routed op）。
- **剩余 full live e2e A/B**：(a) backend layer_idx 注入（多层；单层现成）；(b) **42 层专家权重内存**
  （全驻 ~47GB/rank → LRU/按需，多周硬点；单层可放下）；(c) 8001 拉起 + backend client 指向 worker
  的 routed sock（协议 BE >I + nbytes + int16 bf16）+ live A/B vs 8000。五个组件均已单独 device 验证 +
  入库，剩余是 live 组装 + 内存扩展。



## 2026-07-05 (later-3) —— @pl.jit routed device-run PASS：co-resident live 路径解锁 ✅

- **`_routed_jit_probe.py --device-run`（pypto-lib `ae00e9a`）证明 RECV-tiled routed body 作为 plain
  `@pl.jit` 在 device 上 RUN 正确**（不只是 compile）：真 W8A8 layer 3，ratio_allclose(atol=0.04,
  rtol=0.04) PASS，4.37s。
- **为什么关键（解掉一直卡的 live blocker）**：`@pl.program` 的 `_serve` worker 太重，**不能**和
  vLLM Worker_TP 共卡（co-tenancy → 507018 → card-8 事故）。但 live 8001 的 TP=8 Worker_TP 占满
  cards 8-15，独立 routed `@pl.program` 进程无处可跑。`@pl.jit` 变体轻量、可共卡（现有
  attn/dense/shared/tail 的 @pl.jit op 就是这样共驻在 `_stage_attn_worker.py`）。→ **正确的 live 集成 =
  把 routed op 作为 `@pl.jit` 注册进现有共驻 attn worker（`ChipWorker.register`），而非独立 `_serve`
  进程**；backend `pypto_moe_backend.py` client 连 attn worker 的 socket 即可。
- 这修正了 `deployment/moe-routed-live-wiring.md §4.1` 的 live 路径（`_serve` 独立进程仅适合离线验证，
  我正是用它离线验的；live 用 co-resident @pl.jit）。
- **本 session 累计（全部 device 验证 + 已推送）**：routed 内核真 W8A8 bad=0（fc0bafb）→ `_serve` bf16
  fix + worker round-trip（e17b4ab）→ backend hook `_apply_mlp` + glue selftest（20292aa）→ @pl.jit
  device-run + co-resident 路径（ae00e9a）。剩余 live e2e = 把 routed @pl.jit 注册进 attn worker +
  layer_idx 注入 + 42 层权重内存（LRU/按需）+ 8001 拉起 + A/B。



## 2026-07-05 (later-2) —— MoE routed backend hook `_apply_mlp` 落地 + device glue-test PASS ✅

- **backend hook 代码写完（不是 spec，是可用代码）**：`pypto-lib/tools/step3p5/pypto_moe_backend.py`
  （`20292aa`）monkey-patch `MoECommMethod._apply_mlp` → pypto RoutedExperts worker：
  - `_to_csr(group_list, group_list_type)`：type 1=counts / 0=cumsum→diff，offsets=exclusive
    prefix-sum；`torch.equal` 验证与 `_balanced_csr` 一致。
  - `RoutedClient`（UDS，uint16-view bf16 协议）+ `_pypto_apply_mlp`（pad 到 LOCAL_RECV_MAX、route、
    unpad 回 num_recv；`num_recv>1024` → vanilla fallback）。
  - `install()` 经 sitecustomize autoload（`PYPTO_MOE=1` / `PYPTO_MOE_SOCK` / `PYPTO_MOE_LAYERS`）。
- **device self-test PASS**（`--selftest`，真 W8A8 layer 3，worker on card 8）：num_recv=1024，
  y_shape=(1024,4096)，maxdiff=0.0020，**bad_ratio@0.05=0.0000**。`_apply_mlp` 替换的 compute+glue
  全部在 device 上证对。
- **至此 worker + backend-glue 均已 device 验证**。剩余 live e2e = (1) in-vLLM autoload（8001 全栈拉起
  多步重建 + 8 个 per-rank routed worker + install 进 sitecustomize）；(2) 多层的 layer_idx 注入
  （单层现成）；(3) **42 层专家权重内存**（全驻留 ~47GB/rank → 需 per-layer LRU / 按需加载，真正多周
  硬点）；(4) live A/B vs 8000。见 `deployment/moe-routed-live-wiring.md`。



## 2026-07-05 (later) —— MoE routed worker `routed` op device round-trip 验证通过 + _serve bf16 bug 修复 ✅

- **worker `routed` op device round-trip PASS**：腾空 cards 8-15（8001 已是死状态——Worker_TP
  zombie + 重复 host attn worker，stop 容器 + pkill 清干净）后，在干净 card 8 起 `--serve` worker +
  socket client 打真实往返：真 W8A8 layer 3，`max|y|==max|golden|=0.4551`，maxdiff=0.0020，
  **bad_ratio@0.05=0.0000**，rtt 3.25s。serialize→`compiled(...)`device→deserialize 全路径证通
  （`deployment/moe-routed-live-wiring.md` §4.1 完成）。
- **修复 `_serve` bf16 序列化 bug（pypto-lib `e17b4ab`）**：原 `y1[0].contiguous().numpy().tobytes()`
  在 bfloat16 上 `TypeError: unsupported ScalarType BFloat16`——即 fc0bafb 的 worker op 返回时必崩。
  改 `.view(torch.uint16).numpy().tobytes()`（client 端 uint16→bfloat16 反序列化，2-byte 布局一致）。
  device 计算本身正常（`chip_process dev=8 ready`），仅序列化行。round-trip 测试的价值正在于此。
- **0162 launch gotchas（记入避坑）**：(1) `pkill -f "vllm_routed_experts --serve"` 自匹配自身 shell →
  SIGKILL 自己 → 后续 rm+launch 不执行、无输出 → 用 bracket trick `pkill -f "[v]llm_routed_experts"`；
  (2) netboot/cgroup 激进回收：ssh 里 nohup/setsid/tmux 全在 ssh 断开时被杀（tmux server 都不留）→
  可靠做法是 worker 跑在一条**保持打开的前台 ssh**（后台 hold）+ 另开 ssh poll log/跑 client。
- **边界**：这坐实了 worker 侧（内核 + 真权重 + socket 往返）完全 OK；剩余仍是 backend `_apply_mlp`
  hook（层号注入 + shared 合并 + group_list→CSR）多周工程，见 `deployment/moe-routed-live-wiring.md` §4.2。



## 2026-07-04/05 —— MoE routed-expert 内核真权重验证 + vLLM serving 从零重建 + pypto dense/attn/tail live 逐字对齐 ✅

- **MoE routed-expert per-rank 内核（最后一块 MoE 计算内核）验证通过**：新增
  `pypto-lib/models/step3p5/vllm_routed_experts.py` —— per-rank 36 本地专家的 grouped
  SwiGLU（`N_LOCAL_EXPERTS=36`、`LOCAL_RECV_MAX=1024`、SiLU），**无 collective**，正好是
  vLLM FusedMoE all-to-all dispatch/combine 包裹的 per-rank 计算 seam。body 来自
  `moe.py::_expert_routed`，RECV_TILE=32 行分块（naive `[1024,1280]` FP32 累加器=5MB 会爆
  188KB UB，必须行分块），封成 `@pl.function(Inline)` 塞进 `@pl.program RoutedExperts`
  （chip_orch + host_orch per-rank dispatch）。**关键：tile body 外必须加 `if tile_valid > 0:`
  守卫**（否则 ~31/32 空尾块提交 expert kernel with tile_valid<=0 → 507018；这是第一次 device
  失败的根因）。
  - **device 结果（真实 W8A8，恢复后的 card 8/9）**：synthetic PASS bad_ratio=0.0067；
    **真实 W8A8 layer 3 rank 0 PASS bad_ratio=0.0000**，max|out|=max|ref|=0.428。真权重经
    `weight_loader._load_quantized_expert_projector`（INT8 + `_scale`/`_offset`→BF16）+ HF
    gate/up `[INTER,HIDDEN]`→`[HIDDEN,INTER]` 转置。
  - **worker `routed` op 已实现**：`vllm_routed_experts.py::_serve()` 起最小 UDS worker
    （4-byte len + JSON header + BF16 body），收 BF16 hidden + `offsets`/`counts`，跑编译好的
    RoutedExperts（真实 dequant W8A8 专家），回 BF16 y；host 已验证。`_routed_jit_probe.py`
    另证 RECV-tiled body 也能编成 `@pl.jit`（worker 可像 dense/shared 一样 `register`）。
  - 代码已 push：**`pypto-lib` `fc0bafb`**（csy0225 fork stepfun/develop）。
- **机器事故 + 完整恢复（自伤 → 全恢复）**：首次 routed device-run 误在 -d 8 与 live 8001
  worker **co-tenant** 跑重型 `@pl.program` → 507018 → card 8 Health=Alarm → `npu-smi set -t
  reset -i 8` 在 AMP+HCCS 模式下**重启全部 16 卡**（用户批准）→ 固件 load 卡死（`flag_r=0x6666`/
  `dcmi -8005`）→ `sudo RECOVERY.sh` 重装 driver 但需重启 → host 重启 → **netboot 抹掉
  authorized_keys → SSH 锁死 ~8h**（cluster provisioning 最终恢复 key）。恢复顺序（netboot
  tmpfs 丢失 `/` 全部）：(1) 挂 NVMe（`/dev/nvme0n1`→/mnt/persist、`/dev/nvme1n1`→/data；
  w8a8 ckpt 在 `/data/chensiyu/step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp`）；
  (2) 建 `HwHiAiUser`（否则 driver 装报 0x0091）；(3) `sudo RECOVERY.sh` → driver 25.5.2 +
  firmware 7.8.0.7.220 + ptoas 0.45，**16 卡 Health=OK（card 8 Alarm 清除）**；(4) 修 cann
  symlink → CANN 9.0.0 non-GA（workspace runtime 编译所依赖，RECOVERY.sh 指向 beta.1 是 stale）；
  (5) `apt install libstdc++-12-dev`（CCEC 需 `<cstdint>`）。**铁律**：AMP+HCCS netboot 机上
  **绝不**单卡 `npu-smi set -t reset`（会重启全部卡）+ **绝不**在有 live vLLM worker 的卡上跑重型
  `@pl.program`（co-tenancy → 507018 → 需 root reset）。
- **vLLM serving 从零重建（早先"需 cluster provisioning"的判断是错的）**：用户提示"镜像在某个盘里"
  破局。正确镜像不是 skew 的 lijiahui/vllm-ascend，而是
  **`hub.i.basemind.com/stepcast/stepcast:0.19.0-...`**，从 **docker data-root
  `/mnt/nvme1/chensiyu/docker-data`** 找到（dockerd 已随 netboot 消失，但
  `containers/<id>/config.v2.json`+`hostconfig.json` 存了每个原容器 spec，
  `image/overlay2/repositories.json` 列出镜像）。重建配方（可复现）：
  (1) 挂 NVMe；(2) 从 `/mnt/persist/k8s-install/containerd` 起 containerd（root bind-mount）；
  (3) **runc 1.1.8 `--no-pivot` wrapper**（netboot `/`=rootfs，默认 pivot_root 失败）；
  (4) `nerdctl -n k8s.io pull` 正确 stepcast 镜像；(5) `nerdctl run -d --privileged --network
  host`（privileged→全 NPU）；(6) `nerdctl exec` 起 serve 脚本。**3 个 gotcha**：(a) 不能
  `set -u`（set_env.sh 有 unbound var → 静默退出、0-byte log）；(b) 必须 `export VLLM_USE_V1=1`
  （否则 `hf_overrides must be a dict`）；(c) **DROP `--speculative_config`（MTP）**——draft
  config 再触发 hf_overrides bug；MTP 是 spec-decode，greedy(temp=0) 输出与不带 MTP 完全一致 →
  仍是有效 A/B oracle。**8000 oracle UP（health 200，cards 0-7）**，生成"北京，简称京，是中华人民
  共和国的首都…"。同配方起 8001（cards 8-15）跑 pypto。
- **pypto dense0-2 + attn + tail 在重建平台上 LIVE 且逐字对齐**：8001 pypto = 可用 vanilla boot
  env + `PYPTO_*` 开关（ATTN_BACKEND=1、KV_IPC=1、AB=0、LAYERS=0,1,2、FUSE_MLP_LAYERS=0,1,2、
  TAIL_LMHEAD=1）+ 8 host worker（cards 8-15，8/8 socket）。backend `pypto_attn_backend.py`
  经 `/logs/pypto_patch/sitecustomize.py` autoload。**A/B 结果：3/3 token-EXACT**（8001 pypto vs
  8000 vanilla，temp=0，prompts 北京/中国首都/1+1）。GOTCHA：pypto decode ~2.5s/token（per-layer
  socket round-trip）→ curl 需 `-m150`（否则超时看似"empty"，非 bug，Phase 26 perf）。
  RESTART GOTCHA：kill 旧 8001 后 Worker_TP 仍占 HBM → 用 bracket-pattern `pkill -9` 确认
  HBM<10% 再重启。**live pypto pipeline（attn + dense-MLP + tail lm-head，layers 0-2）在重建
  平台证明正确 = 加 MoE routed experts 的地基**。
- **剩余（full MoE live，多周级）**：接 validated routed 内核 —— worker `routed` op（已实现，需
  device round-trip 在空闲卡验证）+ **backend hook `MoECommMethod._apply_mlp`→`unified_apply_mlp`**
  （映射 `MoEMlpComputeInput.group_list`→CSR offset(cumsum)/count；处理 W8A8 dynamic act-quant），
  覆盖 MoE 层 3-44，再对 8000 oracle 做 live A/B。内核 + hook seam 已定位/验证，集成是多周工程。
- **边界**：本 session 交付 = routed 内核真权重精度闭环 + serving 重建配方 + dense/attn/tail live
  逐字对齐 + worker op 实现；**未做** = backend `_apply_mlp` hook + MoE 层 live A/B（下个 session
  从此继续）。容器侧改动（step3p5.py `tp_in_dp` drop、optimus stub、start 脚本）在 disposable
  container overlay，非 repo；仅 `vllm_routed_experts.py` + `_routed_jit_probe.py` 入 git。



## 2026-07-03 —— 零拷贝 KV-IPC 集成 step 1-5 验证通过 + IPC 主卡点解除 + 重制定 plan ✅

- **背景/纠偏**：项目此前偏成「算子桥接」（每 rank 独立 worker + socket/device-IPC 桥单算子，丢融合收益 + host round-trip ~2.6 tps）。按用户+技术专家 7 步路线，验证「PyPTO runtime 通过 device-IPC 零拷贝接管 vLLM KV 计算」。
- **step 1-5 全部在 0162 card 8 device 实测 PASS**：
  - **step 1**：torch_npu 有 torch.cuda 级 IPC（`rebuild_npu_tensor`/`storage._share_npu_`/`torch_npu.multiprocessing`/`NPUIPCTypes.cpp`）+ 裸 ACL；device tensor 导出 rc=0。**测量到跨进程 import 的 VA 不同但 offset 保留**（`_stage_va_ipc_probe.py`：exporter `0x12c041…`→importer `0x12c1c0…`，`base+4096` 读回正确）。
  - **step 2**：import 的 IPC 指针 → `DeviceTensor` → 真 kernel `bad_ratio=0`（复用 P4/P7）。
  - **step 3**：一 key + `DeviceTensor[block]` 自动 offset，多块 kernel 读取全对（`_stage_vamap_multiblock.py` `VAMAP_MULTIBLOCK_PASS`）。
  - **step 4/5**：45 层 KV 合一 buffer → **1 个 export key** → 1 次 import → **90 条 offset map** → **无 per-tensor MemPool → 无 OOM**；嵌套 offset（层 map + block_table 分页）零拷贝喂 page_attention kernel，跨层 0/22/44 × 块 0/3/7 K/V 全 `bad_ratio=0`（`_stage_kvpool_pageattn.py` `KVPOOL_PAGEATTN_PASS`）。
- **技术解除**：IPC 主卡点根因 = 旧方案「每 tensor 一个 `torch.npu.MemPool`」→ 45 层 90 pool → `rtReserveMemAddress` **207001 OOM**（只撑 4 层）。正解 = 找到真实分配点 `vllm-ascend model_runner_v1._allocate_kv_cache_tensors`（per-layer int8），KV 合一 buffer → **一 key + offset map**。507899（子指针导出）+ 207001（OOM）**双卡点解除**。
- **重制定 plan**：范式定为 out-of-process worker + device-IPC 零拷贝（一 key 整池 map）；socket 桥降级为精度 oracle。新 phase：**24**（step6 整层 live 替换）/**25**（step7 真 module 全网 + Wave-3 whole-model orchestration）/**26**（perf，原 22）。详见 [`../phases/23-zero-copy-kv-ipc-validation.md`](../phases/23-zero-copy-kv-ipc-validation.md)。
- **边界**：验证的是**真实 KV 布局/规模下的机制**；接进 live 8001 服务 loop 是 Phase 24 工程（此前 socket-bridge 已部分打通真实 KV 导出 + decode attention `bad_ratio=0`）。
- **产出脚本**（0162 staging，未入 sub-repo）：`_stage_va_ipc_probe.py`、`_stage_vamap_multiblock.py`、`_stage_kvpool_pageattn.py`。
- **0162 现状**：为腾卡验证 kill 了 8001 + 8 个 pypto attn worker（cards 8-15 空）；**8000 baseline 保留**（cards 0-7，200）。



## 2026-07-02 —— Step3p5 attention 多 position (ctx>1 / prefill) 乱码根因定位 + 修复 ✅

- **症状**：step3p5 full-attention 在**多 position（ctx_len>1 / prefill、带历史的 batched decode）**输出乱码，**单 position（ctx_len=1）正确**。离线复现（`_stage_attn_e2e.py`，`seq_lens=arange(BATCH)+1` crossrow）：row 0（ctx=1）对，rows 1..15 全错（`bad_ratio≈0.90`）。因为 `test_decode_layer_full_dense_st` 只测 ctx=1，一直没暴露；2026-06-30 的 attention device-shared e2e 也是 ctx=1（`bad_ratio=0.0000`），同样掩盖了它。
- **为什么 ctx=1 掩盖 bug**：ctx_len=1 时 softmax 只有一个元素、权重恒=1，attention 输出恒=V₀，**与 q·k 分数无关**。所以错误的 q·k **值**在 ctx=1 完全不可见，只在 ctx>1（按分数加权）时暴露。
- **定位方法**：新建独立最小复现器 `_stage_scope12_qk.py`（standalone L3 `@pl.program`，逐字复制 `attention_full.py` Scope 1（RMSNorm+Q/K/V proj+q_norm/k_norm）+ Scope 2（partial RoPE + KV-cache 写 + all_q_padded 打包）+ Stage-1 QK，per-rank 配置 `apply_perrank_patch`），逐层 dump 对拍 torch golden：`q_proj_norm`✅ `k_proj_norm`✅ `k_cache`✅，唯独 `all_q_padded`（打包后的 Q）**首错在 (row0, col32)**（col32 = `ROTARY_HALF_FULL` = `rot_q_hi` 段起点；`rot_q_lo` 的 cols 0..31 正确）。`REAL_ROPE=1` 时误差更大（all_q_padded 0.19、scores 0.90）。
- **根因**：Scope 2 里 Q 的 partial-RoPE 打包进 `all_q_padded` 是一个 **pypto/ptoas codegen 数值 bug**，定位在 `rot_q_hi` 写入区（列 `ROTARY_HALF_FULL..ROTARY_DIM`）。原写法 `q_block = reshape(slice(q_proj_norm,[1,8*128]),[8,128])` → 对 reshape 后的 `[8,128]` tile 在 col offset 32 切 `q_hi` → `[8,32]` `col_expand_mul` + assemble 到 `all_q_padded` col 32 —— 这条"reshape + col-offset 子列切片 + `[8,32]@col-32` assemble"链路 miscompile。**单行 K 路径（`[1,32]` 切 `k_proj_norm`）正确**，只有多行 Q 出错。
- **修复（model-side，已落地并本地验证）**：把 Q RoPE 打包改成**逐 head 用 `[1, ROTARY_HALF]` 连续切片**（完全镜像已验证正确的 K 路径），逐 head assemble 进 `all_q_padded`。应用到 `pypto-lib/models/step3p5/attention_full.py`（Scope 2）和 `attention_swa.py`（Scope 2；SWA 无 full-row assemble，保留其结构）。数学等价。
- **验证（0162 card 8，修复后）**：`_stage_scope12_qk` scores identity 0.2482→**0.0018**（bf16 噪声）、`REAL_ROPE=1` 0.8998→**0.0000**；`_stage_attn_e2e.py ATTN_PERRANK=1` crossrow 全 decode 层（attn+MLP）0.8374→**0.0000 PASS**；`test_decode_layer_full_dense_st -d 8` 单 position 无回归 **PASS 7.97s**。
- **涉及仓库**：修复在 `pypto-lib/models/step3p5/{attention_full,attention_swa}.py`（**本地工作树，尚未 push**，本次会话按用户要求只推 pypto-project 文档）。复现器 `_stage_scope12_qk.py` + e2e `ATTN_PERRANK`/`ATTN_FULL64` 开关（默认关）在 pypto workspace root（本地）。
- **另一个独立 bug（非本根因）**：`apply_tp1_patch`/unsliced 路径下 Stage-1 `q_padded_row = fa_b*Q_HEAD_PAD_FULL` 与 Scope-2 打包 stride（含 `KV_HEADS_LOCAL`）不一致，仅 `KV_HEADS_LOCAL>1` 触发；生产 per-rank（`KV_HEADS_LOCAL=1`）不受影响。
- **遗留**：SWA 修复已应用+编译通过，但 SWA ST 在共享卡 runtime OOM（tensor-14 需 3.3GB，co-tenant 占内存，非本修复回归）→ SWA runtime + crossrow 精度待空闲卡验证；`prefill_attention_full.py` 已用 `[1,32]` 逐 token 切片，大概率不受影响，待单独确认；深度技术 writeup 按协议应落 `pypto-lib/docs/known-pypto-pitfalls.md`（待 pypto-lib push 时补）；上游 pypto/ptoas codegen bug 待用 `_stage_scope12_qk.py` 提。



## 2026-06-30 —— Step3p5 attention 设备共享 e2e PASS + device-shared 地基提交 ✅

- 在 `gpu-a910x-0162` 打通 **attention 层经 device-IPC 共享 KV 的离线端到端**：独立进程 ctypes 零初始化 `(2,4096,128)` bf16 KV 块 + `aclrtIpcMemGetExportKey`；worker 编译 `select_decode_layer(0)`（full_dense，L3 fork chip child）→ `DistributedWorker` → `rt.import_ipc(key)` → K/V `DeviceTensor` → `rt.run`，对 torch golden（`_torch_attn_no_gate + _torch_dense_mlp`）`bad_ratio=0.0000`。脚本 `_stage_attn_e2e.py`。
- 关键修复 **`DeviceTensor.__getitem__`**：生成的 L3 `host_orch.py` per-rank 切片 `k_cache[r,0:R,0:H]`；新增连续子视图（row-major offset ptr + 降维/resize；非连续内层 slice 报错）。
- option B 底层代码提交（本地 feature 分支 `pypto/device-shared`，未 push）：simpler `18bddac2`（import_ipc 全链路：CTRL_IMPORT_IPC + DistributedWorker.import_ipc）；pypto `0c4b8749`（`DeviceTensor.__getitem__` + import_ipc + 子模块 bump）。8 文件 b-csy-develop↔0162 md5 一致。
- vllm-ascend 镜像源同步到 `0162:/data/chensiyu/hw_project/pypto/vllm-ascend`（shallow，tar `.git` + `git reset --hard`），分支 `pypto/attention-integration`（off fork `fbfe288`），提交 live 集成蓝图 `PYPTO_ATTN_INTEGRATION.md@ba72967`（Option A：复用 `attention_full`，patch `Step3p5DecoderLayer` attention 子块；checkpoint 权重名 / 独立 attention 程序 `build_tp_attention_full_program` / KV-rows ABI / socket 协议 / S1-S4 步骤已逆向）。
- 8001 在线服务恢复（dense 0-2 + shared 3-44），8000=200/8001=200，8 worker，正常出 token。**修正恢复顺序铁律**：先起 8001 做完 TP=8 HCCL init → `Application startup complete` → 再起 pypto worker；worker 占卡 8-15 期间 vLLM HCCL init 会 `hcclCommInitRootInfoConfig error 15 / rtBinaryGetFunction 107000` 全挂，`aclrtResetDeviceForce` 不解。另：`pkill -f pypto_mlp_worker` 自匹配 ssh shell → 用 `'[p]ypto_mlp_worker'`；e2e exporter 须 `aclrtIpcMemClose`（泄漏 exbus 句柄会脏卡）。
- 涉及仓库：`pypto pypto/device-shared:0c4b8749`（local）、`simpler pypto/device-shared:18bddac2`（local）、`vllm-ascend pypto/attention-integration:ba72967`（local，0162）、`pypto-project main`（本提交）。
- 边界：attention 设备共享 **离线 e2e + 机制 + 地基齐备，未接 live vLLM**；live 接线（worker `attn` op + 每层 KV 导出 + 窗口 A/B）按蓝图 S1-S4 推进，最大卡点 KV-rows ABI。



## 2026-06-25 —— Step3p5 BF16 0~47 vLLM-vs-PyPTO detail precision PASS ✅

- 在 `gpu-a910x-0162` isolated vLLM 容器中以 eager + all-to-all 路径采集真实请求 detail dump，checkpoint 为 `/mnt/nvme1/chensiyu/step3p5_flash_release_hf_mtp3_bf16`。
- PyPTO 侧新增逐层 detail 对比工具：主层 `tools/step3p5/pypto_all_layers_detail_compare.py`，MTP3 `tools/step3p5/pypto_mtp3_detail_compare.py`，以及对应 ST。
- 主模型 `0~44`：`3960` checks PASS，worst pass rate `0.9995659589767456`；MTP3 `45~47`：`279` checks PASS，worst pass rate `0.9995659589767456`。
- 组合 ST：`tests/step3p5/test_step3p5_all_layers_detail_st.py tests/step3p5/test_step3p5_mtp3_detail_st.py` → `2 passed in 286.34s`。
- 关键修复：`Step3p5 EPS = 1e-5`（对齐 vLLM `GemmaRMSNorm`）；MoE reference 使用 vLLM fused router dump 的 `topk_ids/topk_weights`。
- BF16 回归数据已打包为 `/mnt/nvme1/chensiyu/logs/step3p5_910b_v017/step3p5_bf16_e2e_st_regression_20260625.tar`，包含 coarse golden、全层 detail、MTP3 detail、final logits artifacts 与报告。
- 本次涉及仓库 commit 组合：`pypto-lib d4c01b9`、`pypto-project b771c7e`（本次文档记录提交，后续文档补记会前进）、`pypto b00c8b23`、`pto-isa e25732f0`、`PTOAS da011a3d`、`simpler c66b4120`。
- BF16 tar SHA256：`bce502f4cbafb61fe541385ab1828d33a1f9c32bdfb7d2009e871adba4c896c4`。



## 2026-06-24 —— Final e2e precision readiness preflight landed 🟡

- 新增 `pypto-lib/tools/step3p5/e2e_precision_readiness.py`，作为最终端到端精度验收的前置门禁。
- 当前 host 级整网 smoke 全绿：`decode_fwd` distributed mock worst pass rate 1.0；`step3p5_decode` synthetic smoke pass rate 1.0。
- 预检明确剩余阻塞：真实 checkpoint 未挂载、vLLM/stepcast oracle 不可见、`Step3p5DecodeFwd.host_orch` 未接 45 层、head_gate parity 策略未定、MoE 8 卡缺 golden 精度。
- pypto-lib pin 更新到 `stepfun/develop:cfe2093`。

## 2026-06-24 —— CANN 9.0.0 non-GA + DecodeLayerMoE 8 卡 ST runtime PASS ✅

- **环境升级**：0162 切到 CANN 9.0.0 non-GA/non-beta，`/usr/local/Ascend/cann` 指向 `/mnt/persist/Ascend/cann-9.0.0/cann-9.0.0`；已重编译 pypto 与 runtime。
- **回归**：`_smoke_program_build` 通过；dense full ST 8.54s PASS；dense SWA ST 15.61s PASS；L3 allreduce 1 passed / 1 skipped。
- **MoE 8 卡**：复现 `507018 / sched_error_code=100` 后重新切分定位，`dispatch-only` PASS、`dispatch+routed` FAIL，最终确认 routed expert 对 `tile_valid <= 0` 的空 tile 仍提交 kernel。加 `if tile_valid > 0` guard 后，`DecodeLayerMoE full_silu_silu --world-size 8` runtime PASS 26.51s。
- **边界**：MoE ST 当前验证 runtime，不带 golden 精度；整网端到端精度对齐仍属于 Phase 20/21 下一步。split dispatch 先保正确性，非 split/fusion 恢复归 Phase 22 perf 优化。

## 2026-06-22（晚） —— 项目跟踪仓库建立 ✅

在 `<dev-host>/data/chensiyu/hw_project/pypto/pypto-project/` 建了
`pypto-project` 作为专属跟踪仓，push 到 `csy0225/pypto-project`（私有
fork-style）。散落 doc 迁移：

- 把 Phase 20/21/22 docs + archive 内容从 `pypto-lib/docs/step3p5/`
  （位置错了 —— 这些是跨仓库议题）迁到 `pypto-project/phases/` +
  `archive/`。
- 写了新顶层入口文档：README.md、STATUS.md、CLAUDE.md（slim）、
  blockers.md。
- 外部 tracker `<workspace>/pypto/CLAUDE.md`（594 行 monolith）退休 ——
  被本仓取代。

**解决**：项目 owner 提的 doc 散乱问题。项目状态 SSOT 现在落在本仓。

## 2026-06-22（下午） —— WIP push 拆分 + dev-workflow docs + Phase 20-22 设计 ✅

### WIP push 拆分

3 个 commit 上 fork csy0225：

- `csy0225/pypto-lib stepfun/develop`: `ffaf5d6 → 73dbd12`
  （tests/step3p5/ 12 个 ST/UT 脚手架 + 中文架构指南，+3381 行）
- `csy0225/pypto-lib wip/step3p5-barrier-allreduce-20260622`: NEW
  `b5bb6ee`（4 文件 -267/+181：barrier-style all_reduce + per_rank
  输入广播）
- `csy0225/pypto stepfun/develop`: `03136bf6 → b00c8b23`
  （10 个 full_rope SSA/scheduling debug repros，+2199 行）

**关键决策**：WIP barrier all_reduce **不进** `stepfun/develop`（会让
dense ST device 0 编译退化 by UB overflow）。侧分支保留意图待后续。

### Dev workflow + pitfalls docs（push: `73dbd12 → a6b5faa`）

- 新增 `pypto-lib/docs/known-pypto-pitfalls.md` §7：
  `pl.range(constant)` 展开不复用 SSA buffer → UB overflow（barrier
  all_reduce blocker 根因 + 3 个 avoidance recipe）。
- 新建 `pypto-lib/docs/dev-workflow-gotchas.md`：5 条 catalog 非 pypto
  workflow 时间坑（stale pyc / 三件套 activation / HTTP/2 timeout /
  netboot SSH / gh CLI 缺席）。

### Phase 20-22 设计落地（push: `a6b5faa → 69f22b1`）

3 个 phase doc，每个 ~200-300 行。这些 doc 后来移到本 `pypto-project`
仓（见上面晚段）。

## 2026-06-22（早） —— 0162 重启后恢复 + 重验 + MoE 507018 复现 ⏸

### 重启后环境恢复

`gpu-a910x-0162` 重启过；三剑合璧都活着（driver 25.5.2、firmware
7.8.0.7.220 chip flash、CANN 9.0.0-beta.1 NVMe symlink）。4 个 git 仓
都在期望 HEAD 上，simpler submodule `a6e06406`。

### Smoke probe 红鲱鱼（已解）

第一次 `python -m models.step3p5._smoke_program_build` 返回 rc=1，
attention_swa.py:396 报 `valid_cols (48) exceeds bound 16`。**根因**：
上次 session `apply_perrank_patch(TP=2)` 实验留下的 stale
`__pycache__/config.cpython-311.pyc`。Python 的 pyc 失效检查只比 source
mtime，不比 module dict 值。

**解决**：`find models/step3p5 -name "*.py" -exec touch {} +` 把
source mtime 顶过 pyc → pyc 失效 → fresh import 读到正确 `TP=8`。归到
workflow gotcha §1。

### 验证基线

| 测试 | 状态 |
|------|------|
| simpler L3 allreduce_distributed -d 0-1 | ✅ `max\|out-expected\|=0` |
| Phase 19 ST-1 full dense | ✅ PASS 7.93s |
| Phase 19 ST-2 swa dense | ✅ PASS 14.85s |
| MoE 6 variants smoke | ✅ 6/6 PASS |
| MoE device runtime（full_silu_silu -d 0） | ⏸ 5s 内 507018 fault |

记到 blocker §2；需要 `P19_DISPATCH_LIMIT` dispatch-cut tool 定位。

## 2026-06-20 —— 5 仓库 rebase 到 origin/main + push fork ✅

把 pypto / pypto-lib / pto-isa / PTOAS / simpler 全 rebase 到
`origin/main`。Audit：

- 4 个 simpler 本地 patch（zero-size view + `--no-as-needed` libhcomm
  + IPC ENABLE_PEER_ACCESS + SDMA_OFF + llvm-strip）都还要保 ——
  上游本周期没 subsume 任何一个。
- 6 个 pypto-lib step3p5 commit 都要保。
- 3 个 pypto commit（DFX env hook + repros + submodule pin）要保。

**结果**（push 到 `csy0225/`）:

- pypto: `926941e0 → 03136bf6`
- pypto-lib: `93826904 → ffaf5d69`
- pto-isa: `109c9f72 → e25732f0`
- simpler: `c66b4120 → a6e06406`

0162 上验证：smoke probe rc=0，simpler L3 allreduce 双卡 golden，
ST-1 dense device PASS，MoE 6/6 smoke PASS。

**Rebuild trap**：`pip install -e .` 第一次失败 due to
`tensor.h:535 buffer_elems` `-Werror=unused-variable`（NDEBUG +
release flag）。修法：别传 `CMAKE_BUILD_TYPE`（用 dev default）。

## 2026-06-19 —— Phase 16 多卡 IPC blocker RESOLVED ✅

`support_shmem_map_exbus=0` cap（filed as simpler#1037）是 driver 能力
缺口。解决要三剑合璧：

1. Driver `25.0.rc1.2 → 25.5.2`
2. Firmware `7.7.0.3.220 → 7.8.0.7.220`（chip flash，持久）
3. CANN `9.0.0-beta.1`（NOT GA —— GA 的 TDT 不推 AICPU
   `libaicpu_extend_kernels.so`，让 simpler init 507018 失败）

加 simpler `comm_hccl.cpp` patch（CANN GA forward-compat alias）。

**Traps**:

- CANN GA vs beta.1：3+ 小时浪费在 GA 上才发现。
- 0162 是 netboot/tmpfs：`/usr/local/Ascend/`、`/etc/`、`~/.ssh/` 重启
  全丢。建 `RECOVERY.sh` 幂等恢复；持久 state 在 NVMe `/mnt/persist/`。
- Kubernetes DaemonSet（`device-plugin`、`npu-exporter`）占着 driver
  `.run --upgrade`。`kubectl drain` 不够 —— 必须 `systemctl stop kubelet`
  + 手动 kill。

**验证**：`aclrtIpcMemImportByKey + ENABLE_PEER_ACCESS` 跨卡 rc=0、
`peer_va == parent ptr`；simpler L3 `allreduce_distributed` 双卡
`max|out-expected|=0` golden。

**0234 路径**：只需升 driver+firmware（CANN 已经对）。`.run` 包 stage
在 0162 `/mnt/persist/ascend-staging/`。归到 blocker §5。

## 2026-06-17 —— Phase 19 MoE blocker 1-4 清掉 + dense ST device PASS ✅

详见 [`prototype-phase-01-19-summary.md`](prototype-phase-01-19-summary.md)
"Phase 19 MoE blocker 解决"。MoE device runtime 507018 仍在（blocker
§2）。Dense ST device 0 通过（full 7.93s，swa 14.85s）。

## 2026-06-15 —— Phase 15 单卡 e2e rc=0 ✅

单 rank decode_layer 端到端跑通 device 0，20 个 dispatched task 完成。
三个层叠修复一起：head_gate ×1 旁路 + `--tp-world-size 1` monkey-patch
+ `LAYER_*_ROWS_DYN` override。`next_hidden_out shape=[1, 16, 4096],
max|value|=0`（dummy zero weight 期望零输出）。Run time 6.69s。

---

## Pin snapshot 历史（降序）

| 日期 | 事件 | pypto | pypto-lib | pto-isa | PTOAS（src） | simpler | ptoas-bin |
|------|------|-------|-----------|---------|--------------|---------|-----------|
| 2026-06-25 | Step3p5 BF16 0~47 detail precision PASS | `stepfun/develop:b00c8b23` | `stepfun/develop:d4c01b9` | `stepfun/develop:e25732f0` | `stepfun/develop:da011a3d` | `c66b4120` | `v0.45` |
| 2026-06-24 | CANN 9.0.0 non-GA + DecodeLayerMoE 8卡 ST | `stepfun/develop:b00c8b23` | `stepfun/develop:cfe2093` | `stepfun/develop:e25732f0` | `stepfun/develop:da011a3d` | `c66b4120` | `v0.45` |
| 2026-06-22 晚 | pypto-project 仓建立 | `develop:b00c8b23` | `develop:9c4773f` | `develop:e25732f0` | `develop:da011a3d` | `a6e06406` | `v0.45` |
| 2026-06-22 下午 | Phase 20-22 设计 + dev-workflow docs | `develop:b00c8b23` | `develop:69f22b1` | `develop:e25732f0` | `develop:da011a3d` | `a6e06406` | `v0.45` |
| 2026-06-20 | 5 仓 rebase + fork push | `develop:03136bf6` | `develop:ffaf5d6` | `develop:e25732f0` | `develop:da011a3d` | `a6e06406` | `v0.45` |
| 2026-06-19 | Phase 16 三剑合璧验证 | `main:a1b066df` | `main:9c5593fb` | `main:109c9f72` | `main:29a8af28` | `afb5c5a9` | `v0.44` |
| 2026-06-17 | Phase 19 blocker 1-4 清掉 | `main:3f421313` | `main:08f71692` | `main:8e436661` | `main:a1efed75` | `6e84154d` | `v0.43` |
| 2026-06-15 | Phase 15 单卡 e2e rc=0 | `main:3f421313` | `main:af4b2ed5` | `main:12e766d1` | `main:5392d5da` | `6e84154d` | `v0.43` |
| 2026-06-05 | Phase 13 re-sync + smoke 绿 | `main:3f421313` | `main:08f71692` | `main:8e436661` | `main:a1efed75` | `6e84154d` | `v0.43` |

---

## 已解 blocker（post-mortems）

### 2026-06-22 —— simpler#1018 libhcomm DT_NEEDED ✅

`comm_init` 段错 —— `hccl_comm.h` 把 HCCL 声明为 weak，x86 默认
`--as-needed` 把 `libhcomm.so` 从 `DT_NEEDED` 删了。修复在 simpler
`a6e06406`：`src/{a2a3,a5}/platform/onboard/host/CMakeLists.txt` 把
`${HCCL_LINK_TARGETS}` 包成 `-Wl,--no-as-needed ... -Wl,--as-needed`。

### 2026-06-19 —— simpler#1037 IPC support_shmem_map_exbus=0 ✅

三剑合璧修复（driver 25.5.2 + firmware 7.8.0.7.220 + CANN beta.1）。
详见上面 2026-06-19 milestone。

### 2026-06-17 —— Phase 19 blocker 1-4 ✅

1. PTOAS v0.44 `pto.tci ui32 {descending=false}` parser：上游 v0.45 fix
   `505abd64`。
2. sh_mlp / gate_matmul L1/UB overflow：是 shape-choice artifact
   （`apply_tp1_patch` 错，`apply_perrank_patch` 对）。
3. dispatch.py 32B 对齐：`PER_RANK_BUCKETS = pad8(...)` 跨 5 文件
   mirror。
4. CCEC bf16 类型转换：`expert_weights` BF16 → FP32 跨 6 个 emission 点。

详见 [`prototype-phase-01-19-summary.md`](prototype-phase-01-19-summary.md)
"Phase 19 MoE blocker 解决"。
