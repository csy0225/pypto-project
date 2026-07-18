# 实时状态

pypto step3p5 项目的实时状态板。**任何 phase / sub-task / blocker 状态
变化都更新这里**。历史细节查 [`archive/`](archive/)。

**最后更新**：2026-07-18

> **0162 clean 环境唯一 stable 版本记录**：
> [`develop/N1/N1-STABLE-ENV-0162-20260717.md`](develop/N1/N1-STABLE-ENV-0162-20260717.md)。
> checkpoint SHA256、pip freeze、三仓/runtime binary、PTOAS/CANN、设备和
> canonical 命令全部内嵌在该单一文档中。

> **2026-07-18 ✅ N1 whole-net rank-local single-submit 已在 0162 通过 canonical P42 20/20**：
> - **当前 stable 对象**：`models.step3p5.decode_layer_single_chip:
>   whole_decode_faithful_real_single_chip`，dispatch pull + combine pull，
>   native W8A8 IPC weights，KV IPC，token 6127。
> - **代码已提交并推送到 fork**：
>   pypto `n1fusion-base`
>   `e49ce111c1503f4fb3e898af4223560cab907a62`；
>   pypto-lib `feat/whole-net-n1-fusion`
>   `3af13f4facbe8db5cd4a6c769e8b9e07e351c7b9`；
>   simpler/runtime 仍为 `36957c6b56700ecba3aeb8dbbedd6240594e01de`。
> - **single-submit 证据**：生成物
>   `WholeDecodeFaithfulRealSingleChip_20260718_105511/orchestration/host_orch.py`
>   只有一个实际 `_submit_chip(whole_chip_orch, device=r)` 调用点；TP=8 下是
>   8 个 rank-local submission，不再是 46 个 layer-level submission/rank。
> - **canonical 20-run**：
>   `workspace/logs_n1/single_submit_cleanup_p42_repeat20_20260718_174948`，
>   `rc=0`、`pass=20/20`、每次 `argmax=303`、
>   `TOP5=[303,9592,1043,768,2086]`，
>   runtime min/mean/max=`0.5536/0.6607/2.2766s`，fingerprint 唯一；
>   20 个 worker-run dmesg 窗口新增 relevant=0。
> - **纠偏**：2026-07-17 64K synthetic 记录里的 `46 dispatches/rank` 是旧
>   main decode 版本的 L3 host→chip control dispatch 数，现在只能作为历史性能
>   对照；当前 main decode 已改为 rank-local single-submit。EP/MoE 内部的
>   `_dispatch_pack_publish/_dispatch_pull/_dispatch_stage` 仍是 MoE 数据重排任务，
>   不与 host submit 计数混淆。
> - **保留诊断边界**：P1 diagnostic 显示 single-submit 与旧 baseline 在 padding /
>   physical rows 的 routed path 上不逐元素等价，但 canonical row0 argmax 不受影响；
>   该诊断不是 release PASS，也不是当前 stable 的反证。若后续追求 padding-row
>   完全等价，应从 `local_routed_x_scale` / route metadata 继续排查。

> **2026-07-17 latest MTP3 synthetic-64K-current-position resident NPU timing（non-canonical）**：
> - 0162 `pypto-lib` 已同步到
>   `53a67326ec419b949657c25bc9ebb35d49c14d27`，latest pipeline 为
>   `whole_decode_faithful_real -> host sampler -> whole_mtp3`。
> - 用户允许跳过真实 64K prefill；本次精确定义为
>   `seq_len=65536/current_position=65535`，synthetic zero-init BF16 KV，
>   fixed `BATCH=16`、仅 row0 meaningful。不是 65,536 个历史 token 后的
>   position 65,536，也不是 canonical 精度测试。
> - resident runtime（prepare/register 一次）复测：main warmup 2 +
>   5 rounds，MTP3 warmup 2 + 10 rounds。只读取 NPU `[STRACE]`：
>   main `device_wall median=456.952ms`、`effective median=383.609ms`；
>   MTP3 `device_wall median=11.757ms`、`effective median=11.401ms`。
> - 当前 main 一步是 `46 dispatches/rank`，MTP3 是 `3 dispatches/rank`。
>   main + MTP3 的独立中位数算术估算为 `468.709ms` NPU device cost；
>   因两 program 间有 host sampler，它不是统一 device clock 的 pipeline wall。
> - `46` 的精确来源是 L3 generated host-orch 的 `45 layer-level
>   _submit_chip + 1 lm_head`；这是 host-orchestration → rank-local chip
>   callable 的 control dispatch，不是 MoE EP token dispatch。EP 的
>   pack/pull/stage/combine 位于每个 MoE layer callable 内部。
> - 旧的 `runtime.run=2.79s/1.98s/4.77s` 是 outer host wall，禁止再当 NPU
>   inference latency；旧 cold MTP3 proxy `44.327ms` 也已被 resident
>   `11.757ms` 替代。
> - worker dmesg 窗口 relevant=0；exporter teardown 后 outer 窗口新增两条
>   `stranded cqe`，归 cleanup。cards 8..15 已清空，0..7 baseline 未动。
> - resident exporter map 同时确认临时 benchmark 的 MTP cache 物理 entry
>   比 program-visible prefix 大 45 倍；不影响本次访问边界，但不能代表
>   production compact MTP cache。temporary 64K ABI/harness 未合入 canonical。
> - 唯一 stable SSOT §11 已修正。resident NPU 证据：
>   `workspace/logs_n1/synth64k_resident_20260717/`。

> **2026-07-16 runtime manifest 审计 + clean-pin formalize**：
> - **旧 20-run 的真实范围**：`pypto-lib 0e7a0fdd` 模型源码 exact-match；
>   pypto 当时为 `5e619dc7 + dirty runtime source`，simpler 当时为
>   `98ce22a6 + dirty runtime source`。因此不得把旧 20-run 写成三仓 clean
>   pin 上直接完成。
> - **相关 standalone 依赖已提交并推送**：
>   pypto-lib `0e7a0fdd`、pypto `e277de9f`、simpler `36957c6b`。
>   0162 三个 release 工作树均 clean；本地 live/debug WIP 仍隔离存在，未混入
>   standalone release。
> - **最终 clean-pin smoke**：0162 日志目录
>   `workspace/logs_n1/release_manifest/final_stack_smoke_20260717_015635`
>   （目录名来自 0162 机器时间），固定 P42 pull+pull、token 6127、
>   native W8A8 IPC/KV IPC；`rc=0`、`RUN done 2.58s`、`argmax=303`、
>   `TOP5=[303,9592,768,1043,410]`。worker dmesg 窗口新增 relevant=0；
>   outer 窗口在 exporter teardown 后新增 1 条 dev14 `stranded cqe`，归 cleanup。
> - **0234 证据边界**：项目既有记录称该机只同步 pypto-lib 后 fresh canonical
>   3/3 stall，但本次 `ssh infra@gpu-a910x-0234...` 返回
>   `Permission denied (publickey,password)`，无法独立复核三仓、runtime
>   binary 和环境。**只拉 pypto-lib 不是同一测试对象**，不得由此直接断言
>   512B 在 0234 失效或已得到跨机器单变量结论。
> - **live 组件边界**：本地 holder/sidecar/KV importer/backend 仍属于独立
>   live WIP，未包含在 standalone 三仓 release 中；live token-exact A/B、
>   per-layer KV 与 redundant-weight HBM 仍未完成。

> **2026-07-16 ✅ Phase 27 standalone canonical stall gate 关闭；Phase 28 live serving 仍进行中**：
> - **发布代码**：pypto-lib `feat/whole-net-n1-fusion` commit
>   `0e7a0fddc90c4f2348f1d59e015fb817a0877a02`
>   （`fix(step3p5): stabilize N1 pull MoE with isolated control signals`），已推送
>   `csy0225/pypto-lib`。
> - **固定准出组合**：gpu-a910x-0162、devices 8..15、`whole_decode_faithful_real`、
>   `P_FAITHFUL_MOE_LAYERS=42`、token 6127、native W8A8 IPC weights、KV IPC、
>   **dispatch fixed-slot pull + combine pull**。
> - **canonical 稳定性与精度**：release commit `0e7a0fdd` exact-source 在
>   fresh exporter pool 连续 **20/20 PASS**，每次 `argmax=303`，runtime
>   min/mean/max=`2.50/2.5605/2.62s`；20 次数值指纹一致。
>   日志：`workspace/logs_n1/signal512/signal512_p42_20_20260717_001135`
>   （0162 机器时间标签）。
> - **整理后复验**：`signal512_final_smoke_20260716_230225`，
>   `2.57s, argmax=303, FINAL_SMOKE=PASS`。exact-source 20-run 的 20 个逐
>   worker-run dmesg 窗口均无新增 fault、`507018`、`running-stalled` 或
>   `stranded CQE`；fresh exporter pool teardown 后 outer 窗口新增 2 条
>   `stranded cqe`（dev8/dev11 exporter），已明确归入 cleanup 边界。
> - **最终最小 layout A/B**：control signal logical view 仍为 `[8,1] INT32`
>   （32B），physical allocation 从 32B 改为 **512B**；216/216 signals 为 512B，
>   相对 offset 全部 `%512=0`。在 0162 上它与历史随机 stall 消失强关联，
>   但不是 matched 单变量因果证明，也不写成跨机器充分条件或某个 signal bit /
>   PUSH/TPUT 已被硬件层证明为唯一根因。
> - **架构边界**：dispatch 生成 `recv_counts + inverse_map`；combine 直接消费
>   dispatch-produced `inverse_map`；self 用 local load、peer 用 remote load；
>   per-layer distinct communication buffers、signed tail、native INT8
>   gate/up/down 全保留，禁止 BF16 dequant 权重回退。
> - **generator gate**：真实 strip → regenerate → byte-compare 已通过：
>   `PRECOMMIT_ROUNDTRIP=PASS`, `ROUNDTRIP_CMP_RC=0`。
> - **排查方法沉淀**：新增
>   [`.claude/skills/pypto-whole-net-hang-debug/`](.claude/skills/pypto-whole-net-hang-debug/)
>   ，包含通用 stall 分类、TASK/CLUSTER/COND、task→kernel→生成源码映射、
>   可选 AICore PC 路径、buffer/layout 审计、证据解析脚本和本次 N1 完整案例。
> - **状态边界**：0162 standalone release gate 已关闭；但项目记录中在
>   pypto-lib 三个 release 文件与 `0e7a0fdd` byte-match 后的 0234 fresh
>   canonical 3/3 stall 仍是独立 active blocker，
>   本次 session 无法通过 SSH 独立复核。Phase 28 live serving 仍另有
>   per-layer paged-KV bridge、重复权重/3-way HBM 和 token-exact A/B。
> - 唯一 standalone 测试口径见 [`N1-CANONICAL-TEST.md`](N1-CANONICAL-TEST.md)；
>   下一会话边界见 [`NEXT-SESSION-N-1.md`](NEXT-SESSION-N-1.md)。

> **2026-07-15 (续²·live single-handoff 实测) 🟡 live 单-handoff plumbing 打通 + co-tenancy patch live 验证；⛔ 3-way HBM 墙（redundant weights）挡住 token-exact [Phase 28]**：
> 本 session 推到**真机 live 单-handoff 实测**，拿到里程碑 + 一个 device-proven 硬 blocker：
> - **✅ live plumbing 端到端打通**：patched vLLM 8001 full-mode（`util 0.57 + --num-gpu-blocks-override 256`，与 8/8 exporters 共存，H=200）→ 请求 `[6127]` 经 `_pypto_full_forward` → AF_UNIX socket → sidecar whole-net holder（8 chips ready、`SIMPLER_COMM_NO_HCCL` bootstrap 干净、权重+KV via IPC、46 args、listening）**全链路路由通**。
> - **✅ co-tenancy patch live 验证**：sidecar 与真 vLLM 同卡共存，`SIMPLER_COMM_NO_HCCL` 无 HcclCommInit 冲突（此前仅 standalone bootstrap 验证，本次真 vLLM 共存坐实）。
> - **✅ vanilla 参照 live**：vLLM 8001 `[6127]` → "，"(303) —— 金标准在 0234 live 确认。
> - **⛔ blocker（device-proven）= 3-way HBM 墙**：whole-net `rt.run` OOM `207001 (size≈16GB)` —— vLLM(25G, W8A8 weights 24G) + exporters(26G, whole-net INT8 weights) + whole-net-run(16G) ≈ 67G > 64G/卡。**根因 = 权重双份冗余**（vLLM 与 exporter 各存一份 step3p5 全模型）。token-exact 303 **未达成**（run 前 OOM，非 hang）。
> - **vLLM 两道内存检查坑（记下）**：`util*total ≤ free`（exporters 占 26G → free 35G → util≤0.577）**且** `util*total > resident_used`（→ util>0.55）→ **窗口极窄 ~0.56–0.57**；再 `--num-gpu-blocks-override` 强制小 KV。util 0.9 挂 check1，util 0.3/0.4 挂 check2。
> - **修法（下一步，gated）**：**HBM 墙 device 二次坐实 = redundant weights，非单纯 ring 旋钮**：`RING_HEAP=4G` run-alloc 16G→OOM；`RING_HEAP=2G` run-alloc 缩到 2.4G（确认 alloc=4×RING_HEAP）**但仍 207001**——whole-net **cumulative** footprint（ring 2G + per-layer comm windows ~3.5G + dep/task pools + activation）> vLLM(25G)+exporters(26G) 后剩的 ~13G free。**根因 = 权重双份**（vLLM W8A8 24G + exporter INT8 26G = 同模型两份，占 51G）。**真修法 = 消除双份**：whole-net sidecar 不自带 26G exporter，改**读 vLLM 常驻权重 via IPC**（whole-net INT8 stacked 布局 ≠ vLLM W8A8，是多日级 model-side 改）；或激进缩 whole-net working（task_window/dep_pool，风险 deadlock）。本段是 2026-07-15 live 实验记录；其 standalone stall 状态已被 2026-07-16 的 20/20 结论覆盖。**净：live 单-handoff plumbing/co-tenancy/IPC 全通，token-exact 当前卡在 redundant-weights HBM 与 per-layer KV/live A/B。**
> - **清理干净**：sidecar+vLLM killed、exporters 8/8 存活、卡回 44%、207001 未 force-reset（无 poison）。

> **2026-07-15 (集成 groundwork，历史里程碑) 🟡 N=1 whole-net → vLLM live single-handoff 组件落地 [Phase 20/M5-M6]**：
> 本段只记录 live 集成组件的形成过程；当时尚未完成的 standalone 稳定性状态已经被
> 2026-07-16 的 canonical P42 20/20 结论覆盖，不再作为当前 gate。
> - **`tools/step3p5/whole_decode_holder.py`（新）= `WholeDecodeHolder`**：把 harness 的 compile→prepare→import_weights→rt.run 抽成常驻 holder（build+prepare 一次，`run()` 复用 rt）。arg 顺序逐字对齐原 harness，device compile OK。
> - **`tools/step3p5/whole_decode_sidecar.py`（新）= sidecar serve + AF_UNIX 协议**：常驻 holder + length-prefixed self-describing 帧（JSON header + bf16-safe tensor blob）+ `WholeDecodeClient`。serve loop 解耦为 `decode_fn`。**offline `--selftest` round-trip PASS**（协议/bf16 编解码/client-server，无 device）。
> - **`tools/step3p5/vllm_monkey_patch.py::_pypto_full_forward`（改）**：从 fail-closed 改为经 `WholeDecodeClient` 驱动 sidecar；collective-consistent fallback（sock absent 全 rank 一致回退，避免 broadcast 死锁）+ rank-0 drive + TP-broadcast replicated next_hidden。**compile OK**（live 验证 gated）。
> - **`pypto/runtime/.../comm_hccl.cpp`（改）= `SIMPLER_COMM_NO_HCCL` co-tenancy patch**：comm_init 跳过 HcclGetRootInfo/HcclCommInitRootInfo（保 run_token+file_barrier）、comm_barrier null-comm no-op、alloc_domain 放宽 null 检查；flag-OFF 无回归，flag-ON bootstrap 通过，后续真 vLLM 共存也已验证。
> - **`tools/step3p5/pypto_whole_decode_backend.py`（新）= 容器 self-contained full-mode backend**：内联 BATCH/HIDDEN + 内联 socket client（与 sidecar 协议逐字一致）+ `_full_forward`（collective fallback + rank-0 drive + broadcast）+ tail compute_logits + `install/uninstall/status/maybe_autoload`（sitecustomize `PYPTO_WHOLE_DECODE=1`，容器无 pypto-lib mount 用）。**offline py_compile + status() PASS**（vLLM step3p5 可 import）。
> - **剩余 gate（device/live）**：(1) whole-net 从共享 ctx=1 KV 改为 per-layer paged-KV bridge，并传真实 attention metadata；(2) 消除 vLLM/exporter 冗余权重，解决 3-way HBM；(3) 完成 live A/B token-exact。当前不得声称 serving 集成已经完成。

> **2026-07-12 (续¹²) ✅✅✅ Blocker B 解除：N=1 整网 42 层真 W8A8 IPC device 跑通干净（3.48s）[Phase 27]**：
> **文档假设（Blocker B = IPC VA 冲突）被 device 证伪，真根因 = gate_topk mrgsort 级联 bug。**
> - **VA 证伪**：在真正的 MoE 路径 `comm_hccl.cpp:703 domain_alloc_via_ipc` 加 VA 诊断（旧诊断加错在 `:431` 未走路径 + `LOG_INFO_V0` 被默认 `info_v=5` 压掉；`logging.getLogger("simpler").setLevel(15)` 放开）→ device 实测：42 层仅 1 个 MoE comm domain window 在 `0x12c041600000`+396MB，8 卡一致，**整段在池 `0x12c1c0000000` 下方无重叠**。IPC 映射表（`pypto_weight_map.rank0.json`）48 key 全 512 对齐/无重叠/`max_end==pool 25.35GiB`/都在池内——VA 表正确。
> - **真根因**：device stall 快照（`ASCEND_GLOBAL_LOG_LEVEL=1`+`ASCEND_PROCESS_LOG_PATH`）= `task_id=3 state=RUNNING kernels=[aiv0:3] core=28 fanin 3/3` 永不完成 = **`gate_topk` AIV kernel 挂死**（`orch_error=8` TENSOR_WAIT_TIMEOUT + S1 running-stalled）。gate.py SCORE_PAD=512 级联 `sort32→16×64 → mrgsort(block_len=64)→4×256 → mrgsort(srt[:,0:512],srt[:,512:1024])`——最后 format2 二路归并被喂两个"各含 2 段"的半块（非单段），违反 format2 前置 = SKILL "format2 半块未排序→状态机不终止→挂死"。**N≤2 clean / N=42 hang**（编译规模触发）。
> - **修复（对齐 DeepSeek v4 gate.py 渐进 format1 链，scale 到 512）**：format2 二路 → `mrgsort(block_len=256)`（4×256→1×1024 全排序）。gate.py + decode_layer.py 10 处去重内联 MoE gate 全改。commit pypto-lib **`4bede85`**（feat/whole-net-n1-fusion）。
> - **验证**：`_stage_whole_faithful_real_ipc -d0-7`（真 W8A8 INT8 IPC 池 25.35GiB/rank，全 42 MoE 层）→ **`REAL_WEIGHT_IPC_RUN_CLEAN` 3.48s，无 stall**。（输出为 0 因 harness 喂 dummy hidden/KV——device 执行路径已干净。）
> - **工具**：harness `--reuse-exporters`（8 exporter 常驻、survive force-reset、bisect 秒级 attach）。
> - **✅ KV cache 也走 IPC 已完成（commit `c61046b`，`--kv-ipc`）**：exporter 把 per-rank `k_cache`/`v_cache` `[4096,128]` bf16 carve 进同一 IPC 池；worker 把它们绑成 `add_inout` StackedDeviceTensor（attention 读 context + 写新 K/V 进共享 peer 内存），替换 dummy host tensor。验证：全 42 层真 W8A8，**权重 + KV 双双走 IPC**，8 卡 → `REAL_WEIGHT_IPC_RUN_CLEAN` 3.39s。**用户"KV+权重都走 IPC"硬约束达成。**
> **剩余**：整网精度 vs vLLM —— **vLLM W8A8 oracle 已在 0234 起来（2026-07-12，port 8000，`step3p5`，health=200，greedy chat golden = "嗯，用户问了一个非常基础的问题：…"，KV cache 26.9GiB，enforce_eager TP=8 EP `--quantization ascend`；注意 vLLM 要 clean CANN env——`acl` 在 `/usr/local/Ascend/cann/python/site-packages`，别被 pypto PYTHONPATH shadow）**。
> **✅ gate 修复已 device 数值验证（`b92031f`）**：`tests/step3p5/_probe_gate_sort.py`（sort-only device probe，绕开 gate_matmul 单卡 unsliced 溢出）→ **`DEVICE RESULT: PASS 7.33s`**，`expert_indices` topk_pair_compare + `expert_weights` ratio_allclose(6e-3) vs torch.topk。**本 session 唯一改动（gate_topk mrgsort）device 数值正确性坐实**（＋整网 device-clean ＋ analytically 完整 merge-sort ＋ DeepSeek 对齐）。
> **整网 token-exact A/B 仍未做（需要非本分支机器）**：N=1 harness 喂 dummy 零输入，缺 tensor-input decode runner + vLLM→whole-net KV bridge + decode-step golden（`test_vllm_golden_e2e_st.py:101` 自标 "decode runner not wired yet"；`decode_acceptance.py` torch ref 跳 attention+routed-MoE；无 standalone full torch forward）。这套是 G5b track 机器（0162 有 oracle+已 debug 的 KV/rope/attention 语义，正查 L17 blowup）。**从零在 N=1 分支重建 full torch reference 语义-mismatch 风险高（违反 correctness>speed）**——建议 port G5b decode-step-golden/live-A/B 或交 G5b track。本 session 改动的 accuracy 风险（gate）已 device 验证；其余网络本 session 未改、历史 model-logic 精度 0.9995。
>
> **2026-07-12 (续¹¹) ⭐⭐⭐ Stage C 达成：INT8-native W8A8 MoE-block device 精度 PASS vs torch W8A8（ccec 修复后）[Phase 27]**：
> 承续¹⁰（Stage B 到 device dispatch + 发现 ccec scale bug）。**修复 gap5_stagec ccec + device 精度验证通过**：
> - **ccec 修复**：dispatch-side `local_routed_x_scale` 从 `[LOCAL_RECV_MAX, SCALE_W_PAD=8]` 改**非 padded contiguous `[1, LOCAL_RECV_MAX]`**；`_expert_routed` 读 `[1,RECV_TILE]` ND2ND row-slice+reshape（DeepSeek recv_scale_dq 模式）；`dispatch_step` repack 用 scalar col-0 read un-pad（a2a 窗口仍 `[.,SCALE_W_PAD]` 保 32B）；`chip_orch` create `[1,LOCAL_RECV_MAX]`。a2a3sim `[14.E] OK`。commit pypto-lib `1379ce2`。
> - **device 精度 PASS**：`_stage_moe_block_precision --layer 3 --dev-offset 0 --ckpt <W8A8> --bypass-gate --torch-golden`（0234 8 卡，真 W8A8 INT8 experts，synthetic 输入）→ **`'moe_out' PASS ratio_allclose(atol=0.04, rtol=0.04, max_error_ratio=0.1)`**，`layer=3 target=ffn_out DEVICE: PASS (27.19s)`。→ **INT8×INT8 W8A8 MoE-block（gate/up/down INT8 matmul + col/row_expand dequant + h_i8 requant + dispatch-side per-token act quant）device 数值正确**（vs torch W8A8-dequant 参考，即 vLLM W8A8 同款数学），无 fractal-32 静默错。
> - **意义**：**精度验证达成**（MoE kernel 级，device-verified）。standalone MoE-block 用 H2D 权重→无 Blocker B，8 卡 EP dispatch collective 跑通干净（27s）。
> **剩余（full e2e）**：whole-net `whole_decode_faithful_real` full IPC e2e 仍卡 **Blocker B**（MoE collective × IPC 池 VA `0x12c1c0000000` = aclrtMalloc HUGE_FIRST 与 runtime arena 冲突；需 runtime VA-placement fix，深度、正交于 INT8）。vs-live-vLLM token-exact 需 W8A8 decode-step dump（0234 无 oracle）。
> **本 session 全 commits**：pypto-lib `cd3ef0d`(A)→`a293fe7`(A5)→`132fedc`→`32b59d3`→`6404385`→`6fe58ee`→`517dd6e`→`1379ce2`(ccec fix)；pypto-project 更新中。


> **2026-07-12 (续¹⁰) ⭐ Stage B device dispatch 到达 + INT8 集成正确性坐实；full-e2e 卡 pre-existing Blocker B；Stage C 缺 oracle dump [Phase 27]**：
> 承 A5（whole_decode_faithful_real INT8 编译通过）。机器 **0234**（经本地 tmux `pypto-ascend-0:0`，与 b-csy NFS 共享 → 我的 INT8 commit 直接可见；8 卡空闲；0162 被两个 vLLM 占满不可用）。
> **Stage B（`_stage_whole_faithful_real_ipc -d0-7`, INT8, heap=4GB）device 结果**：8 exporter 导 **INT8 池 25.35 GiB/rank**（vs 47GB BF16，**存储减半达成**）→ **compile OK** → 8 chip ready → **import_ipc_all OK** → rt.run 到达 device → **前 4 chip 执行完成**（`completed=4/32`：full_dense+2×swa_dense+swa_attn）→ **507018 scheduler-stall @ `stuck_task_id=0x100000003`（scope-1 task3）= 首个 MoE 层跨卡 dispatch collective**。= **pre-existing Blocker B**（MoE collective × IPC 池 VA 0x12c1c0000000 冲突），memory 已预言 INT8 不解此（与 INT8 正交：BF16 IPC 同样 stall）。cards force-reset 干净。
> **INT8 集成正确性坐实**：host loader INT8+scale load PASS（verify_bundle_shapes PASS）；device 到 rt.run + 4 chip 执行；修了 3 个真 bug——(1) scale `[..,N,1]`→squeeze `[..,N]`；(2) exporter map dtype + `_torch_dtype` 不认 int8（池字节对但元数据错→rt.run TypeError）→ 加 int8/fp16；(3) harness int8_routed + scale args。commits pypto-lib `32b59d3`/`6404385`。
> **剩余（均为环境阻塞，非 INT8 代码）**：① **full IPC e2e = Blocker B**（深度 runtime，需 VA-overlap 调查 / dispatch-cut bisect；H2D 替代不可行：200GB host RAM + 40min serial load，harness 已 deprecated）；② **Stage C 精度 = 0234 无 vLLM W8A8 dump**（旧 dump 在老 pod；需重生成 oracle 或 synthetic torch-INT8-golden 跑 `_stage_moe_block_precision`——harness 已 INT8-ready `6404385`）。
> **可跑的下一步**：Stage C 用 `_stage_moe_block_precision --torch-golden` + synthetic 输入（device INT8 MoE-block vs torch W8A8，验 kernel codegen/fractal-32，无需 dump）；或提上游/调查 Blocker B VA-overlap 解 full e2e。


> **2026-07-12 (续⁹) ⭐⭐ G5b token-garbage ROOT CAUSE 定论 + 修复已验证：seq_len=0 未用 batch 行 → NaN 污染 active row（非 SWA/MoE kernel bug）**：
> 承续⁶补。用 offline decode-step golden（prefill dump pos-17 逐 rank 真 KV + 真 per-layer rope + 真 W8A8 BF16-dequant，
> `_stage_whole_decode_run.py --golden-decode-pos 17`，cards 8-15）逐层 out-row0 对拍 vLLM golden，**决定性排除 SWA/MoE kernel bug**：
> - **L1-alone**（直接喂 golden layer_input[1] row0）→ **pass_rate=1.0 max|diff|=0.005** ⇒ SWA kernel 本身正确。
> - **`--golden-fill-batch`**（16 行全填 active token，无 seq_len=0 pad 行）跑 chain → **L0-L4 全 pass 1.0**（full/swa/MoE 全对，真 KV+rope+W8A8）。
> **⇒ 所有 kernel 正确；真根因 = 未用的 decode batch 行 seq_len=0**：whole-decode 编译成 BATCH=16 但单序列 decode 只 1 active 行，
> 其余 15 行 seq_len=0 → flash-attn `fa_ctx_blocks=ceil(0/128)=0` → Stage1-3 空循环不写、Stage4 读**未初始化 scratch** →
> `ctx=oi/li=NaN` → 该行 attn_out=NaN → 传入下一层 input → **跨层污染 active row0** → token garbage。
> **live 触发点** = `vllm_monkey_patch.py:303 _wd_pad_i32` 把 seq_lens pad 成 **0**。**为何从未发现** = swa ST（`test_decode_layer_swa_dense_st.py:288`）用 `seq_lens=torch.ones` 全 ctx=1 从不触发。
> **修复已 device 验证**：未用行 pad 成 **seq_len=1 + 非冲突 dummy slot（N+1+i，被 row0 valid_len 屏蔽）**→ chain **L0-L4 全 pass 1.0**（无 NaN）。
> **已排除**：rope/qk_norm/head-gate/KV-layout（DeepSeek 对齐）/full-attn/SWA/MoE kernel/合成 rope。
> **下步（task 8）**：production 修复 —— 稳健版 kernel guard（attention_full/swa 对 `fa_ctx_blocks==0` 输出 0）或轻量版 live client/sidecar/容器后端 pad seq_len=1；再 restart 8001 mode=full + sidecar + 3-prompt live A/B token-exact
> （现 whole-decode BF16-dequant ~0.9995 vs INT8 vLLM；INT8-native 严格出口在 N=1 track / gap-5，见续⁷/续⁸）。
> memory `g5b_swa_multientry_kv_nan_root_cause`（已修正为 seq_len=0 pad-row 结论）。机器：8001 停以腾 cards 8-15；8000 oracle cards 0-7 全程 200。

> **2026-07-12 (续⁸) ⭐⭐ A5 达成：whole_decode_faithful_real 整网 INT8-native W8A8 编译通过（TP=8）[Phase 27 / NEXT-SESSION-N-1]**：
> 承续⁷（moe.py standalone INT8 已过）。本轮把 INT8-native W8A8 传进 **N=1 整网程序** `whole_decode_faithful_real`（`decode_layer.py`，与 moe.py 解耦、自带 inlined 副本）。
> **手法（agent 三次 600s stall → 改直接编辑）**：range-scoped transform 脚本 `tools/step3p5/_a5_int8_transform.py` 只改 base `_build_whole_decode_faithful_program` 内的 inlined MoE（其余 10+ 程序副本不动）：`_expert_routed`+`expert_routed_step`+`chip_orch` 签名/调用+`host_orch` decls+**全 42 个 per-layer chip_orch 调用**全部穿 INT8+scale（每处 exact-match 断言）。**用 in-kernel per-token 输入量化（照抄 DeepSeek cast 链 FP32→INT32 rint→FP16 round→INT8 trunc）**，不动整网 push-based dispatch（数值 == dispatch-side，见 `gap5_quant_after_dispatch_equiv`）。改 `_gen_faithful_real.py`（emit INT8 `moe_w_*_r` + FP32 `_scale` decls + interleaved chip_orch call）→ 删旧 real builder → 重生成 `decode_layer.py`。
> **验证（0162, TP=8 canonical DistributedConfig）**：
> - `_probe_whole_faithful_canonical --layer-name whole_decode_faithful` → **`COMPILE OK`**（base faithful INT8）
> - `_probe_whole_faithful_canonical --layer-name whole_decode_faithful_real -d 0-7` → **`resolved program=WholeDecodeFaithfulReal` + `COMPILE OK`**（output `WholeDecodeFaithfulReal_20260712_072053`）
> → **整网 INT8 W8A8 完整过 distributed codegen（45 层 + attn + MoE）**。提交 pypto-lib `a293fe7`。
> **剩余（Stage B/C）**：Stage B = 8 卡 device e2e（harness `_stage_whole_faithful_real_ipc.py` 接 `int8_routed=True` exporter + args 加 scale 张量；INT8 池 ~24GB 消 arena-OOM；cards 8-15，真权重 load ~10min）；Stage C = vs vLLM W8A8 精度（需 decode-step golden 或 live A/B；纯 torch detail-compare 只验数学不验 kernel）。fractal-32 partial-tile（valid_shape<32 的 INT8 cube）精度风险留到 Stage B/C device 验证。

> **2026-07-12 (续⁷) ⭐ N=1 W8A8 INT8-native routed MoE kernel 移植完成 + standalone 编译验证通过 [Phase 27 / NEXT-SESSION-N-1]**：
> 目标 = routed MoE 从 BF16-dequant（存 47GB）改成真 W8A8 INT8-native（存 ~24GB INT8 + 片上 W8A8，数学照抄 DeepSeek v4 `expert_routed.py`、与 vLLM W8A8 一致）。**用户确认走 dispatch-side 量化（Option A = DeepSeek 精确对齐 + a2a 半字节 + recv_x 预 fractal 化最强避 gap-5）**。
> **已落地（pypto-lib `feat/whole-net-n1-fusion` @ `cd3ef0d`）**：
> - `moe.py::_expert_routed` → INT8 权重 + per-output-channel FP32 scale；gate/up `matmul(...,out_dtype=INT32)`+`matmul_acc` → `col_expand_mul(row_expand_mul(acc, act_scale[T,1]), w_scale[1,N])` dequant；中间 `h_i8` 照抄 DeepSeek cast 链（FP32→INT32 rint→FP16 round→INT8 trunc，`pl.at(CORE_GROUP)`）；down INT8×INT8+dequant。保留 step3p5 `[K,N]`+b_trans=False（loader 已 transpose，数学等价）+ 逐层 swiglu_limit + routing-weight 在 combine 施加。
> - dispatch-side 量化：`_quant_moe_input` → INT8 x + `[T,SCALE_W_PAD=8]` FP32 per-token scale；`_pack_send_payload`/`ep_all_to_all`/`dispatch_step` 携带并行 scale 窗口，**融进同一个 a2a barrier**。`chip_orch`/`host_orch` 穿 INT8 权重+scale+窗口，input-quant 改**无条件**（swiglu clamp 仍逐层 gated）。
> - `weight_loader.py int8_routed=True`（loader 半）：INT8 routed 权重 + `moe_w_{gate,up,down}_r_scale` keys。
> - codegen 坑修：`_quant_moe_input` = **scheduled InCore（`pl.range`）双输出 kernel**（绕 SplitIncoreOrch 嵌套-InCore + inline-splice 单返回）。
> **验证**：`_compile_moe -p a2a3sim --layer-idx {3,44}` 均 **`[14.E] OK`**（silu + swiglu7/16 编译 clean，0162）。
> **⚠ 关键架构发现（决定剩余工作量）**：N=1 程序 `whole_decode_faithful_real`（`decode_layer.py`）有**自己的 inlined routed-MoE 副本（BF16），与 moe.py 解耦**——`_gen_faithful_real.py` 只 text-transform 已存在的 `_build_whole_decode_faithful_program`，不从 moe.py 重新 inline。故 moe.py INT8 改动**不自动传到 N=1**。
> **剩余（A5，gate Stage B/C）**：改 base `_build_whole_decode_faithful_program` inlined chip_orch(+inlined `_expert_routed`/`dispatch_step`/`_quant_moe_input`)→INT8（照抄 cd3ef0d）+ 改 `_gen_faithful_real.py`（`moe_w_*_r` INT8 decl + scale 参/窗口 + chip_orch call）→ 重生成 decode_layer.py → `_probe_whole_faithful_canonical` smoke-compile；再 harness `int8_routed=True`。之后 Stage B（8 卡 device，INT8 池~24GB 消 arena-OOM）+ Stage C（vs vLLM W8A8 精度）。
> **环境**：0234 DOWN；device/编译移到 **0162**（b-csy NFS 无 python，rsync `models/step3p5/*.py` → 0162 `pypto-lib-n1` worktree 编译）。

> **2026-07-12 (续⁶) ⭐ G5b 数值 bug 定位大幅推进：dense/attention/KV 路径 device 端已证正确，bug 缩到 MoE/INT8 [攻坚 3 结构性排除]**：
> 用 **prefill dump 的 position 17** 构造 decode-step golden（1 token attend 18 KV，全 8 rank 齐），
> 给 `_stage_whole_decode_run.py` 加 `--golden-decode-pos N`（feed layer_input[N] row0 + 逐 rank 注入
> rope.k/qkv.v[0:N+1] 进 k_cache + seq=N+1/slot=N/block=0 + 对拍 device out row0 vs golden out[N]）。
> **device 结果（cards 8-15，真 W8A8 BF16-dequant，SIMPLER_COMM_NO_HCCL=1）**：
> - **L0 full_dense out row0 vs golden[17]：pass_rate=1.000000**（max|diff|=0.28，幅值 7.59 vs 7.88 匹配）
>   → **dense 全路径 device 端正确**：input-RMSNorm/qkv/qk_norm/rope/flash-attn(真 18-entry KV)/head-gate/
>   o_proj/dense-MLP 全对。**rope 之前是文档 #1 嫌疑，现证伪**（neox-64 partial，输入是 qk_norm(q) 非 qkv.q）。
> - **DeepSeek 对齐**：DeepSeek v4 `decode_attention_swa.py:198` 做同样的 `kv_cache_flat = reshape(kv_cache,
>   [B*BLOCKS*BLOCK_SIZE, HEAD_DIM])` + slot/block_table 索引；step3p5 whole-decode 的 flat `[num_slots,128]`
>   与其 `[num_blocks,block_size,1,head_dim]` **字节等价** → KV-layout 与 DeepSeek 对齐，非 bug 嫌疑。
> - head-gate 也已对（worker 算 gate_r，on-device N=16 matmul_acc 已删）。
> **→ token-garbage 根因缩到：MoE 层（L3-44，42/45）/ INT8-vs-BF16 / 跨层累积**。正跑 **45 层 golden chain
> bisect**（逐层注入 KV + 逐层 out row0 对拍）定位首个偏离层。复现器/patch 在 0162
> `_stage_whole_decode_run.py`（`--golden-decode-pos`，备份 `/tmp/_stage_wd.bak_ml_*`）。**机器**：vanilla 8001
> 已停以腾 cards 8-15 跑 offline golden；8000 oracle cards 0-7 全程 200。
>
> **续⁶补（root cause 已定位到 SWA，非 MoE）**：45 层 golden chain bisect（真 KV + **真 per-layer rope**
> build_llama3_yarn/build_plain 注入后）结果：**L0 full_dense pass_rate=1.000000（max|diff|=0.007，真 rope）
> → full attention 全对；L1 swa_dense = NaN（row0），并向后传播 → L2-44 全 NaN**。即**首个偏离层 = L1
> （sliding-window attention），根因 = SWA 在 multi-entry KV（ctx>1）下 device 产 NaN**。**为何从未发现**：
> `test_decode_layer_swa_dense_st.py:288` 用 `seq_lens=torch.ones`（ctx=1，只测 1-token self-attn），
> **SWA multi-entry KV 路径从未 device 测过**；真 decode ctx 递增 → 33 个 SWA 层全 NaN → **就是 token-garbage
> 根因**。已排除：rope/qk_norm/head-gate/KV-layout/full-attn/合成 rope 溢出（真 rope 仍 NaN）。**待定位**：
> attention_swa.py Stage 1-4 里 SWA-specific（SWA_WIN_BLOCKS/SWA_Q_PAD_ALIGNED/eff_ctx/12-head）哪处产 NaN
> —— 下步 L1-alone + attn 中间 dump 定位 + 对齐 DeepSeek v4 `decode_attention_swa.py`。memory:
> `g5b_swa_multientry_kv_nan_root_cause`。

> **2026-07-12 (续⁵) ⭐ G5b co-tenancy crash 彻底解决（file-broadcast）— live 45 层路径 HTTP 200 稳定跑通；剩纯数值 [攻坚 4 结构性 blocker 清除]**：
> **决定性隔离**：offline `--steps 4`（4 forward 复用 prepared rt，真 KV，**无 vLLM co-tenancy**）**全 clean rc=0
> 无 507018** → rt-reuse/资源累积**排除**。→ 一攻的 HcclBroadcast timeout(err9) 和二攻的 507018 **同源** =
> **co-tenancy device 争用**：rank-0 在 sidecar 跑 45 层时，vLLM rank1-7 的 `HcclBroadcast` kernel 在同卡
> 8-15 自旋等待 → 与 sidecar kernel 争用（超时→err9 / fault→507018）。
> **修复（root-cause，仿 G4 NO_HCCL 思路）**：容器后端 `_pypto_full_forward` 把 device 侧 `tp_group.broadcast`
> 换成 **file-based broadcast**（rank-0 写结果到共享 /logs，rank1-7 **CPU-poll** 读，无 device collective）。
> 已部署 /logs（备份 .bak-g5b，`fwd_step` 计数 + `pypto_wd_bcast.{step}.bin` + lazy cleanup）。
> **device 验证**：restart 8001 mode=full（file-bcast 后端）+ 真权重全 45 层 sidecar → **prompt → HTTP 200
> 完成 4 tokens、无 crash、无 507018、无 HcclBroadcast err9**（t=483s，~120s/token 因每步 MoE 权重 copy）。
> **剩余（纯数值，= 攻坚 3）**：生成 token 错误（text=""，finite 但不对；8000 vanilla 同 prompt 出连贯文本）。
> 早先 sweep 只证「无 nan」用的是**合成随机 rope**，非正确性。下步 = 单层 paged-index 数值对拍 vLLM decode
> dump，核 rope 提取（`_wd_rope_from_emb` 是否匹配 step3p5 whole-decode kernel）+ KV 读 + per-layer 权重流。
> **⚠ 清理 hazard**：sidecar SIGTERM teardown 507018 → `force_reset_device(8)` **会 nuke 同卡 co-resident 8001**
> → 停 sidecar 后必须 restart 8001 + 清 card8-15 zombie VLLMWorker。
> **系统**：8001 restart 回 vanilla-serving，8000 oracle 全程 200。fix+复现器在 0162；文档 push。

> **2026-07-12 (续⁴) G5b live A/B 二攻：HCCL broadcast timeout ✅ FIXED，遗留缩到 sidecar 间歇 507018 [攻坚 4]**：
> 承续³。**遗留 A（数值）基本排除**：真权重+真 KV+合成 metadata 单层隔离 sweep（ctx=10/300/4090、block=0/5000/10、
> 多 block）**active 行(row0) 全部 FINITE 无 nan**（27~161）；sidecar 报的「Lx nan」是 padded 行（decode 1 active
> seq，其余 15 行 ctx=0→softmax 0/0），vLLM/容器后端只取 active 行。→ 数值非 blocker（复现器 `_client_wd_sweep.py`）。
> **遗留 B 精确定位为两个 crash mode，其一已修**：
> ① **HCCL broadcast timeout（一攻的 crash）✅ FIXED**：`HcclBroadcast error code 9` = rank-0 在 sidecar 里跑
> 45 层（被每步 MoE 权重 copy 拖慢，分钟级）而 rank1-7 阻塞 `tp_group.broadcast` 等 rank-0，超 HCCL 默认 120s
> connect timeout。修法：`/logs/start_8001_full.sh` 加 `HCCL_CONNECT_TIMEOUT/EXEC_TIMEOUT/EVENT_TIMEOUT=3600`
> （备份 .bak-g5b）→ 二攻 broadcast 不再是 crash 点。
> ② **sidecar 间歇 507018（二攻新暴露的真遗留）**：timeout 修好后 sidecar 跑更远 —— **全 45 层 forward 至少完整
> 完成一次（log 到 L44）**，但**后续 forward 命中 `507018`（chip dev=8 run failed，kernel runtime device fault）**→
> 关 socket → vLLM `RuntimeError: 'sidecar closed connection (header)'` HTTP 500。非单一确定层（跑完 45 层才 fault），
> 疑似多 forward 复用 prepared rt 的资源累积（ring heap？）或 co-tenancy 下间歇 device fault。**下步 = 层/step
> bisect + ring-heap/co-tenancy 调查**（每轮 device ~15min）。
> **系统状态**：sidecar SIGTERM clean（507018 teardown→force_reset 自清卡）；8001 restart 回 vanilla-fallback serving；
> 8000 oracle 全程 200 未受影响。fix + 复现器在 0162；文档 push。

> **2026-07-11 (续³) G5b live A/B 首次拉起：全 45 层真权重 pypto 路径 live 跑通(no-crash dispatch) + 两个精确遗留 [NEXT-SESSION 攻坚 4 首攻]**：
> restart 8001 mode=full（新容器后端 G5b metadata 版）→ health=200 → 起**真权重全 45 层** sidecar
> （`--ckpt` W8A8 + `--kv-ipc-dir` + `--serve` + `SIMPLER_COMM_NO_HCCL=1`, cards 8-15 co-resident 8001）：
> **7 programs / 45 层 / 87 steps 编译 + PREPARE OK + import 8 真 KV 池(0x12c1c0000000) + serve listening，
> 无 OOM、prepare/dispatch 阶段无 507018**。送 prompt → 容器后端路由 decode step 进 sidecar → **全 87 steps
> dispatched 跑完**（= 整条 live single-handoff：embed→socket→metadata→真 KV→45 层→broadcast 机制全通）。
> **遗留 A（数值，= 攻坚 3）**：sidecar per-layer torch-ref 从 L0 报 nan，但该 `max|abs|` 是 **16 行含 padded
> 行**（decode 只 1 active seq，rows 1-15 ctx=0 → softmax 0/0 = nan，与早先 isolated 测试 active 行 non-nan
> 一致）→ **active 行正确性未确认**，需单层 active-行 paged-index 数值对拍（真 KV 读 / block_table→consolidated
> pool offset）。**遗留 B（co-tenancy 稳定性，= 攻坚 4）**：第 2 个请求时 8001 EngineCore 崩在
> vLLM `c10d ProcessGroupHCCL::broadcast`（`_pypto_full_forward` 的 `tp_group.broadcast(next_hidden,src=0)`）
> —— co-resident pypto sidecar 的 device stream 与 vLLM HCCL broadcast 同卡时序冲突（sidecar teardown 507018
> → `aclrtResetDeviceForce(8)` 清 poison）。→ 需 sidecar 每 step 后 device 完全 sync/idle 再让 vLLM broadcast，
> 或换 handoff 时序。**清理**：sidecar SIGTERM clean（force_reset 清卡）；8001 restart 回 vanilla-fallback serving。
> 代码/日志：worker `/tmp/g5b_sidecar45.log`、backend 备份 .bak-g5b、NFS `workspace/g5b_*_20260711_231307`。

> **2026-07-11 (续²) G5b socket 真 metadata 协议 device 验证 + swa const-fold 证伪 ✅ [NEXT-SESSION 攻坚 1+2]**：
> ① **swa_moe const-fold 不再是 blocker**：当前工作树 canonical TP=8 编译 clean（attn_full/attn_swa/
> full_dense/swa_dense/moe_block(swa L3/full L4/L43/L44) 全 COMPILE OK，含 `--kv-ipc-dir` config override）。
> 原「blocker」是 `--smoke` 默认 `--tp 1` 走 `apply_tp1_patch`（unslice，违反铁律）撞 `moe.py:208` parity
> assert，非 const-fold。复现器 `_probe_alllayers_compile.py`。
> ② **socket 真 metadata 协议实装+验证**：把 sidecar 定长 hidden-only 协议换成 **self-describing
> length-prefixed 协议**（`<I hlen>`+JSON header+raw blobs），随 hidden 发 forward_context 的
> `seq_lens`/`block_table`(→BATCH×32 flat)/`slot_mapping` + 首请求发静态 rope（full[4096,64]+swa[4096,128]）。
> 三处同步：sidecar `_WholeDecodeServer.recv_step`+decode-loop `_feed_meta`（每 step copy 进各 attn sh，rope
> 按 full/swa 分流）、in-tree `vllm_monkey_patch.py`、容器后端 `pypto_whole_decode_backend.py`（自包含，已部署
> /logs，备份 .bak-g5b）。提取逻辑镜像 proven per-op `pypto_attn_backend`（prefill→collective fallback）。
> **验证**：offline 协议 round-trip PASS + 容器后端提取 E2E PASS；**device**：sidecar co-resident live 8001
> （NO_HCCL, WD_RING_HEAP=1GB）import 8 真 KV 池（peer_base 0x12c1c0000000）+ 新协议喂 metadata → L0 full-attn
> **active rows non-nan(27.6)**（padded 行 nan=预期，vLLM 只取 active）；8001/8000 全程 health=200，clean。
> **剩 G5b**：step5 单层 paged-index 数值对拍（需真 metadata + decode-step golden）+ step6 live A/B token-exact
> （8001 restart mode=full + 真权重全 45 层 sidecar）。代码在 0162 工作树 + NFS 备份
> `workspace/g5b_*_20260711_231307`（push 阻塞：0162 fork SSH 无 key、PAT 在本地 box）。

> **2026-07-11 (续) G5a LIVE 整网 plumbing device 验证 ✅ [tasks 5-7 大幅推进]**：在 G4 co-tenancy 解除
> （`SIMPLER_COMM_NO_HCCL=1`）基础上，把 whole-decode 接进 live 8001 mode=full 并 device 验证 plumbing：
> ① 自包含容器后端 `/logs/pypto_patch/pypto_whole_decode_backend.py`（内联 BATCH/HIDDEN，sitecustomize
> autoload）**8 rank 全 install** `Step3p5Model.forward -> sidecar`；② startup profiling 靠 **collective
> fallback**（全 rank `os.path.exists(sock)` 一致判定，sock absent → 全回退 original forward）存活 →
> 8001 **health=200**；③ 起 sidecar（`_stage_whole_decode_run.py --serve`，co-resident running 8001，
> NO_HCCL）→ 送 prompt → **HTTP 200 出 token**，8001 log `pypto forward #1 hidden(2,4096)->(2,4096)`
> （8 rank，serving 期 0 fallback），sidecar 收 live hidden 跑 decode。→ **embed + tp broadcast + socket
> 路由 + live co-tenancy 全链 device 验证**（G2 wiring live 证）。**剩 G5b token-exact**：现 dummy-KV + 4
> 层 → token garbage（nan），gate 在 **G3 真 KV-IPC + 全 45 层**。硬坑记录见 `phases/20` §G5a。
> 交付：backend + launch scripts 备份 `workspace/g5_*`；`_pypto_full_forward` + `_WholeDecodeClient`
> （pypto-lib 树 + 容器自包含版）；collective-fallback 修复。机器已清（8001 down / 8000 oracle 200 / cards 8-15 free）。

> **2026-07-11 G4 co-tenancy ✅ DISPATCH-RESOLVED（vLLM+pypto 同卡共存已解）[NEXT-SESSION task 1]**：
> 原症状：idle vLLM 8001 占 cards 8-15 时 whole-decode worker 首个 dispatch 挂 `comm_hccl.cpp:301
> HcclCommInitRootInfo failed: 7`（两 HCCL communicator 同 8 卡不共存；distinct HCCL_IF_BASE_PORT 无效）。
> **根因**：simpler HCCL control comm 是 vestigial（只 init/barrier/destroy，无 AllReduce/Send；唯一
> 消费者 `comm_barrier` dispatch 路径无调用者；数据面+domain 建立已走 file_barrier+IPC peer-access）。
> **修法（root-cause，env-gated）**：`SIMPLER_COMM_NO_HCCL=1` → comm_init 跳过 HcclGetRootInfo/
> HcclCommInitRootInfo（保 run_token 文件+file_barrier），relax null 检查，comm_barrier no-op。默认
> 不变（安全）。patch a2a3 `comm_hccl.cpp`（simpler commit 0162-local `878f3742`，待 push fork）+ 重编
> a2a3 runtime。**device 验证（0162 cards 8-15）**：worker + idle 8001 同卡 → PREPARE OK、all steps
> dispatched、rc=0、无 HcclCommInitRootInfo、8001 health=200；real-weight L0 full_dense torch-ref
> **PASS 1.000**；standalone HCCL vs NO_HCCL L0 **bit-identical**（swa/MoE 差异 = dummy-KV 非确定性，
> HCCL 路径同样，非 NO_HCCL 引入）。runbook：[`deployment/cotenancy-simpler-no-hccl.md`](deployment/cotenancy-simpler-no-hccl.md)。
> **剩余（tasks 5-7）**：G2 `_pypto_full_forward` wiring + G3 real KV import + G5 live A/B（真数值 token-exact 在此定论）。
> **2026-07-11 追加 G2 运行时机制 de-risk（device）**：所有 `_pypto_full_forward` resident holder 依赖的运行时机制已 device 验证 co-resident：
> (1) co-tenancy NO_HCCL ✓；(2) **resident prepared-rt 跨 decode step 复用**（worker `--steps 2` co-resident 8001 → step0+step1 各 dispatch L0-2、`reusing prepared rt`、rc=0、8001 health=200）✓；
> (3) real-weight dispatch co-resident（L0 torch-ref 1.000）✓。→ G2 剩纯 CODE：把 worker build+prepare+dispatch 抽成常驻 holder（module-global rt、manual `__enter__/__exit__`）+ 接 `vllm_monkey_patch.py:233 _pypto_full_forward`（45 层 dispatch loop + resident DeviceTensor residual + live forward_context→attn args）。G3/G5 后续。

> **2026-07-10 (续²) G1 Option-C 真 W8A8 dense/attn device 跑通 [NEXT-SESSION 任务 1+2 完成]**：
> Option-C worker `_stage_whole_decode_run.py` 4 层链（0,1,2,3=3 dense+1 swa_moe，5 步）在 0162
> cards 8-15 真 W8A8 **device rc=0 无 507018**；输出全部 ≠ synth（76/478/520/512 vs 30.9/44.8/59.8）
> → **punch-list item 2（Option-C worker+真设备输出）+ item 3（真 W8A8）device 验证完成**。修 3 个
> host bug（gate_exp 广播 / recon_attn per-rank w_g / `_share` 连续化）。torch-ref 对拍：full-attn(L0)
> + MoE-block(L3 moe_out) **精确 1.000**；SWA-attn 路径稳定 0.994（不累积 → 非层索引错位；满足项目
> `max_error_ratio=0.10` 判据；worker 阈值 0.999 过严）。SWA token-exact 定论留 live A/B。
> 详见 [`archive/milestones-2026-Q2.md` 2026-07-10 (续²)](archive/milestones-2026-Q2.md)。
> **续³（任务 3 完成）**：L43/L44 编译 blocker 修复（`_quant_moe_input` moe.py:1801 `InCore→Inline`，
> #1828 SplitIncoreOrch）；`--layers 0,1,2,44 --ckpt` device rc=0 无 507018。swiglu MoE offline 合成数值
> 不可信（synthetic-only，走 vLLM/live 定论），silu MoE 精确 1.000。**任务 4（45 层链）发现 blocker**：
> worker `_moe_block_sh` stack+share 全权重 → 3 variant `/dev/shm` OOM + 同 variant 多层复用首层权重，
> 需 per-layer weight-stream 重构（与 G2 常驻 weight-IPC 重叠）。
> **下一步 = 任务 4：worker per-layer weight-stream 重构 → 45 层链 → G2 live wiring。**
> **续⁴（任务 4 完成）**：per-layer weight-stream 重构（`_moe_layer_stack` slice-then-stack 修 3.5TB
> mega-stack OOM + `_load_moe_layer_weights` 每 step copy 修同 variant 多层复用首层权重）；chain 0-5
> 证 L3/L4/L5 共享 program 但权重各异、moe_out 各自 torch-ref PASS 1.000。**45 层全链 device rc=0**
> （7 programs, 87 steps, 无 OOM 无 507018）。**full "torch-ref 全层过" offline 不可达**（dummy KV →
> device 残差 L17 NaN，可复现/输入无关；full-chain 正确性必须 live A/B）。**本 session G1 offline
> 收尾：任务 1-4 全 ✅**。下一步 = G2 `_pypto_full_forward` live wiring（任务 5-7）。
> **续⁵（G3 HBM gate 修正）**：cards 8-15 实测 **64GB HBM/卡**（npu-smi HBM Capacity 65536MB）。
> TP=8 sharded：vLLM W8A8 ~24G→3GB/卡 + pypto BF16 ~47G→6GB/卡 + KV ≈ ~10GB/卡 → **fits 64GB**。
> memory 旧记「vLLM24G+pypto47G=OOM」是 aggregate/非-sharded（71G 压一卡）误判 → **G3 HBM 非硬
> blocker**。任务 5-7 剩余 = 纯 INTEGRATION（常驻 whole-decode 服务 + 真 KV import + `_pypto_full_forward`
> + live 8001 A/B），需专门 session，但无 HBM 门槛。

> **2026-07-10 环境确认 latest/consistent + tmov 编译 blocker 解除 + 整网集成真实状态盘点（team `vllm-pypto-e2e`）**：
> 在 0162 `stepfun/develop` 上确认工具链一致且最新：driver `25.5.2` / CANN `9.0.0 non-GA` /
> pypto `5e619dc7`(rebased origin/main) / pto-isa `ecc63…` / PTOAS `72ada0a1`(≈v0.49) / simpler `71e39623`；
> pypto-lib `94aa015c`。cards 0-7 = vanilla oracle(8000)，8-15 空闲。**（注意：`feat/whole-net-n1-fusion`
> 是独立的 N=1 整网融合攻关线 —— 见 [`phases/27-n1-whole-net-fusion.md`](phases/27-n1-whole-net-fusion.md)，
> 2026-07-10 已达成整网**编译**里程碑（`WholeDecodeFaithful` 真实 45 层单 program compile rc=0），
> 分支已 push `csy0225/pypto-lib` HEAD `0fd5afa`。与本 stepfun/develop 主线正交，勿混。主线开发仍在 0162 `stepfun/develop`。）**
> ① **tmov 编译 blocker 解除**：升级后 `full_out_proj_matmul` 报 `pto.tmov` 不支持的 address-space
> pair（910B 无 L1→L1 DMA）。4-agent 定位：N=256 时 out_proj cube RHS [256,256]=128KB 超 L0B 64KB →
> 触发 #1601 Vec-LHS→Mat staging → 非法 Mat→Mat tmov。修复 `OUT_PROJ_N_CHUNK 256→64`（RHS 降到 32KB，
> 原生 stage，数值 parity-safe），commit pypto-lib `d3075ac9`；MoE per-rank compile rc=0。**真正的根因
> 修复（对齐 Qwen3-14B split-K×split-N atomic-add out_proj，或落 `stage_lhs_to_mat` arch-gate）只影响
> prefill 性能，deferred 到 Phase 17/22**（见 blockers.md）。arch-gate 本身是死路（跳过 staging 会重新
> 触发 L0B 溢出）。② **整网集成真实状态盘点（sw-analyst 逐行核对 0162 WT）**：`_pypto_full_forward`
> (vllm_monkey_patch.py:233) 仍 fail-closed stub；**3-scalar layer_idx split 尚未落地**（grep 空，是
> 多层整网的 gating blocker）；Option-C 多程序 worker harness 被弃到 `/tmp/bak_worker*`；tail
> compute_logits 目前委托 vLLM norm+lm_head（非 pypto rms_lm_head kernel）。整网 live A/B 的有序 punch-list：
> (1) 3-scalar split → (2) 恢复 Option-C 7-program worker + 串真设备输出 → (3) 真 W8A8 权重 → (4) standalone
> 45 层链 device 验证 vs vLLM → (5) 接 `_pypto_full_forward` → (6) co-tenancy 测试 → (7) 8001 翻 `full` + live A/B。
> **③ 3-scalar split（Item 1）已 committed（pypto-lib `8b4bf3fa`）**：单个 `layer_idx` 无法同时索引三种布局
> 不同的权重栈（norm[45]abs / attn[full|swa]type-local / dense-MLP[3]dense-order，仅 L0 重合 → 多层拿错权重）。
> 拆成 `norm_layer_idx`/`attn_layer_idx`/`mlp_layer_idx`，共 74 内核 edit + callers + dense ST arity（原子 patch
> 脚本 backup+assert+rollback，reverse-review 语义 GO：index-class/arity/dispatch 全对）。`_smoke_program_build`
> rc=0。**单层行为不变（L0 三者==layer_idx），只改多层索引**；device/多层正确性经 Option-C 整网链 vs vLLM 验证
> （Item 2，下一步）。dense ST 在本树因 pre-existing `moe.py:208 apply_tp1_patch` assert 无法 run（CLAUDE.md
> parity，非本次回归）；MoE ST ScalarSpec redesign（w_gate_d 12-vs-3 层 OOB）deferred。
> **下一步：恢复 Option-C 45 层 worker（`/tmp/bak_realw`）→ 真权重 standalone 链 device 对齐 vs vLLM。**

> **2026-07-09 全栈升级到最新（parity 通过，pypto-lib 已推）+ gap-5 上游定位**：pypto
> `5e619dc7`(rebased origin/main) / pto-isa `ecb6c303` / PTOAS-src `72ada0a1`(v0.49) /
> **ptoas-bin v0.45→v0.49** / simpler `71e39623`；**pypto-lib `1a6c6342→b511da0e`（SplitIncoreOrch
> 移植修复，已 push fork stepfun/develop）**。升级引入的 `#1828 SplitIncoreOrch` MoE 编译回归已修
> （unwrap `_zero_routed_y_buf`/`_serialize_after_shared` 冗余 `pl.at`）；moe_block ffn_out 8 卡
> device PASS + w8a8 e2e 6 passed + decode_acceptance PASS = 与旧版 parity。gap-5 `cast→int8→cube`
> 仍未修（**无上游 commit**，根因 `infer_tile_memory_space_pass.cpp:55-56`；INT8-native gated OFF）。
> 详见 [`archive/milestones-2026-Q2.md` 2026-07-09](archive/milestones-2026-Q2.md) +
> `pypto-lib/docs/upstream-issues/gap5-cast-int8-cube-codegen.md`。**待推**：pypto/simpler/pto-isa/PTOAS
> rebased HEAD（force-with-lease）。下 session：整网 device chain + gap-5 上游修。

> **2026-07-07 stepfun/develop 全仓回归 PASS + 整网集成三大 de-risk（team `vllm-pypto-e2e`）**：
> 五仓均在 stepfun/develop 线（pypto `be90f992` / pypto-lib `1a6c6342` / pto-isa `e25732f0` /
> PTOAS `da011a3d` / runtime `1aa6efb4`；fork 远端 stepfun/develop 已是 `1a6c6342`）。
> ① **MoE-block 8 卡 device 精度回归全 PASS**（`_stage_moe_block_precision --target ffn_out` on-device
> gate vs vLLM ffn_out，atol=0.04）：L3/L4/L43/L44 覆盖全部 4 个 program-class 变体；红队确认 5 个 MoE
> 修复无 CRITICAL/HIGH（routed INT8 仅 gate routed、shared BF16；无 silu 回归；无 [N,1]/跨 rank 列读）。
> ② **整网 45/45 层 COMPILE PASS（Option C 解耦）**：融合 swa_moe 在 L3 编译失败（`attention_swa.py:479`
> 常量折叠 `pl.full([32−12,HEAD_DIM])`），改用「TP-attention 程序 + `select_moe_block`」解耦，全网编译通过
> （`/tmp/wholenet_optc_compile.py`）。③ **47.46GiB 单 key IPC = WORKS**（`/tmp/ipc47_probe.py`）：一个
> `aclrtIpcMemGetExportKey` 覆盖 48GiB、`ImportByKey` 返回同一 VA，零拷贝——解除 live 权重驻留最硬 gate。
> ④ 整网 dense 前缀 TP=8 **device 运行通过**（L0 full_dense / L1,L2 swa_dense 8 卡 DEVICE PASS，rc=0）。
> 上游发现：§10 nesting 已解（`pl.submit`/multi-program DistributedWorker #1706，已在 HEAD）；建议 ptoas
> v0.45→v0.48。**整网 live 端到端对齐仍是多周工程**（离线串联受 dump 缺 KV 限制，真整网精度须 live
> single-handoff A/B）。详见 memory `project_whole_model_pypto_design.md` 2026-07-07 段。

> **2026-07-06 MoE 单块 8 卡 bring-up**：standalone `EpTpMoE` 8 卡真实 W8A8。
> ✅ **gate_topk 8 卡死锁（507018/sched=100）真解决** —— 对照 DeepSeek 定位到错误的
> format2 两路 `mrgsort`（合并未完全排序的半块 → 归并状态机不终止），改成 DeepSeek 式
> format1 渐进链 `sort32→mrgsort(64)→mrgsort(256)`；canonical gate 8 卡跑通 ~20s，
> topk 与 vLLM 一致（gate 仍在 pypto 上算）。✅ **shared expert 路径验证数值正确**（对
> 0.12% torch 参考 PASS，含 ring→barrier-mesh tp_all_reduce）。⏸ **routed 路径精度未过**
> （41.8% 符号翻转，已隔离到 `_expert_routed` grouped-GEMM，排除 gate/权重/act-quant/
> dispatch/combine 数据搬运；下一步逐级设备 dump 定位）。代码：`csy0225/pypto-lib`
> 分支 `wip/moe-gate-fix-20260706`（956aede）。详见
> [`deployment/troubleshooting-moe-block-8card-gate-topk.md`](deployment/troubleshooting-moe-block-8card-gate-topk.md)。

> **2026-07-03 方向纠偏 + 卡点解除**：零拷贝 KV-IPC 集成 step 1-5 device 验证通过，
> IPC 主卡点（507899 / 207001 MemPool OOM）经「一 key 整池 map」正解**解除**；
> 范式回正为 PyPTO runtime 接管（非算子桥接）。详见
> [`phases/23-zero-copy-kv-ipc-validation.md`](phases/23-zero-copy-kv-ipc-validation.md)
> （验证报告 + 技术问题 + 重制定 plan：Phase 24 整层 / 25 全网 / 26 perf）。

---

## 阶段跟踪

| 阶段 | 标题 | 状态 | 详情 |
|-----:|------|------|------|
| **1** | **pypto kernel 原型** | ✅ **已完成** | [`archive/prototype-phase-01-19-summary.md`](archive/prototype-phase-01-19-summary.md) |
| **2** | **vLLM Ascend 后端集成** | 🟡 **进行中**（设计已落） | 见下 |

### Phase 2 sub-phases

| Sub-phase | 范围 | 状态 | 文档 | 估时 |
|-----------|------|------|------|------|
| **2.0（Phase 20）** | vLLM monkey-patch e2e — 整模型 patch `Step3p5Model.forward`；单卡/TP=8 mixed/full PyPTO runner 接入 | 🟡 **待实现**；dump-based 精度 blocker 已清，但 production backend 未接 | [`phases/20-vllm-backend-monkey-patch.md`](phases/20-vllm-backend-monkey-patch.md) | 3-4 周 |
| **2.1（Phase 21）** | 与 vLLM 原生精度对比 harness；L1/L2/L3 三层 | ✅ **dump-based 精度闭环完成**（BF16 decode、W8A8 decode、W8A8 prefill）；待 Phase20 backend 后做在线 L1/L2/L3 gate | [`phases/21-precision-validation.md`](phases/21-precision-validation.md) | 1-2 周补在线 gate |
| **2.2（Phase 22）** | Perf baseline + 两轮优化；TP=8 多卡 | 📐 设计已落；gate Phase 21 + 2 个硬 blocker | [`phases/22-perf-baseline.md`](phases/22-perf-baseline.md) | 6-8 周 |
| **2.7（Phase 27）** | N=1 单 `@pl.program` whole-net standalone | ✅ **canonical P42 完成**：0162 native W8A8/KV-IPC pull+pull 20/20，commit `0e7a0fdd` | [`phases/27-n1-whole-net-fusion.md`](phases/27-n1-whole-net-fusion.md) | standalone gate closed |
| **2.8（Phase 28）** | N=1 whole-net → vLLM live single-handoff | 🟡 standalone gate 已清；per-layer KV、redundant-weight/3-way HBM、live token-exact A/B 待完成 | [`phases/28-n1-live-integration.md`](phases/28-n1-live-integration.md) | dedicated live session |

**到 v1.0 production decode 的总目标**：自 2026-06-22 起约 12-16 周
（含 gate 任务的并行投入）。

---

## Phase 2 交付物分级（跟踪现在到了哪个 sub-version）

| Tier | 能跑什么 | 需要 Phase 2 哪几部分 | 需要清掉哪些 blocker |
|------|----------|----------------------|----------------------|
| **v0.1** | 单卡 dense + mixed-mode MoE 走 vLLM | Phase 20 | 无 |
| **v0.2** | 单卡 45 层 mixed-mode（dense pypto + MoE vLLM eager） | Phase 20 | 无 |
| **v0.3** | TP=8 多卡 dense + mixed-mode MoE | Phase 20 + Phase 22.1-3 | ✅ kernel blocker 已清；待 vLLM harness |
| **v1.0** | TP=8 / EP=8 全 pypto MoE + perf 数发布 | Phase 20-22 全完 | 待整网精度 + perf 优化（split task 融合） |

**当前口径拆分**：

- ✅ **dump-based 精度验证已闭环**：BF16 decode、W8A8 decode、W8A8 prefill（1k/4k/8k/32k/64k/128k）均已用 vLLM eager detail dump 作为 oracle，在 PyPTO 侧 reference/detail/final-logits 口径通过。
- 🟡 **production backend 仍未完成**：上述验证并不等价于“真实 vLLM 请求已经走 PyPTO NPU full runner”。后续仍需 Phase 20 把 `Step3p5DecodeFwd`/prefill runner、权重翻译、KV/cache ABI、vLLM monkey-patch 接入生产路径。
- 🟡 **真实 PyPTO NPU prefill kernel 仍待开发**：本轮 W8A8 prefill 是 vLLM golden + PyPTO reference 对齐；`prefill_moe.py` 的 L1 overflow 仍阻塞真正 PyPTO NPU prefill kernel。

---




### Step3p5 MoE 层 + tail 融合程序 (FusedMoELmHead) 编译通过 — Phase 25.3 building block (2026-07-04)

> **2026-07-04 追加 — mixed dense-MLP + MoE 融合编译通过（§10 命名墙已破）**：给
> `_dense_mlp_body_tp` 的 pl-tensor 局部名加 `dm_` 前缀（`patch_dmlp_rename.py`，行为不变、
> standalone 仍 rc=0），解决了 dense-MLP body 与 MoE chip_orch 同名 RMSNorm 局部（post_norm/
> resid1_fp32/scaled/normed…）在 `pl.inline` 下的冲突；再给 `_build_decode_layer_moe_program` 加
> `with_dense_mlp`（`patch_moe_dense_mixed.py`）——chip_orch 在 attention 之后内联 dense-MLP、其输出
> 作为 resid1 喂进 MoE。`_stage_mixed_compile.py -p a2a3` **COMPILE OK**：一个 **mixed dense-MLP +
> EP+TP MoE decode 层融进一个 @pl.program chip_orch**。§5(mixed shape)+§10(many-body inline+命名) 对
> dense+MoE 已证。剩余：mixed RUN（需 config-bound-shrink 让层堆叠权重装得下）、full-45 mixed
> chip_orch（decode_fwd.chip_orch 逐层 dispatch，仍 tail-only 占位）→ full-45 COMPILE、25.4 live
> 分片权重 ABI（唯一内存可行 run 路径）。

在腾出 cards 8-15（临时下线 live 8001，8000 vanilla 保留作 oracle）后推进整网融合：

- **基础复验**：`test_decode_layer_moe_st --variant full_silu_silu --world-size 8 -p a2a3`
  （`MOE_ST_DEV_OFFSET=8` cards 8-15，当前树 pypto-lib `2df9613`+工作树改动）→ `next_hidden_out`
  golden **PASS 28.9s**。注意 MoE ST 用**零权重专家**（golden=attention resid1），验证程序 8 卡
  能跑（无 507018）+ attention，不含 expert-precision。
- **FusedMoELmHead 编译通过**：给 `_build_decode_layer_moe_program` 加 `with_lmhead` 开关——镜像已工作的
  dense `FusedDenseLmHead`，把 `rms_lm_head` 作为独立 `lm_head_orch`(@pl.function Orchestration) 内联，
  host_orch 两遍循环（先所有 chip_orch 再所有 lm_head_orch）。**host COMPILE OK**
  （`_stage_moe_lmhead_compile.py` -p a2a3，完整 frontend→IR→ptoas→distributed codegen）→ 一个 MoE
  decode 层（attn + EP+TP MoE）与末尾 RMSNorm+LM-head **融进一个 @pl.program**。这是整网融合的 building
  block（与 dense FusedDenseLmHead 平行）；§5 mixed-shape / §10 many-body inline 对 MoE+tail 成立。
- **8 卡 RUN 被 weight-staging OOM 挡住（非融合/精度/507018 问题）**：`--with-lmhead` ST run →
  `rtMalloc failed 207001 (size≈3.02GB) tensor 14` at `runtime_maker.cpp:209`。tensor 14 = `wo`
  [tp,46080,4096]bf16=3.02GB（45×1024 层堆叠 full-attn wo）；MoE-only 能分配这 3GB，加 tail 的
  lm_head_weight(≈1GB)+logits 后 bind OOM。降 `PTO2_RING_HEAP` 4GB→1GB 无效（非 ring heap 累积；疑为
  full multi-rank input 在 bind 设备上的 staging）。属**测试 harness 供给问题**（ST 总是分配全 45 层
  堆叠权重），与融合正确性正交；tail-fusion 数值机制已由 dense 版 FusedDenseLmHead（8 卡 bad=0.0000）证明。
  下一步修法：单层测试收缩堆叠权重动态界 / 只喂 layer_idx 切片 / 缩小 lm_head 测试词表 / 诊断 3GB alloc
  在 61GB-free 卡上为何失败。
- **边界/交付**：`patch_moe_lmhead.py`（加 with_lmhead，applied 到 pypto-lib decode_layer.py 工作树，
  备份 /tmp/decode_layer.py.bak_moelmhead）、`patch_moe_st_lmhead.py`（ST --with-lmhead）、
  `_stage_moe_lmhead_compile.py`（host 编译 probe，PASS）。均在 0162 project root，**未 git commit**。
  live MoE 整网仍是 model-B 全网程序（Phase 25.4，多周级；per-card model-A 因 EP 跨卡不可行）。

### Step3p5 decode 尾部 (final RMSNorm + LM-head) 接入 live 并逐字对齐 (2026-07-04)

decode 的 **tail（末尾 zero-centered RMSNorm + vocab 切片 LM-head）现已完全在 pypto 上运行**，
在线 vLLM 8001 与 vanilla 8000 逐字（token-exact）一致。

- **内核根因修复（pypto-lib `2df9613`，已推送 fork stepfun/develop）**：`rms_lm_head` 在一个
  `pl.at(CORE_GROUP)` scope 产生 `final_normed` 暂存、在另一组 matmul scope 消费。作为顶层
  `@pl.jit` 能过（monolithic `lm_real` PASS），但被 inline 进 `@pl.program` chip_orch（decode_fwd /
  mtp / fused lm_head_orch）时，orchestration 把每个 CORE_GROUP scope 拆成独立 task，RMSNorm-scope →
  1007 个 matmul-scope 对 `final_normed` 的 fan-out 依赖被误追踪 → logits 错（~99.9% 或 60x 放大）。
  **修复**：把 RMSNorm + 所有 vocab-block matmul 折进**单一 scope**，只算一次 `inv_rms`，在 matmul
  循环内**内联归一化**每个 k-chunk（不再 materialize `final_normed`，无跨 scope/跨 task GM 暂存）。
  另 `_build_tp_rms_lm_head_program` 用 `vocab_per_tp=VOCAB_LOCAL`（非 `VOCAB//tp_size`），使单卡
  tp_size=1 构建仍按 per-rank 宽度切词表（对 8 卡规范构建后向兼容）。
- **验证**：8×910B 上 `TpRmsLmHead` 8 卡 PASS、`test_rms_lm_head` 单卡 PASS、`FusedDenseLmHead`
  PASS（bad 0.0000）；worker 离线 round-trip max 与 golden 一致（bad 0.077=bf16 噪声）。
- **live 接入**：worker `_stage_attn_worker.py` 加 `--fuse-lmhead`（构建 TpRmsLmHead(tp=1) + 加载
  `model.norm`/本 rank `lm_head` vocab 切片 + `lm_head()` RPC）；backend `_stage_attn_backend.py` 加
  `PYPTO_TAIL_LMHEAD=1`（替换 `Step3p5ForCausalLM.compute_logits`：每 rank 算 vocab 分片 →
  `tensor_model_parallel_all_gather(dim=-1)` → 全词表 `[.,128896]` logits）。fallback 为**响亮**（打印
  + 计数，非静默），startup profiling / worker 未起时才回退。
- **在线 A/B（8001 dense0-2 fused + tail 全 pypto vs 8000 vanilla，temp=0 seed=42，3 prompt）**：
  **全部逐字一致**；8 rank 均 `first compute_logits SUCCESS logits.shape=(1,128896)`；serving 期间
  fallback 增量 **0**。W8A8 的 `lm_head.weight`/`model.norm.weight` 均为 BF16（未量化）。
- **边界**：worker/backend/probe 代码仍在 0162 工作树（非 git 仓，未提交，属 live 集成 harness）；
  剩余 full decode 替换 = MoE 层 3-44 的 MLP/MoE 本体 pypto 化（model-B EP，最大一块；这些层 attention
  已在 Phase 24.3 live）。

### Step3p5 attention 多 position (ctx>1 / prefill) 乱码根因定位 + 修复 (2026-07-02)

full-attention 多 position（prefill / 带历史 batched decode）乱码、单 position 正确的 bug **根因找到并修复**。

- **根因**：Scope 2 里 Q 的 partial-RoPE 打包进 `all_q_padded` 的 pypto/ptoas **codegen 数值 bug**，定位在 `rot_q_hi` 段（列 `ROTARY_HALF_FULL..ROTARY_DIM`）。`reshape([8,128])` + col-32 子列切片 + `[8,32]@col-32` assemble 链路 miscompile；单行 K 路径（`[1,32]`）正确。
- **为什么一直没发现**：ctx=1 输出恒=V₀ 与 q·k 分数无关，掩盖了错误的 q 值；只有 ctx>1 暴露。`test_decode_layer_full_dense_st` 只测 ctx=1。
- **定位工具**：`_stage_scope12_qk.py`（逐字复制 Scope1/2/QK 的 standalone L3 程序，逐层 dump 对拍 golden）—— `q_proj_norm`/`k_proj_norm`/`k_cache` 全对，唯 `all_q_padded` 首错 (row0,col32)。
- **修复**：Q RoPE 打包改成逐 head `[1, ROTARY_HALF]` 连续切片（镜像 K 路径），落在 `pypto-lib/models/step3p5/{attention_full,attention_swa}.py`（**本地工作树，未 push**）。
- **验证（0162 card 8）**：`_stage_scope12_qk` scores 0.25/0.90→~0；`_stage_attn_e2e ATTN_PERRANK=1` crossrow 全 decode 层 0.8374→**0.0000 PASS**；dense ST 无回归 PASS 7.97s。
- **遗留**：SWA 修复已应用+编译过，runtime 待空闲卡（共享卡 OOM）；prefill 路径待确认；深度 writeup 待落 `pypto-lib/docs/known-pypto-pitfalls.md`；上游 codegen bug 待提（复现器 `_stage_scope12_qk.py`）。详见 [`archive/milestones-2026-Q2.md`](archive/milestones-2026-Q2.md) 2026-07-02 段。

### Step3p5 attention 设备共享 e2e PASS + device-shared 地基提交 (2026-06-30)

在 0162 打通 **attention 层经 device-IPC 共享 KV 的离线端到端**，并把 option B（device-mem IPC）底层代码提交到 feature 分支。

- ✅ **attention e2e**：独立进程 ctypes 零初始化 `(2,4096,128)` bf16 KV 块 + `aclrtIpcMemGetExportKey`；worker 编译 `select_decode_layer(0)`（full_dense，L3 fork chip child）→ `DistributedWorker` → `rt.import_ipc(key)` → `DeviceTensor` K/V → `rt.run`，输出对 torch golden（`_torch_attn_no_gate + _torch_dense_mlp`）`bad_ratio=0.0000`。证明 forked chip child 原地读写跨进程 IPC KV 正确。脚本 `_stage_attn_e2e.py`。
- ✅ **关键修复 `DeviceTensor.__getitem__`**：生成的 L3 `host_orch.py` 对每个输入 per-rank 切片 `k_cache[r,0:R,0:H]`，DeviceTensor 之前无下标；新增返回连续子视图（offset ptr + 降维/resize slice）。
- ✅ **device-shared 地基已提交**（本地 feature 分支，未 push）：simpler `pypto/device-shared@18bddac2`（import_ipc 全链路）；pypto `pypto/device-shared@0c4b8749`（`DeviceTensor.__getitem__` + `DistributedWorker.import_ipc` + 子模块 bump）。
- ✅ **vllm-ascend 镜像源同步**：`/data/chensiyu/hw_project/pypto/vllm-ascend`（0162），分支 `pypto/attention-integration`（off fork `fbfe288`），提交 live 集成蓝图 `PYPTO_ATTN_INTEGRATION.md@ba72967`（Option A：复用 `attention_full`，patch `Step3p5DecoderLayer` attention 子块；权重名/落点/KV-rows ABI 已逆向）。
- ✅ **8001 在线服务恢复**：腾卡跑 e2e 后恢复 dense(0-2)+shared(3-44)，8000=200/8001=200，8 worker，正常出 token。学到**正确恢复顺序**：先起 8001 做完 HCCL init → `Application startup complete` → 再起 worker（否则 worker 占卡 8-15 致 vLLM TP=8 HCCL init `rtBinaryGetFunction 107000` 全挂；`aclrtResetDeviceForce` 不解）。
- 下一步：按蓝图 S1-S4 把 `attention_full` 接进 live vLLM（worker `attn` op + 每层 KV 导出 + 窗口 A/B）。最大卡点 = **KV-rows ABI**（`attention_full` 编译 `KV_CACHE_ROWS` 须等于 vLLM 真实 `num_blocks*block_size`，远大于 e2e 用的 4096）。

边界：attention 设备共享**离线 e2e 已通 + 机制+地基齐备**，但**尚未接 live vLLM**；MoE-routed（EP）、tail 仍待。

### Step3p5 dense-MLP 真实 PyPTO kernel vLLM 集成 (2026-06-28)

在 0162 完成 **vLLM-Ascend + PyPTO 端到端集成**：dense 层（global 0,1,2）的 SwiGLU 由 **真实 PyPTO @pl NPU kernel** 计算，替代 vLLM 原生 `Step3p5MLP` 的矩阵乘；vLLM 保留 API / KV / 调度 / 显存 / RMSNorm / TP all_reduce。

- ✅ **架构（agent 投票）**：Option C（独立 host worker + IPC）+ Topology B（每 rank 一 worker，1:1，cards 8-15）+ Unix socket on shared mount。kernel 只算 per-rank partial（gate_up+silu+down，无 collective/无 rmsnorm），vLLM 做 RMSNorm（前）+ `tensor_model_parallel_all_reduce`（后）。
- ✅ **关键发现**：PyPTO 运行时本就是多进程（`chip_process` 子进程），故 host worker 不损失"同进程"收益；host-pypto 与 container-torch_npu **同物理卡共存实测通过**；W8A8 模型 dense 层权重是 BF16（只 MoE 量化），worker 直接读 W8A8 ckpt 无需反量化。
- ✅ **离线/单元**：kernel@device vs golden PASS；worker round-trip（真实 ckpt 权重）layer 0/1/2 `bad_ratio=0.0000`；跨容器 UDS bridge PASS；sum(8 rank partials) vs full SwiGLU `bad_ratio≈0.019` PASS。
- ✅ **在线 A/B**：patched 8001（`step3.5-flash-w8a8-pypto-densemlp`，TP=EP=8）与 baseline 8000 同 prompt `max_tokens=8` 输出**逐字一致**，**0 fallback**；patch 在 8 个 TP worker 进程经 sitecustomize 自动安装。
- ⚠ **性能**：当前 host round-trip（d2h→UDS→h2d，每 16 行一 tile）为正确性优先，patched ~2.6 tps vs baseline ~4.9-9.4 tps（baseline 含 MTP speculative）；perf benefit 待 Phase 22 device-IPC/零拷贝 + 全模型覆盖。
- 交付物（pypto-lib）：`models/step3p5/vllm_dense_mlp.py`、`tools/step3p5/pypto_mlp_worker.py`、`tools/step3p5/pypto_dense_mlp_backend.py`(monkey patch)、`tools/step3p5/test_pypto_dense_mlp_e2e.py`、`tools/step3p5/PYPTO_DENSE_MLP_E2E_REPORT.md`。

边界：dense 3/45 层走真实 kernel；attention（需 KV/block-table ABI）、MoE（需清 507018）、tail 仍由 vLLM 原生执行。这是"真实 @pl kernel 进 vLLM 在线 loop"的第一个完整闭环。

### Step3p5 45-layer online layer replacement smoke (2026-06-27)

在 0162 `stepcast-vllm-w8a8` 容器内完成 `PYPTO_STEP3P5_PATCH_MODE=layer_ref` 在线 smoke：

- ✅ `layer_ref` mode 替换全部 45 个 `Step3p5DecoderLayer.forward` 的 Python orchestration（input RMSNorm → attention backend → residual → post RMSNorm → MLP/MoE backend → residual），重用 vLLM 的 attention/KV 与 dense/MoE NPU kernels。
- ✅ 带 patch 的 vLLM 服务：port `8001`，served model `step3.5-flash-w8a8-pypto-layer`，TP=EP=8，`--quantization ascend`，eager。
- ✅ 在线 `/v1/completions` E2E PASS：prompt `请用一句话介绍北京。`，`max_tokens=1`，HTTP 200，top-1 text `?\n`，top-5 logprobs 正常返回。
- ✅ 多 token/长上下文补充验证 PASS：`max_tokens=4/8/16` + `1k prompt, max_tokens=8` 共 4 个 case，patched `layer_ref` 与 unpatched baseline 输出文本 4/4 完全一致。
- ✅ coverage artifact：`pypto_layer_ref_calls.json` 显示 `num_layers_observed=45`、`num_layers_replaced=45`、`all_observed_layers_replaced=true`，每层 `0..44` 均记录 patched layer_ref 调用。
- ✅ 与现有 unpatched baseline 服务（port `8000`）同 prompt top-1 对齐：均输出 `?\n`。

报告：`/mnt/nvme1/chensiyu/logs/step3p5_910b_w8a8_pypto_layer_v001/PYPTO_LAYER_REF_E2E_REPORT.md`。代码提交：`pypto-lib` `099aefa`；随后 coverage report 提交 `408a041`。

边界：这是 **45-layer online layer orchestration replacement**，但 heavy math 仍复用 vLLM NPU kernels；真正 @pl PyPTO full-network replacement 仍需把 45 个 per-layer @pl program wire 进 `Step3p5DecodeFwd.host_orch`。

ABI probe：新增 `PYPTO_STEP3P5_FORWARD_CONTEXT_REPORT` 后在线 dump 显示 vLLM `ForwardContext` 在 layer-level 可见，但当前 vLLM-Ascend eager path 暴露到 Python 的 `attn_metadata=None`、`slot_mapping={}`、sample `kv_cache` shape `[0]`，说明真正 PyPTO runner 不能只依赖 `ForwardContext`，还需要继续从 vLLM-Ascend `model_runner.input_batch` / attention backend 内部拿 block table、slot mapping 和 KV cache view。

### Step3p5 live vLLM parameter metadata contract (2026-06-27)

为 Phase 20 `nn.Module -> PyPTO bundle` 翻译补齐在线参数命名/shape contract：

- ✅ `vllm_monkey_patch.py` 新增 `PYPTO_STEP3P5_DUMP_PARAM_META`，在 tail patch 首次 `compute_logits` 时 dump live `Step3p5ForCausalLM.named_parameters()` metadata。
- ✅ 0162 `stepcast-vllm-w8a8` 容器内已生成 `/logs/pypto_tail_param_meta.json`（host 映射：`/mnt/nvme1/chensiyu/logs/step3p5_910b_w8a8_v001/pypto_tail_param_meta.json`），共 `744` 个 local-rank 参数。
- ✅ `weight_translate.py --vllm-param-meta ...` 校验 live vLLM 参数命名/shape/dtype 与 PyPTO 预期 local-rank contract 一致：`ok=true`，`num_expected=744`，`num_observed=744`，无 missing/extra/mismatch。

代码提交：`pypto-lib` `a59c7fe`；随后 `weight_translate.py --emit-vllm-transform-plan` 已输出 live vLLM -> PyPTO decode bundle transform plan（qkv split、gate_up split、MoE w13/w2 dequant/orientation 等），代码提交 `c4fca8a`。下一步是把 transform plan 落成真正的 in-memory tensor extraction。

### Step3p5 vLLM + PyPTO monkey-patch tail E2E smoke (2026-06-26)

Phase 20 monkey-patch surface 已在 0162 的 stepcast 容器内完成在线 smoke：

- ✅ `sitecustomize.py` autoload 验证：容器内 `tools.step3p5.vllm_monkey_patch.status()` 返回 `installed=True, mode=tail`，patch 目标模块 `/vllm-workspace/vllm/vllm/model_executor/models/step3p5.py`。
- ✅ 带 patch 的 vLLM 服务：`PYPTO_STEP3P5_PATCH_MODE=tail`，port `8001`，served model `step3.5-flash-w8a8-pypto`，TP=EP=8，`--quantization ascend`，eager。
- ✅ 在线端到端请求 PASS：`/v1/completions`，prompt `请用一句话介绍北京。`，`max_tokens=1`，HTTP 200，输出 top-1 text `?\n`。
- ✅ 与现有 unpatched baseline 服务（port `8000`）同 prompt/top-1 对齐：均输出 `?\n`。

报告与 artifacts：`/mnt/nvme1/chensiyu/logs/step3p5_910b_w8a8_pypto_patch_v001/PYPTO_TAIL_PATCH_E2E_REPORT.md`。

边界：这是 **tail-mode online smoke**，证明 vLLM monkey patch/autoload/服务端到端路径可用；**full-network replacement 仍未完成**，因为 `Step3p5DecodeFwd.host_orch` 仍未 wire 45 层 per-layer NPU program，`full` mode 保持 fail-closed。

### Step3p5 W8A8 prefill precision closure (2026-06-26)

在 0162 目标机完成 W8A8 prefill 多长度精度闭环，流程对齐 decode 阶段：vLLM eager W8A8 detail dump 作为 oracle，PyPTO 侧复算非 attention-core per-layer detail，并对 final RMSNorm + LM-head logits 做端到端比较。

- ✅ 覆盖长度：`1k / 4k / 8k / 32k / 64k / 128k`（`1024, 4096, 8192, 32768, 65536, 131072`）。
- ✅ vLLM W8A8 prefill golden：`/mnt/nvme1/chensiyu/logs/step3p5_910b_w8a8_prefill_v001/golden_step3p5_w8a8_prefill_vllm_sampled`；长序列采用 sampled detail dump（每个 forward 最多 128 rows）并裁剪到 PyPTO comparator 所需 tensor，避免 128k full dump 爆盘。
- ✅ detail + final logits 报告：`/mnt/nvme1/chensiyu/logs/step3p5_910b_w8a8_prefill_v001/pypto_prefill_precision/STEP3P5_W8A8_PREFILL_REPORT.json`，`ok=true`；acceptance 为 sampled W8A8 prefill detail `pass_rate >= 0.997`，final logits 全 case PASS。
- ✅ 各长度 worst pass rate：1k `0.999349`，4k `0.998698`，8k `0.999023`，32k `0.999349`，64k `0.999756`，128k `0.997559`。
- ✅ ST：`STEP3P5_PREFILL_REPORT_ROOT=/mnt/nvme1/chensiyu/logs/step3p5_910b_w8a8_prefill_v001/pypto_prefill_precision PYTHONPATH=. pytest -q tests/step3p5/test_step3p5_w8a8_prefill_st.py` PASS (`1 passed in 0.01s`)。

W8A8 prefill 回归数据包：`/mnt/nvme1/chensiyu/logs/step3p5_910b_w8a8_prefill_v001/step3p5_w8a8_prefill_regression_20260626.tar`，SHA256 `cd34f034e017c68437547e5f7f453a2f6b481a1e97e162a89ac21c422fe76b6e`。报告归档：[`archive/step3p5-w8a8-prefill-delivery-20260626.md`](archive/step3p5-w8a8-prefill-delivery-20260626.md)。代码提交：`pypto-lib` `81252e9`（随后 Phase 20 config-align 工具提交推进到 `e616407`）。

### Step3p5 W8A8 vLLM-vs-PyPTO precision closure (2026-06-26)

本轮按 BF16 golden 构造方式在 0162 重新部署 W8A8 checkpoint `/mnt/nvme1/chensiyu/step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp`，vLLM 使用 `--quantization ascend`、eager、TP=EP=8、NPU 8-15、port 8001 采集 int8/W8A8 detail dump，没有复用 BF16 golden。

- ✅ W8A8 权重加载：`weight_loader.py` 支持 `quant_model_weights.safetensors.index.json`，按 per-expert `weight_scale/weight_offset` 反量化 routed MoE INT8 权重到 PyPTO bundle。
- ✅ W8A8 routed MoE reference：detail compare 自动启用 dynamic activation quantization，匹配 vLLM W8A8_DYNAMIC routed expert 路径。
- ✅ vLLM W8A8 golden：`/mnt/nvme1/chensiyu/logs/step3p5_910b_w8a8_v001/golden_step3p5_w8a8_vllm_20260626_004648`，`beijing_1tok` 共 5944 个 `.pt` dump。
- ✅ acceptance：`decode_acceptance_w8a8_rank0.json` `ok=true`，48 层 dispatcher 覆盖。
- ✅ 主 45 层 detail：`pypto_all_layers_detail_compare_w8a8_beijing1_atol1_report.json` `ok=true`，3960 checks，worst pass rate `0.9995659589767456`。
- ✅ final logits e2e：`pypto_final_logits_from_vllm_w8a8/final_logits_report.json` `ok=true`，full-vocab pass rate `1.0`，argmax token `3648` 匹配。
- ✅ ST：`pytest -q tests/step3p5/test_weight_loader_w8a8.py tests/step3p5/test_step3p5_w8a8_e2e_st.py` PASS (`6 passed in 1.30s`)。

W8A8 回归数据包：`/mnt/nvme1/chensiyu/logs/step3p5_910b_w8a8_v001/step3p5_w8a8_e2e_st_regression_20260626.tar`，SHA256 `6f0a0f8e61f54d160325150917474209a0e493e987a77318aaeb1519c3915909`。端到端测试报告：[`archive/step3p5-w8a8-e2e-delivery-20260626.md`](archive/step3p5-w8a8-e2e-delivery-20260626.md)（目标机原件：`/mnt/nvme1/chensiyu/logs/step3p5_910b_w8a8_v001/STEP3P5_W8A8_E2E_ST_REPORT.md`）。代码提交：`pypto-lib` `b918e60`。


### Step3p5 BF16 vLLM-vs-PyPTO detail precision closure (2026-06-25)

本轮在 0162 isolated vLLM 容器中使用 BF16 checkpoint `/mnt/nvme1/chensiyu/step3p5_flash_release_hf_mtp3_bf16` 采集真实请求 detail dump，并在 PyPTO 侧完成逐层 tensor-input 对齐：

- ✅ 权重加载/dispatcher acceptance：`tools/step3p5/decode_acceptance.py --json` PASS。
- ✅ final logits 全 step：`pypto_final_logits_from_vllm_all_steps_eps1e5/final_logits_report.json` `ok=True`。
- ✅ 主 45 层 detail：`0~44` 共 `3960` checks，worst pass rate `0.9995659589767456`，报告 `/mnt/nvme1/chensiyu/logs/step3p5_910b_v017/pypto_all_layers_detail_compare_topk_final_atol02_report.json`。
- ✅ MTP3 detail：`45~47` 共 `279` checks，worst pass rate `0.9995659589767456`，报告 `/mnt/nvme1/chensiyu/logs/step3p5_910b_v017/pypto_mtp3_detail_compare_report.json`。
- ✅ ST：`test_step3p5_all_layers_detail_st.py` + `test_step3p5_mtp3_detail_st.py` 组合 PASS (`2 passed in 286.34s`)。

关键修复：`models/step3p5/config.py` 的 `EPS` 从 `1e-6` 修正为 vLLM `GemmaRMSNorm` 实际使用的 `1e-5`；MoE 对齐使用 vLLM fused router dump 的 `topk_ids/topk_weights` 驱动 PyPTO MoE reference。

BF16 回归数据包：`/mnt/nvme1/chensiyu/logs/step3p5_910b_v017/step3p5_bf16_e2e_st_regression_20260625.tar`。

**本次涉及仓库 commit 组合（记录于 2026-06-26）**：

| 仓库 | 分支/用途 | Commit | 备注 |
|------|-----------|--------|------|
| `pypto-lib` | `stepfun/develop` | `b198dcd` | vLLM forward-context ABI probe；45-layer coverage `408a041` |
| `pypto-project` | `main` | `b771c7e` | 首次记录本次验收状态的文档提交；本段会由后续文档提交推进 |
| `pypto` | `stepfun/develop` | `b00c8b23` | 本次未改代码；沿用当前 pin |
| `pto-isa` | `stepfun/develop` | `e25732f0` | 本次未改代码；沿用当前 pin |
| `PTOAS` | `stepfun/develop` | `da011a3d` | 本次未改代码；沿用当前 pin |
| `simpler` | submodule/runtime pin | `c66b4120` | 本次未改代码；沿用当前 pin |

BF16 回归数据包：`/mnt/nvme1/chensiyu/logs/step3p5_910b_v017/step3p5_bf16_e2e_st_regression_20260625.tar`，SHA256 `bce502f4cbafb61fe541385ab1828d33a1f9c32bdfb7d2009e871adba4c896c4`。



### MoE 8-card precision ST update (2026-06-24 evening)

本轮在 0162 当前代码和 CANN 9.0.0 non-GA 环境下补齐了 `test_decode_layer_moe_st --world-size 8` 的 rank-wise golden：

- ✅ dense 多卡基线：`test_decode_layer_full_dense_multirank_st -p a2a3 -d 0,1,2,3,4,5,6,7` PASS，8 rank `bad_ratio=0.0004`。
- ✅ MoE smoke：真实模型 MoE variants compile-only 全 PASS。
- ✅ MoE 8 卡真实模型组合 golden：`full_silu_silu` / `full_swiglu7_silu` / `full_swiglu7_swiglu16` / `swa_silu_silu` / `swa_swiglu7_silu` PASS。

相关 0162 日志位于 `/data/chensiyu/hw_project/pypto/workspace/moe8-precision-st-*.log`；代码侧进展同步在 `pypto-lib` 的 `test_decode_layer_moe_st.py` 和 `docs/upstream-issues/step3p5-moe-8card-fence-gap.md`。

### Final e2e precision gate preflight (2026-06-24; superseded by 2026-06-25/26 dumps)

`pypto-lib/tools/step3p5/e2e_precision_readiness.py` 作为早期预检保留；其中“checkpoint/vLLM 不可见”等环境 blocker 已由 0162 上 BF16/W8A8 dump-based precision 闭环解除。当前仍有效的结论是：

- ✅ `decode_fwd` torch distributed mock：worst pass rate 1.0。
- ✅ `step3p5_decode` synthetic smoke：pass rate 1.0。
- ✅ MoE 8 卡 ST 已补 rank-wise golden；真实模型会遇到的 5 个 MoE variant 全 PASS。
- 🟡 `Step3p5DecodeFwd`/prefill runner 尚未接入真实 vLLM online backend；见 Phase 20 production backend blocker。
- 🟡 head_gate ×1 parity 策略仍待在线 backend L1 gate 决策。

## 🎯 Decode 接管 gap 盘点（2026-07-10，下 session 集中攻破）

**目标口径**：pypto 接管 step3p5 整网 decode = **live single-handoff A/B**（8001 vLLM 内 pypto 跑完
整 45 层 decode，token-exact vs 8000 vanilla）。

**已就绪（地基）**：
- 逐层精度离线验证过（dense / MoE-block 8 卡 / tail），数学正确。
- 编译级 blocker 全清：tmov（`d3075ac9`）+ 3-scalar 多层索引（`8b4bf3fa`）；`_smoke_program_build` rc=0。
- 47GiB 单 key IPC ✓；N=7 co-prepare device-clean ✓；Option-C 43-45/45 层 COMPILE ✓。
- dense L0-2 + tail 曾 live token-exact（早期 session）。

**还差的工作（按关键路径排序，"集中攻破"顺序）**：

| # | Gate | 内容 | 估时 | 依赖 |
|--:|------|------|------|------|
| **G1** | **Option-C 链 device 验证**（🟡 4 层进行中） | worker `_stage_whole_decode_run.py --worker` 已建（N=7 dedup + worker 自补 norm/residual + 3-scalar + gate_exp）。**Pass 1 synth 机制 rc=0**（4 层全 5 步派发、无 507018）+ **Pass 2 真 W8A8 rc=0**（47GB 载入 + `moe_out` 0→3.5 真专家跑通）已达成。**剩**：dense/attn 真权重 wiring（现 synth）+ torch-ref 逐层对拍 + L43/L44 SplitIncoreOrch(Option C) → 扩 45 层。详见 archive 2026-07-10 (续)。 | 1-2 session | 无（split committed） |
| **G2** | **`_pypto_full_forward` live wiring** | `vllm_monkey_patch.py:233`（当前 fail-closed）：install() 建常驻 DistributedWorker holder；import KV pool（pypto_kvpool）+ 权重 pool；45 层 dispatch loop + 常驻 DeviceTensor residual handoff；读 live `forward_context` slot_mapping/block_table 进 attn args；final hidden copy 回。 | 1 session | G1 |
| **G3** | **HBM 共存 / 权重策略（live 硬 gate）** | vLLM W8A8(~24GB) + pypto BF16 pool(47GB) = OOM(>64GB)。方案 (a) standalone（vLLM 卸载，~58GB，验机制非 live）先行；(b) **gap-5**：pypto 直接吃 vLLM 常驻 W8A8 + in-kernel dequant（INT8×INT8→INT32，primitive 已 device-validated）→ ~31GB 可共存。gap-5 是 net-new kernel 工作但 primitive 已验证。 | gap-5 2-3 session | G1 |
| **G4** | **co-tenancy 507018 device 测试** | whole-decode worker 在与 resident-idle vLLM 共卡时能否干净 dispatch（single-handoff vLLM-idle 可能绕开 per-layer co-tenancy 507018）。未测。 | 0.5 session | G2 |
| **G5** | **tail** | pypto rms_lm_head 重新接进 compute_logits（当前委托 vLLM），或保留 vLLM tail（decode 接管可接受 lm_head 留 vLLM）。 | 次要 | - |

**Deferred（不阻塞 decode 接管）**：prefill MoE L1 overflow（TASK-29）、MTP、Qwen3-aligned out_proj 根因修复（prefill perf）、MoE ST `w_gate_d` OOB redesign、单层 ST harness 腐坏修复（moe.py:208 / gate_r）。

**下 session 单点突破 = G1**（Option-C 45 层链 standalone device 跑通 + vs vLLM 对齐）。worker 骨架在
`/tmp/bak_realw`，3-scalar caller 已就位（`_stage_whole_decode_run.py` type-local）。

---

## 立即可做的下一步（按优先级）

0. **（新，2026-07-03 主线）Phase 24 —— step 6 整层 live 替换（P0）**：把已验证的
   零拷贝 KV-IPC「一 key 整池 map」接进 live 8001 —— patch `_allocate_kv_cache_tensors`
   产出「一 buffer + 一 key + map」（取代会 OOM 的 per-tensor MemPool）→ worker 一次
   import 整池建 VA-map → page_attention 走 map + block_table → 扩到全 45 层 decode
   A/B `bad_ratio=0`。详见 [`phases/23-zero-copy-kv-ipc-validation.md`](phases/23-zero-copy-kv-ipc-validation.md) §5。
   完成后 → Phase 25（step 7 真 module 全网 + Wave-3 whole-model orchestration）→ Phase 26（perf）。

1. **Phase 20 backend 接入（P1）**：`config_align.py` 已启动并在 W8A8 checkpoint 上 PASS（pypto-lib `e616407`）；`weight_translate.py` 已提供 per-rank bundle manifest/export contract（pypto-lib `0511d27`）；`vllm_monkey_patch.py` 已提供 tail/shadow/full patch surface（full 目前 fail-closed，等待真实 runner；pypto-lib `9718083`；autoload helper `588610e`，已在 stepcast 容器内通过 `sitecustomize` autoload smoke，并完成 tail-mode vLLM 在线 E2E 请求 PASS）；live vLLM parameter metadata contract 已验证（pypto-lib `a59c7fe`）。下一步接 vLLM `nn.Module` in-memory 权重翻译，并把 `Step3p5DecodeFwd`/runner 接到 vLLM 请求路径。
2. **真实 PyPTO prefill NPU kernel（P2）**：重构 `prefill_moe.py`，用 multi-step gate/up chunking 清 L1 overflow，完成 1k~128k NPU prefill ST。
3. **在线精度 gate（P3）**：Phase 20 backend 能跑后，补 vLLM patched backend 的 L1/L2/L3 gate；当前 dump-based precision artifacts 作为 oracle/regression baseline。
4. **性能 baseline（P3/P4）**：做 decode-only TPS/ITL、prefill TTFT、1k~128k 性能曲线；分析 MoE dispatch/combine、TP/EP 通信、host launch overhead。
5. **MTP speculative 集成（P4）**：把 MTP 拼进 `decode_fwd` 和 vLLM speculative pipeline；该项不阻塞当前 correctness。 

---

## 组件 Pin Snapshot（最新一行）

| 日期 | 事件 | pypto | pypto-lib | pto-isa | PTOAS（src） | simpler（submodule） | ptoas-bin |
|------|------|-------|-----------|---------|--------------|---------------------|-----------|
| 2026-07-17 | N=1 stable environment freeze：唯一 SSOT 见 `develop/N1/N1-STABLE-ENV-0162-20260717.md`；clean stack 在 0162 P42 pull+pull smoke `rc=0,argmax=303` | `n1fusion-base:e277de9f` | `feat/whole-net-n1-fusion:0e7a0fdd` | `main:ecb6c303` | `main:72ada0a1` | `n1fusion-base:36957c6b` | `0.45 (sha fe7949da…)` |
| 2026-07-16 | N=1 standalone runtime manifest formalize：相关 pypto/simpler dirty runtime 支持已提交；最终 clean stack 在 0162 P42 pull+pull smoke `rc=0,argmax=303` | `n1fusion-base:e277de9f` | `feat/whole-net-n1-fusion:0e7a0fdd` | `main:ecb6c303` | `main:72ada0a1` | `n1fusion-base:36957c6b` | historical label `v0.49`（实际 binary 见 stable SSOT） |
| 2026-07-16 | N=1 model exact-source canonical closure：`0e7a0fdd` 在 0162 P42 native W8A8/KV-IPC pull+pull 20/20 `argmax=303`；当时 runtime 实际为 pypto `5e619dc7` + dirty support、simpler `98ce22a6` + dirty support，后由上一行 clean pins formalize | `5e619dc7 + dirty runtime source` | `feat/whole-net-n1-fusion:0e7a0fdd` | `main:ecb6c303` | `main:72ada0a1` | `98ce22a6 + dirty runtime source` | v0.49 |
| 2026-07-12 | on-device head-gate 恢复（matmul_acc N=16 修复）→ 整网 per-layer gate_r 解除 blocker；whole_decode_faithful_real TP=8 COMPILE OK；L1 A/B 暴露 pre-existing attn NaN | （未改本 session） | `feat/whole-net-n1-fusion:f07da3b`（attention_full/swa Scope 1.f on-device gate + gate_r=block-diag R + 3 probe） | （未改） | （未改） | （未改） | v0.49 |
| 2026-07-10 | tmov 修复 + 3-scalar layer_idx split（整网多层 gating blocker）committed + push fork stepfun/develop | `stepfun/develop:5e619dc7` | `stepfun/develop:47c260e3`（`d3075ac9` tmov chunk64 + `8b4bf3fa` 3-scalar split + `47c260e3` ST arity；fork `b511da0→47c260e`） | `main:ecb6c303` | `main:72ada0a1` | `71e39623` | v0.49 |
| 2026-07-07 | MoE-block 精度全 PASS 合并到集成分支 + import_ipc 全网 push + 0162/fork 对齐（L44 shared-swiglu16 clamp + router_bias BF16 补齐到 backend/worker 线；三仓 push；0162 rebase 到最新） | `stepfun/develop:be90f992`（DeviceTensor.__getitem__ slicing + distributed_runner import_ipc glue；submodule→simpler 1aa6efb） | `stepfun/develop:1a6c634`（L44 精度修复 cherry-pick 到 bb9e683 merge 线：routed a2a + INT8 + shared clamp `2b00bec` + router_bias BF16 `1a6c634`） | `stepfun/develop:e25732f0`（未改） | `stepfun/develop:da011a3d`（未改） | `stepfun/develop:1aa6efb`（import_ipc device-IPC key import `c236194`/rebased `1e55bba` + timeout 实验 + comm PID-whitelist fix `25a0544`） | `v0.45` |
| 2026-07-05 | backend↔co-resident-worker code path 完成（BE 协议 + layer targeting + 容器安全 import，selftest bad=0） | `stepfun/develop:b00c8b23` | `stepfun/develop:dbad26d`（pypto_moe_backend co-resident 协议 + container-safe；已推送 fork） | `stepfun/develop:e25732f0` | `stepfun/develop:da011a3d` | `c66b4120` | `v0.45` |
| 2026-07-05 | co-resident worker routed op device PASS（dense+routed 同 ChipWorker，无 co-tenancy） | `stepfun/develop:b00c8b23` | `stepfun/develop:0249700`（pypto_mlp_worker routed op；已推送 fork） | `stepfun/develop:e25732f0` | `stepfun/develop:da011a3d` | `c66b4120` | `v0.45` |
| 2026-07-05 | @pl.jit routed device-run PASS（真 W8A8，co-resident live 路径解锁） | `stepfun/develop:b00c8b23` | `stepfun/develop:ae00e9a`（_routed_jit_probe --device-run；已推送 fork） | `stepfun/develop:e25732f0` | `stepfun/develop:da011a3d` | `c66b4120` | `v0.45` |
| 2026-07-05 | MoE routed backend hook `_apply_mlp` + device glue-test PASS（bad=0.0000） | `stepfun/develop:b00c8b23` | `stepfun/develop:20292aa`（pypto_moe_backend.py；已推送 fork） | `stepfun/develop:e25732f0` | `stepfun/develop:da011a3d` | `c66b4120` | `v0.45` |
| 2026-07-05 | worker `routed` op device round-trip PASS（bad=0.0000）+ _serve bf16 fix | `stepfun/develop:b00c8b23` | `stepfun/develop:e17b4ab`（_serve bf16 序列化修复；已推送 fork） | `stepfun/develop:e25732f0` | `stepfun/develop:da011a3d` | `c66b4120` | `v0.45` |
| 2026-07-05 | MoE routed-expert per-rank 内核（真 W8A8 bad=0.0000）+ worker `routed` op | `stepfun/develop:b00c8b23` | `stepfun/develop:fc0bafb`（vllm_routed_experts.py + _routed_jit_probe.py；已推送 fork） | `stepfun/develop:e25732f0` | `stepfun/develop:da011a3d` | `c66b4120` | `v0.45` |
| 2026-07-04 | tail lm_head 单-scope 修复 + 接入 live（token-exact） | `stepfun/develop:b00c8b23` | `stepfun/develop:2df9613`（single-scope rms_lm_head inline fix；已推送 fork） | `stepfun/develop:e25732f0` | `stepfun/develop:da011a3d` | `c66b4120` | `v0.45` |
| 2026-06-27 | Phase20 online 45-layer layer_ref replacement + context ABI probe | `stepfun/develop:b00c8b23` | `stepfun/develop:b198dcd`（forward-context probe; 45/45 layer coverage `408a041`） | `stepfun/develop:e25732f0` | `stepfun/develop:da011a3d` | `c66b4120` | `v0.45` |
| 2026-06-22 | Phase 2 设计落地；建项目跟踪仓 | `stepfun/develop:b00c8b23` | `stepfun/develop:b918e60`（W8A8 precision alignment；BF16 0~47 detail ST 基线 `d4c01b9`） | `stepfun/develop:e25732f0` | `stepfun/develop:da011a3d` | `a6e06406` | `v0.45` |

历史 pin snapshot 见 [`archive/milestones-2026-Q2.md`](archive/milestones-2026-Q2.md)。

---

## 当前 Blocker / Deferred Items

| # | Blocker | 严重度 | gate 什么 | Owner | 详情 |
|--:|---------|--------|-----------|-------|------|
| N1-S-0162 | **N=1 standalone release gate（0162 scope）** | ✅ RESOLVED 2026-07-16 | `0e7a0fdd` exact-source P42 pull+pull，20/20 `argmax=303` | team-lead | [`N1-CANONICAL-TEST.md`](N1-CANONICAL-TEST.md) |
| N1-S-0234 | **0234 上 pypto-lib 同步后的 whole-net stall（完整对象未确认）** | 🔴 Active / 未独立复核 | 重新取得 SSH 后核对三仓、runtime binary、环境并重跑 canonical | team-lead | [`blockers.md`](blockers.md) §N1-S-0234 |
| N1-L | **Phase 28 live：per-layer KV + redundant-weight/3-way HBM** | 🔴 Active | live single-handoff token-exact A/B | team-lead | [`phases/28-n1-live-integration.md`](phases/28-n1-live-integration.md) |
| G4 | **co-tenancy：whole-decode worker HCCL world 与 vLLM 同卡冲突** | ✅ RESOLVED (dispatch) | ~~live single-handoff~~ → 已解，`SIMPLER_COMM_NO_HCCL=1` | team-lead | [`deployment/cotenancy-simpler-no-hccl.md`](deployment/cotenancy-simpler-no-hccl.md) |
| 1 | Phase 20 production backend 未接入 | 🟡 功能 | 真实 vLLM 请求走 PyPTO runner | 未指派 | `phases/20-vllm-backend-monkey-patch.md` |
| 2 | Prefill MoE L1 overflow（TASK-29） | 🟡 功能/性能 | 真实 PyPTO NPU prefill kernel + TTFT | 未指派 | [`blockers.md`](blockers.md) §2 |
| 3 | head_gate × 1 旁路 — vLLM 原生语义偏离 | 🟡 精度 | 在线 backend L1 layer parity | TASK-L（pto-isa 上游） | [`blockers.md`](blockers.md) §1 |
| 5 | MTP 集成进 `decode_fwd` | 🟢 Deferred | speculative decoding 吞吐 | 未指派 | [`blockers.md`](blockers.md) §6 |

---

## `gpu-a910x-0162`（Phase 16 验证机）目前已确认能跑

| 组件 | 验证 | 备注 |
|------|------|------|
| driver 25.5.2 | ✅ 2026-06-22 | `npu-smi info -t board -i 0` 报上 |
| firmware 7.8.0.7.220 | ✅（chip flash） | 跨重启持久 |
| CANN 9.0.0 non-GA/non-beta | ✅ `/usr/local/Ascend/cann` → `/mnt/persist/Ascend/cann-9.0.0/cann-9.0.0` | 2026-06-24 已重装并重编译 pypto/runtime |
| simpler L3 allreduce_distributed -d 0-1 | ✅ 2026-06-24 | 1 passed / 1 skipped（pytest harness） |
| pypto-lib 前端 smoke rc=0 | ✅ 2026-06-24 | `_smoke_program_build` 通过 |
| Decode dense full ST @ device 0 | ✅ 8.54s（ratio_allclose PASS，2026-06-24） | CANN 9.0.0 non-GA 重编译后验证 |
| Decode dense SWA ST @ device 0 | ✅ 15.61s（ratio_allclose PASS，2026-06-24） | CANN 9.0.0 non-GA 重编译后验证 |
| Phase 19 MoE real-model variants smoke compile | ✅ PASS | TP=8 per-rank slice 路径 |
| Decode MoE full_silu_silu ST @ 8 cards | ✅ golden PASS 32.61s（2026-06-24） | rank-wise golden；retry 后通过 |
| Decode MoE full_swiglu7_silu ST @ 8 cards | ✅ golden PASS 27.64s（2026-06-24） | full attention + routed swiglu7 + shared silu |
| Decode MoE full_swiglu7_swiglu16 ST @ 8 cards | ✅ golden PASS 26.74s（2026-06-24） | full attention + routed swiglu7 + shared swiglu16 |
| Decode MoE swa_silu_silu ST @ 8 cards | ✅ golden PASS 33.61s（2026-06-24） | SWA + routed/shared silu |
| Decode MoE swa_swiglu7_silu ST @ 8 cards | ✅ golden PASS 35.97s（2026-06-24） | retry 后通过；前一轮出现 transient 507018 |
| Phase 15 单卡 e2e | ✅ rc=0，20 tasks complete | head_gate ×1 旁路 + TP=1 patch 路径 |

---

## `gpu-a910x-0234` 当前状态

> **2026-07-16 审计边界**：本次尝试
> `ssh infra@gpu-a910x-0234.host.platform.shaipower.com` 返回
> `Permission denied (publickey,password)`。下文的 0234 运行结果来自既有项目记录，
> 尚未由本次 session 独立复核；不要把“只拉 pypto-lib”或“core==commit”的
> 记录当成完整三仓/build/environment manifest。

**已升级（2026-07-10 核对）**：driver `25.5.2` / firmware `7.8.0.7.220` / CANN `9.0.0-beta.1`（三剑合璧齐）。
Phase 16 cap 缺口**不再是** 0234 的问题（旧记录 25.5.1/7.8.0.6.201 已过时）。0234 曾成功跑多卡
（2026-06-24 MR-golden dense 8 卡 PASS、2026-06-29 L3 allreduce `max|out-expected|=0`）。
**历史纠错**：2026-07-10 曾将 507899 误判为节点级 IPC poison；后续 reboot
未解决，clean runtime rebuild + 关闭 `SIMPLER_ENABLE_PTO_SDMA_WORKSPACE`
后 baseline 恢复，证明实际是 stale/mismatched runtime 与配置问题。当前 0234
因 SSH 权限不可达，既不能继续标为 poisoned，也不能标为已验证健康。
