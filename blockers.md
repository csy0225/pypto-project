# 活跃 Blocker

阻塞项目进展的 **open** issue 的 SSOT。每条：症状 / 根因 / 当前状态 / 解除条件 / 链接。

**协议**：blocker 解决时，**删掉本文件这一节** → 到 [`postmortems/`](postmortems/) 建一篇
五段复盘（模板 [`postmortems/TEMPLATE.md`](postmortems/TEMPLATE.md)）+ 更新
[`STATUS.md`](STATUS.md) blocker 摘要。已解问题不留在本文件。

**已解 blocker 的复盘去向**：见 [`postmortems/README.md`](postmortems/README.md)。
如 507899/507018、co-tenancy(G4)、tmov Vec-LHS、gate_topk、多程序 co-prepare 死锁、
gap-5、scheduler-timeout、attention 乱码、G5b import_ipc、swa_moe const-fold 等均已归档。

**最后检视**：2026-08-03。

---

## 🟡 ACTIVE — DEPLOY-REPRO：镜像内 git 工作树 dirty，pin 集不足以复现验证环境（部分已解）

**症状**：按 `deployment/docker/build.sh` + 记录 pin 全新构建的镜像，整网 CI 在
`_reset_persistent_domains → orch.copy_to` 报
`device pointer 0x… is not a live allocation on worker 0 (… or an interior pointer)`；
而 0728 的 candidate 镜像同样 pin 却能跑。

**根因**：0728 三个 candidate 镜像
（`step3p5-b404a3c9-{,ci-cleanup-,ci-final-}20260728`）都带一份**未提交**的 simpler
补丁（span-aware child provenance，`worker.py +133/−19` + `orchestrator.py +2/−2`），
`build.sh` 严格按 pin clone 拿不到。已发布的 0726 镜像不含该补丁。
详见 [`postmortems/14`](postmortems/14-image-dirty-worktree-unreproducible-pins.md)。

**已解部分**：补丁已逐字节入库为 `csy0225/simpler@8459d60f`（`stepfun/develop`），并由
`csy0225/pypto@6933b1aa` 的 `runtime` submodule gitlink 固定（Dockerfile 的显式 checkout
会被 `git submodule update` 覆盖，只改 spec 无效）；`img_regress.sh` 已增加
`IMAGE_WORKTREE_CLEAN_AUDIT`（五仓逐个查 dirty）；已发布镜像
`stepfun-develop-20260729-allreduce-push` 在该基线上回归全绿。

**仍 open 的部分**：
1. 本地 `workspace/pypto/runtime` 工作树里还有一版**更新的** WIP（`worker.py +222/−47`，
   给 `exact_malloc_live` 加 `malloc_size` 上界，另带 119 行单测），**设备上从未验证**，
   未入库。需独立验证后再决定是否前进 pin，否则它会重复制造同类不可复现状态。
2. 0728 三个 candidate 的既有验证结论（C/D/G 的 BS1/2/16、N=256、Main 8-step）
   都建立在未入库补丁上，需在 `8459d60f` 基线上复核后才能作为可复现证据引用。
3. 其余四仓（pypto / pto-isa / PTOAS / pypto-lib）是否也曾以 dirty 工作树参与过
   历史镜像构建，尚未回溯核查。

**解除条件**：① 本地 WIP 增量完成设备验证并入库或明确废弃；② 0728 的 C/D/G 结论在
`8459d60f` 基线上复现；③ 历史镜像的 dirty 回溯完成。

---

## 🔴 ACTIVE — N1-S-0234：pypto-lib 同步后记录的 whole-net stall，待完整 manifest 复核

**范围必须分开**：`N1-S-0162 = release-qualified`（`0e7a0fdd` exact-source / P42 /
pull+pull，20/20 argmax=303）；`N1-S-0234 = active / root cause unknown`（项目记录称
pypto-lib 三个 release 文件与 `0e7a0fdd` byte-match 后 devices 0..7 fresh canonical
3/3 stall，但完整 pypto/simpler/runtime binary/environment 等价性未验证）。

0162 的 release gate 已关闭，**不能外推到 0234**，也不能把"0234 只拉了 pypto-lib"
当成三仓/binary/环境一致。2026-07-16 `ssh infra@gpu-a910x-0234...` 返回
`Permission denied (publickey,password)`，该 3/3 结果只能标记为**既有记录、未独立复核**。

**优先排查**：① 核对三仓 commit/dirty/submodule；② editable/source 实际加载文件 +
runtime `.so` hash/mtime；③ CANN/PTOAS/checkpoint/device/ring env；④ 对齐后仍 stall
则存同轮 TASK/CLUSTER/COND + `kernel_config.py` + build hash + dmesg delta；⑤ 不把
512B signal isolation 当跨机器充分条件或唯一根因。

**解除条件**：恢复 0234 访问 → 生成完整 manifest → 按 [`reference/canonical-test.md`](reference/canonical-test.md)
重跑；若仍 stall，定位机器/runtime/environment delta 或新通信边界。参见
[`postmortems/07-whole-net-scheduler-timeout.md`](postmortems/07-whole-net-scheduler-timeout.md)。

---

## 🔴 ACTIVE — Phase 28 live serving：per-layer KV bridge + 3-way HBM / redundant weights

**当前代码边界（2026-08-03）**：`pypto-lib stepfun/develop@7099476b7c4f13112b159e237e7a64344803caf0`
只保留 `models.step3p5.decode_fwd:whole_decode_step3p5`。C/D/G、BS1 correctness、
Attention/Vec I1 与 Wave5 TP all-reduce stability 已合入；Wave5 在 0162
release-qualified。本节只描述 Phase 28 live serving 缺口，不再把 BS1 的旧
`6127` 结果视为当前代码状态，也不把 0162 的结论外推到其它机器。
`models/step3p5_opt`
package、`whole_decode_opt` 和 `WholeDecodeOpt` 已删除。0726 已发布镜像内
canonical-only N=256 与清理前 canonical 镜像 token/hidden `256/256` exact、
`max_abs_diff=0`、TP spread `0.0`，所以 compatibility removal regression 已关闭。
对同一 vanilla oracle raw 为 `240/256=93.75%`，低于历史 95% raw gate；不能写成
raw PASS，也不能外推成完整 Main+MTP serving 已平替。该结论不覆盖 N1-S-0234。
2026-07-27 已删除 retired 0724 unroll source、rollback selector 和自定义
Main module/name 参数；后续 blocker 定位只允许 canonical。

**当前 live serving blocker**：

> **2026-07-22 更新（device 0162, stepfun/develop `a632c42e` = hidden-only 集成）**：
> per-layer KV bridge **已接线并可跑 multi-step**——`_stage_main_hidden_only --steps 8`
> 用 per-step `block_table/slot_mapping/seq_lens` 常驻 decode。**多步精度 = NORMAL,
> pypto == vanilla vLLM 逐 token 一致**。已 device 定论：重启 vanilla W8A8 oracle
> （containerd/k8s 容器,`sudo nsenter -t <sleep-infinity-pid> bash /logs/start_8000_oracle.sh`,
> cards 0-7,port 8000）并查它自己对相同 bare-token context 的下一 token 分布——
> `[6127]→303`、`[6127,303]→1207`、`[6127,303,1207]→`**`6127`**（北京 -2.8685;
> 19384 题目是 vanilla 的 #2 -2.9935）。**vanilla step2 自己就输出 6127,与 pypto 一致**;
> harness `DEFAULT_ORACLE_TOKENS[2]=19384` 是**过时/不同 setup 生成的常量**(step2 是
> near-tie,只有它对 BOS/template setup 敏感;step0/1 margin 大所以任何 setup 都对)。
> teacher-forced 8-step = 7/8(唯一 miss 就是这个 stale-oracle 的 step2)。严格自回归
> harness 显示 2/8 纯属"一次翻转污染后续输入"的 chain artifact,非精度问题。
> **结论:多decode精度 blocker 已解决,整网 forward 数值忠实、逐 token 对齐 vanilla。**
> 历史"near-tie/未完全正常"表述作废。详见 memory `n1_multidecode_neartie_faithful_a632c42e`。

> **2026-08-04 更新（device 0162 cards 0-7, 分支 `feat/vllm-live-front-wiring`, 镜像
> `vllm-pypto:wave5-local`）——co-resident 整条 round-trip WIRING 已在设备上打通到
> decode ABI；下一墙 = live prefill**：
>
> 本 session 把 tail-only vLLM（`--load-format pypto` + `PYPTO_STEP3P5_TAIL_ONLY=1`,
> kept=3/skipped=109539）与常驻 whole-net sidecar **同卡 0-7 co-resident** 跑通到
> 真实请求进入 gate。解决/验证：
> 1. **block_table ABI 修复**：live vLLM 默认 `max-model-len`(~262144)→ block table
>    宽 2048 → flat `16×2048=32768`，而 compiled `BLOCK_TABLE_FLAT_DYN(BTF)=512`
>    (`USER_BATCH_DYN=16 × ceil(MAX_SEQ 4096/128)=32`)。**修法：vLLM launcher 固定
>    `--max-model-len 4096`** 让 block table 宽=32 → flat 512 = BTF。（`whole_decode_holder.py:536`
>    的 `table.numel()!=BTF` 校验。）
> 2. **G4 co-tenancy NO_HCCL 补丁不在任何发布镜像里**：release/Wave 镜像都为 standalone
>    canonical（cards 8-15）构建，`comm_hccl.cpp` 无 `SIMPLER_COMM_NO_HCCL` gate（image
>    `.so` grep=0），故 env flag 空转、sidecar `comm_init` 撞 `HcclCommInitRootInfo
>    failed: 7`。**修法：从 git `878f3742` 重建 patch，在 wave5 镜像内 patch
>    `src/a2a3/platform/onboard/host/comm_hccl.cpp`（5 处 anchor）+ `build_runtimes
>    --platforms a2a3` 重编，把 patched `libhost_runtime.so`（host_build_graph +
>    tensormap_and_ringbuffer）mount 进 sidecar**。重建后 gate_count=1。
>    → device 验证：sidecar 常驻、weight+KV IPC 零拷贝导入、`whole_decode_step3p5`
>    在 8 chip **co-resident 编译+prewarm+run**（`simpler_run` device_wall spans，
>    ~50ms），**0 次 HcclCommInitRootInfo failed、0 次 507018、无 card poison / 无
>    force-reset**（co-tenancy hazard 未触发）。
> 3. **下一墙 = live prefill（H4，非 wiring 缺陷）**：真实请求首个 forward 是 **prefill**
>    (`AscendAttentionState.PrefillNoCache`, `prompt_token_ids_len=1, num_computed_tokens=0`)；
>    `whole_decode_step3p5` 是 **decode-only**，gate（`vllm_monkey_patch.py`
>    `classify_decode_gate`）在 `extract_pypto_decode_plan` 上正确 fail-closed
>    (`DecodeMetadataError: unsupported attention state PrefillNoCache`)，EngineCore 退出。
>    要端到端出 token，必须先 prefill 填 KV 再 decode——需要 wire prefill program/bridge
>    或 prefill KV-fill 路径。跟踪见 phase 28 H4。
>
> 未推送：分支 `feat/vllm-live-front-wiring` 3 个 fix commit（`a9573180` load_format
> coerce、`c9af2a6a` MTP profile no-op hoist、`d35a71bf` KVPOOL MTP-optional）待用户
> 授权后 HTTP/1.1+PAT push。NO_HCCL patch 目前只在镜像内重建（`nohccl_patch.py` +
> `build_nohccl.sh` 在 0162 `live-front-wiring/patches/`），未 bake 进发布镜像。

1. **live per-layer paged-KV bridge**：standalone canonical path 已有
   resident per-layer KV 并完成多步回归；缺口是从真实 vLLM paged KV pool
   导入 per-layer BF16 slice，并按请求/step 传
   `block_table`/`slot_mapping`/`seq_lens` 与 dynamic batch metadata。
2. **3-way HBM / redundant weights**：vLLM W8A8 常驻权重 + exporter 的 whole-net INT8
   IPC 权重 + whole-net runtime working set 同时存在时，0162 live 报 `207001` OOM。
   不是 standalone stall，也不是调小 ring heap 能解决；需消除 vLLM/exporter 重复权重，
   或做等价 in-place/shared-weight 方案。
3. **独立 live front + 同代 MTP absolute gate**：sidecar 默认 canonical Main wiring 已完成，
   但真实 online request 接管、current Main 输出进入 MTP 后的 absolute
   token/hidden oracle 尚未闭环。

**解除条件**：完成 live paged-KV/dynamic batch 接线 + device 验证；解决重复权重与
live HBM 预算；完成独立 live-front A/B 和同代 MTP absolute gate。详见
[`planning/phases/28-n1-live-integration.md`](planning/phases/28-n1-live-integration.md)、
[`design/vllm-pypto/02-detailed-design.md`](design/vllm-pypto/02-detailed-design.md)。

> **历史定位结论降级**：旧文档把 PUSH/TPUT/某 stuck kernel/signal bit 写成唯一硬件根因
> 的结论已撤下（详见 [`postmortems/12-integration-churn-meta.md`](postmortems/12-integration-churn-meta.md)）。

---

## 🟡 Phase 20 production backend 未接入（功能）

**症状**：BF16/W8A8 decode 与 W8A8 prefill 的结论来自 vLLM eager detail dump + PyPTO
reference/detail/final-logits replay——证明数值路径可对齐，但**不是** production backend。

**根因**：`Step3p5DecodeFwd`/prefill runner、`Step3p5Model.forward` monkey-patch、runtime
weight bundle 注入、KV/block_table/slot_mapping ABI 尚未接成在线请求路径。

**解除条件**：① `config_align.py` 校验 vLLM `hf_config` vs PyPTO constants；②
`weight_translate.py` 支持 vLLM module → PyPTO bundle；③ runner 接入 vLLM 请求路径
（decode-only 能返回 token）；④ Phase 21 在线 L1/L2/L3 gate 通过。详见
[`design/vllm-pypto/`](design/vllm-pypto/)。**Owner**：未指派。

---

## 🟡 Prefill MoE L1 overflow（TASK-29）

**症状**：`models/step3p5/prefill_moe.py` 编译时 `moe_gate_up` L1 buffer overflow
（~5MB > 限）。prefill MoE 层编译不过。**根因**：prefill 在宽 SEQ 维（如 4096 vs decode
BATCH=16）上跑，decode UB 装得下的 MoE 结构到 prefill 爆 L1。

**解除条件**：重设计 prefill_moe，加 multi-step gate_up chunking（~1-2 周）。**decode-only
perf 绕路**：合成数据预填 KV 到目标 length，跳 prefill 测 decode-only TPS/ITL（见
[`archive/completed-phases/22-perf-baseline.md`](archive/completed-phases/22-perf-baseline.md)）。
gate Phase 17。**Owner**：未指派。

---

## 🟡 head_gate 剩余：L1 A/B 暴露的整网 MoE NaN

**已解部分**：`matmul_acc N=16` codegen bug 已修，on-device head-gate 已在
`attention_full/swa` Scope 1.f 恢复（gate_r 承载 layer-independent block-diag R）——
详见 [`postmortems/09-attention-multiposition-corruption.md`](postmortems/09-attention-multiposition-corruption.md)。

**剩余**：L1 ctx=1 A/B（tid 6127 → 期望 303）曾 `logits=nan`。bisect 定位 NaN 在 42 层
INT8 W8A8 routed-MoE（单层即复现），非 attention——属 gap-5 territory，见
[`postmortems/10-gap5-attention-quant-scope.md`](postmortems/10-gap5-attention-quant-scope.md)。
**解除条件**：per-op MoE dump 定位首个 NaN 算子 → 修 → 重跑 L1。**Owner**：TASK-L 上游。

---

## 🟢 Deferred — MTP 集成进 decode

3 个 MTP 层有 kernel（`models/step3p5/mtp.py`）但没拼进 decode。speculative decoding
吞吐倍率，**不在 Phase 2 关键路径**。gate Phase 22 baseline 后 2-4 周。**Owner**：未指派。

---

## 🔴 Final e2e precision prerequisites

**gate**：最终验收"端到端精度正确且无阻塞"。预检
`python tools/step3p5/e2e_precision_readiness.py --batch 2`（host smoke 全绿）。
**剩余前置**：① 真实权重目录挂载；② vLLM/stepcast 原生 Step3p5 代码可见；③ live backend
接入（Phase 20）；④ head_gate vLLM parity 策略。**解除条件**：真实权重 + vLLM oracle 可见 +
`decode_fwd` 接线完成 + 8 rank logits shard concat 对齐 vLLM top-k。

---

## 怎么加新 blocker

1. 按严重度（🔴 Critical / 🟡 功能 / 🟢 Deferred）插入。
2. 写 症状 / 根因 / 当前状态 / 解除条件 / 链接。
3. 链回症状首次出现处（phase doc 或 [`archive/milestones-2026-Q2.md`](archive/milestones-2026-Q2.md)）。
4. 在 [`STATUS.md`](STATUS.md) blocker 摘要表加一行。
5. 解决后 → 删本节 + 建 [`postmortems/`](postmortems/) 复盘。
