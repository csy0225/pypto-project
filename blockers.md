# 活跃 Blocker

阻塞项目进展的 **open** issue 的 SSOT。每条：症状 / 根因 / 当前状态 / 解除条件 / 链接。

**协议**：blocker 解决时，**删掉本文件这一节** → 到 [`postmortems/`](postmortems/) 建一篇
五段复盘（模板 [`postmortems/TEMPLATE.md`](postmortems/TEMPLATE.md)）+ 更新
[`STATUS.md`](STATUS.md) blocker 摘要。已解问题不留在本文件。

**已解 blocker 的复盘去向**：见 [`postmortems/README.md`](postmortems/README.md)。
如 507899/507018、co-tenancy(G4)、tmov Vec-LHS、gate_topk、多程序 co-prepare 死锁、
gap-5、scheduler-timeout、attention 乱码、G5b import_ipc、swa_moe const-fold 等均已归档。

**最后检视**：2026-07-18。

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

**边界**：0162 standalone N=1 canonical stall gate 已关闭（`dispatch pull + combine pull`，
真 W8A8 + 真 KV-IPC + P42，连续 20/20 `argmax=303`，见
[`reference/canonical-test.md`](reference/canonical-test.md)）。该结论不覆盖 N1-S-0234。

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

1. **per-layer KV bridge**：whole-net standalone substrate 仍是 45 层共享的
   `k_cache/v_cache`，只覆盖 ctx=1。live multi-token decode 需从 vLLM paged KV pool
   导入 per-layer BF16 KV slice，并按 decode step 传 `block_table`/`slot_mapping`/`seq_lens`。
   （**注**：hidden-only `a632c42e` 已在 standalone 侧实现 per-step KV 常驻并跑通 8 步；
   live 侧从真实 vLLM paged pool 导入仍待接。）
2. **3-way HBM / redundant weights**：vLLM W8A8 常驻权重 + exporter 的 whole-net INT8
   IPC 权重 + whole-net runtime working set 同时存在时，0162 live 报 `207001` OOM。
   不是 standalone stall，也不是调小 ring heap 能解决；需消除 vLLM/exporter 重复权重，
   或做等价 in-place/shared-weight 方案。token-exact live A/B 尚未完成。

**解除条件**：完成 per-layer KV model-side 改造 + device 验证；解决重复权重与 live HBM
预算；再按 Phase 28 标准做 token-exact A/B 验收。详见
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
