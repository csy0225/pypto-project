# Phase 24 —— 整层 live 替换（一 key 整池 map + page_attention）

> **组件 pin**：2026-07-03，0162 cards 8-15，CANN `9.0.0` non-GA，ptoas-bin `v0.45`。
> 代码 artifacts：本目录 [`../artifacts/phase24/`](../artifacts/phase24/)（staging 验证脚本 +
> kvpool backend patch + worker one-key-pool patch 摘录）。
>
> 承接 [`23-zero-copy-kv-ipc-validation.md`](23-zero-copy-kv-ipc-validation.md)（step 1-5 机制验证）。
> 本 phase = 把机制接进 live 8001。

---

## 24.1 ✅ 整池映射（一 buffer / 一 key / map，无 OOM）

`artifacts/phase24/_stage_kvpool_backend.py`（部署 `/logs/pypto_patch/pypto_kvpool_backend.py`
+ sitecustomize `PYPTO_KVPOOL` autoload）。patch `vllm_ascend model_runner_v1.
_allocate_kv_cache_tensors`：把全部 45 层 K/V 合并进 **一个 `torch.npu.MemPool` buffer(90 MiB)**
→ **一个 export key** + `map[(layer,K|V)]=byte_off`（90 条），每 rank 写 `pypto_kvpool.key.rankN`
/ `pypto_kvpool_map.json.rankN`。

**验证**：8 rank 全 `consolidated 45 layers -> ONE buffer ... ONE key`；`Application startup
complete`，8001=200，连贯出 token，**无 OOM**（旧 per-tensor MemPool 在 4 层/90 pool 撞 207001）。

## 24.2 ✅ worker 源自 one-key pool（layer-0 A/B bad_ratio=0）

`artifacts/phase24/worker_one_key_pool_attn_setup.patch.txt`（改 `_stage_attn_worker.py::
_AttnService.attn_setup`）：读 `pypto_kvpool.key/map.rankN`（`PYPTO_KVPOOL_DIR`=host log 目录），
**一次 import 整池** → 每层 `k_dt/v_dt = DeviceTensor(pool_base + map_off, (1,4096,128), bf16)`。

**验证**：worker `KV one-key-pool import rank=0 pool_base=0x12c1c0000000 map_entries=90`
+ `one-key-pool layer=0 k_off=0 v_off=1048576`；layer-0 decode A/B **`bad_ratio=0.0000`**
（max|d|~0.001），连贯。零 per-layer key、零 per-tensor MemPool。

## 24.3 ✅ 全 45 层 attention（对齐 baseline，无 OOM）

`start_8001_kvpool_attn_all.sh`（`PYPTO_ATTN_LAYERS=0..44`, AB=0）+ 8 workers `--layer 0..44`。

**验证**：`one-key-pool layer=0..44`（45/45 imported），1080+ pypto attn SUCCESS，**零
507018/OOM/错误**；`max_tokens=2` 输出 `'，你会'` **逐字 == baseline 8000**。perf 36s/2tok
（45 socket round-trips/token）= **Phase 26 性能项，非正确性**。

**结论**：零拷贝 KV takeover + OOM-killer 在全 45 层 attention 上 live 闭环，输出对齐 vanilla。

## 24.4 ⚠ 整层（attn+MLP）—— 两处阻塞

**目标**：dense 层(0-2) = attn(pypto) + dense MLP(pypto) 整层；MoE 层(3-44) = attn(pypto) +
MoE(pypto)。

- **dense 整层 ❌ 卡「双 worker 同卡 co-tenancy」**：同时起 attn worker + dense-MLP worker
  （每卡 2 个 pypto `chip_process` + 1 个 vLLM TP worker）→ attn worker 首次 kernel run 崩：
  `chip_process dev=8: run_prepared failed with code 13`，attn socket 全灭 → 全 fallback vanilla
  （输出仍对齐 baseline，因 fallback=vanilla=正确，但 attn 实际没走 pypto）。24.2/24.3（attn-only
  同卡）无此问题，**唯一新变量 = 第二个 pypto chip_process**。两个 pypto chip_process 各自 ACL
  context + heap ring，同卡争用 → code 13。
  **正解 = 单 worker/rank 同时做 attn+MLP（共享一个 chip_process）**，是 worker 重构，不是 config。
  （attn 与 dense MLP 各自已独立验证：24.2/24.3 + 2026-06-28 dense MLP e2e。）
- **MoE 整层 ❌ 卡 507018**：MoE dispatch 的 AICPU stream sync timeout（见 simpler
  Device-Error-Codes wiki）。上游 device fault，`blockers.md` 既有 blocker。

## Status

- **24.1 ✅ 24.2 ✅ 24.3 ✅**（live 2026-07-03）：零拷贝一 key 整池 KV takeover，全 45 层
  attention 对齐 baseline，无 OOM。**Phase 24 核心达成。**
- **24.4 ⏸ 阻塞**：dense 整层需单-worker-per-rank 重构（消除双 chip_process 同卡争用）；
  MoE 整层卡 507018。
- **下一步**：(1) 单 worker 合并 attn+dense-MLP（解 co-tenancy）→ dense 整层闭环；(2) 507018
  → MoE 整层；(3) Phase 25 整网 host_orch 48 层融合（Wave-3）；(4) Phase 26 perf（消 socket
  round-trip）。

## 运行配置 / 复现

- 8001 启动脚本（均在 `/mnt/nvme1/chensiyu/logs/step3p5_910b_w8a8_v001/`）：
  `start_8001_kvpool.sh`（24.1 隔离）/ `start_8001_kvpool_attn.sh`（24.2 layer-0 A/B）/
  `start_8001_kvpool_attn_all.sh`（24.3 全 45 层）/ `start_8001_dense_wholelayer.sh`（24.4-dense，
  co-tenancy 阻塞）。
- worker：attn = `_stage_attn_worker.py server --layer <list> --kv-rows 4096`
  + `PYPTO_KVPOOL_DIR=<host log dir>`；dense-MLP = `tools.step3p5.pypto_mlp_worker
  --dense-layers 0,1,2`。
