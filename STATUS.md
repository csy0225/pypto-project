# 实时状态（STATUS）

> **T1 = 只写此刻为真。** 每条带 sha / digest / 门结论；历史过程去
> [`archive/milestones-2026-Q2.md`](archive/milestones-2026-Q2.md)，长证据去
> [`benchmark/`](benchmark/)，未决去 [`blockers.md`](blockers.md)。
>
> **最后更新：2026-08-25。预算 ≤130 行。**

## 0. Agent 判定当前状态的强制顺序

1. 先读本文件与 [`planning/handoff.md`](planning/handoff.md)，再查 GitHub 远端
   `refs/heads/stepfun/develop`；不得用本地 branch/worktree 名推断当前 tip。
2. 区分源码 tip（SRC）、镜像内容（IMG）和部署运行合同；三者不能互相代替。
3. 镜像只认 manifest/config digest、fresh pull 与 immutable gate。
4. 性能数字除 digest 外还必须绑定**完整有效环境变量**。当前 r10 发布性能合同为
   `PYPTO_H4_RESIDENT=all`；镜像 Config 未 bake 该值。

## 1. 当前 SRC（五仓远端 `stepfun/develop`）

| 仓库 / 组件 | 当前值 |
|---|---|
| pypto | `519b588a7a6461cac0e443e853accf29479c1d15` |
| pypto-lib | `fe641929dbf959d887ad111f3bd7cac0b73fa34b` |
| pto-isa | `cd4a3d3f7a1a27fcfe536f617e9bca3008929664` |
| PTOAS source / binary | `307d0484a9e7d5e36f01b253d2bebe4d2f45fe81` / `v0.57` |
| simpler | `85a82c454074c069315ed6485033c3c2b136e562` |
| vLLM patch | `1b3e538c35999e62b6d24e0651b3a85b7d16c826` |
| Main 入口 | `models.step3p5.decode_fwd:whole_decode_step3p5` |
| `decode_fwd.py` sha256 | `da36c09dc275838ee364f76342d74717338ef313d912ba2b372808530489dd14` |

五个远端 ref 已于 2026-08-25 逐项复核。`simpler` 是非 fast-forward 更新，旧 tip
`e2efebcb…` 已备份到 `backup/stepfun-develop-pre-upgrade-20260824-e2efebcb`。
pypto-lib 已用 exact lease 从 `bf3ff440` fast-forward 到 `fe641929`，远端
`refs/heads/stepfun/develop` 已复核。

⚠ 本页的 **upgrade image r9** 是发布序号；它与历史 MoE dispatch 候选 **R9 NO-GO**
不是同一对象，后者仍不得恢复。

## 2. 当前 release-admitted IMG

```text
tag:      hub.i.basemind.com/stepcast/vllm-pypto:stepfun-upgrade-20260825-r10
manifest: sha256:8510f30e1f2a2f2edcaa834c831165b349a4aca1212b655ca2a02ed6b3e9907b
config:   sha256:38ebba41d6aa0c49940c03e2e7c6fa42d85b61d631c143d38944683d0c657b5f
```

最终合同 `71/71 PASS`：
`0162:…/r10-release-admission-20260825-150350/release_contract.json`
（SHA256 `bcdd0b11d346e450dca49b8434544de5566b7fc0ad1a38c715815a41958dafca`）。
r9 manifest `b637f00c…90f6` 保留为回退镜像。

## 3. r10 权威门结论

| 门 | 结论 |
|---|---|
| Precision | **PASS** `127/128 = 99.21875%`，门限 95%，唯一 mismatch step `94`；accepted oracle SHA `eb561cf8…a241c2` |
| H4 数值等价 | **PASS**：`all/none` output token `128/128` exact，hidden tensor `256/256 torch.equal` |
| Full liveness | **PASS**：Main 8-step、MTP single、MTP batch16 全部 `rc=0` |
| ITL，发布合同 `all` | **PASS**：64K/1000 p50 `21.742 ms`、mean `22.296`、p99 `27.286` |
| immutable A/B/A | **PASS**：p50 `22.524 / 21.821 / 22.580 ms`，midpoint `22.552`，收益 `−0.731 ms / −3.241%`；hidden SHA `567b206b…e27f03e`、token `14371` exact |
| 六档 BS correctness | **PASS**：BS `1/2/4/7/8/16` hidden exact `6/6`，arm health `12/12`，TP spread `0`、inactive exact zero |
| 前五层 hidden | **PASS**：L3/L4 `torch.equal=true`，SHA `5aca3716…108ee8b9` / `0308be31…e400a4` |
| 前五层 DFX | **PASS**：8/8 chip/merged swimlane，analyzer 与 outer admission `pass=true`、`blockers=[]` |
| Final contract | **PASS**：`71/71`，pypto-lib develop exact-lease 已同步 |

前五层是 `L0_full_dense`、`L1_swa_dense`、`L2_swa_dense`、`L3_swa_moe`、
`L4_full_moe`。outer 产物：
`0162:…/r10-outer-swimlane-dfx-20260825-141817/`。完整发布证据见
[`benchmark/2026-08-25-moe-fusion-image-release.md`](benchmark/2026-08-25-moe-fusion-image-release.md)。

## 4. 当前性能与功能边界

- H4 `all` 把 4 个 RoPE 表 + 4 个 gate-R 常量一次上传并常驻，日志为
  `8 args, 99.64 MiB/rank`；`bind.args` p50 `6.461 → 0.063 ms`。
- `21.742 ms` 是 **r10 + `PYPTO_H4_RESIDENT=all`** 的长跑性能，不是镜像默认值。
- 六档门是 correctness gate，计时仅为 warmup `1` / iters `1` 诊断；BS8
  `35.124 → 40.801 ms`、BS16 `51.317 → 51.868 ms` 单次回退，不得写成多 BS 性能全赢。
- r10 raw DFX schema 是 `chip_swimlane_records.json`；不得复制/改名伪造旧
  `l2_swimlane_records.json`。

## 5. 当前功能边界

- production live backend / live prefill / 真实 paged-KV 仍未闭环；本次 release admission
  不等价于 Phase 28 live serving 已完成。
- standalone BS16×每请求 64K 仍受 HBM 容量门限制，不能把 OOM 当整网性能数据。

## 6. 当前下一步

1. 把 `PYPTO_H4_RESIDENT=all` 接入正式 deployment，并在 exact launcher 上重跑
   startup contract + 64K ITL。
2. 继续 Phase 28 live serving
   与 paged-KV/dynamic batch，消除 live 3-way HBM 重复权重
   → live token-exact A/B。
3. 收口 `UPSTREAM-NOTIFY-FENCE`；任何拉近 payload store 与 credit 的 AR 改动，在 fence
   落地前继续禁止。

## 7. 机器状态口径

0162：driver `25.5.2` / firmware `7.8.0.7.220` / CANN `9.0.0-beta.1`。
每次作业前重新用 `sudo -n fuser` + `npu-smi info -t proc-mem` 双查，不能沿用旧 session
的空闲结论；禁止 reset NPU 或破坏性 kill。

## 8. Blocker 摘要

| Blocker | 当前缺口 | 严重度 / gate |
|---|---|---|
| H4-DEPLOY-CONTRACT | r10 发布性能要求 `PYPTO_H4_RESIDENT=all`，正式 launcher 尚未发现显式 env | 🔴 生产性能口径 |
| UPSTREAM-NOTIFY-FENCE | notify 的 invalidate 在 payload drain 前，缺 pre-CMO `PIPE_ALL` | 🔴 AR correctness |
| N1-S-0234 | 0234 同步后 whole-net stall，尚未独立复核 | 🔴 0234 可用性 |
| Phase 28 live | live prefill + paged-KV + 3-way HBM 未闭环 | 🔴 live serving |
| DEPLOY-REPRO | 历史 dirty 镜像的剩余回溯未完成 | 🟡 可复现性 |
| Phase 20 backend / Prefill MoE / head_gate / final e2e | 见 [`blockers.md`](blockers.md) | 🟡 功能/精度 |
| MTP 集成进 decode | speculative 吞吐，不在当前关键路径 | 🟢 Deferred |
