# 实时状态（STATUS）

> **T1 = 只写此刻为真。** 每条带 sha / digest / 门结论；历史过程去
> [`archive/milestones-2026-Q2.md`](archive/milestones-2026-Q2.md)，长证据去
> [`benchmark/`](benchmark/)，未决去 [`blockers.md`](blockers.md)。
>
> **最后更新：2026-08-24。预算 ≤130 行。**

## 0. Agent 判定当前状态的强制顺序

1. 先读本文件与 [`planning/handoff.md`](planning/handoff.md)，再查 GitHub 远端
   `refs/heads/stepfun/develop`；不得用本地 branch/worktree 名推断当前 tip。
2. 区分源码 tip（SRC）、镜像内容（IMG）和部署运行合同；三者不能互相代替。
3. 镜像只认 manifest/config digest、fresh pull 与 immutable gate。
4. 性能数字除 digest 外还必须绑定**完整有效环境变量**。同一 r9 digest 在
   `PYPTO_H4_RESIDENT=none/all` 下分别约 `27.8/22.3 ms`。

## 1. 当前 SRC（五仓远端 `stepfun/develop`）

| 仓库 / 组件 | 当前值 |
|---|---|
| pypto | `519b588a7a6461cac0e443e853accf29479c1d15` |
| pypto-lib | `bf3ff4400082f74b35fbdb5b3e0f5f4bf51ce373` |
| pto-isa | `cd4a3d3f7a1a27fcfe536f617e9bca3008929664` |
| PTOAS source / binary | `307d0484a9e7d5e36f01b253d2bebe4d2f45fe81` / `v0.57` |
| simpler | `85a82c454074c069315ed6485033c3c2b136e562` |
| vLLM patch | `1b3e538c35999e62b6d24e0651b3a85b7d16c826` |
| Main 入口 | `models.step3p5.decode_fwd:whole_decode_step3p5` |
| `decode_fwd.py` sha256 | `671a5df8a07e09303c398871fd1772f306b2998ea3e8168048588de6cc3fa323` |

五个远端 ref 已于 2026-08-24 逐项复核。`simpler` 是非 fast-forward 更新，旧 tip
`e2efebcb…` 已备份到 `backup/stepfun-develop-pre-upgrade-20260824-e2efebcb`。

⚠ 本页的 **upgrade image r9** 是发布序号；它与历史 MoE dispatch 候选 **R9 NO-GO**
不是同一对象，后者仍不得恢复。

## 2. 当前 IMG（已发布）

```text
tag:      hub.i.basemind.com/stepcast/vllm-pypto:stepfun-upgrade-20260824-r9
manifest: sha256:b637f00c66d4dc976c053c617d2e19e6d6d66f68f4bef30250984da7a71690f6
config:   sha256:f6c8f72eecad0a9d40d0c4ea55afaab09dd4e2f5fe54d6a091e332465e421dae
```

- registry push、raw manifest/config 校验与 fresh pull 均 PASS；
- 镜像内 pins 与 §1 完全一致；
- 本次升级 release admission 在 0162 为 `pass=true`；
- **性能部署合同必须显式设置 `PYPTO_H4_RESIDENT=all`**。镜像 Config Env 没有
  bake 该值；直接启动的默认值是 `none`。

最终合同：
`0162:/mnt/persist/chensiyu/workspace/upgrade-20260821/r9-release-admission-20260824-151848/release_contract.json`
（SHA256 `1cd646e31cd6ce4dd0f3817219c297690b5ab1d355ab47c71eaafe489b2a08a6`）。
版本矩阵见 [`deployment/version-matrix.md`](deployment/version-matrix.md)。

## 3. r9 权威门结论

| 门 | 结论 |
|---|---|
| Precision | **PASS** `127/128 = 99.21875%`，门限 95%，唯一 mismatch step `94`；accepted oracle SHA `eb561cf8…a241c2` |
| H4 数值等价 | **PASS**：accepted 与 alternate 两组 oracle 中，`all/none` 输出序列各自完全一致 |
| Full liveness | **PASS**：Main 8-step、MTP single、MTP batch16 全部 `rc=0` |
| ITL，默认 `none` | 64K/1000 p50 `27.812 ms`、mean `28.164`、p99 `32.583` |
| ITL，发布合同 `all` | **PASS**：64K/1000 p50 `22.253 ms`、mean `22.426`、p99 `27.206` |
| 前五层 hidden | **PASS**：L3/L4 `torch.equal=true`，SHA `5aca3716…108ee8b9` / `0308be31…e400a4` |
| 前五层 DFX | **PASS**：8/8 rank `chip_swimlane_records.json`，analyzer `pass=true`、`blockers=[]`，recv_meta ready |

前五层是 `L0_full_dense`、`L1_swa_dense`、`L2_swa_dense`、`L3_swa_moe`、
`L4_full_moe`。正式产物：
`0162:…/upgrade-20260821/outer-swimlane-r9-h4-20260824-151416/`。

DFX analyzer 按职责仍保留 `PENDING_EXTERNAL_GATE/publication_allowed=false`，因为它不消费
outer hidden-state；不得手改。最终 outer admission 已独立消费 L3/L4 exact 并由 release
contract 给出整体 `pass=true`。

完整证据与两臂 ITL 对账：
[`benchmark/2026-08-24-upgrade-r9-release.md`](benchmark/2026-08-24-upgrade-r9-release.md)。

## 4. 当前性能与功能边界

- H4 `all` 把 4 个 RoPE 表 + 4 个 gate-R 常量一次上传并常驻，日志为
  `8 args, 99.64 MiB/rank`；`bind.args` p50 `6.461 → 0.063 ms`。
- `22.253 ms` 不是镜像默认性能，而是 **r9 + `PYPTO_H4_RESIDENT=all`** 的性能。
- r9 raw DFX schema 是 `chip_swimlane_records.json`；不得复制/改名伪造旧
  `l2_swimlane_records.json`。

## 5. 当前功能边界

- production live backend / live prefill / 真实 paged-KV 仍未闭环；本次 release admission
  不等价于 Phase 28 live serving 已完成。
- standalone BS16×每请求 64K 仍受 HBM 容量门限制，不能把 OOM 当整网性能数据。

## 6. 当前下一步

1. 把 `PYPTO_H4_RESIDENT=all` 写入正式 serving launcher / deployment manifest，并在
   exact deployment 上重跑 startup contract + 64K ITL；或改默认值后重发镜像。
2. 继续 Phase 28：live prefill → live paged-KV/dynamic batch → 消除 live 3-way HBM 重复权重
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
| R9-H4-DEPLOY-CONTRACT | r9 默认 `none≈27.8 ms`，发布性能 `all=22.253 ms`，正式 launcher 尚未发现显式 env | 🔴 生产性能口径 |
| UPSTREAM-NOTIFY-FENCE | notify 的 invalidate 在 payload drain 前，缺 pre-CMO `PIPE_ALL` | 🔴 AR correctness |
| N1-S-0234 | 0234 同步后 whole-net stall，尚未独立复核 | 🔴 0234 可用性 |
| Phase 28 live | live prefill + paged-KV + 3-way HBM 未闭环 | 🔴 live serving |
| DEPLOY-REPRO | 历史 dirty 镜像的剩余回溯未完成 | 🟡 可复现性 |
| Phase 20 backend / Prefill MoE / head_gate / final e2e | 见 [`blockers.md`](blockers.md) | 🟡 功能/精度 |
| MTP 集成进 decode | speculative 吞吐，不在当前关键路径 | 🟢 Deferred |
