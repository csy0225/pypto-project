# 实时状态（STATUS）

> **T1 = 只写此刻为真。** 每条带 sha / digest / 门结论；历史过程去
> [`archive/milestones-2026-Q2.md`](archive/milestones-2026-Q2.md)，长证据去
> [`benchmark/`](benchmark/)，未决去 [`blockers.md`](blockers.md)。
>
> **最后更新：2026-09-02。预算 ≤130 行。**

## 0. Agent 判定当前状态的强制顺序

1. 先读本文件与 [`planning/handoff.md`](planning/handoff.md)，再查 GitHub 远端
   `refs/heads/stepfun/develop`；不得用本地 branch/worktree 名推断当前 tip。
2. 区分源码 tip（SRC）、镜像内容（IMG）和部署运行合同；三者不能互相代替。
3. 镜像只认 manifest/config digest、fresh pull 与 immutable gate。
4. 性能数字必须绑定完整环境。r12 `20.973 ms`（1000 iter）与 candidate
   `20.516 ms`（100 iter）不是严格同合同；正式只认各自 matched 门。H4=`all` 由
   launcher/OCI 注入，r12/r15 Config Env 与代码默认均未 bake。

## 1. 当前 SRC（五仓远端 `stepfun/develop`）
| 仓库 / 组件 | 当前值 |
|---|---|
| pypto | `655c7bda7b0a0b495a3387b2570ea68c4a857a40` |
| pypto-lib | `a745ab659c68afca01de37870e29ccb9648d7c87` |
| pto-isa | `cd4a3d3f7a1a27fcfe536f617e9bca3008929664` |
| PTOAS source / binary | `307d0484a9e7d5e36f01b253d2bebe4d2f45fe81` / `v0.57` |
| simpler | `85a82c454074c069315ed6485033c3c2b136e562` |
| vLLM patch | `1b3e538c35999e62b6d24e0651b3a85b7d16c826` |
| Main 入口 | `models.step3p5.decode_fwd:whole_decode_step3p5` |
| `decode_fwd.py` sha256 | `cdb2bb26ddc0ca773bcddd0629bfc7bdfa5c426a334e26dde4364aacd867f348` |
2026-09-01 已用 `ls-remote` exact 复核 `pypto=655c7bda`、`pypto-lib=a745ab659`；
前者修复 local-owner persistent-reset ABI 回退，后者优化 routed GMM latch participant；
parent 分别为 `14de90fd`、`e6c7d8ec`。
## 2. 最新 candidate IMG（尚未 release-admitted）
```text
tag(local): stepfun-upgrade-20260902-a745-k8-r15
manifest:   sha256:19f51d373c5f9d6171ccf3306f260066e873eda48efca23f5d77b4d6f5e64a7f
config:     sha256:7e5dd8683fda03e3e51a0b5217ae71ab82052173f3659db60fd689ea833ed6eb
pins:       pypto 655c7bda / pypto-lib a745ab659
```

r14b/r15 本地 tag 指向同一 manifest/config；local build/audit PASS，但 registry tag
仍不存在且缺 push credential。matched immutable H4=`all` A/B/A 为
`21.617/20.516/21.257 ms`，reset 修复 gain `0.921 ms / 4.296%` > `0.616 ms`；
5-case extended admission v2 PASS。`20.516` 未刷新历史 source-overlay `20.172 ms`，
相对 r12 `20.973 ms` 的 `−0.457 ms` 仅方向性。route v2/v1 合同未闭合，不能称 release IMG。
## 3. 当前 release-admitted IMG
```text
tag:      hub.i.basemind.com/stepcast/vllm-pypto:stepfun-upgrade-20260826-r12
manifest: sha256:ba42fd19b3af0144a835e95a4a6925ed89ea700624f696b221e93a54e6eb805d
config:   sha256:b36f0cec3a8b64e5e17e273c63d69694730bd8b904e69c2806c3d73a5233f08f
pins:     pypto 14de90fd / pypto-lib e6c7d8ec
```

最终合同 `1844/1844 PASS`：
`0162:…/moe-fusion-release-20260826-r12/release-admission-r12-20260826-224620/release_contract.json`
（SHA256 `511a545956aee4cef7264a74460bd04862846e377ef71eb01619ae4ddbf87f3a`）。
r11 manifest `401ead7d…a67b12` 保留为直接回退镜像，r10 为更早回退。

## 4. r12 权威门结论
| 门 | 结论 |
|---|---|
| Registry / fresh pull | **PASS**：tag/digest raw manifest/config exact；digest-only audit/smoke/source-bake 与 non-privileged device smoke 全过 |
| Main precision H4 `all` | **PASS**：`126/128 = 98.4375%`，mismatch steps `[20,69]`，hidden finite、TP spread `0` |
| Main precision H4 `none` | **PASS**：同为 `126/128 = 98.4375%`，与 `all` 同一 mismatch，hidden finite、TP spread `0` |
| MTP BS1 / BS16 | **PASS**：tokens `[6178,410,303]`，三层 hidden pass rate 全 `1.0`、max abs diff `0` |
| Main dep-only DFX | **PASS**：8/8 `deps.json`，hidden/token exact，tail token `43640`，TP spread `0` |
| 设备/安全合同 | **PASS**：显式卡 `0–7`，保护 `8–15`；major/minor exact，`privileged=false`，无 source/runtime/core overlay |
| Immutable gate | **PASS**：runner SHA `9c9cb8bc…3234d8`；run contract `214946fc…1daf8`；admission `0b4b44f9…69ec` |
| Final contract | **PASS**：`status=release-admitted`，1844 checks 全 true，五仓远端 exact |

完整证据见
[`benchmark/2026-08-27-whole-step-host-graph-submit-r12-release.md`](benchmark/2026-08-27-whole-step-host-graph-submit-r12-release.md)。

## 5. 本镜像新增优化与性能边界
- `pypto@14de90fd` 对 prepared 参数生成稳定 descriptor/token，在参数身份未变化时复用
  TaskArgs signature/cache；未知或可变对象 fail-open 回完整校验，缓存有界。
- `free()` 与 submit 的 validation/graph-build proof window 互斥，避免缓存命中后使用已释放
  Buffer。正式生成的 host loop 仍是 serial 8-rank / 8 个独立 chip submit；
  commit 标题虽含 `parallelize rank submit`，本门未证明 parallel/native group-submit。
- r11 基座 source-overlay A/B/A（warmup `10`、measured `100`、8 ranks、ctx 64K）：
  ITL `21.6805 → 21.1150 ms`（`−0.5655 ms / −2.608%`）；
  graph build `4.0926 → 2.2743 ms`（`−44.429%`）；
  graph→first-runner `3.0981 → 1.6130 ms`（`−47.936%`）；
  runner wave `−0.2817 ms`，submit envelope `−0.2875 ms / −23.887%`；
  graph→chip done `−1.7624 ms / −8.443%`。
- `bind.args` 仅 `0.054449 → 0.054669 ms`（`+0.000220 ms`，候选 ITL 的
  `0.259%`，`no_clear_change`），不再投入优化。各 span 有重叠，不得相加。
- H4 deployment：r12 source-default-all A/B/A `30.516/22.606/29.440 ms`，
  exact launcher 64K/1000 p50 `20.973 ms`；H4=`all` 由 launcher 注入，`none` 可回退。
- K8 reset：固定 a745 的 immutable A/B/A `21.617/20.516/21.257 ms`，正式收益
  `0.921 ms / 4.296%`；control prefix `46,080 B`，B `memset_all=462.277 us`。
- 历史 a745 source-overlay 候选为 `20.172 ms`；当前 `20.516 ms` 不是历史新低。
  与 r12 `20.973 ms` 的差值小于 `0.616 ms` floor 且合同不同，不作显著提升声明。
  详见 [`benchmark/2026-09-02-k8-historical-performance-reconciliation.md`](benchmark/2026-09-02-k8-historical-performance-reconciliation.md)。
- 旧 r13 `655+e6c7d8ec=21.562 ms` 是未匹配旧栈，不能代表当前组合。

## 6. 当前功能边界

- r12 immutable gate 不声明 whole-swimlane；dep-only DFX 只证明依赖图与 hidden/token exact。
- production live backend / live prefill / 真实 paged-KV 仍未闭环；release admission
  不等价于 Phase 28 live serving 已完成。
- standalone BS16×每请求 64K 仍受 HBM 容量门限制，不能把 OOM 当整网性能数据。
- candidate IMG 尚未按 a745 provenance 采集 route publication/route identity；
  旧 r12 count-only sidecar 不可复用。

## 7. 当前下一步

1. 为已构建 r15 注入临时 registry write credential，完成 push/raw manifest/config/fresh pull。
2. 闭合 a745 `local-routes.v2` exporter 与 `recv-meta.v1` validator，并跑完整 route/release gate。
3. 补 r15 64K/1000 及 old/new immutable 同合同门；通过前不声明历史性能新高。
4. 继续 Phase 28 live serving，并收口 `UPSTREAM-NOTIFY-FENCE`。

## 8. 机器状态口径

0162：driver `25.5.2` / firmware `7.8.0.7.220` / CANN `9.0.0-beta.1`。
2026-09-02 文档/发布审计未启动新 workload；复核 16/16 NPU process-free，container/task/runc/shim 全空。每次作业前仍须重新用 `sudo -n fuser` + `npu-smi info -t proc-mem` 双查；
containerd/BuildKit 与驱动内核线程不是占卡 workload。

## 9. Blocker 摘要

| Blocker | 当前缺口 | 严重度 / gate |
|---|---|---|
| R15-PUBLICATION | registry 缺 push credential；route exporter=`local-routes.v2`、validator=`recv-meta.v1`；旧 r12 sidecar 不可复用 | 🔴 IMG 准入 |
| K8-HISTORICAL-PERF | matched 回退修复 PASS；`20.516 ms` 尚未刷新/复现历史 `20.172 ms` | 🟡 性能声明 |
| UPSTREAM-NOTIFY-FENCE | notify 的 invalidate 在 payload drain 前，缺 pre-CMO `PIPE_ALL` | 🔴 AR correctness |
| N1-S-0234 | 0234 同步后 whole-net stall，尚未独立复核 | 🔴 0234 可用性 |
| Phase 28 live | live prefill + paged-KV + 3-way HBM 未闭环 | 🔴 live serving |
| DEPLOY-REPRO | 历史 dirty 镜像的剩余回溯未完成 | 🟡 可复现性 |
| Phase 20 backend / Prefill MoE / head_gate / final e2e | 见 [`blockers.md`](blockers.md) | 🟡 功能/精度 |
| MTP 集成进 decode | speculative 吞吐，不在当前关键路径 | 🟢 Deferred |
