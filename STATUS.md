# 实时状态（STATUS）

> **T1 = 只写此刻为真。** 每条带 sha / digest / 门结论；历史过程去
> [`archive/milestones-2026-Q2.md`](archive/milestones-2026-Q2.md)，长证据去
> [`benchmark/`](benchmark/)，未决去 [`blockers.md`](blockers.md)。
>
> **最后更新：2026-09-01。预算 ≤130 行。**

## 0. Agent 判定当前状态的强制顺序

1. 先读本文件与 [`planning/handoff.md`](planning/handoff.md)，再查 GitHub 远端
   `refs/heads/stepfun/develop`；不得用本地 branch/worktree 名推断当前 tip。
2. 区分源码 tip（SRC）、镜像内容（IMG）和部署运行合同；三者不能互相代替。
3. 镜像只认 manifest/config digest、fresh pull 与 immutable gate。
4. 性能数字必须绑定完整环境。2026-08-29 H4 deployment 门绑定 r12 digest；
   2026-09-01 matched candidate 绑定下方 candidate digest。正式 launcher 默认注入
   `PYPTO_H4_RESIDENT=all`；r12 Config Env 与代码默认仍未 bake，三层不能互代。

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
manifest: sha256:19f51d373c5f9d6171ccf3306f260066e873eda48efca23f5d77b4d6f5e64a7f
config:   sha256:7e5dd8683fda03e3e51a0b5217ae71ab82052173f3659db60fd689ea833ed6eb
pins:     pypto 655c7bda / pypto-lib a745ab659
```

matched immutable H4=`all` A/B/A 为 `21.617 / 20.516 / 21.257 ms`
（A=`14de+a745`，B=`655+a745`）；H4 PASS（gain `0.921 ms` >
required `0.616 ms`）；5-case extended admission v2 checks 16/16 PASS。尚不含
a745 `recv_meta` route publication，不能称 release IMG。
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
- 2026-08-29 H4 deployment 收口：r12 **source-default-all** matched
  `none/default/none` p50 `30.516/22.606/29.440 ms`，default=`all` 相对 midpoint
  收益 `7.372 ms / 24.591%`，
  三臂 hidden SHA `ee8ae6…db96a`、token `43640` exact；exact launcher 在父 env
  unset 下 64K/1000 p50 `20.973 ms`、RC=0。`none` 保留为回退。
- 旧 r13 `655+e6c7d8ec` 的 `21.562 ms` 是未匹配的旧栈；本轮不要拿它
  代表最新 `655+a745ab659`。local-owner reset 修复将 control prefix
  固定为 `46,080 B`，相对 full window `11,842,560 B`，B 臂
  `memset_all` p50 为 `462.277 us`。

## 6. 当前功能边界

- r12 immutable gate 不声明 whole-swimlane；dep-only DFX 只证明依赖图与 hidden/token exact。
- production live backend / live prefill / 真实 paged-KV 仍未闭环；release admission
  不等价于 Phase 28 live serving 已完成。
- standalone BS16×每请求 64K 仍受 HBM 容量门限制，不能把 OOM 当整网性能数据。
- candidate IMG 尚未按 a745 provenance 采集 route publication/route identity；
  旧 r12 count-only sidecar 不可复用。

## 7. 当前下一步

1. 以 canonical `655+a745` 重建 successor image，重新采集 a745
   `recv_meta` route publication，并完成完整 release contract。
2. 继续 Phase 28 live serving、paged-KV/dynamic batch 与 3-way HBM 收口。
3. 收口 `UPSTREAM-NOTIFY-FENCE`；任何拉近 payload store 与 credit 的 AR 改动，在 fence
   落地前继续禁止。

## 8. 机器状态口径

0162：driver `25.5.2` / firmware `7.8.0.7.220` / CANN `9.0.0-beta.1`。
2026-09-01 extended gate 后复核为 16/16 NPU process-free；container/task/runc/shim
全空。每次作业前仍须重新用 `sudo -n fuser` + `npu-smi info -t proc-mem` 双查；
containerd/BuildKit 与驱动内核线程不是占卡 workload。

## 9. Blocker 摘要

| Blocker | 当前缺口 | 严重度 / gate |
|---|---|---|
| A745-ROUTE-PUBLICATION | candidate `655+a745` 尚无匹配 `recv_meta` sidecar；旧 r12 sidecar schema/provenance 不可复用 | 🔴 IMG 准入 |
| UPSTREAM-NOTIFY-FENCE | notify 的 invalidate 在 payload drain 前，缺 pre-CMO `PIPE_ALL` | 🔴 AR correctness |
| N1-S-0234 | 0234 同步后 whole-net stall，尚未独立复核 | 🔴 0234 可用性 |
| Phase 28 live | live prefill + paged-KV + 3-way HBM 未闭环 | 🔴 live serving |
| DEPLOY-REPRO | 历史 dirty 镜像的剩余回溯未完成 | 🟡 可复现性 |
| Phase 20 backend / Prefill MoE / head_gate / final e2e | 见 [`blockers.md`](blockers.md) | 🟡 功能/精度 |
| MTP 集成进 decode | speculative 吞吐，不在当前关键路径 | 🟢 Deferred |
