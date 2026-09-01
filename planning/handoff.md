# 接力上下文（Handoff）

> **T6 = 纯指针 + “现在接什么”。最后更新：2026-09-01。预算 ≤90 行。**
>
> 当前真相 → [`../STATUS.md`](../STATUS.md)　落地台账 →
> [`../progress/landed.md`](../progress/landed.md)　未决 →
> [`../blockers.md`](../blockers.md)　r12 证据 →
> [`../benchmark/2026-08-27-whole-step-host-graph-submit-r12-release.md`](../benchmark/2026-08-27-whole-step-host-graph-submit-r12-release.md)　J3 CAND 证据 →
> [`../benchmark/2026-08-30-routed-gmm-active-worker-dual-latch.md`](../benchmark/2026-08-30-routed-gmm-active-worker-dual-latch.md)

## 当前交付基线

```text
release-admitted r12:
  tag      stepfun-upgrade-20260826-r12
  manifest sha256:ba42fd19b3af0144a835e95a4a6925ed89ea700624f696b221e93a54e6eb805d
  config   sha256:b36f0cec3a8b64e5e17e273c63d69694730bd8b904e69c2806c3d73a5233f08f
  pypto    14de90fd74b3c0716f94b9d4eafdd004d4eaed73
  pypto-lib e6c7d8ec34a05c3051ccf0dd169639f40f041a57

rollback r11:
  manifest sha256:401ead7da4f957f6532e380fa1a138eda733fe1dc04b40eabc67d79d62a67b12
```

最终合同：

```text
0162:/mnt/persist/chensiyu/workspace/moe-fusion-release-20260826-r12/
  release-admission-r12-20260826-224620/release_contract.json
sha256 511a545956aee4cef7264a74460bd04862846e377ef71eb01619ae4ddbf87f3a
```

`1844/1844 PASS`：registry/fresh pull、五仓 pin、baked runtime、Main H4
`all/none`、MTP BS1/BS16、dep-only DFX、显式设备与 non-privileged 安全合同均闭环。

2026-09-01 canonical SRC 已前进到 `pypto@655c7bda` +
`pypto-lib@a745ab659`（两次 exact-lease push 后 `ls-remote` exact）。
local-owner reset 回退修复的 matched candidate H4=`all` A/B/A 为
`21.617/20.516/21.257 ms`，5-case extended admission v2 PASS。
candidate image 尚未完成 a745 `recv_meta` publication，当前 release IMG
仍是 r12，不能把 candidate digest 写成 release-admitted。

whole-step 收益证据是 **r11 immutable 基座上的 source-overlay A/B/A**：
ITL `21.6805 → 21.1150 ms`（`−2.608%`）、graph build `−44.429%`、
graph→first-runner `−47.936%`、submit envelope `−23.887%`。正式合同仍为
`serial-eight-rank`、`group_size=1`、`group_submit=0`；不能写成 r12 immutable
性能 A/B/A，也不能写成 native group-submit 已过门。`bind.args` 仅占候选 ITL
`0.259%` 且 `no_clear_change`，不再优化。

H4 deployment contract 已于 2026-08-29 收口：三个 canonical launcher 默认注入
`PYPTO_H4_RESIDENT=all`，`none` 保留为回退；r12 source-default-all matched A/B/A 收益
`7.372 ms / 24.591%`，exact launcher 64K/1000 p50 `20.973 ms`。证据见
[`2026-08-29-h4-resident-deployment-contract.md`](../benchmark/2026-08-29-h4-resident-deployment-contract.md)。

当前 canonical `655+a745` 的 matched candidate H4=`all` A/B/A 为
`21.617/20.516/21.257 ms`，5-case extended admission v2 PASS；candidate image
尚未完成 a745 provenance 匹配的 `recv_meta` publication，当前 release IMG 仍为 r12。

## 现在接什么（按优先级，只有三条）

### 1. 为 `655+a745` 重建 successor IMG 并完成 route publication

先采集与 a745 provenance 绑定的 `recv_meta` + per-route identity sidecar，
再重跑 publication DFX 与完整 release contract；旧 r12 count-only sidecar 不可复用。

### 2. 继续 Phase 28 live serving

继续 live prefill → paged-KV/dynamic batch → 消除 3-way HBM → live token-exact A/B。

### 3. 收口 upstream notify fence

任何拉近 remote payload store 与 credit notify 的改动，必须先补
`UPSTREAM-NOTIFY-FENCE` 的 pre-CMO `PIPE_ALL`。

## 操作约束

- 性能数字必须绑定 manifest/config、完整命令、effective env 与 overlay 身份。
- r12 只声明 dep-only DFX，不声明 whole-swimlane；`655+a745` candidate
  correctness PASS 也不等于 IMG release admission。
- 每次占卡前重新查锁、`nerdctl ps`、`sudo -n fuser` 和 NPU process。
- 禁止 NPU reset、破坏性 kill、source/runtime overlay 冒充 image gate。
