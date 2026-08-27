# 接力上下文（Handoff）

> **T6 = 纯指针 + “现在接什么”。最后更新：2026-08-27。预算 ≤90 行。**
>
> 当前真相 → [`../STATUS.md`](../STATUS.md)　落地台账 →
> [`../progress/landed.md`](../progress/landed.md)　未决 →
> [`../blockers.md`](../blockers.md)　r12 证据 →
> [`../benchmark/2026-08-27-whole-step-host-graph-submit-r12-release.md`](../benchmark/2026-08-27-whole-step-host-graph-submit-r12-release.md)

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

whole-step 收益证据是 **r11 immutable 基座上的 source-overlay A/B/A**：
ITL `21.6805 → 21.1150 ms`（`−2.608%`）、graph build `−44.429%`、
graph→first-runner `−47.936%`、submit envelope `−23.887%`。正式合同仍为
`serial-eight-rank`、`group_size=1`、`group_submit=0`；不能写成 r12 immutable
性能 A/B/A，也不能写成 native group-submit 已过门。`bind.args` 仅占候选 ITL
`0.259%` 且 `no_clear_change`，不再优化。

## 现在接什么（按优先级，只有三条）

### 1. 把 H4 运行合同接入正式 deployment

r12 Config Env 仍未 bake `PYPTO_H4_RESIDENT=all`。在正式 launcher 显式接线后，
按 exact deployment 重跑 startup contract + 64K ITL。

### 2. 继续 Phase 28 live serving

继续 live prefill → paged-KV/dynamic batch → 消除 3-way HBM → live token-exact A/B。

### 3. 收口 upstream notify fence

任何拉近 remote payload store 与 credit notify 的改动，必须先补
`UPSTREAM-NOTIFY-FENCE` 的 pre-CMO `PIPE_ALL`。

## 操作约束

- 性能数字必须绑定 manifest/config、完整命令、effective env 与 overlay 身份。
- r12 只声明 dep-only DFX，不声明 whole-swimlane。
- 每次占卡前重新查锁、`nerdctl ps`、`sudo -n fuser` 和 NPU process。
- 禁止 NPU reset、破坏性 kill、source/runtime overlay 冒充 image gate。
