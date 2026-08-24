# 接力上下文（Handoff）

> **T6 = 纯指针 + “现在接什么”。最后更新：2026-08-24。预算 ≤90 行。**
>
> 当前真相 → [`../STATUS.md`](../STATUS.md)　落地台账 →
> [`../progress/landed.md`](../progress/landed.md)　未决 →
> [`../blockers.md`](../blockers.md)　完整 r9 证据 →
> [`../benchmark/2026-08-24-upgrade-r9-release.md`](../benchmark/2026-08-24-upgrade-r9-release.md)

## 已完成的交付基线

```text
image:
  hub.i.basemind.com/stepcast/vllm-pypto:stepfun-upgrade-20260824-r9
  manifest sha256:b637f00c66d4dc976c053c617d2e19e6d6d66f68f4bef30250984da7a71690f6
  config   sha256:f6c8f72eecad0a9d40d0c4ea55afaab09dd4e2f5fe54d6a091e332465e421dae

stepfun/develop:
  pypto      519b588a7a6461cac0e443e853accf29479c1d15
  pypto-lib  bf3ff4400082f74b35fbdb5b3e0f5f4bf51ce373
  pto-isa    cd4a3d3f7a1a27fcfe536f617e9bca3008929664
  PTOAS      307d0484a9e7d5e36f01b253d2bebe4d2f45fe81
  simpler    85a82c454074c069315ed6485033c3c2b136e562
```

最终合同：

```text
0162:/mnt/persist/chensiyu/workspace/upgrade-20260821/
  r9-release-admission-20260824-151848/release_contract.json
```

`pass=true`：registry fresh pull、precision `127/128`、Main/MTP liveness、
L3/L4 exact、8/8 chip swimlane 与五仓远端同步均已闭环。

## 现在接什么（按优先级，只有三条）

### 1. 把 H4 运行合同接入正式 deployment ← 最高优先

同一 r9 digest：

```text
unset/default none             64K/1000 p50 27.812 ms
PYPTO_H4_RESIDENT=all          64K/1000 p50 22.253 ms
```

镜像 Config Env 没有 bake `PYPTO_H4_RESIDENT`，代码默认 `none`。正式 serving
launcher / manifest 必须显式设置：

```bash
PYPTO_H4_RESIDENT=all
```

接线后在 exact deployment 上重跑 startup contract + 64K ITL，并把 effective env
写入产物。若选择改代码默认或 bake 镜像，必须重建、重新发布并重跑完整 r9 门。
跟踪：[`../blockers.md`](../blockers.md) `R9-H4-DEPLOY-CONTRACT`。

### 2. 继续 Phase 28 live serving

按顺序做：

1. live prefill / KV-fill；
2. vLLM paged-KV + dynamic batch；
3. 消除 vLLM 权重 + exporter 权重 + runtime working set 的 3-way HBM；
4. live token-exact A/B + Main→MTP absolute gate。

入口：
[`phases/28-live-integration.md`](phases/28-live-integration.md)、
[`../design/vllm-pypto/02-detailed-design.md`](../design/vllm-pypto/02-detailed-design.md)。

### 3. 收口 upstream notify fence

`remote_store` 紧接自身 credit `notify` 时，pre-CMO 缺 `PIPE_ALL`。任何合并波次、
按 peer 融合或拉近 payload/credit 的改动都必须先落 fence，再跑 A/B/A + precision。
入口：
[`../blockers.md`](../blockers.md) `UPSTREAM-NOTIFY-FENCE`、
[`../design/performance/06-upstream-asks.md`](../design/performance/06-upstream-asks.md)。

## 操作约束

- 性能数字必须同时绑定 manifest/config、完整命令与 effective env；digest 相同不代表运行合同相同。
- r9 正式 DFX raw schema 是 `chip_swimlane_records.json`，不得伪造旧
  `l2_swimlane_records.json`。
- analyzer 的 `PENDING_EXTERNAL_GATE` 是职责边界，outer admission 才消费 hidden；
  不得手改 analyzer 报告。
- 每次占卡前重新查锁、`nerdctl ps`、`sudo -n fuser` 与 NPU process；禁止 reset NPU
  或破坏性 kill。
- 历史 MoE dispatch 候选 R9 NO-GO 与 upgrade image r9 不是同一对象；前者不得恢复。
