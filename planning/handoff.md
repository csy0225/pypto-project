# 接力上下文（Handoff）

> **T6 = 纯指针 + “现在接什么”。最后更新：2026-08-25。预算 ≤90 行。**
>
> 当前真相 → [`../STATUS.md`](../STATUS.md)　落地台账 →
> [`../progress/landed.md`](../progress/landed.md)　未决 →
> [`../blockers.md`](../blockers.md)　r10 证据 →
> [`../benchmark/2026-08-25-moe-fusion-image-release.md`](../benchmark/2026-08-25-moe-fusion-image-release.md)

## 当前交付基线

```text
release-admitted r10:
  tag      stepfun-upgrade-20260825-r10
  manifest sha256:8510f30e1f2a2f2edcaa834c831165b349a4aca1212b655ca2a02ed6b3e9907b
  config   sha256:38ebba41d6aa0c49940c03e2e7c6fa42d85b61d631c143d38944683d0c657b5f
  pypto-lib fe641929dbf959d887ad111f3bd7cac0b73fa34b

rollback r9:
  manifest sha256:b637f00c66d4dc976c053c617d2e19e6d6d66f68f4bef30250984da7a71690f6
```

最终合同：

```text
0162:/mnt/persist/chensiyu/workspace/moe-fusion-release-20260825/
  r10-release-admission-20260825-150350/release_contract.json
sha256 bcdd0b11d346e450dca49b8434544de5566b7fc0ad1a38c715815a41958dafca
```

`71/71 PASS`：registry/compile/liveness/precision/ITL、immutable A/B/A、六档
correctness、outer DFX、develop sync 均闭环。A/B/A p50
`22.524 / 21.821 / 22.580 ms`，收益 `−0.731 ms / −3.241%`；三臂 hidden
SHA `567b206b…e27f03e`、token `14371` exact。64K/1000 p50 `21.742 ms`。

```text
stepfun/develop:
  pypto      519b588a7a6461cac0e443e853accf29479c1d15
  pypto-lib  fe641929dbf959d887ad111f3bd7cac0b73fa34b
  pto-isa    cd4a3d3f7a1a27fcfe536f617e9bca3008929664
  PTOAS      307d0484a9e7d5e36f01b253d2bebe4d2f45fe81
  simpler    85a82c454074c069315ed6485033c3c2b136e562
```

六档 BS `1/2/4/7/8/16` hidden exact `6/6`、health `12/12`。该门计时仅为
warmup `1` / iters `1` 诊断：BS8 `35.124 → 40.801 ms`、BS16
`51.317 → 51.868 ms`，不得宣称多 BS 性能全面收益。

## 现在接什么（按优先级，只有三条）

### 1. 把 H4 运行合同接入正式 deployment

r10 必须显式运行：

```bash
PYPTO_H4_RESIDENT=all
```

镜像 Config Env 未 bake 此值。正式 launcher 接线后重跑 startup contract +
64K ITL。跟踪：`H4-DEPLOY-CONTRACT`。

### 2. 继续 Phase 28 live serving

继续 live prefill → paged-KV/dynamic batch → 消除 3-way HBM → live token-exact A/B。

### 3. 收口 upstream notify fence

任何拉近 remote payload store 与 credit notify 的改动，必须先补
`UPSTREAM-NOTIFY-FENCE` 的 pre-CMO `PIPE_ALL`。

## 操作约束

- 性能数字必须绑定 manifest/config、完整命令和 effective env。
- 正式 DFX raw schema 是 `chip_swimlane_records.json`；outer admission 才消费 hidden。
- 每次占卡前重新查锁、`nerdctl ps`、`sudo -n fuser` 和 NPU process。
- 禁止 NPU reset、破坏性 kill、source/runtime overlay 冒充 image gate。
