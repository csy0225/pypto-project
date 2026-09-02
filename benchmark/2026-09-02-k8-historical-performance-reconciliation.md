# 2026-09-02 · K8 回退修复与历史性能口径对账

> **状态：证据对账完成；本轮未启动新 device workload。**
> 本文统一解释历史 `20.973 ms`、`20.172 ms` 与当前 `20.516 ms`，并限定
> r15 可以使用的性能声明。所有只读核对均在 0162 完成。

## 1. 修正后的结论

| 日期 / 证据 | 源码栈 | 运行形态 | measured iterations | ITL p50 | 可以声明 |
|---|---|---|---:|---:|---|
| 2026-08-29 H4 exact launcher | `14de90fd+e6c7d8ec` | r12 immutable image，launcher 注入 H4=`all` | 1000 | `20.973 ms` | r12 deployment 合同在该单臂长门下成立 |
| 2026-08-29 routed-GMM B 臂 | `14de90fd+a745ab659` | r12 image + candidate source overlay | 100 | **`20.172 ms`** | a745 source-overlay A/B/A 收益成立；不是 IMG 结果 |
| 2026-09-01 K8 reset B 臂 | `655c7bda+a745ab659` | digest-only immutable image，non-privileged | 100 | `20.516 ms` | 655 相对同代 `14de+a745` 回退基线的修复收益成立 |

三组都使用 0162、cards `0-7`、BS1、ctx `65536`、`num_blocks=512`、
warmup `10` 和 `PYPTO_H4_RESIDENT=all`，但它们不是同一完整运行合同。

当前 matched immutable A/B/A 的正式结论是：

```text
A1  14de+a745  21.617 ms
B   655+a745   20.516 ms
A2  14de+a745  21.257 ms

baseline midpoint  21.437 ms
gain                 0.921 ms / 4.296%
required              0.616 ms
verdict                PASS
```

该结论证明 **local-owner persistent-reset 回退已修复**。它不证明当前候选刷新了
历史绝对最优：

- 相对 r12 exact-launcher `20.973 ms`，当前数值低 `0.457 ms / 2.179%`；
  但迭代数、镜像、runner、安全模型和 effective env 不同，且 `0.457 ms < 0.616 ms`，
  只能作方向性 sanity check；
- 相对历史 source-overlay 候选 `20.172 ms`，当前数值高 `0.344 ms`；
  不能写成“新历史最优”或“在旧最优上继续获得 `0.921 ms`”。

## 2. 旧优化没有丢失

历史 `20.172 ms` 的 B 臂就是 `pypto@14de90fd + pypto-lib@a745ab659`。
历史 candidate diff 与当前 `e6c7d8ec..a745ab659` 的 SHA256 均为：

```text
4cc1e10b7949d57893fbf4dc19e6170ffce771dacc92eb1abf31e2e95c2e7c5e
```

9 个变更文件逐 SHA 一致，`decode_fwd.py` 均为
`cdb2bb26ddc0ca773bcddd0629bfc7bdfa5c426a334e26dde4364aacd867f348`。
因此 a745 routed-GMM active-worker dual-latch 没有被遗漏或改坏。

真正被改坏的是跨仓 K8 reset ABI：

- K8 runtime 只识别 legacy 7-control/9-data layout；
- local-owner 改成 4-control/4-data 后无法命中 profile，回退为每步完整清零
  `11,842,560 B`；
- 历史 `20.172 ms` B 臂自身也记录了 `k8_prefix_applied=false`，
  `memset_all` p50 `1027.168 us`；
- `pypto@655c7bda` 新增 local-owner exact profile，将 control prefix 固定为
  `46,080 B`，当前 B 臂 `memset_all` p50 为 `462.277 us`。

所以当前代码形态是“a745 旧优化 + 655 K8 回退修复”，不是用后者替换前者。

## 3. 为什么不能按旧绝对值直接相减

历史 a745 source-overlay B 臂与当前 immutable A 臂名义上都是
`14de90fd+a745ab659`，但跨 campaign 的 p50 分别为 `20.172 ms` 与 baseline
midpoint `21.437 ms`，绝对偏移约 `1.265 ms`。已确认的合同差异包括：

- r12 source overlay 与 r14 immutable baked source；
- 历史/当前 image manifest、outer runner 和 distributed-runner 身份不同；
- privileged 容器与 non-privileged + explicit device-cgroup；
- 单臂/ABA 位置及额外 runtime env 不同；
- 与 r12 `20.973 ms` 单臂比较时另有 1000 与 100 measured iterations 的差异；
- 历史运行未保留原始 sample 数组，不能补做 bootstrap 显著性检验；
- 旧 `20.973 ms` 核心 admission/run-contract/patch SHA 仍匹配，但 evidence 包中的
  `driver.log` 已不匹配封存 SHA，不能把整包当完全 immutable 证据。

因此不能用 `20.172 - 0.921` 推导“应该达到 `19.251 ms`”，也不能把
`20.516 - 20.172` 单独归因为代码回退。跨历史绝对值必须用新的 matched 合同重测。

## 4. 发布声明与下一门

当前允许写：

- “matched immutable A/B/A 证明 local-owner reset 回退修复，收益
  `0.921 ms / 4.296%`”；
- “最新 digest 在 H4=`all`、64K、warmup10/iters100 合同下 p50 为
  `20.516 ms`”。

当前禁止写：

- “r15 刷新历史最优”；
- “相对上一发布版本显著提升 `0.457 ms`”；
- “a745 的 `20.172 ms` 已由当前 immutable image 原合同复现”。

发布性能声明前需在 0162 补齐：

1. r15 同口径 64K/1000 long gate，保存原始 samples；
2. old/new immutable digest 使用同一 runner、安全模型、effective env 与 arm 顺序的
   matched comparison；不得用 source overlay 冒充 image gate；
3. OCI spec 留证 `PYPTO_H4_RESIDENT=all`，并把结果与 registry digest 绑定。

registry push 与 a745 route-publication schema 是独立 release blocker，不能用性能门替代。

## 5. 证据

```text
H4 exact launcher:
/mnt/persist/chensiyu/workspace/perf-2026q3/h4-deploy-contract-20260829/
  exact-launcher-20260829-104651-1389455/

routed-GMM source-overlay A/B/A:
/mnt/persist/chensiyu/workspace/perf-2026q3/r12-gmm-softsync-validation-20260829/
  runs/h4-all-aba-20260829-230334-1791325-441605187/

K8 reset immutable A/B/A:
/mnt/persist/chensiyu/workspace/perf-2026q3/k8-a745-matched-validation-20260901/
  runs/h4-k8-a745-immutable-20260901-211101-2847484-669318413/
```
