# 专项：升级后 IPC interior pointer 被 address-free dispatch 拒绝

| 字段 | 值 |
|------|----|
| **子系统** | whole-net / runtime |
| **error signature** | `raw-pointer DeviceTensor cannot be dispatched` |
| **首次出现** | 2026-08-22 |
| **状态** | ✅ 已解 |
| **相关证据** | [`../benchmark/2026-08-24-upgrade-r9-release.md`](../benchmark/2026-08-24-upgrade-r9-release.md) |

## 1. 背景（Background）

五仓升级到新 pypto/simpler wire ABI 后，需要在 0162 上完成 Main、MTP、前五层
hidden/swimlane/DFX 的 immutable-image admission。权重和 KV 由 IPC pool 导入，holder
经常使用 pool 内的 interior slice，而不是独立 malloc 的 base pointer。

## 2. 现象（Symptom）

早期升级候选修完 `whole_decode_holder` 后，Main liveness 与 N=128 precision 已通过，
但前五层 holder 在首次 dispatch 失败：

```text
TypeError: Parameter 'input_rms__ssa_v0' shard 0: a raw-pointer DeviceTensor
cannot be dispatched by DistributedWorker; use this same
DistributedWorker.alloc_tensor() to create it.
```

`five_layer_moe_holder.py`、`five_layer_moe_route_holder.py` 仍直接切
`device_tensor(key)[start:stop]`；`mtp_layer_holder.py` 用裸 `data_ptr` 重建 reshape。
当时整网 liveness 又带了 `--skip-mtp`，所以 Main 绿不能覆盖这三个调用点。

## 3. 根因（Root Cause）

升级后的 address-free wire ABI 不再在 dispatch descriptor 中携带裸地址，而是从
`arg.buffer.tensor(...)` 导出 provenance。IPC pool 的 interior pointer 若没有一个
以该 slice VA 为 base 的 Buffer，就会同时被 runtime provenance guard 和 tensor-arg
构造拒绝。

迁移时发生了两次“只修一半”：

1. 上游只吸收了 IPC range 的部分能力，dispatch 仍要求精确 Buffer provenance；
2. 我方先只修了 Whole holder，没有枚举 five-layer route/non-route 与 MTP 的同族调用点。

因此这不是单一入口 bug，而是一类“IPC interior slice 必须保留/重建 Buffer identity”的
全调用点合同。

## 4. 如何解决（Fix）

最终发布 pins：

```text
simpler     85a82c454074c069315ed6485033c3c2b136e562
pypto       519b588a7a6461cac0e443e853accf29479c1d15
pypto-lib   bf3ff4400082f74b35fbdb5b3e0f5f4bf51ce373
```

修复组合为：

- simpler 为导入区域提供 slice-aware Buffer；
- pypto 的 imported tensor / reshape 保留 Buffer provenance；
- pypto-lib 的 holder 统一走 `device_tensor_slice()` 或 buffer-preserving reshape，
  不再用裸 `data_ptr` 造派生张量。

r9 immutable digest 上 Main 8-step、MTP single、MTP batch16 全部 PASS；combined
five-layer gate 的 L3/L4 `torch.equal=true`，8/8 rank 产出
`chip_swimlane_records.json`，outer admission `pass=true`。

## 5. 走过的弯路（Detours / What We Got Wrong）

- ❌ 只看到 Whole liveness 绿，就认为 IPC provenance 已全局解决。证伪：five-layer
  holder 首次 dispatch 仍报同一 raw-pointer 错误。
- ❌ 用 `--skip-mtp` 的整网门推断 MTP 也安全。证伪：`mtp_layer_holder.py` 正在遗漏清单内。
- ❌ 只修导入入口，不审 slice / reshape 派生路径。address-free ABI 的合同落在**最终 arg**
  是否带 Buffer，而不只落在最初 IPC import 是否成功。

## 6. 如何避免（Prevention）

- 升级 Buffer/provenance ABI 时，用代码搜索枚举所有 `DeviceTensor(data_ptr, ...)`、
  `device_tensor(...)[slice]` 与 reshape 调用点，按 Whole / five-layer / route / MTP 建矩阵。
- Main liveness、MTP liveness、focused hidden/swimlane 是正交门；禁止用一个替代另一个。
- admission 合同必须记录实际覆盖的 holder 与 `--skip-*`，避免“绿但没跑到”的假闭环。
- 对 IPC interior slice，只允许保留 Buffer 的正式 API；禁止重新包装裸地址。
