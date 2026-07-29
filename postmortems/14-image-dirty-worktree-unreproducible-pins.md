# 专项：已发布/已验证镜像内 git 工作树 dirty，记录的 pin 集无法复现验证环境

| 字段 | 值 |
|------|----|
| **子系统** | deployment / 镜像发布流程 |
| **error signature** | `Worker.copy_to: device pointer 0x… is not a live allocation on worker 0 (wrong worker, freed/stale, or an interior pointer)`；上游表现为整网 CI `main_hidden_8step rc=1` |
| **首次出现** | 2026-07-29（缺陷自 2026-07-28 起已存在） |
| **状态** | ✅ 已解（补丁入库为 `csy0225/simpler@8459d60f`，审计已补） |
| **相关** | [`deployment/docker/README.md`](../deployment/docker/README.md)、[`deployment/version-matrix.md`](../deployment/version-matrix.md)、[`13-tp-allreduce-pull-notify-race.md`](13-tp-allreduce-pull-notify-race.md) |

## 1. 背景（Background）

PERF-C4（TP all-reduce 改 reduce-scatter + push all-gather）在 0162 上验证全绿后，
按 `deployment/docker/build.sh` + 新 spec 构建镜像
`stepfun-develop-20260729-allreduce-push`（pypto-lib 前进到 `cfbdcce8`，其余四仓与
ptoas-bin 保持已验证 pin），推 registry，然后在**镜像内**做完整回归。

此前 C4 的所有真机回归都跑在 0728 的 candidate 镜像
`step3p5-b404a3c9-ci-final-20260728` 上（并挂载被测的两个模型文件）。

## 2. 现象（Symptom）

新构建镜像的 audit / smoke / 五仓 pin / `ALLREDUCE_PUSH_PRESENT` **全部 PASS**，
但整网 CI 与三次重复 8-step 全部失败：

```
whole_decode_holder.py:607 run()
 → distributed_runner.py:1213 _reset_persistent_domains → orch.copy_to
 → worker.py  _child_prov_require_live
ValueError: Worker.copy_to: device pointer 0x12cd40100000 is not a live allocation
            on worker 0 (wrong worker, freed/stale, or an interior pointer)
SINGLE_CHIP_HIDDEN_CI=FAIL   stage=main_hidden_8step rc=1
```

指针值每次完全相同（确定性，非 race）。失败点是 runtime **每次 request 前清零
retained comm window** 的路径。

## 3. 根因（Root Cause）

**0728 的三个 candidate 镜像都带着一份未提交、未记录的 simpler 补丁；`build.sh`
严格按 pin clone，必然拿不到它。**

单变量实验先排除了自己的改动：把**改动前**的 mesh 版 `decode_fwd.py` + `dense_mlp.py`
挂到新镜像上，以**完全相同的指针**同样失败 → 失败属于镜像构建，与 all-reduce 无关。

随后逐镜像核对 `git status`：

| 镜像 | simpler HEAD | 工作树 | span-aware | pypto-lib |
|---|---|---|---|---|
| `step3p5-b404a3c9-20260728` | `216e7632` | **dirty 2 文件** | 有 | `b404a3c9` |
| `step3p5-b404a3c9-ci-cleanup-20260728` | `216e7632` | **dirty 2 文件** | 有 | `b404a3c9` |
| `step3p5-b404a3c9-ci-final-20260728` | `216e7632` | **dirty 2 文件** | 有 | `b404a3c9` |
| `stepfun-develop-20260726-step3p5-only`（已发布） | `216e7632` | clean | **无** | `53eb7212` |

补丁内容：把 child provenance 检查从**单点**升级为**范围（span）感知**——新增
`_child_span_in_ipc_region` / `_child_span_in_domain_range` / `_child_ptr_in_domain_range`
/ `_child_tracked_domain_ids`，给 `_child_prov_require_live` 与
`_child_prov_record_domain` 加 span/extent 参数，并由 `Orchestrator.copy_to` 传入拷贝长度。

为什么非它不可：`_reset_persistent_domains` 按 zero-buffer 的 chunk（1 MiB）分段
`copy_to` 整个窗口，`base + 1MiB` 这种**interior pointer 本身不是一条已登记的
allocation**。step3p5 的 per-layer MoE all-reduce scratch 是 `42×16×4096×2 = 5.5 MiB`，
必然要走第二个 chunk → 单点检查必然拒。窗口尺寸在 C4 前后**未变**，所以 mesh 版一样中。

上游 `origin/main` 是更旧的版本（连 `_child_ptr_in_ipc_region` 都没有），该能力**不在上游**。

## 4. 如何解决（Fix）

1. **抢救**：补丁只存在于镜像内（未提交/未推），先从 `ci-final` 导出到
   `workspace/rescue-simpler-cifinal/`（`simpler-cifinal-uncommitted.patch`，
   sha256 `97a274ae4424…`，base `216e7632`）。
2. **入库**：用独立 git worktree 从 `216e7632` 起，贴入镜像内的 `worker.py` +
   `orchestrator.py`，**逐字节校验 diff 与镜像一致**后提交 →
   `csy0225/simpler` `stepfun/develop` `216e7632 → 8459d60f`（fast-forward）。
3. **重建**：把 spec 的 `SIMPLER_COMMIT` 指到 `8459d60f…`（`Dockerfile:58` 用全 40 位
   SHA 校验），重建并重跑完整回归。
   ⚠ **只改 spec 不够**：Dockerfile 的显式 `checkout ${SIMPLER_COMMIT}` 在更早的 layer，
   后面 `pip install -e pypto` / `build_runtimes` 的 `git submodule update` 会把 runtime
   切回 gitlink（镜像内 reflog：`216e7632 → 8459d60f → 216e7632`）。**必须同时在
   `csy0225/pypto` 提 gitlink bump**（`git update-index --cacheinfo 160000,<sha>,runtime`
   → `6933b1aa`）并前进 `PYPTO_COMMIT`。这一步是靠"推前先本地核对镜像内容"发现的。
4. **补审计**：`img_regress.sh` 增加 `IMAGE_WORKTREE_CLEAN_AUDIT`——对
   pypto / pypto-lib / pto-isa / PTOAS / simpler 逐个 `git status --porcelain`，
   任一 dirty 即 FAIL 并打印文件列表。

**版本选择**：入库的是 0728 镜像里那一版（worker.py `+133/−19`），因为它与 C4
40 个 decode step 全绿的 runtime **逐字节一致**。本地工作树里另有一版更新的 WIP
（`+222/−47`，额外给 `exact_malloc_live` 加 `malloc_size` 上界，并带 119 行单测），
**设备上从未验证**，故意未纳入；它应作为独立一步先验证再前进 pin。

## 5. 走过的弯路 / 教训（Detours）

- **审计写了一半**：`img_regress.sh` 一开始就查了 credential 泄漏和 canonical-only
  符号，**唯独没查工作树是否 dirty**。"pin 对 + 冒烟过"被当成了"环境可复现"，
  这是缺口从 0728 漏到 0729 的直接原因。**pin 相同不等于内容相同。**
- **差点误判为自己的回归**：失败点在 IPC/provenance，很容易归因到刚改的 collective。
  避免误判靠的是**单变量对照**（把改动前的文件挂到新镜像上）——先证明"与我无关"，
  再去找真因。
- **"已验证镜像"的信任边界**：0728 三个 candidate 的 C/D/G 结论（BS1/2/16、N=256、
  Main 8-step）同样建立在这份未提交补丁上，因此在补丁入库前它们也无法按记录的 pin
  复现。这不是 C4 引入的缺陷，但 C4 是第一次全新构建、于是第一次撞上。
- **补丁差点永久丢失**：它只存在于镜像层里，镜像一删即失。凡是"镜像能跑但 pin 跑不通"
  的情况，**第一步应当是把 diff 导出存档**，而不是先去改构建。
