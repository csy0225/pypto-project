# Milestones —— 2026 Q2

按 session 划分的 milestone 日志，append-only，按日期降序。
高层 Phase 01-19 总结见
[`prototype-phase-01-19-summary.md`](prototype-phase-01-19-summary.md)。

## 2026-06-22（晚） —— 项目跟踪仓库建立 ✅

在 `<dev-host>/data/chensiyu/hw_project/pypto/pypto-project/` 建了
`pypto-project` 作为专属跟踪仓，push 到 `csy0225/pypto-project`（私有
fork-style）。散落 doc 迁移：

- 把 Phase 20/21/22 docs + archive 内容从 `pypto-lib/docs/step3p5/`
  （位置错了 —— 这些是跨仓库议题）迁到 `pypto-project/phases/` +
  `archive/`。
- 写了新顶层入口文档：README.md、STATUS.md、CLAUDE.md（slim）、
  blockers.md。
- 外部 tracker `<workspace>/pypto/CLAUDE.md`（594 行 monolith）退休 ——
  被本仓取代。

**解决**：项目 owner 提的 doc 散乱问题。项目状态 SSOT 现在落在本仓。

## 2026-06-22（下午） —— WIP push 拆分 + dev-workflow docs + Phase 20-22 设计 ✅

### WIP push 拆分

3 个 commit 上 fork csy0225：

- `csy0225/pypto-lib stepfun/develop`: `ffaf5d6 → 73dbd12`
  （tests/step3p5/ 12 个 ST/UT 脚手架 + 中文架构指南，+3381 行）
- `csy0225/pypto-lib wip/step3p5-barrier-allreduce-20260622`: NEW
  `b5bb6ee`（4 文件 -267/+181：barrier-style all_reduce + per_rank
  输入广播）
- `csy0225/pypto stepfun/develop`: `03136bf6 → b00c8b23`
  （10 个 full_rope SSA/scheduling debug repros，+2199 行）

**关键决策**：WIP barrier all_reduce **不进** `stepfun/develop`（会让
dense ST device 0 编译退化 by UB overflow）。侧分支保留意图待后续。

### Dev workflow + pitfalls docs（push: `73dbd12 → a6b5faa`）

- 新增 `pypto-lib/docs/known-pypto-pitfalls.md` §7：
  `pl.range(constant)` 展开不复用 SSA buffer → UB overflow（barrier
  all_reduce blocker 根因 + 3 个 avoidance recipe）。
- 新建 `pypto-lib/docs/dev-workflow-gotchas.md`：5 条 catalog 非 pypto
  workflow 时间坑（stale pyc / 三件套 activation / HTTP/2 timeout /
  netboot SSH / gh CLI 缺席）。

### Phase 20-22 设计落地（push: `a6b5faa → 69f22b1`）

3 个 phase doc，每个 ~200-300 行。这些 doc 后来移到本 `pypto-project`
仓（见上面晚段）。

## 2026-06-22（早） —— 0162 重启后恢复 + 重验 + MoE 507018 复现 ⏸

### 重启后环境恢复

`gpu-a910x-0162` 重启过；三剑合璧都活着（driver 25.5.2、firmware
7.8.0.7.220 chip flash、CANN 9.0.0-beta.1 NVMe symlink）。4 个 git 仓
都在期望 HEAD 上，simpler submodule `a6e06406`。

### Smoke probe 红鲱鱼（已解）

第一次 `python -m models.step3p5._smoke_program_build` 返回 rc=1，
attention_swa.py:396 报 `valid_cols (48) exceeds bound 16`。**根因**：
上次 session `apply_perrank_patch(TP=2)` 实验留下的 stale
`__pycache__/config.cpython-311.pyc`。Python 的 pyc 失效检查只比 source
mtime，不比 module dict 值。

**解决**：`find models/step3p5 -name "*.py" -exec touch {} +` 把
source mtime 顶过 pyc → pyc 失效 → fresh import 读到正确 `TP=8`。归到
workflow gotcha §1。

### 验证基线

| 测试 | 状态 |
|------|------|
| simpler L3 allreduce_distributed -d 0-1 | ✅ `max\|out-expected\|=0` |
| Phase 19 ST-1 full dense | ✅ PASS 7.93s |
| Phase 19 ST-2 swa dense | ✅ PASS 14.85s |
| MoE 6 variants smoke | ✅ 6/6 PASS |
| MoE device runtime（full_silu_silu -d 0） | ⏸ 5s 内 507018 fault |

记到 blocker §2；需要 `P19_DISPATCH_LIMIT` dispatch-cut tool 定位。

## 2026-06-20 —— 5 仓库 rebase 到 origin/main + push fork ✅

把 pypto / pypto-lib / pto-isa / PTOAS / simpler 全 rebase 到
`origin/main`。Audit：

- 4 个 simpler 本地 patch（zero-size view + `--no-as-needed` libhcomm
  + IPC ENABLE_PEER_ACCESS + SDMA_OFF + llvm-strip）都还要保 ——
  上游本周期没 subsume 任何一个。
- 6 个 pypto-lib step3p5 commit 都要保。
- 3 个 pypto commit（DFX env hook + repros + submodule pin）要保。

**结果**（push 到 `csy0225/`）:

- pypto: `926941e0 → 03136bf6`
- pypto-lib: `93826904 → ffaf5d69`
- pto-isa: `109c9f72 → e25732f0`
- simpler: `c66b4120 → a6e06406`

0162 上验证：smoke probe rc=0，simpler L3 allreduce 双卡 golden，
ST-1 dense device PASS，MoE 6/6 smoke PASS。

**Rebuild trap**：`pip install -e .` 第一次失败 due to
`tensor.h:535 buffer_elems` `-Werror=unused-variable`（NDEBUG +
release flag）。修法：别传 `CMAKE_BUILD_TYPE`（用 dev default）。

## 2026-06-19 —— Phase 16 多卡 IPC blocker RESOLVED ✅

`support_shmem_map_exbus=0` cap（filed as simpler#1037）是 driver 能力
缺口。解决要三剑合璧：

1. Driver `25.0.rc1.2 → 25.5.2`
2. Firmware `7.7.0.3.220 → 7.8.0.7.220`（chip flash，持久）
3. CANN `9.0.0-beta.1`（NOT GA —— GA 的 TDT 不推 AICPU
   `libaicpu_extend_kernels.so`，让 simpler init 507018 失败）

加 simpler `comm_hccl.cpp` patch（CANN GA forward-compat alias）。

**Traps**:

- CANN GA vs beta.1：3+ 小时浪费在 GA 上才发现。
- 0162 是 netboot/tmpfs：`/usr/local/Ascend/`、`/etc/`、`~/.ssh/` 重启
  全丢。建 `RECOVERY.sh` 幂等恢复；持久 state 在 NVMe `/mnt/persist/`。
- Kubernetes DaemonSet（`device-plugin`、`npu-exporter`）占着 driver
  `.run --upgrade`。`kubectl drain` 不够 —— 必须 `systemctl stop kubelet`
  + 手动 kill。

**验证**：`aclrtIpcMemImportByKey + ENABLE_PEER_ACCESS` 跨卡 rc=0、
`peer_va == parent ptr`；simpler L3 `allreduce_distributed` 双卡
`max|out-expected|=0` golden。

**0234 路径**：只需升 driver+firmware（CANN 已经对）。`.run` 包 stage
在 0162 `/mnt/persist/ascend-staging/`。归到 blocker §5。

## 2026-06-17 —— Phase 19 MoE blocker 1-4 清掉 + dense ST device PASS ✅

详见 [`prototype-phase-01-19-summary.md`](prototype-phase-01-19-summary.md)
"Phase 19 MoE blocker 解决"。MoE device runtime 507018 仍在（blocker
§2）。Dense ST device 0 通过（full 7.93s，swa 14.85s）。

## 2026-06-15 —— Phase 15 单卡 e2e rc=0 ✅

单 rank decode_layer 端到端跑通 device 0，20 个 dispatched task 完成。
三个层叠修复一起：head_gate ×1 旁路 + `--tp-world-size 1` monkey-patch
+ `LAYER_*_ROWS_DYN` override。`next_hidden_out shape=[1, 16, 4096],
max|value|=0`（dummy zero weight 期望零输出）。Run time 6.69s。

---

## Pin snapshot 历史（降序）

| 日期 | 事件 | pypto | pypto-lib | pto-isa | PTOAS（src） | simpler | ptoas-bin |
|------|------|-------|-----------|---------|--------------|---------|-----------|
| 2026-06-22 晚 | pypto-project 仓建立 | `develop:b00c8b23` | `develop:9c4773f` | `develop:e25732f0` | `develop:da011a3d` | `a6e06406` | `v0.45` |
| 2026-06-22 下午 | Phase 20-22 设计 + dev-workflow docs | `develop:b00c8b23` | `develop:69f22b1` | `develop:e25732f0` | `develop:da011a3d` | `a6e06406` | `v0.45` |
| 2026-06-20 | 5 仓 rebase + fork push | `develop:03136bf6` | `develop:ffaf5d6` | `develop:e25732f0` | `develop:da011a3d` | `a6e06406` | `v0.45` |
| 2026-06-19 | Phase 16 三剑合璧验证 | `main:a1b066df` | `main:9c5593fb` | `main:109c9f72` | `main:29a8af28` | `afb5c5a9` | `v0.44` |
| 2026-06-17 | Phase 19 blocker 1-4 清掉 | `main:3f421313` | `main:08f71692` | `main:8e436661` | `main:a1efed75` | `6e84154d` | `v0.43` |
| 2026-06-15 | Phase 15 单卡 e2e rc=0 | `main:3f421313` | `main:af4b2ed5` | `main:12e766d1` | `main:5392d5da` | `6e84154d` | `v0.43` |
| 2026-06-05 | Phase 13 re-sync + smoke 绿 | `main:3f421313` | `main:08f71692` | `main:8e436661` | `main:a1efed75` | `6e84154d` | `v0.43` |

---

## 已解 blocker（post-mortems）

### 2026-06-22 —— simpler#1018 libhcomm DT_NEEDED ✅

`comm_init` 段错 —— `hccl_comm.h` 把 HCCL 声明为 weak，x86 默认
`--as-needed` 把 `libhcomm.so` 从 `DT_NEEDED` 删了。修复在 simpler
`a6e06406`：`src/{a2a3,a5}/platform/onboard/host/CMakeLists.txt` 把
`${HCCL_LINK_TARGETS}` 包成 `-Wl,--no-as-needed ... -Wl,--as-needed`。

### 2026-06-19 —— simpler#1037 IPC support_shmem_map_exbus=0 ✅

三剑合璧修复（driver 25.5.2 + firmware 7.8.0.7.220 + CANN beta.1）。
详见上面 2026-06-19 milestone。

### 2026-06-17 —— Phase 19 blocker 1-4 ✅

1. PTOAS v0.44 `pto.tci ui32 {descending=false}` parser：上游 v0.45 fix
   `505abd64`。
2. sh_mlp / gate_matmul L1/UB overflow：是 shape-choice artifact
   （`apply_tp1_patch` 错，`apply_perrank_patch` 对）。
3. dispatch.py 32B 对齐：`PER_RANK_BUCKETS = pad8(...)` 跨 5 文件
   mirror。
4. CCEC bf16 类型转换：`expert_weights` BF16 → FP32 跨 6 个 emission 点。

详见 [`prototype-phase-01-19-summary.md`](prototype-phase-01-19-summary.md)
"Phase 19 MoE blocker 解决"。
