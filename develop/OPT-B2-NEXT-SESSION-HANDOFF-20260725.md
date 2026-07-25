# step3p5_opt B2 loop-form — 交接文档（2026-07-25，后续改造完成）

> 承接 `2026-07-25-loop-form-status-and-507018-diagnosis.md`（那份覆盖 507018 环境问题 + B2 loop-form 结构）。
> **本份覆盖：环境已解、TP bug 已修、L03 accuracy 分叉已修复，以及 0162
> 镜像设备上的最终回归结果。**
> 团队：team-lead（协调）+ b1-weights（权重/holder/dump）+ c1-comm（通信/根因静态分析）。

---

## 0. 一句话状态

opt loop-form 的**架构 / 编译 / 8 卡设备执行 / TP 一致性 / full-net 首 token
精度均已打通**。正式改造包括 MoE direct-Out residual SSA 修复，以及
`OPT_STOP_AFTER=0` full graph 执行 L1/L2 dense loop 的条件修复。0162 上
full 首 token 为 303，L0-L44 逐层 bit-exact；N=22/N=40 截断回归也分别在
L0-L24/L0-L42 bit-exact。

> **结论优先（2026-07-26 设备回归）**：此前“tail/组件污染 runtime loop”的
> 判断是误导；真正的 full blocker 是 `OPT_STOP_AFTER=0` 错误跳过 L1/L2。
> 修复为 `if OPT_STOP_AFTER == 0 or OPT_STOP_AFTER >= 2:` 后，full graph
> 首 token 恢复为 303。8-step teacher-forced standalone harness 中，
> opt 与 baseline 的主 hidden 及每一步 L0-L44 均逐字相同；两者共同在 step 2
> 对内置 oracle 出现 1 个位置不匹配（7/8），所以该位置不是 opt 回归。

---

## 1. 已解决（本 session 实打实的成果）

### 1.1 环境（0724 镜像绕开裸机漂移）
- 之前卡很久的 `507018` = 0162 裸机 provisioning 漂移（07-20 重启没跑 RECOVERY.sh），**不是 opt/代码**。
- 解法：**用 0724 blessed 镜像** `hub.i.basemind.com/stepcast/vllm-pypto:stepfun-develop-20260724`（pypto ca21ab5f / pypto-lib fd26b1be / pto-isa ecb6c303 / PTOAS fc8c6cae / simpler 216e7632 / ptoas v0.50 + CANN beta.1）。镜像里 opt 设备跑通，无 507018。
- 0162 重启恢复：nvme 手动挂（不在 fstab）+ RECOVERY.sh + runc `--no-pivot` wrapper + containerd 手动起（非 systemd）。详见 `deployment/machine-recovery.md`。

### 1.2 TP 一致性 bug（signal-stride）—— 已修 + 设备验证
- **现象**：opt 首 token 输出 19、`hidden_tp_spread=384.5`（8 卡 hidden 不一致）。
- **根因（c1 定位）**：MoE loop 里 **4 个 cross-rank 信号栈**（`moe_count_done_sig` / `moe_data_done_sig` / `moe_combine_done_sig` / `moe_sh_signal`）的 per-layer stride 写错：用了 `moe_win_off = layer_idx*BATCH(16)`，应为 `moe_sig_off = layer_idx*COMM_SIGNAL_STRIDE_I32(128)`，且 buf/window 多了 `*n_ranks`。→ **从物理层 21 起 signal slice 越界** → 跨 rank Ge barrier 读越界垃圾 → rank 发散。
- **修复（b1，commit `5d7eda44`）**：4 栈的 16 处 slice offset `moe_win_off→moe_sig_off`（含 L43 `_43`/L44 `_44` 变体）+ 4 处 buf 去 `*n_ranks` → `NUM_MOE_LAYERS_TOTAL*COMM_CONTROL_SIGNAL_BYTES` + 4 处 window `[N*n_ranks,1]→[N*COMM_SIGNAL_STRIDE_I32,1]`。对齐 baseline sibling。
- **设备验证**：full-net `tp_spread` **384.5 → 0.0**（8 卡完全一致）。bracket 精确验证越界 onset：`OPT_MOE_LAYERS=20` clean / `=22` diverge（层 21 越界）。
- ⚠ **回归教训**：这个 fix 一度因 b1 的 `git checkout`（修 dump 时）被 revert，重建时 buggy 版带回来（tp_spread 又变 288）；已重新 apply 并 **commit**（不再留 working-tree-only）。

---

## 2. 已解决的 blocker：L03 accuracy 分叉（修复前历史现象与定位过程）

### 2.1 修复前历史现象
- 以下现象均来自正式改造前的旧版本，仅用于说明 blocker 的起点；
  **不是当前代码和当前回归结果**。
- opt full-net：`tp_spread=0.0`（TP 一致）**但 argmax=565 ≠ 303**，`hidden_row0_abs_max=692`，finite。
- baseline（同镜像同输入 seed-token 6127）：argmax==303 token_exact ✅ —— **oracle 确认，565 是真 opt bug**。

### 2.2 定位：per-layer hidden dump（opt vs 自校验-303 baseline）
逐层 `ratio_allclose` / cosine 对比（`compare_per_layer.py`）：

```
L00: PASS     cos=1.000  opt|max=16.125  base|max=16.125   ← L0 (dense-full) 完全一致
L01: DIVERGE  cos=0.000  opt|max=0       base|max=23.625   ← opt 全 0 (dump bug, 见 2.4)
L02: DIVERGE  cos=0.000  opt|max=0       base|max=26.875   ← opt 全 0 (dump bug)
L03: DIVERGE  cos=0.977  opt|max=15.688  base|max=33       ← 首个真实 compute 分叉, 量级 0.48×
L04-L09: cos 0.94-0.97, 量级 0.27-0.59×
L10-L39: cos 0.998,     量级 1.05-1.13×  (opt 反而略大)
L40-L44: cos 0.996→0.846, 量级 1.07-1.14×
```

**判读**：
- L0 完全对 → 输入 + dense-full 计算 + dump 都正确。
- L1/L2 opt=0 是 **dump artifact**（见 2.4），不是 compute（L03 收到了 L2 的真输出）。
- **L03（第一个 MoE 层，swa_moe，layer_idx=0）= 首个真实分叉**。signature = **方向基本对（cos 0.977）+ 量级系统偏（0.48×）**，典型 scaling / 丢分量。
- opt dump 量级曲线 tracks baseline（都中段涨到 ~3500 尾段掉）→ **MoE dump 可信**，L03+ 是真分叉不是 stale。

### 2.3 已排除的根因（全部有证据）
| 候选 | 结论 | 证据 |
|---|---|---|
| 环境 / 507018 | 排除 | 0724 镜像跑通 |
| TP/signal-stride | **已修** | tp_spread 384.5→0 |
| layer dispatch (%4 full/swa) | 排除 | 逐层对齐 baseline 层表 |
| MoE variant / L43/L44 特化 | 排除 | c1 静态核 |
| norm/gate/residual 顺序 | 排除 | MoE body 与 baseline **逐字一致 (0 diff)** |
| **attention 权重 offset** | 排除（**误判 3 次**） | 见 2.5 铁律 |
| W8A8 expert **scale** 逐层映射 | 排除 | b1+c1 核 offset/stacking/shape 全一致 |
| recv_counts layout | 排除 | per-rank 预筛表，my_rank 在 write 侧 |
| H3 data 窗口跨迭代 reset | 排除 | 全 write-before-read / 显式 zero，与 baseline 等价 |
| host wiring（权重 + 非权重 54 arg） | 排除 | c1 逐 arg 核 |
| SSA loop-carry (prev_hidden handoff) | 排除 | ConvertToSSA 自动闭合 Pattern B（bare reassignment 合法） |
| IfStmt branch phi-merge (差1) | 排除 | ConvertIf 自动 phi-merge，无需 pl.yield_ |
| B' routed-path SSA 遮蔽 | 存疑（见 2.4） | 静态分析 routed 只 method-arg，dense compute 正确证 masking 只打 pl.assemble-Out |

### 2.4 dump 机制本身有 SSA-masking（关键 caveat，导致定位循环）
- **现象**：dense loop 的 per-layer dump（L1/L2）恒为 0；MoE loop 的 dump（L3+）正常非 0。
- **机制**：`pl.assemble(Out, pl.slice(SRC, ...))` 里 SRC = `create_tensor + reassign` 的 tensor（如 dense 的 `h_next`），在 `pl.range` unroll 下被读成 **zero-init version**。
- **b1 的 offset-fix 失败**（commit `b57afebb`：把 dense dump offset 从 inline `[layer_idx+1,0,0]` 改 pre-computed）→ **L1/L2 仍全 0** → 证明 **masking 是 source-based（create_tensor+reassign 的 SSA version），不是 offset-based**。
- **重要影响**：这**部分 revive 了 site-1 假设**（见 2.6）——c1 之前证伪 site-1 靠的是"trigger=inline 算术 offset"前提，已被 b1 的失败 fix 推翻。
- **对定位的影响**：dump 工具自己受同类 bug 影响 → 用 dump 定位是**循环的**。**所以探针必须用 proven-reliable 的 MoE-dump pattern**（per_layer_hidden 路子，L03+ 抓到非 0，证明可信），不能用会被 masked 的写法。

### 2.5 ⚠ 铁律：attention 权重 offset 误判（本 session 犯 3 次）
- **反复误判**：看到 opt MoE loop `full_idx=(layer_idx-1)//4` / `swa_idx=layer_idx-full_before`，对比 baseline **global** `full_wq[12]`/`swa_wq[33]` 的 offset（dense 在 slot 0/0-1），得出"opt 少算了 dense 前置层，要 +1/+2"。**这是错的。**
- **真相**：opt MoE loop 读的是 **holder split 出来的 MoE-only 栈** `moe_full_wq[10]` / `moe_swa_wq[30]`（`decode_fwd.py:4058/4065`）。holder 用 `Wsub(KEY_WQ_FULL, _MOE_FULL_SLOTS=range(1,11))` / `Wsub(KEY_WQ_SWA, range(2,32))`（`whole_decode_holder.py:44-45`）把 canonical slot **1-10 / 2-31 抽进栈的行 0-9 / 0-29**。**dense 偏移已经在 holder 层做掉了**。
- **端到端 trace（物理 L3, first swa_moe）**：baseline 读 global swa_wq 行 2（=L3）；opt 读 moe_swa_wq 行 0 = holder `Wsub[2:32]` 行 0 = global 行 2 = **同一份 L3 权重**。opt 对。
- **+1/+2 会**：读错层权重（L4→L8 的）+ **越界**（full_idx=10 / swa_idx=31 超 10/30 行栈）。
- **内化**：opt 用 holder split MoE-only 栈，offset **必然**比 baseline global 少一个 dense count，**这个差是对的，不是 bug**。

### 2.6 历史 leading 假设（已被后续证据取代）：site-1 `stash_resid_hold`
- opt `swa_moe_chip_orch` L2248: `resid_hold = pl.assemble(resid_hold, pl.slice(resid1, ...), [0,_r0])`，SRC `resid1` = create_tensor+reassign（attention 输出）。**命中 source-based masking pattern**（2.4）。
- 若 masked → `resid_hold=0` → section D `next_hidden = 0 + moe_out = moe_out`（**丢 attention residual**）。
- **量级 fit**：第一个 MoE 层 attention residual 与 moe_out 大体同向，丢 resid → 同向缩小 → **正好 0.48× + cos 0.977**。
- **后续结论**：这个假设不是本轮 full 565 的根因。保存并对比 full 与
  stop3 的 `ConvertToSSA`/最终 IR 后，发现 full 模式实际跳过了 L1/L2 dense
  loop；stop3 反而执行了 L1/L2。因此 L3 起的旧分叉来自错误的
  `OPT_STOP_AFTER=0` 条件，而不是 tail Out 污染 runtime MoE loop。

---

## 3. [已完成] 原下一步（决定性验证与改造）

### 3.1 根因与正式代码改造
- **direct-Out residual SSA 修复**：`full_moe_chip_orch`、
  `swa_moe_chip_orch` 以及 L43/L44 specialized orchestration 不再使用
  `create_tensor -> attention reassign -> assemble` 的 residual handoff；
  attention 直接写入 `resid_hold` Out，post norm 和 residual add 直接消费
  该 Out。该修复已在 L3 单层、N=2/3/4/10/20 以及 stop3 N=40 回归中验证。
- **full dense-loop 条件修复**：`OPT_STOP_AFTER=0` 表示 full graph，必须执行
  L1/L2；条件从 `OPT_STOP_AFTER >= 2` 改为
  `OPT_STOP_AFTER == 0 or OPT_STOP_AFTER >= 2`。
- 保持 signal stride 修复不变：
  `moe_sig_off = layer_idx * COMM_SIGNAL_STRIDE_I32`，signal window
  使用 512B stride；未改 attention weight offset。
- 删除临时 `dbg_out` 顶层 host ABI、holder buffer 和 harness dump；
  保留正式 `per_layer_hidden` accuracy dump。

### 3.2 IR 与设备证据
- IR 对比 artifact：`0162:/tmp/opt_ir_compare_20260726_002149/`。
- full 修复前最终 IR 没有 L1/L2 dense loop，直接从 L0 进入 L3 MoE loop；
  stop3 IR 明确包含 L1/L2 dense loop。这解释了此前所有“加入 L43 污染
  runtime loop”的失败实验。
- full 正式单步 artifact：
  `0162:/tmp/opt_full_formal_20260726_002843/`。
  compile OK，`output_token=303`，`token_exact=true`，
  `hidden_tp_spread=0.0`，finite，`hidden_row0_abs_max=660`。
- full 单步逐层比较：
  `0162:/tmp/perlayer_173036/baseline/` 对
  `0162:/tmp/opt_full_formal_20260726_002843/`，L00-L44 共 45/45
  `max_abs_diff=0`。

### 3.3 截断与多步回归
- stop3 N=22：
  `0162:/tmp/opt_stop3_n22_formal2_20260726_004117/`；
  L00-L24 共 25/25 exact，`hidden_tp_spread=0.0`，
  `hidden_row0_abs_max=3424`。截断 token 不与 full expected 303 比较。
- stop3 N=40：
  `0162:/tmp/opt_stop3_n40_formal2_20260726_004427/`；
  L00-L42 共 43/43 exact，`hidden_tp_spread=0.0`，
  `hidden_row0_abs_max=141`。截断 token 不与 full expected 303 比较。
- full 8-step opt：
  `0162:/tmp/opt_full_8step_20260726_0110/`；
  baseline：
  `0162:/tmp/baseline_8step_20260726_0118/`。
  两边每步主 hidden `exact=True, max_abs_diff=0`，每步 L00-L44
  45/45 exact，TP spread 均为 0，hidden 均 finite。
- 两边对同一内置 oracle 都是 7/8：step 0/1、3/4/5/6/7 exact，
  step 2（输入 token 1207）共同输出 6127，而 oracle 为 19384。
  该差异在 baseline 中完全复现，不应作为 opt blocker。

### 3.4 不应合入的临时方向
- `external-tail`、`autodeps`、`PYPTO_MEM_PLANNER=ptoas` 以及各种
  tail/carry 临时 variant 均未解决 full 565 或无法编译，不纳入正式 patch。

---

## 4. 环境 / 跑法（0724 镜像，可直接复制）

```bash
# 0162 恢复（重启后）：手动挂 nvme + RECOVERY.sh + containerd（见 deployment/machine-recovery.md）
# 镜像 + 路径
CD=/mnt/persist/k8s-install/containerd
IMG=hub.i.basemind.com/stepcast/vllm-pypto:stepfun-develop-20260724
CKPT=/data/chensiyu/step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp
NEW=/mnt/persist/chensiyu/perf-opt-ws   # 0162 device 用的 checkout（team-lead scp 同步 local perf2 → 这里）

# per-layer dump run（baseline 自校验 303 + opt-dump），individual bind-mount（proven）：
#   mount: $NEW/models/step3p5_opt (opt dir) + baseline decode_layer_single_chip_hidden.py
#          + whole_decode_holder.py + _stage_main_hidden_only.py + compare_per_layer.py
#   COMMON flags: --net host --ipc host --privileged --security-opt apparmor=unconfined
#                 --security-opt seccomp=unconfined $DEVS(davinci8-15) + davinci_manager/hisi_hdc/devmm_svm
#                 -v driver:ro -v $CKPT:ro --shm-size 32g
#   baseline: python -m tests.step3p5.harnesses._stage_main_hidden_only --device 8..15 --out $OUT/baseline --ckpt $CKPT --steps 1 --seed-token 6127
#   opt:      同上 + --layer-module models.step3p5_opt.decode_fwd --layer-name whole_decode_opt --out $OUT/opt
#   compare（另起容器，必须带 --net host + --security-opt apparmor/seccomp）：
#     python /compare.py $OUT/opt $OUT/baseline 00
# 参考脚本：team-lead local /tmp/perlayer_run2.sh（fresh 时间戳 OUT 目录，无需 rm）

# 诊断旋钮（decode_fwd.py 模块级）：
#   OPT_STOP_AFTER=0 全网 / =1 只L0 / =2 L0+dense-swa loop / =3 L0+dense+OPT_MOE_LAYERS 个 MoE 层
#   OPT_MOE_LAYERS=N MoE 循环层数（默认 40）
```

**已保存的有效 dump**（0162 `/tmp`，重启前）：`/tmp/perlayer_173036/{baseline,opt}/per_layer_step00/layer00-44.pt`（baseline 303 有效参照，可复用）。

---

## 5. 组件 pin / commit（fork csy0225，branch perf/step3p5-bc-v2）

| 组件 | 状态 |
|---|---|
| opt `models/step3p5_opt/decode_fwd.py` | signal-stride fix (`5d7eda44`) + per-layer dump (`5d7eda44`/`42e862a5`) + dense-dump offset fix (`b57afebb`，**未 work**) committed |
| dump instrumentation | opt 5 站点 + baseline 45 站点（`decode_layer_single_chip_hidden.py`，+460 行纯加，303 不受影响，self-check 过） |
| holder / harness | `per_layer_hidden` 接线 + `--layer-module/--layer-name` |
| `tools/step3p5/compare_per_layer.py` | 已 track，正式用于 ratio_allclose + cosine + first-divergence + JSON 比较 |
| `per_layer_hidden` instrumentation | 已正式保留；用于 baseline/opt 逐层 hidden 回归，full L0-L44、stop3 L0-L24/L0-L42 均已验证 |
| `dbg_components` L3 组件 dump | 不再需要；已由 full/stop3 IR 对比定位真正 blocker，未纳入正式 patch |
| 当前正式 patch | 已提交为 `0d9f66bc`；未合入临时 tail/autodeps/PTOAS 方向 |

**⚠ 教训**：signal fix + dump 一度是未 commit 的 working-tree 改动，被 `git checkout` 冲掉 3 次。**所有 fix 立即 commit**。

---

## 6. 团队协作复盘（给下个 session 的方法论提醒）

- **churn 根因**：调试一个 pl.range unroll 的 codegen SSA-masking bug，且 **dump 工具自身受同类 bug 影响**，导致定位循环 + 多个静态假设被反复证伪（attention-offset 3 次、scale、routed、resid_hold 反复）。
- **有效的**：falsify-before-assert（每个假设都被静态/设备证伪或证实）；两 agent 对抗核对；端到端物理 trace 破误判。
- **教训**：(1) 静态"offset 值不同"≠"错"，要看 holder split / 端到端物理映射；(2) dump 工具在同 bug 类里不可轻信，探针要用 proven-reliable pattern + sanity；(3) fix 立即 commit。

---

## 7. 最终状态

- 正式代码改造已完成并提交为 `0d9f66bc`：保留 MoE direct-Out residual SSA 修复，
  并修正 `OPT_STOP_AFTER=0` 必须执行 L1/L2 dense loop 的 full-graph 条件。
- 0162 正式回归：full 首 token 为 303，L0-L44 共 45/45 exact；stop3
  N=22 为 L0-L24 共 25/25 exact，stop3 N=40 为 L0-L42 共 43/43 exact。
- full 8-step standalone teacher-forced 回归中，opt 与 baseline 的主 hidden
  以及每一步 L0-L44 均 exact。两边对同一 oracle 都为 7/8，step 2 的
  `1207 -> 6127` 与 oracle `19384` 的差异在 baseline 中同样出现，不作为
  opt regression。
