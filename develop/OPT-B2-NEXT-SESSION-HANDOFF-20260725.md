# step3p5_opt B2 loop-form — 交接文档（2026-07-25，后续改造完成）

> **CURRENT OVERRIDE（2026-07-26）**：本文前半部分保留的是定位过程和
> rename 前历史命令，不再作为当前 source of truth。当前 canonical 实现是
> `models.step3p5.decode_fwd:whole_decode_step3p5`，
> `step3p5_opt` 仅保留 compatibility shim；fork
> `stepfun/develop@29547af6`，发布镜像
> `stepfun-develop-20260726-step3p5@sha256:f58708d2…`。当前操作入口、pin 和
> 未关闭边界以本文 §8.9、`STATUS.md`、`planning/handoff.md` 和
> `deployment/version-matrix.md` 为准。

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

---

## 8. 2026-07-26 最终回归与 0724 provenance 核验（本节覆盖前述待办）

### 8.1 最终代码状态

- 当前 active checkout：`/data/chensiyu/hw_project/pypto/workspace/vllm-pypto`
- 分支：`perf/step3p5-bc-v2`
- 最终 HEAD：`703739fb`
  - `d4491b32`：为 MTP KV IPC import 增加 `region_bytes`，修复 0724
    runtime 的 child-pointer provenance dispatch failure；
  - `0e85777c`：更新过时的 hidden-only contract 单测，使其接受正式的
    `per_layer_hidden` accuracy dump；
  - `667c949c`：清理 `decode_fwd.py` 的历史 trailing whitespace，AST 语义树
    保持完全一致；
  - `09897feb`：修正 whole-sidecar MTP protocol selftest 的 position/slot
    夹具，生产 slot 公式未改变。
  - `ad478abb`：为 production `whole_decode_sidecar.py` 增加显式 Main
    `--layer-module/--layer-name` wiring；这是 256-step 与 sidecar 设备回归
    实际使用的 commit；
  - `703739fb`：不改变已验证的 opt 数学实现，将
    `models.step3p5_opt.decode_fwd:whole_decode_opt` 设为 harness/production
    sidecar 的 release 默认；只有显式 `--baseline-main` 才回退到 0724
    canonical baseline。
- 当前工作树 clean；临时 `diagnostic-no-oracle` MTP wrapper 未进入代码仓。

### 8.2 与 0724 镜像的严格边界

固定镜像：

```text
hub.i.basemind.com/stepcast/vllm-pypto:stepfun-develop-20260724
digest: sha256:2b0dc4612796a34bea6720ccb4bf8fa3af4ea406cdd0f12add34586ca860d7e0
```

镜像内部 pin 已再次确认：

```text
pypto-lib: fd26b1be8683764befcb2f85423dd4d60f0bcaf5
pypto:     ca21ab5fcfd8203165928428302d273c377db5c6
pto-isa:   ecb6c303f797749f811a494742c3c08156aacabb
ptoas:     0.50
pypto:     0.1.0
CANN:      9.0.0-beta.1
driver:    25.5.2
```

当前分支与 0724 镜像 `pypto-lib` base 的 merge-base 正好是
`fd26b1be8683764befcb2f85423dd4d60f0bcaf5`；相对该 base 的最终 tree
差异收敛为以下 10 个文件：

```text
A models/step3p5/_compile_mtp_layer_hidden.py
M models/step3p5/decode_layer_single_chip_hidden.py
A models/step3p5_opt/__init__.py
A models/step3p5_opt/decode_fwd.py
M tests/step3p5/harnesses/_stage_main_hidden_only.py
A tools/step3p5/compare_per_layer.py
M tools/step3p5/pypto_mtp_kv_ipc.py
M tools/step3p5/whole_decode_holder.py
M tests/step3p5/unit/test_main_hidden_only_contract.py
M tools/step3p5/whole_decode_sidecar.py
```

0724 镜像中的以下基线文件与当前分支逐字一致：

```text
tools/step3p5/pypto_weight_ipc.py
tools/step3p5/pypto_kv_ipc.py
tests/step3p5/ci/run_whole_network_ci.py
```

因此，结论不是“整个 checkout 与镜像逐字相同”：当前 opt loop、holder
4-bucket split、per-layer instrumentation 和 MTP provenance fix 都是有意的
overlay；但没有混入其他 runtime/toolchain，公共 base、未改 helper 和镜像
内部组件均保持 0724 pin。

### 8.3 pypto-image-verify 回归

0162 使用设备 `8,9,10,11,12,13,14,15`，同一 checkpoint：

```text
/data/chensiyu/step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp
```

skill smoke 重新通过：

```text
[smoke] ptoas   : ptoas 0.50
[smoke] pypto   : 0.1.0
[smoke] simpler : OK
[smoke] runtime : .../a2a3/dispatcher/libsimpler_aicpu_dispatcher.so
[smoke] PASS
```

并确认 `ldd /workspace/ptoas-bin/ptoas` 的 `not found` 数为 0。

重新运行了三份 Main：

```text
0724 image pristine:
  /tmp/pypto_regress_20260726_d4491b32/image_pristine_main
current baseline:
  /tmp/pypto_regress_20260726_d4491b32/baseline
current opt:
  /tmp/pypto_regress_20260726_d4491b32/opt
```

结果：

```text
Main token chain（baseline/opt 共同）:
step0  6127 -> 303
step1   303 -> 1207
step2  1207 -> 6127    # harness stale expected=19384
step3 19384 -> 872
step4   872 -> 428
step5   428 -> 6127
step6  6127 -> 4231
step7  4231 -> 2636
```

- baseline 与 opt：8/8 个 `main_stepXX_hidden.pt` BF16 `torch.equal`；
- baseline 与 opt：每一步 L0-L44 均 45/45 exact，共 360 个逐层输出，
  `max_abs_diff=0`；
- 0724 image pristine 与 current baseline：8/8 step hidden exact；
- 0724 image pristine 与 current opt：8/8 step hidden exact；
- 每一步 `hidden_finite=true`、`hidden_tp_spread=0.0`。

因此，当前 Main opt 已证明与 **0724 镜像 pristine Main** 以及当前 baseline
在本 canonical 8-step teacher-forced 输入链上逐字对齐。step2 的 7/8 报告
是 stale oracle 问题：baseline、opt 和镜像 pristine 三者均实际得到
`1207 -> 6127`，并非 opt regression。

#### production sidecar opt wiring 设备验证

已在 0162 使用同一固定镜像、同一 checkpoint、设备 `8..15`，对新独立
快照
`/mnt/persist/chensiyu/pypto_image_verify_20260726_ad478abb`
做了 production sidecar 的最小真实设备验证：

```text
sidecar holder program: whole_decode_opt
compile: OK
AF_UNIX request: input token 6127 / valid_tokens=1
response: next_hidden shape [1,4096]
response program: whole_decode_opt
hidden vs current opt Main step0: exact=True, max_abs_diff=0
CPU tail token: 303
dt_sec: 2.271489...
```

artifact：

```text
/tmp/pypto_sidecar_opt_ad478abb2/sidecar_opt_report.json
```

因此，`whole_decode_sidecar.py` 的 opt 选择已经不再只存在于 standalone
`_stage_main_hidden_only` harness：`run_sidecar()` 和
`run_combined_sidecar()` 都能构造并执行 loop-form Main。该设备证据产生于
`ad478abb` 的显式
`--layer-module models.step3p5_opt.decode_fwd
--layer-name whole_decode_opt`；最终 `703739fb` 将同一实现设为默认，
`--baseline-main` 才回退到 0724 baseline。只提供自定义 module/name 的
其中一个仍会直接拒绝，避免半配置。
本次验证覆盖了 sidecar 的真实 compile、IPC attach、单次 dispatch 和
socket 返回；没有把 0..7 上的 vLLM front 服务停掉，也没有宣称已完成
独立的 live vLLM Main+MTP 全链路 A/B。

### 8.4 MTP 回归与 oracle 代际边界

使用 2026-07-18 的 pinned MTP oracle：

```text
/data/chensiyu/hw_project/pypto/workspace/logs_n1/
live_mtp3_patch_ci4_inline_runtime_20260718_220645
```

当前 MTP 程序在 0724 runtime 下 active-batch=1 和 16 均通过：

```text
absolute layer 45/46/47:
  hidden_pass_rate = 1.0
  hidden_max_abs_diff = 0.0
  hidden_tp_spread = 0.0
  tokens = [6178, 410, 303]
```

对应 artifact：

```text
/tmp/pypto_regress_20260726_d4491b32/mtp_oracle_ab1
/tmp/pypto_regress_20260726_d4491b32/mtp_oracle_ab16
```

这证明 `d4491b32` 的 MTP KV pool provenance 修复以及 MTP compile/dispatch
路径有效。

但旧 MTP oracle 的 previous hidden 来自旧 Main 代际，不能直接作为当前
0724 image Main 的端到端 oracle。证据：

```text
old P42_nh_row0.pt row0 absmax = 612
current 0724 Main step0 hidden absmax = 660
max_abs_diff = 146
mean_abs_diff = 11.060949325561523
torch.equal = false
```

用当前 baseline step0 hidden 和当前 opt step0 hidden 分别喂入同一个临时
diagnostic MTP wrapper（只跳过旧 hidden/token oracle 断言）时，三层 hidden
仍然逐字相同，active-batch=1 的 token 也相同：

```text
baseline = opt:
  MTP45 hidden exact, token 303
  MTP46 hidden exact, token 303
  MTP47 hidden exact, token 54107
```

仓内 CPU ctx=1 reference 也用当前 0724 Main step0 hidden 重新运行过：

```text
reference tokens: [303, 303, 42209]
device baseline/opt tokens: [303, 303, 54107]

MTP45: pass_rate=1.0, max_abs_diff=0.25
MTP46: pass_rate=1.0, max_abs_diff=0.50
MTP47: pass_rate=1.0, max_abs_diff=0.75
```

因此，当前 baseline/opt MTP 之间没有观察到差异，但 MTP 的**绝对精度/
token 对齐仍未闭环**：旧 `[6178,410,303]` 是旧 Main 代际结果，CPU
reference 与设备在第三层也存在 token 分歧。该 CPU reference 当前只能作
诊断参考，不能替代同代 live vLLM A/B 或已验证的 device oracle。

production sidecar wiring 已完成，最终 `703739fb` 已把 opt 设为默认。正常
生产启动不再需要 Main 选择参数：

```bash
python -m tools.step3p5.whole_decode_sidecar \
  --serve-all \
  ...
```

需要 rollback 时显式增加 `--baseline-main`。`--layer-module/--layer-name`
仍保留给成对指定其他自定义 Main，不能与 `--baseline-main` 同时使用。

### 8.5 单测与最终判定

相关合同/单测：

```text
23 passed, 1 skipped
tests/step3p5/unit/test_mtp_layer_holder.py
tests/step3p5/unit/test_main_hidden_only_contract.py
tests/step3p5/unit/test_mtp_hidden_only_contract.py
tests/step3p5/ci/test_whole_network_ci.py
```

最终判定：

1. **当前 Main opt 可以在 0724 镜像中替代当前 baseline Main 的计算路径**：设备上
   8-step 主 hidden 与逐层输出均 bit-exact，且直接与镜像 pristine Main
   交叉核对通过。
2. **当前版本不是“与 0724 镜像所有源码逐字一致”**：它是从
   `fd26b1be` 演进出来的 opt overlay；runtime/toolchain 和公共 base
   一致，差异文件仅限上述 10 个预期文件。
3. **当前版本尚不能完全平替 `stepfun/develop` base 的 production
   Main+MTP pipeline**：
   - production sidecar 的 opt wiring 已完成并通过 0162 设备上的真实
     compile/AF_UNIX 单请求验证；`703739fb` 已将同一 opt 设为默认，但尚未
     完成独立 live vLLM front 接管验证；
   - current Main→MTP 的绝对 token/hidden oracle 尚未同代闭环；
   - 旧 MTP oracle 的 active-batch 1/16 通过，只能证明旧输入代际下的
     MTP 程序和 IPC 路径可用。
4. **可以确认的范围**：0724 runtime/toolchain provenance 正确；standalone
   Main opt 与 0724 pristine Main、当前 baseline 均逐步/逐层 bit-exact；
   current baseline/opt MTP A/B 也逐层 exact；production sidecar opt wiring
   已完成。要达到“完全平替”仍需同代 MTP absolute oracle 和独立 live vLLM
   front 接管验证；本次 replacement regression 不应被描述为完整
   `stepfun/develop` Main+MTP serving 已完成。

### 8.6 2026-07-26：256-step vLLM A/B 与发布收口

在上述单步、8-step、0724 provenance 和 sidecar 设备验证之后，使用固定
0724 镜像在 0162 做了用户指定的 **256-step vLLM 长序列回归**：

```text
image:  hub.i.basemind.com/stepcast/vllm-pypto:stepfun-develop-20260724
digest: sha256:2b0dc4612796a34bea6720ccb4bf8fa3af4ea406cdd0f12add34586ca860d7e0
seed:   6127
steps:  256
oracle: vanilla vLLM cards 0-7
baseline/opt: current code cards 8-15, teacher-forced
```

严格区分两条结论：

| gate | 结果 | 发布口径 |
|------|------|----------|
| vanilla raw alignment | opt `240/256=93.75%`；baseline `240/256=93.75%` | 旧 `>=95%` raw gate 未通过 |
| opt replacement equivalence | token `256/256` exact；hidden `256/256` exact；`max_abs_diff=0`；TP spread `0.0` | **通过，可作为 replacement regression PASS** |

opt 与 baseline 在 256 步每一步 token 完全相同、hidden 逐字节相同；
`step127`、`step128` 跨 KV block 边界和 `step255` 均通过。raw miss 为：

```text
2, 20, 49, 52, 57, 62, 125, 131,
151, 153, 161, 162, 187, 221, 231, 252
```

同一显式上下文下对 vanilla miss 重查，观察到 fresh top-1 切换和近似
tie；同时 baseline 完全复现 240/256。因此 `93.75%` 是当前 vanilla
raw oracle 结果，不能被写成无条件 vanilla precision PASS，也不能归因
为 opt regression。

回归产物：

```text
0162:/tmp/live_ab_opt_ad478abb_n256_20260726/raw_alignment_summary.json
0162:/tmp/live_ab_opt_ad478abb_n256_20260726/baseline_opt_n256_compare.json
0162:/tmp/live_ab_opt_ad478abb_n256_20260726/vanilla_miss_requery.json
```

上述 device artifact 绑定 `ad478abb`；最终 `703739fb` 只改变默认入口和对应
合同/文档，没有改变 `whole_decode_opt` 的程序实现。新镜像需再验证无显式
Main 参数时实际选择 opt，以及 `--baseline-main` rollback 合同。

### 8.7 2026-07-26 新镜像发布与默认入口复验

最终镜像已构建并推送：

```text
image:  hub.i.basemind.com/stepcast/vllm-pypto:stepfun-develop-20260726-opt-b2
digest: sha256:0b22fcef3477488b82e7c8b6fd72341b55102605563faa093d911fa238830270
config: sha256:3792c1b6496092f04a73c6ef13db3d7dadc063afe15a1822c66df8ddd7e941e4
```

0162 静态验证：

```text
pypto-lib / vllm-pypto = 703739fb
pypto                    = ca21ab5f
simpler                  = 216e7632
pto-isa                  = ecb6c303
PTOAS                    = fc8c6cae
ptoas                    = 0.50
ptoas ldd not found      = 0
pypto-smoke              = PASS
image git credential audit = PASS
```

设备 8–15 上：

- 不传 Main 选择参数时实际打印
  `program=whole_decode_opt`；
- `--baseline-main` 实际打印
  `program=whole_decode_faithful_real_single_chip_hidden_only`；
- 两者在内置 8-step teacher-forced 链上均为 7/8，仅 step 2 命中已知
  stale oracle，输出均为 `6127`；
- 新镜像默认 opt 完整 N=256 为 `240/256=93.75%` raw；
- 新镜像默认 opt 与此前 `ad478abb` current baseline：
  token chain `256/256` exact，hidden `256/256` exact，
  `max_abs_diff=0`，TP spread `0.0`；
- `step127`、`step128`、`step255` 均通过。

新镜像默认入口 artifact：

```text
0162:/tmp/image_verify_20260726_0b22fcef_n256_default/
```

因此 `703739fb` 的默认入口切换和镜像封装没有改变已经验证的 replacement
行为。vanilla raw 95% gate 仍未通过，发布口径不变。

安全说明：第一次构建的旧 digest `sha256:285514c1…` 在源码 checkout 的
`.git/config` 中保留了 credential-bearing remote URL，已废弃。Dockerfile
现按 immutable vLLM commit 构建，并在 clone 后 scrub GitHub remote 与
submodule URL；最终 `sha256:0b22fcef…` 已在 0162 验证
`IMAGE_GIT_CREDENTIAL_AUDIT=PASS`。

发布收口后的 active workspace：

```text
pypto-lib / vllm-pypto: 703739fb (perf/step3p5-bc-v2)
pypto:     ca21ab5fcfd8203165928428302d273c377db5c6
simpler:   216e7632267ae815c484cdeba7991c87fabf3086
pto-isa:   ecb6c303f797749f811a494742c3c08156aacabb
PTOAS:     fc8c6caee561914b4fb991dfc8427bb63194269e
ptoas:     v0.50
```

旧 pypto-lib experiment checkout 的 dirty 内容已保存到：

```text
/tmp/pypto_workspace_dirty_backup_20260726
```

active workspace 已不再保留这些 variant checkout；工具链仓保持独立，
没有把 pypto-lib 内容覆盖到 `pypto`、`pto-isa` 或 `PTOAS`。历史
`workspace/_ws_archive_20260723` 仍有 root-owned 归档文件，因未获提权
删除批准未强删；它不是 active source checkout，也未用于构建或运行。

### 8.8 已完成优化、改造前后与收益

| 优化 | 改造前 | 改造后 | 已确认收益 |
|------|--------|--------|------------|
| **A1 可观测性** | whole-net 主要以“是否跑通”判断，缺少统一的多步 token/hidden、finite、TP spread 和边界 step 证据 | harness 生成 N=256 report、逐 step hidden，并检查 hidden finite、TP spread、step127/128/255 | 回归结果可审计；当前 TP spread `0.0`，且能把 vanilla raw gate 与 replacement gate 分开 |
| **B1 resident 权重池的 opt zero-copy view ABI** | 0724 baseline 已经一次 IPC import、跨 step resident，但 opt 没有从 canonical FULL `[12,...]` / SWA `[33,...]` 栈取得 10/30 层 MoE-only 连续 bucket 的 ABI | `Wsub()` 对每 rank 做 FULL `1:11`、SWA `2:32` outermost contiguous slice，再以 `StackedDeviceTensor` 跨 rank 绑定；不 materialize 新权重 | B2 可以用 dynamic `pl.slice(layer_idx)`；相对复制 opt 专用 attention buckets，按当前 shape 避免约 `965 MiB≈0.94 GiB/rank` 额外设备副本。0724 原本已 resident，**不能**写成“消除 24 GiB/rank/step H2D” |
| **B2 45 层 loop-form** | historical `decode_layer.py` 31,686 行，MoE 主体按 40 个 layer site 重复描述 | current `decode_fwd.py` 4,775 行；L1/L2=`pl.range(2)`、L3-L42=`pl.range(40)`，L43/L44 保留必要 specialization | 主体源码约减少 **84.97%**，MoE runtime loop body `40→1`；N=256 opt↔baseline token/hidden `256/256` exact、`max_abs_diff=0`。没有同环境 compiler wall-clock A/B，不宣称编译加速比例 |
| **C2 dispatch/combine pull** | source rank push/`remote_store`，跨 die 写完成存在竞争和随机 stall/507018 风险路径 | fixed-slot peer-major，consumer `remote_load` pull；combine 固定槽 pull-back，本地 bucket 与 peer pull 分开 | 当前 0162 N=256 无 stall、TP spread `0.0`，liveness 可重复性提高；通信字节数不变，不宣称带宽下降 |

边界：

- C1 单 window + `moe_epoch` + `WaitCmp.Ge` 尚未进入 current release；
  `~766 MB→十几 MB` 仍只是设计目标。
- B3 KV resident/in-place、D1/D2、C3、E1、F1/F2/F3、G1 以及 live
  production front/KV/MTP/HBM 尚未闭环，不能计入本次收益。
- 当前 harness warm `run_sec` 仅用于记录运行时间，不是完整 serving ITL；
  没有旧 31k-line implementation 的同环境 A/B，因此不写“B2 已加速 X%”。

### 8.9 canonical `step3p5` 正式发布（覆盖 §8.1–8.7 的 active 口径）

最终 active release 已不再使用 `perf/step3p5-bc-v2@703739fb` 或
`stepfun-develop-20260726-opt-b2` 作为默认入口。正式状态为：

```text
pypto-lib / vllm-pypto:
  branch stepfun/develop
  commit 29547af6c3c5b7db2a75c1fd5e0110959d2a7624

canonical Main:
  models.step3p5.decode_fwd:whole_decode_step3p5

compatibility shim only:
  models.step3p5_opt.decode_fwd:whole_decode_opt

image:
  hub.i.basemind.com/stepcast/vllm-pypto:stepfun-develop-20260726-step3p5
  digest sha256:f58708d2c6cc60474fa98da38aef23a128b850173e4bf5ab6336590b83e2afbc
  config sha256:a9d4c288b0aeb15d912724d6df717ffac37a27f6826ac33ec864ecf775104ca3
```

rename 后 canonical 文件保留唯一真实 PyPTO program；`step3p5_opt` 的
26 行文件只 re-export canonical object，不再维护第二份程序。0162 已验证：

- 镜像内六个 pin、`ptoas 0.50`、credential audit、canonical/shim import
  identity 和 `/workspace/pypto-smoke.sh` 全部 PASS；
- 默认 8-step device smoke 实际打印
  `program=whole_decode_step3p5`，hidden 全 finite、TP spread `0.0`；
  仅旧硬编码 oracle 的 step2 预期 `19384` 与实际 `6127` 不同，其余
  `7/8` exact；
- rename 前后 N=256 token/hidden `256/256` bit-exact，
  `max_abs_diff=0`，step127/128/255 PASS；
- vanilla raw 仍为 `240/256=93.75%`，低于历史 95% gate，不能写成
  vanilla precision PASS。
- 新镜像完整 N=256 artifact：
  `/tmp/newimage_step3p5_n256_20260726/`；与既有 canonical artifact
  token/hidden `256/256` exact、`max_abs_diff=0`、TP spread `0.0`，
  step127/128/255 PASS，raw miss 与此前 16 个位置完全一致。

active workspace：

```text
local workspace/vllm-pypto: stepfun/develop@29547af6
local workspace/pypto-lib:  detached@29547af6
0162 workspace/vllm-pypto:  stepfun/develop@29547af6
0162 workspace/pypto-lib:   detached@29547af6
0162 workspace/pypto-lib-n1: detached@29547af6
0162 workspace/pypto:       detached@ca21ab5f
0162 workspace/pypto/runtime: detached@216e7632
0162 workspace/pto-isa:     ecb6c303
0162 workspace/PTOAS:       detached@fc8c6cae
```

0162 的根 `workspace/pypto-lib` 历史 dirty work 已保存到 git stash
`archive pre-canonical root workspace 2026-07-26`，随后切到 detached
`29547af6`；`/mnt/persist/chensiyu/perf-opt-ws` 未操作，仍只作历史实验
worktree，不是 active release source。
