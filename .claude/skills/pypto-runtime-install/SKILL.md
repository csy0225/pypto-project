---
name: pypto-runtime-install
description: >
  从零在一台 Ascend 910B 硬件上装好 pypto 运行时环境的分步 runbook（参照 0162
  stepfun/develop 现状）。当用户拿到新机器/新 pod、拉了 GitHub 仓库代码，问"接下来
  装环境的步骤是什么 / 怎么跑起来 / 怎么避免装错"时使用。覆盖：Phase 16 三剑合璧
  版本核验、5 仓库拉取与 pin、Python venv、pip install -e、simpler runtime 构建、
  ptoas-bin、三件套激活、smoke/L3/canonical 验证，以及每步的高频踩坑与规避。
---

# pypto 运行时环境安装 runbook（Ascend 910B）

> 目标：**用户拿到一台硬件 → 拉仓库 → 按本文步骤把 pypto 运行时装到能跑
> smoke + 多卡 allreduce + 单卡 ST**。每步都给"正确做法 + 高频踩坑"。
> 参照 `gpu-a910x-0162` 上 `stepfun/develop` 的现状。
>
> **权威出处（出问题先查这几篇）**：
> - 版本硬绑定：[`deployment/phase16-three-pillars.md`](../../deployment/phase16-three-pillars.md)
> - 版本兼容矩阵：[`deployment/version-matrix.md`](../../deployment/version-matrix.md)
> - 机器恢复/升级：[`deployment/machine-recovery.md`](../../deployment/machine-recovery.md)
> - 唯一可复现环境 SSOT：[`develop/N1/N1-STABLE-ENV-0162-20260717.md`](../../develop/N1/N1-STABLE-ENV-0162-20260717.md)
> - 多卡 IPC 报错复盘：[`postmortems/01-multirank-ipc-507899-507018.md`](../../postmortems/01-multirank-ipc-507899-507018.md)

---

## 0. 全景：装完长什么样

一个 workspace 目录（NVMe/持久盘上，**不要放 tmpfs**）里并列 5 个仓库 + 1 个
binary + 1 个 venv：

```text
<workspace>/                 # 如 /data/chensiyu/hw_project/pypto/workspace
├── pypto/                   # 框架；simpler 是它的 submodule (pypto/runtime)
│   └── runtime/             # = simpler（PTO runtime）
├── pypto-lib/               # step3p5 模型 + kernel
├── pto-isa/                 # tile ISA（$PTO_ISA_ROOT 指这里）
├── PTOAS/                   # 字节码 assembler 源码
├── ptoas-bin/               # ptoas 二进制发布（v0.45），进 $PATH/$LD_LIBRARY_PATH
├── .venv311/                # Python 3.11 venv
└── activate.sh              # 激活脚本（只设 venv+PTOAS，不设 CANN！见 Step 6）
```

**硬前提（不满足则多卡必挂，先做 Step 1-2）**：driver `25.5.2` + firmware
`7.8.0.7.220` + CANN `9.0.0-beta.1`(**非 GA**) —— Phase 16 三剑合璧，缺一不可。

---

## 用法（给 agent：分两阶段执行，最终反馈"环境是否可用"）

本 skill 指导 agent 把一台新机器的 pypto 环境弄到**可用**，分两阶段：

- **Phase A · 环境安装**（Step 1–6）：核验/装 driver·firmware·CANN → 拉 5 仓 → venv+pip → build_runtimes → 激活。
- **Phase B · 安装后 CI smoke**（Step 7）：跑权威 CI runner，**通过即判定"环境可用"**。

agent 执行要点：
- 按 Step 顺序，每步先跑"检查"命令，达标才继续；不达标按该步「踩坑」修。
- Phase A 里 driver/firmware/CANN 的 **sudo / 一次性操作**若不达标：**先停下告诉用户**（可能要 root、走 [`machine-recovery.md`](../../deployment/machine-recovery.md)），**不要擅自装驱动**。
- 最终**反馈模板**（回给用户）：
  ```
  环境: 可用 ✅ / 不可用 ❌
  卡在: <Step N / 无>
  证据: CI smoke Main tokens=[...]（或多卡 allreduce max=0）
  待办: <需用户做的一次性事项：升 driver / 挂 checkpoint / …>
  ```

---

## Step 1 · 核验硬件 / driver / firmware

```bash
npu-smi info -t board -i 0 | grep -E "Software|Firmware"
# 期望：Software Version : 25.5.2   |   Firmware Version : 7.8.0.7.220
npu-smi info                       # 看 8/16 张卡 health=OK、HBM=65536MB
```

- ✅ 达标 → 去 Step 2。
- ❌ 版本低 / 卡不全 → **先按 [`deployment/machine-recovery.md`](../../deployment/machine-recovery.md) 升级 driver+firmware（成对升，重启）**，别继续往下装。

> **踩坑**：
> - driver+firmware **必须成对**，`support_shmem_map_exbus` cap 由两者共同 gate；只升一个 → 跨卡 IPC `aclrtIpcMemImportByKey 507899`。
> - netboot/tmpfs 机器（如 0162）**重启会丢 driver**，走 machine-recovery 的 `RECOVERY.sh`；持久盘 (`/mnt/persist`,`/data`) 不丢。
> - firmware 有时 `npu-smi` 直接 readback 显示 `NA`（chip flash 已刷但读不回），以部署记录为准。

## Step 2 · CANN `9.0.0-beta.1`（**非 GA**）

```bash
ls -la /usr/local/Ascend/cann            # 应 symlink 到持久盘上的 non-GA/beta.1 安装
test -f /usr/local/Ascend/cann/set_env.sh && echo OK
```

> **踩坑（最容易错的一条）**：
> - **绝对不能用 CANN GA**。GA 的 TDT 不把 `Ascend-aicpu_extend_syskernels.tar.gz`
>   推到 AICPU 端 → simpler init `507018 (BootstrapDispatcher)`。必须 beta.1
>   /non-GA。详见 [`phase16-three-pillars.md`](../../deployment/phase16-three-pillars.md) "CANN GA failure mode"。
> - 集群自动化可能把 CANN 悄悄 revert 成 GA →**装前先备份** beta.1，坏了从备份恢复 symlink。
> - 不要"顺手升级 CANN"，除非 Huawei 出了新 beta/GA 且验证过。

## Step 3 · 拉 5 个仓库到 workspace + 对 pin

从 GitHub fork（`csy0225/*`）克隆，全部切 `stepfun/develop`。**simpler 是 pypto
的 submodule**，不要单独平级 clone。

```bash
export WS=/data/chensiyu/hw_project/pypto/workspace   # 换成你的持久盘路径
mkdir -p "$WS" && cd "$WS"

git clone -b stepfun/develop https://github.com/csy0225/pypto.git
git -C pypto submodule update --init --recursive      # 拉出 pypto/runtime = simpler
git clone -b stepfun/develop https://github.com/csy0225/pypto-lib.git
git clone -b stepfun/develop https://github.com/csy0225/pto-isa.git
git clone -b stepfun/develop https://github.com/csy0225/PTOAS.git
```

**当前 stepfun/develop 参考 pin**（2026-07-18 N=1 single-submit 合入；精确可复现
pin 以 [`N1-STABLE-ENV`](../../develop/N1/N1-STABLE-ENV-0162-20260717.md) §2 为准）：

| 仓库 | 参考 commit |
|------|-------------|
| pypto | `9ec303f6` |
| pypto-lib | `e1513d22` |
| simpler (pypto/runtime submodule) | `c7fdc574` |
| pto-isa | `ecb6c303`（≈ origin/main） |
| PTOAS (src) | `72ada0a1` |
| ptoas-bin | `v0.45`（binary，见 Step 5） |

> **踩坑**：
> - **只拉 pypto-lib 不构成同一测试对象**——pypto 的 `StackedDeviceTensor`/`import_ipc_all`、simpler 的 forked-child ACL IPC import、runtime build 产物缺一都跑不通。5 仓要一起对齐。
> - simpler 的 pin **由 pypto 的 submodule 决定**；改 simpler 后必须 `git submodule update` 并在 pypto 侧 commit submodule pin。
> - 私有 fork 拉不动时用 PAT（`/data/chensiyu/secrets/github.env`），push/pull 走 `git -c http.version=HTTP/1.1`（默认 HTTP/2 在部分内网 130s 静默超时）。

## Step 4 · Python venv + 安装 pypto / simpler

```bash
cd "$WS"
python3.11 -m venv .venv311
source .venv311/bin/activate
python -m pip install -U pip

# CPU torch + 基础依赖（0162 stable：torch 2.12.1+cpu / numpy 2.4.6 / safetensors 0.8.0）
pip install "torch==2.12.1+cpu" numpy safetensors pytest

# 装 pypto 框架（editable，带 native 扩展）
pip install --no-build-isolation -e "$WS/pypto"
# 装 simpler runtime（editable）
pip install --no-build-isolation -e "$WS/pypto/runtime"
```

> **踩坑**：
> - **不要传 `CMAKE_BUILD_TYPE=Release`**——会撞 `tensor.h buffer_elems -Werror=unused-variable`，用 cmake dev default 即可。
> - rebase / 换 pin 后第一次 build 先清缓存：`rm -rf "$WS"/pypto/build/cp311-* "$WS"/pypto/build/cache "$WS"/pypto/build/lib`。
> - Python 必须 3.11（venv 名 `.venv311`）。

## Step 5 · 构建 simpler runtime (a2a3) + ptoas-bin 就位

```bash
# ptoas-bin：从 PTOAS release 下载 v0.45 的 ptoas-bin-x86_64.tar.gz，解到 $WS/ptoas-bin
#   校验：$WS/ptoas-bin/bin/ptoas --version  ->  ptoas 0.45
# （运行 ptoas 前必须先 source activate.sh，否则缺 libMLIR...so.19.1）

# 构建 a2a3 平台 runtime .so（host_runtime / aicpu_kernel / aicore_kernel / dispatcher）
cd "$WS/pypto"
python -m simpler_setup.build_runtimes --platforms a2a3 --clone-protocol https
```

> **踩坑（netboot 机器重启后常缺这两个构建依赖）**：
> - 缺 cmake → `RuntimeError: CMake configuration not found`：装进 venv `pip install "cmake==3.31.6"`。
> - 缺 `libstdc++-12-dev` → ccec 编 aicore_kernel 报 `fatal error: 'cstdint' file not found`：`sudo apt-get install -y libstdc++-12-dev`（ccec 的 clang 选 gcc-12 工具链）。
> - 用 `--clone-protocol https`（容器内 ssh clone 会 hang）。
> - **a2a3 HOST 编译报 `PLATFORM_ONBOARD_SCHEDULER_TIMEOUT_MS` 未定义**：common HOST 代码
>   `src/common/platform/onboard/host/device_runner_base.cpp`（`resolve_onboard_timeout_config()`）
>   引用该宏，某些旧 fork 状态下 **a2a3 的 `src/a2a3/platform/include/common/platform_config.h`
>   漏定义（a5 有、a2a3 没有）→ 只有 a2a3 HOST 编译挂**。它只是 onboard AICPU 调度
>   no-progress 超时默认值（compile-time 常量 = `10000` ms），**不影响已建好的旧 `.so`
>   运行行为**。**当前 stepfun/develop 已修**（a2a3 `platform_config.h` 已定义 =10000，与
>   a5 一致）；卡在旧 checkout 的两条出路：① 拉到当前 pin；② 一行修复——在 a2a3 的
>   `platform_config.h` 补 `constexpr int32_t PLATFORM_ONBOARD_SCHEDULER_TIMEOUT_MS = 10000;`
>   （照抄 a5 的值）。**先判断影响范围**：若已有能跑的旧 runtime `.so`，仅运行不必重编（旧 install 仍可用）；只有要重建 a2a3 runtime 时才需要这个定义。

## Step 6 · 三件套激活（**每个新 shell 都要**）

`activate.sh` **只**激活 venv + 设 PTOAS/PATH，**不 source CANN**。每个新 shell：

```bash
source /usr/local/Ascend/cann/set_env.sh      # ← activate.sh 不做这步！
source "$WS/activate.sh"
export PTO_ISA_ROOT="$WS/pto-isa"
```

**或直接** `WS=<你的workspace> source .claude/skills/pypto-runtime-install/activate-pypto.sh`
（本 skill 附带，把上面三件套 + canonical 运行时参数一次性设好，含 python-not-in-PATH 兜底）。

> **踩坑**：
> - 少 source CANN → KernelCompiler `OSError: 'ASCEND_HOME_PATH' not set`；少 `PTO_ISA_ROOT` → `OSError: PTO_ISA_ROOT not set`。
> - 个别机器 `source activate.sh` 后 `python` 不在 PATH → 再补 `source "$WS/.venv311/bin/activate"`。
> - **跑过 monkey-patch 测试（`apply_perrank_patch`/`cfg.X=Y`）后**，下次 fresh run 前清 pyc：`find "$WS"/pypto-lib/models/step3p5 -name "*.py" -exec touch {} +`（否则 patched 全局被序列化进 `.pyc` 污染）。

## Step 7 · Phase B —— 安装后 CI smoke（判定"环境可用"）

分两级：先做**不需 checkpoint 的多卡 sanity**，再跑**权威 CI smoke**。

**① 多卡 collective sanity（无需 checkpoint，先跑这个）**

```bash
cd "$WS/pypto/runtime"
python examples/workers/l3/allreduce_distributed/main.py -p a2a3 -d 0-1
#   期望：max |out - expected| = 0.000e+00（Phase 16 三剑合璧真正生效的证据）
```

- 报 `507899` → driver/firmware 没到位（回 Step 1）。
- 报 `507018 (BootstrapDispatcher)` → CANN 是 GA（回 Step 2）。
- 决策树见 [`postmortems/01-multirank-ipc-507899-507018.md`](../../postmortems/01-multirank-ipc-507899-507018.md)。

**② 权威 CI smoke（需 checkpoint + cards 8–15）—— 通过即"环境可用"**

当前 live 线唯一权威 runner（preflight ckpt/PTO-ISA/cards → Main 45 层 hidden-only
8-step → MTP45/46/47 → 清理 exporter；详见
`pypto-lib/tests/step3p5/ci/WHOLE_NETWORK_CI.md`）：

```bash
cd "$WS/pypto-lib"
python -m tests.step3p5.ci.run_whole_network_ci \
  --ckpt /data/chensiyu/step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp \
  --devices 8,9,10,11,12,13,14,15 \
  --out /tmp/n1_single_chip_hidden_ci \
  --artifact-dir "$WS/logs_n1_0162/single_chip_hidden_ci"
```

**期望（通过判据）**：
- Main tokens = `[303, 1207, 19384, 872, 428, 6127, 4231, 2636]`
- MTP  tokens = `[6178, 410, 303]`
- 无 stall / 无残余 exporter PID。

> 数值正确与无 stall 是两个独立 gate。step1 对、step2 错 → 先查 token/embedding、
> metadata、KV 地址、repeated-run state，再查 weights/算子。

> **agent 判定**：① allreduce max=0 且 ② CI smoke Main/MTP tokens 匹配 → **环境可用 ✅**，
> 按 §用法 反馈模板回报。缺 checkpoint / 只有部分卡 → 至少跑 ① 并注明"CI smoke 待 ckpt"。

## Step 8 · （深入）N=1 canonical 复现细节

Phase B ② 的更细粒度复现（fresh exporter pool + worker、清理铁律、环境变量、
逐 rank 检查）见 [`reference/canonical-test.md`](../../reference/canonical-test.md) 与
[`N1-STABLE-ENV`](../../develop/N1/N1-STABLE-ENV-0162-20260717.md) §6。

> **踩坑**：exporter 仍存活时**严禁** `rm -rf "$OUT"`（会把 IPC pool 生命周期问题误诊成 kernel stall）；先 `touch "$OUT/STOP"` + `wait` 各 exporter PID 再清 marker。

---

## 高频踩坑速查（装错就查这张表）

| 现象 / 报错 | 根因 | 修法 |
|------------|------|------|
| `aclrtIpcMemImportByKey ... 507899` | driver<25.5.2 或 firmware<7.8.0.7.220 | 成对升 driver+firmware（Step 1） |
| simpler init `507018 (BootstrapDispatcher)` | CANN 是 GA，非 beta.1 | 换 non-GA/beta.1 + 恢复 symlink（Step 2） |
| `ASCEND_HOME_PATH not set` | 没 source CANN set_env.sh | 三件套第 1 行（Step 6） |
| `PTO_ISA_ROOT not set` | 没 export PTO_ISA_ROOT | 三件套第 3 行（Step 6） |
| `python` 找不到 | activate.sh 后 venv 未激活 | `source $WS/.venv311/bin/activate`（Step 6） |
| ptoas 缺 `libMLIR...so.19.1` | 没先 source activate.sh | 先激活再跑 ptoas（Step 5） |
| build `buffer_elems -Werror` | `CMAKE_BUILD_TYPE=Release` | 用 dev default，别传（Step 4） |
| `CMake configuration not found` | venv 缺 cmake | `pip install cmake==3.31.6`（Step 5） |
| ccec `'cstdint' file not found` | 缺 libstdc++-12-dev | `apt-get install libstdc++-12-dev`（Step 5） |
| a2a3 HOST 编译 `PLATFORM_ONBOARD_SCHEDULER_TIMEOUT_MS` 未定义 | 旧 fork 状态 a2a3 `platform_config.h` 缺该宏（a5 有），common `device_runner_base.cpp` 引用 | 拉当前 pin（已修=10000）；或在 a2a3 header 补 `constexpr int32_t PLATFORM_ONBOARD_SCHEDULER_TIMEOUT_MS = 10000;`。仅 compile-time，不影响旧 `.so` 运行（Step 5） |
| ptoas parse error | pypto 越过动 MLIR op 的 commit，ptoas-bin 太旧 | bump ptoas-bin ≥ v0.45（Step 5） |
| 跑通一次后结果变/污染 | monkey-patch 后 stale `.pyc` | `find models/step3p5 -name "*.py" -exec touch {} +`（Step 6） |
| git push/pull 130s 超时 | 内网 HTTP/2 | `git -c http.version=HTTP/1.1 ...`（Step 3） |

## 铁律（装环境时不可违反）

1. **Phase 16 三剑合璧缺一不可**（driver 25.5.2 / firmware 7.8.0.7.220 / CANN 9.0.0-beta.1）。
2. **CANN 绝不用 GA**。
3. **5 仓一起对齐 pin + simpler 走 submodule**，不单独拉某一个。
4. **持久盘装**（netboot/tmpfs 机器重启丢 driver/venv，走 machine-recovery）。
5. **每个新 shell 三件套激活**；monkey-patch 后清 pyc。
6. 验证以 **Phase B** 为准：多卡 `allreduce max=0` + CI smoke（`run_whole_network_ci` 的 Main/MTP tokens 匹配）= **环境可用**；缺一不算装好。
