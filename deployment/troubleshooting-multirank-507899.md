# 多卡 IPC 报错排查 —— 507899 / 507018

**适用对象**：在 Ascend 910B / A2A3 上跑多卡 step3p5 或 simpler L3
（`allreduce_distributed` 等），撞到 `aclrtIpcMemImportByKey 507899`
或 simpler init `507018`，怀疑是「明明文档说修好了，怎么还报错」的人。

> 这个问题的「解」是 [`phase16-three-pillars.md`](phase16-three-pillars.md)
> 的三剑合璧版本绑定。**还在报错 99% 是三件套没配齐**——只升了一两件，
> 或 CANN 用成 GA 了。本文给一个自助诊断流程。

---

## 一句话根因

跨卡 IPC 依赖驱动 capability `support_shmem_map_exbus`。这个 cap 由
**driver + firmware 共同 gate**，且 simpler init 还需要 **CANN beta.1**
把 AICPU 库推到设备端。**三件必须同时对**，缺任一件都会在不同环节报错。

---

## 诊断决策树（先看你报的是哪个错误码）

```
报错码是？
│
├─ 507899  (aclrtIpcMemImportByKey failed)
│   └─ cap 还是 0 → driver / firmware 没升全
│       ├─ driver  == 25.5.2 ?        ── 否 → 升 driver（见下）
│       ├─ firmware== 7.8.0.7.220 ?   ── 否 → 升 firmware（见下）
│       └─ 两个都对但仍 507899 ?
│           └─ flag 没传对 → ImportByKey 必须传
│              ACL_RT_IPC_MEM_IMPORT_FLAG_ENABLE_PEER_ACCESS (0x1)
│              （flag=DEFAULT/0 在修好的 driver 上仍 507899）
│
├─ 507018  (simpler init / BootstrapDispatcher,
│           aclrtSynchronizeStream failed)
│   └─ cap 已修好（不再 507899），但 CANN 用成 GA 了
│       └─ CANN == 9.0.0-beta.1 ?  ── 否（是 GA）→ 换回 beta.1
│          GA 的 TDT 不推 libaicpu_extend_kernels.so 到 AICPU 设备端，
│          BootstrapDispatcher 找不到 DynTileFwkKernelServerInit
│
└─ comm_init 段错 (HcclGetRootInfo 处崩)
    └─ 这是更早的 simpler#1018，不是本文范围
       → 确认 simpler 带 host CMakeLists `--no-as-needed` patch
         （libhost_runtime.so 的 DT_NEEDED 要有 libhcomm.so）
```

**关键直觉**：错误码会随你修复的进度往后挪。
`507899 → 507018 → 跑通` 就是三件套逐件补齐的正常轨迹。
**507899 变 507018 是进步**（driver/firmware 已对，只差 CANN）。

---

## ⚠ 新增根因（2026-07-10）：三件套全对 + flag 对，仍 507899 → SDMA workspace force-ON

**如果 driver 25.5.2 / firmware 7.8.0.7.220 / CANN beta.1 三件全对、ImportByKey flag 也对，
但多卡仍 507899，且 507899 之前紧跟着 `[SDMA] aclrtSynchronizeStream (aicpu) failed`** ——
这不是三件套问题，是 **simpler runtime 的 `SIMPLER_ENABLE_PTO_SDMA_WORKSPACE` 被 force-ON**。

- 根因：`ensure_sdma_workspace()`→`SdmaWorkspaceManager::Init()` 在 `comm_hccl.cpp:815`
  （`domain_alloc_via_ipc` 内）发一个 AICPU `aclnnShmemSdmaStarsQuery`，在本机 driver/CANN 上
  fault（507018），**毒化紧跟其后的跨卡 `aclrtIpcMemImportByKey`（507899）**。它按自己的 CMake
  注释是 "logically orthogonal to HCCL comm bootstrap, only needed by sdma_async_completion_demo"
  —— comm 不该依赖它。
- 为什么升级后才出现：Phase-16 验证栈（simpler `a6e06406`）本带 "SDMA OFF" patch；升级到
  origin `71e39623` 把这个 patch **丢了**并 force-ON。（升级栈 device 路径此前只做过 compile
  验证，没在真机跑过 → 潜藏 regression。）
- **判据**：对比 log —— 2026-06-29 成功 log 里 `[SDMA] aclrtSynchronizeStream (aicpu) failed`
  计数 = 0（当时 SDMA OFF）；升级栈失败 log 里每次都有，且在 507899 之前。
- **另一个独立症状**：同一份 stale/mismatched `.so` 会让**单卡** `hello_world -d 0` 也挂
  `aclrtSynchronizeStream (AICPU) failed 507018`（跟多卡无关）。

### 修复（两步都要）

```bash
# 1) 单卡 AICPU 507018：clean 重编 runtime（stale .so）
cd $WS/pypto && python -m simpler_setup.build_runtimes --platforms a2a3   # 注意：--clone-protocol 已移除
python examples/beginner/hello_world.py -p a2a3 -d 0    # 期望 PASS

# 2) 多卡 IPC 507899：关掉 SDMA workspace
#    pypto/runtime/src/a2a3/platform/onboard/host/CMakeLists.txt:42
#      set(SIMPLER_ENABLE_PTO_SDMA_WORKSPACE OFF)   # 并把 PTO_ISA_ROOT FATAL_ERROR + include 收进 if(...ON)
mv $WS/pypto/runtime/build/cache $WS/pypto/runtime/build/cache.bak.$(date +%s)   # 强制 reconfigure
cd $WS/pypto && python -m simpler_setup.build_runtimes --platforms a2a3
cd $WS/pypto/runtime/examples/workers/l3/allreduce_distributed
python main.py -p a2a3 -d 0-7 --mode twophase   # 期望 8 卡全 max|out-expected|=0.000e+00 ✅
```

**通用教训**：升级到 origin pin 后，必须审计"哪些本地 patch 被 origin 吞了 / 丢了"（这次丢的是
SDMA-OFF）；升级栈只做 compile 验证不够，device 路径（单卡 hello + `allreduce -d 0-7`）必须重新验证。

---

## 排查清单（按顺序核对）

```bash
# 1. driver + firmware（一条命令同时看）
npu-smi info -t board -i 0 | grep -E "Software|Firmware"
#   期望:
#     Software Version : 25.5.2
#     Firmware Version : 7.8.0.7.220
#   firmware 烧进 chip flash，跨重启持久；driver 在 tmpfs 机器上重启会丢

# 2. CANN —— 必须 beta.1，不是 GA
ls -la /usr/local/Ascend/cann-9.0.0-beta.1 && \
  test -f /usr/local/Ascend/cann-9.0.0-beta.1/set_env.sh && echo "CANN beta.1 OK"
#   若实际指向 /usr/local/Ascend/cann-9.0.0（GA）→ 这就是 507018 的原因

# 3. simpler runtime 带 --no-as-needed patch（comm_init 不段错的前提）
readelf -d <workspace>/pypto/runtime/build/lib/a2a3/onboard/.../libhost_runtime.so \
  | grep NEEDED
#   期望: DT_NEEDED 里出现 libhcomm.so
```

任一项不符 → 按 [`machine-recovery.md`](machine-recovery.md) 升级。

---

## 修复动作

### 升 driver + firmware（507899）

`.run` 包已 stage 在 `gpu-a910x-0162:/mnt/persist/ascend-staging/`：

```
Ascend-hdk-910b-npu-driver_25.5.2_linux-x86-64.run
Ascend-hdk-910b-npu-firmware_7.8.0.7.220.run
```

升级前**必须**先 drain Kubernetes daemonset（`device-plugin` /
`npu-exporter` 占着 `/dev/davinci*`，否则 `--upgrade` 报设备 busy）：

```bash
systemctl stop kubelet
systemctl stop bip-agent          # 视集群而定
# 确认没有进程占着 /dev/davinci*
sudo ./Ascend-hdk-910b-npu-driver_25.5.2_linux-x86-64.run --upgrade --quiet
sudo ./Ascend-hdk-910b-npu-firmware_7.8.0.7.220.run --upgrade --quiet
reboot
```

完整步骤（NVMe 持久化、K8s drain 顺序、netboot/tmpfs 注意事项）见
[`machine-recovery.md`](machine-recovery.md)。

### 换回 CANN beta.1（507018）

升 driver/firmware 时**不要**顺手把 CANN 升成 GA。如果已经是 GA：

```bash
# 把 host 的 CANN 切回 beta.1（保留新 driver+firmware）
# beta.1 install 应已备份在 persistent storage
source /usr/local/Ascend/cann-9.0.0-beta.1/set_env.sh
```

⚠ **升级前给 CANN beta.1 备份**，防集群自动化脚本把它覆盖成 GA：

```bash
sudo cp -a /usr/local/Ascend/cann-9.0.0-beta.1 \
  /<persistent>/cann-9.0.0-beta.1.backup-$(date +%Y%m%d)
```

---

## 验证修好了

两层验证，从底层 API 到 e2e：

```bash
# Layer 1: 纯 IPC probe（fork 双进程，无 simpler 依赖）
# probe2 binary 在 gpu-a910x-0162:/data/chensiyu/probe/probe2
./probe2
#   期望: ImportByKey(flag=ENABLE_PEER_ACCESS) -> 0
#         peer_va == parent ptr （跨卡 same-VA mapping 建立）

# Layer 2: simpler L3 双卡 allreduce e2e
source /usr/local/Ascend/cann-9.0.0-beta.1/set_env.sh
source <workspace>/activate.sh
export PTO_ISA_ROOT=<workspace>/pto-isa
cd <workspace>/pypto/runtime
python examples/workers/l3/allreduce_distributed/main.py -p a2a3 -d 0-1
#   期望: max |out - expected| = 0.000e+00 两卡都通过 (golden match)
```

---

## 反证：为什么这不是 simpler 的 bug

同 pod / 同 8 卡 / 同 CANN / 同旧 driver 上，**vLLM-Ascend TP=8 能跑通**：

```bash
vllm serve <model> --tensor-parallel-size 8   # 8 ranks 全 ready
```

vLLM-Ascend 走 CANN HCCL **集合通信**（`init_process_group("hccl")`），
完全不碰 `aclrtIpcMemImportByKey` / `halShmemOpenHandleByDevId`——HCCL
collective 绕开了 user-space 跨卡 device-memory IPC。所以 chip / runtime /
HCCL 都健康；缺的是 simpler DIY-IPC 后端依赖的「跨卡 Ex-Bus shmem
mapping」这个特定 driver capability。

---

## 试过但没用的（别重复踩）

| 尝试 | 结果 |
|------|------|
| ImportByKey 前先 `aclrtDeviceEnablePeerAccess` | 还是 507899（device peer access ≠ exbus shmem cap） |
| 只升 driver 不升 firmware（或反之） | cap 仍 0，继续 507899（两者共同 gate） |
| 升完 driver/firmware 顺手升 CANN 到 GA | cap 修好了但变 507018（AICPU 库没推） |
| 换不同 chip pair (0↔1 / 0↔7 / 6↔7) | 都崩，不是 chip-pair 路由问题 |

---

## Issue / 链接

| Issue / PR | 性质 | 状态 |
|---|---|---|
| [simpler#1037](https://github.com/hw-native-sys/simpler/issues/1037) | 本问题（507899 driver cap 缺口） | ✅ RESOLVED 2026-06-19（driver 25.5.2 + firmware 7.8.0.7.220 + CANN beta.1） |
| [simpler#1018](https://github.com/hw-native-sys/simpler/issues/1018) | 前置：comm_init 段错（`--no-as-needed`） | 修复已验证，待上游合入 |
| [simpler#1023](https://github.com/hw-native-sys/simpler/pull/1023) | 单卡 507018 早期 zero-shape view（独立问题） | 待上游合入 |

- 上游 issue 完整记录（含 probe 输出、CANN GA trap 详解）：
  `pypto/docs/upstream-issues/step3p5-multirank-shmem-exbus.md` §Resolution
- 版本绑定 spec：[`phase16-three-pillars.md`](phase16-three-pillars.md)
- 升级 / 恢复 runbook：[`machine-recovery.md`](machine-recovery.md)
- 0234 待升级 blocker：[`../blockers.md`](../blockers.md) §5
- 完整部署 skill：`pypto/runtime/.claude/skills/ascend-phase16-deploy/SKILL.md`

---

## 各机器现状（速查）

| 机器 | driver | firmware | CANN | 多卡 e2e |
|------|--------|----------|------|----------|
| `gpu-a910x-0162` | 25.5.2 ✅ | 7.8.0.7.220 ✅ | beta.1 ✅ | ✅ 验证通过 |
| `gpu-a910x-0234` | 25.5.1 ⚠ | 7.8.0.6.201 ⚠ | beta.1 ✅ | ❌ 507899（待升 driver+firmware，CANN 别动） |
