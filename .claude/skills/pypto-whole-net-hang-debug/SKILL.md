---
name: pypto-whole-net-hang-debug
description: 使用隔离日志、PTO2 stall 分类、scheduler TASK/CLUSTER 寄存器快照、task→kernel→生成源码映射、可选 AICore PC 映射、跨 rank 通信边界审计和 buffer 布局 A/B，定位 PyPTO/Ascend 整网程序的 507018、running-stalled、依赖死锁、随机 hang 或 kernel 不完成问题。适用于整模型、多层单 program、跨卡 collective、W8A8 MoE、间歇性卡死，以及需要复核地址对齐、dtype、padding、batch、初始化、buffer 生命周期和发布稳定性的场景。
---

# PyPTO 整网 Hang 定位

## 目标与纪律

把“整网卡死”收敛为可复核的证据链：

```text
固定被测对象
-> 隔离采集 host/device/dmesg/build 证据
-> 分类 stall 机制
-> task 映射到 exact kernel
-> 跨 rank 还原最早阻塞边界
-> 审计 buffer/同步/数值契约
-> 单变量 A/B
-> canonical 稳定性 + 精度准出
```

始终遵守：

1. 先检查框架设计约束，再检查局部代码。
2. 把“精度正确”和“无 stall”作为两个独立 gate；二者必须同时通过。
3. `507018` 只是 host 侧通用失败，不能单独证明 deadlock、OOM 或 kernel bug。
4. 不把 `stuck_task_id` 当 kernel `func_id`。
5. 没有真实 AICore PC 就只定位到 kernel/阶段，禁止编造指令级结论。
6. 诊断旋钮、重试、增大 timeout、缩层数只能分类，不能作为发布修复。
7. 保持 native W8A8；禁止用 BF16-dequant 权重回退掩盖问题。
8. 用“已证实 / 强证据 / 候选 / 已排除”标注结论强度。

写排障结论时不要只写“现象 → 修复”两句话。至少交代：

```text
前置对象和框架约束
-> 观测到的现象及失败阶段
-> 为什么当时会提出这个假设
-> 决定性实验和实际结果
-> 结论的证据等级
-> 哪些修改保留、哪些只用于诊断、哪些结论被撤回
-> 这个结论能覆盖哪些对象，不能覆盖哪些对象
```

如果同一个错误码在多个阶段出现，按时间和测试对象拆开写；不要为了简短把
多个 blocker 压成一个“根因”。长时间、多 session 的实践案例见
`references/n1-case-study.md` 的时间线和 blocker 演化矩阵。

面向新读者时，正文首次出现 `P1/P20/P42/K2` 等历史简写必须写出完整变量、
MoE 层数和中文含义；日志原文可保留缩写，但不能让其与 task、kernel、rank、
batch 或优先级混淆。项目术语的通俗解释见案例文档 §1.2。

处理 N1 类问题时，同时读取：

- `../pypto-dev-constraints/SKILL.md`；
- 项目唯一 canonical 测试文档；
- `/data/chensiyu/hw_project/pypto/pypto_top_level_documents/` 中 shape、layout、
  sharding、runtime、memory 文档；目录不存在时先报告环境缺口，不要用局部源码
  代替顶层设计约束。

详细实践见 [references/n1-case-study.md](references/n1-case-study.md)。
处理持续多个 session、同一错误码反复出现的 N1 类事件时，先读该案例的
§2～§4：不要只读最终 512B 结果；先按时间区分环境、dispatch、alias、OOM、
deterministic kernel bug、精度 bug 和 probabilistic stall，再复用后半部分的
task/kernel、跨 rank、buffer 和 release 方法。

## 0. 先冻结被测对象

在第一次复现前记录：

```text
program / branch / commit / dirty diff / generator hash
machine / devices / driver / firmware / CANN / PTOAS / runtime
模型层数 / dispatch+combine 组合 / native W8A8 权重来源 / KV 来源
真实输入 / batch / padding 语义 / golden / 精度判据
ring/task-window/dep-pool/timeout 环境变量
run id / source hash / exact build directory
```

禁止把以下对象混成同一基线：

- 随机输入与真实 token；
- P1/P20 与完整 P42；
- compile-only 与 device run；
- `RUN_CLEAN` 与数值 golden；
- clean 一次与稳定通过；
- push+push、pull+push、pull+pull；
- native INT8 权重与 BF16-dequant 权重。

历史偶发 clean 只能证明数学路径“有机会正确”，不能证明稳定基线。

## 1. 先画边界，再运行

为每层建立边界图，不从某一条报错直接钻进局部 kernel：

```text
producer
-> payload 写入/发布
-> fence
-> notify
-> peer wait
-> local/remote load
-> consumer
```

为每个中间对象建立 ledger：

| 字段 | 必填内容 |
|---|---|
| layer/phase | 唯一层后缀和所属边界 |
| buffer | signal、payload、scratch、output |
| logical | shape、valid_shape、dtype |
| physical | nbytes、base/relative offset、alignment |
| ownership | local、peer-readable、writer ranks |
| lifetime | first producer、last consumer、可回收点 |
| init | runtime whole-window zero、显式 memset、未定义 |
| protocol | Set/AtomicAdd、Eq/Ge、expected generation |
| alias | 与前后 buffer 是否共线或复用 |

先找最早失去 forward progress 的边界，而不是最后打印错误的函数。

## 2. 为每次运行创建隔离证据包

使用独立 run 目录：

```bash
RUN=/path/to/logs/<case>-$(date +%Y%m%d-%H%M%S)
mkdir -p "$RUN/ascend"
export ASCEND_PROCESS_LOG_PATH="$RUN/ascend"

dmesg -T > "$RUN/dmesg.before.txt"
# 运行唯一 canonical 命令；保存 stdout/stderr、rc、build_dir、source hash
dmesg -T > "$RUN/dmesg.after.txt"
diff -u "$RUN/dmesg.before.txt" "$RUN/dmesg.after.txt" \
  > "$RUN/dmesg.diff.txt" || true
```

要求：

- `ASCEND_PROCESS_LOG_PATH` 目录必须预先创建；
- 不从共享 `~/ascend/log/debug` 猜 PID/时间；
- 若机器提供 `task-submit`，整个压力循环只申请一次设备锁；
- 保存所有 rank，而不只看 rank0；
- 保存失败 build，至少保留 `kernel_config.py`、orchestration C++、kernel C++、
  task dependency dump、memory report 和 perf hints；
- 记录失败发生在 exporter/compile/prepare/import/run 哪个阶段；
- exporter 或 checkpoint 在 `rt.run` 前失败，不属于 kernel stall。

可先运行解析器：

```bash
python .claude/skills/pypto-whole-net-hang-debug/scripts/analyze_stall.py \
  --log "$RUN/host.log" \
  --log "$RUN/ascend" \
  --build-dir "$BUILD_DIR" \
  --orchestration full_moe_chip_orch \
  --dmesg-before "$RUN/dmesg.before.txt" \
  --dmesg-after "$RUN/dmesg.after.txt"
```

解析器只汇总证据和候选映射，不自动宣布根因。它按输入文件和
`[STALL thread=... idle_iterations=...]` 快照分组；不同 run、rank 或日志
文件的 timeout/TASK/kernel 不会自动拼接。没有同轮 RUNNING TASK/CORE 时，
不得用另一份历史日志补齐映射；WAIT/READY kernel 只作为 dependency context。

## 3. 先分类机制

### 3.1 读取错误码和计数

优先读取：

```text
orch_error_code
sched_error_code
sub_class
completed/total
running / ready / waiting / orch_done
stuck_task_id / stuck_core
TASK / SUMMARY / CLUSTER
```

常见 orchestrator code：

| code | 含义 |
|---:|---|
| 1 | scope deadlock |
| 2 | heap-ring deadlock |
| 3 | flow-control deadlock |
| 4 | dep-pool overflow |
| 8 | tensor-wait timeout |
| 10 | scope-tasks overflow |
| 11 | tensormap overflow |
| 100 | scheduler timeout |

不要用 ring/dep-pool 调参替代分类；只有对应 detector/code 出现时，容量方向才有直接证据。

### 3.2 解释 PTO2 stall subclass

| subclass | 快照 | 第一调查方向 |
|---|---|---|
| S1 running-stalled | 无进展快照中存在至少一个已分配到 core 的 RUNNING task | 该 task 及其前一跨层边界：kernel 内 wait、异常、DMA/同步、越界 |
| S3 ready-but-all-idle | READY 存在但所有 core idle | scheduler/resource/dispatch |
| S4 dependency-deadlock | 只有 WAIT，fanin 永不满足 | DAG、missing dependency、cycle、wiring |
| S5 orchestrator-starvation | 已提交任务结束但 orch 未结束 | orchestrator 提交/终止逻辑 |

其他签名：

- `Task Allocator Deadlock` / structural head-of-line：容量或 scope 回收；
- `Timeout (N cycles): producer/consumers`：具体 spin wait；
- `HandleTaskTimeout`：OS op timeout，不能自动叫 deadlock；
- 抬高 timeout 后仍耗完整预算不完成：支持“真 stall 而非单纯慢”，但不是修复。

## 4. 读 TASK、CLUSTER 和寄存器

`TASK` 行回答：

```text
哪个 ring/task
RUNNING / READY / WAIT
fanin 当前值/目标值
该 task 包含哪些 aic/aiv kernel id
RUNNING 在哪个 core/thread
```

`CLUSTER` 行回答：

```text
core idle/busy
kernel id
task id
COND register 状态
```

解释 `COND`：

- `ack`：采样瞬间硬件仍报告 ACK；只证明当时尚未观察到完成，不证明持续
  forward progress，也不区分计算、spin wait 或 DMA wait；
- `fin`：硬件报告完成，但软件 slot 仍 busy；
- `fin ANOMALY`：检查 completion polling/recycle bookkeeping，同时考虑诊断
  采样与正常回收并发；不要直接写成产品 race 根因。

`cond_tok/running_tok/pending_tok` 用于核对 completion bookkeeping，不是 PC。
普通 stall 快照的 `COND` 也不是 AICore program counter。

## 5. 正确完成 task → kernel → 源码映射

raw task id 编码：

```text
ring_id  = raw >> 32
local_id = raw & 0xffffffff
```

例如：

```text
4294967319 = 0x100000017 = ring 1 / local task 23
```

这不代表 `func_id=23`。

正确链条：

1. 从同一轮 device `TASK` 行取 `kernels=[aic:X aiv0:Y aiv1:Z]`；
2. 使用该次 exact build 的
   `next_levels/<orch>/kernel_config.py`；
3. 将 `func_id=Y` 映射为 kernel 名和生成源文件；
4. 在 `orchestration/<orch>.cpp` 中定位该 task 的提交和前后依赖；
5. 对照 dependency dump，确认 producer/consumer；
6. 若多个 orchestration 的同一 func id 名称不同，保留候选，不能任选一个。

禁止：

- 按 `completed=78/81` 猜“接近末尾所以是 combine”；
- 用另一轮 build 的 `kernel_config.py`；
- build 已删除后凭记忆恢复映射；
- 把 downstream WAIT task 当成实际 RUNNING hang 点。

## 6. PC 指针是二级证据

普通 PTO2 stall 快照通常只读 `COND`，不自动读取 AICore PC。按以下分支处理。

### 有真实 AICore PC

1. 保存报告 `error_pc/current_pc/start_pc` 的原始 plog/exception；
2. 保存同一轮加载的 `.so`、kernel binary 和 debug map；
3. 仅当地址来自同一 loaded image 时计算：

   ```text
   pc_offset = error_pc - kernel_start_pc
   ```

4. 使用 CCE/PTOAS 对应的 map/disassembler 映射 offset；
5. 同时检查 PC 指令和它前一个配对操作：
   - wait 前的 producer notify；
   - load 前的 publish/fence；
   - notify 前的 data movement；
6. 记录工具版本和 binary hash。

不要假设 GNU `objdump` 能解 AICore binary，也不要把 host CPU stack PC 当 AICore PC。

### 没有真实 AICore PC

先定位到 exact kernel，再：

- 在 kernel 内加入有界 phase marker；
- 二分 kernel 阶段；
- 拆出最小 faithful probe；
- dump signal generation、offset、count、producer completion；
- 保持协议和数据形状不变。

结论写为“定位到 kernel/边界”，不能写“定位到某条指令”。

## 7. 跨 rank 反推最早阻塞边界

同时比较所有 rank：

```text
rank A: RUNNING producer/dispatch
rank B: RUNNING pull/combine
rank C: WAIT downstream
```

不同 rank 停在不同深度并不等于多个独立根因。执行：

1. 按层和通信 generation 排序各 rank 快照；
2. 找最早未完成的跨 rank 协议；
3. 检查该协议的最后一个成功配对：
   - payload 是否写完；
   - fence 是否覆盖正确 pipe；
   - notify 是否每 peer/每 generation 一次；
   - wait expected 是否与初始化和复用一致；
   - self 路径是否错误地走 remote；
   - peer offset/count 是否来自同一 snapshot；
4. 检查下游 RUNNING 是否可能只是等待前一边界数据；
5. 对跨 run 漂移做统计，不把一次采样写成固定层根因。

## 8. 高优先级 buffer 与布局审计

### 8.1 存储与 tile 对齐

必须分别检查四类规则：

```text
tensor storage:
prod(shape) * sizeof(dtype) % 512 == 0
真实逻辑范围放 valid_shape

UB Vec tile 每行:
cols * sizeof(dtype) % 32 == 0
FP32/INT32: cols % 8 == 0
BF16/FP16: cols % 16 == 0
INT8: cols % 32 == 0

GM <-> UB tile:
按 512B DMA 约束审计

L2 cache line:
512B；BF16 trailing 256、FP32 128、INT8 512 elements
```

`[N,1] FP32` slice 可能绕过静态检查并在运行时报 `507018`。

### 8.2 control signal 隔离

逻辑 signal 可为：

```text
[8,1] INT32 = 32B
```

物理 allocation 不应自动等于逻辑字节。审计：

```text
physical_nbytes >= 512
(actual_window_base + relative_offset) % 512 == 0
signal 与 payload 不共 512B line
不同协议的 atomic signal 不共 line
whole window size 满足 allocator 对齐
```

仅扩大到 512B 但 base 未对齐，仍可能跨两条 cache line。相对 offset 和
window size 对齐不能单独证明实际地址对齐；记录 allocator 的 base 保证，
或逐 rank 审计实际 `base` 与 `(base+offset)%512`。

N1 当前保留 512B physical signal isolation，但这是该平台/allocator 条件下
的结构性选择，不是跨平台定律，也不是跨机器充分条件。

### 8.3 生命周期、初始化和复用

- RAW-only v1 依赖 non-aliasing intermediate memory；
- 为重叠层分配 distinct signal/payload window；
- 不复用仍有远端 writer/reader 的通信窗口；
- 首次 notify/wait 前清零 signal；
- 按语义清零 routed/gather destination；
- 区分 runtime whole-window memset 和 logical tensor 初始化；
- 旧 signal 可能使 `Ge` 提前通过，也可能让 peer 永久等待。

### 8.4 dtype、padding 和 batch

- logical shape、physical bytes、dtype reinterpret 必须一致；
- native W8A8 保持 INT8 weights + FP32 scales；
- routed input 和 clamp 后中间激活均使用 per-token INT8 dynamic quant/requant；
- shared expert 保持 BF16，不套 routed quant；
- `router_bias` 做 BF16 round，EPS=`1e-5`，采用 layer-specific swiglu clamp/limit；
- golden 必须为 W8A8-specific，不复用 BF16 golden；
- 从 exact generated/runtime argument dump 验证实际 dtype、scale 和 requant
  路径，不能只看 Python 声明；
- signed tail 先判断再 cast 为 unsigned/index；
- 覆盖单有效 row + padding rows、单 batch、多 batch；
- 明确 padding 行是否初始化、是否进入 reduction/softmax/route；
- `seq_len=0` 或空 tile 必须有定义行为，不能读取未初始化 scratch。

### 8.5 片上容量

以生成报告为准：

```text
UB physical 192KB；常用编译预算 184KB
L1 512KB
L0A 64KB
L0B 64KB
L0C 128KB
```

读取 `report/memory_after_AllocateMemoryAddr.txt` 的 used/limit、
buffer address range 和 live range；不要只凭源码 shape 估算。

## 9. 设计最小 A/B，而不是堆补丁

每个实验只改一个候选变量，并记录：

```text
hypothesis
unchanged invariants
changed variable
predicted signature
actual signature
evidence strength
next decision
```

推荐顺序：

1. exact build + 日志增强，不改语义；
2. 复核 framework invariant；
3. 固定通信边界，采集 signal/count/offset；
4. 只改物理布局或只改同步 generation；
5. 最小 faithful probe；
6. 完整 canonical 复验。

P1/P20 可用于缩短诊断，但不能代替 P42 release。若最终问题只在完整深度暴露，
直接使用完整深度进行决定性 A/B，避免被中间态误导。

## 10. 准出和文档记录

发布前同时满足：

- exact final source；
- generator 真实 strip → regenerate → byte-compare；
- compile/static checks；
- canonical 完整层数；
- 连续稳定性次数达到测试文档要求；
- 每次数值 golden 正确且指纹稳定；
- fresh exporter/进程环境；
- dmesg before/after 无新增相关 fault；
- 最终 commit、branch、日志路径、build/source hash 可追溯。

结论模板：

```text
已证实：
- [直接日志/寄存器/映射/测试支持]

强证据：
- [单变量 A/B + 稳定性数据]

尚未证明：
- [没有 PC、没有 bit-level trace、没有唯一性证明]

已排除：
- [对应实验和证据]

保留的结构性修复：
- [因架构/精度/生成器一致性保留，但不冒充 stall 唯一根因]
```

跨机器发布时增加 scope：

```text
machine scope:
  0162: release-qualified / evidence path
  0234: active or not independently verified
```

不得把一台机器的 20/20 直接写成所有机器的 standalone resolved。

完成后更新 active handoff、canonical 文档和项目状态；删除已失效的 active prompt，
历史错误结论只在案例复盘中以“已纠正”方式保留。
