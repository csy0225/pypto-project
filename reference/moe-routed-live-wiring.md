# MoE routed-expert live wiring 集成 spec（源码落地版）

> 目标：把已验证的 `pypto-lib/models/step3p5/vllm_routed_experts.py`（RoutedExperts
> per-rank grouped-GEMM，真 W8A8 bad_ratio=0.0000）接进 live 8001，替换 MoE 层
> (3-44) 的 routed-expert 计算，再对 8000 vanilla oracle 做 live A/B。
>
> 本文档基于 **running 容器 `vllm-8001`（镜像 stepcast:0.19.0）内的真实
> vllm_ascend 源码**（2026-07-05 读取），不是猜测。所有行号针对该镜像。

---

## 1. 精确 hook 点（已定位）

`vllm_ascend/ops/fused_moe/moe_comm_method.py`：

```
class MoECommMethod(ABC):                       # :82  基类，所有 comm impl 继承
    def fused_experts(self, fused_experts_input):   # :117
        ... prepare → token_dispatch → build_mlp_compute_input ...
        mlp_output = self._apply_mlp(mlp_compute_input)   # :144  ← 唯一 per-rank 计算
        routed_out = self.token_dispatcher.token_combine(mlp_output, ...)  # :147
        return FusedExpertsResult(routed_out=..., ...)

    def _apply_mlp(self, mlp_compute_input) -> torch.Tensor:   # :160  ← HOOK 在这里
        return unified_apply_mlp(mlp_compute_input=mlp_compute_input)
```

**为什么这是正确的 seam**：`_apply_mlp` 在 token_dispatch **之后**、token_combine
**之前**——输入 tokens 已按 local expert 排序聚合（all-to-all dispatch 完成），
输出直接喂 combine。这正是 `RoutedExperts` 内核覆盖的范围（per-rank，无 collective；
dispatch/combine 仍由 vLLM 做）。且 `_apply_mlp` 在**基类**上定义（AllGather / MC2 /
AlltoAll / FusedMC2 全部继承同一个），所以一个 monkey-patch 覆盖所有 comm 模式，
与 8001 具体用哪种 comm 无关。

**不要 hook `unified_apply_mlp` 或 `quant_apply_mlp`**（moe_mlp.py）——那是自由函数、
签名复杂、且 W8A8/MXFP 分支多；hook 基类方法 `_apply_mlp` 语义最干净。

---

## 2. 输入/输出契约（`MoEMlpComputeInput`，moe_stage_contracts.py:128）

hook 收到的 `mlp_compute_input` 字段（已核对源码）：

| 字段 | 类型 | 含义（对接 RoutedExperts） |
|------|------|---------------------------|
| `hidden_states` | Tensor `[num_recv, HIDDEN]` **BF16** | 本 rank dispatch 后收到的 tokens（按 local expert 排序、连续）→ `local_routed_x` |
| `group_list` | Tensor `[n_local_experts]` | `group_list_type==1` 时是**每个 local expert 的 token 数** → `local_expert_count` |
| `group_list_type` | int | `1`=counts，`0`=cumsum。cumsum→count 用 `torch.diff`（见 moe_mlp.py cumsum_group_list） |
| `dynamic_scale` | Tensor\|None | **入口通常 None**（BF16 hidden 尚未 per-token quant）→ 正是我们要拦截的点 |
| `weights.w1/w2` | Tensor/list | vanilla 的 W8A8 int8 专家（**我们不用**，用自己 dequant-BF16） |
| `weights.w1_scale/w1_offset/w1_scale_bias` | Tensor | W8A8 反量化元数据（**我们不用**，worker 侧已 dequant） |
| `quant.is_quant` | bool | W8A8 时 True |
| `activation` | str | `"silu"`（step3p5 routed 是 SiLU；RoutedExperts 已是 SiLU） |

返回：`torch.Tensor [num_recv, HIDDEN]`（BF16），喂给 `token_combine`。

**关键精度决策**：vanilla W8A8 路径（`quant_apply_mlp` moe_mlp.py:84）会先
`npu_dynamic_quant(hidden BF16 → int8 + per-token scale)`，再 int8 grouped-matmul +
`npu_dequant_swiglu_quant`。我们在 **BF16 hidden 入口拦截**，用 **dequant-BF16 专家**
做 BF16 grouped-GEMM ——这是 W8A8 的**参考精度路径**，正是 `vllm_routed_experts.py`
离线 golden（torch dequant-W8A8 ref）对齐到 **bad_ratio=0.0000** 的那条路径。所以
数值上是 W8A8-dequant reference，干净、无需复刻 int8 kernel。（activation 量化误差被
去掉 = 更接近 dequant reference，A/B 对 8000 vanilla 可能有 <1e-2 量级差异，属预期；
若要 bit-exact 需在 worker 内复刻 per-token act-quant，是后续可选项。）

---

## 3. RoutedExperts 侧契约（`vllm_routed_experts.py`，已验证）

```
RoutedExperts(@pl.program)  inputs:
  local_routed_x      [LOCAL_RECV_MAX=1024, HIDDEN=4096] BF16
  local_expert_offset [N_LOCAL_EXPERTS=36] INT32   # 每个 expert 在 x 中的起始行 = counts 的 exclusive cumsum
  local_expert_count  [N_LOCAL_EXPERTS=36] INT32   # = group_list（group_list_type==1）
  w_gate/w_up         [36, HIDDEN, INTER=1280] BF16 (dequant-W8A8, HF gate/up 已转置 [INTER,HIDDEN]→[HIDDEN,INTER])
  w_down              [36, INTER, HIDDEN] BF16 (dequant, [HIDDEN,INTER]→[INTER,HIDDEN])
  output local_routed_y [1024, HIDDEN] BF16
```

- **rank r 拥有 global experts `[r*36 .. r*36+36)`**；dispatch 后本 rank 收到的 tokens
  恰好是这 36 个 local expert 的（vLLM EP 保证）。所以 `group_list` 的 36 项直接就是
  local_expert_count，顺序一致。
- **`LOCAL_RECV_MAX=1024` 上限**：hook 必须 `assert num_recv <= 1024`（decode batch 小，
  通常成立）。若 prefill/大 batch 超限 → 需按 1024 分块多次调用（后续项）。
- 权重是 **per-layer** 的：worker 按 `layer_idx` 加载对应层的 dequant-W8A8 专家
  （`_real_weights(ckpt, layer, rank)` 已实现）。

---

## 4. 落地步骤（下个 session 实施）

### 4.1 worker `routed` op（已实现，需 device round-trip 验证）
`vllm_routed_experts.py::_serve(sock, device, ckpt, layer, rank)` 已就绪：UDS，
4-byte len + JSON header(`{op:"routed", num_recv, offsets, counts, layer}`) + BF16 body。
- **待做**：腾空一张卡（8001 下线 cards 8-15）跑一次真实 device round-trip
  （client 发 BF16 hidden + offsets/counts → 收 y → 比对 `main()` 的 golden）。
  内核本身已 device-PASS，这步只验 socket 往返。**不要在有 live worker 的卡上跑**（co-tenancy 507018）。
- **多层支持**：worker 需能按请求里的 `layer` 切换/缓存该层 experts（预编译一份 RoutedExperts
  program，权重按 layer 换）。42 个 MoE 层 × 36 experts × [4096,1280] BF16 ≈ 每层 ~450MB×3；
  单卡放不下全 42 层 → worker 按需 load + LRU，或每 rank 常驻若干层。（内存策略是后续项。）

### 4.2 backend monkey-patch `_apply_mlp`（核心，多周）
在 8001 的 `/logs/pypto_patch/` 加一个 backend patch（沿用 `sitecustomize.py` autoload
机制，与现有 `pypto_attn_backend.py` 平行）：

```python
# 伪代码 —— 落在 pypto_moe_backend.py
import vllm_ascend.ops.fused_moe.moe_comm_method as mcm
_orig = mcm.MoECommMethod._apply_mlp

def _pypto_apply_mlp(self, mlp_in):
    if not _PYPTO_MOE_ENABLED or 当前层 not in MOE_LAYERS:   # 层号如何拿到见下
        return _orig(self, mlp_in)
    x = mlp_in.hidden_states            # [num_recv, HIDDEN] BF16
    gl = mlp_in.group_list
    counts = gl if mlp_in.group_list_type == 1 else torch.diff(cat([gl[:1], gl]))
    offsets = exclusive_cumsum(counts)  # int32
    assert x.shape[0] <= 1024
    y = _pypto_routed_client(x, offsets, counts, layer_idx, rank)  # UDS 到 worker
    return y                            # [num_recv, HIDDEN] BF16 → token_combine
mcm.MoECommMethod._apply_mlp = _pypto_apply_mlp
```

**未解难点（这是"多周"的真实原因）**：
1. **层号**：`_apply_mlp` 的入参不带 layer_idx。需要一条把「当前正在算第几层」传进来的路
   （eager forward 时 layer module 顺序执行；可在 `Step3p5MoE.forward` / `FusedMoE.forward`
   外层用 threadlocal / 计数器注入 layer_idx，或 patch 更外层的 `FusedMoE.forward` 拿到 self.layer_idx）。
2. **token 布局对齐**：确认 dispatch 后 `hidden_states` 行序与 `group_list` 累加边界严格一致
   （dump 一次真实 tensor 核对：`hidden_states[offset[e]:offset[e]+count[e]]` 应全属 expert e）。
3. **shared expert**：step3p5 MoE 层还有 shared expert（`vllm_shared_mlp.py` 已验证）。routed
   与 shared 在 vLLM 里的合并点（`SharedFusedMoE`）要一并接管或让 shared 走已验证的 `share_expert`
   patch，避免双算。
4. **性能**：per-layer socket round-trip（现 attn ~2.5s/token）叠加 MoE 会更慢，属 Phase 26。

### 4.3 live A/B
8001 加 `PYPTO_MOE=1` + `MOE_LAYERS=3,..,44` → 起服务 → 对 8000 vanilla 同 prompt
（temp=0，curl `-m300`）逐 token 比对。先接 1 层（layer 3）验证正确，再全 42 层。

---

## 5. 验证链（每步必须 PASS 才进下一步）

1. worker `routed` op device round-trip（空闲卡）：client golden bad_ratio ≈ 0。
2. 离线 `_apply_mlp` 拦截单测：抓一次真实 `MoEMlpComputeInput`（8001 dump），
   `_pypto_apply_mlp` vs `_orig` 输出对比（BF16-dequant ref 差异 <1e-2）。
3. 单层 live（只接 layer 3）：8001 A/B vs 8000，输出连贯 + token 基本对齐。
4. 全 42 MoE 层 live A/B。

---

## 6. 边界 / 现状（2026-07-05）

- **已完成**：RoutedExperts 内核真 W8A8 bad_ratio=0.0000（`pypto-lib fc0bafb`）；worker
  `routed` op 实现（host-verified）；hook seam + 契约**已从 running 源码核对**（本文档）。
- **未完成**：4.1 device round-trip、4.2 backend patch（含层号注入 + shared 合并）、4.3 A/B。
- **机器**：16 卡 OK；8000 oracle + 8001 pypto(dense0-2+attn+tail) 均 live。MoE @pl.program
  device 测试**必须先腾空卡**（co-tenancy → 507018 → card Alarm → root reset 级联，见
  `archive/milestones-2026-Q2.md` 2026-07-04/05）。
