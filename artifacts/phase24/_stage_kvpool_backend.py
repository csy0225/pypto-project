"""Phase 24.1 —— vLLM KV-pool 一 key 整池映射 patch（取代 per-tensor MemPool）。

隔离验证 Phase 24 最有风险的一步：**vLLM 能否在「所有 attention 层的 K/V 合并进
一个 backing buffer（一个 export key + offset map）」下正常 boot + serve**，同时
证明它**不再撞 45 层 per-tensor MemPool 的 207001 OOM**。

只 patch `vllm_ascend.worker.model_runner_v1.NPUModelRunner._allocate_kv_cache_tensors`：
  orig() 返回 {layer_name: (k_int8, v_int8, ...)} 后，把目标层的所有 K/V 挪进
  ONE `torch.npu.MemPool` 里的 ONE `torch.zeros(total, int8)`（块基址 → 可导出），
  每层 K/V 换成该 buffer 的连续 view，记录 `map[(layer,K|V)]=byte_off`，导出 ONE
  key，把 key + map 写到 /logs 供 worker（Phase 24.2）import。

**不**替换 attention forward（那是 24.2 + 现有 attn backend）——所以 vanilla
attention 照常在这些 view 上跑，8001 正常出 token。这样 24.1 只验证一件事：
consolidated-buffer KV + 一 key + map 下 vLLM 正常 + 无 OOM。

Env:
  PYPTO_KVPOOL=1                     # 启用本 patch
  PYPTO_KVPOOL_LAYERS=               # 逗号列表；空=所有 attention 层
  PYPTO_KVPOOL_DIR=/logs             # key/map 落点
  PYPTO_KVPOOL_KEY=pypto_kvpool.key  # 单 key 文件名
  PYPTO_KVPOOL_MAP=pypto_kvpool_map.json
"""
from __future__ import annotations

import ctypes
import json
import os
import re

import torch

KEY_BUF = 256
_LAYER_IDX_RE = re.compile(r"\.layers\.(\d+)\.")
_POOLS: list = []   # pin MemPool + buffer for serving lifetime
_BIG: list = []
_PATCHED = False


def _dir() -> str:
    return os.environ.get("PYPTO_KVPOOL_DIR", "/logs")


def _layers() -> "frozenset[int] | None":
    raw = os.environ.get("PYPTO_KVPOOL_LAYERS", "")
    if not raw.strip():
        return None  # all attention layers
    return frozenset(int(x) for x in raw.split(",") if x.strip())


def _layer_idx(name: str) -> "int | None":
    m = _LAYER_IDX_RE.search(name)
    return int(m.group(1)) if m else None


def _export_key(dptr: int, nbytes: int) -> bytes:
    acl = ctypes.CDLL("libascendcl.so")
    acl.aclrtIpcMemGetExportKey.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p, ctypes.c_size_t, ctypes.c_uint64]
    key = ctypes.create_string_buffer(KEY_BUF)
    rc = acl.aclrtIpcMemGetExportKey(ctypes.c_void_p(dptr), ctypes.c_size_t(nbytes), key, KEY_BUF, ctypes.c_uint64(0x1))
    if rc != 0:
        raise RuntimeError(f"aclrtIpcMemGetExportKey rc={rc} dptr={hex(dptr)} nbytes={nbytes}")
    return key.raw[:KEY_BUF]


def install() -> "dict":
    global _PATCHED  # noqa: PLW0603
    if _PATCHED:
        return {"ok": True, "already": True}
    import vllm_ascend.worker.model_runner_v1 as mr  # noqa: PLC0415
    from vllm.distributed import get_tensor_model_parallel_rank  # noqa: PLC0415

    runner_cls = mr.NPUModelRunner
    orig = runner_cls._allocate_kv_cache_tensors
    want = _layers()

    def patched(self, kv_cache_config):
        raw = orig(self, kv_cache_config)
        targets = []  # (layer_name, idx, k, v, rest)
        total = 0
        for layer_name, kv in list(raw.items()):
            idx = _layer_idx(layer_name)
            if idx is None or (want is not None and idx not in want):
                continue
            if (isinstance(kv, (tuple, list)) and len(kv) >= 2
                    and torch.is_tensor(kv[0]) and torch.is_tensor(kv[1])):
                k, v = kv[0], kv[1]
                kb = k.numel() * k.element_size()
                vb = v.numel() * v.element_size()
                targets.append((layer_name, idx, k, v, tuple(kv[2:])))
                total += kb + vb
        if not targets:
            return raw

        # ONE MemPool + ONE buffer for ALL target-layer K/V (1 VA reservation).
        pool = torch.npu.MemPool()
        with torch.npu.use_mem_pool(pool):
            big = torch.zeros(total, dtype=torch.int8, device=self.device)
        _POOLS.append(pool)
        _BIG.append(big)
        base = int(big.data_ptr())

        off = 0
        kvmap: "dict[str, dict]" = {}
        for (layer_name, idx, k, v, rest) in targets:
            kb = k.numel() * k.element_size()
            vb = v.numel() * v.element_size()
            k_view = big[off:off + kb].view(k.dtype).view(k.shape)
            kvmap[f"L{idx}.K"] = {"offset": off, "nbytes": kb, "shape": list(k.shape)}
            off += kb
            v_view = big[off:off + vb].view(v.dtype).view(v.shape)
            kvmap[f"L{idx}.V"] = {"offset": off, "nbytes": vb, "shape": list(v.shape)}
            off += vb
            raw[layer_name] = (k_view, v_view, *rest)
        # originals now unreferenced -> reclaim so peak stays ~1x KV, not 2x.
        torch.npu.empty_cache()

        key = _export_key(base, total)
        rank = get_tensor_model_parallel_rank()
        d = _dir()
        keyname = os.environ.get("PYPTO_KVPOOL_KEY", "pypto_kvpool.key")
        mapname = os.environ.get("PYPTO_KVPOOL_MAP", "pypto_kvpool_map.json")
        with open(os.path.join(d, f"{keyname}.rank{rank}"), "wb") as f:
            f.write(key)
        with open(os.path.join(d, f"{mapname}.rank{rank}"), "w") as f:
            json.dump({"rank": rank, "pool_base": base, "pool_bytes": total,
                       "num_layers": len(targets), "map": kvmap}, f)
        print(f"[pypto-kvpool] rank{rank} consolidated {len(targets)} layers -> ONE buffer "
              f"base={hex(base)} bytes={total} ({total/1048576:.1f} MiB) ONE key exported; "
              f"map entries={len(kvmap)}", flush=True)
        return raw

    runner_cls._allocate_kv_cache_tensors = patched
    _PATCHED = True
    return {"ok": True, "patched": True}


def maybe_autoload() -> "dict":
    """sitecustomize entry: install iff PYPTO_KVPOOL=1."""
    if os.environ.get("PYPTO_KVPOOL", "") == "1":
        return install()
    return {"ok": True, "skipped": "PYPTO_KVPOOL != 1"}


def status() -> "dict":
    return {"installed": _PATCHED, "layers": sorted(_layers()) if _layers() else "all"}

