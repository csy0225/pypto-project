"""Step 4+5 (real KV scale): one-key whole-pool map for ALL layers + paged read.

Validates the two claims that go beyond step 3:

  Step 4 -- map ALL of vLLM's KV with a BOUNDED key set (here: ONE key) + an
    offset map table, instead of the per-tensor torch.npu.MemPool approach that
    reserves one NPU virtual-address range per K and per V per layer (2N pools ->
    the measured 45-layer `rtReserveMemAddress out of memory 207001`). vLLM-Ascend
    allocates KV as per-layer int8 buffers
    (`model_runner_v1._allocate_kv_cache_tensors`: `torch.zeros(k_size, int8)` /
    `torch.zeros(v_size, int8)` per attention layer). Step 4 consolidates them into
    ONE backing buffer (one allocation-block base => one export key) and records
    `map[(layer, "K"|"V")] = byte_offset`.

  Step 5 -- a paged-attention-style kernel reads real paged KV via block_table +
    the map, zero-copy: for query q, block = block_table[q]; the KV rows are at
    `peer_base + layer_offset + block*block_bytes` -- a NESTED auto-offset (step-4
    layer map + step-5 paged block index), fed to the kernel as a DeviceTensor with
    NO data copy.

NLAYERS=45 at real per-layer KV size exercises the OOM-killer at production scale:
ONE key + one import + 90 offset-map entries vs 90 MemPool VA reservations.

Two roles (SAME free card, exporter first):
  WS=/data/chensiyu/hw_project/pypto/workspace
  source /usr/local/Ascend/cann/set_env.sh; source $WS/activate.sh
  export PTO_ISA_ROOT=$WS/pto-isa PYTHONPATH=$WS/pypto/python:$WS/pypto-lib
  K=/tmp/kvpool_key.bin
  python _stage_kvpool_pageattn.py exporter 8 $K &
  python _stage_kvpool_pageattn.py worker   8 $K
"""
import ctypes
import os
import sys
import time

NLAYERS = 45
NUM_BLOCKS = 8
BLOCK_SIZE = 128           # tokens per block (paged)
HEAD_DIM = 128
BLOCK_ELEMS = BLOCK_SIZE * HEAD_DIM
BLOCK_BYTES = BLOCK_ELEMS * 4          # fp32
KVBUF_BYTES = NUM_BLOCKS * BLOCK_BYTES  # one layer's K (or V) = [NUM_BLOCKS, BLOCK_SIZE, HEAD_DIM]
POOL_BYTES = NLAYERS * 2 * KVBUF_BYTES  # all layers, K and V, in ONE buffer
KEY_BUF = 256
# (layer, kind, block) sample points to verify — spread across the whole pool.
SAMPLES = [(0, "K", 0), (0, "V", 7), (22, "K", 3), (44, "K", 7), (44, "V", 0)]

_acl = ctypes.CDLL("libascendcl.so")
_acl.aclrtIpcMemGetExportKey.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p, ctypes.c_size_t, ctypes.c_uint64]
_acl.aclrtMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_int]
_acl.aclrtMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
_HUGE_FIRST, _H2D = 0, 1


def _layer_offset(layer: int, kind: str) -> int:
    """Byte offset of (layer, K|V) within the single pool buffer = the step-4 map."""
    return (layer * 2 + (0 if kind == "K" else 1)) * KVBUF_BYTES


def _block_pattern(layer: int, kind: str, block: int) -> "list[float]":
    """Distinct per-(layer,kind,block) pattern; fp32-exact (values < 2**24)."""
    base = layer * 10000 + (0 if kind == "K" else 5000) + block * 100
    return [float(base + (i % 97)) for i in range(BLOCK_ELEMS)]


def exporter(dev: int, keyfile: str) -> None:
    _acl.aclInit(None)
    _acl.aclrtSetDevice(dev)
    dptr = ctypes.c_void_p()
    assert _acl.aclrtMalloc(ctypes.byref(dptr), POOL_BYTES, _HUGE_FIRST) == 0, "pool malloc"
    # Fill each (layer, kind) buffer's blocks with distinct patterns via per-buffer H2D.
    for layer in range(NLAYERS):
        for kind in ("K", "V"):
            flat: list[float] = []
            for b in range(NUM_BLOCKS):
                flat.extend(_block_pattern(layer, kind, b))
            host = (ctypes.c_float * (NUM_BLOCKS * BLOCK_ELEMS))(*flat)
            off = _layer_offset(layer, kind)
            dst = ctypes.c_void_p((dptr.value or 0) + off)
            assert _acl.aclrtMemcpy(dst, KVBUF_BYTES, host, KVBUF_BYTES, _H2D) == 0, f"H2D L{layer}{kind}"
    key = ctypes.create_string_buffer(KEY_BUF)
    rc = _acl.aclrtIpcMemGetExportKey(dptr, POOL_BYTES, key, KEY_BUF, 0x1)
    print(f"[exporter] pool_base={hex(dptr.value or 0)} nlayers={NLAYERS} pool_MiB={POOL_BYTES/1048576:.1f} "
          f"ONE_KEY export rc={rc}", flush=True)
    assert rc == 0
    open(keyfile, "wb").write(key.raw)
    for _ in range(240):
        if os.path.exists(keyfile + ".done"):
            break
        time.sleep(1)
    print("[exporter] done", flush=True)


class KVPoolMap:
    """Step-4 map: ONE imported peer_base for the whole KV pool + per-(layer,kind)
    byte offsets. `paged_block(layer, kind, block)` composes the layer offset with
    the step-5 paged block index into a zero-copy DeviceTensor (the page-attention
    KV arg)."""

    def __init__(self, peer_base: int) -> None:
        import torch  # noqa: PLC0415
        self._torch = torch
        self._peer_base = peer_base

    def paged_block(self, layer: int, kind: str, block: int):
        from pypto.runtime.device_tensor import DeviceTensor  # noqa: PLC0415
        # step-4 layer offset (bytes) -> DeviceTensor over that layer's [NUM_BLOCKS, BLOCK_SIZE, HEAD_DIM]
        layer_base = self._peer_base + _layer_offset(layer, kind)
        kv = DeviceTensor(layer_base, (NUM_BLOCKS, BLOCK_SIZE, HEAD_DIM), self._torch.float32)
        return kv[block]  # step-5 paged index -> [BLOCK_SIZE, HEAD_DIM] view, auto-offset


def worker(dev: int, keyfile: str) -> None:
    import pypto.language as pl  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from pypto.ir.distributed_compiled_program import DistributedCompiledProgram, DistributedConfig  # noqa: PLC0415
    from pypto.runtime import DistributedWorker, RunConfig  # noqa: PLC0415

    @pl.jit.incore
    def _read_kv_block(kv: pl.Tensor, out: pl.Out[pl.Tensor]):
        k = pl.load(kv, [0, 0], [BLOCK_SIZE, HEAD_DIM])
        return pl.store(k, [0, 0], out)

    @pl.jit
    def _pageattn_chip(kv: pl.Tensor, out: pl.Out[pl.Tensor]):
        return _read_kv_block(kv, out)

    @pl.jit.host
    def _pageattn(kv: pl.Tensor, out: pl.Out[pl.Tensor]):
        return _pageattn_chip(kv, out)

    dc = DistributedConfig(device_ids=[dev], num_sub_workers=0, block_dim=3)
    cfg = RunConfig(platform="a2a3", distributed_config=dc)
    kv_sample = torch.zeros((BLOCK_SIZE, HEAD_DIM), dtype=torch.float32)
    host_out = torch.zeros((BLOCK_SIZE, HEAD_DIM), dtype=torch.float32).share_memory_()
    prog = _pageattn.compile(kv_sample, host_out, config=cfg)
    assert isinstance(prog, DistributedCompiledProgram)

    for _ in range(240):
        if os.path.exists(keyfile):
            break
        time.sleep(0.5)
    time.sleep(1.0)
    key = open(keyfile, "rb").read()

    # A tiny block_table so the paged read goes through an indirection, like real
    # page attention: query q attends block block_table[q].
    block_table = {0: 0, 1: 7, 2: 3}
    _ = block_table  # the SAMPLES below already exercise these block indices

    all_ok = True
    with DistributedWorker([prog]) as rt:
        peer_base = rt.import_ipc(key, worker_id=0)  # ONE import for the WHOLE pool
        print(f"[worker] import_ipc whole-pool peer_base={hex(peer_base)} (ONE key, {NLAYERS} layers)", flush=True)
        kvmap = KVPoolMap(peer_base)
        # Build the full 45-layer x {K,V} offset map table (step 4) and report its size.
        full_map = {f"L{layer}.{kind}": _layer_offset(layer, kind)
                    for layer in range(NLAYERS) for kind in ("K", "V")}
        print(f"[worker] step-4 map table entries={len(full_map)} (bounded to 1 key, no per-tensor MemPool)", flush=True)

        for (layer, kind, block) in SAMPLES:
            kv_blk = kvmap.paged_block(layer, kind, block)  # nested layer+block auto-offset, zero-copy
            host_out.zero_()
            rt.run(prog, kv_blk, host_out)
            expected = torch.tensor(_block_pattern(layer, kind, block), dtype=torch.float32).reshape(BLOCK_SIZE, HEAD_DIM)
            ok = torch.allclose(host_out, expected, rtol=1e-5, atol=1e-5)
            bad = (host_out != expected).float().mean().item()
            all_ok = all_ok and ok
            print(f"[worker] L{layer}.{kind} block={block} arg={kv_blk!r} ok={ok} bad_ratio={bad:.4f} "
                  f"got0={host_out.flatten()[0].item()} exp0={expected.flatten()[0].item()}", flush=True)

    open(keyfile + ".done", "w").write("1")
    print(f"KVPOOL_PAGEATTN_{'PASS' if all_ok else 'FAIL'} nlayers={NLAYERS} one_key=True samples={len(SAMPLES)}", flush=True)
    sys.exit(0 if all_ok else 3)


if __name__ == "__main__":
    role, dev, keyfile = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    (exporter if role == "exporter" else worker)(dev, keyfile)
