"""Step 3 (synthetic): VA-map + auto-offset at kernel launch, multi-block.

Extends P7 (`_stage_p7_import_ipc_validate.py`) from a single IPC buffer to a
MULTI-BLOCK pool addressed through a VA-map — the exact mechanism step 4/5 need
for a real vLLM paged-KV pool (one export key for the whole pool, per-block
access by offset).

Validated fact this rests on (measured on 0162 by `_stage_va_ipc_probe.py`):
cross-process import returns a DIFFERENT VA but offset is preserved, so a block
at pool_base+off imports to peer_base+off. The VA-map is therefore a pure base
map: `pypto_block_ptr = peer_base + block * block_bytes`, realized here via
`DeviceTensor(peer_base, pool_shape, dtype)[block]` (DeviceTensor.__getitem__).

Two roles (SAME free card, exporter first):
  exporter: aclrtMalloc ONE contiguous pool [NUM_BLOCKS, TILE, TILE] fp32; fill
            block b with a DISTINCT pattern flat[i] = b*1000 + (i % 97); export
            ONE key for the pool base; wait for <keyfile>.done.
  worker:   compile the trivial read kernel; DistributedWorker([decode]) forks
            the chip child; import_ipc(key) -> peer_base; register it in a VAMap;
            for several DIFFERENT block indices, VAMap.block(b) auto-offsets to a
            per-block DeviceTensor and the kernel reads it -> must equal block b's
            pattern. ALL blocks correct => VA-map + auto-offset works multi-block.

Run (maintenance window, 8001 down, free card N):
  WS=/data/chensiyu/hw_project/pypto/workspace
  source /usr/local/Ascend/cann/set_env.sh; source $WS/activate.sh
  export PTO_ISA_ROOT=$WS/pto-isa PYTHONPATH=$WS/pypto/python:$WS/pypto-lib
  K=/tmp/vamap_key.bin
  python _stage_vamap_multiblock.py exporter 8 $K &
  python _stage_vamap_multiblock.py worker   8 $K
"""
import ctypes
import os
import sys
import time

TILE = 128
NUM_BLOCKS = 8
BLOCK_ELEMS = TILE * TILE
BLOCK_BYTES = BLOCK_ELEMS * 4  # fp32
POOL_BYTES = NUM_BLOCKS * BLOCK_BYTES
KEY_BUF = 256
TEST_BLOCKS = (0, 1, 3, 7)  # first, second, middle, last — distinct offsets

_acl = ctypes.CDLL("libascendcl.so")
_acl.aclrtIpcMemGetExportKey.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p, ctypes.c_size_t, ctypes.c_uint64]
_acl.aclrtMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_int]
_acl.aclrtMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
_HUGE_FIRST, _H2D = 0, 1


def _block_pattern(b: int) -> "list[float]":
    """Distinct per-block pattern so a wrong offset reads visibly wrong data."""
    return [float(b * 1000 + (i % 97)) for i in range(BLOCK_ELEMS)]


def exporter(dev: int, keyfile: str) -> None:
    _acl.aclInit(None)
    _acl.aclrtSetDevice(dev)
    dptr = ctypes.c_void_p()
    assert _acl.aclrtMalloc(ctypes.byref(dptr), POOL_BYTES, _HUGE_FIRST) == 0
    flat: list[float] = []
    for b in range(NUM_BLOCKS):
        flat.extend(_block_pattern(b))
    host = (ctypes.c_float * (NUM_BLOCKS * BLOCK_ELEMS))(*flat)
    assert _acl.aclrtMemcpy(dptr, POOL_BYTES, host, POOL_BYTES, _H2D) == 0
    key = ctypes.create_string_buffer(KEY_BUF)
    rc = _acl.aclrtIpcMemGetExportKey(dptr, POOL_BYTES, key, KEY_BUF, 0x1)
    print(f"[exporter] pool_base={hex(dptr.value or 0)} blocks={NUM_BLOCKS} export rc={rc}", flush=True)
    assert rc == 0
    open(keyfile, "wb").write(key.raw)
    for _ in range(240):
        if os.path.exists(keyfile + ".done"):
            break
        time.sleep(1)
    print("[exporter] done", flush=True)


class VAMap:
    """Maps an exported pool's identity to its imported per-process base, and
    yields per-block DeviceTensor views by auto-offset (step-3 mechanism).

    This is the synthetic stand-in for step 4's real table
    ``{(layer, "K"|"V") -> imported_peer_base}``; ``block(key_id, b)`` is the
    per-launch auto-offset the page-attention kernel arg will use in step 5.
    """

    def __init__(self) -> None:
        import torch  # noqa: PLC0415
        self._torch = torch
        self._pools: dict[str, tuple[int, tuple[int, ...]]] = {}

    def register(self, key_id: str, peer_base: int, pool_shape: tuple[int, ...]) -> None:
        self._pools[key_id] = (peer_base, pool_shape)

    def block(self, key_id: str, b: int):
        from pypto.runtime.device_tensor import DeviceTensor  # noqa: PLC0415
        peer_base, pool_shape = self._pools[key_id]
        pool = DeviceTensor(peer_base, pool_shape, self._torch.float32)
        return pool[b]  # __getitem__ -> DeviceTensor(peer_base + b*block_elems*4, (TILE,TILE))


def worker(dev: int, keyfile: str) -> None:
    import pypto.language as pl  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from pypto.ir.distributed_compiled_program import DistributedCompiledProgram, DistributedConfig  # noqa: PLC0415
    from pypto.runtime import DistributedWorker, RunConfig  # noqa: PLC0415

    @pl.jit.incore
    def _read_blk(token: pl.Tensor, kv: pl.Tensor, logits: pl.Out[pl.Tensor]):
        t = pl.load(token, [0, 0], [TILE, TILE])
        k = pl.load(kv, [0, 0], [TILE, TILE])
        return pl.store(pl.add(t, k), [0, 0], logits)

    @pl.jit
    def _decode_chip(token: pl.Tensor, kv: pl.Tensor, logits: pl.Out[pl.Tensor]):
        return _read_blk(token, kv, logits)

    @pl.jit.host
    def _decode(token: pl.Tensor, kv: pl.Tensor, logits: pl.Out[pl.Tensor]):
        return _decode_chip(token, kv, logits)

    dc = DistributedConfig(device_ids=[dev], num_sub_workers=0, block_dim=3)
    cfg = RunConfig(platform="a2a3", distributed_config=dc)
    host_token = torch.zeros((TILE, TILE), dtype=torch.float32).share_memory_()
    host_logits = torch.zeros((TILE, TILE), dtype=torch.float32).share_memory_()
    kv_sample = torch.zeros((TILE, TILE), dtype=torch.float32)
    decode_c = _decode.compile(host_token, kv_sample, host_logits, config=cfg)
    assert isinstance(decode_c, DistributedCompiledProgram)

    for _ in range(240):
        if os.path.exists(keyfile):
            break
        time.sleep(0.5)
    time.sleep(1.0)
    key = open(keyfile, "rb").read()

    all_ok = True
    with DistributedWorker([decode_c]) as rt:
        peer_base = rt.import_ipc(key, worker_id=0)
        print(f"[worker] import_ipc pool peer_base={hex(peer_base)}", flush=True)
        vamap = VAMap()
        vamap.register("synthetic_kv_pool", peer_base, (NUM_BLOCKS, TILE, TILE))

        for b in TEST_BLOCKS:
            blk = vamap.block("synthetic_kv_pool", b)  # auto-offset DeviceTensor
            host_token.zero_()
            host_logits.zero_()
            rt.run(decode_c, host_token, blk, host_logits)
            expected = torch.tensor(_block_pattern(b), dtype=torch.float32).reshape(TILE, TILE)
            ok = torch.allclose(host_logits, expected, rtol=1e-5, atol=1e-5)
            bad = (host_logits != expected).float().mean().item()
            all_ok = all_ok and ok
            print(f"[worker] block={b} data_ptr={blk!r} ok={ok} bad_ratio={bad:.4f} "
                  f"got0={host_logits.flatten()[0].item()} exp0={expected.flatten()[0].item()}", flush=True)

    open(keyfile + ".done", "w").write("1")
    print(f"VAMAP_MULTIBLOCK_{'PASS' if all_ok else 'FAIL'} blocks={list(TEST_BLOCKS)}", flush=True)
    sys.exit(0 if all_ok else 3)


if __name__ == "__main__":
    role, dev, keyfile = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    (exporter if role == "exporter" else worker)(dev, keyfile)
