#!/usr/bin/env python3
"""Step-1/Step-3 confirmatory probe (device required; run on a FREE card).

Closes the two open questions the prior P-track never explicitly measured:

  (A) VA EQUALITY across processes: does aclrtIpcMemImportByKey return the SAME
      virtual address as the exporter's device pointer, or a different one?
      This decides how hard step-3 "auto offset at kernel launch" is:
        - same VA  -> a sub-tensor at base+off is already valid as-is (trivial).
        - diff VA  -> need a per-block base map: pypto_ptr = peer_base + (off).
      Also checks that a base+OFFSET sub-pointer imports to peer_base+offset
      (offset preservation), the mechanism P5/P8 rely on.

  (B) torch_npu HIGH-LEVEL IPC API surface: does torch_npu expose a torch.cuda
      style tensor-IPC (torch.multiprocessing reductions / storage._share_npu_ /
      get_ipc_handle), or must we stay on raw ACL aclrtIpcMem* (what P2/P4/P7
      already use)?

Usage:  python _stage_va_ipc_probe.py --device <free_card_id>
PASS when: data round-trips (got==exp) AND both VA facts are printed clearly.
"""
from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import multiprocessing as mp
import os
import sys

ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2
ACL_MEM_MALLOC_HUGE_FIRST = 0
EXPORT_FLAG_DISABLE_PID_VALIDATION = 0x1
IMPORT_FLAG_DEFAULT = 0x0
OFFSET = 4096  # sub-pointer offset to test offset preservation


def load_acl() -> ctypes.CDLL:
    acl = ctypes.CDLL(ctypes.util.find_library("ascendcl") or "libascendcl.so")
    acl.aclInit.argtypes = [ctypes.c_char_p]
    acl.aclrtSetDevice.argtypes = [ctypes.c_int32]
    acl.aclrtResetDevice.argtypes = [ctypes.c_int32]
    acl.aclrtMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_int]
    acl.aclrtFree.argtypes = [ctypes.c_void_p]
    acl.aclrtMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    acl.aclrtSynchronizeDevice.argtypes = []
    acl.aclrtIpcMemGetExportKey.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p, ctypes.c_size_t, ctypes.c_uint64]
    acl.aclrtIpcMemImportByKey.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p, ctypes.c_uint64]
    acl.aclrtIpcMemClose.argtypes = [ctypes.c_char_p]
    for fn in (acl.aclInit, acl.aclrtSetDevice, acl.aclrtResetDevice, acl.aclrtMalloc,
               acl.aclrtFree, acl.aclrtMemcpy, acl.aclrtSynchronizeDevice,
               acl.aclrtIpcMemGetExportKey, acl.aclrtIpcMemImportByKey, acl.aclrtIpcMemClose):
        fn.restype = ctypes.c_int
    return acl


def check(rc: int, what: str) -> None:
    if rc != 0:
        raise RuntimeError(f"{what} failed rc={rc}")


def expected_bytes(n: int) -> bytes:
    return bytes((i % 251) for i in range(n))


def exporter(device: int, nbytes: int, key_len: int, conn) -> None:
    acl = load_acl()
    dev_ptr = ctypes.c_void_p()
    key_buf = ctypes.create_string_buffer(key_len)
    try:
        check(acl.aclInit(None), "exporter aclInit")
        check(acl.aclrtSetDevice(device), f"exporter setDevice({device})")
        check(acl.aclrtMalloc(ctypes.byref(dev_ptr), nbytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc")
        src = ctypes.create_string_buffer(expected_bytes(nbytes), nbytes)
        check(acl.aclrtMemcpy(dev_ptr, nbytes, ctypes.cast(src, ctypes.c_void_p), nbytes, ACL_MEMCPY_HOST_TO_DEVICE), "H2D")
        check(acl.aclrtSynchronizeDevice(), "exporter sync")
        check(acl.aclrtIpcMemGetExportKey(dev_ptr, nbytes, key_buf, key_len, EXPORT_FLAG_DISABLE_PID_VALIDATION), "getExportKey")
        print(f"[exporter] dev_ptr={hex(dev_ptr.value or 0)} nbytes={nbytes}", flush=True)
        conn.send((key_buf.raw, int(dev_ptr.value or 0)))
        assert conn.recv() == "done"
        acl.aclrtIpcMemClose(key_buf)
        acl.aclrtFree(dev_ptr)
    except BaseException as exc:  # noqa: BLE001
        try:
            conn.send(("ERR", repr(exc)))
        except Exception:
            pass
        raise
    finally:
        acl.aclrtResetDevice(device)


def importer(device: int, nbytes: int, key_len: int, conn) -> bool:
    acl = load_acl()
    try:
        key, exp_ptr = conn.recv()
        if key == "ERR":
            raise RuntimeError(f"exporter error: {exp_ptr}")
        check(acl.aclInit(None), "importer aclInit")
        check(acl.aclrtSetDevice(device), f"importer setDevice({device})")
        imported = ctypes.c_void_p()
        key_buf = ctypes.create_string_buffer(key, key_len)
        check(acl.aclrtIpcMemImportByKey(ctypes.byref(imported), key_buf, IMPORT_FLAG_DEFAULT), "importByKey")
        imp_ptr = int(imported.value or 0)
        same_va = imp_ptr == exp_ptr
        print(f"[importer] imported_ptr={hex(imp_ptr)} exporter_ptr={hex(exp_ptr)} SAME_VA={same_va}", flush=True)
        print(f"[VA-VERDICT] {'SAME across process -> step3 offset is trivial' if same_va else 'DIFFERENT -> need per-block base map (pypto=peer_base+off)'}", flush=True)

        # full-buffer readback
        dst = ctypes.create_string_buffer(nbytes)
        check(acl.aclrtMemcpy(ctypes.cast(dst, ctypes.c_void_p), nbytes, imported, nbytes, ACL_MEMCPY_DEVICE_TO_HOST), "D2H full")
        # offset sub-pointer readback (offset preservation)
        sub = ctypes.c_void_p(imp_ptr + OFFSET)
        dst2 = ctypes.create_string_buffer(nbytes - OFFSET)
        check(acl.aclrtMemcpy(ctypes.cast(dst2, ctypes.c_void_p), nbytes - OFFSET, sub, nbytes - OFFSET, ACL_MEMCPY_DEVICE_TO_HOST), "D2H sub")
        check(acl.aclrtSynchronizeDevice(), "importer sync")

        exp = expected_bytes(nbytes)
        ok_full = bytes(dst.raw) == exp
        ok_sub = bytes(dst2.raw) == exp[OFFSET:]
        print(f"[importer] ok_full={ok_full} ok_sub(off={OFFSET})={ok_sub}", flush=True)
        conn.send("done")
        return ok_full and ok_sub
    finally:
        acl.aclrtResetDevice(device)


def probe_torch_npu_ipc_api() -> None:
    """Part B: is there a torch.cuda-style high-level tensor-IPC in torch_npu?"""
    print("\n[B] torch_npu high-level IPC API surface", flush=True)
    try:
        import torch
        import torch_npu  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        print(f"[B] torch_npu import failed: {exc!r}", flush=True)
        return
    candidates = [
        "torch.multiprocessing.reductions.rebuild_cuda_tensor",
        "torch.multiprocessing.reductions.reduce_tensor",
        "torch_npu.multiprocessing",
    ]
    import importlib
    for dotted in candidates:
        parts = dotted.split(".")
        obj = None
        for i in range(len(parts), 0, -1):
            try:
                mod = importlib.import_module(".".join(parts[:i]))
            except Exception:
                continue
            obj = mod
            for a in parts[i:]:
                obj = getattr(obj, a, None)
                if obj is None:
                    break
            break
        print(f"[B] {'OK ' if obj is not None else '-- '}{dotted}", flush=True)
    # storage-level IPC handle methods
    try:
        t = torch.arange(64, dtype=torch.int32, device="npu:0")
        st = t.untyped_storage()
        for m in ("_share_npu_", "_share_cuda_", "_share_filename_cpu_"):
            print(f"[B] storage.{m}: {'present' if hasattr(st, m) else 'absent'}", flush=True)
        # does torch.multiprocessing know how to reduce an NPU tensor?
        from torch.multiprocessing.reductions import reduce_tensor
        try:
            red = reduce_tensor(t)
            print(f"[B] reduce_tensor(npu_tensor) -> {red[0].__name__}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[B] reduce_tensor(npu_tensor) raised: {exc!r}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[B] storage probe skipped: {exc!r}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=int(os.environ.get("PROBE_DEVICE", "0")))
    ap.add_argument("--nbytes", type=int, default=1 << 20)  # 1 MiB (matches live KV block)
    ap.add_argument("--key-len", type=int, default=1024)
    ap.add_argument("--skip-torch", action="store_true")
    args = ap.parse_args()

    ctx = mp.get_context("spawn")
    a, b = ctx.Pipe(duplex=True)
    p = ctx.Process(target=exporter, args=(args.device, args.nbytes, args.key_len, a), name="va-exporter")
    p.start()
    ok = False
    try:
        ok = importer(args.device, args.nbytes, args.key_len, b)
    finally:
        p.join(timeout=15)

    if not args.skip_torch:
        probe_torch_npu_ipc_api()

    verdict = "PASS" if (ok and p.exitcode == 0) else "FAIL"
    print(f"\nVA_IPC_PROBE_{verdict} device={args.device} nbytes={args.nbytes} exporter_exit={p.exitcode}", flush=True)
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
