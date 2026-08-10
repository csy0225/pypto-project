#!/usr/bin/env python3
"""Claude independent recompute of P2 phase-0 critical-tail, from raw slots only.

Does NOT use codex's analyze_latest_entry.py. Reads the original timing_records.pt
in place and re-derives critical_tail = max(rank_exit) - max(rank_entry) using the
slot layout read directly out of the kernel source:

    slot 0 = magic
    slot 1 = rank
    slot 2 = total ticks (ar_end - ar_entry)
    slot 4 = ar_entry low  32 bits
    slot 5 = ar_entry high 32 bits

Emits provenance (absolute path + sha256) for every input it touches.
"""
import hashlib
import json
import statistics
import sys
from pathlib import Path

import torch

TICKS_MHZ = 50.0
MAGIC = {
    "prod_total": 0x41525031,
    "prod_phase0": 0x41503031,
    "prod_coarse": 0x41504331,
    "prod_step": 0x41505331,
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pct(values):
    ordered = sorted(values)
    n = len(ordered)
    return {
        "n": n,
        "min": ordered[0],
        "p50": statistics.median(ordered),
        "mean": statistics.fmean(ordered),
        "p95": ordered[min(n - 1, int(n * 0.95))],
        "max": ordered[-1],
    }


PHASE_LABELS = [
    "self_tput_drain",
    "w1_first_pre_notify_fence",
    "w1_first_notify_call",
    "w1_remaining_notify",
    "w1_wait",
    "reduce_owned",
    "publish_issue",
    "w2_first_pre_notify_fence",
    "w2_first_notify_call_incl_publish_drain",
    "w2_remaining_notify",
    "w2_wait",
    "local_copy_issue",
    "w3_first_pre_notify_fence",
    "w3_first_notify_call_incl_copy_drain",
    "w3_remaining_notify",
    "w3_wait",
    "final",
    "tail",
]


def analyze(pt_path: Path, mode: str):
    rec = torch.load(pt_path, map_location="cpu", weights_only=False)
    if not isinstance(rec, torch.Tensor):
        raise TypeError(f"{pt_path}: expected Tensor, got {type(rec)}")
    n_epoch, n_rank, _ = rec.shape

    magic_ok = bool(torch.all(rec[:, :, 0] == MAGIC[mode]))
    rank_ok = bool(
        torch.all(rec[:, :, 1] == torch.arange(n_rank, dtype=torch.int32).reshape(1, n_rank))
    )

    lo = rec[:, :, 4].to(torch.int64) & 0xFFFFFFFF
    hi = rec[:, :, 5].to(torch.int64) & 0xFFFFFFFF
    entry = (hi << 32) | lo
    # prod_total: slot2 = total. prod_phase0: slot2 = phase count (18), slot3 = total.
    total_slot = 3 if mode == "prod_phase0" else 2
    total = rec[:, :, total_slot].to(torch.int64) & 0xFFFFFFFF
    exit_ = entry + total

    crit_ticks = exit_.max(dim=1).values - entry.max(dim=1).values
    latest_entry_rank = entry.argmax(dim=1)
    last_exit_rank = exit_.argmax(dim=1)
    same = int((latest_entry_rank == last_exit_rank).sum())

    # duration of the latest-entry rank itself (codex's "latest-entry selected")
    sel = total.gather(1, latest_entry_rank.unsqueeze(1)).squeeze(1)

    from collections import Counter

    out = {
        "mode": mode,
        "epochs": n_epoch,
        "ranks": n_rank,
        "magic_ok": magic_ok,
        "rank_ok": rank_ok,
        "critical_tail_us": pct((crit_ticks.double() / TICKS_MHZ).tolist()),
        "latest_entry_selected_us": pct((sel.double() / TICKS_MHZ).tolist()),
        "pooled_total_us": pct((total.double() / TICKS_MHZ).flatten().tolist()),
        "latest_entry_eq_last_exit": f"{same}/{n_epoch}",
        "latest_entry_rank_counts": dict(Counter(latest_entry_rank.tolist())),
    }

    if mode != "prod_phase0":
        return out

    # 18 phase deltas live in slots 8..25; take them on the latest-entry rank only.
    ph = rec[:, :, 8:26].to(torch.int64) & 0xFFFFFFFF
    idx = latest_entry_rank.view(-1, 1, 1).expand(-1, 1, ph.shape[2])
    ph_sel = ph.gather(1, idx).squeeze(1)  # [epoch, 18]

    err = (ph_sel.sum(dim=1) - sel).tolist()
    out["phase_sum_error_ticks"] = {"min": min(err), "max": max(err)}
    out["phase_us_p50"] = {
        lbl: statistics.median((ph_sel[:, i].double() / TICKS_MHZ).tolist())
        for i, lbl in enumerate(PHASE_LABELS)
    }

    # Per-epoch calibrated split, then percentile (never sum independent percentiles).
    us = ph_sel.double() / TICKS_MHZ
    w1_ctl = us[:, 1] + us[:, 2] + us[:, 3]
    first_peer_ctl = us[:, 1] + us[:, 2]          # no-payload calibration from Wave1
    w2_ctl = us[:, 7] + first_peer_ctl + us[:, 9]
    w3_ctl = us[:, 12] + first_peer_ctl + us[:, 14]
    w2_pub_excess = us[:, 8] - first_peer_ctl
    w3_copy_excess = us[:, 13] - first_peer_ctl
    waits = us[:, 4] + us[:, 10] + us[:, 15]
    control = w1_ctl + w2_ctl + w3_ctl + waits
    data = us[:, 0] + us[:, 5] + us[:, 6] + w2_pub_excess + us[:, 11] + w3_copy_excess
    tailf = us[:, 16] + us[:, 17]
    med = lambda t: statistics.median(t.tolist())
    out["calibrated_p50"] = {
        "w1_notify_control": med(w1_ctl),
        "w2_notify_control": med(w2_ctl),
        "w2_publish_completion_excess": med(w2_pub_excess),
        "w3_notify_control": med(w3_ctl),
        "w3_copy_completion_excess": med(w3_copy_excess),
        "explicit_waits": med(waits),
        "control_total_incl_waits": med(control),
        "control_total_excl_waits": med(control - waits),
        "data_compute_total": med(data),
        "final_tail": med(tailf),
        "accounted_total": med(control + data + tailf),
        "per_epoch_accounting_error_max_us": max((control + data + tailf - us.sum(dim=1)).abs().tolist()),
    }
    # notify count law: per-peer marginal cost from the remaining-6-peer phases
    out["notify_law"] = {
        "w1_remaining_6peer_us_p50": med(us[:, 3]),
        "w2_remaining_6peer_us_p50": med(us[:, 9]),
        "w3_remaining_6peer_us_p50": med(us[:, 14]),
        "marginal_per_peer_us": (med(us[:, 3]) + med(us[:, 9]) + med(us[:, 14])) / 18.0,
        "first_peer_control_us_p50": med(first_peer_ctl),
    }
    return out


def main(argv):
    out = {
        "analyzer": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        "author": "claude (independent recompute; does not use codex analyze_latest_entry.py)",
        "arms": [],
    }
    for spec in argv[1:]:
        path_s, mode = spec.rsplit(":", 1)
        p = Path(path_s).resolve()
        res = analyze(p, mode)
        res["input"] = {"path": str(p), "sha256": sha256(p), "bytes": p.stat().st_size}
        out["arms"].append(res)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv)
