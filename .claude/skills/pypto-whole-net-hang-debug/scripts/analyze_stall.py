#!/usr/bin/env python3
"""Summarize PyPTO/PTO2 stall evidence without claiming a root cause."""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from dataclasses import asdict, dataclass
import difflib
import json
from pathlib import Path
import re
import sys
from typing import Iterable, Iterator


TIMEOUT_RE = re.compile(
    r"PTO2 scheduler timeout sub_class=(?P<sub_class>\S+).*?"
    r"completed=(?P<completed>\d+)/(?P<total>\d+).*?"
    r"running=(?P<running>\d+)\s+ready=(?P<ready>\d+)\s+"
    r"waiting=(?P<waiting>\d+)\s+orch_done=(?P<orch_done>\d+).*?"
    r"stuck_task_id=(?P<task>-?\d+)\s+stuck_core=(?P<core>-?\d+)"
)
TASK_RE = re.compile(
    r"TASK\s+ring=(?P<ring>\d+)\s+task_id=(?P<task>-?\d+)\s+"
    r"state=(?P<state>RUNNING|READY|WAIT)\s+"
    r"fanin_refcount=(?P<fanin_ref>-?\d+)/(?P<fanin>-?\d+)\s+"
    r"kernels=\[aic:(?P<aic>-?\d+)\s+aiv0:(?P<aiv0>-?\d+)\s+"
    r"aiv1:(?P<aiv1>-?\d+)\]"
)
SNAPSHOT_RE = re.compile(
    r"\[STALL\s+thread=(?P<thread>\d+)\s+idle_iterations=(?P<idle>\d+)\]"
)
SUMMARY_RE = re.compile(
    r"SUMMARY\s+completed=(?P<completed>\d+)/(?P<total>\d+).*?"
    r"scan_ready=(?P<ready>\d+)\s+scan_waiting=(?P<waiting>\d+)\s+"
    r"scan_running=(?P<running>\d+)"
)
CORE_RE = re.compile(
    r"core(?P<core>\d+)\(busy\s+kernel=(?P<kernel>-?\d+)\s+"
    r"task=(?P<task>-?\d+)\s+cond_reg_state=(?P<cond>ack|fin)"
    r"(?P<anomaly>\s+ANOMALY)?"
)
ERROR_CODE_RES = {
    "orch_error_code": re.compile(r"orch_error(?:_code)?[=:]\s*(?P<code>-?\d+)"),
    "sched_error_code": re.compile(r"sched_error(?:_code)?[=:]\s*(?P<code>-?\d+)"),
}
GENERIC_PATTERNS = {
    # Exclude fractional timestamps such as "...14.507018]".
    "507018": re.compile(r"(?<![\d.])507018(?!\d)"),
    "507014": re.compile(r"(?<![\d.])507014(?!\d)"),
    "507899": re.compile(r"(?<![\d.])507899(?!\d)"),
    "allocator_deadlock": re.compile(
        r"Task Allocator Deadlock|Provable head-of-line", re.IGNORECASE
    ),
    "spin_timeout": re.compile(r"Timeout \(\d+ cycles\).*producer/consumers"),
    "handle_task_timeout": re.compile(r"HandleTaskTimeout"),
    "definite_fault": re.compile(
        r"page fault|illegal VA|illegal instruction|DMA.*fault|UB.*fault",
        re.IGNORECASE,
    ),
    "suspicious_devmm": re.compile(
        r"\bdevmm\b", re.IGNORECASE
    ),
    "stranded_cqe": re.compile(r"stranded CQE", re.IGNORECASE),
}

MAX_INPUT_FILES = 4096
MAX_INPUT_BYTES = 256 * 1024 * 1024


@dataclass(frozen=True)
class Timeout:
    source: str
    line: int
    sub_class: str
    completed: int
    total: int
    running: int
    ready: int
    waiting: int
    orch_done: int
    task_id: int
    ring_id: int
    local_task_id: int
    stuck_core: int


@dataclass(frozen=True)
class Task:
    source: str
    line: int
    ring: int
    task_id: int
    local_task_id: int
    stall_thread: int | None
    idle_iterations: int | None
    snapshot_id: str | None
    state: str
    fanin_refcount: int
    fanin_count: int
    aic: int
    aiv0: int
    aiv1: int


@dataclass(frozen=True)
class Core:
    source: str
    line: int
    core: int
    kernel: int
    task_id: int
    local_task_id: int
    stall_thread: int | None
    idle_iterations: int | None
    snapshot_id: str | None
    cond: str
    anomaly: bool


@dataclass(frozen=True)
class Kernel:
    config: str
    orchestration: str
    func_id: int
    name: str
    core_type: str | None
    source: str | None


def decode_task_id(raw: int) -> tuple[int, int]:
    if raw < 0:
        return -1, -1
    return raw >> 32, raw & 0xFFFFFFFF


def snapshot_id(stall_thread: int | None, idle_iterations: int | None) -> str | None:
    if stall_thread is None or idle_iterations is None:
        return None
    return f"thread={stall_thread}/idle={idle_iterations}"


def iter_input_files(
    inputs: Iterable[str],
    *,
    max_files: int = MAX_INPUT_FILES,
    max_bytes: int = MAX_INPUT_BYTES,
) -> Iterator[Path]:
    seen: set[Path] = set()
    total_files = 0
    for raw in inputs:
        path = Path(raw)
        if path.is_dir():
            candidates = sorted(
                p for p in path.rglob("*") if p.is_file() and not p.is_symlink()
            )
        else:
            candidates = [path]
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            if not candidate.is_file() or candidate.is_symlink():
                continue
            try:
                size = candidate.stat().st_size
            except OSError as exc:
                print(f"warning: skip {candidate}: {exc}", file=sys.stderr)
                continue
            if size > max_bytes:
                print(
                    f"warning: skip {candidate}: {size} bytes exceeds "
                    f"max input size {max_bytes}",
                    file=sys.stderr,
                )
                continue
            total_files += 1
            if total_files > max_files:
                raise ValueError(
                    f"input file count exceeds max_files={max_files}; "
                    "pass isolated log files rather than an entire workspace"
                )
            seen.add(resolved)
            yield candidate


def read_text(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        raise OSError(f"not a regular file: {path}")
    if path.stat().st_size > MAX_INPUT_BYTES:
        raise OSError(
            f"{path} exceeds max input size {MAX_INPUT_BYTES}; "
            "use an isolated log file"
        )
    return path.read_text(encoding="utf-8", errors="replace")


def literal_string(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def source_tail(node: ast.AST) -> str | None:
    """Recover the generated source suffix from str(_ROOT_DIR / ... / "x.cpp")."""

    def path_parts(expr: ast.AST) -> list[str] | None:
        if isinstance(expr, ast.Call) and expr.args:
            return path_parts(expr.args[0])
        if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Div):
            left = path_parts(expr.left)
            right = path_parts(expr.right)
            if left is None or right is None:
                return None
            return left + right
        if isinstance(expr, ast.Name) and expr.id == "_ROOT_DIR":
            return []
        value = literal_string(expr)
        if value is not None:
            return [value]
        return None

    parts = path_parts(node)
    return "/".join(parts) if parts else None


def parse_kernel_config(path: Path) -> list[Kernel]:
    try:
        tree = ast.parse(read_text(path), filename=str(path))
    except SyntaxError as exc:
        raise ValueError(f"cannot parse {path}: {exc}") from exc

    kernels_node: ast.List | ast.Tuple | None = None
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if any(isinstance(t, ast.Name) and t.id == "KERNELS" for t in targets):
            value = node.value
            if isinstance(value, (ast.List, ast.Tuple)):
                kernels_node = value
            break
    if kernels_node is None:
        return []

    orchestration = path.parent.name
    result: list[Kernel] = []
    for entry in kernels_node.elts:
        if not isinstance(entry, ast.Dict):
            continue
        fields: dict[str, ast.AST] = {}
        for key, value in zip(entry.keys, entry.values):
            if key is None:
                continue
            key_text = literal_string(key)
            if key_text is not None:
                fields[key_text] = value
        func_node = fields.get("func_id")
        name_node = fields.get("name")
        if not (
            isinstance(func_node, ast.Constant)
            and isinstance(func_node.value, int)
            and name_node is not None
        ):
            continue
        name = literal_string(name_node)
        if name is None:
            continue
        core_type = literal_string(fields["core_type"]) if "core_type" in fields else None
        source = source_tail(fields["source"]) if "source" in fields else None
        result.append(
            Kernel(
                config=str(path),
                orchestration=orchestration,
                func_id=func_node.value,
                name=name,
                core_type=core_type,
                source=source,
            )
        )
    return result


def collect_kernel_configs(
    build_dirs: Iterable[str],
    explicit_configs: Iterable[str],
    orchestration_filter: str | None,
) -> list[Kernel]:
    paths: list[Path] = [Path(p) for p in explicit_configs]
    for build_dir in build_dirs:
        paths.extend(sorted(Path(build_dir).rglob("kernel_config.py")))
    kernels: list[Kernel] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if orchestration_filter and orchestration_filter not in str(path):
            continue
        kernels.extend(parse_kernel_config(path))
    return kernels


def ordered_delta(before: list[str], after: list[str]) -> dict:
    """Return an ordered before/after delta, or mark it unreliable.

    ``dmesg -T`` normally grows by appending new lines.  A multiset subtraction
    loses ordering and can treat an old line as a new event after ring
    rotation.  Accept only a prefix/suffix sequence relation.  If neither is
    available, retain a small diagnostic diff but do not call it a reliable
    time-window delta.
    """

    if after[: len(before)] == before:
        return {
            "reliable": True,
            "method": "before-prefix",
            "new_lines": after[len(before) :],
        }
    max_overlap = min(len(before), len(after))
    overlap = 0
    for size in range(max_overlap, 0, -1):
        if before[-size:] == after[:size]:
            overlap = size
            break
    if overlap:
        return {
            "reliable": True,
            "method": "suffix-prefix-overlap",
            "new_lines": after[overlap:],
        }

    matcher = difflib.SequenceMatcher(a=before, b=after, autojunk=False)
    inserted: list[str] = []
    for tag, _i1, _i2, j1, j2 in matcher.get_opcodes():
        if tag in {"insert", "replace"}:
            inserted.extend(after[j1:j2])
    return {
        "reliable": False,
        "method": "sequence-diff-no-stable-overlap",
        "new_lines": inserted,
    }


def new_lines(before: list[str], after: list[str]) -> list[str]:
    """Compatibility wrapper; callers needing trust must use ordered_delta."""

    return ordered_delta(before, after)["new_lines"]


def parse_logs(paths: Iterable[Path]) -> dict:
    timeouts: list[Timeout] = []
    tasks: list[Task] = []
    cores: list[Core] = []
    summaries: list[dict] = []
    error_codes: dict[str, Counter[int]] = {
        "orch_error_code": Counter(),
        "sched_error_code": Counter(),
    }
    signatures = Counter()

    for path in paths:
        try:
            lines = read_text(path).splitlines()
        except (OSError, UnicodeError) as exc:
            print(f"warning: skip {path}: {exc}", file=sys.stderr)
            continue
        for line_no, line in enumerate(lines, 1):
            snapshot_match = SNAPSHOT_RE.search(line)
            stall_thread = (
                int(snapshot_match.group("thread")) if snapshot_match else None
            )
            idle_iterations = (
                int(snapshot_match.group("idle")) if snapshot_match else None
            )
            current_snapshot_id = snapshot_id(stall_thread, idle_iterations)
            timeout_match = TIMEOUT_RE.search(line)
            if timeout_match:
                values = timeout_match.groupdict()
                task_id = int(values["task"])
                ring_id, local_id = decode_task_id(task_id)
                timeouts.append(
                    Timeout(
                        source=str(path),
                        line=line_no,
                        sub_class=values["sub_class"],
                        completed=int(values["completed"]),
                        total=int(values["total"]),
                        running=int(values["running"]),
                        ready=int(values["ready"]),
                        waiting=int(values["waiting"]),
                        orch_done=int(values["orch_done"]),
                        task_id=task_id,
                        ring_id=ring_id,
                        local_task_id=local_id,
                        stuck_core=int(values["core"]),
                    )
                )
            task_match = TASK_RE.search(line)
            if task_match:
                values = task_match.groupdict()
                task_id = int(values["task"])
                _, local_id = decode_task_id(task_id)
                tasks.append(
                    Task(
                        source=str(path),
                        line=line_no,
                        ring=int(values["ring"]),
                        task_id=task_id,
                        local_task_id=local_id,
                        stall_thread=stall_thread,
                        idle_iterations=idle_iterations,
                        snapshot_id=current_snapshot_id,
                        state=values["state"],
                        fanin_refcount=int(values["fanin_ref"]),
                        fanin_count=int(values["fanin"]),
                        aic=int(values["aic"]),
                        aiv0=int(values["aiv0"]),
                        aiv1=int(values["aiv1"]),
                    )
                )
            summary_match = SUMMARY_RE.search(line)
            if summary_match:
                summaries.append(
                    {
                        "source": str(path),
                        "line": line_no,
                        "stall_thread": stall_thread,
                        "idle_iterations": idle_iterations,
                        "snapshot_id": current_snapshot_id,
                        **{k: int(v) for k, v in summary_match.groupdict().items()},
                    }
                )
            for core_match in CORE_RE.finditer(line):
                values = core_match.groupdict()
                task_id = int(values["task"])
                _, local_id = decode_task_id(task_id)
                cores.append(
                    Core(
                        source=str(path),
                        line=line_no,
                        core=int(values["core"]),
                        kernel=int(values["kernel"]),
                        task_id=task_id,
                        local_task_id=local_id,
                        stall_thread=stall_thread,
                        idle_iterations=idle_iterations,
                        snapshot_id=current_snapshot_id,
                        cond=values["cond"],
                        anomaly=bool(values["anomaly"]),
                    )
                )
            for name, regex in ERROR_CODE_RES.items():
                for match in regex.finditer(line):
                    code = int(match.group("code"))
                    if code != 0:
                        error_codes[name][code] += 1
            for name, regex in GENERIC_PATTERNS.items():
                if regex.search(line):
                    signatures[name] += 1

    return {
        "timeouts": timeouts,
        "tasks": tasks,
        "cores": cores,
        "summaries": summaries,
        "error_codes": error_codes,
        "signatures": signatures,
    }


def kernel_ids_from_evidence(parsed: dict) -> list[int]:
    """Return only same-snapshot RUNNING/core IDs, never WAIT/READY context."""

    ids: set[int] = set()
    for task in parsed["tasks"]:
        if task.state != "RUNNING":
            continue
        ids.update(k for k in (task.aic, task.aiv0, task.aiv1) if k >= 0)
    for core in parsed["cores"]:
        if core.kernel >= 0:
            ids.add(core.kernel)
    return sorted(ids)


def dependency_kernel_ids(parsed: dict) -> list[int]:
    ids: set[int] = set()
    for task in parsed["tasks"]:
        if task.state == "RUNNING":
            continue
        ids.update(k for k in (task.aic, task.aiv0, task.aiv1) if k >= 0)
    return sorted(ids)


def mappings_for(kernel_ids: Iterable[int], kernels: Iterable[Kernel]) -> dict[int, list[Kernel]]:
    result: dict[int, list[Kernel]] = {}
    all_kernels = list(kernels)
    for func_id in kernel_ids:
        result[func_id] = [kernel for kernel in all_kernels if kernel.func_id == func_id]
    return result


def classify_hint(timeout: Timeout) -> str:
    if timeout.running:
        return (
            "S1: a task was RUNNING on a core at the no-progress snapshot; "
            "inspect that task and its preceding boundary"
        )
    if timeout.ready:
        return "S3: scheduler/resource/dispatch path"
    if timeout.waiting:
        return "S4: dependency/fanin/wiring path"
    if not timeout.orch_done:
        return "S5: orchestrator submission/termination path"
    return "unknown: inspect accounting/corruption"


def parsed_for_source(parsed: dict, source: str) -> dict:
    return {
        "timeouts": [item for item in parsed["timeouts"] if item.source == source],
        "tasks": [item for item in parsed["tasks"] if item.source == source],
        "cores": [item for item in parsed["cores"] if item.source == source],
        "summaries": [
            item for item in parsed["summaries"] if item["source"] == source
        ],
    }


def parsed_for_snapshot(parsed: dict, source: str, current_snapshot_id: str) -> dict:
    return {
        "timeouts": [],
        "tasks": [
            item
            for item in parsed["tasks"]
            if item.source == source and item.snapshot_id == current_snapshot_id
        ],
        "cores": [
            item
            for item in parsed["cores"]
            if item.source == source and item.snapshot_id == current_snapshot_id
        ],
        "summaries": [
            item
            for item in parsed["summaries"]
            if item["source"] == source
            and item.get("snapshot_id") == current_snapshot_id
        ],
    }


def mapping_dict(ids: Iterable[int], kernels: list[Kernel]) -> dict[str, list[dict]]:
    return {
        str(func_id): [asdict(kernel) for kernel in candidates]
        for func_id, candidates in mappings_for(ids, kernels).items()
    }


def serialize(parsed: dict, kernels: list[Kernel], dmesg: dict | None) -> dict:
    sources = sorted(
        {
            *(item.source for item in parsed["timeouts"]),
            *(item.source for item in parsed["tasks"]),
            *(item.source for item in parsed["cores"]),
            *(item["source"] for item in parsed["summaries"]),
        }
    )
    source_reports: dict[str, dict] = {}
    for source in sources:
        scoped = parsed_for_source(parsed, source)
        snapshot_ids = sorted(
            {
                *(item.snapshot_id for item in scoped["tasks"] if item.snapshot_id),
                *(item.snapshot_id for item in scoped["cores"] if item.snapshot_id),
            }
        )
        snapshot_reports: dict[str, dict] = {}
        for current_snapshot_id in snapshot_ids:
            snapshot_scoped = parsed_for_snapshot(
                parsed, source, current_snapshot_id
            )
            stuck_ids = kernel_ids_from_evidence(snapshot_scoped)
            dep_ids = dependency_kernel_ids(snapshot_scoped)
            snapshot_reports[current_snapshot_id] = {
                "tasks": [asdict(item) for item in snapshot_scoped["tasks"]],
                "cores": [asdict(item) for item in snapshot_scoped["cores"]],
                "stuck_kernel_mappings": mapping_dict(stuck_ids, kernels),
                "dependency_context_mappings": mapping_dict(dep_ids, kernels),
                "correlation": (
                    "same-source same-snapshot RUNNING/core evidence"
                    if stuck_ids
                    else "no RUNNING/core kernel in this snapshot"
                ),
            }
        source_reports[source] = {
            "timeouts": [
                asdict(item) | {"hint": classify_hint(item)}
                for item in scoped["timeouts"]
            ],
            "tasks": [asdict(item) for item in scoped["tasks"]],
            "cores": [asdict(item) for item in scoped["cores"]],
            "summaries": scoped["summaries"],
            "snapshot_reports": snapshot_reports,
            "correlation": (
                "timeouts are not joined to TASK/core snapshots automatically; "
                "snapshot mappings are diagnostic context only"
            ),
        }

    return {
        "timeouts": [asdict(item) | {"hint": classify_hint(item)} for item in parsed["timeouts"]],
        "tasks": [asdict(item) for item in parsed["tasks"]],
        "cores": [asdict(item) for item in parsed["cores"]],
        "summaries": parsed["summaries"],
        "error_codes": {
            name: dict(counter) for name, counter in parsed["error_codes"].items()
        },
        "signatures": dict(parsed["signatures"]),
        "source_reports": source_reports,
        "cross_source_correlation_attempted": False,
        "dmesg": dmesg,
        "caveats": [
            "507018 alone does not identify deadlock, OOM, or a kernel root cause.",
            "stuck_task_id is a ring/local task id, not a kernel func_id.",
            "COND ack/fin is not an AICore program counter.",
            "Evidence from different source files is not correlated automatically.",
            "WAIT/READY mappings are dependency context, not the stuck kernel.",
            "Kernel-config mappings remain candidates until the config is bound to the same run/build.",
            "Multiple kernel-config candidates require exact orchestration/build selection.",
        ],
    }


def print_markdown(report: dict) -> None:
    print("# PyPTO stall evidence summary")
    print()
    print(
        "> Evidence is grouped by source file. The tool never joins a timeout "
        "from one file to TASK/kernel IDs from another file."
    )
    print()
    print("## Scheduler timeouts")
    if not report["timeouts"]:
        print("- none parsed")
    for item in report["timeouts"]:
        print(
            f"- `{item['source']}:{item['line']}` {item['sub_class']} "
            f"completed={item['completed']}/{item['total']} "
            f"running/ready/waiting={item['running']}/{item['ready']}/{item['waiting']} "
            f"task={item['task_id']}=(ring {item['ring_id']}, local {item['local_task_id']}) "
            f"core={item['stuck_core']}; next={item['hint']}"
        )
    print()
    print("## TASK / register evidence")
    if not report["tasks"] and not report["cores"]:
        print("- none parsed")
    for item in report["tasks"]:
        print(
            f"- TASK `{item['source']}:{item['line']}` state={item['state']} "
            f"task={item['task_id']} local={item['local_task_id']} "
            f"fanin={item['fanin_refcount']}/{item['fanin_count']} "
            f"kernels=aic:{item['aic']},aiv0:{item['aiv0']},aiv1:{item['aiv1']}"
        )
    for item in report["cores"]:
        suffix = " ANOMALY" if item["anomaly"] else ""
        print(
            f"- CORE `{item['source']}:{item['line']}` core={item['core']} "
            f"kernel={item['kernel']} task={item['task_id']} "
            f"cond={item['cond']}{suffix}"
        )
    print()
    print("## Error signatures")
    print(f"- orchestrator codes: {report['error_codes']['orch_error_code'] or '{}'}")
    print(f"- scheduler codes: {report['error_codes']['sched_error_code'] or '{}'}")
    print(f"- generic signatures: {report['signatures'] or '{}'}")
    print()
    print("## Per-source candidate kernel mappings")
    if not report["source_reports"]:
        print("- no TASK/core evidence found")
    for source, source_report in report["source_reports"].items():
        print(f"- source `{source}`: {source_report['correlation']}")
        for current_snapshot_id, snapshot_report in source_report[
            "snapshot_reports"
        ].items():
            print(f"  - snapshot `{current_snapshot_id}`: {snapshot_report['correlation']}")
            mappings = snapshot_report["stuck_kernel_mappings"]
            if not mappings:
                print("    - no same-snapshot RUNNING/core kernel id")
            for func_id, candidates in mappings.items():
                if not candidates:
                    print(
                        f"    - RUNNING func_id={func_id}: "
                        "no candidate config mapping found"
                    )
                    continue
                print(f"    - RUNNING func_id={func_id}:")
                for kernel in candidates:
                    print(
                        f"      - `{kernel['orchestration']}` `{kernel['name']}` "
                        f"core={kernel['core_type']} source={kernel['source']} "
                        f"config=`{kernel['config']}`"
                    )
                if len(candidates) > 1:
                    print(
                        "      - warning: ambiguous configs; "
                        "bind the exact run/build/orchestration"
                    )
            dep_mappings = snapshot_report["dependency_context_mappings"]
            if dep_mappings:
                print(
                    "    - WAIT/READY dependency-context func_ids="
                    + ",".join(dep_mappings)
                    + " (not treated as the stuck kernel)"
                )
    if report["dmesg"] is not None:
        print()
        print("## dmesg delta")
        print(
            f"- new lines={report['dmesg']['new_line_count']} "
            f"relevant={report['dmesg']['relevant_line_count']} "
            f"reliable={report['dmesg']['reliable']} "
            f"method={report['dmesg']['method']}"
        )
        for line in report["dmesg"]["relevant_lines"][:50]:
            print(f"  - `{line}`")
    print()
    print("## Mandatory caveats")
    for caveat in report["caveats"]:
        print(f"- {caveat}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log",
        action="append",
        default=[],
        help="host/device log file or directory; repeatable",
    )
    parser.add_argument(
        "--build-dir",
        action="append",
        default=[],
        help="exact build directory; searches kernel_config.py recursively",
    )
    parser.add_argument(
        "--kernel-config",
        action="append",
        default=[],
        help="explicit kernel_config.py; repeatable",
    )
    parser.add_argument(
        "--orchestration",
        help="substring used to narrow kernel_config paths, e.g. full_moe_chip_orch",
    )
    parser.add_argument("--dmesg-before")
    parser.add_argument("--dmesg-after")
    parser.add_argument(
        "--dmesg-diff",
        help="preferred isolated dmesg diff; lines beginning with '+' are treated as new",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON instead of Markdown")
    args = parser.parse_args()
    if not args.log:
        parser.error("at least one --log is required")
    if bool(args.dmesg_before) != bool(args.dmesg_after):
        parser.error("--dmesg-before and --dmesg-after must be provided together")
    if args.dmesg_diff and args.dmesg_before:
        parser.error("--dmesg-diff is mutually exclusive with before/after")
    return args


def main() -> int:
    args = parse_args()
    try:
        log_paths = list(iter_input_files(args.log))
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    parsed = parse_logs(log_paths)
    try:
        kernels = collect_kernel_configs(
            args.build_dir, args.kernel_config, args.orchestration
        )
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    dmesg_report = None
    if args.dmesg_diff:
        try:
            diff_lines = read_text(Path(args.dmesg_diff)).splitlines()
        except OSError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        delta = [
            line[1:]
            for line in diff_lines
            if line.startswith("+") and not line.startswith("+++")
        ]
        delta_info = {
            "reliable": True,
            "method": "explicit-diff",
            "new_lines": delta,
        }
    elif args.dmesg_before:
        try:
            before = read_text(Path(args.dmesg_before)).splitlines()
            after = read_text(Path(args.dmesg_after)).splitlines()
        except OSError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        delta_info = ordered_delta(before, after)
        delta = delta_info["new_lines"]
    else:
        delta_info = None

    if delta_info is not None:
        relevant = [
            line
            for line in delta
            if any(regex.search(line) for regex in GENERIC_PATTERNS.values())
        ]
        dmesg_report = {
            "reliable": delta_info["reliable"],
            "method": delta_info["method"],
            "new_line_count": len(delta),
            "relevant_line_count": len(relevant),
            "relevant_lines": relevant,
        }

    report = serialize(parsed, kernels, dmesg_report)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print_markdown(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
