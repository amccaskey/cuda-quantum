#!/usr/bin/env python3
"""Measure folded versus explicitly unrolled P0 representation growth.

This is the machine-readable paper-evaluation driver. The corresponding
user-facing example lives in ``examples/standalone/06_folding_ablation.py``.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import signal
import subprocess
import sys
import time

import cudaq.logical as qlx



def _run_worker(command, stream, *, timeout: int):
    """Run one matrix worker and tear down its process tree on timeout."""

    if len(command) < 2 or Path(command[1]).resolve() != Path(__file__).resolve():
        raise RuntimeError("workers may execute only this study file")
    process = subprocess.Popen(
        command,
        stdout=stream,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=(os.name == "posix"),
    )
    try:
        returncode = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired as error:
        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        else:
            process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            if os.name == "posix":
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                process.kill()
            process.wait()
        stream.write(f"\nWORKER_TIMEOUT_SECONDS={timeout}\n")
        stream.flush()
        raise subprocess.TimeoutExpired(command, timeout) from error
    return subprocess.CompletedProcess(command, returncode)


QLX_SOURCE = {
    "module": str(qlx.__file__),
    "version": getattr(qlx, "__version__", "unknown"),
}



def _git(repo: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *arguments],
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()


def _git_state(repo: Path) -> dict[str, object]:
    diff = subprocess.check_output(
        ["git", "-C", str(repo), "diff", "--binary", "HEAD"],
    )
    return {
        "commit": _git(repo, "rev-parse", "HEAD"),
        "branch": _git(repo, "branch", "--show-current"),
        "describe": _git(repo, "describe", "--always", "--dirty"),
        "dirty": bool(_git(repo, "status", "--porcelain")),
        "tracked_diff_sha256": "sha256:" + hashlib.sha256(diff).hexdigest(),
    }


def _peak_rss_mib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _walk(operation):
    yield operation
    for region in operation.regions:
        for block in region.blocks:
            for child in block.operations:
                yield from _walk(child.operation)


def _artifact_profile(build) -> dict[str, object]:
    started = time.perf_counter()
    module = build._fresh_module()
    counts = Counter(operation.name for operation in _walk(module.operation))
    mlir = module.operation.get_asm(assume_verified=True).encode("utf-8")
    bundle = build.serialize()
    return {
        "content_sha256": build.content_sha256,
        "stage": str(build.stage),
        "facets": [str(value) for value in build.facets],
        "structural_operation_count": sum(counts.values()),
        "operation_counts": dict(sorted(counts.items())),
        "mlir_bytes": len(mlir),
        "portable_bundle_bytes": len(bundle),
        "inspection_wall_s": time.perf_counter() - started,
        "process_peak_rss_mib_after_inspection": _peak_rss_mib(),
    }


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _host_record() -> dict[str, object]:
    cpu_model = None
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    total_memory = None
    try:
        import psutil

        total_memory = psutil.virtual_memory().total
    except Exception:
        pass
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_model": cpu_model,
        "logical_cpus": os.cpu_count(),
        "total_memory_bytes": total_memory,
        "python": sys.version,
        "python_executable": sys.executable,
        "dependencies": {
            name: _package_version(name)
            for name in ("numpy", "mpmath", "psutil")
        },
    }


def _program(mode: str, count: int):
    def kernel() -> None:
        data = qlx.allocate(1, state=qlx.types.zero)
        if mode == "folded":
            data[0], = qlx.ops.repeat(
                count,
                carries=(data[0],),
                body=lambda _iteration, value: (qlx.h(value),),
            )
        else:
            for _ in range(count):
                data[0] = qlx.h(data[0])
        qlx.discard(data)

    return qlx.program(kernel, name=f"{mode}_h_{count}")


def _run_matrix(arguments: argparse.Namespace) -> int:
    counts = (1, 4, 16, 64, 256) if arguments.quick_matrix else (
        1, 4, 16, 64, 256, 1024, 4096,
    )
    if arguments.matrix_output_dir is None:
        raise RuntimeError("--matrix requires --matrix-output-dir")
    output_dir = arguments.matrix_output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()) and not (arguments.resume or arguments.force):
        raise FileExistsError(
            f"refusing to use nonempty {output_dir}; pass --resume or --force"
        )
    records = []
    by_count = {}
    for count in counts:
        by_count[count] = {}
        for mode in ("folded", "expanded"):
            output = output_dir / f"{mode}-{count}.json"
            log = output.with_suffix(".log")
            if not (output.exists() and arguments.resume):
                if (output.exists() or log.exists()) and not arguments.force:
                    raise FileExistsError(
                        f"refusing to overwrite {output} or {log}"
                    )
                command = (
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--qlx-repo",
                    str(arguments.qlx_repo.resolve()),
                    "--mode",
                    mode,
                    "--count",
                    str(count),
                    "--output",
                    str(output),
                    "--force",
                )
                with log.open("w", encoding="utf-8") as stream:
                    completed = _run_worker(
                        command, stream,
                        timeout=arguments.worker_timeout_seconds,
                    )
                if completed.returncode:
                    raise subprocess.CalledProcessError(
                        completed.returncode, command
                    )
            record = json.loads(output.read_text(encoding="utf-8"))
            if (
                record.get("schema") != "qlx.paper.folding-ablation-point/v2"
                or record.get("mode") != mode
                or record.get("repeat_count") != count
                or record.get("study_sha256")
                != hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
                or record.get("source", {}).get("imported_qlx") != QLX_SOURCE
                or record.get("logical_actions", {}).get("qlx_standard_h")
                != count
            ):
                raise RuntimeError(f"invalid folding result {output}")
            by_count[count][mode] = record
            records.append(
                {
                    "mode": mode,
                    "count": count,
                    "path": str(output),
                    "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                }
            )
        if (
            by_count[count]["folded"]["logical_actions"]
            != by_count[count]["expanded"]["logical_actions"]
        ):
            raise RuntimeError(
                f"folded and expanded semantic counts differ at {count}"
            )
    receipt = {
        "schema": "qlx.paper.folding-ablation-matrix/v2",
        "source": QLX_SOURCE,
        "study_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "counts": list(counts),
        "worker_timeout_seconds": arguments.worker_timeout_seconds,
        "records": records,
        "acceptance": {"folded_expanded_semantics_match": True},
    }
    destination = output_dir / "matrix-receipt.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)
    print(json.dumps({"receipt": str(destination), "records": len(records)}))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qlx-repo", type=Path, required=True)
    parser.add_argument("--mode", choices=("folded", "expanded"))
    parser.add_argument("--count", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--matrix", action="store_true")
    parser.add_argument("--matrix-output-dir", type=Path)
    parser.add_argument("--quick-matrix", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--worker-timeout-seconds",
        type=int,
        default=1_800,
        help="positive wall-clock limit for each isolated matrix point",
    )
    arguments = parser.parse_args()
    if arguments.worker_timeout_seconds <= 0:
        parser.error("--worker-timeout-seconds must be positive")
    if arguments.matrix:
        return _run_matrix(arguments)
    if arguments.mode is None or arguments.count is None or arguments.output is None:
        parser.error("point mode requires --mode, --count, and --output")
    if arguments.count <= 0:
        parser.error("--count must be positive")
    if arguments.output.exists() and not arguments.force:
        parser.error(f"refusing to overwrite {arguments.output}; pass --force")

    started = time.perf_counter()
    definition = _program(arguments.mode, arguments.count)
    compile_started = time.perf_counter()
    build = qlx.compile(definition)
    compile_wall = time.perf_counter() - compile_started
    profile = _artifact_profile(build)
    estimate_started = time.perf_counter()
    estimate = qlx.estimate(build, tier=qlx.estimate.Tier.LOGICAL)
    estimate_wall = time.perf_counter() - estimate_started
    payload = {
        "schema": "qlx.paper.folding-ablation-point/v2",
        "study_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "mode": arguments.mode,
        "repeat_count": arguments.count,
        "source": {
            "qlx_repo": str(arguments.qlx_repo.resolve()),
            "imported_qlx": QLX_SOURCE,
            "qlx_git": _git_state(arguments.qlx_repo.resolve()),
        },
        "host": _host_record(),
        "compile_wall_s": compile_wall,
        "logical_estimate_wall_s": estimate_wall,
        "logical_actions": dict(estimate.actions),
        "artifact": profile,
        "process_peak_rss_mib": _peak_rss_mib(),
        "total_wall_s": time.perf_counter() - started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_suffix(arguments.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, arguments.output)
    print(json.dumps({
        "mode": arguments.mode,
        "count": arguments.count,
        "compile_wall_s": compile_wall,
        "mlir_bytes": profile["mlir_bytes"],
        "structural_operations": profile["structural_operation_count"],
        "peak_rss_mib": payload["process_peak_rss_mib"],
        "output": str(arguments.output),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
