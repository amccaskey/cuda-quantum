#!/usr/bin/env python3
"""Correctness-gated CUDA-Q Logical compiler performance benchmark."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import re
import signal
import statistics
import subprocess
import sys
import time
from typing import Any, Callable


PRODUCT_REPO = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class Case:
    name: str
    receipt_name: str
    command: Callable[[Path, Path, Path], list[str]]
    validate: Callable[[dict[str, Any]], dict[str, Any]]


def _positive_number(value: Any, name: str) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value <= 0):
        raise RuntimeError(f"{name} must be a finite positive number")
    return float(value)


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None:
        raise RuntimeError(f"{name} must be a SHA-256 digest")
    return value


def _validate_folding(record: dict[str, Any]) -> dict[str, Any]:
    if record.get("schema") != "qlx.paper.folding-ablation-point/v2":
        raise RuntimeError("folding receipt has an unexpected schema")
    if record.get("mode") != "folded" or record.get("repeat_count") != 16:
        raise RuntimeError("folding receipt does not describe folded count 16")
    if record.get("logical_actions") != {"qlx_standard_h": 16}:
        raise RuntimeError("folding receipt does not contain sixteen H actions")
    artifact = record.get("artifact", {})
    if artifact.get("stage") != "p0" or artifact.get("structural_operation_count") != 8:
        raise RuntimeError("folding receipt lacks the expected compact P0 program")
    if (_digest(artifact.get("content_sha256"), "folding content_sha256") !=
            "sha256:6decda19e70a72660bccf246ddb9f603bca1439fc97eed13fb074def06c8f180"):
        raise RuntimeError("folding receipt differs from the reviewed P0 artifact")
    return {
        "compile_wall_seconds": _positive_number(
            record.get("compile_wall_s"), "compile_wall_s"
        ),
        "study_peak_rss_mib": _positive_number(
            record.get("process_peak_rss_mib"), "process_peak_rss_mib"
        ),
        "logical_estimate_wall_seconds": _positive_number(
            record.get("logical_estimate_wall_s"), "logical_estimate_wall_s"
        ),
        "structural_operations": artifact["structural_operation_count"],
    }


def _validate_rsa_factory(record: dict[str, Any]) -> dict[str, Any]:
    if record.get("schema") != "qlx.paper.rsa2048/v1":
        raise RuntimeError("RSA factory receipt has an unexpected schema")
    if record.get("status") != "passed" or record.get("factory_only") is not True:
        raise RuntimeError("RSA factory correctness checks did not pass")
    factory = record.get("detailed_factory", {})
    if (factory.get("schedule_entries") != 688 or
            factory.get("physical_qubits") != 142808):
        raise RuntimeError(
            "RSA factory receipt does not cover the reviewed P3 workload")
    if (_digest(factory.get("p3_sha256"), "RSA p3_sha256") !=
            "sha256:72f2df65add8617bcbc3d8cc01c26531dc0c2ef42d431d4fd6d8ebfc83630fa0" or
            _digest(factory.get("schedule_sha256"), "RSA schedule_sha256") !=
            "sha256:9fcfb3171d29a026d7f5c64330d9a322464a6c556b1efe3e8f0a323aa17fcc9e"):
        raise RuntimeError("RSA factory receipt differs from the reviewed P3 artifact")
    return {
        "schedule_entries": int(
            _positive_number(factory.get("schedule_entries"), "schedule_entries")
        ),
        "physical_qubits": int(
            _positive_number(factory.get("physical_qubits"), "physical_qubits")
        ),
        "makespan_ns": _positive_number(factory.get("makespan_ns"), "makespan_ns"),
    }


CASES = {
    "folding-16": Case(
        name="folding-16",
        receipt_name="folding-16.json",
        command=lambda worktree, receipt, python: [
            str(python),
            str(worktree / "preview" / "logical" / "benchmarks" /
                "paper_evaluations" / "folding_ablation.py"),
            "--qlx-repo",
            str(worktree),
            "--mode",
            "folded",
            "--count",
            "16",
            "--output",
            str(receipt),
            "--force",
        ],
        validate=_validate_folding,
    ),
    "rsa2048-factory": Case(
        name="rsa2048-factory",
        receipt_name="rsa2048-factory.json",
        command=lambda worktree, receipt, python: [
            str(python),
            str(worktree / "preview" / "logical" / "benchmarks" /
                "paper_evaluations" / "rsa2048.py"),
            "--factory-only",
            "--output",
            str(receipt),
            "--force",
        ],
        validate=_validate_rsa_factory,
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(worktree: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(worktree), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _git_state(worktree: Path) -> dict[str, Any]:
    status = _git(worktree, "status", "--short", "--untracked-files=all")
    tracked_diff = subprocess.run(
        ["git", "-C", str(worktree), "diff", "--binary", "HEAD"],
        check=True,
        capture_output=True,
    ).stdout
    return {
        "path": str(worktree.resolve()),
        "commit": _git(worktree, "rev-parse", "HEAD"),
        "branch": _git(worktree, "branch", "--show-current"),
        "dirty": bool(status),
        "status": status.splitlines(),
        "tracked_diff_sha256": hashlib.sha256(tracked_diff).hexdigest(),
    }


def _install_prefix(worktree: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        prefix = explicit.resolve()
    else:
        cache = worktree / "build" / "preview" / "logical" / "CMakeCache.txt"
        prefix = None
        if cache.is_file():
            for line in cache.read_text(encoding="utf-8").splitlines():
                if line.startswith("CMAKE_INSTALL_PREFIX:"):
                    prefix = Path(line.split("=", maxsplit=1)[1]).resolve()
                    break
        if prefix is None:
            raise FileNotFoundError(
                f"no logical CMake install prefix recorded in {cache}"
            )
    if not (prefix / "cudaq" / "logical" / "__init__.py").is_file():
        raise FileNotFoundError(
            f"{prefix} does not contain an installed cudaq.logical package"
        )
    return prefix


def _cache_value(cache: Path, name: str) -> str:
    for line in cache.read_text(encoding="utf-8").splitlines():
        if line.startswith(name + ":"):
            return line.split("=", maxsplit=1)[1]
    raise RuntimeError(f"{cache} lacks {name}")


def _refresh_source_install(worktree: Path, prefix: Path,
                            output_dir: Path) -> dict[str, Any]:
    """Rebuild and install the checkout being measured before any workload."""

    build_dir = worktree / "build" / "preview" / "logical"
    cache = build_dir / "CMakeCache.txt"
    if not cache.is_file():
        raise RuntimeError(f"source build is not configured: {cache}")
    if (Path(_cache_value(cache, "CMAKE_HOME_DIRECTORY")).resolve() !=
            worktree / "preview" / "logical" or
            Path(_cache_value(cache, "CMAKE_INSTALL_PREFIX")).resolve() != prefix):
        raise RuntimeError("Logical source build or install prefix differs")
    commands = (
        ["cmake", "--build", str(build_dir), "--parallel", "8"],
        ["cmake", "--install", str(build_dir)],
    )
    log = output_dir / "source-build.log"
    with log.open("w", encoding="utf-8") as output:
        for command in commands:
            output.write("$ " + " ".join(command) + "\n")
            output.flush()
            completed = subprocess.run(command, cwd=worktree, stdout=output,
                                       stderr=subprocess.STDOUT, check=False)
            if completed.returncode:
                raise RuntimeError(
                    f"source build failed (exit={completed.returncode}); "
                    f"see {log}")
    return {
        "configured_source": str((worktree / "preview" / "logical").resolve()),
        "build_directory": str(build_dir.resolve()),
        "install_prefix": str(prefix),
        "commands": commands,
        "log": str(log.relative_to(output_dir)),
        "log_sha256": _sha256(log),
        "source_commit": _git(worktree, "rev-parse", "HEAD"),
        "status": "passed",
    }


def _workload_python(worktree: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        executable = explicit.expanduser().absolute()
    else:
        cache = worktree / "build" / "preview" / "logical" / "CMakeCache.txt"
        executable = None
        if cache.is_file():
            for line in cache.read_text(encoding="utf-8").splitlines():
                if line.startswith("Python3_EXECUTABLE:"):
                    # Preserve a virtual-environment symlink. Resolving it to
                    # the base interpreter would drop that environment's
                    # site-packages while retaining only its Python ABI.
                    executable = Path(
                        line.split("=", maxsplit=1)[1]).expanduser().absolute()
                    break
        if executable is None:
            raise FileNotFoundError(
                f"no Python3_EXECUTABLE recorded in {cache}"
            )
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise FileNotFoundError(
            f"workload Python executable is unavailable: {executable}"
        )
    return executable


def _stop_process_group(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
    else:
        process.terminate()
    try:
        process.wait(timeout=5)
        return
    except subprocess.TimeoutExpired:
        pass
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
    else:
        process.kill()
    process.wait()


def _run(
    *,
    command: list[str],
    log_path: Path,
    environment: dict[str, str],
    working_directory: Path,
    timeout_seconds: int,
) -> tuple[float, float, int, bool]:
    started = time.perf_counter()
    timed_out = False
    measurement_path = log_path.with_name("resource-usage.json")
    wrapped_command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--_measure",
        str(measurement_path),
        json.dumps(command),
    ]
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            wrapped_command,
            cwd=working_directory,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=(os.name == "posix"),
        )
        try:
            returncode = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            _stop_process_group(process)
            returncode = process.returncode
    wall_seconds = time.perf_counter() - started
    peak_rss_mib = 0.0
    if measurement_path.is_file():
        measurement = json.loads(measurement_path.read_text(encoding="utf-8"))
        if (measurement.get("schema") !=
                "cudaq.logical.process-resource-usage/v1" or
                measurement.get("returncode") != returncode):
            raise RuntimeError(f"invalid workload resource usage: {measurement_path}")
        wall_seconds = _positive_number(measurement["wall_seconds"],
                                        "wall_seconds")
        peak_rss_mib = _positive_number(measurement["peak_rss_mib"], "peak_rss_mib")
    elif returncode == 0 and not timed_out:
        raise RuntimeError(f"successful workload omitted {measurement_path}")
    return (
        wall_seconds,
        peak_rss_mib,
        returncode,
        timed_out,
    )


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for name in sorted({run["case"] for run in runs}):
        selected = [run for run in runs if run["case"] == name]
        wall = [run["wall_seconds"] for run in selected]
        rss = [run["peak_rss_mib"] for run in selected]
        result[name] = {
            "repetitions": len(selected),
            "wall_seconds": {
                "minimum": min(wall),
                "median": statistics.median(wall),
                "maximum": max(wall),
            },
            "peak_rss_mib": {
                "minimum": min(rss),
                "median": statistics.median(rss),
                "maximum": max(rss),
            },
        }
    return result


def _measure_worker(measurement_path: Path, command_json: str) -> int:
    command = json.loads(command_json)
    if not isinstance(command, list) or not all(
        isinstance(value, str) for value in command
    ):
        raise TypeError("measurement worker command must be a JSON string list")
    started = time.perf_counter()
    completed = subprocess.run(command, check=False)
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    divisor = 1024.0 * 1024.0 if sys.platform == "darwin" else 1024.0
    _atomic_json(
        measurement_path,
        {
            "schema": "cudaq.logical.process-resource-usage/v1",
            "wall_seconds": time.perf_counter() - started,
            "peak_rss_mib": usage.ru_maxrss / divisor,
            "returncode": completed.returncode,
        },
    )
    return completed.returncode


def main() -> int:
    if len(sys.argv) == 4 and sys.argv[1] == "--_measure":
        return _measure_worker(Path(sys.argv[2]), sys.argv[3])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cudaq-worktree",
        type=Path,
        default=PRODUCT_REPO,
        help=("source-built CUDA-Q checkout containing preview/logical; "
              "defaults to the checkout containing this benchmark"),
    )
    parser.add_argument(
        "--cudaq-python-path",
        type=Path,
        help="CMake install prefix; defaults to the logical CMake cache value",
    )
    parser.add_argument(
        "--python-executable",
        type=Path,
        help=("Python used for benchmark workloads; defaults to the "
              "interpreter recorded by logical CMake"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=tuple(CASES),
        default=list(CASES),
    )
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=1_800)
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    if arguments.repetitions <= 0 or arguments.warmups < 0:
        parser.error("--repetitions must be positive and --warmups nonnegative")
    if arguments.timeout <= 0:
        parser.error("--timeout must be positive")

    worktree = arguments.cudaq_worktree.resolve()
    if not (worktree / "preview" / "logical").is_dir():
        parser.error(f"{worktree} does not contain preview/logical")
    workload_python = _workload_python(worktree, arguments.python_executable)
    missing_studies = []
    for case in CASES.values():
        study = Path(
            case.command(worktree, Path("receipt.json"), workload_python)[1])
        if not study.is_file():
            missing_studies.append(str(study))
    if missing_studies:
        parser.error("paper checkout is missing benchmark studies: " +
                     ", ".join(missing_studies))
    install_prefix = _install_prefix(worktree, arguments.cudaq_python_path)
    output_dir = arguments.output_dir.resolve()
    result_path = output_dir / "logical-compiler-performance.json"
    if result_path.exists() and not arguments.force:
        parser.error(f"refusing to overwrite {result_path}; pass --force")
    output_dir.mkdir(parents=True, exist_ok=True)
    source_build = _refresh_source_install(worktree, install_prefix, output_dir)

    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(install_prefix), existing_pythonpath) if part
    )

    runs = []
    for case_name in arguments.cases:
        case = CASES[case_name]
        for iteration in range(-arguments.warmups, arguments.repetitions):
            warmup = iteration < 0
            label = (
                f"warmup-{iteration + arguments.warmups + 1}"
                if warmup
                else f"run-{iteration + 1}"
            )
            run_dir = output_dir / case_name / label
            run_dir.mkdir(parents=True, exist_ok=True)
            receipt = run_dir / case.receipt_name
            log = run_dir / "stdout.log"
            command = case.command(worktree, receipt, workload_python)
            wall, peak_rss, returncode, timed_out = _run(
                command=command,
                log_path=log,
                environment=environment,
                working_directory=worktree,
                timeout_seconds=arguments.timeout,
            )
            if timed_out or returncode:
                raise RuntimeError(
                    f"{case_name} {label} failed (exit={returncode}, "
                    f"timed_out={timed_out}); see {log}"
                )
            record = json.loads(receipt.read_text(encoding="utf-8"))
            correctness = case.validate(record)
            imported = record.get("source", {}).get("imported_qlx", record.get("source", {}))
            module = imported.get("module")
            if (not isinstance(module, str)
                    or not Path(module).resolve().is_relative_to(install_prefix)):
                raise RuntimeError(f"{case_name} imported cudaq.logical outside {install_prefix}")
            if warmup:
                continue
            runs.append(
                {
                    "case": case_name,
                    "repetition": iteration + 1,
                    "command": command,
                    "wall_seconds": wall,
                    "peak_rss_mib": peak_rss,
                    "returncode": returncode,
                    "timed_out": timed_out,
                    "receipt": str(receipt.relative_to(output_dir)),
                    "receipt_sha256": _sha256(receipt),
                    "resource_usage": str(
                        log.with_name("resource-usage.json").relative_to(output_dir)),
                    "resource_usage_sha256": _sha256(
                        log.with_name("resource-usage.json")),
                    "log": str(log.relative_to(output_dir)),
                    "correctness": correctness,
                }
            )
            print(
                f"{case_name} run {iteration + 1}: {wall:.3f} s, "
                f"{peak_rss:.1f} MiB peak RSS",
                flush=True,
            )

    payload = {
        "schema": "cudaq.logical.compiler-performance/v3",
        "status": "passed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark": str(Path(__file__).resolve()),
        "benchmark_sha256": _sha256(Path(__file__)),
        "python": {
            "runner_executable": sys.executable,
            "workload_executable": str(workload_python),
            "version": platform.python_version(),
            "platform": platform.platform(),
            "pythonpath": environment["PYTHONPATH"],
        },
        "ci_image": environment.get("CUDAQ_DEVDEPS_IMAGE"),
        "configuration": {
            "cases": arguments.cases,
            "output_directory": str(output_dir),
            "repetitions": arguments.repetitions,
            "warmups": arguments.warmups,
            "timeout_seconds": arguments.timeout,
            "memory_measurement": "isolated worker RUSAGE_CHILDREN ru_maxrss",
            "cudaq_install_prefix": str(install_prefix),
        },
        "sources": {
            "cudaq": _git_state(worktree),
            "source_build": source_build,
            "study_sha256": {
                CASES[name].name: _sha256(
                    Path(CASES[name].command(
                        worktree, Path("x"), workload_python)[1]))
                for name in arguments.cases
            },
        },
        "runs": runs,
        "summary": _summary(runs),
    }
    _atomic_json(result_path, payload)
    print(result_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
