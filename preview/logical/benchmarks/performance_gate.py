#!/usr/bin/env python3
"""Fail CI on incomplete benchmark evidence or runtime and RSS regressions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import sys

from logical_compiler_performance import CASES, __file__ as BENCHMARK_RUNNER


def finite_positive(value: object, label: str) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value <= 0):
        raise ValueError(f"{label} must be a finite positive number")
    return float(value)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{label} must be a SHA-256 digest")
    return value


def _git_output(worktree: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(worktree), *arguments],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _relative_file(summary: Path, value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} path is missing")
    relative = Path(value)
    directory = summary.resolve().parent
    resolved = (directory / relative).resolve()
    if (relative.is_absolute() or
            not resolved.is_relative_to(directory) or
            not resolved.is_file()):
        raise ValueError(f"{label} is unavailable")
    return resolved


def _cache_value(cache: Path, name: str) -> str:
    for line in cache.read_text(encoding="utf-8").splitlines():
        if line.startswith(name + ":"):
            return line.split("=", maxsplit=1)[1]
    raise ValueError(f"{cache} lacks {name}")


def checked_summary(path: Path, cases: set[str], *, verify_receipts=True,
                    verify_local_source=True, expected_commit=None) -> dict:
    result = json.loads(path.read_text(encoding="utf-8"))
    if (result.get("schema") != "cudaq.logical.compiler-performance/v3"
            or result.get("status") != "passed"):
        raise ValueError(f"{path} is not a passed compiler benchmark")
    summary = result.get("summary", {})
    if set(summary) != cases or set(result.get("configuration", {}).get("cases", [])) != cases:
        raise ValueError(f"{path} must contain exactly {sorted(cases)}")
    runs = result.get("runs", [])
    if {run.get("case") for run in runs} != cases:
        raise ValueError(f"{path} has an unexpected run inventory")
    studies = result.get("sources", {}).get("study_sha256", {})
    if set(studies) != cases:
        raise ValueError(f"{path} has an incomplete study inventory")
    for case, digest in studies.items():
        _digest(digest, f"{case} study source")
    _digest(result.get("benchmark_sha256"), "benchmark runner source")
    if verify_local_source:
        source = result["sources"]["cudaq"]
        worktree = Path(source["path"]).resolve()
        current_commit = _git_output(worktree, "rev-parse", "HEAD")
        if (source.get("commit") != current_commit or
                (expected_commit is not None and
                 current_commit != expected_commit) or
                source.get("dirty") is not False or source.get("status") or
                _git_output(worktree, "status", "--porcelain=v1")):
            raise ValueError(f"{path} does not describe a clean expected source cut")
        if result["benchmark_sha256"] != _sha256(Path(BENCHMARK_RUNNER).read_bytes()):
            raise ValueError(f"{path} benchmark runner source differs")
        source_build = result["sources"].get("source_build", {})
        build_dir = worktree / "build" / "preview" / "logical"
        prefix = Path(result["configuration"]["cudaq_install_prefix"]).resolve()
        if (source_build.get("status") != "passed" or
                source_build.get("source_commit") != current_commit or
                source_build.get("configured_source") !=
                    str(worktree / "preview" / "logical") or
                source_build.get("build_directory") != str(build_dir) or
                source_build.get("install_prefix") != str(prefix) or
                source_build.get("commands") != [
                    ["cmake", "--build", str(build_dir), "--parallel", "8"],
                    ["cmake", "--install", str(build_dir)],
                ]):
            raise ValueError(f"{path} lacks a matching source build")
        cache = build_dir / "CMakeCache.txt"
        if (Path(_cache_value(cache, "CMAKE_HOME_DIRECTORY")).resolve() !=
                worktree / "preview" / "logical" or
                Path(_cache_value(cache, "CMAKE_INSTALL_PREFIX")).resolve() != prefix):
            raise ValueError(f"{path} source build configuration differs")
        build_log = _relative_file(path, source_build.get("log"), "source build log")
        if _digest(source_build.get("log_sha256"), "source build log") != _sha256(
                build_log.read_bytes()):
            raise ValueError(f"{path} source build log differs")
        for case in cases:
            script = Path(CASES[case].command(
                worktree, Path("unused.json"),
                Path(result["python"]["workload_executable"]))[1])
            if studies[case] != _sha256(script.read_bytes()):
                raise ValueError(f"{path}: {case} study source differs")
    for case in cases:
        selected = [run for run in runs if run.get("case") == case]
        if len(selected) != summary[case].get("repetitions") or not selected:
            raise ValueError(f"{path}: {case} has incomplete runs")
        for run in selected:
            if run.get("returncode") != 0 or run.get("timed_out") is not False:
                raise ValueError(f"{path}: {case} did not complete")
            if not isinstance(run.get("correctness"), dict) or not run["correctness"]:
                raise ValueError(f"{path}: {case} lacks correctness evidence")
            receipt_sha256 = _digest(run.get("receipt_sha256"),
                                     f"{case} receipt")
            usage_sha256 = _digest(run.get("resource_usage_sha256"),
                                   f"{case} resource usage")
            if verify_receipts:
                receipt = _relative_file(path, run.get("receipt"),
                                         f"{case} receipt")
                raw = receipt.read_bytes()
                if _sha256(raw) != receipt_sha256:
                    raise ValueError(f"{path}: {case} receipt digest differs")
                record = json.loads(raw)
                expected = CASES[case].validate(record)
                if run["correctness"] != expected:
                    raise ValueError(f"{path}: {case} correctness differs from receipt")
                prefix = Path(result["configuration"]["cudaq_install_prefix"]).resolve()
                imported = record.get("source", {}).get(
                    "imported_qlx", record.get("source", {}))
                module = imported.get("module")
                if (not isinstance(module, str) or
                        not Path(module).resolve().is_relative_to(prefix)):
                    raise ValueError(f"{path}: {case} imported an unrelated install")
                worktree = Path(result["sources"]["cudaq"]["path"]).resolve()
                python = Path(result["python"]["workload_executable"])
                original_output = Path(
                    result["configuration"]["output_directory"]).resolve()
                recorded_receipt = original_output / run["receipt"]
                if run.get("command") != CASES[case].command(
                        worktree, recorded_receipt, python):
                    raise ValueError(f"{path}: {case} command differs")
                usage = _relative_file(path, run.get("resource_usage"),
                                       f"{case} resource usage")
                usage_raw = usage.read_bytes()
                if _sha256(usage_raw) != usage_sha256:
                    raise ValueError(f"{path}: {case} resource usage differs")
                measurement = json.loads(usage_raw)
                if (measurement.get("schema") !=
                        "cudaq.logical.process-resource-usage/v1" or
                        measurement.get("returncode") != run["returncode"] or
                        measurement.get("wall_seconds") != run["wall_seconds"] or
                        measurement.get("peak_rss_mib") != run["peak_rss_mib"]):
                    raise ValueError(f"{path}: {case} measurements differ")
            for field in ("wall_seconds", "peak_rss_mib"):
                finite_positive(run.get(field), f"{case} {field}")
        for field in ("wall_seconds", "peak_rss_mib"):
            median = finite_positive(summary[case][field].get("median"), f"{case} median {field}")
            if median != statistics.median(run[field] for run in selected):
                raise ValueError(f"{path}: {case} {field} median differs from runs")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--limits", type=Path, default=Path(__file__).with_name("performance_limits.json"))
    parser.add_argument("--baseline", type=Path, help="successful target-branch performance artifact")
    parser.add_argument("--expected-commit", help="required candidate source commit")
    arguments = parser.parse_args()
    limits = json.loads(arguments.limits.read_text(encoding="utf-8"))
    if limits.get("schema") != "cudaq.logical.performance-limits/v1":
        parser.error("unknown performance limits schema")
    cases = set(limits["cases"])
    candidate = checked_summary(arguments.candidate, cases,
                                expected_commit=arguments.expected_commit)
    baseline = (checked_summary(arguments.baseline, cases,
                                verify_receipts=False,
                                verify_local_source=False)
                if arguments.baseline else None)
    if baseline:
        if candidate["sources"]["study_sha256"] != baseline["sources"]["study_sha256"]:
            raise ValueError("benchmark study source changed; approve a fresh baseline")
        if candidate["python"]["platform"] != baseline["python"]["platform"]:
            raise ValueError("benchmark baseline uses a different platform")
        if candidate.get("ci_image") != baseline.get("ci_image"):
            raise ValueError("benchmark baseline uses a different CI image")
    failures = []
    for case in sorted(cases):
        for field in ("wall_seconds", "peak_rss_mib"):
            observed = candidate["summary"][case][field]["median"]
            cap = finite_positive(limits["cases"][case][field], f"{case} {field} limit")
            if observed > cap:
                failures.append(f"{case} {field}: {observed:.2f} exceeds {cap:.2f}")
            if baseline:
                previous = baseline["summary"][case][field]["median"]
                ratio = finite_positive(limits["baseline_max_ratio"][field], f"{field} ratio")
                if observed > previous * ratio:
                    failures.append(
                        f"{case} {field}: {observed:.2f} exceeds "
                        f"target-branch {previous:.2f} by more than {ratio:.2f}x")
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    print(f"Performance gate passed for {', '.join(sorted(cases))}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, TypeError, ValueError, RuntimeError,
            subprocess.CalledProcessError, OSError) as error:
        print(f"Invalid performance evidence: {error}", file=sys.stderr)
        raise SystemExit(1) from error
