# CUDA-Q Logical compiler performance gate

The runner measures wall time and peak resident memory in a fresh process for
each of three end-to-end workloads:

| Case | Work exercised | Receipt sentinel |
| --- | --- | --- |
| `folding-16` | Folded P0 construction and logical estimation | Sixteen H actions represented by a compact P0 program |
| `rsa2048-factory` | Detailed AutoCCZ factory P3 scheduling | Completed factory schedule and authenticated P3 artifact |
| `fermi-p3-l4` | Pinnacle Fermi--Hubbard L=4 P3 projection | Full workload schedule and authenticated P3 artifact |

The three study scripts under `paper_evaluations/` are the benchmark inputs.
They do not import an in-repository application library. The Fermi workload
requires the optional `cudaq-logical[rus]` dependency (`pygridsynth`).

Build and install CUDA-Q from source, and configure `preview/logical` against
that installation. The runner rebuilds and reinstalls Logical from this
checkout immediately before measuring. Run:

```bash
python3 -m pip install 'pygridsynth==2.0.0'
python3 preview/logical/benchmarks/logical_compiler_performance.py \
  --output-dir /tmp/cudaq-logical-performance
python3 preview/logical/benchmarks/performance_gate.py \
  --candidate /tmp/cudaq-logical-performance/logical-compiler-performance.json
```

By default the runner reads the install prefix and Python interpreter from
`build/preview/logical/CMakeCache.txt`. `--cudaq-python-path` and
`--python-executable` override those paths. `--cases`, `--repetitions`, and
`--warmups` are useful for local diagnosis; the gate requires all three cases.

The runner rejects a failed source rebuild, failed workload, missing or
nonfinite memory measurement, wrong installed package, or an incomplete
correctness receipt. The receipts must retain the reviewed P0/P3 artifact
digests; RSA and Fermi also retain
reviewed schedule entry and physical qubit counts. Legitimate changes to those
workloads require reviewing and updating the sentinels. Its JSON summary
includes source and study hashes, source build and workload commands, logs,
raw process-resource records, per-run results, and median measurements. Keep
the summary and its entire output directory together.

`performance_limits.json` has committed upper bounds for both metrics. The
gate reopens each candidate receipt and raw process-resource record, checks
their hashes and values, and requires the candidate's recorded source build,
commit, and study sources to match a clean local checkout. Use
`--expected-commit` to name the required source cut. The gate always applies
the committed bounds. An optional `--baseline` summary also limits
median wall time to 1.5 times and median RSS to 1.25 times the baseline. The
baseline must use the same study source and platform; changes to a benchmark
study need an explicitly approved fresh baseline.

The reference measurements were approximately 0.7 s / 164 MiB for folding,
20 s / 405 MiB for RSA factory, and 302 s / 11.2 GiB for Fermi. The committed
bounds include headroom for build and host variation; they catch substantial
regressions. Compare runs on the same runner class and build configuration.
