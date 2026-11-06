# Reference results and validation scope

The latest matched vendor comparison, on **5 November 2026**, used a single
Ascend 910B2 and the exact workload in [BENCHMARK.md](BENCHMARK.md). Five warmups
and twenty measured rounds per block, two blocks per mode, gave forty measured
rounds per configuration. The metric is the median of output tokens divided by
complete four-request round duration, including input processing.

| Engine | Median output tokens/s | Block 1 / block 2 |
|---|---:|---:|
| PyPTO | 12.050310 | 12.066006 / 12.041306 |
| vLLM-Ascend 0.23.0, default | 12.211804 | 12.333565 / 11.796778 |
| vLLM-Ascend 0.23.0, eager | 6.155393 | 6.255977 / 6.005122 |

PyPTO's pooled median was **1.32% below default vLLM**. The ordering changed
between temporal halves: PyPTO trailed by 2.17% in one and led by 2.07% in the
other. Default vLLM's block medians differed by 4.35%, versus 0.20% for PyPTO.
The defensible description is **near parity on this bounded workload**, with
neither a repeatable PyPTO win nor formal equivalence established.

Default vLLM resolved to `VLLM_COMPILE` and `FULL_AND_PIECEWISE` NPU graphs;
both logs showed graph capture completing. Eager disabled compilation and
graphs. The approximately 1.96× ratio against eager is a separate control and
should not be presented as the overall comparison against vLLM.

| Prompt length | PyPTO TTFT, ms | Default vLLM TTFT, ms | PyPTO latency, ms | Default vLLM latency, ms |
|---|---:|---:|---:|---:|
| 5, fact | 270.15 | 859.92 | 663.75 | 1067.01 |
| 5, arithmetic | 269.74 | 871.03 | 663.43 | 1091.52 |
| 2048 | 705.74 | 873.39 | 1884.87 | 1495.95 |
| 2050 | 1594.30 | 1308.02 | 1984.41 | 1527.13 |

These are medians. TTFT includes scheduling, prompt processing and output
delivery. It does not isolate prefill arithmetic, and the 2050-token tail is
slower on PyPTO. Default vLLM completes the longer requests sooner.

## Correctness

All **4,800 generated IDs across 600 requests**, including warmups, matched the
frozen greedy output fixture. Every request returned eight tokens. Checkpoint
hashes, source snapshots, NPU placement and sequential allocation were audited.
This is token equivalence on four prompts, not full-logit vendor equivalence,
long-generation validation or a language-quality benchmark.

The subsequent interface cleanup passed 623 serving unit tests, 16 model
contract tests, seven native program builds and an NPU regression with 368
generated IDs, 48 complete logit vectors, and eight decode/eight prefill cache
cases. Generated C++ was byte-identical across 756 files before and after that
cleanup. Those logit comparisons are **between PyPTO paths**, not with vLLM.
They cover resumed/packed prefill, owner reordering, changing batches, state-slot
reuse and untouched cache storage.

## Relationship to this preview

The recorded comparison predates the final interface cleanup and a subsequent
small tile-tuning change. The latter improved a separate internal measurement
from 12.058150 to 12.085138 tokens/s (0.22%); it has no fresh matched vendor run.
The current preview includes that retained tuning. Do not describe the table
above as a new measurement of the exact preview commits, or combine the later
PyPTO number with the earlier vendor number to claim a win.

[reference-results.json](reference-results.json) preserves the historical
warmup/measured timing samples and token IDs, with machine paths and process
metadata omitted. It also records hashes of the original raw block files.
The snapshot identifiers preserve provenance; the new runner's integrity checks
apply to new results, not this historical export. New runs always write into
separate ignored directories and do not overwrite this reference.
