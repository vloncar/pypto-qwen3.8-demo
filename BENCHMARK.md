# Benchmark against vLLM-Ascend

## What is measured

This benchmark uses the same streaming client, raw prompts, greedy settings,
checkpoint and physical card for both engines. It times tokenization, queueing,
scheduling, sampling, dispatch and token delivery. Model startup, compilation,
checkpoint hashing, warmups and result writing are excluded.

| Setting | Value |
|---|---|
| Checkpoint | Eco-Tech/Qwen3.8-27B-w8a8, revision and SHA-256 files pinned |
| Device | One reserved Ascend 910B2, physical card 0 |
| Input lengths | 5, 5, 2048 and 2050 tokens; raw text, no chat template |
| Output | Eight tokens/request, greedy, seed 42, EOS stopping disabled |
| Capacity | Concurrency 2; context 4096; token budget 2048; TP=1 |
| Prefix caching | Disabled in both engines |
| Prefill | Chunked; PyPTO preserves complete fused chunks |
| Throughput | 32 output tokens divided by the complete four-request round time |
| TTFT | Client-slot admission, before tokenization, to first token-bearing event |
| Request latency | Same admission point through terminal stream event |
| Transport | In-process streaming engine API, no HTTP |

TTFT includes queueing and dispatch; it is not an isolated prefill kernel timer.
The benchmark reports median round throughput, per-prompt TTFT/latency and p95,
raw samples, individual block summaries, resolved graph modes and worker audits.
PyPTO uses 128-token pages; vLLM retains its automatic hybrid-cache layout and
internal scheduling. These are complete serving-stack comparisons.

The order is **PyPTO → default vLLM → eager vLLM → eager vLLM → default vLLM →
PyPTO**. Each block starts a fresh engine, does five warmup rounds and twenty
measured rounds in one process. Each configuration therefore has forty measured
rounds. Both vendor modes are always reported; eager is a diagnostic control.
The code records whether default vLLM actually enabled compilation/NPU graphs.

## Run from a Docker host

The automated coordinator requires Python 3.10+, Docker, the vLLM image and an
already prepared, running PyPTO development container with `task-submit`.
It reserves physical card 0 for the whole six-block comparison. It does not
install the development container or the site's allocator.

1. Complete [setup](SETUP.md) inside the development container and verify the
   checkpoint. Keep its environment and source revisions unchanged during runs.
2. Copy this companion repository onto the **Docker host**. It needs only the
   Python standard library there. A checked-out copy or `docker cp` both work.
3. Copy `config.example.json` to `config.local.json` on the host and fill in
   the development container's name, Python executable, environment file,
   checkpoint path and result directory. All five paths/names refer to the
   **container**. The checkpoint argument below is the **host** path.
4. Make the pinned image available and launch:

```bash
cd /path/to/pypto-qwen3.8-demo
# Download once, if it is not already installed:
docker pull quay.io/ascend/vllm-ascend:v0.23.0

python3 run_host.py /path/on/host/to/models/qwen/checkpoint \
  --config config.local.json
```

The launcher checks the image digest in `versions.json` and refuses a moved tag.
It mounts only physical card 0 and the Ascend driver interfaces, with 32 GiB of
vendor-container shared memory. Host paths under `/usr/local/Ascend` must match
the installation; adapt the mount list explicitly if your host differs.
The model and scripts are mounted read-only. Results are written separately.

The launcher takes the intersection of the host/controller and development
container CPU sets and gives vLLM the same Docker CPU quota, shares, memory limit
and NUMA restriction where configured. Both launch environments use
`OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=16`. Workers' actual Torch
thread counts, affinities and NPU devices are audited outside timing. vLLM's
own internal CPU-binding and thread policy is retained. Shared host CPU activity
can still introduce variation; this is not an exclusive CPU reservation.

PyPTO compiles into a fresh directory in each block. Allow time for two PyPTO
compilations, six model initializations and vendor graph capture. The block
limit is one hour; queue wait is separately limited to one hour. Status updates
print every thirty seconds. Do not start a second NPU workload on the card.

## Results and checks

The host creates `results-<timestamp>-<pid>/` beside the scripts and copies it
back into `<dev_results_root>/returned/` in the development container.

- `REPORT.md`: headline medians and completeness status.
- `comparison.json`: both vendor comparisons, per-prompt timings and block variation.
- `block-*/results.json`: every measured and warmup sample and generated token ID.
- `session.json`, `lease-done.json`, logs: allocation, ordering, image and cleanup evidence.
- `bundle/`: the exact portable benchmark inputs used for that run.

`compare.py` independently checks timing arithmetic, expected IDs, all rounds,
source/checkpoint/fixture hashes, CPU allocation, worker placement, sequential
execution and complete cleanup. Run it again against the preserved bundle if
you have since changed your working copy:

```bash
python3 /path/to/results-.../bundle/compare.py /path/to/results-...
```

A failed block makes the comparison incomplete. Passing eager mode does not
replace a failed default mode. Inspect `resolved_engine_config` and vendor
capture logs before describing a run as graph-enabled. Do not combine an old
vendor number with a new PyPTO number into a claimed matched comparison.

The frozen expected outputs check exact generated token IDs on four prompts.
They do not compare full vendor logits or establish broad language quality.
Two temporally balanced blocks reduce linear drift but do not establish a
sub-percent improvement or peak/saturation throughput.

## Recovery

Ctrl+C stops the named vendor container before releasing the allocator grant,
and stops the PyPTO child process group. Partial results are retained.
If Docker communication fails, cleanup cannot be confirmed and the reservation
may stay held. If the host disappears, the holder retains the grant while a
vendor workload might still be running.

Inspect the **specific** vendor container and task ID recorded in `session.json`.
Stop that container first. After confirming that no benchmark work remains,
use `docker exec <dev-container> task-submit --status <task-id>` and your site's
`--kill`/`--cancel` recovery commands for that exact task. Do not release a grant
while its vendor container is still running.

## Host-only verification

```bash
python3 -m unittest -v test_protocol.py test_packaging.py
```

These tests cover timing/concurrency, incorrect-output rejection, incomplete
comparisons, changed sources, lost reservations and child cleanup. They do not
replace a full host Docker run. The portable coordinator has not yet completed
a new six-block Docker run; its predecessor produced the historical results.
