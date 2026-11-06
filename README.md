# Qwen3.8-27B on PyPTO

A reproducible preview of **Qwen3.8-27B W8A8 text generation on Ascend**, using
PyPTO kernels for the complete 64-layer hybrid Gated DeltaNet/full-attention
model, from embedding through prefill, decode and vocabulary projection.
The CPU handles weight preparation, request metadata, scheduling and sampling.
Model arithmetic and cache updates execute on the NPU.

The latest matched reference comparison reached **12.05 output tokens/s with
PyPTO versus 12.21 with default, graph-enabled vLLM-Ascend** on one Ascend 910B2.
That is near parity on a small, fixed workload, not an established performance
win. All 4,800 generated token IDs matched across 600 requests. See
[results and limitations](RESULTS.md) before quoting these numbers.

**Publication status: local preview.** The branch/repository URLs in
[versions.json](versions.json) are planned destinations and are not yet
published. Use local checkout overrides in the [setup guide](SETUP.md) today.

| Component | Preview branch | Role |
|---|---|---|
| pypto-lib | `preview/qwen3.8-27b` | Kernels, model composition, HOST factories, capability manifest and numerical validation |
| pypto-serving | `preview/qwen3.8-27b` | Checkpoint staging, hybrid caches, request scheduling, runners, CLI and HTTP service |
| This repository | `main` | Pinned setup, generation commands, matched vLLM benchmark and reference results |

Exact source commits, compiler/runtime revisions and vendor image digest are
recorded in [versions.json](versions.json). The serving integration follows the
DeepSeek structure: kernels and public HOST signatures belong to pypto-lib;
serving owns weights, caches and request dispatch. The preview uses an explicit
`PYPTO_LIB_ROOT` checkout because the serving submodule has not been advanced to
the new kernel code. These are runnable preview branches; upstream review and
commit splitting remain separate work.

1. [Set up the pinned sources and run generation](SETUP.md).
2. [Run the matched benchmark against default and eager vLLM](BENCHMARK.md).
3. [Read the measured results and correctness scope](RESULTS.md).
4. [Inspect package validation](VALIDATION.md).
5. [Publish later](PUBLISH.md), then use the [short announcement](SHARE.md).

The supported path is text-only, TP=EP=1, with independent replicas possible.
The complete serving comparison validates concurrency two, context 4096 and a
2048-token prefill budget. Wider batches, other checkpoints, multimodal inputs
and long-generation quality require their own validation.

This repository includes no model weights or generated device binaries.
Source is provided under [CANN Open Software License Agreement v2.0](LICENSE),
consistent with the PyPTO projects. Model weights and vendor dependencies retain
their own licenses.
