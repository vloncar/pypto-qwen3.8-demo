# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Matching vLLM-ascend v0.23.0 streaming-engine benchmark (run in the host image)."""
import argparse
import asyncio
import importlib.metadata
import json
import os
from pathlib import Path
import time
import traceback

from benchmark import Event, benchmark, write_json
from audit import cpu_record, initial_result
from verify_checkpoint import verify


async def run(args, suite, result):
    import vllm_ascend.utils as ascend_utils
    ascend_utils.adapt_patch(is_global_patch=False)
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.sampling_params import RequestOutputKind, SamplingParams
    from vllm.v1.engine.async_llm import AsyncLLM

    s = suite['settings']
    kwargs = dict(model=str(args.model), dtype='bfloat16', quantization='ascend',
        trust_remote_code=False, tensor_parallel_size=s['tensor_parallel_size'], enforce_eager=args.variant == 'vllm-eager',
        worker_extension_cls='audit.BenchmarkWorkerAudit',
        max_model_len=s['max_model_len'], max_num_seqs=s['max_num_seqs'],
        max_num_batched_tokens=s['max_num_batched_tokens'],
        enable_prefix_caching=False, enable_chunked_prefill=True,
        gpu_memory_utilization=s['memory_utilization'], seed=s['seed'], generation_config='vllm',
        disable_log_stats=True, enable_log_requests=False, stream_interval=1)
    start = time.perf_counter()
    engine = AsyncLLM.from_engine_args(AsyncEngineArgs(**kwargs))
    result['startup_s_excluded'] = time.perf_counter() - start
    result['engine_config'] = kwargs
    result['resolved_engine_config'] = {
        'block_size': engine.vllm_config.cache_config.block_size,
        'async_scheduling': engine.vllm_config.scheduler_config.async_scheduling,
        'max_num_seqs': engine.vllm_config.scheduler_config.max_num_seqs,
        'max_num_batched_tokens': engine.vllm_config.scheduler_config.max_num_batched_tokens,
        'max_model_len': engine.vllm_config.model_config.max_model_len,
        'enforce_eager': engine.vllm_config.model_config.enforce_eager,
        'compilation_mode': str(engine.vllm_config.compilation_config.mode),
        'graph_mode': str(engine.vllm_config.compilation_config.cudagraph_mode),
        'compilation_config': str(engine.vllm_config.compilation_config),
    }
    try:
        tokenizer = engine.get_tokenizer()
        for prompt in suite['prompts']:
            if tokenizer.encode(prompt['text'], add_special_tokens=False) != prompt['token_ids']:
                raise ValueError(f"Tokenizer mismatch: {prompt['id']}")
        sampling = SamplingParams(temperature=0.0, top_p=1.0, top_k=-1, seed=s['seed'],
            max_tokens=s['max_new_tokens'], min_tokens=s['max_new_tokens'], ignore_eos=True,
            output_kind=RequestOutputKind.DELTA)

        async def stream(prompt, request_id):
            tokens = []
            async for out in engine.generate(prompt['text'], sampling, request_id,
                                            tokenization_kwargs={'add_special_tokens': False}):
                if list(out.prompt_token_ids) != prompt['token_ids']:
                    raise ValueError(f"Engine prompt IDs differ: {prompt['id']}")
                if len(out.outputs) != 1:
                    raise ValueError('Expected one completion')
                completion = out.outputs[0]
                tokens.extend(completion.token_ids)
                yield Event(len(tokens), out.finished, list(tokens) if out.finished else None,
                            completion.finish_reason or '')
        result['worker_audit'] = await engine.collective_rpc('qwen_demo_resource_audit', timeout=60)
        for worker in result['worker_audit']:
            if not worker['device'].startswith('npu:'):
                raise ValueError('vLLM worker is not on the NPU')
        await benchmark(stream, suite, result, args.output)
        result['worker_audit_after'] = await engine.collective_rpc('qwen_demo_resource_audit', timeout=60)
    finally:
        engine.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--lease-id', required=True)
    parser.add_argument('--block-index', type=int, required=True)
    parser.add_argument('--variant', choices=['vllm-default', 'vllm-eager'], required=True)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    suite = json.loads((here / 'suite.json').read_text())
    result = initial_result('vllm-ascend', args, suite)
    try:
        result['checkpoint'] = verify(args.model, here / 'checkpoint_manifest.json')
        if result['checkpoint']['mismatches']:
            raise ValueError('Checkpoint verification failed')
        result['versions'] = {n: importlib.metadata.version(n)
                              for n in ('vllm', 'vllm-ascend', 'torch', 'torch-npu', 'transformers')}
        if not result['versions']['vllm'].startswith('0.23.0'):
            raise ValueError('Expected vLLM 0.23.0')
        os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        for name in list(os.environ):
            if name.startswith('VLLM_PTO'):
                del os.environ[name]
        result['parent_audit'] = cpu_record()
        asyncio.run(run(args, suite, result))
        result['status'] = 'pass'
        print('VLLM_BENCHMARK_PASS', json.dumps(result['summary']), flush=True)
    except BaseException as exc:
        result.update(status='failed', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
        raise
    finally:
        write_json(args.output, result)


if __name__ == '__main__':
    main()
