# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Benchmark the verified Qwen3.8 model through PyPTO's streaming serving engine."""
import argparse
import asyncio
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import time
import traceback

from benchmark import Event, benchmark, write_json
from audit import cpu_record, initial_result, install_pypto_worker_audit, verify_candidate, verify_toolchain
from verify_checkpoint import verify

install_pypto_worker_audit()


async def run(args, suite, result):
    from pypto_serving import AsyncLLMEngine, GenerateConfig
    from pypto_serving.cli.main import build_parser, build_serving_engine_config
    from pypto_serving.model.tokenizer import load_tokenizer

    s = suite['settings']
    cli = build_parser().parse_args([
        '--model', str(args.model), '--platform', 'a2a3', '--device', str(args.device),
        '--max-model-len', str(s['max_model_len']), '--max-num-seqs', str(s['max_num_seqs']),
        '--max-num-batched-tokens', str(s['max_num_batched_tokens']),
        '--block-size', str(s['pypto_block_size']), '--no-enable-prefix-caching',
        '--npu-memory-utilization', str(s['memory_utilization']),
    ])
    config = build_serving_engine_config(cli)
    args.compiled_dir.mkdir(parents=True, exist_ok=False)
    os.environ['PYPTO_PROG_BUILD_DIR'] = str(args.compiled_dir.resolve())
    config.executor_kwargs['use_compile_cache'] = False
    tokenizer = load_tokenizer(args.model)
    for prompt in suite['prompts']:
        if tokenizer.encode(prompt['text']) != prompt['token_ids']:
            raise ValueError(f"Tokenizer mismatch: {prompt['id']}")
    generation = GenerateConfig(max_new_tokens=s['max_new_tokens'], temperature=0.0,
                                top_p=1.0, seed=s['seed'], stream=True, ignore_eos=True)
    engine = AsyncLLMEngine(config=config, tokenizer=tokenizer)
    result['engine_config'] = {'executor': config.executor_cls,
        'async_scheduling': config.resolve_async_scheduling(),
        'homogeneous_prefill_decode': config.resolve_runtime_config().requires_homogeneous_prefill_decode,
        'compiled_dir': str(args.compiled_dir),
        'preserve_prefill_chunks': config.resolve_runtime_config().preserve_prefill_chunks}

    async def stream(prompt, request_id):
        async for out in engine.add_request(request_id, prompt['text'], generation):
            yield Event(out.completion_tokens, out.finished,
                        list(out.token_ids) if out.finished else None, out.finish_reason)
    try:
        start = time.perf_counter()
        await engine.start()
        result['startup_s_excluded'] = time.perf_counter() - start
        result['worker_audit'] = json.loads(Path(os.environ['QWEN_DEMO_WORKER_AUDIT']).read_text())
        if not result['engine_config']['preserve_prefill_chunks']:
            raise ValueError('Chunk preservation not enabled')
        await benchmark(stream, suite, result, args.output)
    finally:
        await engine.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--compiled-dir', type=Path, required=True)
    parser.add_argument('--lease-id', required=True)
    parser.add_argument('--block-index', type=int, required=True)
    parser.add_argument('--variant', choices=['pypto'], required=True)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    suite = json.loads((here / 'suite.json').read_text())
    result = initial_result('pypto-serving', args, suite)
    os.environ['PYTHONWARNINGS'] = ','.join(filter(None, [os.environ.get('PYTHONWARNINGS'),
        'error:Sampler received non-finite logits:UserWarning']))
    logging.basicConfig(level=logging.WARNING)
    try:
        result['candidate_before'] = verify_candidate()
        result['toolchain'] = verify_toolchain()
        import golden
        import pypto
        import pypto_serving
        result['modules'] = {m.__name__: m.__file__ for m in (golden, pypto, pypto_serving)}
        for module, root in ((golden, 'PYPTO_LIB_ROOT'), (pypto_serving, 'PYPTO_SERVING_ROOT')):
            if not Path(module.__file__).resolve().is_relative_to(Path(os.environ[root]).resolve()):
                raise ValueError('Imported module outside pinned checkout: ' + module.__name__)
        print('MODULES', result['modules'], flush=True)
        result['checkpoint'] = verify(args.model, here / 'checkpoint_manifest.json')
        if result['checkpoint']['mismatches']:
            raise ValueError('Checkpoint verification failed')
        result['versions'] = {n: importlib.metadata.version(n)
                              for n in ('torch', 'transformers', 'safetensors')}
        import torch
        torch.set_num_threads(16)
        torch.backends.mkldnn.enabled = False
        result['parent_audit'] = cpu_record()
        asyncio.run(run(args, suite, result))
        result['candidate_after'] = verify_candidate()
        if result['candidate_before'] != result['candidate_after']:
            raise ValueError('Candidate changed during block')
        result['status'] = 'pass'
        print('PYPTO_BENCHMARK_PASS', json.dumps(result['summary']), flush=True)
    except BaseException as exc:
        result.update(status='failed', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
        raise
    finally:
        write_json(args.output, result)


if __name__ == '__main__':
    main()
