# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Identical streaming client, timing boundaries, validation, and statistics for both engines."""
import asyncio
from dataclasses import dataclass
import json
import math
from pathlib import Path
import statistics
import time

from verify_checkpoint import sha256


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


@dataclass
class Event:
    count: int
    finished: bool
    token_ids: list | None = None
    finish_reason: str = ''


def stats(values):
    values = sorted(values)
    if not values or not all(math.isfinite(v) for v in values):
        raise ValueError('Missing or non-finite timing samples')
    def percentile(q):
        index = (len(values) - 1) * q
        lo, hi = math.floor(index), math.ceil(index)
        return values[lo] + (values[hi] - values[lo]) * (index - lo)
    return {'n': len(values), 'min': values[0], 'median': statistics.median(values),
            'mean': statistics.mean(values), 'p95': percentile(.95), 'max': values[-1],
            'stdev': statistics.stdev(values) if len(values) > 1 else 0.0}


async def run_round(stream, suite, label):
    semaphore = asyncio.Semaphore(suite['settings']['client_concurrency'])
    round_start = time.perf_counter()

    async def request(prompt):
        async with semaphore:
            start = time.perf_counter()
            first = None
            last_count = 0
            final = None
            events = []
            async for event in stream(prompt, f'{label}-{prompt["id"]}'):
                now = time.perf_counter()
                if event.count < last_count:
                    raise ValueError('Token count moved backwards')
                if event.count > last_count:
                    if first is None:
                        first = now
                    events.append({'elapsed_s': now - start, 'token_count': event.count})
                last_count = event.count
                if event.finished:
                    final = event
                    end = now
            if final is None or first is None:
                raise ValueError(f'Missing token-bearing final stream: {prompt["id"]}')
            return {'id': prompt['id'], 'submitted_s': start - round_start,
                    'ttft_s': first - start, 'latency_s': end - start,
                    'completed_s': end - round_start, 'token_ids': final.token_ids,
                    'completion_tokens': final.count, 'finish_reason': final.finish_reason,
                    'events': events}

    tasks = [asyncio.create_task(request(prompt)) for prompt in suite['prompts']]
    try:
        outputs = await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    elapsed = time.perf_counter() - round_start
    # Validation and printing are outside the round's measured interval.
    for prompt, row in zip(suite['prompts'], outputs, strict=True):
        if row['token_ids'] != prompt['expected_token_ids']:
            raise ValueError(f'Greedy output differs: {prompt["id"]}: {row["token_ids"]}')
        if row['completion_tokens'] != suite['settings']['max_new_tokens']:
            raise ValueError(f'Incomplete output: {prompt["id"]}')
        if not 0 < row['ttft_s'] <= row['latency_s'] <= elapsed:
            raise ValueError('Invalid timing order')
        if row['finish_reason'] not in ('length', 'FINISHED_LENGTH'):
            raise ValueError(f'Unexpected termination: {row["finish_reason"]}')
    count = sum(row['completion_tokens'] for row in outputs)
    return {'round': label, 'wall_s': elapsed, 'output_tokens': count,
            'output_tokens_per_s': count / elapsed, 'outputs': outputs}


def summarize(runs):
    result = {'round_wall_s': stats([r['wall_s'] for r in runs]),
              'output_tokens_per_s': stats([r['output_tokens_per_s'] for r in runs]),
              'aggregate_output_tokens_per_s': sum(r['output_tokens'] for r in runs)
              / sum(r['wall_s'] for r in runs), 'per_prompt': {}}
    for index, output in enumerate(runs[0]['outputs']):
        rows = [r['outputs'][index] for r in runs]
        result['per_prompt'][output['id']] = {
            'ttft_ms': stats([r['ttft_s'] * 1000 for r in rows]),
            'latency_ms': stats([r['latency_s'] * 1000 for r in rows]),
        }
    return result


async def benchmark(stream, suite, result, output):
    result.update(status='running', warmup=[], runs=[])
    write_json(output, result)
    settings = suite['settings']
    for phase, rounds in [('warmup', settings['warmup_rounds']), ('runs', settings['measured_rounds'])]:
        for index in range(rounds):
            row = await asyncio.wait_for(run_round(stream, suite, f'{phase}-{index}'), timeout=300)
            result[phase].append(row)
    result['summary'] = summarize(result['runs'])
    result['repeatable_token_ids'] = True  # Every warmup/measured output matched the frozen baseline.


def source_hashes(here):
    return {p.name: sha256(p) for p in sorted(here.glob('*.py'))}
