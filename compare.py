# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Independently validate every sample and the single-grant balanced protocol."""
import argparse
import json
import math
from pathlib import Path

from audit import HERE, verify_bundle
from benchmark import source_hashes, summarize, write_json
from verify_checkpoint import sha256


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate(result):
    require(result['status'] == 'pass' and result['repeatable_token_ids'], 'Benchmark did not pass')
    suite = result['suite']
    require(suite == json.loads((HERE / 'suite.json').read_text()), 'Suite differs')
    require(result['suite_sha256'] == sha256(HERE / 'suite.json'), 'Suite hash differs')
    require(result['protocol'] == json.loads((HERE / 'protocol.json').read_text()), 'Protocol differs')
    require(result['protocol_sha256'] == sha256(HERE / 'protocol.json'), 'Protocol hash differs')
    require(result['runner_sources_sha256'] == source_hashes(HERE), 'Runner sources differ')
    require(result['bundle_manifest_sha256'] == sha256(HERE / 'bundle_manifest.json'), 'Bundle differs')
    require(result['physical_device'] == '0', 'Physical device differs')
    manifest = json.loads((HERE / 'checkpoint_manifest.json').read_text())
    require(result['checkpoint'] == {
        'repo': manifest['repo'], 'revision': manifest['revision'], 'mismatches': [],
        'verified_sha256': {k: v['sha256'] for k, v in manifest['files'].items()}}, 'Checkpoint differs')
    for key, count in [('warmup', suite['settings']['warmup_rounds']),
                       ('runs', suite['settings']['measured_rounds'])]:
        require(len(result[key]) == count, 'Incomplete rounds')
        for run in result[key]:
            require(len(run['outputs']) == len(suite['prompts']), 'Missing request output')
            total = 0
            require(math.isfinite(run['wall_s']) and run['wall_s'] > 0, 'Invalid round time')
            for row, prompt in zip(run['outputs'], suite['prompts'], strict=True):
                require(row['id'] == prompt['id'] and row['token_ids'] == prompt['expected_token_ids'],
                        'Output IDs or request order differs')
                require(row['completion_tokens'] == suite['settings']['max_new_tokens'], 'Incomplete tokens')
                require(0 <= row['submitted_s'] < row['completed_s'] <= run['wall_s'], 'Invalid submission time')
                require(0 < row['ttft_s'] <= row['latency_s'] <= run['wall_s'], 'Invalid request timing')
                require(math.isclose(row['submitted_s'] + row['latency_s'], row['completed_s'], abs_tol=1e-8),
                        'Completion time differs')
                require(row['finish_reason'] in ('length', 'FINISHED_LENGTH'), 'Unexpected finish reason')
                require(bool(row['events']), 'No token events')
                previous_count, previous_time = 0, 0
                for event in row['events']:
                    require(previous_count < event['token_count'] <= row['completion_tokens'], 'Invalid token event')
                    require(previous_time <= event['elapsed_s'] <= row['latency_s'], 'Invalid event timing')
                    previous_count, previous_time = event['token_count'], event['elapsed_s']
                require(previous_count == row['completion_tokens'], 'Missing final tokens')
                require(row['events'][0]['elapsed_s'] == row['ttft_s'], 'TTFT differs from first token event')
                total += row['completion_tokens']
            require(run['output_tokens'] == total and run['output_tokens_per_s'] == total / run['wall_s'],
                    'Throughput denominator differs')
    require(result['summary'] == summarize(result['runs']), 'Summary differs from raw measurements')
    require(result['allowed_cpus'] == result['parent_audit']['cpu_affinity'], 'Parent CPU allocation differs')
    audits = [result['parent_audit']]
    # Both launch environments are fixed. Retain vendor-internal thread policy
    # if it changes its own worker's environment after initialization.
    fixed_thread_audits = [result['parent_audit']]
    if result['variant'] == 'pypto':
        require(result['engine'] == 'pypto-serving', 'Wrong engine')
        candidate = json.loads((HERE / 'candidate_manifest.json').read_text())
        require(result['candidate_before'] == result['candidate_after'] == {
            'manifest_sha256': sha256(HERE / 'candidate_manifest.json'),
            'sources_verified': len(candidate['source_sha256']),
            'repositories': candidate['repositories']}, 'Candidate verification differs')
        require(result['engine_config']['preserve_prefill_chunks'], 'Chunk preservation not active')
        require(result['worker_audit']['torch_num_threads'] == 16, 'PyPTO worker threads differ')
        require(result['worker_audit']['executor'] == 'Qwen38PyptoExecutor', 'Wrong PyPTO executor')
        require(result['worker_audit']['platform'] == 'a2a3', 'Wrong PyPTO platform')
        audits.append(result['worker_audit'])
        fixed_thread_audits.append(result['worker_audit'])
    else:
        require(result['engine'] == 'vllm-ascend', 'Wrong engine')
        require(result['resolved_engine_config']['enforce_eager'] == (result['variant'] == 'vllm-eager'),
                'Unexpected vLLM execution mode')
        require(len(result['worker_audit']) == len(result['worker_audit_after']) == 1, 'Worker count differs')
        for worker in result['worker_audit'] + result['worker_audit_after']:
            require(worker['device'] == 'npu:0', 'Vendor worker device differs')
            require(worker['parameter_devices'] == ['npu:0'], 'Vendor model parameters left the NPU')
            audits.append(worker)
    for audit in audits:
        require(audit['cpu_affinity'] and set(audit['cpu_affinity']) <= set(result['allowed_cpus']),
                'Worker CPU affinity exceeds allocation')
        require(audit['torch_num_threads'] > 0, 'Invalid worker thread count')
    for audit in fixed_thread_audits:
        for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
            require(audit['env'][key] == '16', 'Thread environment differs: ' + key)


def compare(directory):
    verify_bundle()
    session = json.loads((directory / 'session.json').read_text())
    protocol = json.loads((HERE / 'protocol.json').read_text())
    require(session['status'] == 'completed' and session['cleanup_complete'], 'Session incomplete')
    require(session['protocol'] == protocol, 'Session protocol differs')
    require(session['bundle_manifest_sha256'] == sha256(HERE / 'bundle_manifest.json'), 'Session bundle differs')
    require(len(session['blocks']) == len(protocol['order']), 'Incomplete block order')
    lease = json.loads((directory / 'lease-done.json').read_text())
    require(lease['lease_id'] == session['lease_id'] and lease['physical_device'] == '0', 'Lease identity differs')
    require(len(lease['history']) == 6, 'Lease history incomplete')
    valid, failures = {}, []
    previous_end = 0
    for index, (block, variant) in enumerate(zip(session['blocks'], protocol['order'], strict=True)):
        require(block['index'] == index and block['variant'] == variant, 'Block order differs')
        require(block['started_unix_s'] >= previous_end and block['ended_unix_s'] > block['started_unix_s'],
                'Overlapping blocks')
        previous_end = block['ended_unix_s']
        history = lease['history'][index]
        require((history['index'], history['variant'], history['exit_code']) ==
                (index, variant, block['exit_code']), 'Lease history differs')
        try:
            require(block['exit_code'] == 0, 'Launcher failed with exit=' + str(block['exit_code']))
            result = json.loads((directory / block['directory'] / 'results.json').read_text())
            validate(result)
            require(result['lease_id'] == session['lease_id'] and result['block_index'] == index
                    and result['variant'] == variant, 'Result block identity differs')
            require(result['allowed_cpus'] == session['allowed_cpus'], 'CPU allocation differs')
            valid[index] = result
        except (ValueError, KeyError, OSError) as exc:
            failures.append({'block': index, 'variant': variant, 'error': str(exc)})
    summaries = {}
    for variant in dict.fromkeys(protocol['order']):
        indices = [i for i, name in enumerate(protocol['order']) if name == variant]
        if all(i in valid for i in indices):
            summaries[variant] = summarize([row for i in indices for row in valid[i]['runs']])
            summaries[variant]['blocks'] = [{'index': i, 'summary': valid[i]['summary'],
                'worker_audit': valid[i]['worker_audit'],
                'resolved_engine_config': valid[i].get('resolved_engine_config', valid[i]['engine_config'])}
                for i in indices]
    ratios = {}
    if 'pypto' in summaries:
        for variant in ('vllm-default', 'vllm-eager'):
            if variant in summaries:
                ratio = summaries['pypto']['output_tokens_per_s']['median'] / summaries[variant]['output_tokens_per_s']['median']
                vendor_indices = [i for i, name in enumerate(protocol['order']) if name == variant]
                ratios[variant] = {'median_throughput_ratio_pypto_over_vllm': ratio,
                    'pypto_difference_percent': 100 * (ratio - 1),
                    'first_half_ratio': valid[0]['summary']['output_tokens_per_s']['median'] /
                                        valid[vendor_indices[0]]['summary']['output_tokens_per_s']['median'],
                    'second_half_ratio': valid[5]['summary']['output_tokens_per_s']['median'] /
                                         valid[vendor_indices[1]]['summary']['output_tokens_per_s']['median']}
    return {'status': 'pass' if not failures else 'incomplete',
            'kind': 'same_card_single_grant_balanced_six_block_engine_comparison',
            'lease_id': session['lease_id'], 'task_id': session['task_id'], 'physical_device': '0',
            'failures': failures, 'summaries': summaries, 'ratios': ratios,
            'validated_output_ids': sum((len(v['warmup']) + len(v['runs'])) * 32 for v in valid.values()),
            'limitations': ['Four fixed prompts, eight outputs each, concurrency two; not peak throughput.',
                'Fresh engine processes per block; startup/compilation/checkpoint verification excluded.',
                'Same CPU allocation and thread environment; internal engine thread/affinity policies are audited.',
                'Two blocks per mode control linear drift but do not establish a broad or sub-percent win.',
                'A failed default-mode run does not justify a best-vendor-performance claim.']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    args = parser.parse_args()
    result = compare(args.results)
    write_json(args.results / 'comparison.json', result)
    lines = ['# Qwen3.8 serving comparison', '', 'Status: ' + result['status'], '',
             '| Configuration | Median output tok/s | Median suite seconds |',
             '|---|---:|---:|']
    for variant, summary in result['summaries'].items():
        lines.append(f"| {variant} | {summary['output_tokens_per_s']['median']:.6f} | {summary['round_wall_s']['median']:.6f} |")
    lines += ['', 'All request TTFT, latency, p95, variability, worker settings, and raw samples are in comparison.json and block results.', '']
    lines += ['- ' + text for text in result['limitations']]
    if result['failures']:
        lines += ['', 'Failed blocks:', ''] + ['- ' + str(f) for f in result['failures']]
    text = '\n'.join(lines) + '\n'
    (args.results / 'REPORT.md').write_text(text)
    print(text)
    raise SystemExit(0 if result['status'] == 'pass' else 1)
