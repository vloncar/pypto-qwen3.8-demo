# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Host-only timing, rejection, reservation, and cleanup tests. No NPU access."""
import asyncio
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

import benchmark
from audit import HERE
from benchmark import Event, run_round, source_hashes, summarize, write_json
from compare import compare, validate
from run_host import HostRun
from verify_checkpoint import sha256


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.suite = json.loads((HERE / 'suite.json').read_text())
        cls.protocol = json.loads((HERE / 'protocol.json').read_text())

    async def stream(self, prompt, request_id):
        await asyncio.sleep(.001)
        yield Event(1, False)
        await asyncio.sleep(.001)
        yield Event(8, True, prompt['expected_token_ids'], 'length')

    def result(self, variant='pypto', index=0):
        row = asyncio.run(run_round(self.stream, self.suite, 'test'))
        checkpoint = json.loads((HERE / 'checkpoint_manifest.json').read_text())
        candidate = json.loads((HERE / 'candidate_manifest.json').read_text())
        audit = {'cpu_affinity': list(range(16)), 'torch_num_threads': 16,
                 'env': {k: '16' for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')},
                 'executor': 'Qwen38PyptoExecutor', 'platform': 'a2a3',
                 'device': 'npu:0', 'parameter_devices': ['npu:0']}
        verified = {'manifest_sha256': sha256(HERE / 'candidate_manifest.json'),
                    'sources_verified': len(candidate['source_sha256']),
                    'repositories': candidate['repositories']}
        result = {'status': 'pass', 'repeatable_token_ids': True, 'variant': variant,
                  'engine': 'pypto-serving' if variant == 'pypto' else 'vllm-ascend',
                  'suite': self.suite, 'suite_sha256': sha256(HERE / 'suite.json'),
                  'protocol': self.protocol, 'protocol_sha256': sha256(HERE / 'protocol.json'),
                  'runner_sources_sha256': source_hashes(HERE),
                  'bundle_manifest_sha256': sha256(HERE / 'bundle_manifest.json'),
                  'physical_device': '0', 'allowed_cpus': list(range(16)),
                  'parent_audit': audit, 'worker_audit': audit if variant == 'pypto' else [audit],
                  'worker_audit_after': [audit], 'candidate_before': verified, 'candidate_after': verified,
                  'engine_config': {'preserve_prefill_chunks': True},
                  'resolved_engine_config': {'enforce_eager': variant == 'vllm-eager'},
                  'lease_id': 'test-lease', 'block_index': index,
                  'checkpoint': {'repo': checkpoint['repo'], 'revision': checkpoint['revision'],
                      'mismatches': [], 'verified_sha256': {k: v['sha256'] for k, v in checkpoint['files'].items()}},
                  'warmup': [copy.deepcopy(row) for _ in range(5)],
                  'runs': [copy.deepcopy(row) for _ in range(20)]}
        result['summary'] = summarize(result['runs'])
        return result

    def test_shared_stream_concurrency_and_ttft(self):
        active = 0
        peak = 0
        async def stream(prompt, request_id):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            yield Event(0, False)
            async for event in self.stream(prompt, request_id):
                yield event
            active -= 1
        row = asyncio.run(run_round(stream, self.suite, 'test'))
        self.assertEqual(peak, 2)
        self.assertEqual(row['output_tokens'], 32)
        self.assertGreater(row['outputs'][2]['submitted_s'], row['outputs'][0]['completed_s'])
        self.assertEqual(row['outputs'][0]['ttft_s'], row['outputs'][0]['events'][0]['elapsed_s'])

    def test_no_writes_or_progress_print_between_rounds(self):
        result = {}
        writes = []
        with patch.object(benchmark, 'write_json', side_effect=lambda p, r: writes.append(
                (len(r['warmup']), len(r['runs'])))), patch('builtins.print') as prints:
            asyncio.run(benchmark.benchmark(self.stream, self.suite, result, Path('/unused')))
        self.assertEqual(writes, [(0, 0)])
        prints.assert_not_called()
        self.assertEqual((len(result['warmup']), len(result['runs'])), (5, 20))

    def test_bad_stream_fails(self):
        async def wrong(prompt, request_id):
            yield Event(8, True, [0] * 8, 'length')
        async def unfinished(prompt, request_id):
            yield Event(1, False)
        for stream in (wrong, unfinished):
            with self.assertRaises(ValueError):
                asyncio.run(run_round(stream, self.suite, 'wrong'))

    def test_rejects_tampered_results(self):
        original = self.result()
        validate(original)
        for key, value in [('physical_device', '1'), ('status', 'failed'), ('runs', []),
                           ('suite_sha256', 'bad'), ('protocol_sha256', 'bad'),
                           ('bundle_manifest_sha256', 'bad'), ('checkpoint', {}),
                           ('candidate_after', {}), ('allowed_cpus', [999])]:
            with self.subTest(key=key):
                changed = copy.deepcopy(original)
                changed[key] = value
                with self.assertRaises((ValueError, KeyError)):
                    validate(changed)
        for key, value in [('token_ids', [0] * 8), ('finish_reason', 'stop'), ('events', []),
                           ('ttft_s', -1), ('completion_tokens', 7), ('submitted_s', -1)]:
            with self.subTest(key=key):
                changed = copy.deepcopy(original)
                changed['runs'][0]['outputs'][0][key] = value
                with self.assertRaises(ValueError):
                    validate(changed)
        for variant in ('vllm-default', 'vllm-eager'):
            result = self.result(variant)
            validate(result)
            result['worker_audit'] = copy.deepcopy(result['worker_audit'])
            result['worker_audit_after'] = copy.deepcopy(result['worker_audit_after'])
            for worker in result['worker_audit'] + result['worker_audit_after']:
                worker['torch_num_threads'] = 1
                worker['env']['OMP_NUM_THREADS'] = '1'
            validate(result)  # Vendor internal defaults are audited, not overwritten.
            result['resolved_engine_config']['enforce_eager'] = not result['resolved_engine_config']['enforce_eager']
            with self.assertRaises(ValueError):
                validate(result)

    def test_full_comparison_and_partial_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            session = {'status': 'completed', 'cleanup_complete': True, 'protocol': self.protocol,
                       'bundle_manifest_sha256': sha256(HERE / 'bundle_manifest.json'),
                       'lease_id': 'test-lease', 'task_id': 'task_test',
                       'allowed_cpus': list(range(16)), 'blocks': []}
            history = []
            for index, variant in enumerate(self.protocol['order']):
                name = f'block-{index:02d}-{variant}'
                write_json(directory / name / 'results.json', self.result(variant, index))
                session['blocks'].append({'index': index, 'variant': variant, 'exit_code': 0,
                    'directory': name, 'started_unix_s': index * 2, 'ended_unix_s': index * 2 + 1})
                history.append({'index': index, 'variant': variant, 'exit_code': 0})
            write_json(directory / 'session.json', session)
            write_json(directory / 'lease-done.json', {'lease_id': 'test-lease', 'physical_device': '0',
                                                      'history': history})
            result = compare(directory)
            self.assertEqual(result['status'], 'pass')
            self.assertEqual(result['validated_output_ids'], 4800)
            self.assertEqual(set(result['ratios']), {'vllm-default', 'vllm-eager'})
            missing = directory / session['blocks'][1]['directory'] / 'results.json'
            missing.unlink()
            result = compare(directory)
            self.assertEqual(result['status'], 'incomplete')
            self.assertNotIn('vllm-default', result['ratios'])
            self.assertIn('vllm-eager', result['ratios'])

    def test_host_never_releases_before_vendor_stop(self):
        host = HostRun.__new__(HostRun)
        host.config = {'dev_container': 'test-container'}
        host.created_lease = True
        host.task_id = None
        host.lease = '/unused'
        host.output = Path('/unused')
        actions = []
        host.stop_vendor = Mock(side_effect=lambda: actions.append('stop_vendor'))
        host.dev = Mock(side_effect=lambda *a, **kw: actions.append(a))
        with patch('run_host.command'):
            host.cleanup()
        self.assertEqual(actions[:2], ['stop_vendor', ('touch', '/unused/release')])
        host.stop_vendor = Mock(side_effect=RuntimeError('Cannot stop vendor'))
        host.dev.reset_mock()
        with self.assertRaises(RuntimeError):
            host.cleanup()
        host.dev.assert_not_called()

    def test_vendor_stops_on_lost_lease(self):
        host = HostRun.__new__(HostRun)
        host.container = 'test'
        host.output = Path('/results/test')
        host.model = Path('/model')
        host.cpu_options = []
        host.cpu_list = '0'
        host.run_id = 'test'
        host.protocol = self.protocol
        host.state = Mock(side_effect=RuntimeError('Lease lost'))
        host.stop_vendor = Mock()
        process = Mock()
        process.poll.return_value = None
        with tempfile.TemporaryDirectory() as temporary, patch('run_host.subprocess.Popen', return_value=process):
            with self.assertRaisesRegex(RuntimeError, 'Lease lost'):
                host.vendor(1, 'vllm-default', Path(temporary))
        host.stop_vendor.assert_called_once()
        process.wait.assert_called_once()


class LeaseTests(unittest.TestCase):
    def test_protocol_and_release_stops_child_group(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            write_json(directory / 'config.json', {'lease_id': 'test', 'cpu_list': '0'})
            (directory / 'host-heartbeat').touch()
            code = ('from pathlib import Path;from lease import serve;import sys;'
                    'serve(Path(sys.argv[1]), [sys.executable,"-c","import time;time.sleep(60)"])')
            env = dict(os.environ, PYTHONPATH=str(HERE), ASCEND_RT_VISIBLE_DEVICES='0')
            process = subprocess.Popen([sys.executable, '-c', code, str(directory)], env=env,
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                self.wait_for(lambda: (directory / 'state.json').exists())
                write_json(directory / 'request.json', {'index': 0, 'variant': 'pypto'})
                self.wait_for(lambda: json.loads((directory / 'state.json').read_text())['status'] == 'pypto_running')
                child = json.loads((directory / 'state.json').read_text())['child_pid']
                (directory / 'release').touch()
                stdout, stderr = process.communicate(timeout=25)
                self.assertEqual(process.returncode, 0, stdout + stderr)
                with self.assertRaises(ProcessLookupError):
                    os.kill(child, 0)
                self.assertTrue((directory / 'done.json').exists())
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.communicate(timeout=25)

    def test_stale_host_keeps_reservation(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            write_json(directory / 'config.json', {'lease_id': 'test', 'cpu_list': '0'})
            (directory / 'host-heartbeat').touch()
            os.utime(directory / 'host-heartbeat', (time.time() - 100, time.time() - 100))
            code = 'from pathlib import Path;from lease import serve;import sys;serve(Path(sys.argv[1]))'
            env = dict(os.environ, PYTHONPATH=str(HERE), ASCEND_RT_VISIBLE_DEVICES='0')
            process = subprocess.Popen([sys.executable, '-c', code, str(directory)], env=env,
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                self.wait_for(lambda: (directory / 'state.json').exists())
                self.assertEqual(json.loads((directory / 'state.json').read_text())['status'],
                                 'host_lost_holding_reservation')
                self.assertIsNone(process.poll())
                (directory / 'release').touch()
                process.communicate(timeout=10)
                self.assertEqual(process.returncode, 0)
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.communicate(timeout=25)

    def wait_for(self, predicate):
        deadline = time.monotonic() + 10
        while not predicate():
            if time.monotonic() > deadline:
                self.fail('Lease test timed out')
            time.sleep(.1)


if __name__ == '__main__':
    unittest.main()
