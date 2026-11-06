# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Portable checkout and stale-source rejection checks; no NPU access."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from audit import verify_candidate


class PackagingTests(unittest.TestCase):
    def test_wrong_revision_and_modified_source_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'model.py').write_text('original')
            import hashlib
            manifest = {'repositories': {'pypto-lib': 'revision'},
                        'source_sha256': {'pypto-lib/model.py': hashlib.sha256(b'original').hexdigest()}}
            (root / 'candidate_manifest.json').write_text(json.dumps(manifest))
            with patch('audit.HERE', root), patch.dict(os.environ, PYPTO_LIB_ROOT=str(root),
                    PYPTO_SERVING_ROOT=str(root)), patch('audit.subprocess.check_output', return_value='revision\n') as git:
                self.assertEqual(verify_candidate()['sources_verified'], 1)
                git.return_value = 'wrong\n'
                with self.assertRaisesRegex(ValueError, 'Wrong source revision'):
                    verify_candidate()
                git.return_value = 'revision\n'
                (root / 'model.py').write_text('changed')
                with self.assertRaisesRegex(ValueError, 'Candidate source changed'):
                    verify_candidate()

    def test_host_uses_config_and_copies_only_portable_payload(self):
        import shutil
        from run_host import DIGEST, HERE, HostRun
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            names = list(json.loads((HERE / 'bundle_manifest.json').read_text())) + ['bundle_manifest.json']
            for name in names:
                shutil.copyfile(HERE / name, root / name)
            (root / 'environment.local.sh').write_text('PRIVATE SENTINEL')
            model = root / 'checkpoint'
            model.mkdir()
            (model / 'quant_model_weights.safetensors.index.json').write_text('{}')
            config = {'dev_container': 'chosen-container', 'dev_python': '/chosen/python',
                      'dev_env': '/chosen/env.sh', 'dev_results_root': '/chosen/results',
                      'dev_checkpoint': '/chosen/checkpoint'}
            def command(argv, **kwargs):
                if argv[:3] == ['docker', 'image', 'inspect']:
                    output = json.dumps([{'RepoDigests': [DIGEST], 'Id': 'image'}])
                elif argv[:2] == ['docker', 'inspect']:
                    output = json.dumps([{'State': {'Running': True}, 'HostConfig': {}, 'Id': 'dev'}])
                else:
                    output = ''
                return subprocess.CompletedProcess(argv, 0, output, '')
            def dev(*argv, **kwargs):
                output = ('task_20261106_000000_123' if argv[0] == 'task-submit' else
                          json.dumps(list(range(16))) if argv[0] == '/chosen/python' else 'OK')
                return subprocess.CompletedProcess(argv, 0, output, '')
            with patch('run_host.HERE', root), patch('run_host.command', side_effect=command), \
                    patch('run_host.os.sched_getaffinity', return_value=set(range(16))):
                host = HostRun(model, config)
                host.dev = Mock(side_effect=dev)
                host.wait_state = Mock(return_value={'physical_device': '0', 'lease_id': host.run_id})
                host.prepare()
                copied = {p.name for p in (host.output / 'bundle').iterdir()}
                self.assertEqual(copied, set(names))
                self.assertNotIn('environment.local.sh', copied)
                holder = json.loads((host.output / 'control/config.json').read_text())
                self.assertEqual(holder['dev_env'], '/chosen/env.sh')
                self.assertEqual(holder['dev_checkpoint'], '/chosen/checkpoint')
                host.dev.assert_any_call('bash', host.lease + '/bundle/preflight.sh',
                                         '/chosen/env.sh', '/chosen/python', timeout=120)

    def test_checkout_refuses_existing_work(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'pypto-lib').mkdir()
            sentinel = root / 'pypto-lib' / 'keep.txt'
            sentinel.write_text('keep')
            result = subprocess.run([sys.executable, str(Path(__file__).with_name('checkout.py')), str(root)],
                                    text=True, capture_output=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn('already contains', result.stderr)
            self.assertEqual(sentinel.read_text(), 'keep')


if __name__ == '__main__':
    unittest.main()
