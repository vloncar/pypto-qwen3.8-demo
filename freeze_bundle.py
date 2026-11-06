# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Refresh integrity hashes after reviewing intentional benchmark changes."""
import json
from pathlib import Path

from verify_checkpoint import sha256


HERE = Path(__file__).resolve().parent
NAMES = (
    'audit.py', 'benchmark.py', 'compare.py', 'lease.py', 'run_host.py',
    'run_pypto.py', 'run_vllm.py', 'verify_checkpoint.py', 'test_protocol.py',
    'checkout.py', 'freeze_bundle.py', 'test_packaging.py',
    'run_pypto.sh', 'preflight.sh', 'suite.json', 'checkpoint_manifest.json',
    'candidate_manifest.json', 'protocol.json', 'versions.json',
)


if __name__ == '__main__':
    manifest = {name: sha256(HERE / name) for name in sorted(NAMES)}
    (HERE / 'bundle_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('Pinned', len(manifest), 'benchmark files')
