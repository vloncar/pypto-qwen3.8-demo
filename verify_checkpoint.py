# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Verify the exact pinned files used by the reference comparison."""
import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify(model_dir, manifest_path):
    manifest = json.loads(manifest_path.read_text())
    mismatches = []
    verified = {}
    for name, info in manifest['files'].items():
        path = model_dir / name
        if not path.is_file():
            mismatches.append(f'{name}: missing')
            continue
        if path.stat().st_size != info['bytes']:
            mismatches.append(f'{name}: size differs')
            continue
        actual = sha256(path)
        if actual != info['sha256']:
            mismatches.append(f'{name}: SHA-256 differs')
        else:
            verified[name] = actual
            print('VERIFIED', name, flush=True)
    return {'repo': manifest['repo'], 'revision': manifest['revision'],
            'verified_sha256': verified, 'mismatches': mismatches}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model_dir', type=Path)
    args = parser.parse_args()
    result = verify(args.model_dir, Path(__file__).with_name('checkpoint_manifest.json'))
    print(json.dumps(result, indent=2))
    raise SystemExit(bool(result['mismatches']))
