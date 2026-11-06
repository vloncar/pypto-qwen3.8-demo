# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Clone the pinned source revisions without changing existing checkouts."""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--lib-url', help='Override the unpublished pypto-lib URL with a local checkout')
    parser.add_argument('--serving-url', help='Override the unpublished serving URL with a local checkout')
    parser.add_argument('--compiler-url', help='Optional local compiler source')
    parser.add_argument('--no-submodules', action='store_true', help='Checkout validation only; not a runnable install')
    args = parser.parse_args()
    versions = json.loads(Path(__file__).with_name('versions.json').read_text())
    destination = args.destination.expanduser().resolve()
    repositories = versions['repositories']
    if any((destination / name).exists() for name in repositories):
        parser.error('Destination already contains a target repository; choose an empty directory')
    overrides = {'pypto-lib': args.lib_url, 'pypto-serving': args.serving_url, 'pypto': args.compiler_url}
    destination.mkdir(parents=True, exist_ok=True)
    for name, info in repositories.items():
        target = destination / name
        subprocess.run(['git', 'clone', '--no-checkout', overrides[name] or info['url'], str(target)], check=True)
        subprocess.run(['git', '-C', str(target), 'checkout', '--detach', info['commit']], check=True)
        if not args.no_submodules and name in ('pypto', 'pypto-serving'):
            command = ['git', '-C', str(target), 'submodule', 'update', '--init', '--recursive']
            if name == 'pypto-serving':
                command += ['--', 'pypto-lib']
            subprocess.run(command, check=True)
    print('Pinned sources ready in', destination)


if __name__ == '__main__':
    main()
