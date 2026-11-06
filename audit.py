# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Untimed provenance and actual worker resource audits; no inference-path hooks."""
import json
import importlib.metadata
import os
from pathlib import Path
import platform
import sys
import subprocess

from benchmark import source_hashes, write_json
from verify_checkpoint import sha256


HERE = Path(__file__).resolve().parent


def verify_bundle():
    manifest = json.loads((HERE / 'bundle_manifest.json').read_text())
    for name, expected in manifest.items():
        if sha256(HERE / name) != expected:
            raise ValueError('Bundle changed: ' + name)
    return sha256(HERE / 'bundle_manifest.json')


def verify_candidate():
    manifest = json.loads((HERE / 'candidate_manifest.json').read_text())
    roots = {'pypto-lib': Path(os.environ['PYPTO_LIB_ROOT']),
             'pypto-serving': Path(os.environ['PYPTO_SERVING_ROOT'])}
    for name, expected in manifest['repositories'].items():
        actual = subprocess.check_output(['git', '-C', str(roots[name]), 'rev-parse', 'HEAD'],
                                         text=True).strip()
        if actual != expected:
            raise ValueError('Wrong source revision: ' + name)
    for name, expected in manifest['source_sha256'].items():
        repository, relative = name.split('/', 1)
        if sha256(roots[repository] / relative) != expected:
            raise ValueError('Candidate source changed: ' + name)
    return {'manifest_sha256': sha256(HERE / 'candidate_manifest.json'),
            'sources_verified': len(manifest['source_sha256']),
            'repositories': manifest['repositories']}


def cpu_record():
    import torch
    return {'pid': os.getpid(), 'cpu_affinity': sorted(os.sched_getaffinity(0)),
            'torch_num_threads': torch.get_num_threads(),
            'torch_num_interop_threads': torch.get_num_interop_threads(),
            'mkldnn_enabled': torch.backends.mkldnn.enabled,
            'env': {k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                'OPENBLAS_NUM_THREADS', 'ASCEND_RT_VISIBLE_DEVICES', 'VLLM_WORKER_MULTIPROC_METHOD')},
            'architecture': platform.machine()}


def verify_toolchain():
    from pypto.backend._ptoas_locate import check_ptoas_version, find_ptoas_binary
    from pypto.runtime.runtime_pin import check_runtime_pin

    versions = json.loads((HERE / 'versions.json').read_text())
    root = Path(os.environ['PYPTO_ROOT'])
    for path, expected in ((root, versions['repositories']['pypto']['commit']),
                           (root / 'runtime', versions['runtime_commit'])):
        actual = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
        if actual != expected:
            raise ValueError('Toolchain checkout differs: ' + str(path))
    if (root / 'runtime/pto_isa.pin').read_text().strip() != versions['pto_isa_commit']:
        raise ValueError('PTO ISA pin differs')
    installed = check_runtime_pin().installed
    if installed != versions['runtime_commit']:
        raise ValueError('Installed runtime build stamp differs')
    binary = find_ptoas_binary()
    if not binary:
        raise ValueError('PTOAS not found; set PTOAS_ROOT')
    assembler = check_ptoas_version(binary).strip()
    if assembler != 'ptoas ' + versions['ptoas']:
        raise ValueError('PTOAS release differs: ' + assembler)
    return {'pypto_commit': versions['repositories']['pypto']['commit'],
            'runtime_build_commit': installed, 'ptoas': assembler,
            'packages': {name: importlib.metadata.version(name) for name in
                         ('torch', 'numpy', 'transformers', 'safetensors')}}


def initial_result(engine, args, suite):
    bundle_hash = verify_bundle()
    allowed = sorted(int(x) for x in os.environ['QWEN_DEMO_CPU_LIST'].split(','))
    os.sched_setaffinity(0, allowed)
    if os.environ.get('ASCEND_RT_VISIBLE_DEVICES') != '0':
        raise ValueError('Expected reserved physical card 0')
    return {'schema_version': 2, 'engine': engine, 'variant': args.variant,
            'block_index': args.block_index, 'lease_id': args.lease_id,
            'status': 'starting', 'suite_sha256': sha256(HERE / 'suite.json'),
            'suite': suite, 'protocol': json.loads((HERE / 'protocol.json').read_text()),
            'protocol_sha256': sha256(HERE / 'protocol.json'),
            'bundle_manifest_sha256': bundle_hash,
            'runner_sources_sha256': source_hashes(HERE),
            'physical_device': os.environ['ASCEND_RT_VISIBLE_DEVICES'],
            'allowed_cpus': allowed, 'python': sys.version, 'command': sys.argv}


def install_pypto_worker_audit():
    from pypto_serving.serving.server.serving_worker import WorkerProcess
    original = WorkerProcess.init_device_and_model

    def audited_init(self):
        pages = original(self)
        record = cpu_record()
        record.update(executor=type(self.executor).__name__,
                      platform=self.config.platform,
                      logical_devices=list(self.config.worker_device_ids()))
        if record['torch_num_threads'] != 16:
            raise ValueError('PyPTO worker did not inherit 16 CPU threads')
        write_json(Path(os.environ['QWEN_DEMO_WORKER_AUDIT']), record)
        return pages

    WorkerProcess.init_device_and_model = audited_init


class BenchmarkWorkerAudit:
    """vLLM's supported worker extension, called only before and after timing."""

    def qwen_demo_resource_audit(self):
        record = cpu_record()
        record['device'] = str(self.device)
        record['parameter_devices'] = sorted({str(p.device) for p in self.get_model().parameters()})
        return record


if __name__ == '__main__':
    print(json.dumps({'bundle': verify_bundle(), 'candidate': verify_candidate(),
                      'toolchain': verify_toolchain()}, indent=2))
