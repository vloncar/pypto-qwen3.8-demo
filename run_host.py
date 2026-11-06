# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Host coordinator: six balanced blocks, one allocator reservation, automatic result return."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
import traceback

from audit import verify_bundle
from benchmark import write_json


HERE = Path(__file__).resolve().parent
VERSIONS = json.loads((HERE / 'versions.json').read_text())
IMAGE = VERSIONS['vllm_image']
DIGEST = VERSIONS['vllm_digest']


def command(argv, *, check=True, timeout=60):
    result = subprocess.run([str(x) for x in argv], text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, timeout=timeout)
    if check and result.returncode:
        raise RuntimeError(f'Command failed ({result.returncode}): {shlex.join(map(str, argv))}\n'
                           + result.stdout + result.stderr)
    return result


class HostRun:
    def __init__(self, model, config):
        self.config = config
        self.model = model
        self.run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + str(os.getpid())
        self.output = HERE / ('results-' + self.run_id)
        self.lease = self.config['dev_results_root'] + '/leases/' + self.run_id
        self.container = 'qwen38-compare-' + self.run_id
        self.task_id = None
        self.created_lease = False
        self.granted = False
        self.session = {'status': 'starting', 'lease_id': self.run_id, 'blocks': [],
                        'bundle_manifest_sha256': verify_bundle()}
        self.protocol = json.loads((HERE / 'protocol.json').read_text())
        self.output.mkdir()
        (self.output / 'control').mkdir()

    def dev(self, *args, **kwargs):
        return command(['docker', 'exec', self.config['dev_container'], *args], **kwargs)

    def put_json(self, name, value):
        local = self.output / 'control' / name
        write_json(local, value)
        command(['docker', 'cp', local, f'{self.config["dev_container"]}:{self.lease}/{name}.incoming'])
        self.dev('mv', self.lease + '/' + name + '.incoming', self.lease + '/' + name)

    def state(self):
        script = ('import json,time;from pathlib import Path;'
                  'p=Path(__import__("sys").argv[1]);(p/"host-heartbeat").touch();'
                  'print(json.dumps({"done":(p/"done.json").exists(),'
                  '"state":json.loads((p/"state.json").read_text()) if (p/"state.json").exists() else None}))')
        record = json.loads(self.dev(self.config['dev_python'], '-c', script, self.lease).stdout)
        if record['done']:
            raise RuntimeError('Allocator holder exited; stopping the benchmark')
        state = record['state']
        if state and (time.time() - state['heartbeat_unix_s'] > 20
                      or state['status'] == 'host_lost_holding_reservation'):
            raise RuntimeError('Allocator holder heartbeat was lost')
        return state

    def wait_state(self, index, status, timeout):
        deadline = time.monotonic() + timeout
        progress = 0
        while time.monotonic() < deadline:
            state = self.state()
            if state and state['block_index'] == index and state['status'] == status:
                return state
            if time.monotonic() > progress:
                print('Waiting:', state['status'] if state else 'allocator queue', flush=True)
                progress = time.monotonic() + 30
            time.sleep(2)
        raise TimeoutError('Timed out waiting for ' + status)

    def prepare(self):
        if not (self.model / 'quant_model_weights.safetensors.index.json').is_file():
            raise ValueError('Checkpoint directory is incorrect: ' + str(self.model))
        image = json.loads(command(['docker', 'image', 'inspect', IMAGE]).stdout)[0]
        if DIGEST not in image.get('RepoDigests', []):
            raise ValueError('Image digest differs from the pinned v0.23.0 baseline')
        dev = json.loads(command(['docker', 'inspect', self.config['dev_container']]).stdout)[0]
        if not dev['State']['Running']:
            raise ValueError('Development container is not running')
        resources = dev['HostConfig']
        # Record only resource settings, not container environment or credentials.
        resource_keys = ('CpuPeriod', 'CpuQuota', 'NanoCpus', 'CpuShares', 'CpusetCpus',
                         'CpusetMems', 'Memory', 'MemorySwap', 'ShmSize')
        self.session['dev_resources'] = {k: resources.get(k) for k in resource_keys}
        dev_cpus = json.loads(self.dev(self.config['dev_python'], '-c',
            'import os,json;print(json.dumps(sorted(os.sched_getaffinity(0))))').stdout)
        self.cpus = sorted(set(dev_cpus).intersection(os.sched_getaffinity(0)))
        if len(self.cpus) < 16:
            raise ValueError('Fewer than 16 common allowed CPUs; review resource allocation')
        self.cpu_list = ','.join(map(str, self.cpus))
        self.session.update(allowed_cpus=self.cpus, image_id=image['Id'],
                            image_digest=DIGEST, dev_container_id=dev['Id'], protocol=self.protocol)
        self.cpu_options = ['--cpuset-cpus=' + self.cpu_list]
        if resources.get('NanoCpus', 0):
            self.cpu_options += ['--cpus=' + str(resources['NanoCpus'] / 1e9)]
        elif resources.get('CpuQuota', 0) > 0:
            self.cpu_options += ['--cpu-quota=' + str(resources['CpuQuota']),
                                 '--cpu-period=' + str(resources.get('CpuPeriod') or 100000)]
        if resources.get('CpuShares', 0):
            self.cpu_options += ['--cpu-shares=' + str(resources['CpuShares'])]
        if resources.get('CpusetMems'):
            self.cpu_options += ['--cpuset-mems=' + resources['CpusetMems']]
        if resources.get('Memory', 0):
            self.cpu_options += ['--memory=' + str(resources['Memory'])]
            if resources.get('MemorySwap', 0):
                self.cpu_options += ['--memory-swap=' + str(resources['MemorySwap'])]
        self.dev('mkdir', '-p', self.lease, self.config['dev_results_root'] + '/returned')
        self.created_lease = True
        bundle = self.output / 'bundle'
        bundle.mkdir()
        names = list(json.loads((HERE / 'bundle_manifest.json').read_text())) + ['bundle_manifest.json']
        for name in names:
            target = bundle / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(HERE / name, target)
        command(['docker', 'cp', bundle, f'{self.config["dev_container"]}:{self.lease}/bundle'])
        self.put_json('config.json', dict(self.config, lease_id=self.run_id, cpu_list=self.cpu_list))
        self.dev('touch', self.lease + '/host-heartbeat')
        # Fail before reserving the card if the pinned source snapshot has changed.
        audit = self.dev('bash', self.lease + '/bundle/preflight.sh',
                         self.config['dev_env'], self.config['dev_python'], timeout=120)
        (self.output / 'candidate-preflight.log').write_text(audit.stdout)
        command_text = shlex.join([self.config['dev_python'], self.lease + '/bundle/lease.py', self.lease])
        allocation = self.dev('task-submit', '--device', '0', '--ptoas', 'none',
                              '--timeout', '3600', '--max-time', '0', command_text)
        (self.output / 'allocation.txt').write_text(allocation.stdout + allocation.stderr)
        matches = re.findall(r'task_\d+_\d+_\d+', allocation.stdout)
        if not matches:
            raise ValueError('Allocator did not return a task ID')
        self.task_id = matches[0]
        self.session['task_id'] = self.task_id
        write_json(self.output / 'session.json', self.session)
        print('Waiting for physical card 0:', self.task_id, flush=True)
        state = self.wait_state(-1, 'ready', 3650)
        if state['physical_device'] != '0' or state['lease_id'] != self.run_id:
            raise ValueError('Unexpected grant identity')
        self.granted = True

    def stop_vendor(self):
        found = command(['docker', 'ps', '-aq', '--filter', 'name=^/' + self.container + '$'])
        if not found.stdout.strip():
            return
        command(['docker', 'stop', '-t', '15', self.container], timeout=45)
        state = json.loads(command(['docker', 'inspect', self.container]).stdout)[0]['State']
        if state['Running']:
            raise RuntimeError('Vendor container is still running; reservation must stay held')
        command(['docker', 'rm', self.container])

    def vendor(self, index, variant, output):
        extra = ['--device=/dev/devmm_svm'] if Path('/dev/devmm_svm').exists() else []
        args = ['docker', 'run', '--pull=never', '--name', self.container,
            '--network=host', '--ipc=private', '--shm-size=32g', *self.cpu_options,
            '--device=/dev/davinci0', '--device=/dev/davinci_manager', '--device=/dev/hisi_hdc', *extra,
            '-v', '/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro',
            '-v', '/usr/local/Ascend/firmware:/usr/local/Ascend/firmware:ro',
            '-v', '/usr/local/sbin:/usr/local/sbin:ro',
            '-v', '/etc/ascend_install.info:/etc/ascend_install.info:ro',
            '-v', str(self.model) + ':/model:ro', '-v', str(self.output / 'bundle') + ':/work:ro',
            '-v', str(output) + ':/out',
            '-e', 'ASCEND_RT_VISIBLE_DEVICES=0', '-e', 'VLLM_WORKER_MULTIPROC_METHOD=spawn',
            '-e', 'OMP_NUM_THREADS=16', '-e', 'MKL_NUM_THREADS=16', '-e', 'OPENBLAS_NUM_THREADS=16',
            '-e', 'QWEN_DEMO_CPU_LIST=' + self.cpu_list, '-e', 'PYTHONPATH=/work',
            '-e', 'HF_HUB_OFFLINE=1', '-e', 'TRANSFORMERS_OFFLINE=1', '-e', 'PYTHONUNBUFFERED=1',
            '-e', 'VLLM_LOGGING_LEVEL=WARNING',
            '--entrypoint', '/bin/bash', DIGEST, '-lc',
            'npu-smi info > /out/npu-before.txt; python /work/run_vllm.py "$@"; '
            'rc=$?; npu-smi info > /out/npu-after.txt; exit "$rc"', 'benchmark',
            '--model', '/model', '--output', '/out/results.json', '--lease-id', self.run_id,
            '--block-index', str(index), '--variant', variant]
        write_json(output / 'launch-command.json', args)
        process = None
        try:
            with (output / 'run.log').open('w') as log:
                process = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT)
                deadline = time.monotonic() + self.protocol['block_timeout_s']
                progress = 0
                while process.poll() is None:
                    self.state()
                    if time.monotonic() > deadline:
                        raise TimeoutError('Vendor block exceeded one hour')
                    if time.monotonic() > progress:
                        print(f'Block {index + 1}/6 {variant}: running (log: {output / "run.log"})', flush=True)
                        progress = time.monotonic() + 30
                    time.sleep(2)
                rc = process.returncode
        finally:
            self.stop_vendor()
            if process is not None:
                process.wait(timeout=30)
        return rc

    def run(self):
        self.prepare()
        self.session['status'] = 'running'
        for index, variant in enumerate(self.protocol['order']):
            output = self.output / f'block-{index:02d}-{variant}'
            output.mkdir()
            print(f'Starting block {index + 1}/6: {variant}; 5 warmups + 20 measured rounds', flush=True)
            started = time.time()
            self.put_json('request.json', {'index': index, 'variant': variant})
            if variant == 'pypto':
                state = self.wait_state(index, 'block_done', self.protocol['block_timeout_s'] + 60)
                rc = state['exit_code']
                command(['docker', 'cp', f'{self.config["dev_container"]}:{self.lease}/results/{output.name}/.', output], timeout=120)
            else:
                self.wait_state(index, 'vendor_reserved', 30)
                rc = self.vendor(index, variant, output)
                self.put_json(f'vendor-done-{index:02d}.json', {'exit_code': rc})
                self.wait_state(index, 'block_done', 30)
            row = {'index': index, 'variant': variant, 'directory': output.name, 'exit_code': rc,
                   'started_unix_s': started, 'ended_unix_s': time.time()}
            self.session['blocks'].append(row)
            write_json(self.output / 'session.json', self.session)
            print(f'Block {index + 1}/6 completed, exit={rc}', flush=True)
            if rc:
                print('Failure retained; continuing the other configurations. See', output / 'run.log', flush=True)
        self.session['status'] = 'completed'
        write_json(self.output / 'session.json', self.session)

    def cleanup(self):
        # Confirm host NPU work is stopped before allowing the allocator to release.
        self.stop_vendor()
        if self.created_lease:
            self.dev('touch', self.lease + '/release')
            if self.task_id:
                self.dev('task-submit', '--cancel', self.task_id, check=False)
                deadline = time.monotonic() + 45
                while self.granted and time.monotonic() < deadline:
                    done = self.dev('test', '-f', self.lease + '/done.json', check=False)
                    if done.returncode == 0:
                        command(['docker', 'cp', f'{self.config["dev_container"]}:{self.lease}/done.json', self.output / 'lease-done.json'])
                        break
                    time.sleep(1)
                else:
                    if self.granted:
                        raise RuntimeError('Lease did not acknowledge cleanup; inspect ' + self.task_id)
                log = self.dev('task-submit', '--log', self.task_id, check=False)
                (self.output / 'allocation.log').write_text(log.stdout + log.stderr)
            # Preserve partial PyPTO results too if interrupted inside a block.
            command(['docker', 'cp', f'{self.config["dev_container"]}:{self.lease}/results', self.output / 'pypto-lease-results'],
                    check=False, timeout=120)

    def return_results(self):
        self.dev('mkdir', '-p', self.config['dev_results_root'] + '/returned')
        command(['docker', 'cp', self.output, f'{self.config["dev_container"]}:{self.config["dev_results_root"]}/returned/'], timeout=120)
        print(f'Results copied back to {self.config["dev_container"]}:{self.config["dev_results_root"]}/returned/{self.output.name}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--config', type=Path, required=True, help='Host-side copy of config.local.json')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    required = {'dev_container', 'dev_python', 'dev_env', 'dev_checkpoint', 'dev_results_root'}
    if set(config) != required or any(not isinstance(v, str) or not v for v in config.values()):
        parser.error('Config must contain exactly the nonempty fields in config.example.json')
    if any(not config[k].startswith('/') for k in required - {'dev_container'}):
        parser.error('Container paths must be absolute')
    run = HostRun(args.checkpoint.expanduser().resolve(), config)
    def interrupted(signum, frame):
        raise KeyboardInterrupt('Host received signal ' + str(signum))
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, interrupted)
    rc = 1
    try:
        run.run()
        rc = 0
    except BaseException:
        run.session.update(status='failed', error=traceback.format_exc())
        traceback.print_exc()
    finally:
        # A second terminal interrupt must not skip cleanup.
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, signal.SIG_IGN)
        try:
            run.cleanup()
            run.session['cleanup_complete'] = True
        except BaseException:
            rc = 1
            run.session['cleanup_error'] = traceback.format_exc()
            traceback.print_exc()
            print('Cleanup was not confirmed. Reservation may remain held:', run.task_id, file=sys.stderr)
        write_json(run.output / 'session.json', run.session)
        if rc == 0:
            result = command([sys.executable, HERE / 'compare.py', run.output], check=False)
            (run.output / 'comparison.log').write_text(result.stdout + result.stderr)
            print(result.stdout, end='', flush=True)
            if result.returncode:
                print(result.stderr, file=sys.stderr)
                rc = result.returncode
        (run.output / 'exit-code.txt').write_text(str(rc) + '\n')
        try:
            run.return_results()
        except BaseException:
            traceback.print_exc()
            print('Results remain on the host:', run.output, file=sys.stderr)
            rc = 1
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
