# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run PyPTO inside the allocator grant; hold it while the host runs vLLM."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback

from audit import verify_bundle
from benchmark import write_json


def stop_group(process):
    """Stop the whole benchmark process group, including spawned workers."""
    if process is None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        process.poll()
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            return
        time.sleep(.2)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=10)


def serve(lease, pypto_command=None):
    config = json.loads((lease / 'config.json').read_text())
    here = Path(__file__).resolve().parent
    protocol = json.loads((here / 'protocol.json').read_text())
    if os.environ.get('ASCEND_RT_VISIBLE_DEVICES') != '0':
        raise ValueError('Allocator did not grant physical card 0')
    state = {'status': 'ready', 'physical_device': '0', 'lease_id': config['lease_id'],
             'block_index': -1, 'pid': os.getpid(), 'history': []}
    process = None
    log = None
    started = None

    def interrupted(signum, frame):
        raise KeyboardInterrupt('Lease received signal ' + str(signum))

    previous = {s: signal.signal(s, interrupted) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        while not (lease / 'release').exists():
            host_age = time.time() - (lease / 'host-heartbeat').stat().st_mtime
            if host_age > 90:
                stop_group(process)
                process = None
                if log:
                    log.close()
                    log = None
                # A foreign Docker workload may still own the NPU. Keep the
                # allocator lock until the host explicitly confirms cleanup.
                state['status'] = 'host_lost_holding_reservation'
            if process is not None:
                rc = process.poll()
                if rc is not None or time.monotonic() - started > protocol['block_timeout_s']:
                    state.update(status='block_done', exit_code=rc if rc is not None else 124)
                    stop_group(process)
                    process = None
                    log.close()
                    log = None
                    state['history'].append({'index': state['block_index'],
                        'variant': 'pypto', 'exit_code': state['exit_code'], 'ended_unix_s': time.time()})
            request_path = lease / 'request.json'
            if request_path.exists() and state['status'] in ('ready', 'block_done'):
                request = json.loads(request_path.read_text())
                index = request['index']
                if index > state['block_index']:
                    if index != state['block_index'] + 1 or request['variant'] != protocol['order'][index]:
                        raise ValueError('Unexpected block order')
                    state.update(block_index=index, variant=request['variant'], started_unix_s=time.time())
                    if request['variant'] == 'pypto':
                        output = lease / 'results' / f"block-{index:02d}-pypto"
                        output.mkdir(parents=True)
                        command = pypto_command or ['bash', str(here / 'run_pypto.sh'), str(output),
                            config['lease_id'], str(index), config['cpu_list'],
                            config['dev_env'], config['dev_checkpoint'], config['dev_python']]
                        log = (output / 'run.log').open('w')
                        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                                   start_new_session=True)
                        started = time.monotonic()
                        state.update(status='pypto_running', child_pid=process.pid)
                    else:
                        state.update(status='vendor_reserved')
            if state['status'] == 'vendor_reserved':
                finished_path = lease / f"vendor-done-{state['block_index']:02d}.json"
                if finished_path.exists():
                    finished = json.loads(finished_path.read_text())
                    state['history'].append({'index': state['block_index'], 'variant': state['variant'],
                        'exit_code': finished['exit_code'], 'ended_unix_s': time.time()})
                    state.update(status='block_done', exit_code=finished['exit_code'])
            state['heartbeat_unix_s'] = time.time()
            write_json(lease / 'state.json', state)
            time.sleep(.5)
    except BaseException:
        state.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        stop_group(process)
        if log:
            log.close()
        state['stopped_unix_s'] = time.time()
        write_json(lease / 'done.json', state)
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('lease', type=Path)
    args = parser.parse_args()
    verify_bundle()
    serve(args.lease)
