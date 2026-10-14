#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Prove the selected NCCL mapping, install prepared transport, then serve."""
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import threading
import tempfile
import time
import urllib.request
sys.dont_write_bytecode=True
_CONTROLLER=None
HEALTH_PATH=Path('/cache/spark-ring-controller-health.json')


def source_map_digest(files):
    return hashlib.sha256(json.dumps(files,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def compact_controller_report(report):
    result=dict(report)
    files=result.pop('tensorfold_source_sha256')
    if not isinstance(files,dict) or not files:raise RuntimeError('Missing controller source map')
    result.update(tensorfold_source_digest_sha256=source_map_digest(files),
                  tensorfold_source_file_count=len(files),
                  engine_manifest_sha256=os.environ.get('SPARK_RING_ENGINE_MANIFEST_SHA256'))
    return result


def process_start_ticks(pid,proc_root=Path('/proc')):
    # The comm field may contain spaces and parentheses. Fields after its last
    # closing parenthesis start at stat field 3; starttime is field 22.
    text=(Path(proc_root)/str(pid)/'stat').read_text()
    fields=text.rsplit(')',1)[1].split()
    if fields[0] in ('Z','X','x'):raise RuntimeError('Controller service process is not live')
    return int(fields[19])


def write_controller_snapshot(report,path=HEALTH_PATH,proc_root=Path('/proc')):
    path=Path(path);pid=os.getpid()
    envelope={'unix_seconds':time.time(),'service_pid':pid,
              'service_start_ticks':process_start_ticks(pid,proc_root),
              'host_boot_id':(Path(proc_root)/'sys/kernel/random/boot_id').read_text().strip(),
              'report':compact_controller_report(report)}
    fd,temporary=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'w') as stream:
            json.dump(envelope,stream,sort_keys=True,separators=(',',':'),allow_nan=False)
            stream.write('\n')
        os.replace(temporary,path)
    finally:
        if os.path.exists(temporary):os.unlink(temporary)
    return envelope


def read_controller_snapshot(path=HEALTH_PATH,proc_root=Path('/proc'),expected_pid=1):
    path=Path(path)
    with path.open() as stream:
        metadata=os.fstat(stream.fileno())
        if metadata.st_mode & 0o777 != 0o600:raise RuntimeError('Controller snapshot mode differs')
        row=json.load(stream)
    pid=row.get('service_pid');ticks=row.get('service_start_ticks');stamp=row.get('unix_seconds')
    if (type(pid) is not int or pid!=expected_pid or type(ticks) is not int or ticks<=0
            or type(stamp) not in (float,int) or not math.isfinite(stamp) or stamp<=0
            or not isinstance(row.get('report'),dict)):
        raise RuntimeError('Controller snapshot envelope differs')
    if (process_start_ticks(pid,proc_root)!=ticks or
            row.get('host_boot_id')!=(Path(proc_root)/'sys/kernel/random/boot_id').read_text().strip()):
        raise RuntimeError('Controller snapshot process/boot identity differs')
    return row


def health_watch(stop,controller,health):
    """Check only host/proxy health; a native fault poisons every owned rank."""
    url='http://'+health['address']+':'+str(health['port'])
    headers={'Authorization':'Bearer '+health['token']}
    next_report=0
    while not stop.wait(2):
        try:
            request=urllib.request.Request(url+'/health',headers=headers)
            with urllib.request.urlopen(request,timeout=3) as response:row=json.load(response)
            if (row.get('plan_id')!=health['plan_id'] or row.get('ready') is not True or
                row.get('all_peers_ready') is not True or row.get('failure') is not None):raise RuntimeError('Four-rank native mesh became unhealthy')
            controller.check_health()
            # Commit the machine receipt before any potentially interleaved
            # human log write. This contains no CUDA query or synchronization.
            snapshot=write_controller_snapshot(controller.report())
            now=time.monotonic()
            if now>=next_report:
                # Host proxy counters only: no CUDA synchronization or GPU stats.
                print('SPARK_RING_ROCENANTE_ACTIVE '+json.dumps(snapshot['report']),flush=True)
                next_report=now+60
        except BaseException as error:
            print('SPARK_RING_NATIVE_FATAL '+json.dumps({'rank':health['rank'],'error':str(error)}),flush=True)
            try:
                request=urllib.request.Request(url+'/fault',data=b'',headers=headers,method='POST')
                with urllib.request.urlopen(request,timeout=2) as response:response.read()
            except Exception:pass
            # No retry or transport fallback after native work has begun.
            os._exit(70)


def main():
    global _CONTROLLER
    library=Path(os.environ['TF_NCCL_LIB']).resolve(strict=True)
    expected=os.environ['SPARK_RING_NCCL_SHA256'];actual=hashlib.sha256(library.read_bytes()).hexdigest()
    if actual!=expected:raise RuntimeError('Selected NCCL binary hash changed')
    loaded=ctypes.CDLL(str(library),mode=ctypes.RTLD_GLOBAL);version=ctypes.c_int()
    if loaded.ncclGetVersion(ctypes.byref(version))!=0 or version.value!=23203:raise RuntimeError('Runtime requires patched NCCL2.32.3')
    import torch
    from tensorfold.cuda import comm
    if Path(comm._library()._name).resolve(strict=True)!=library:raise RuntimeError('TensorFold selected another NCCL library')
    maps=sorted({line.split(maxsplit=5)[-1].strip() for line in Path('/proc/self/maps').read_text().splitlines() if '/libnccl' in line and len(line.split(maxsplit=5))==6})
    if not maps or any(Path(path).resolve()!=library for path in maps):raise RuntimeError('Unexpected NCCL process mappings')
    for name,digest in json.loads(os.environ['SPARK_RING_EXPERIMENT_SOURCES']).items():
        if hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()!=digest:raise RuntimeError('Experimental source changed: '+name)
    print('SPARK_RING_LIBRARY '+json.dumps({'rank':int(os.environ['SPARK_RING_RANK']),'library':str(library),'sha256':actual,
        'nccl_version':version.value,'maps':maps,'torch_version':torch.__version__,'image_digest':os.environ['SPARK_RING_IMAGE_DIGEST'],
        'hcas':os.environ['NCCL_IB_HCA'],'comm':os.environ['TF_GLM_COMM'],'hc_exchange':os.environ['TF_GLM_HC_EXCHANGE']}),flush=True)
    from rocenante_adapter import install
    _CONTROLLER=install(json.loads(os.environ['SPARK_RING_ROCENANTE_SETTINGS']),os.environ['SPARK_RING_VENDOR_ROOT'],
        os.environ['SPARK_RING_VENDOR_MANIFEST_SHA256'],os.environ['SPARK_RING_ROCENANTE_MODE'])
    print('SPARK_RING_ROCENANTE_INSTALLED '+json.dumps(_CONTROLLER.report()),flush=True)
    health=json.loads(Path('/opt/spark-ring/mesh-health.json').read_text())
    if health['rank']!=int(os.environ['SPARK_RING_RANK']):raise RuntimeError('Mesh health configuration belongs to another rank')
    stop=threading.Event();watch=threading.Thread(target=health_watch,args=(stop,_CONTROLLER,health),daemon=True);watch.start()
    from tensorfold.cli import main as tensorfold_main
    sys.argv[0]='tensorfold'
    try:tensorfold_main()
    finally:
        stop.set()
        # TensorFold owns captured graphs until teardown; keep proxy buffers alive.
        try:report=_CONTROLLER.report()
        except Exception as error:report={'failure':str(error)}
        print('SPARK_RING_ROCENANTE_EXIT '+json.dumps(report),flush=True)


if __name__=='__main__':main()
