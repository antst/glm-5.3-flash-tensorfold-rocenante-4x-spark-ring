#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Prove the selected NCCL mapping, install prepared transport, then serve."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time
import urllib.request
sys.dont_write_bytecode=True
_CONTROLLER=None


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
            now=time.monotonic()
            if now>=next_report:
                # Host proxy counters only: no CUDA synchronization or GPU stats.
                print('SPARK_RING_ROCENANTE_ACTIVE '+json.dumps(controller.report()),flush=True)
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
