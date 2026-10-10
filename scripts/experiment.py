#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Operate the isolated prepared-RoCEnante/NCCL experiment; plan stays offline."""
import argparse
import hashlib
import ipaddress
import json
from pathlib import Path
import shlex
import sys
import uuid
sys.dont_write_bytecode=True
from common import (IMAGE, environment, library_path, load_config, ordered_hcas,
                    state_dir, topology_sha256, validate_local_storage)
from cluster import Operator
from network import canonical, live_guard, make_plan, marker_identity, operate, resolved

OWNER='tensorfold-rocenante-experiment-v1'
VENDOR_MANIFEST_SHA256='19b6189f2856655b6919bdcfcb4974bcf9dc7b71882f179b814de36c3615bf89'
NCCL_COMMIT='12df1a11afad322be5a204a2db890161cbf8131d'
NCCL_PATCH_SHA256='a0deca2f21b8cad56bc68a5f5468936b7a6b894db160cfa1d62a6a3d59358836'
SOURCE_FILES=('common.py','cluster.py','network.py','experiment.py','runtime.py','rocenante_adapter.py',
              'qualify.py','collective-probe.py','rocenante-probe.py','capture-function-counters.py')
RECEIPT_FLAGS=('qualified','byte_exact_passed','cuda_graph_passed','mixed_size_passed','inplace_passed',
               'payload_all_hcas_passed','native_health_passed','host_configuration_unchanged',
               'qp_diagnostics_all_hcas_passed','hardware_mesh_payload_passed','hardware_mesh_passed','asymmetric_cache_passed')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):h.update(block)
    return h.hexdigest()


def json_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def prepare_config(path,mode=None):
    cfg=load_config(path)
    if cfg['recipe_id']!=OWNER or cfg['container_name']!='spark-ring-glm53-rocenante-experiment':
        raise ValueError('Use the isolated RoCEnante recipe/container ownership namespace')
    if cfg['api_port']==8888 or cfg['bootstrap_port']==29551 or 'rocenante' not in Path(cfg['cache_root']).name:
        raise ValueError('Experiment needs separate ports and local rocenante cache')
    canonical(cfg)
    e=cfg.setdefault('experiment',{})
    defaults={'mode':'rocenante','transport':'prepared-rocenante-tp4','min_gather_bytes':1,
              'max_gather_bytes':16777216,'paths_per_peer':2,'sircl_enabled':False,'scope':'all-gather',
              'vendor_root':'third_party/prepared-b12x','vendor_manifest_sha256':VENDOR_MANIFEST_SHA256,
              'timeout_seconds':600,'compile_workers':0,'forward_window_bytes':131072,'forward_chunk_bytes':32768,
              'native_blocks':8}
    for key,value in defaults.items():e.setdefault(key,value)
    if mode is not None:e['mode']=mode
    if e['mode'] not in ('nccl','rocenante') or e['transport']!='prepared-rocenante-tp4' or e['scope']!='all-gather' or e['sircl_enabled'] is not False:
        raise ValueError('This experiment uses prepared RoCEnante or identical NCCL control, without SIRCL')
    if e['paths_per_peer']!=2 or e['vendor_manifest_sha256']!=VENDOR_MANIFEST_SHA256:
        raise ValueError('Exact current prepared framework and two paths per peer required')
    if (type(e['max_gather_bytes']) is not int or not 16<=e['max_gather_bytes']<=16777216 or e['max_gather_bytes']%16 or
        type(e['min_gather_bytes']) is not int or not 1<=e['min_gather_bytes']<=e['max_gather_bytes']):
        raise ValueError('Invalid raw all-gather admission limits')
    if type(e['timeout_seconds']) is not int or not 10<=e['timeout_seconds']<=3600 or type(e['compile_workers']) is not int or not 0<=e['compile_workers']<=8:
        raise ValueError('Invalid preparation timeout/compile concurrency')
    if type(e['native_blocks']) is not int or e['native_blocks'] not in (4,8,16):
        raise ValueError('Native gather blocks must be4,8 or16; threads are fixed at512')
    if 'native_threads' in e and (type(e['native_threads']) is not int or e['native_threads']!=512):
        raise ValueError('Native gather threads are fixed at512')
    if (type(e['forward_window_bytes'])!=int or e['forward_window_bytes'] not in (32768,65536,131072,262144) or
        type(e['forward_chunk_bytes'])!=int or e['forward_chunk_bytes'] not in (16384,32768,65536) or
        e['forward_chunk_bytes']>e['forward_window_bytes']):
        raise ValueError('Choose bounded pinned native forward windows32/64/128/256KiB and chunks16/32/64KiB')
    cfg['model_dir']=cfg.get('model_dir',str(Path(cfg['model_root'])/'glm53-exl3'))
    cfg['drafter_dir']=cfg.get('drafter_dir',str(Path(cfg['model_root'])/'dflash2'))
    validate_local_storage(cfg)
    return cfg


def vendor_path(cfg):
    p=Path(cfg['experiment']['vendor_root'])
    return p if p.is_absolute() else Path(cfg['shared_root'])/p


def vendor_identity(cfg):
    from rocenante_adapter import verify_vendor
    root,manifest=verify_vendor(vendor_path(cfg),VENDOR_MANIFEST_SHA256)
    return {'path':str(root),'manifest_sha256':VENDOR_MANIFEST_SHA256,'files':len(manifest['files'])}


def source_hashes(cfg):
    return {name:sha(Path(cfg['shared_root'])/'scripts'/name) for name in SOURCE_FILES}


def rank_settings(cfg,rank,run_id=None):
    e=cfg['experiment']
    return {'rank':rank,'world':4,'hca_names':ordered_hcas(cfg['nodes'][rank]),
            'peer_hosts':[n['name'] for n in cfg['nodes']],
            'store_prefix':cfg['recipe_id']+'/'+(run_id or 'model'),
            'min_gather_bytes':e['min_gather_bytes'],'max_gather_bytes':e['max_gather_bytes'],
            'timeout_seconds':e['timeout_seconds'],'compile_workers':e['compile_workers'],'device':'cuda:0',
            'native_blocks':e['native_blocks'],'native_threads':512}


def experiment_identity(cfg,nccl_sha256):
    library=library_path(cfg);build=library.parent/'build-receipt.json'
    receipt=json.loads(build.read_text())
    if (receipt.get('source_commit')!=NCCL_COMMIT or receipt.get('patch_sha256')!=NCCL_PATCH_SHA256 or
        receipt.get('nccl_get_version')!=23203 or receipt.get('native_cpu_test_passed') is not True):
        raise RuntimeError('Fresh NCCL2.32.3 binary/build CPU proof required')
    if sha(library)!=nccl_sha256:raise RuntimeError('NCCL identity changed')
    binary,binary_sha=marker_identity(cfg)
    inventory=json.loads((state_dir(cfg)/'network-inventory.json').read_text())
    plan=make_plan(cfg,inventory)
    if not plan.get('complete'):raise RuntimeError('Exact inventory-backed mesh plan required')
    settings={k:v for k,v in cfg['experiment'].items() if k not in ('mode','network_policy')}
    return {'schema':'tensorfold-rocenante-experiment-identity/v1','image_digest':IMAGE,
            'library_sha256':nccl_sha256,'nccl_build_receipt_sha256':sha(build),'vendor':vendor_identity(cfg),
            'marker_sha256':binary_sha,'marker_receipt_sha256':sha(resolved(cfg,'marker_receipt','lib/mesh/build-receipt.json')),
            'network_plan_id':plan['plan_id'],'network_planner_sha256':plan['upstream_planner_sha256'],
            'sources':source_hashes(cfg),'hosts':[n['name'] for n in cfg['nodes']],
            'topology_sha256':topology_sha256(cfg),'profile':cfg['profile'],'settings':settings,
            'transport_env_by_host':{n['name']:__import__('common').transport_environment(environment(cfg,r,nccl_sha256)) for r,n in enumerate(cfg['nodes'])}}


def verify_boot_qualification(qualification,guard):
    """Both hardware receipts must cover this exact four-host boot generation."""
    boots=guard.get('network_boot_ids')
    if not isinstance(boots,dict) or len(boots)!=4 or set(boots)!=set(guard.get('hosts',{})):
        raise RuntimeError('Fresh four-host boot identities are missing from the live mesh guard')
    try:
        if any(not isinstance(value,str) or str(uuid.UUID(value))!=value for value in boots.values()):raise ValueError()
    except (ValueError,AttributeError,TypeError):
        raise RuntimeError('Live mesh contains an invalid boot identity') from None
    for mode in ('baseline','prepared'):
        if qualification[mode].get('network_boot_ids')!=boots:
            raise RuntimeError(mode+': qualification belongs to a previous host boot; rerun both hardware qualifications')


class ExperimentOperator(Operator):
    def qualification(self,path,digest):
        baseline=super().qualification(path,digest)
        custom_path=self.state/'rocenante-qualification.json';custom=json.loads(custom_path.read_text())
        for flag in RECEIPT_FLAGS:
            if custom.get(flag) is not True:raise RuntimeError('Fresh RoCEnante qualification requires '+flag+'=true')
        identity=experiment_identity(self.config,digest)
        if custom.get('identity')!=identity:raise RuntimeError('Prepared transport/source/topology/settings/build identity differs; rerun fresh qualification')
        if custom.get('baseline_receipt_sha256')!=sha(path):raise RuntimeError('Fresh NCCL oracle receipt changed')
        if sorted(custom.get('modes_qualified',[]))!=['nccl','rocenante'] or custom.get('network_plan_id')!=identity['network_plan_id']:
            raise RuntimeError('Qualification does not prove the exact hardware mesh and both modes')
        return {'baseline':baseline,'prepared':custom}
    def command(self,rank,library,digest):
        cache,command=super().command(rank,library,digest)
        cache=cache/self.config['experiment']['mode']
        # Replace only the original owned cache mount.
        command=[s.replace('src='+str(cache.parent)+',dst=/cache','src='+str(cache)+',dst=/cache') for s in command]
        source={name:sha(Path(self.config['shared_root'])/'scripts'/name) if (Path(self.config['shared_root'])/'scripts'/name).is_file() else 'QUALIFY_SOURCE_SHA256' for name in SOURCE_FILES}
        extra={'SPARK_RING_ROCENANTE_MODE':self.config['experiment']['mode'],
               'SPARK_RING_ROCENANTE_SETTINGS':json.dumps(rank_settings(self.config,rank),sort_keys=True,separators=(',',':')),
               'SPARK_RING_VENDOR_ROOT':'/opt/spark-ring/vendor',
               'SPARK_RING_VENDOR_MANIFEST_SHA256':VENDOR_MANIFEST_SHA256,
               'SPARK_RING_EXPERIMENT_SOURCES':json.dumps(source,sort_keys=True,separators=(',',':')),
               'XDG_CACHE_HOME':'/cache','B12X_COMPILE_CACHE_DIR':'/cache/b12x',
               'B12X_ROCE_CACHE_DIR':'/cache/roce','CUTE_DSL_CACHE_DIR':'/cache/cute',
               'SIRCL_ENABLED':'0'}
        index=command.index(IMAGE)
        health_path='/var/tmp/tensorfold-rocenante-mesh/'+self.config['recipe_id']+'/health.json'
        mounts=['--mount','type=bind,src='+str(vendor_path(self.config))+',dst=/opt/spark-ring/vendor,readonly',
                '--mount','type=bind,src='+health_path+',dst=/opt/spark-ring/mesh-health.json,readonly']
        args=mounts
        for key,value in sorted(extra.items()):args+=['-e',key+'='+value]
        command[index:index]=args
        return cache,command
    def preflight(self,rank,library,digest):
        guard=getattr(self,'mesh_guard',None) or live_guard(self.config)
        payload={'sources':source_hashes(self.config),'shared_root':self.config['shared_root'],
                 'vendor':str(vendor_path(self.config)),'vendor_manifest_sha256':VENDOR_MANIFEST_SHA256,
                 'hcas':ordered_hcas(self.config['nodes'][rank]),
                 'allowed_marker_pids':guard['allowed_marker_pids_by_host'][self.config['nodes'][rank]['name']]}
        code='data='+repr(payload)+'\n'+r'''
import hashlib,json,pathlib,subprocess
if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip():raise RuntimeError('Active GPU workload; experimental start never stops another model')
for name,wanted in data['sources'].items():
 if hashlib.sha256((pathlib.Path(data['shared_root'])/'scripts'/name).read_bytes()).hexdigest()!=wanted:raise RuntimeError('Shared experiment source changed')
manifest=pathlib.Path(data['vendor'])/'source-manifest.json'
if hashlib.sha256(manifest.read_bytes()).hexdigest()!=data['vendor_manifest_sha256']:raise RuntimeError('Prepared vendor manifest changed')
for name,wanted in json.loads(manifest.read_text())['files'].items():
 if hashlib.sha256((manifest.parent/name).read_bytes()).hexdigest()!=wanted:raise RuntimeError('Prepared vendor source changed')
rows=json.loads(subprocess.check_output(['rdma','-j','resource','show','qp'],text=True) or '[]')
if any(r.get('ifname','').split('/')[0] in data['hcas'] and r.get('pid') not in data['allowed_marker_pids'] and
       not (r.get('pid') is None and r.get('comm')=='ib_core' and r.get('type')=='GSI') for r in rows):raise RuntimeError('Unowned fabric QPs are already in use')
print(json.dumps({'sources_verified':True,'vendor_verified':True,'gpu_idle':True,'fabric_qps_idle':True}))
'''
        extras=json.loads(self.remote_python(rank,code))
        result=super().preflight(rank,library,digest);result['experiment']=extras
        if any(f['gid']!=3 for f in result['rails']):raise RuntimeError('Current prepared profile requires all16GIDs at3')
        return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',nargs='?',choices=('plan','check','start','status','stop','logs','wait'))
    p.add_argument('--plan',action='store_true');p.add_argument('--config',required=True)
    p.add_argument('--mode',choices=('nccl','rocenante'));p.add_argument('--timeout',type=int,default=3600);p.add_argument('--no-wait',action='store_true')
    args=p.parse_args()
    if args.plan:
        if args.action not in (None,'plan'):p.error('--plan cannot be combined with a host action')
        args.action='plan'
    if not args.action:p.error('Choose an explicit action')
    cfg=prepare_config(args.config,args.mode);op=ExperimentOperator(cfg);library=library_path(cfg)
    if args.action=='plan':
        digest=sha(library) if library.is_file() else 'BUILD_AND_QUALIFY_NCCL_SHA256'
        print(json.dumps({'offline':True,'mode':cfg['experiment']['mode'],'hosts':[n['name'] for n in cfg['nodes']],
            'required_receipts':['state/collective-qualification.json','state/rocenante-qualification.json'],
            'commands':[{'rank':r,'host':n['ssh'],'cache':str(op.command(r,library,digest)[0]),'argv':op.command(r,library,digest)[1]} for r,n in enumerate(cfg['nodes'])]},indent=2));return
    if any(ipaddress.IPv4Address(n['management_address']) in ipaddress.IPv4Network('192.0.2.0/24') for n in cfg['nodes']):p.error('Configure real management addresses before host operations')
    if args.action=='status':print(json.dumps(op.parallel(op.inspect),indent=2));return
    if args.action=='stop':print(json.dumps(op.parallel(op.stop_one),indent=2));return
    if args.action=='logs':
        for rank,node in enumerate(cfg['nodes']):print(node['name']+'\n'+op.remote(rank,shlex.join(['docker','logs','--tail','100',cfg['container_name']])+' 2>&1'))
        return
    if args.action=='wait':op.wait(args.timeout);return
    library,digest=op.library_identity();qualification=op.qualification(state_dir(cfg)/'collective-qualification.json',digest)
    inventory=json.loads((state_dir(cfg)/'network-inventory.json').read_text());plan=make_plan(cfg,inventory)
    network=live_guard(cfg);verify_boot_qualification(qualification,network);op.mesh_guard=network
    checks=op.parallel(lambda rank:op.preflight(rank,library,digest))
    if any(c['existing_container'] is not None for c in checks.values()):raise RuntimeError('Owned experiment container exists; use status/stop before a fresh start')
    if any(c['memory_available_gib']<64 for c in checks.values()):raise RuntimeError('Every rank requires at least64GiB available before launch')
    op.event('experiment-preflight-passed',mode=cfg['experiment']['mode'],hosts=checks,network=network)
    if args.action=='check':print(json.dumps({'preflight':True,'hosts':checks},indent=2));return
    started=[]
    try:
        for rank in (3,2,1,0):
            cache,command=op.command(rank,library,digest);started.append(rank)
            op.event('experiment-container-start',host=cfg['nodes'][rank]['name'],mode=cfg['experiment']['mode'],command=command,cache=str(cache))
            cid=op.remote(rank,shlex.join(['mkdir','-p',str(cache)])+'\n'+shlex.join(command)).strip()
            print(cfg['nodes'][rank]['name']+': started '+cid[:12],flush=True)
        if not args.no_wait:op.wait(args.timeout)
    except BaseException:
        for rank in reversed(started):
            try:op.stop_one(rank)
            except Exception as error:op.event('failed-start-owned-cleanup-error',rank=rank,error=str(error))
        raise


if __name__=='__main__':main()
