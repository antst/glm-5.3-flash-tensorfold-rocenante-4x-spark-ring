#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Python owns argument parsing and receipts; the shell never expands source inputs.
set -euo pipefail
SPARK_RING_BUILDER_FILE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")
export SPARK_RING_BUILDER_FILE
exec python3 -B - "$@" <<'SPARK_RING_NCCL_PYTHON'
"""Build pinned NCCL2.32.3 in a NEW local directory; never install globally.

--plan is offline/read-only and creates no directories. The actual build requires
--source (clean pinned Git), --archive (exact Git tar), or explicit --fetch.
CPU tests and optional image loading never initialize CUDA or create a communicator.
"""
import argparse
import ctypes
import hashlib
import json
import os
import posixpath
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime, timezone

sys.dont_write_bytecode = True
BUILDER = Path(os.environ.get('SPARK_RING_BUILDER_FILE', __file__)).resolve()
ROOT = BUILDER.parent.parent
COMMIT = '12df1a11afad322be5a204a2db890161cbf8131d'
ARCHIVE_SHA = 'a66ba2b0ac77e7339a71397292134d7ab33c3642af495a170f0c85f27a769008'
BASE_TREE = 'de760d9f34b52dee589604eeeccef9a09df8a979'
PATCH_SHA = 'a0deca2f21b8cad56bc68a5f5468936b7a6b894db160cfa1d62a6a3d59358836'
IMAGE = 'ghcr.io/miaai-lab/glm-5.3-flash-exl3-2x-dgx-sparks-tensorfold@sha256:bc34d7d63f978cf601f42863b284bc95a567c50c10e9adb0866a635be568bf5f'
SHARED_FS = {'nfs','nfs4','cifs','smb3','9p','ceph','glusterfs','lustre'}

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',',':')).encode()

def sha(path):
    with Path(path).open('rb') as file:
        return hashlib.file_digest(file,'sha256').hexdigest()

def run(argv, cwd=None, log=None, env=None):
    argv = list(map(str,argv))
    if log:
        with Path(log).open('w') as out:
            result = subprocess.run(argv,cwd=cwd,env=env,text=True,stdout=out,stderr=subprocess.STDOUT)
    else:
        result = subprocess.run(argv,cwd=cwd,env=env,text=True,capture_output=True)
    if result.returncode:
        detail = Path(log).read_text()[-6000:] if log else result.stderr[-6000:]
        raise RuntimeError(f'command failed {argv!r}: {detail}')
    return Path(log).read_text() if log else result.stdout

def git(source,*arguments):
    return run(['git','-c',f'safe.directory={source}','-C',source,*arguments],
               env=dict(os.environ,GIT_OPTIONAL_LOCKS='0',GIT_TERMINAL_PROMPT='0')).strip()

def manifest():
    value=json.loads((ROOT/'nccl-sources.json').read_text())
    if (value.get('schema'),value.get('commit'),value.get('version'),value.get('archive_sha256'),value.get('patch',{}).get('sha256')) != (
            'tensorfold-rocenante-nccl-sources/v1',COMMIT,'2.32.3',ARCHIVE_SHA,PATCH_SHA):
        raise ValueError('NCCL source/version/patch pin differs')
    if sha(ROOT/value['patch']['path']) != PATCH_SHA:
        raise ValueError('NCCL cumulative patch bytes differ')
    return value

def source_proof(source):
    source=Path(source).resolve()
    commit=git(source,'rev-parse','HEAD')
    status=git(source,'status','--porcelain=v1','--untracked-files=all')
    if commit != COMMIT or status:
        raise ValueError('NCCL source must be a clean checkout of the exact commit')
    return {'commit':commit,'tree':git(source,'rev-parse','HEAD^{tree}'),'source_clean':True,'git_status_porcelain':status}

def validate_paths(build,publish):
    build=Path(build).resolve()
    if build.exists(): raise ValueError('build directory must be new')
    ancestor=build
    while not ancestor.exists(): ancestor=ancestor.parent
    fs=run(['findmnt','-T',ancestor,'-n','-o','FSTYPE']).strip()
    if fs in SHARED_FS or fs.startswith('fuse.'):
        raise ValueError('compile storage must be local')
    publish=Path(publish).resolve() if publish else None
    if publish and (publish.exists() or publish==build or build in publish.parents or publish in build.parents):
        raise ValueError('publish directory must be new and separate')
    return build,publish

def toolchain(cuda):
    if platform.machine() not in ('aarch64','arm64'): raise ValueError('native build requires ARM64')
    cuda=Path(cuda).resolve()
    paths={name:shutil.which(name) for name in ('git','gcc','g++','make','readelf','ldd','findmnt')}
    paths.update(nvcc=str(cuda/'bin/nvcc'),cuobjdump=str(cuda/'bin/cuobjdump'))
    if any(not value or not Path(value).is_file() for value in paths.values()):
        raise ValueError('missing existing native tool; builder installs no packages')
    return {'platform':platform.machine(),'cuda_root':str(cuda),'paths':paths,
            'versions':{name:run([paths[name],'--version']) for name in ('gcc','g++','nvcc','ldd')}}

def extract(archive,destination):
    if sha(archive)!=ARCHIVE_SHA: raise ValueError('source archive SHA256 differs')
    with tarfile.open(archive,'r:') as tar:
        members=tar.getmembers()
        names=set()
        for member in members:
            path=PurePosixPath(member.name)
            if path.is_absolute() or '..' in path.parts or member.name in names or not (member.isfile() or member.isdir() or member.issym()):
                raise ValueError('unsafe or duplicate source archive member')
            names.add(member.name)
        links={member.name for member in members if member.issym()}
        for member in members:
            if any(str(parent) in links for parent in PurePosixPath(member.name).parents):
                raise ValueError('source member has a symbolic-link parent')
            if member.issym():
                target=posixpath.normpath(posixpath.join(posixpath.dirname(member.name),member.linkname))
                if member.linkname.startswith('/') or target.startswith('../') or target not in names or target in links:
                    raise ValueError('unsafe source symbolic link')
        destination.mkdir()
        for member in sorted(members,key=lambda item:bool(item.issym())):
            path=destination/member.name
            if member.isdir(): path.mkdir(parents=True,exist_ok=True)
            elif member.issym():
                path.parent.mkdir(parents=True,exist_ok=True);path.symlink_to(member.linkname)
            else:
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_bytes(tar.extractfile(member).read())
                path.chmod(member.mode & 0o777)

def inventory(source):
    names=git(source,'ls-files','-z').split('\0')
    return {name:(hashlib.sha256(os.readlink(source/name).encode()).hexdigest() if (source/name).is_symlink() else sha(source/name)) for name in names if name}

def restore_export_metadata(source,pins):
    restored=[]
    for item in pins['archive_export_subst']:
        path=source/item['path'];original=item['original_utf8'].encode()
        if sha(path)!=item['expanded_sha256'] or hashlib.sha256(original).hexdigest()!=item['original_sha256']:
            raise ValueError('archive export-subst metadata differs from fixed bytes')
        path.write_bytes(original)
        restored.append({'path':item['path'],'archive_sha256':item['expanded_sha256'],'restored_git_source_sha256':item['original_sha256'],
                         'reason':'Restore Git archive export-subst metadata to the exact original committed blob; no code change.'})
    return restored

def image_loader(library_dir,image,task,actions):
    if image!=IMAGE: raise ValueError('loader proof must use exact pinned Mia image')
    info=json.loads(run(['docker','image','inspect',image]))[0]
    name='spark-ring-rocenante-nccl-loader-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ').lower()
    code="""import ctypes,hashlib,json,pathlib
p=pathlib.Path('/opt/nccl/libnccl.so.2.32.3')
library=ctypes.CDLL(str(p));version=ctypes.c_int()
assert library.ncclGetVersion(ctypes.byref(version))==0 and version.value==23203
print(json.dumps({'passed':True,'library_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'nccl_get_version':version.value,'context_created':False,'session_created':False,'communicator_created':False,'rdma_endpoint_created':False}))
"""
    argv=['docker','run','--rm','--name',name,'--label','org.spark-ring.task=rocenante-nccl-build','--runtime','runc',
          '--env','NVIDIA_VISIBLE_DEVICES=void','--env','CUDA_VISIBLE_DEVICES=','--network','none','--read-only',
          '--cap-drop','ALL','--security-opt','no-new-privileges','--pids-limit','128','--memory','2g','--cpus','2',
          '--tmpfs','/tmp:rw,nosuid,nodev,size=16m','--mount',f'type=bind,src={library_dir},dst=/opt/nccl,readonly',
          '--entrypoint','python3',image,'-B','-c',code]
    actions.append({'action':'temporary_container','name':name,'command':argv,'gpu':False,'infiniband':False,'network':'none','auto_remove':True})
    (task/'build-actions.json').write_text(json.dumps(actions,indent=2)+'\n')
    result=json.loads(run(argv))
    if run(['docker','ps','-aq','--filter','name=^/'+name+'$']).strip():
        raise RuntimeError('temporary CPU loader container was not removed')
    result.update(image_digest=image,image_id=info['Id'],container_name=name,container_removed=True)
    (task/'image-loader-proof.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

def build_marker(args,pins,chain,task,actions):
    if not args.marker_source: return None
    marker=pins['marker']
    if sha(args.marker_source)!=marker['source_sha256']: raise ValueError('marker source SHA256 differs')
    directory=task/'mesh';directory.mkdir()
    source=directory/'mlx5_rdma_tx_rewrite_probe.c';shutil.copyfile(args.marker_source,source)
    binary=directory/marker['binary_name']
    argv=[chain['paths']['gcc'],'-O2','-Wall','-Wextra',source,'-o',binary,'-libverbs','-lmlx5']
    run(argv,log=directory/'compile.log')
    elf=run([chain['paths']['readelf'],'-h',binary]);dynamic=run([chain['paths']['readelf'],'-d',binary])
    deps=run([chain['paths']['ldd'],binary])
    if 'AArch64' not in elf or 'not found' in deps: raise RuntimeError('marker ELF/dependencies failed')
    receipt={'schema':'tensorfold-rocenante-mesh-marker-build/v1','created_at':datetime.now(timezone.utc).isoformat(),
             'source':marker,'binary':binary.name,'binary_sha256':sha(binary),'binary_bytes':binary.stat().st_size,
             'source_sha256':marker['source_sha256'],'source_commit':marker['source_commit'],
             'elf_machine':'AArch64','needed':re.findall(r'\(NEEDED\).*?\[(.*?)\]',dynamic),
             'compiler':chain['versions']['gcc'],'compile_command':list(map(str,argv)),
             'binary_executed':False,'rdma_endpoints_created':False,'hardware_qualified':False}
    linker=Path('/lib/ld-linux-aarch64.so.1')
    loader_argv=[linker,'--list',binary]
    loader_output=run(loader_argv)
    if 'not found' in loader_output: raise RuntimeError('marker CPU dynamic linker proof failed')
    receipt.update(cpu_loader_passed=True,cpu_loader_command=list(map(str,loader_argv)),cpu_loader_entrypoint_executed=False)
    (directory/'cpu-loader-list.txt').write_text(loader_output)
    (directory/'build-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    (directory/'readelf-header.txt').write_text(elf);(directory/'readelf-dynamic.txt').write_text(dynamic)
    (directory/'loader-dependencies.txt').write_text(deps)
    if args.publish_mesh:
        output=Path(args.publish_mesh).resolve()
        if output.exists(): raise ValueError('marker publish directory must be new')
        output.parent.mkdir(parents=True,exist_ok=True);output.mkdir()
        for file in directory.iterdir(): shutil.copy2(file,output/file.name)
        actions.append({'action':'new_marker_publish_directory','path':str(output),'binary_executed':False})
        (task/'build-actions.json').write_text(json.dumps(actions,indent=2)+'\n')
    print('Pinned hardware marker compiled and recorded; not executed.',flush=True)
    return receipt

def build(args,pins):
    task,publish=validate_paths(args.build_dir,args.publish)
    chain=toolchain(args.cuda_root)
    proof=source_proof(args.source) if args.source else None
    if args.archive and sha(args.archive)!=ARCHIVE_SHA: raise ValueError('source archive SHA256 differs')
    if args.marker_source and sha(args.marker_source)!=pins['marker']['source_sha256']: raise ValueError('marker source differs')
    if args.publish_mesh and Path(args.publish_mesh).exists(): raise ValueError('marker publish directory must be new')
    if args.loader_image:
        if args.loader_image!=IMAGE: raise ValueError('loader image differs')
        run(['docker','image','inspect',args.loader_image]) # no automatic pull
    builder=BUILDER.read_bytes()
    task.mkdir()
    (task/'builder-executed.sh').write_bytes(builder)
    actions=[{'action':'new_local_directory','path':str(task)}]
    def save_actions(): (task/'build-actions.json').write_text(json.dumps(actions,indent=2)+'\n')
    save_actions()
    archive=task/'nccl-source.tar'
    if args.source:
        run(['git','-c','core.autocrlf=false','-c','tar.umask=0022','-c',f'safe.directory={Path(args.source).resolve()}','-C',args.source,'archive','--format=tar','-o',archive,COMMIT])
    elif args.archive: shutil.copyfile(args.archive,archive)
    else:
        fetched=task/'fetched-source'
        run(['git','init',fetched],log=task/'git-init.log')
        run(['git','-C',fetched,'fetch','--depth=1',pins['repository'],COMMIT],log=task/'git-fetch.log',env=dict(os.environ,GIT_TERMINAL_PROMPT='0'))
        run(['git','-C',fetched,'checkout','--detach','FETCH_HEAD'],log=task/'git-checkout.log')
        proof=source_proof(fetched)
        run(['git','-c','core.autocrlf=false','-c','tar.umask=0022','-C',fetched,'archive','--format=tar','-o',archive,COMMIT])
        actions.append({'action':'fixed_source_fetch','repository':pins['repository'],'commit':COMMIT,'destination':str(fetched)})
        save_actions()
    source=task/'source';extract(archive,source)
    normalized_metadata=restore_export_metadata(source,pins)
    run(['git','init',source],log=task/'source-init.log')
    git(source,'add','--all')
    base_tree=git(source,'write-tree')
    if base_tree!=BASE_TREE: raise RuntimeError('base source tree differs from the fixed official commit')
    if proof and base_tree!=proof['tree']: raise RuntimeError('archive base tree differs from commit')
    before=inventory(source)
    patch=task/'nccl-2.32.3-dual-pci-domain.patch'
    shutil.copyfile(ROOT/pins['patch']['path'],patch)
    git(source,'apply','--check',patch);git(source,'apply','--index',patch)
    patched_tree=git(source,'write-tree');patched=inventory(source)
    proof={'commit':COMMIT,'base_tree':base_tree,'patched_tree':patched_tree,'archive_sha256':sha(archive),
           'archive_export_subst_restored':normalized_metadata,
           'patch_sha256':sha(patch),'base_source_clean':True,'proof_kind':'exact_pinned_Git_archive_and_index_trees',
           'base_file_sha256':before,'patched_file_sha256':patched,
           'base_inventory_sha256':hashlib.sha256(canonical(before)).hexdigest(),
           'patched_inventory_sha256':hashlib.sha256(canonical(patched)).hexdigest()}
    (task/'source-proof.json').write_text(json.dumps(proof,indent=2)+'\n')
    env=dict(os.environ,PATH=str(Path(chain['cuda_root'])/'bin')+os.pathsep+os.environ.get('PATH',''),CUDA_VISIBLE_DEVICES='')
    cpu=[chain['paths']['g++'],'-std=c++11','-O2','-Wall','-Wextra','-Werror',source/'tests/routing_handle/compat.cc','-o',task/'routing-handle-test']
    run(cpu,log=task/'cpu-test-build.log',env=env)
    run([task/'routing-handle-test'],log=task/'cpu-tests.log',env=env)
    marker=build_marker(args,pins,chain,task,actions)
    command=[chain['paths']['make'],f'-j{args.jobs}','src.build',f"CUDA_HOME={chain['cuda_root']}",
             f"CUDA_LIB={chain['cuda_root']}/lib64",'NVCC_GENCODE=-gencode=arch=compute_121,code=sm_121','NVCC_THREADS=1']
    print('Pinned source/archive/patch verified; CPU ABI/GID/PCI-root checks passed. Compiling NCCL2.32.3 SM121 with at most2 jobs.',flush=True)
    run(command,cwd=source,log=task/'build.log',env=env)
    if inventory(source)!=patched: raise RuntimeError('tracked source changed during compilation')
    if args.source: source_proof(args.source)
    libdir=task/'lib';libdir.mkdir();library=libdir/'libnccl.so.2.32.3'
    shutil.copyfile(source/'build/lib/libnccl.so.2.32.3',library);library.chmod(0o755)
    (libdir/'libnccl.so.2').symlink_to(library.name);(libdir/'libnccl.so').symlink_to('libnccl.so.2')
    for name in ('LICENSE.txt','ThirdPartyNotices.txt'): shutil.copyfile(source/name,libdir/name)
    elf=run([chain['paths']['readelf'],'-h',library]);dynamic=run([chain['paths']['readelf'],'-d',library])
    deps=run([chain['paths']['ldd'],library],env=env);cubins=run([chain['paths']['cuobjdump'],'--list-elf',library])
    for name,text in [('readelf-header.txt',elf),('readelf-dynamic.txt',dynamic),('loader-dependencies.txt',deps),('sm121-elfs.txt',cubins)]:
        (task/name).write_text(text)
    if 'AArch64' not in elf or 'sm_121' not in cubins or 'not found' in deps: raise RuntimeError('ELF/SM121/dependency proof failed')
    version=ctypes.c_int();native=ctypes.CDLL(str(library))
    if native.ncclGetVersion(ctypes.byref(version))!=0 or version.value!=23203: raise RuntimeError('CPU ncclGetVersion is not23203')
    loader={'passed':False,'status':'not_requested','context_created':False,'communicator_created':False}
    if args.loader_image: loader=image_loader(libdir,args.loader_image,task,actions)
    receipt={'schema':'tensorfold-rocenante-nccl-native-build/v1','generated_utc':datetime.now(timezone.utc).isoformat(),
      'base_commit':COMMIT,'source_commit':COMMIT,'base_tree':base_tree,'patched_tree':patched_tree,'source_tree':patched_tree,
      'source_archive_sha256':sha(archive),'patch_sha256':PATCH_SHA,'library':library.name,'library_sha256':sha(library),
      'upstream_recipe_archive_sha256':pins['upstream_recipe_archive']['sha256'],
      'upstream_recipe_archive_reproduced':False,
      'library_bytes':library.stat().st_size,'nccl_get_version':version.value,'elf_machine':'AArch64','sm121_embedded':True,
      'soname':re.findall(r'\(SONAME\).*?\[(.*?)\]',dynamic)[0],'needed':re.findall(r'\(NEEDED\).*?\[(.*?)\]',dynamic),
      'native_cpu_test_passed':True,'native_cpu_compatibility_test_passed':True,'native_cpu_library_load_passed':True,
      'image_loader_proof':loader,'target_image_cpu_loader_passed':bool(loader['passed']),
      'marker_build':marker,
      'target_image_loader_qualified':False,'hardware_collective_qualified':False,'source':proof,
      'build':{'toolchain':chain,'jobs':args.jobs,'architecture':'121','cpu_compile_command':list(map(str,cpu)),
               'compile_command':list(map(str,command)),'builder_sha256':hashlib.sha256(builder).hexdigest(),
               'source_manifest_sha256':sha(ROOT/'nccl-sources.json')},
      'limits':'CPU build/ABI/version/image loader only; no CUDA context, communicator, RDMA, GPU, inference or serving proof.'}
    (task/'build-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    if publish:
        publish.parent.mkdir(parents=True,exist_ok=True);publish.mkdir();actions.append({'action':'new_publish_directory','path':str(publish)});save_actions()
        for file in libdir.iterdir():
            if file.is_symlink(): (publish/file.name).symlink_to(os.readlink(file))
            else: shutil.copy2(file,publish/file.name)
        for name in ('build-receipt.json','builder-executed.sh','source-proof.json','nccl-source.tar','nccl-2.32.3-dual-pci-domain.patch',
                     'build-actions.json','cpu-test-build.log','cpu-tests.log','build.log','readelf-header.txt','readelf-dynamic.txt','loader-dependencies.txt','sm121-elfs.txt'):
            shutil.copyfile(task/name,publish/name)
        if args.loader_image: shutil.copyfile(task/'image-loader-proof.json',publish/'image-loader-proof.json')
    print(json.dumps({'receipt':str(task/'build-receipt.json'),'published':str(publish),'base_tree':base_tree,'patched_tree':patched_tree,
                     'library_sha256':receipt['library_sha256'],'nccl_get_version':23203,'image_cpu_loader_passed':loader['passed']},indent=2),flush=True)
    return receipt

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',action='store_true')
    group=parser.add_mutually_exclusive_group()
    group.add_argument('--source',type=Path);group.add_argument('--archive',type=Path);group.add_argument('--fetch',action='store_true')
    parser.add_argument('--build-dir',type=Path);parser.add_argument('--publish',type=Path)
    parser.add_argument('--jobs',type=int,choices=(1,2),default=2)
    parser.add_argument('--cuda-root',type=Path,default=Path('/usr/local/cuda'))
    parser.add_argument('--loader-image')
    parser.add_argument('--marker-source',type=Path,help='optional exact pinned marker C source; compiled but never executed')
    parser.add_argument('--publish-mesh',type=Path,help='new marker artifact directory; requires --marker-source')
    args=parser.parse_args(argv);pins=manifest()
    if args.loader_image and args.loader_image!=IMAGE: raise ValueError('loader image differs')
    if args.publish_mesh and not args.marker_source: raise ValueError('--publish-mesh requires --marker-source')
    if args.plan:
        proof=source_proof(args.source) if args.source else None
        if args.archive and sha(args.archive)!=ARCHIVE_SHA: raise ValueError('archive differs')
        print(json.dumps({'mode':'offline_plan','writes':False,'network':False,'ssh':False,'gpu':False,'source':pins,
                         'existing_source_proof':proof,'jobs':args.jobs,'build_dir':str(args.build_dir) if args.build_dir else None,
                         'publish':str(args.publish) if args.publish else None,'loader_image':args.loader_image},indent=2))
        return 0
    if not args.build_dir or not (args.source or args.archive or args.fetch):
        raise ValueError('build requires NEW --build-dir and --source, --archive or explicit --fetch')
    build(args,pins)
    return 0

if __name__=='__main__':
    try: raise SystemExit(main())
    except (ValueError,RuntimeError) as error: raise SystemExit(str(error))
SPARK_RING_NCCL_PYTHON
