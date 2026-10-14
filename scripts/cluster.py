#!/usr/bin/env python3
"""Operate the pinned Mia engine on a configured four-Spark ring."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import ipaddress
import json
from pathlib import Path
import shlex
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
sys.dont_write_bytecode=True

from common import (IMAGE, IMAGE_DIGEST, MODEL_PINS, environment, library_path,
                    load_config, ssh_args, state_dir, topology_sha256,
                    transport_environment, validate_local_storage, validate_profile)


class Operator:
    def __init__(self, config):
        self.config = config
        self.state = state_dir(config)
        self.lock = threading.Lock()

    def event(self, action, **details):
        with self.lock:
            self.state.mkdir(parents=True, exist_ok=True)
            with (self.state / "events.jsonl").open("a") as output:
                output.write(json.dumps({"utc": datetime.now(timezone.utc).isoformat(),
                                         "action": action, **details}, sort_keys=True) + "\n")

    def remote(self, rank, script, timeout=120):
        marker = "SPARK_RING_OPERATOR_SCRIPT"
        if marker in script:
            raise ValueError("literal heredoc marker occurs in script")
        command = f"sudo -n -s <<'{marker}'\nset -eu\n{script}\n{marker}"
        node = self.config["nodes"][rank]
        result = subprocess.run(ssh_args(node, self.config) + [command], text=True,
                                capture_output=True, timeout=timeout)
        if result.returncode:
            raise RuntimeError(f"{node['name']}: {result.stderr.strip() or result.stdout.strip()}")
        return result.stdout

    def remote_python(self, rank, code, timeout=120):
        return self.remote(rank, "python3 <<'SPARK_RING_PYTHON'\n" + code + "\nSPARK_RING_PYTHON", timeout)

    def parallel(self, function):
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(function, range(4)))
        return dict(zip((node["name"] for node in self.config["nodes"]), results))

    def library_identity(self):
        path = library_path(self.config)
        path.resolve(strict=True)
        return path, hashlib.sha256(path.read_bytes()).hexdigest()

    def qualification(self, path, digest):
        document = json.loads(path.read_text())
        for flag in ("qualified", "correctness_passed", "payload_all_hcas_passed", "cuda_graph_passed",
                     "host_configuration_unchanged", "qp_diagnostics_all_hcas_passed"):
            if document.get(flag) is not True:
                raise RuntimeError(f"{path}: required {flag}=true")
        if document.get("library_sha256") != digest:
            raise RuntimeError("qualification library SHA256 differs from the selected binary")
        if document.get("image_digest") not in (IMAGE, IMAGE_DIGEST):
            raise RuntimeError("qualification image differs from the pinned Mia image")
        names = [node["name"] for node in self.config["nodes"]]
        if document.get("hosts") != names or document.get("topology_sha256") != topology_sha256(self.config):
            raise RuntimeError("qualification must cover the exact configured hosts/topology")
        expected_hcas = {n["name"]: [f["hca"] for f in n["fabric"]] for n in self.config["nodes"]}
        recorded_hcas = document.get("required_hcas_by_host")
        if recorded_hcas is None:
            recorded_hcas = {name: document.get("required_hcas", []) for name in names}
        if set(recorded_hcas) != set(names) or any(len(recorded_hcas[name]) != 4 or
                set(recorded_hcas[name]) != set(expected_hcas[name]) for name in names):
            raise RuntimeError("qualification must prove all four configured functions on every host")
        expected_env = {node["name"]: transport_environment(environment(self.config, rank, digest))
                        for rank, node in enumerate(self.config["nodes"])}
        recorded_env = document.get("transport_env_by_host")
        if recorded_env is None:
            recorded_env = {name: document.get("transport_env") for name in names}
        if recorded_env != expected_env:
            raise RuntimeError("qualified transport settings differ; rerun scripts/qualify.py for this configuration")
        return document

    def command(self, rank, library, digest):
        cfg, p = self.config, self.config["profile"]
        node = cfg["nodes"][rank]
        cache = Path(cfg["cache_root"]) / (IMAGE_DIGEST[7:19] + "-" + digest[:12]) / node["name"]
        cmd = ["docker", "run", "-d", "--name", cfg["container_name"],
               "--label", "spark-ring.recipe=" + cfg["recipe_id"],
               "--label", "spark-ring.root=" + cfg["shared_root"],
               "--label", "spark-ring.nccl-sha256=" + digest,
               "--gpus", "all", "--ipc=host", "--network", "host", "--shm-size", "16g",
               "--device", "/dev/infiniband", "--cap-add", "IPC_LOCK",
               "--ulimit", "memlock=-1", "--ulimit", "stack=67108864",
               "--log-driver", "json-file", "--log-opt", "max-size=100m", "--log-opt", "max-file=3",
               "--mount", f"type=bind,src={cfg['model_dir']},dst=/models/glm53-exl3,readonly",
               "--mount", f"type=bind,src={cfg['drafter_dir']},dst=/models/dflash2,readonly",
               "--mount", f"type=bind,src={library.parent},dst=/opt/spark-ring/lib,readonly",
               "--mount", f"type=bind,src={cfg.get('runtime_source_root', cfg['shared_root']+'/scripts')},dst=/opt/spark-ring/scripts,readonly",
               "--mount", f"type=bind,src={cache},dst=/cache"]
        from engine_binding import mounts as engine_mounts
        cmd += engine_mounts(cfg)
        for key, value in sorted(environment(cfg, rank, digest).items()):
            cmd += ["-e", key + "=" + value]
        cmd += [IMAGE, "python", "/opt/spark-ring/scripts/runtime.py", "serve", "/models/glm53-exl3",
                "--tp", "4", "--rank", str(rank), "--master", cfg["nodes"][0]["management_address"],
                "--master-port", str(cfg["bootstrap_port"]), "--context", str(p["context"]),
                "--parallel", str(p["parallel"]), "--max-tokens", str(p["max_tokens"]),
                "--drafter", "/models/dflash2", "--thinking", "--vision"]
        if rank == 0:
            cmd += ["--name", "GLM-5.3-Flash-EXL3", "--host", "0.0.0.0", "--port", str(cfg["api_port"])]
        return cache, cmd

    def inspect(self, rank):
        code = "name=" + repr(self.config["container_name"]) + "\n" + r'''
import json,subprocess
p=subprocess.run(['docker','inspect',name],text=True,capture_output=True)
if p.returncode: print(json.dumps({'exists':False}))
else:
 c=json.loads(p.stdout)[0]
 print(json.dumps({'exists':True,'id':c['Id'],'state':c['State'],'labels':c['Config'].get('Labels',{})}))
'''
        return json.loads(self.remote_python(rank, code))

    def preflight(self, rank, library, digest):
        cfg, node = self.config, self.config["nodes"][rank]
        payload = {"node": node, "image": IMAGE, "library": str(library), "sha256": digest,
                   "model_dir": cfg["model_dir"], "drafter_dir": cfg["drafter_dir"], "model_root": cfg["model_root"],
                   "pins": MODEL_PINS, "scripts": cfg.get("runtime_source_root", cfg["shared_root"] + "/scripts"), "mtu": cfg["mtu"],
                   "shared_root": cfg["shared_root"], "cache_root": cfg["cache_root"],
                   "container": cfg["container_name"], "free_ports": [cfg["bootstrap_port"], cfg["api_port"]] if rank == 0 else []}
        code = "data=" + repr(payload) + "\n" + r'''
import hashlib,ipaddress,json,pathlib,subprocess
def run(argv):return subprocess.check_output(argv,text=True,stderr=subprocess.STDOUT)
image=json.loads(run(['docker','image','inspect',data['image']]))[0]
if image['Architecture']!='arm64':raise RuntimeError('Expected ARM64 Spark image')
if hashlib.sha256(pathlib.Path(data['library']).read_bytes()).hexdigest()!=data['sha256']:raise RuntimeError('Shared library digest mismatch')
if not (pathlib.Path(data['scripts'])/'runtime.py').is_file():raise RuntimeError('Shared package scripts missing')
addresses={x['ifname']:x for x in json.loads(run(['ip','-j','-4','addr','show']))}
shared=pathlib.Path(data['shared_root']).resolve()
for name in ('model_root','model_dir','drafter_dir','cache_root'):
 path=pathlib.Path(data[name]).resolve()
 if path==shared or shared in path.parents:raise RuntimeError(name+' must not be inside shared_root')
 while not path.exists():path=path.parent
 filesystem=run(['stat','-f','-c','%T',str(path)]).strip().lower()
 if filesystem.startswith('nfs') or filesystem in ('cifs','smb2','smb3'):raise RuntimeError(name+' must use local storage, not '+filesystem)
node=data['node']
if node['management_address'] not in [x['local'] for x in addresses[node['management_interface']]['addr_info']]:raise RuntimeError('Management address/interface mismatch')
rails=[]
for f in node['fabric']:
 base=pathlib.Path('/sys/class/infiniband')/f['hca']/'ports/1'
 if 'ACTIVE' not in (base/'state').read_text():raise RuntimeError(f['hca']+' is not ACTIVE')
 if set(x.name for x in (base.parent.parent/'device/net').iterdir())!={f['interface']}:raise RuntimeError('HCA/netdev mismatch')
 item=addresses[f['interface']]
 if item['mtu']!=data['mtu'] or not any(a['local']==f['address'] and a['prefixlen']==f['prefixlen'] for a in item['addr_info']):raise RuntimeError('Fabric address/prefix/MTU mismatch')
 gids=[]
 for path in (base/'gids').iterdir():
  try:gid=ipaddress.IPv6Address(path.read_text().strip())
  except (ValueError,OSError):continue
  if str(gid.ipv4_mapped)==f['address'] and 'v2' in (base/'gid_attrs/types'/path.name).read_text():gids.append(int(path.name))
 if len(gids)!=1:raise RuntimeError('Need exactly one IPv4 RoCEv2 GID per configured rail')
 rails.append({**f,'gid':gids[0],'active':True})
snapshots=[]
for name,pin in data['pins'].items():
 directory=pathlib.Path(data['model_dir'] if name=='glm53-exl3' else data['drafter_dir'])
 receipt=json.loads((pathlib.Path(data['model_root'])/(name+'-receipt.json')).read_text())
 if receipt.get('complete') is not True or (receipt.get('repo'),receipt.get('revision'))!=tuple(pin):raise RuntimeError('Missing complete pinned download receipt')
 for file in receipt['files']:
  part=pathlib.Path(file['path'])
  if part.is_absolute() or '..' in part.parts:raise RuntimeError('Unsafe snapshot receipt path')
  actual=directory/part
  if not actual.is_file() or actual.stat().st_size!=file['size']:raise RuntimeError('Snapshot file/size mismatch: '+str(actual))
  if file.get('sha256') and file['size']<64*1024*1024 and hashlib.sha256(actual.read_bytes()).hexdigest()!=file['sha256']:raise RuntimeError('Snapshot small-file SHA256 mismatch: '+str(actual))
 snapshots.append({'name':name,'revision':pin[1],'files':len(receipt['files'])})
model=pathlib.Path(data['model_dir'])
for name in ['config.json','quantization_config.json','model.safetensors.index.json','tokenizer.json']:
 if not (model/name).is_file() or (model/name).stat().st_size==0:raise RuntimeError('Missing target file '+name)
index=json.loads((model/'model.safetensors.index.json').read_text());shards=set(index['weight_map'].values())
if not shards:raise RuntimeError('Empty target shard index')
for name in shards:
 part=pathlib.Path(name)
 if part.is_absolute() or '..' in part.parts or not (model/part).is_file():raise RuntimeError('Invalid/missing target shard')
draft=pathlib.Path(data['drafter_dir'])
if not (draft/'config.json').is_file() or not list(draft.glob('*.safetensors')):raise RuntimeError('Drafter snapshot missing')
p=subprocess.run(['docker','inspect',data['container']],text=True,capture_output=True)
container=json.loads(p.stdout)[0] if p.returncode==0 else None
listeners=run(['ss','-H','-ltn']).splitlines()
busy=[port for port in data['free_ports'] if any(line.split()[3].rsplit(':',1)[-1]==str(port) for line in listeners)]
if busy and container is None:raise RuntimeError('Required TCP ports busy: '+str(busy))
available=int(next(line.split()[1] for line in pathlib.Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))*1024
print(json.dumps({'image_id':image['Id'],'library_sha256':data['sha256'],'rails':rails,'snapshots':snapshots,
 'memory_available_gib':available/1024**3,'existing_container':None if container is None else {'id':container['Id'],'labels':container['Config'].get('Labels',{})}}))
'''
        return json.loads(self.remote_python(rank, code))

    def stop_one(self, rank):
        node, cfg = self.config["nodes"][rank], self.config
        current = self.inspect(rank)
        if not current["exists"]:
            return {"changed": False, "reason": "container absent"}
        labels = current["labels"]
        if labels.get("spark-ring.recipe") != cfg["recipe_id"] or labels.get("spark-ring.root") != cfg["shared_root"]:
            raise RuntimeError(f"{node['name']}: refusing to stop a container owned by another deployment")
        logs = self.state / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        archive = logs / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + node["name"] + ".log")
        self.event("container_stop_begin", host=node["name"], container_id=current["id"], archive=str(archive))
        archive.write_text(self.remote(rank, shlex.join(["docker", "logs", "--timestamps", cfg["container_name"]]) + " 2>&1"))
        self.remote(rank, shlex.join(["docker", "stop", "-t", "30", cfg["container_name"]]) + "\n" +
                    shlex.join(["docker", "rm", cfg["container_name"]]))
        self.event("container_stop_complete", host=node["name"], container_id=current["id"])
        return {"changed": True, "log_archive": str(archive)}

    def wait(self, timeout):
        url = f"http://{self.config['nodes'][0]['management_address']}:{self.config['api_port']}/v1/models"
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            for host, row in self.parallel(self.inspect).items():
                if not row["exists"] or not row["state"]["Running"]:
                    raise RuntimeError(f"{host}: rank exited before HTTP readiness: {row}")
            try:
                with urllib.request.urlopen(url, timeout=5) as response:
                    models = json.load(response)
                if any(item.get("id") == "GLM-5.3-Flash-EXL3" for item in models.get("data", [])):
                    self.event("api_ready", url=url, models=models)
                    print(json.dumps({"api_ready": True, "url": url, "models": models}, indent=2))
                    return
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
                pass
            time.sleep(10)
        raise RuntimeError(f"API readiness timed out after {timeout}s; use logs to inspect each rank")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", choices=["plan", "check", "start", "wait", "status", "stop", "logs"])
    parser.add_argument("--config", type=Path, help="explicit reviewed site JSON; required for every operation")
    parser.add_argument("--plan", action="store_true", help="print commands offline; no SSH, GPU or filesystem mutation")
    parser.add_argument("--qualification", type=Path)
    parser.add_argument("--nccl-library", type=Path)
    parser.add_argument("--model-dir")
    parser.add_argument("--drafter-dir")
    parser.add_argument("--cache-root")
    for name in ("context", "parallel", "multi-window", "cache-entries", "max-tokens"):
        parser.add_argument("--" + name, type=int)
    for name in ("cache-gib", "reserve-gib"):
        parser.add_argument("--" + name, type=float)
    parser.add_argument("--port", type=int)
    parser.add_argument("--protocol", choices=["auto", "Simple", "LL", "LL128"])
    parser.add_argument("--thread-thresholds")
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--no-wait", action="store_true")
    args = parser.parse_args()
    if args.plan:
        if args.action not in (None, "plan"):
            parser.error("--plan cannot be combined with a host action")
        args.action = "plan"
    if args.action is None or args.config is None:
        parser.error("choose an action and explicitly supply --config")
    cfg = load_config(args.config)
    for name in ("context", "parallel", "multi_window", "cache_entries", "max_tokens", "cache_gib", "reserve_gib", "protocol", "thread_thresholds"):
        if getattr(args, name) is not None:
            cfg["profile"][name] = getattr(args, name)
    if args.parallel is not None and args.multi_window is None:
        cfg["profile"]["multi_window"] = 64 if args.parallel > 4 else 32
    cfg["profile"] = validate_profile(cfg["profile"])
    if args.port is not None:
        if not 1 <= args.port <= 65535 or args.port == cfg["bootstrap_port"]:
            parser.error("--port must be a valid API port distinct from bootstrap_port")
        cfg["api_port"] = args.port
    if args.nccl_library is not None:
        cfg["library"] = str(args.nccl_library.absolute())
    for name in ("model_dir", "drafter_dir", "cache_root"):
        default = str(Path(cfg["model_root"]) / ("glm53-exl3" if name == "model_dir" else "dflash2"))
        value = getattr(args, name) or cfg.get(name) or (cfg["cache_root"] if name == "cache_root" else default)
        if not Path(value).is_absolute() or any(c in value for c in (",", "\n", "\r")):
            parser.error("model/cache paths must be absolute without commas/newlines")
        cfg[name] = value
    validate_local_storage(cfg)
    op = Operator(cfg)
    if args.action == "plan":
        library = library_path(cfg)
        digest = hashlib.sha256(library.read_bytes()).hexdigest() if library.is_file() else "BUILD_AND_QUALIFY_LIBRARY_SHA256"
        print("# Offline plan only; fresh qualification required before start. No SSH/GPU/state writes.")
        for rank, node in enumerate(cfg["nodes"]):
            cache, command = op.command(rank, library, digest)
            print("# rank", rank, node["name"], "target", node["ssh"], "local cache", cache)
            print(shlex.join(command))
        return
    doc_networks = [ipaddress.IPv4Network(prefix) for prefix in ("192.0.2.0/24", "198.51.100.0/24", "203.0.113.0/24")]
    if any(ipaddress.IPv4Address(n["management_address"]) in prefix for n in cfg["nodes"] for prefix in doc_networks):
        parser.error("example management addresses are documentation-only; configure your real hosts before host operations")
    if args.action == "status":
        print(json.dumps(op.parallel(op.inspect), indent=2)); return
    if args.action == "stop":
        print(json.dumps(op.parallel(op.stop_one), indent=2)); return
    if args.action == "logs":
        for rank, node in enumerate(cfg["nodes"]):
            print(node["name"])
            print(op.remote(rank, shlex.join(["docker", "logs", "--tail", "100", cfg["container_name"]]) + " 2>&1"))
        return
    if args.action == "wait":
        op.wait(args.timeout); return
    library, digest = op.library_identity()
    receipt = args.qualification or state_dir(cfg) / "collective-qualification.json"
    qualified = op.qualification(receipt, digest)
    checks = op.parallel(lambda rank: op.preflight(rank, library, digest))
    op.event("preflight_complete", library_sha256=digest, qualification=str(receipt), hosts=checks)
    (op.state / "preflight-latest.json").write_text(json.dumps({"hosts": checks, "library_sha256": digest,
                                                              "qualification": qualified}, indent=2) + "\n")
    if args.action == "check":
        print(json.dumps(checks, indent=2)); return
    for name, check in checks.items():
        if check["existing_container"] is not None:
            raise RuntimeError(f"{name}: container already exists; inspect status and stop its own deployment before starting")
        if check["memory_available_gib"] < 64:
            raise RuntimeError(f"{name}: initial TP4 launch requires at least64 GiB MemAvailable")
    started = []
    try:
        for rank in (3, 2, 1, 0):
            node = cfg["nodes"][rank]
            cache, command = op.command(rank, library, digest)
            op.event("container_start_begin", host=node["name"], cache=str(cache), command=command)
            started.append(rank)
            cid = op.remote(rank, shlex.join(["mkdir", "-p", str(cache)]) + "\n" + shlex.join(command)).strip()
            op.event("container_start_complete", host=node["name"], container_id=cid)
            print(f"{node['name']}: rank{rank} started ({cid[:12]})", flush=True)
        if not args.no_wait:
            op.wait(args.timeout)
    except BaseException:
        for rank in reversed(started):
            try:
                op.stop_one(rank)
            except Exception as error:
                op.event("failed_start_cleanup_error", rank=rank, error=str(error))
        raise


if __name__ == "__main__":
    # This file retains the shared operator implementation. Its public CLI
    # delegates to the isolated recipe's complete qualification/startup gates.
    from experiment import main as experiment_main
    experiment_main()
