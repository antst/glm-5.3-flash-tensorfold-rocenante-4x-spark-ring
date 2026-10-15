#!/usr/bin/env python3
"""Configuration and transport contract; importing this module changes no state."""
import ipaddress
import hashlib
import json
import math
from pathlib import Path
import re

IMAGE_DIGEST = "sha256:14b5072e8e9918f1080153f57321415a387bc7d78cf9c9b44a7aeae2d4399ccf"
IMAGE = "ghcr.io/miaai-lab/glm-5.3-flash-exl3-2x-dgx-sparks-tensorfold@" + IMAGE_DIGEST
MODEL_PINS = {
    "glm53-exl3": ("Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold", "078455ffe6472f9a52fbc1139f58b9db2881b25c"),
    "dflash2": ("incoai/GLM-5.3-Flash-DFlash2", "bf582e4eacc1810f76656d1811693ff6c6737d2a"),
}
PROFILE = {"context": 1048576, "parallel": 8, "multi_window": 64,
           "cache_entries": 32, "max_tokens": 32768, "cache_gib": 24.0,
           "reserve_gib": 20.0, "protocol": "auto", "thread_thresholds": None,"nccl_channels":4}
PEER_HCA_MAPS = ("1=0/2,2=0/3,3=1/3", "0=1/3,2=0/2,3=0/3",
                 "0=1/2,1=1/3,3=0/2", "0=0/2,1=1/2,2=1/3")


def ordered_hcas(node):
    return [next(p['hca'] for p in node['fabric'] if (p['physical_port'],p['rail'])==(port,rail))
            for rail in (0,1) for port in (0,1)]


def _name(value, label):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]*", value):
        raise ValueError(f"{label} must be an explicit interface/name without whitespace")
    return value


def _absolute(value, label):
    path = Path(value)
    if not path.is_absolute() or any(c in str(path) for c in (",", "\n", "\r")):
        raise ValueError(f"{label} must be an absolute path without commas/newlines")
    return str(path)


def validate_profile(profile):
    p = {**PROFILE, **profile}
    for field in ("context", "parallel", "multi_window", "cache_entries", "max_tokens"):
        if type(p[field]) is not int:
            raise ValueError(f"profile.{field} must be an integer")
    if "multi_window" not in profile:
        p["multi_window"] = 64 if p["parallel"] > 4 else 32
    if not 1024 <= p["context"] <= 1048576 or not 0 < p["max_tokens"] <= p["context"]:
        raise ValueError("context must be 1024..1048576 and max_tokens 1..context")
    if not 1 <= p["parallel"] <= 8 or p["multi_window"] not in range(16, 65, 8) or p["cache_entries"] < 1:
        raise ValueError("parallel 1..8, window 16..64 in steps of8, and positive cache_entries required")
    if any(type(p[k]) not in (int, float) or not math.isfinite(p[k]) for k in ("cache_gib", "reserve_gib")):
        raise ValueError("cache_gib/reserve_gib must be finite numbers")
    if p["cache_gib"] <= 0 or p["reserve_gib"] < 0:
        raise ValueError("cache_gib must be positive; reserve_gib must be nonnegative")
    if "kept_bytes_gib" in p:
        value = p["kept_bytes_gib"]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError("profile.kept_bytes_gib must be a finite nonnegative number")
    for field in ("keep_reasoning_entries", "keep_reasoning_mb"):
        if field in p and (type(p[field]) is not int or p[field] < 0):
            raise ValueError(f"profile.{field} must be a nonnegative integer")
    if p["protocol"] not in ("auto", "Simple", "LL", "LL128"):
        raise ValueError("protocol must be auto, Simple, LL or LL128")
    if p["thread_thresholds"] is not None and not re.fullmatch(r"[0-9]+(?: [0-9]+)*", p["thread_thresholds"]):
        raise ValueError("thread_thresholds must be space-separated unsigned integers")
    if type(p['nccl_channels'])!=int or p['nccl_channels'] not in (4,8,16):
        raise ValueError('nccl_channels must be4,8 or16; every choice requires fresh qualification')
    return p


def load_config(path):
    """Read/validate JSON without SSH, directory creation or GPU imports."""
    config_path = Path(path).absolute()
    cfg = json.loads(config_path.read_text())
    cfg["_config_path"] = str(config_path)
    cfg["shared_root"] = _absolute(cfg["shared_root"], "shared_root")
    cfg["model_root"] = _absolute(cfg.get("model_root", "/var/tmp/spark-ring-models"), "model_root")
    cfg["cache_root"] = _absolute(cfg.get("cache_root", "/var/tmp/spark-ring-tensorfold-cache"), "cache_root")
    validate_local_storage(cfg)
    cfg.setdefault("state_root", str(Path(cfg["shared_root"]) / "state"))
    cfg["state_root"] = _absolute(cfg["state_root"], "state_root")
    cfg.setdefault("library", "lib/libnccl.so.2")
    cfg.setdefault("image", IMAGE)
    if cfg["image"] != IMAGE:
        raise ValueError("This baseline requires the immutable pinned Mia image; an upgrade requires review and fresh qualification")
    cfg.setdefault("container_name", "spark-ring-glm53-tp4")
    cfg.setdefault("recipe_id", "tensorfold-ring-tp4-public-v1")
    _name(cfg["container_name"], "container_name")
    _name(cfg["recipe_id"], "recipe_id")
    cfg.setdefault("bootstrap_port", 29551)
    cfg.setdefault("api_port", 8888)
    if any(type(cfg[k]) is not int or not 1 <= cfg[k] <= 65535 for k in ("bootstrap_port", "api_port")):
        raise ValueError("bootstrap_port/api_port must be valid TCP ports")
    if cfg["bootstrap_port"] == cfg["api_port"]:
        raise ValueError("bootstrap and API ports must differ")
    cfg.setdefault("fabric_address_range", "10.77.0.0/16")
    address_range = ipaddress.IPv4Network(cfg["fabric_address_range"])
    cfg.setdefault("fabric_prefixlen", 24)
    cfg.setdefault("mtu", 9000)
    if cfg["fabric_prefixlen"] != 24 or cfg["mtu"] != 9000:
        raise ValueError("This qualified recipe uses /24 fabric subnets and MTU9000")
    nodes = cfg["nodes"]
    if len(nodes) != 4:
        raise ValueError("Exactly four nodes ordered by rank are required")
    names, management = [], []
    for rank, node in enumerate(nodes):
        names.append(_name(node["name"], f"nodes[{rank}].name"))
        address = str(ipaddress.IPv4Address(node["management_address"]))
        management.append(address)
        if ipaddress.IPv4Address(address) in address_range:
            raise ValueError("Management address must be outside the isolated fabric address range")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]*@" + re.escape(address), node["ssh"]):
            raise ValueError("Each node.ssh must explicitly be USER@management_address")
        node.setdefault("host_key_alias", node["name"])
        _name(node["host_key_alias"], "host_key_alias")
        _name(node["management_interface"], "management_interface")
        rails = node["fabric"]
        if len(rails) != 4 or len({f["hca"] for f in rails}) != 4 or len({f["interface"] for f in rails}) != 4:
            raise ValueError(f"{node['name']}: four unique HCA/netdev functions are mandatory")
        if {(f["physical_port"], f["rail"]) for f in rails} != {(0, 0), (0, 1), (1, 0), (1, 1)}:
            raise ValueError("Each physical port requires both PCIe rails0/1")
        for f in rails:
            _name(f["hca"], "hca")
            _name(f["interface"], "interface")
            if f["interface"] == node["management_interface"]:
                raise ValueError("Fabric and management interfaces must differ")
            f.setdefault("prefixlen", cfg["fabric_prefixlen"])
            if f["prefixlen"] != cfg["fabric_prefixlen"] or ipaddress.IPv4Address(f["address"]) not in address_range:
                raise ValueError("Fabric addresses must be in fabric_address_range with the configured prefix")
            interface = ipaddress.IPv4Interface(f"{f['address']}/{f['prefixlen']}")
            if interface.ip in (interface.network.network_address, interface.network.broadcast_address):
                raise ValueError("Fabric address must be a usable host address")
            if type(f["peer_rank"]) is not int or not 0 <= f["peer_rank"] < 4 or f["peer_rank"] == rank:
                raise ValueError("Fabric peer_rank must identify another configured rank")
        for port in (0, 1):
            if len({f["peer_rank"] for f in rails if f["physical_port"] == port}) != 1:
                raise ValueError("Both functions of a physical port must connect to the same neighbor")
        if len({f["peer_rank"] for f in rails}) != 2:
            raise ValueError("Each ring node requires exactly two neighbors")
    if len(set(names)) != 4 or len(set(management)) != 4:
        raise ValueError("Node names and management addresses must be unique")
    addresses = [f["address"] for n in nodes for f in n["fabric"]]
    if len(set(addresses)) != 16:
        raise ValueError("The sixteen fabric function addresses must be unique")
    subnets = [str(ipaddress.IPv4Interface(f"{f['address']}/{f['prefixlen']}").network)
               for n in nodes for f in n["fabric"]]
    if any(subnets.count(network) != 2 for network in set(subnets)):
        raise ValueError("Every isolated fabric subnet must contain exactly the two configured peer functions")
    for rank, node in enumerate(nodes):
        for f in node["fabric"]:
            peers = [p for p in nodes[f["peer_rank"]]["fabric"] if p["interface"] == f["peer_interface"]]
            if len(peers) != 1:
                raise ValueError("peer_interface must identify one peer fabric function")
            peer = peers[0]
            if peer["peer_rank"] != rank or peer["peer_interface"] != f["interface"] or peer["rail"] != f["rail"]:
                raise ValueError("Fabric links must be reciprocal and use matching PCIe rails")
            if ipaddress.IPv4Interface(f"{f['address']}/{f['prefixlen']}").network != ipaddress.IPv4Interface(f"{peer['address']}/{peer['prefixlen']}").network:
                raise ValueError("Each directly connected peer rail needs the same isolated subnet")
    reached, pending = set(), [0]
    while pending:
        rank = pending.pop()
        if rank not in reached:
            reached.add(rank)
            pending.extend(f["peer_rank"] for f in nodes[rank]["fabric"])
    if len(reached) != 4:
        raise ValueError("Fabric must be a connected four-node ring")
    cfg["profile"] = validate_profile(cfg.get("profile", {}))
    ssh = cfg.setdefault("ssh_options", {})
    for key in ("known_hosts_file", "identity_file"):
        if ssh.get(key):
            p = Path(ssh[key])
            ssh[key] = str(p if p.is_absolute() else config_path.parent / p)
    library_path(cfg)
    if cfg.get('runtime_source_root'):
        cfg['runtime_source_root'] = _absolute(cfg['runtime_source_root'], 'runtime_source_root')
    from engine_binding import identity as engine_identity, flags as engine_flags
    engine_flags(cfg)
    engine_identity(cfg)
    if cfg.get('tier') is not None:
        t=cfg['tier']
        required={'root','gib','min_free_gib','min_tokens','turns','host_mib','read_host_mib','stage_mib','read_blocks'}
        if not isinstance(t,dict) or set(t)!=required: raise ValueError('Exact NVMe tier settings required')
        _absolute(t['root'],'tier.root')
        if not str(t['root']).startswith('/var/tmp/spark-ring-nvme-tier'): raise ValueError('Owned local NVMe tier path required')
        if type(t['gib']) is not int or not 16<=t['gib']<=256: raise ValueError('Bounded tier.gib16..256')
        if t['min_free_gib']!=64 or t['min_tokens']!=8192 or t['turns']!=4: raise ValueError('Pinned tier retention/disk floor required')
        if t['host_mib']!=512 or t['read_host_mib']!=256 or t['stage_mib']!=256 or t['read_blocks']!=16: raise ValueError('Pinned tier staging budget required')
        if cfg.get('engine_flags',{}).get('TF_GLM_POOL_TOKENS') not in ('6098944','262144','524288'): raise ValueError('Explicit tier RAM pool ceiling required')
    return cfg


def library_path(config):
    path = Path(config["library"])
    path = path if path.is_absolute() else Path(config["shared_root"]) / path
    if path.name != "libnccl.so.2":
        raise ValueError("library must name the reviewed libnccl.so.2 alias")
    return path


def state_dir(config):
    return Path(config["state_root"])


def validate_local_storage(config):
    """Do not put weight-download or compilation locks below the shared repo."""
    shared = Path(config["shared_root"]).resolve()
    for name in ("model_root", "cache_root", "model_dir", "drafter_dir"):
        if config.get(name):
            path = Path(config[name]).resolve()
            if path == shared or shared in path.parents:
                raise ValueError(f"{name} must be local storage outside shared_root; shared download/JIT locks are unsupported")


def ssh_args(node, config=None):
    """Strict per-command SSH; trust must already exist, never populated here."""
    argv = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
            "-o", "ServerAliveInterval=15", "-o", "IdentitiesOnly=yes",
            "-o", "IdentityAgent=none", "-o", "UpdateHostKeys=no",
            "-o", "StrictHostKeyChecking=yes", "-o", "HostKeyAlias=" + node["host_key_alias"]]
    options = (config or {}).get("ssh_options", {})
    if options.get("known_hosts_file"):
        argv += ["-o", "UserKnownHostsFile=" + options["known_hosts_file"]]
    if options.get("identity_file"):
        argv += ["-i", options["identity_file"]]
    return argv + [node["ssh"]]


def topology_sha256(config):
    nodes = [{k: n[k] for k in ("name", "management_address", "management_interface", "fabric")}
             for n in config["nodes"]]
    data = {"nodes": nodes, "mtu": config["mtu"], "fabric_prefixlen": config["fabric_prefixlen"],
            "fabric_address_range": config["fabric_address_range"]}
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def transport_environment(env):
    names = {k for k in env if k.startswith("NCCL_") and not k.startswith("NCCL_DEBUG")}
    names |= {"NCCL_PROTO", "NCCL_THREAD_THRESHOLDS", "NCCL_IB_GID_INDEX",
              "TF_GLM_COMM", "TF_GLM_HC_EXCHANGE", "GLOO_SOCKET_IFNAME", "TORCH_USE_RTLD_GLOBAL"}
    from engine_binding import FLAG_VALUES
    names |= set(FLAG_VALUES) & set(env)
    names |= {k for k in env if k.startswith('B12X_ROCE_')}
    return {key: env.get(key) for key in sorted(names)}


def environment(config, rank, digest):
    node, profile = config["nodes"][rank], config["profile"]
    env = {
        "TF_NCCL_LIB": "/opt/spark-ring/lib/libnccl.so.2", "LD_PRELOAD": "/opt/spark-ring/lib/libnccl.so.2",
        "TORCH_USE_RTLD_GLOBAL": "1", "SPARK_RING_NCCL_SHA256": digest,
        "SPARK_RING_RANK": str(rank), "SPARK_RING_IMAGE_DIGEST": IMAGE_DIGEST,
        "MASTER_ADDR": config["nodes"][0]["management_address"], "MASTER_PORT": str(config["bootstrap_port"]),
        "NCCL_VERSION": "2.32.3", "NCCL_SWITCHLESS_RING_ONLY": "1", "NCCL_ALGO": "Ring",
        "NCCL_NET": "IB", "NCCL_NET_PLUGIN": "none", "NCCL_IB_DISABLE": "0",
        "NCCL_IB_HCA": "=" + ",".join(hca + ":1" for hca in ordered_hcas(node)),
        "NCCL_IB_ROCE_VERSION_NUM": "2", "NCCL_IB_ADDR_FAMILY": "AF_INET",
        "NCCL_IB_ADDR_RANGE": config["fabric_address_range"], "NCCL_IB_SUBNET_PREFIX_LEN": str(config["fabric_prefixlen"]),
        "NCCL_IB_SUBNET_AWARE_ROUTING": "1", "NCCL_IB_MERGE_NICS": "0",
        "NCCL_IB_EXTENDED_IPV4_GIDS": "1", "NCCL_IB_PRESERVE_PCI_DOMAIN": "1", "NCCL_IB_ROUTE_DIAGNOSTICS": "1",
        "NCCL_CROSS_NIC": "1", "NCCL_MIN_NCHANNELS": str(profile['nccl_channels']), "NCCL_MAX_NCHANNELS": str(profile['nccl_channels']),
        "NCCL_IB_QPS_PER_CONNECTION": "1", "NCCL_SOCKET_IFNAME": "=" + node["management_interface"],
        "NCCL_CUMEM_ENABLE": "0", "GLOO_SOCKET_IFNAME": node["management_interface"],
        "NCCL_DEBUG": "INFO", "NCCL_DEBUG_SUBSYS": "INIT,NET,GRAPH",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HOME": "/cache/huggingface",
        "TORCH_EXTENSIONS_DIR": "/cache/torch_extensions", "MAX_JOBS": "8", "TORCH_CUDA_ARCH_LIST": "12.1",
        "TENSORFOLD_NO_UPDATE_CHECK": "1", "TF_GLM_COMM": "nccl", "TF_GLM_HC_SPLIT": "1",
        "TF_GLM_HC_EXCHANGE": "gather", "TF_GLM_PREFILL_OVERLAP": "2", "TF_GLM_DENSE": "q4",
        "TF_GLM_KV": "fp8", "TF_GLM_MTP": "auto", "TF_GLM_KDA_CHUNKED": "1",
        "TF_GLM_COPY_DRAFTS": "1", "TF_GLM_COPY_MAX": "15", "TF_GLM_DFLASH_POLICY": "fnc7:0.3",
        "TF_GLM_WIDE_GRAPHS": "16", "TF_GLM_COPY_REPLY_MATCH": "16", "TF_GLM_MULTI_LONE": "0",
        "TF_GLM_MULTI_WINDOW": str(profile["multi_window"]), "TF_GLM_CACHE_ENTRIES": str(profile["cache_entries"]),
        "TF_GLM_CACHE_GIB": str(profile["cache_gib"]), "TENSORFOLD_MEMORY_RESERVE_GIB": str(profile["reserve_gib"]),
        "TF_GLM_DISPLAY_KV_MIB": "0", "TF_GLM_CLEAR_THINKING": "0", "TF_GLM_EFFORT_TAIL": "0",
        "TF_GLM_L2PF": "1", "TF_GLM_EXL3_LOADS": "nc", "TF_GLM_EXL3_DEC_ORDER": "0",
        "TF_GLM_SHARED_PREFIX": "1", "TF_GLM_MULTI_PREFILL": "1", "TF_GLM_STREAM_SMOOTH": "0",
        "TF_GLM_FILL_BUDGET_MS": "200", "TF_GLM_FILL_DRAFTS": "1", "TF_GLM_MAX_QUEUED": "4",
        "TENSORFOLD_NUCLEUS_UNION": "1",
        "B12X_ROCE_HCA": ",".join(ordered_hcas(node)),
        "B12X_ROCE_OPPOSITE_PATHS": "2", "B12X_ROCE_PEER_HCA_MAP": PEER_HCA_MAPS[rank],
        "B12X_ROCE_GID_INDEX": "3", "B12X_ROCE_PEER_TIMEOUT_S": "600",
        "B12X_ROCE_FORWARD_WINDOW_BYTES":str(config.get('experiment',{}).get('forward_window_bytes',131072)),
        "B12X_ROCE_FORWARD_CHUNK_BYTES":str(config.get('experiment',{}).get('forward_chunk_bytes',32768)),
    }
    if profile["protocol"] != "auto":
        env["NCCL_PROTO"] = profile["protocol"]
    if profile["thread_thresholds"]:
        env["NCCL_THREAD_THRESHOLDS"] = profile["thread_thresholds"]
    if "kept_bytes_gib" in profile:
        env["TF_GLM_KEPT_BYTES_GIB"] = str(profile["kept_bytes_gib"])
    for field, name in (("keep_reasoning_entries", "TF_GLM_KEEP_REASONING"), ("keep_reasoning_mb", "TF_GLM_KEEP_REASONING_MB")):
        if field in profile:
            env[name] = str(profile[field])
    from engine_binding import environment as engine_environment
    env.update(engine_environment(config))
    return env
