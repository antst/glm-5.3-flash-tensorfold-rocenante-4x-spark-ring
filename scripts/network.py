#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Plan and own the minimal RoCEnante hardware mesh; preserve the IP underlay.

Planning never contacts hosts or writes files. Inventory is read-only on hosts.
Only explicit prepare/apply/rollback actions change owned state. No software
forwarding, firewall, NAT, static neighbors or appliance services are installed.
"""
import argparse
import dataclasses
import datetime
import hashlib
import importlib.util
import inspect
import ipaddress
import json
import os
from pathlib import Path
import secrets
import sys
import time
import uuid
sys.dont_write_bytecode=True

from common import load_config, state_dir, topology_sha256
from cluster import Operator

PLANNER_SHA256 = 'b201ede9554bd6a02cc09a07a6accf242a63b2d46744bab9132ee4a7e318dbda'
SOURCE_COMMIT = '6d27dd10a9f97da21278ae504ea6235f8634f303'
MARKER_SOURCE_SHA256 = '8684a6961b8e86aa474fa2310ff71e4cdf219a63a72ceb5593b2f95e54812792'


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def resolved(config, name, default):
    path = Path(config.get('network', {}).get(name, default))
    return path if path.is_absolute() else Path(config['shared_root']) / path


def planner(config):
    path = resolved(config, 'upstream_planner', 'third_party/sparkring/fabric.py')
    expected = config.get('network', {}).get('upstream_planner_sha256', PLANNER_SHA256)
    if expected != PLANNER_SHA256 or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError('Network planner is not the exact audited upstream source')
    spec = importlib.util.spec_from_file_location('rocenante_pinned_fabric', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def canonical(config):
    for rank, node in enumerate(config['nodes']):
        for port in node['fabric']:
            expected = (rank + (1 if port['physical_port'] == 0 else -1)) % 4
            if port['peer_rank'] != expected:
                raise ValueError('Current prepared RoCEnante requires f0=next rank and f1=previous rank')
            wanted = ('rocep1s0f' if port['rail'] == 0 else 'roceP2p1s0f') + str(port['physical_port'])
            if port['hca'] != wanted:
                raise ValueError('Prepared upstream transport uses its exact four canonical HCA names')


def make_plan(config, inventory=None):
    """Use upstream dataclasses/validation/selection entirely in memory."""
    canonical(config)
    f = planner(config)
    ranks, missing = [], []
    for rank, node in enumerate(config['nodes']):
        ports = []
        for port in sorted(node['fabric'], key=lambda p: (p['physical_port'], p['rail'])):
            facts = (inventory or {}).get('hosts', {}).get(node['name'], {}).get('fabric', {}).get(port['hca'], {})
            mac = port.get('mac') or facts.get('mac')
            if not isinstance(mac, str) or not f._MAC.fullmatch(mac):
                missing.append(node['name'] + '/' + port['hca']); continue
            direction = 'clockwise' if port['physical_port'] == 0 else 'counter_clockwise'
            ports.append(f.Port(direction, port['rail'], port['interface'], port['hca'], port['address'],
                                mac.lower(), port['peer_rank'], f._opposite_direction(direction), port['rail']))
        ranks.append(f.Rank(rank, node['ssh'], node['management_interface'], tuple(ports)))
    if missing:
        return {'schema': 'tensorfold-rocenante-network-plan/v1', 'complete': False,
                'requires_inventory': missing, 'hosts': [n['name'] for n in config['nodes']],
                'topology_sha256': topology_sha256(config), 'host_changes': False}
    identity = {'topology_sha256': topology_sha256(config), 'ports': [dataclasses.asdict(r) for r in ranks]}
    # Keyword construction below avoids silently relying on an upstream field order.
    topology = f.FabricTopology(source_path=Path('<in-memory>'), sha256=digest(identity), group_id=77,
        socket_direct_functions=2, shared_diagonal_flow_label=True, endpoint_route_strategy='adjacent_gateway',
        tc_rule_identity_overrides=(), flow_label_base=16383, standard_ether_type=0x0800, marked_ether_type=0x88b5,
        standard_udp_destination_port=4791, roce_gid_index=3, bounded_runtime_seconds=7200,
        expected_ethernet_mtu=9000, expected_roce_mtu=4096, host_helper_path='/UNSUPPORTED/external-fabric-orchestrator',
        remote_state_root='/run/tensorfold-rocenante', ranks=tuple(ranks))
    f._validate_topology(topology)
    selected = f.build_rocenante_plan(f.build_plan(topology))
    nodes = []
    marker = str(resolved(config, 'marker_binary', 'lib/mesh/mlx5-rdma-tx-marker'))
    for rank, node in enumerate(config['nodes']):
        routes = [r for r in selected.routes if r.source_rank == rank]
        rules = [r for r in selected.tc_rules if r.intermediate_rank == rank]
        markers = [m for m in selected.markers if m.source_rank == rank]
        if len(routes) != 2 or len(rules) != 2 or len(markers) != 2 or any(r.permanent_final_neighbor for r in routes):
            raise ValueError('Unexpected current profile diagonal resource geometry')
        nodes.append({'rank': rank, 'name': node['name'], 'node': node,
            'routes': [{'object': dataclasses.asdict(r), 'add': f.route_command(r, add=True), 'delete': f.route_command(r, add=False)} for r in routes],
            'rules': [{'object': dataclasses.asdict(r), 'add': f.tc_rule_command(r, add=True), 'delete': f.tc_rule_command(r, add=False)} for r in rules],
            'markers': [{'device': m.rdma_device, 'argv': [marker, '--device', m.rdma_device, '--source-port', '65535',
                        '--replacement-ethertype', '0x88b5', '--attach', '--managed']} for m in markers],
            'qdiscs': sorted({r.ingress_netdev for r in rules})})
    return {'schema': 'tensorfold-rocenante-network-plan/v1', 'complete': True,
        'topology_sha256': topology_sha256(config), 'upstream_plan_sha256': selected.sha256,
        'upstream_planner_sha256': PLANNER_SHA256, 'source_commit': SOURCE_COMMIT,
        'hosts': [n['name'] for n in config['nodes']], 'nodes': nodes, 'inventory': f.rocenante_inventory(selected),
        'neighbors': [], 'software_forwarding_changes': [], 'firewall_changes': [], 'address_changes': [],
        'marker_binary': marker, 'recipe_id': config['recipe_id'], 'plan_id': digest({'identity': identity, 'plan': selected.sha256})}


def firewall_policy_sha256(text):
    """Hash policy while ignoring only save timestamps and chain counters."""
    import hashlib, re
    lines=[]
    for line in text.splitlines():
        if line.startswith('#'):continue
        # iptables-save without -c still emits live counters on chain declarations.
        # Do not remove numeric text from rules or quoted rule comments.
        lines.append(re.sub(r'^(:\S+\s+\S+)\s+\[\d+:\d+\]$',r'\1 [COUNTERS]',line))
    return hashlib.sha256('\n'.join(lines).encode()).hexdigest()


def inventory_worker(data):
    """Read facts only; command failures remain visible instead of becoming defaults."""
    import datetime, hashlib, ipaddress, json, os, pathlib, re, subprocess, time
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    node = data['node']; commands = []
    def query(argv, required=True):
        start = time.monotonic(); result = subprocess.run(argv, text=True, capture_output=True, timeout=20)
        commands.append({'argv': argv, 'returncode': result.returncode, 'seconds': round(time.monotonic()-start, 4)})
        if result.returncode and required: raise RuntimeError('Read-only query failed: '+repr(argv)+': '+result.stderr.strip())
        return {'stdout': result.stdout, 'stderr': result.stderr, 'returncode': result.returncode}
    def rows(argv): return json.loads(query(argv)['stdout'] or '[]')
    addresses = {r['ifname']: r for r in rows(['ip', '-j', 'address', 'show'])}
    facts = {'started_utc': started, 'node': node['name'], 'management_address': node['management_address'],
             'boot_id': pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
             'management_interface': node['management_interface'], 'fabric': {},
             'routes4': rows(['ip', '-j', '-4', 'route', 'show', 'table', 'all']),
             'routes6': rows(['ip', '-j', '-6', 'route', 'show', 'table', 'all']),
             'devlink': query(['devlink', '-j', 'dev', 'param', 'show'], required=False),
             'devlink_statistics': query(['devlink','-s','-j','dev','show'],required=False),
             'affinity_cpus': sorted(os.sched_getaffinity(0)),
             'gpu_processes': query(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits']),
             'rdma_qps': query(['rdma', '-j', 'resource', 'show', 'qp']), 'firewall_sha256': {}, 'sysctls': {}}
    facts['configuration_sha256']={}
    for directory in ('/etc/netplan','/etc/NetworkManager/system-connections','/run/NetworkManager/system-connections'):
        for path in pathlib.Path(directory).rglob('*'):
            if path.is_file() and not path.is_symlink():facts['configuration_sha256'][str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    mgmt = addresses[node['management_interface']]
    if node['management_address'] not in [a['local'] for a in mgmt.get('addr_info', []) if a['family']=='inet']:
        raise RuntimeError('Management identity mismatch')
    for port in node['fabric']:
        base = pathlib.Path('/sys/class/infiniband')/port['hca']; iface = addresses[port['interface']]
        pci = (base/'device').resolve().name
        associated = sorted(p.name for p in (base/'device/net').iterdir())
        gids = []
        for entry in (base/'ports/1/gids').iterdir():
            try:
                kind = (base/'ports/1/gid_attrs/types'/entry.name).read_text().strip()
                value = entry.read_text().strip()
            except OSError:
                # Unallocated driver slots return EINVAL; they are not entries.
                continue
            if kind=='RoCE v2' and ipaddress.IPv6Address(value).ipv4_mapped == ipaddress.IPv4Address(port['address']):gids.append(int(entry.name))
        verbs=query(['ibv_devinfo','-d',port['hca'],'-i','1'])
        match=re.search(r'active_mtu:\s+([^\n]+)',verbs['stdout'])
        f = {'mac': iface.get('address'), 'interface': port['interface'], 'hca': port['hca'], 'pci': pci,
             'associated_netdevs': associated, 'mtu': iface.get('mtu'), 'flags': iface.get('flags'),
             'ipv4': [{'address': a['local'], 'prefixlen': a['prefixlen']} for a in iface.get('addr_info', []) if a['family']=='inet'],
             'gid_indices': sorted(gids), 'state': (base/'ports/1/state').read_text().strip(),
             'active_mtu': match.group(1).strip() if match else None, 'verbs':verbs,
             'driver': query(['ethtool', '-i', port['interface']]),
             'features':query(['ethtool','-k',port['interface']]),
             'tc_filters': rows(['tc', '-j', '-s', 'filter', 'show', 'dev', port['interface'], 'ingress']),
             'tc_qdiscs': rows(['tc', '-j', 'qdisc', 'show', 'dev', port['interface']]),
             'nm_connection': query(['nmcli', '-g', 'GENERAL.CON-UUID,GENERAL.CONNECTION', 'device', 'show', port['interface']], required=False)}
        for key in ('forwarding', 'rp_filter'):
            path=pathlib.Path('/proc/sys/net/ipv4/conf')/port['interface']/key; facts['sysctls'][str(path)]=path.read_text().strip()
        facts['fabric'][port['hca']]=f
    facts['sysctls']['/proc/sys/net/ipv4/ip_forward']=pathlib.Path('/proc/sys/net/ipv4/ip_forward').read_text().strip()
    for scope in ('all','default'):
        for key in ('forwarding','rp_filter'):
            path=pathlib.Path('/proc/sys/net/ipv4/conf')/scope/key;facts['sysctls'][str(path)]=path.read_text().strip()
    facts['firewall_hash_scheme']='iptables-save-policy-v2'
    for binary in ('iptables-save','ip6tables-save'):
        facts['firewall_sha256'][binary]=firewall_policy_sha256(query([binary])['stdout'])
    facts['commands']=commands; facts['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    return facts


def collect(config):
    operator=Operator(config)
    def one(rank):
        code=inspect.getsource(firewall_policy_sha256)+'\n'+inspect.getsource(inventory_worker)+'\nprint(__import__("json").dumps(inventory_worker('+repr({'node':config['nodes'][rank]})+')))'
        return json.loads(operator.remote_python(rank,code,timeout=150))
    return {'schema':'tensorfold-rocenante-network-inventory/v1','utc':utc(),
            'topology_sha256':topology_sha256(config),'hosts':operator.parallel(one)}


def fabric_users(rows,hcas,allowed_pids=()):
    """rdma JSON is flat on Ubuntu; also handle older nested qp arrays."""
    unexpected=[];allowed=set(allowed_pids)
    for row in rows:
        if row.get('ifname','').split('/')[0] not in hcas:continue
        entries=row.get('qp') if isinstance(row.get('qp'),list) else [row]
        for entry in entries:
            pid=entry.get('pid')
            if pid is None and entry.get('comm')=='ib_core' and entry.get('type')=='GSI':continue
            if pid not in allowed:unexpected.append({**entry,'ifname':row.get('ifname')})
    return unexpected


def reload_statistics(document,pci):
    entry=document.get('dev',{}).get(pci,{})
    count=entry
    for key in ('stats','reload','driver_reinit','unspecified'):
        count=count.get(key) if isinstance(count,dict) else None
    failed=entry.get('reload_failed',False) if entry else None
    if type(count)!=int or count<0 or type(failed)!=bool:raise RuntimeError('Driver reload statistics unavailable')
    return {'driver_reinit':count,'failed':failed}


def validate_inventory(config, inventory, *, idle=False, prepared=False, allowed_pids_by_host=None):
    if inventory.get('topology_sha256') != topology_sha256(config):
        raise ValueError('Inventory belongs to another topology')
    for node in config['nodes']:
        host=inventory['hosts'][node['name']]
        if idle and host['gpu_processes']['stdout'].strip():raise RuntimeError(node['name']+': active GPU workload')
        qps=json.loads(host['rdma_qps']['stdout'] or '[]')
        devices={p['hca'] for p in node['fabric']}
        if idle and fabric_users(qps,devices,(allowed_pids_by_host or {}).get(node['name'],())):
            raise RuntimeError(node['name']+': active fabric QPs')
        devlink=json.loads(host['devlink']['stdout'])['param']
        statistics=json.loads(host['devlink_statistics']['stdout'])
        for port in node['fabric']:
            observed=host['fabric'][port['hca']]
            if (observed['associated_netdevs']!=[port['interface']] or observed['gid_indices']!=[3] or
                observed['ipv4']!=[{'address':port['address'],'prefixlen':24}] or observed['mtu']!=9000 or
                'ACTIVE' not in observed['state'] or not isinstance(observed['active_mtu'],str) or not observed['active_mtu'].startswith('4096')):
                raise RuntimeError(node['name']+': fabric identity/GID/MTU mismatch '+port['hca'])
            params={r['name']:{v['cmode']:v['value'] for v in r['values']} for r in devlink['pci/'+observed['pci']]}
            if params.get('flow_steering_mode',{}).get('runtime')!='hmfs':raise RuntimeError('Current mesh requires tested hmfs steering')
            if params.get('hairpin_num_queues',{}).get('driverinit')!=4:raise RuntimeError('Current mesh requires four hairpin queues')
            if prepared and params.get('hairpin_queue_size',{}).get('driverinit')!=8192:raise RuntimeError('Run explicit stopped-stack network prepare first')
            reload=reload_statistics(statistics,'pci/'+observed['pci'])
            if prepared and (reload['failed'] or reload['driver_reinit']<1):raise RuntimeError('Hairpin driverinit values have not been applied by a successful reload')
            offload=next((line.split(':',1)[1].strip() for line in observed['features']['stdout'].splitlines() if line.startswith('hw-tc-offload:')),None)
            if offload is None or (prepared and not offload.startswith('on')):raise RuntimeError('Hardware TC offload unavailable/disabled')


def hardware_rule_attestation(rule, rows):
    """Verify the exact upstream hardware match, rewrite bytes and egress.

    Derived from SparkRing inspect_fabric.py validate_rule (Apache-2.0).
    """
    # tc prints EtherType/pedit words as bare hex (for example '08000000').
    # Filter handles are JSON integers or ordinary prefixed numeric strings.
    def number(value):return int(value,16) if isinstance(value,str) else int(value)
    def handle(value):return int(value,0) if isinstance(value,str) else int(value)
    matches=[r for r in rows if r.get('pref')==rule['preference'] and r.get('kind')=='flower' and
             handle(r.get('options',{}).get('handle',0))==rule['handle']]
    if len(matches)!=1:raise RuntimeError('Missing or conflicting owned TC flower identity')
    options=matches[0]['options'];keys=options.get('keys',{})
    if (keys.get('src_mac','').lower()!=rule['match_ethernet_source'] or
        keys.get('dst_mac','').lower()!=rule['match_ethernet_destination'] or number(keys.get('eth_type',0))!=0x88b5 or
        options.get('skip_sw') is not True or options.get('in_hw') is not True):
        raise RuntimeError('TC match or hardware-only placement differs')
    actions=options.get('actions',[])
    if [a.get('kind') for a in actions]!=['pedit','pedit','mirred']:raise RuntimeError('TC action family differs')
    destination=bytes.fromhex(rule['rewrite_ethernet_destination'].replace(':',''))
    expected=[(12,0x08000000,0xFFFF),(0,int.from_bytes(destination[:4],'big'),0),(4,int.from_bytes(destination[4:]+b'\0\0','big'),0xFFFF)]
    edits=[item for a in actions[:2] for item in a.get('keys',[])]
    actual=[(i.get('offset'),number(i.get('val',0)),number(i.get('mask',0))) for i in edits]
    if actual!=expected or any(i.get('htype')!='eth' or i.get('cmd')!='set' for i in edits):raise RuntimeError('TC rewrite bytes differ')
    if actions[2].get('to_dev')!=rule['egress_netdev'] or actions[2].get('mirred_action')!='redirect':raise RuntimeError('TC redirect differs')
    if any(any(a.get('stats',{}).get(field,0) for field in ('sw_packets','sw_bytes','drops')) for a in actions):
        raise RuntimeError('TC reports software transit or drops')
    return {'in_hw':True,'skip_sw':True,'rule_identity':dict(rule),'software_counter_available':all('sw_packets' in a.get('stats',{}) for a in actions),
            'actions_stats':[a.get('stats',{}) for a in actions]}


def hardware_payload_delta(before,after):
    """Prove positive hardware transit on every one of the eight diagonal rules.

    Use one action's counters per rule to avoid counting the same packet three
    times. Aggregate counters are hardware-only when skip_sw and in_hw both
    attest the exact rule. Missing software counters are reported explicitly.
    """
    if before.get('plan_id')!=after.get('plan_id') or before.get('network_boot_ids')!=after.get('network_boot_ids'):
        raise RuntimeError('Hardware mesh plan/boot generation changed during payload proof')
    hosts=before.get('hosts',{})
    if len(hosts)!=4 or set(hosts)!=set(after.get('hosts',{})):raise RuntimeError('Hardware payload proof requires all four hosts')
    results=[]
    for name in sorted(hosts):
        prior=hosts[name].get('rules',[]);current=after['hosts'][name].get('rules',[])
        if len(prior)!=2 or len(current)!=2:raise RuntimeError('Hardware payload proof requires two diagonal rules per host')
        for index,(old,new) in enumerate(zip(prior,current)):
            if (not old.get('rule_identity') or old.get('rule_identity')!=new.get('rule_identity') or
                any(row.get(key) is not True for row in (old,new) for key in ('skip_sw','in_hw'))):
                raise RuntimeError('TC rule identity or hardware-only placement changed')
            old_stats=old.get('actions_stats',[]);new_stats=new.get('actions_stats',[])
            if len(old_stats)!=3 or len(new_stats)!=3:raise RuntimeError('Expected exact three-action TC statistics')
            candidates=[];software_available=True
            for action,(a,b) in enumerate(zip(old_stats,new_stats)):
                for field in ('sw_packets','sw_bytes','drops'):
                    for stats in (a,b):
                        value=stats.get(field,0)
                        if type(value)!=int or value!=0:raise RuntimeError('TC reports software transit, drops, or invalid error counters')
                    if field.startswith('sw_') and (field not in a or field not in b):software_available=False
                for packet_field,byte_field,source in (('hw_packets','hw_bytes','explicit_hardware'),('packets','bytes','aggregate_hardware_only')):
                    if any(field not in stats for field in (packet_field,byte_field) for stats in (a,b)):continue
                    values=[stats[field] for stats in (a,b) for field in (packet_field,byte_field)]
                    if any(type(value)!=int or value<0 for value in values):raise RuntimeError('Invalid TC hardware counters')
                    packets=b[packet_field]-a[packet_field];byte_count=b[byte_field]-a[byte_field]
                    if packets<0 or byte_count<0:raise RuntimeError('TC hardware counter reset')
                    if packets>0 and byte_count>0:candidates.append((source!='explicit_hardware',action,source,packets,byte_count))
            if not candidates:raise RuntimeError(name+': diagonal rule '+str(index)+' carried no measured hardware payload')
            _,action,source,packets,byte_count=min(candidates)
            results.append({'host':name,'rule_index':index,'rule_identity':new['rule_identity'],'action_index':action,
                            'counter_source':source,'hardware_packets':packets,'hardware_bytes':byte_count,
                            'software_counter_available':software_available,'software_fallback_forbidden':True,
                            'drops':0,'hardware_payload_passed':True})
    return {'passed':True,'rules':results,'rule_count':len(results)}


def guardian_worker(configuration):
    """Volatile authenticated health guard; stop only this recipe's local model.

    Each host watches its exact marker PIDs and polls all four local health
    endpoints. A marker/guardian failure therefore stops every owned rank.
    This is a failure guard, not automatic recovery or a persistent service.
    """
    import datetime, hashlib, http.server, json, os, pathlib, signal, subprocess, threading, time, urllib.request
    root=pathlib.Path(configuration['directory']); stop=threading.Event(); local={'ready':False,'failure':None,'all_peers_ready':False}
    def event(action,**extra):
        with (root/'guardian-events.jsonl').open('a') as out:out.write(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'action':action,**extra})+'\n')
        os.chmod(root/'guardian-events.jsonl',0o600)
    def markers_ready():
        if pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()!=configuration['boot_id']:return False
        records=json.loads((root/'journal.json').read_text())['markers']
        if len(records)!=2:return False
        for record in records:
            if record.get('boot_id')!=configuration['boot_id']:return False
            proc=pathlib.Path('/proc')/str(record['pid'])
            fields=(proc/'stat').read_text().rsplit(')',1)[1].split()
            if fields[0] not in ('R','S','D','I') or int(fields[19])!=record['start_ticks']:return False
            if (proc/'cmdline').read_bytes().rstrip(b'\0').decode().split('\0')!=record['argv']:return False
        return True
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path!='/health' or self.headers.get('Authorization')!='Bearer '+configuration['token']:
                self.send_response(403);self.end_headers();return
            response=json.dumps({'ready':local['ready'],'failure':local['failure'],'all_peers_ready':local['all_peers_ready'],
                                 'plan_id':configuration['plan_id'],'rank':configuration['rank']}).encode()
            self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(response)));self.end_headers();self.wfile.write(response)
        def do_POST(self):
            if self.path!='/fault' or self.headers.get('Authorization')!='Bearer '+configuration['token']:
                self.send_response(403);self.end_headers();return
            local['failure']='owned-runtime-reported-native-fault';local['ready']=False;local['all_peers_ready']=False
            event('owned-runtime-native-fault');self.send_response(200);self.end_headers()
        def log_message(self,*args):pass
    server=http.server.ThreadingHTTPServer((configuration['address'],configuration['port']),Handler)
    server.daemon_threads=True
    def stop_model(reason):
        # No inference/clock commands. Refuse to act on another deployment's name.
        name=configuration['container_name']
        result=subprocess.run(['docker','inspect',name],text=True,capture_output=True,timeout=10)
        if result.returncode:return
        container=json.loads(result.stdout)[0];labels=container['Config'].get('Labels',{})
        if labels.get('spark-ring.recipe')!=configuration['recipe_id'] or labels.get('spark-ring.root')!=configuration['shared_root']:
            event('refuse-unowned-model-stop',reason=reason);return
        if container['State'].get('Running'):
            event('stop-owned-model-on-mesh-fault',reason=reason,container=name)
            result=subprocess.run(['docker','stop','--time','10',name],text=True,capture_output=True,timeout=20)
            event('stop-result',returncode=result.returncode)
    def monitor():
        bad=0;armed=False;started=time.monotonic()
        while not stop.wait(1):
            try:local['ready']=markers_ready()
            except (OSError,ValueError,KeyError):local['ready']=False
            okay=local['ready']
            for address in configuration['peers']:
                try:
                    request=urllib.request.Request('http://'+address+':'+str(configuration['port'])+'/health',headers={'Authorization':'Bearer '+configuration['token']})
                    with urllib.request.urlopen(request,timeout=.5) as response:row=json.load(response)
                    okay=okay and row.get('ready') is True and row.get('plan_id')==configuration['plan_id'] and row.get('failure') is None
                except Exception:okay=False
            local['all_peers_ready']=okay
            if okay:armed=True;bad=0
            else:bad+=1
            if (armed and bad>=3) or (not armed and time.monotonic()-started>60):
                local['failure']='native-mesh-health-failed';local['ready']=False;event('mesh-failure');stop_model(local['failure'])
                # Remain alive, permanently unhealthy, until explicit owned rollback.
                while not stop.wait(1):stop_model(local['failure'])
    signal.signal(signal.SIGTERM,lambda *_:stop.set());signal.signal(signal.SIGINT,lambda *_:stop.set())
    thread=threading.Thread(target=monitor,daemon=True);thread.start();event('guard-started',rank=configuration['rank'],port=configuration['port'])
    server.timeout=.5
    while not stop.is_set():server.handle_request()
    server.server_close();event('guard-stopped')


def mesh_worker(data, action):
    """Privileged worker owns exact task resources; preserves unknown existing state."""
    import datetime, fcntl, hashlib, json, os, pathlib, re, signal, socket, subprocess, tarfile, time, urllib.request
    if os.geteuid()!=0:raise RuntimeError('Mesh worker requires sudo -n -s')
    nodeplan=data['nodeplan'];node=nodeplan['node'];recipe=data['recipe_id']
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.:-]*',recipe):raise ValueError('Unsafe owner')
    directory=pathlib.Path('/var/tmp/tensorfold-rocenante-mesh')/recipe
    boot_id=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    if not re.fullmatch(r'[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}',boot_id):raise RuntimeError('Invalid kernel boot identity')
    expected={'recipe_id':recipe,'plan_id':data['plan_id'],'rank':nodeplan['rank']}
    def query(argv,check=True):
        p=subprocess.run(argv,text=True,capture_output=True,timeout=30)
        if check and p.returncode:raise RuntimeError(repr(argv)+': '+p.stderr.strip())
        return p
    def write(path,value):
        temporary=path.with_name(path.name+'.tmp');temporary.write_text(json.dumps(value,indent=2)+'\n');os.chmod(temporary,0o600);os.replace(temporary,path)
    def owner():
        if directory.parent.is_symlink() or directory.parent.stat().st_uid!=0 or directory.parent.stat().st_mode&0o022:
            raise RuntimeError('Mesh state parent must be a root-owned directory without write access for others')
        if directory.is_symlink() or not directory.is_dir() or directory.stat().st_uid!=0 or directory.stat().st_mode&0o077:
            raise RuntimeError('Mesh state must be a root-only owned directory')
        recorded=json.loads((directory/'owner.json').read_text())
        if {key:recorded.get(key) for key in expected}!=expected or not isinstance(recorded.get('boot_id'),str):
            raise RuntimeError('Mesh state belongs to a different plan or lacks its original boot identity')
    def event(name,**extra):
        with (directory/'changes.jsonl').open('a') as out:out.write(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'action':name,**extra})+'\n')
        os.chmod(directory/'changes.jsonl',0o600)
    def mutate(argv):
        event('command-start',argv=argv);result=query(argv,False)
        event('command-finish',argv=argv,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)
        if result.returncode:raise RuntimeError('Mesh mutation failed: '+repr(argv)+': '+result.stderr.strip())
    def idle():
        if query(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits']).stdout.strip():raise RuntimeError('Active GPU workload')
        rows=json.loads(query(['rdma','-j','resource','show','qp']).stdout or '[]')
        devices={p['hca'] for p in node['fabric']}
        if directory.exists() or directory.is_symlink():owner()
        records=json.loads((directory/'journal.json').read_text())['markers'] if (directory/'journal.json').is_file() else []
        known=set()
        for marker in records:
            if marker_identity(marker,False):known.add(marker['pid'])
        if fabric_users(rows,devices,known):raise RuntimeError('Active unowned fabric QPs')
        for proc in pathlib.Path('/proc').iterdir():
            if not proc.name.isdigit() or int(proc.name) in known:continue
            try:argv=(proc/'cmdline').read_bytes().rstrip(b'\0').decode().split('\0')
            except (OSError,UnicodeError):continue
            if '--source-port' in argv and '--device' in argv and '--attach' in argv:
                if argv[argv.index('--source-port')+1]=='65535' and argv[argv.index('--device')+1] in devices:
                    raise RuntimeError('Unowned source marker already occupies a selected fabric HCA')
    def route_identity(item):
        o=item['object'];rows=json.loads(query(['ip','-j','-4','route','show','exact',o['destination_ipv4']+'/32']).stdout or '[]')
        if not rows:return None
        if len(rows)!=1 or rows[0].get('dev')!=o['source_netdev'] or rows[0].get('gateway')!=o['gateway_ipv4'] or rows[0].get('prefsrc')!=o['source_ipv4']:
            raise RuntimeError('Conflicting existing endpoint route')
        return rows[0]
    def route_present(item):return route_identity(item) is not None
    def rule_present(item):
        o=item['object'];rows=json.loads(query(['tc','-j','-s','filter','show','dev',o['ingress_netdev'],'ingress']).stdout or '[]')
        matches=[r for r in rows if r.get('pref')==o['preference'] and r.get('kind')=='flower' and int(str(r.get('options',{}).get('handle','0')),0)==o['handle']]
        if not matches:return False
        return hardware_rule_attestation(o,rows)
    def marker_identity(record,require_ready=True):
        if record.get('boot_id')!=boot_id:
            if require_ready:raise RuntimeError('Owned marker belongs to a previous boot; run stopped-stack rollback')
            return False
        proc=pathlib.Path('/proc')/str(record['pid'])
        try:
            fields=(proc/'stat').read_text().rsplit(')',1)[1].split()
            if int(fields[19])!=record['start_ticks']:raise RuntimeError('Marker PID identity changed')
            if fields[0]=='Z':
                if require_ready:raise RuntimeError('Owned marker has exited')
                return False
            argv=(proc/'cmdline').read_bytes().rstrip(b'\0').decode().split('\0')
            if int(fields[19])!=record['start_ticks'] or argv!=record['argv'] or hashlib.sha256((proc/'exe').read_bytes()).hexdigest()!=data['marker_sha256']:raise RuntimeError('Marker PID identity changed')
        except FileNotFoundError:
            if require_ready:raise RuntimeError('Owned marker is no longer running')
            return False
        if fields[0] not in ('R','S','D','I'):raise RuntimeError('Marker exited/stopped')
        return True
    if action=='preflight':
        idle()
        if (directory/'journal.json').is_file() and json.loads((directory/'journal.json').read_text()).get('boot_id')!=boot_id:
            raise RuntimeError('Owned mesh journal belongs to a previous boot; run stopped-stack rollback first')
        if hashlib.sha256(pathlib.Path(data['marker_binary']).read_bytes()).hexdigest()!=data['marker_sha256']:raise RuntimeError('Shared marker hash mismatch')
        for item in nodeplan['routes']:route_present(item)
        for item in nodeplan['rules']:rule_present(item)
        if not (directory/'journal.json').is_file() or not json.loads((directory/'journal.json').read_text()).get('guardian'):
            with socket.socket() as listener:
                listener.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
                listener.bind((node['management_address'],data['health_port']))
        return {'preflight':True}
    if action=='rollback-preflight':
        idle();owner()
        return {'idle':True,'boot_id':boot_id}
    if action=='stage':
        idle()
        if directory.exists() or directory.is_symlink():owner();return {'staged':True,'existing':True}
        if directory.parent.is_symlink():raise RuntimeError('Symlink mesh state parent')
        directory.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
        if directory.parent.stat().st_uid!=0 or directory.parent.stat().st_mode&0o022:raise RuntimeError('Unsafe mesh state parent')
        directory.mkdir(mode=0o700)
        write(directory/'owner.json',{**expected,'boot_id':boot_id});write(directory/'before.json',data['before'])
        with tarfile.open(directory/'configuration-before.tar.gz','w:gz') as archive:
            for base in ('/etc/netplan','/etc/NetworkManager/system-connections','/run/NetworkManager/system-connections'):
                if pathlib.Path(base).exists():archive.add(base,arcname=base.lstrip('/'))
        os.chmod(directory/'configuration-before.tar.gz',0o600)
        write(directory/'journal.json',{'boot_id':boot_id,'resources':[],'markers':[],'driver_changes':[],'guardian':None,'retired_boots':[]})
        event('staged-root-only-backup',owner=expected);return {'staged':True,'directory':str(directory)}
    owner()
    with (directory/'lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        journal=json.loads((directory/'journal.json').read_text())
        def checkpoint():write(directory/'journal.json',journal)
        if journal.get('boot_id')!=boot_id and action!='rollback':
            raise RuntimeError('Owned mesh journal belongs to a previous boot; run stopped-stack rollback first')
        if action=='prepare':
            if journal['markers'] or journal.get('guardian'):raise RuntimeError('Rollback owned mesh before a stopped-stack driver preparation')
            idle();parameters=json.loads(query(['devlink','-j','dev','param','show']).stdout)['param']
            for port in node['fabric']:
                pci='pci/'+data['before']['fabric'][port['hca']]['pci']
                names={p['name']:{v['cmode']:v['value'] for v in p['values']} for p in parameters[pci]}
                if names['flow_steering_mode']['runtime']!='hmfs' or names['hairpin_num_queues']['driverinit']!=4:raise RuntimeError('Unexpected steering/queue count; refusing unreviewed changes')
                old=names['hairpin_queue_size']['driverinit']
                if old not in (8192,1024):raise RuntimeError('Unexpected existing hairpin queue size')
                reload=reload_statistics(json.loads(query(['devlink','-s','-j','dev','show',pci]).stdout),pci)
                if old!=8192 or reload['failed'] or reload['driver_reinit']==0:
                    change={'pci':pci,'name':'hairpin_queue_size','cmode':'driverinit','before':old,'after':8192,'pending':True,'reload_before':reload}
                    journal['driver_changes'].append(change);checkpoint()
                    if old!=8192:mutate(['devlink','dev','param','set',pci,'name','hairpin_queue_size','value','8192','cmode','driverinit'])
                    mutate(['devlink','dev','reload',pci,'action','driver_reinit'])
                    after=reload_statistics(json.loads(query(['devlink','-s','-j','dev','show',pci]).stdout),pci)
                    if after['failed'] or after['driver_reinit']!=reload['driver_reinit']+1:raise RuntimeError('Driver reload did not complete exactly once')
                    change['pending']=False;change['reload_after']=after;checkpoint()
                features=query(['ethtool','-k',port['interface']]).stdout
                offload=next((line.split(':',1)[1].strip() for line in features.splitlines() if line.startswith('hw-tc-offload:')),None)
                if offload is None or (offload.startswith('off') and '[fixed]' in offload):raise RuntimeError('Hardware TC offload cannot be enabled')
                if offload.startswith('off'):
                    journal.setdefault('offload_changes',[]).append({'interface':port['interface'],'before':'off','after':'on'});checkpoint()
                    mutate(['ethtool','-K',port['interface'],'hw-tc-offload','on'])
            return {'prepared':True,'driver_changes':journal['driver_changes'],'offload_changes':journal.get('offload_changes',[])}
        if action=='apply':
            idle()
            if hashlib.sha256(pathlib.Path(data['marker_binary']).read_bytes()).hexdigest()!=data['marker_sha256']:raise RuntimeError('Marker binary changed')
            # Refuse all conflicts before any network object is added.
            for item in nodeplan['routes']:route_present(item)
            for item in nodeplan['rules']:rule_present(item)
            for item in nodeplan['routes']:
                if route_present(item):continue
                record={'kind':'route','item':item,'pending':True};journal['resources'].append(record);checkpoint()
                mutate(item['add']);record['pending']=False;checkpoint()
            for interface in nodeplan['qdiscs']:
                rows=json.loads(query(['tc','-j','qdisc','show','dev',interface]).stdout or '[]')
                if not any(r.get('kind')=='clsact' for r in rows):
                    if any(r.get('kind')=='ingress' for r in rows):raise RuntimeError('Conflicting ingress qdisc')
                    record={'kind':'qdisc','interface':interface,'pending':True};journal['resources'].append(record);checkpoint()
                    mutate(['tc','qdisc','add','dev',interface,'clsact']);record['pending']=False;checkpoint()
            for item in nodeplan['rules']:
                if rule_present(item):continue
                record={'kind':'rule','item':item,'pending':True};journal['resources'].append(record);checkpoint()
                mutate(item['add']);record['pending']=False;checkpoint();rule_present(item)
            for item in nodeplan['markers']:
                previous=[m for m in journal['markers'] if m['device']==item['device']]
                if previous:marker_identity(previous[0]);continue
                stdout=directory/(item['device']+'.stdout.jsonl');stderr=directory/(item['device']+'.stderr.log')
                with stdout.open('w') as out,stderr.open('w') as err:
                    process=subprocess.Popen(item['argv'],stdout=out,stderr=err,start_new_session=True)
                os.chmod(stdout,0o600);os.chmod(stderr,0o600)
                ticks=int((pathlib.Path('/proc')/str(process.pid)/'stat').read_text().rsplit(')',1)[1].split()[19])
                record={'device':item['device'],'pid':process.pid,'start_ticks':ticks,'boot_id':boot_id,'argv':item['argv'],'stdout':str(stdout),'stderr':str(stderr)}
                journal['markers'].append(record);checkpoint();event('start-owned-marker',**record)
                deadline=time.monotonic()+10
                while not stdout.read_text().strip():
                    if process.poll() is not None:raise RuntimeError('Marker failed: '+stderr.read_text())
                    if time.monotonic()>deadline:raise RuntimeError('Marker readiness timed out')
                    time.sleep(.05)
                ready=json.loads(stdout.read_text().splitlines()[0])
                if (ready.get('device')!=item['device'] or any(ready.get(k) is not True for k in ('attached','managed','rdma_tx_matcher','sigint_sigterm_cleanup')) or
                    ready.get('action')!='ethertype_rewrite' or ready.get('source_port')!=65535 or ready.get('lifetime_seconds','missing') is not None):raise RuntimeError('Native marker readiness differs')
                record['ready']=ready;checkpoint()
            if not journal.get('guardian'):
                health={'directory':str(directory),'address':node['management_address'],'port':data['health_port'],
                        'token':data['health_token'],'peers':data['management_addresses'],'rank':nodeplan['rank'],
                        'plan_id':data['plan_id'],'container_name':data['container_name'],'recipe_id':recipe,'shared_root':data['shared_root'],'boot_id':boot_id}
                write(directory/'health.json',health)
                script=data['guardian_source']+'\nguardian_worker(__import__("json").loads(__import__("pathlib").Path(__file__).with_name("health.json").read_text()))\n'
                (directory/'guardian.py').write_text(script);os.chmod(directory/'guardian.py',0o600)
                args=['/usr/bin/python3','-B',str(directory/'guardian.py')]
                with (directory/'guardian.stdout.log').open('a') as out,(directory/'guardian.stderr.log').open('a') as err:
                    process=subprocess.Popen(args,stdout=out,stderr=err,start_new_session=True)
                os.chmod(directory/'guardian.stdout.log',0o600);os.chmod(directory/'guardian.stderr.log',0o600)
                ticks=int((pathlib.Path('/proc')/str(process.pid)/'stat').read_text().rsplit(')',1)[1].split()[19])
                journal['guardian']={'pid':process.pid,'start_ticks':ticks,'boot_id':boot_id,'argv':args};checkpoint();event('start-owned-guardian',pid=process.pid)
            return {'applied':True,'directory':str(directory),'markers':journal['markers']}
        if action=='check':
            for item in nodeplan['routes']:
                if not route_present(item):raise RuntimeError('Owned mesh route missing')
            rules=[rule_present(item) for item in nodeplan['rules']]
            if not all(rules):raise RuntimeError('Owned mesh TC rule missing')
            if len(journal['markers'])!=2:raise RuntimeError('Exactly two owned markers required')
            for record in journal['markers']:marker_identity(record)
            request=urllib.request.Request('http://'+node['management_address']+':'+str(data['health_port'])+'/health',headers={'Authorization':'Bearer '+data['health_token']})
            with urllib.request.urlopen(request,timeout=2) as response:health=json.load(response)
            if health.get('ready') is not True or health.get('all_peers_ready') is not True or health.get('failure') is not None or health.get('plan_id')!=data['plan_id']:
                raise RuntimeError('Four-rank native mesh guardian is not healthy')
            return {'ready':True,'plan_id':data['plan_id'],'boot_id':boot_id,'rules':rules,'markers':journal['markers'],'health':health}
        if action=='rollback':
            idle();warnings=[];removed_routes=[]
            if journal.get('boot_id')!=boot_id:
                # Volatile resources from the previous boot are no longer owned.
                # Retain the history and never signal a PID reused after reboot.
                retired={key:journal.get(key) for key in ('boot_id','resources','markers','guardian')}
                retired['retired_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
                journal.setdefault('retired_boots',[]).append(retired)
                journal.update(boot_id=boot_id,resources=[],markers=[],guardian=None)
                checkpoint();event('retire-previous-boot-resources',previous_boot_id=retired['boot_id'],current_boot_id=boot_id)
            guardian=journal.get('guardian')
            if guardian:
                if guardian.get('boot_id')!=boot_id:raise RuntimeError('Guardian record boot identity differs from active journal')
                proc=pathlib.Path('/proc')/str(guardian['pid'])
                if proc.exists():
                    fields=(proc/'stat').read_text().rsplit(')',1)[1].split()
                    if int(fields[19])!=guardian['start_ticks'] or (fields[0]!='Z' and (proc/'cmdline').read_bytes().rstrip(b'\0').decode().split('\0')!=guardian['argv']):
                        raise RuntimeError('Guardian PID identity changed')
                    if fields[0]!='Z':
                        os.kill(guardian['pid'],signal.SIGTERM);event('stop-owned-guardian',pid=guardian['pid'])
                        for attempt in range(100):
                            if not proc.exists() or proc.joinpath('stat').read_text().rsplit(')',1)[1].split()[0]=='Z':break
                            time.sleep(.05)
                        else:raise RuntimeError('Owned guardian failed to stop')
                journal['guardian']=None;checkpoint()
            for record in reversed(journal['markers']):
                if marker_identity(record,False):
                    os.kill(record['pid'],signal.SIGTERM);event('stop-owned-marker',pid=record['pid'],device=record['device'])
                    for attempt in range(100):
                        proc=pathlib.Path('/proc')/str(record['pid'])
                        if not proc.exists() or proc.joinpath('stat').read_text().rsplit(')',1)[1].split()[0]=='Z':break
                        time.sleep(.05)
                    else:raise RuntimeError('Marker did not detach after SIGTERM')
            journal['markers']=[];checkpoint()
            for record in reversed(journal['resources']):
                if record['pending']:warnings.append('Retained ambiguous pending '+record['kind']);continue
                if record['kind']=='qdisc':
                    interface=record['interface']
                    if any(json.loads(query(['tc','-j','filter','show','dev',interface,direction]).stdout or '[]') for direction in ('ingress','egress')):
                        warnings.append('Retained nonempty qdisc '+interface);continue
                    mutate(['tc','qdisc','del','dev',interface,'clsact'])
                else:
                    item=record['item']
                    if record['kind']=='route':
                        actual=route_identity(item)
                        if actual is not None:
                            mutate(item['delete'])
                            removal={'node':node['name'],'owner':expected,'boot_id':boot_id,'route_identity':actual,
                                     'resource':record,'resource_sha256':hashlib.sha256(json.dumps(record,sort_keys=True,separators=(',',':')).encode()).hexdigest()}
                            removed_routes.append(removal);event('remove-owned-route',attestation=removal)
                    elif rule_present(item):mutate(item['delete'])
                journal['resources'].remove(record);checkpoint()
            if data.get('restore_driver'):
                for change in reversed(journal['driver_changes']):
                    # Installed devlink requires a parameter name after a PCI
                    # selector. Global show is supported and identifies it exactly.
                    parameters=json.loads(query(['devlink','-j','dev','param','show']).stdout)['param'][change['pci']]
                    names={p['name']:{v['cmode']:v['value'] for v in p['values']} for p in parameters}
                    current=names[change['name']][change['cmode']]
                    if current not in (change['before'],change['after']):raise RuntimeError('Driver parameter changed outside this recipe; refusing rollback overwrite')
                    if current!=change['before'] or change.get('pending') or change.get('rollback_pending'):
                        reload=reload_statistics(json.loads(query(['devlink','-s','-j','dev','show',change['pci']]).stdout),change['pci'])
                        change['rollback_pending']=True;checkpoint()
                        if current!=change['before']:mutate(['devlink','dev','param','set',change['pci'],'name',change['name'],'value',str(change['before']),'cmode',change['cmode']])
                        mutate(['devlink','dev','reload',change['pci'],'action','driver_reinit'])
                        after=reload_statistics(json.loads(query(['devlink','-s','-j','dev','show',change['pci']]).stdout),change['pci'])
                        if after['failed'] or after['driver_reinit']!=reload['driver_reinit']+1:raise RuntimeError('Driver rollback reload failed')
                        event('restore-owned-driver-value',change=change,reload_after=after)
                    journal['driver_changes'].remove(change);checkpoint()
                for change in list(journal.get('offload_changes',[])):
                    features=query(['ethtool','-k',change['interface']]).stdout
                    current=next((line.split(':',1)[1].strip().split()[0] for line in features.splitlines() if line.startswith('hw-tc-offload:')),None)
                    if current not in (change['before'],change['after']):raise RuntimeError('Hardware offload changed outside this recipe; refusing rollback overwrite')
                    if current!=change['before']:mutate(['ethtool','-K',change['interface'],'hw-tc-offload',change['before']])
                    journal['offload_changes'].remove(change);checkpoint()
            return {'rolled_back':not warnings,'warnings':warnings,'retained_driver_changes':journal['driver_changes'],
                    'removed_routes':removed_routes,'boot_id':boot_id}
    raise ValueError('Unsupported worker action')


def marker_identity(config):
    library=resolved(config,'marker_binary','lib/mesh/mlx5-rdma-tx-marker')
    receipt=resolved(config,'marker_receipt','lib/mesh/build-receipt.json')
    data=json.loads(receipt.read_text());expected=hashlib.sha256(library.read_bytes()).hexdigest()
    # Accept only the exact current managed-attachment source, with real CPU loader proof.
    if data.get('source_commit')!=SOURCE_COMMIT or data.get('source_sha256')!=MARKER_SOURCE_SHA256 or data.get('binary_sha256')!=expected:
        raise ValueError('Marker receipt does not identify the exact built managed helper')
    if data.get('cpu_loader_passed') is not True:raise ValueError('Marker CPU loader proof missing')
    return str(library),expected


def project_owned_removed_routes(before, after, removals, nodeplan, recipe_id, plan_id):
    """Exclude exact current-boot owned rows actually removed by the worker."""
    if not isinstance(before.get('boot_id'),str) or before['boot_id']!=after.get('boot_id'):
        raise RuntimeError('Host boot changed during rollback preservation')
    remaining=list(before['routes4']);proof=[]
    expected={'recipe_id':recipe_id,'plan_id':plan_id,'rank':nodeplan['rank']}
    for removal in removals:
        record=removal.get('resource',{})
        if (removal.get('owner')!=expected or removal.get('node')!=nodeplan['node']['name'] or
            removal.get('boot_id')!=before['boot_id'] or record.get('kind')!='route' or record.get('pending') is not False or
            record.get('item') not in nodeplan['routes'] or removal.get('resource_sha256')!=digest(record)):
            raise RuntimeError('Rollback route removal lacks exact journal/plan/boot ownership')
        row=removal.get('route_identity');o=record['item']['object']
        if (not isinstance(row,dict) or row.get('dst') not in (o['destination_ipv4'],o['destination_ipv4']+'/32') or
            row.get('dev')!=o['source_netdev'] or row.get('gateway')!=o['gateway_ipv4'] or row.get('prefsrc')!=o['source_ipv4']):
            raise RuntimeError('Removed route differs from the owned endpoint route')
        matches=[index for index,candidate in enumerate(remaining) if candidate==row]
        if len(matches)!=1 or row in after['routes4']:
            raise RuntimeError('Removed route must match one full before identity and be absent afterward')
        remaining.pop(matches[0]);proof.append(removal)
    return {**before,'routes4':remaining},proof


def preservation(before, after, *, planned_routes=(), allow_driver_reload=False):
    """Allow only the reviewed endpoint routes; never mask unrelated defaults/policy."""
    for key in ('configuration_sha256','sysctls','firewall_sha256'):
        if before.get(key)!=after.get(key):raise RuntimeError('Unplanned change to '+key)
    metric_changes=[]
    def connected_identity(row,facts):
        if row.get('protocol')!='kernel' or row.get('scope')!='link':return None
        for hca,port in facts['fabric'].items():
            for address in port['ipv4']:
                subnet=str(ipaddress.IPv4Network(address['address']+'/'+str(address['prefixlen']),strict=False))
                if row.get('dev')==port['interface'] and row.get('dst')==subnet and row.get('prefsrc')==address['address']:
                    return hca,'pci/'+port['pci']
        return None
    # NM allocates new automatic metrics after a verified driver reload. Admit
    # only that field on the exact configured connected route, never defaults.
    if allow_driver_reload:
        for prior in before['routes4']:
            identity=connected_identity(prior,before)
            if identity is None:continue
            hca,pci=identity
            matches=[row for row in after['routes4'] if connected_identity(row,after)==identity and row.get('dst')==prior.get('dst')]
            if len(matches)!=1:continue
            current=matches[0]
            if prior.get('metric')==current.get('metric'):continue
            if {k:v for k,v in prior.items() if k!='metric'}!={k:v for k,v in current.items() if k!='metric'}:
                raise RuntimeError('Connected fabric route changed beyond automatic metric')
            old=reload_statistics(json.loads(before['devlink_statistics']['stdout']),pci)
            new=reload_statistics(json.loads(after['devlink_statistics']['stdout']),pci)
            if old['failed'] or new['failed'] or new['driver_reinit']!=old['driver_reinit']+1:
                raise RuntimeError('Connected metric change lacks exactly one successful PCI driver reload')
            metric_changes.append({'hca':hca,'pci':pci,'destination':prior['dst'],'interface':prior['dev'],
                                   'prefsrc':prior['prefsrc'],'before_metric':prior.get('metric'),'after_metric':current.get('metric'),
                                   'reload_before':old,'reload_after':new})
    admitted={(row['destination'],row['interface'],row['prefsrc']) for row in metric_changes}
    def normalize(rows,ipv4=False):
        ignore={'expires','lastuse','used','cache'}
        return sorted(json.dumps({k:v for k,v in row.items() if k not in ignore and not
                                 (ipv4 and k=='metric' and (row.get('dst'),row.get('dev'),row.get('prefsrc')) in admitted)},sort_keys=True) for row in rows)
    # ip -j emits a host destination as a bare IPv4 on installed iproute2;
    # recognize only that host's two equivalent spellings, never its subnet.
    wanted={spelling for r in planned_routes for spelling in
            (r['object']['destination_ipv4'],r['object']['destination_ipv4']+'/32')}
    current=[r for r in after['routes4'] if r.get('dst') not in wanted]
    prior=[r for r in before['routes4'] if r.get('dst') not in wanted]
    if normalize(prior,True)!=normalize(current,True) or normalize(before['routes6'])!=normalize(after['routes6']):
        raise RuntimeError('Unplanned route change')
    for hca,port in before['fabric'].items():
        other=after['fabric'][hca]
        for key in ('mac','interface','hca','pci','associated_netdevs','mtu','ipv4','nm_connection'):
            if port[key]!=other[key]:raise RuntimeError('Unplanned fabric identity/profile change '+hca+'/'+key)
    return {'passed':True,'connected_metric_changes':metric_changes}


def operate(config, plan, inventory, action, *, restore_driver=False):
    if not plan.get('complete'):raise ValueError('An exact inventory-backed plan is required')
    binary,binary_sha=marker_identity(config)
    operator=Operator(config)
    port=config.get('network',{}).get('health_port',29661)
    if type(port)!=int or not 1024<=port<=65535 or port in (config['api_port'],config['bootstrap_port']):raise ValueError('Invalid distinct mesh health port')
    token_file=state_dir(config)/'mesh-health-token'
    if action in ('apply','prepare') and not token_file.exists():
        token_file.parent.mkdir(parents=True,exist_ok=True)
        try:
            descriptor=os.open(token_file,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
            with os.fdopen(descriptor,'w') as stream:stream.write(secrets.token_hex(32))
        except FileExistsError:pass
    token=token_file.read_text().strip() if token_file.exists() else ''
    before_inventory=inventory
    def one(rank, requested, restore=None):
        payload={'nodeplan':plan['nodes'][rank],'recipe_id':config['recipe_id'],'plan_id':plan['plan_id'],
                 'marker_binary':binary,'marker_sha256':binary_sha,'restore_driver':restore_driver if restore is None else restore,
                 'before':before_inventory['hosts'][config['nodes'][rank]['name']], 'health_port':port,'health_token':token,
                 'management_addresses':[n['management_address'] for n in config['nodes']],
                 'guardian_source':inspect.getsource(guardian_worker),'container_name':config['container_name'],'shared_root':config['shared_root']}
        code=inspect.getsource(fabric_users)+'\n'+inspect.getsource(reload_statistics)+'\n'+inspect.getsource(hardware_rule_attestation)+'\n'+inspect.getsource(mesh_worker)+'\npayload='+repr(payload)+'''\ntry:
 result=mesh_worker(payload,'''+repr(requested)+''')
except Exception as error:
 import sys
 print(type(error).__name__+': '+str(error),file=sys.stderr)
 sys.exit(1)
print(__import__("json").dumps(result))
'''
        return json.loads(operator.remote_python(rank,code,timeout=300))
    if action=='rollback':
        # The saved inventory binds the physical plan, not a stale boot's route
        # metrics. All-host idle checks precede any rollback worker mutation.
        before_inventory=collect(config)
        if make_plan(config,before_inventory)['plan_id']!=plan['plan_id']:
            raise RuntimeError('Actual physical identities changed since the rollback plan')
        operator.parallel(lambda rank:one(rank,'rollback-preflight'))
        save(state_dir(config)/'network-before-rollback.json',before_inventory)
        operator.event('mesh-rollback-fresh-before',plan_id=plan['plan_id'])
    if action in ('apply','prepare'):
        current=collect(config);validate_inventory(config,current,idle=True,prepared=action=='apply')
        if make_plan(config,current)['plan_id']!=plan['plan_id']:raise RuntimeError('Actual physical identities changed since plan')
        operator.parallel(lambda rank:one(rank,'preflight'))
        # All-host read-only preconditions precede even backup-directory creation.
        staged=operator.parallel(lambda rank:one(rank,'stage'));operator.event('mesh-backups-staged',hosts=staged)
    operator.event('mesh-action-start',mesh_action=action,plan_id=plan['plan_id'])
    try:
        result=operator.parallel(lambda rank:one(rank,action))
        if action in ('prepare','apply','rollback'):
            # Driver reinitialization can briefly remove links/GIDs. Retry only
            # observations, record every failure, and require a valid final state.
            deadline=time.monotonic()+60
            while True:
                try:
                    after=collect(config)
                    if action in ('prepare','apply'):validate_inventory(config,after,prepared=True)
                    break
                except Exception as error:
                    if action!='prepare' or time.monotonic()>=deadline:raise
                    operator.event('mesh-prepare-observation-not-ready',error=str(error));time.sleep(1)
            # Retain observations even when the guard rejects the action. This
            # receipt is explicitly unqualified and never used for admission.
            save(state_dir(config)/('network-observed-'+action+'.json'),
                 {'schema':'tensorfold-rocenante-action-observation/v1','qualified':False,
                  'preservation_verified':False,'utc':utc(),'action':action,'inventory':after})
            for rank,node in enumerate(config['nodes']):
                prior=before_inventory['hosts'][node['name']];removed=[]
                if action=='rollback':
                    if result[node['name']].get('rolled_back') is not True:
                        raise RuntimeError('Rollback retained ambiguous owned resources; inspect the journal')
                    prior,removed=project_owned_removed_routes(prior,after['hosts'][node['name']],
                        result[node['name']].get('removed_routes',[]),plan['nodes'][rank],config['recipe_id'],plan['plan_id'])
                result[node['name']]['preservation']=preservation(prior,after['hosts'][node['name']],
                             planned_routes=plan['nodes'][rank]['routes'] if action=='apply' else (),
                             allow_driver_reload=action=='prepare' or (action=='rollback' and restore_driver))
                if action=='rollback':result[node['name']]['preservation']['owned_removed_routes']=removed
                operator.event('mesh-preservation-passed',host=node['name'],report=result[node['name']]['preservation'])
            save(state_dir(config)/('network-after-'+action+'.json'),after)
        if action=='apply':
            deadline=time.monotonic()+55
            while True:
                try:
                    health=operator.parallel(lambda rank:one(rank,'check'));break
                except Exception as error:
                    if time.monotonic()>=deadline:raise
                    operator.event('mesh-guardian-observation-not-ready',error=str(error));time.sleep(1)
            for name in result:result[name]['live_guard']=health[name]
    except Exception as error:
        operator.event('mesh-action-failed',mesh_action=action,error=str(error))
        if action in ('apply','prepare'):
            cleanup={}
            for rank,node in enumerate(config['nodes']):
                try:cleanup[node['name']]=one(rank,'rollback',restore=action=='prepare')
                except Exception as e:cleanup[node['name']]={'error':str(e)}
            save(state_dir(config)/('network-failed-'+action+'-rollback.json'),cleanup)
        raise
    save(state_dir(config)/('network-'+action+'.json'),{'utc':utc(),'plan_id':plan['plan_id'],'hosts':result})
    operator.event('mesh-action-finish',mesh_action=action,hosts=result)
    return result


def live_guard(config):
    """Fresh four-rank exact route/TC/marker/guardian proof, safe with models running."""
    inventory=json.loads((state_dir(config)/'network-inventory.json').read_text())
    plan=make_plan(config,inventory)
    hosts=operate(config,plan,inventory,'check')
    return {'plan_id':plan['plan_id'],'hosts':hosts,
            'network_boot_ids':{name:row['boot_id'] for name,row in hosts.items()},
            'allowed_marker_pids_by_host':{name:[m['pid'] for m in row['markers']] for name,row in hosts.items()}}


def save(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp-'+uuid.uuid4().hex)
    temp.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');os.chmod(temp,0o600);os.replace(temp,path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',nargs='?',choices=('plan','inventory','check','prepare','apply','rollback'))
    parser.add_argument('--plan',action='store_true');parser.add_argument('--config',required=True)
    parser.add_argument('--inventory',help='explicit saved read-only inventory for exact MAC identities')
    parser.add_argument('--output',help='inventory receipt path (inventory action only)')
    parser.add_argument('--restore-driver',action='store_true',help='rollback only: restore journaled prior driver values with a stopped-stack reload')
    args=parser.parse_args()
    if args.plan:
        if args.action not in (None,'plan'):parser.error('--plan cannot be combined with a host action')
        args.action='plan'
    if not args.action:parser.error('explicit action is required')
    config=load_config(args.config)
    if args.action=='plan':
        inventory=json.loads(Path(args.inventory).read_text()) if args.inventory else None
        print(json.dumps(make_plan(config,inventory),indent=2,sort_keys=True));return
    if any(ipaddress.IPv4Address(n['management_address']) in ipaddress.IPv4Network('192.0.2.0/24') for n in config['nodes']):
        parser.error('Documentation-only addresses cannot be used for host actions')
    if args.action=='inventory':
        report=collect(config);path=Path(args.output) if args.output else state_dir(config)/'network-inventory.json'
        save(path,report);Operator(config).event('network-read-only-inventory',path=str(path))
        print(json.dumps({'saved':str(path),'hosts':list(report['hosts']),'host_changes':False}));return
    if not args.inventory:parser.error('Controlled actions require --inventory saved from the reviewed topology')
    inventory=json.loads(Path(args.inventory).read_text());plan=make_plan(config,inventory)
    if args.restore_driver and args.action!='rollback':parser.error('--restore-driver applies only to rollback')
    print(json.dumps(operate(config,plan,inventory,args.action,restore_driver=args.restore_driver),indent=2))


if __name__=='__main__':main()
