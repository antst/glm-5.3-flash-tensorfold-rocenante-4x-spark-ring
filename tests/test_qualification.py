# SPDX-License-Identifier: Apache-2.0
"""CPU-only negative qualification guards. No Torch, SSH, CUDA or NIC calls."""
import copy
import ast
import importlib.util
import io
import json
from pathlib import Path
from config_fixtures import config_file
import struct
import socket
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / filename)
    obj = importlib.util.module_from_spec(spec); spec.loader.exec_module(obj);return obj

qualifier = module('qualification_under_test', 'qualify.py')
probe = module('rocenante_probe_under_test', 'rocenante-probe.py')
HCAS = ['a', 'b', 'c', 'd']

def paths():
    rows = []
    for peer in (1, 2, 3):
        for path in (0, 1):
            rows.append(dict(peer_rank=peer, path_index=path, device=HCAS[(peer + path) % 4],
                qp_number=peer * 10 + path, remote_qp_number=peer * 100 + path,
                local_hca_index=(peer + path) % 4, remote_hca_index=path, physical_hops=1 if peer != 2 else 2,
                payload_writes=0, payload_bytes=0, physical_hop_payload_bytes=0,
                flag_writes=0, send_completions=0, completion_errors=0))
    return rows

def counters(value):
    return {'devices': {hca: {'pci_address':str(i), 'netdev':hca,
        'function_rdma': {'rx_vport_rdma_unicast_bytes':value, 'tx_vport_rdma_unicast_bytes':value,
                         'rx_vport_rdma_unicast_packets':value // 64, 'tx_vport_rdma_unicast_packets':value // 64},
        'hw_requests_and_errors': {field:0 for field in probe.ERROR_FIELDS}} for i,hca in enumerate(HCAS)}}


def geometry_report(blocks=8):
    """Complete synthetic transport evidence to isolate the geometry gate."""
    settings={'min_gather_bytes':1,'max_gather_bytes':16777216,'vendor_manifest_sha256':'vendor'}
    if blocks!=8:settings['native_blocks']=blocks
    before_paths=paths();after_paths=copy.deepcopy(before_paths)
    for row in after_paths:
        row.update(payload_writes=1,payload_bytes=1000,physical_hop_payload_bytes=1000)
    native={'native_active':True,'failure':None,
            'native_stats':{'error_seq':0,'abort_seq':0,'wait_stopped':False,'ctrl_seq':12,'completed_seq':12},
            'path_counters':after_paths,'selection_cache':'disabled','live_native_graphs':0,
            'adapter_sha256':qualifier.sha(ROOT/'scripts/rocenante_adapter.py'),
            'vendor_manifest_sha256':'vendor','gid_indices':[3,3,3,3],'hca_names':HCAS,
            'native_threads':512,'native_blocks':blocks,'runtime_native_threads':512,'runtime_native_blocks':blocks}
    report={'rank':0,'world':4,'library_sha256':'library','required_hcas':HCAS,
            'library':qualifier.CONTAINER_LIBRARY,'resolved_nccl':qualifier.CONTAINER_LIBRARY,
            'mapped_nccl':[qualifier.CONTAINER_LIBRARY],
            **{flag:True for flag in ('byte_exact_passed','cuda_graph_passed','mixed_size_passed','inplace_passed',
                                     'payload_all_hcas_passed','native_health_passed','teardown_passed')},
            'results':[],'native_final':copy.deepcopy(native),
            'extra_alias_results':[{'dtype':dtype,'correct':True,'cuda_graph_correct':True} for dtype in probe.DTYPES],
            'mixed_results':[{'correct':True,'graph_eager_nccl_ordering_passed':True} for _ in range(9)],
            'native_proxy':{'abi_version':6,'blob_bytes':456,'sha256':'a'*64}}
    for dtype in probe.DTYPES:
        for size in probe.case_sizes(settings):
            for inplace in (False,True):
                admitted=settings['min_gather_bytes']<=size<=settings['max_gather_bytes']
                row={'dtype':dtype,'send_bytes':size,'inplace':inplace,'correct':True,'cuda_graph_correct':True,
                     'oracle_ms':1.,'oracle_graph_ms':1.,'eager_ms':1.,'graph_ms':1.,'repeats':1,
                     'native_admitted':admitted,'native_after':copy.deepcopy(native)}
                if admitted:
                    payload=size>16
                    row.update(counter_before=counters(0),counter_after=counters(1000000000),
                               native_before={**copy.deepcopy(native),'path_counters':before_paths},
                               payload_gate_required=payload,payload_all_hcas_passed=True if payload else None)
                    row['function_traffic']=probe.function_delta(row['counter_before'],row['counter_after'],size,2,require_payload=payload)
                    row['native_path_traffic']=probe.path_delta(before_paths,after_paths,0,HCAS,require_both=payload)
                else:row.update(fallback_reason='capacity',fallback_reason_verified=True)
                report['results'].append(row)
    return report,settings


class QualificationTests(unittest.TestCase):
    def test_requested_and_runtime_geometry_accepts_each_bounded_configuration(self):
        for blocks in (4,8,16):
            with self.subTest(blocks=blocks):
                report,settings=geometry_report(blocks)
                self.assertTrue(qualifier.validate_rocenante_report(report,0,HCAS,'library',settings))
    def test_geometry_missing_wrong_boolean_or_float_rejected_final_and_per_case(self):
        report,settings=geometry_report()
        for scope in ('native_final','admitted_case','fallback_case'):
            for field in ('native_threads','native_blocks','runtime_native_threads','runtime_native_blocks'):
                for value in (None,True,False,256,4,512.0 if 'threads' in field else 8.0):
                    bad=copy.deepcopy(report)
                    target=bad['native_final'] if scope=='native_final' else next(
                        row['native_after'] for row in bad['results'] if row['native_admitted'] is (scope=='admitted_case'))
                    if value is None:target.pop(field)
                    else:target[field]=value
                    with self.subTest(scope=scope,field=field,value=value):
                        self.assertFalse(qualifier.validate_rocenante_report(bad,0,HCAS,'library',settings))
    def test_configured_geometry_rejects_unbound_or_coerced_values(self):
        report,settings=geometry_report()
        for key,value in (('native_blocks',True),('native_blocks',8.0),('native_blocks',2),
                          ('native_blocks',16),('native_threads',True),('native_threads',512.0),('native_threads',256)):
            with self.subTest(key=key,value=value):
                self.assertFalse(qualifier.validate_rocenante_report(report,0,HCAS,'library',{**settings,key:value}))
    def test_bootstrap_preflight_reuses_time_wait_but_rejects_live_listener(self):
        # Execute the actual socket availability block from the root worker.
        # The fixture creates only an isolated loopback TCP listener.
        tree=ast.parse((ROOT/'scripts/qualify.py').read_text())
        worker=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='worker')
        block=next(node for node in ast.walk(worker) if isinstance(node,ast.With)
                   and any(isinstance(item.context_expr,ast.Call)
                           and isinstance(item.context_expr.func,ast.Attribute)
                           and isinstance(item.context_expr.func.value,ast.Name)
                           and item.context_expr.func.value.id=='socket'
                           and item.context_expr.func.attr=='socket' for item in node.items))
        code=compile(ast.fix_missing_locations(ast.Module(body=[block],type_ignores=[])),'qualifier-port-preflight','exec')
        with socket.socket() as server:
            server.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
            server.bind(('127.0.0.1',0));server.listen(1);server.settimeout(5)
            port=server.getsockname()[1]
            namespace={'socket':socket,'node':{'management_address':'127.0.0.1'},'payload':{'master_port':port}}
            with self.assertRaises(OSError):exec(code,namespace)
            with socket.create_connection(server.getsockname(),timeout=5) as client:
                accepted,_=server.accept();accepted.close()
                self.assertEqual(client.recv(1),b'')
        # Closing the accepted server socket first leaves this port TIME_WAIT.
        # The TCPStore-compatible probe permits reuse but never an active server.
        with socket.socket() as ordinary:
            with self.assertRaises(OSError):ordinary.bind(('127.0.0.1',port))
        exec(code,namespace)
    def test_offline_plan_no_processes_or_writes(self):
        with config_file() as fixture, patch.object(sys,'argv',['qualify.py','plan','--config',str(fixture),'--transport','rocenante']), \
             patch('subprocess.run',side_effect=AssertionError('No processes')), \
             patch.object(Path,'mkdir',side_effect=AssertionError('No directories')), \
             patch.object(Path,'write_text',side_effect=AssertionError('No writes')),patch('sys.stdout',new_callable=io.StringIO) as output:
            qualifier.main()
        value=json.loads(output.getvalue());self.assertEqual(value['transport'],'rocenante')
        self.assertEqual(value['cache_layouts'],['symmetric','asymmetric'])
    def test_raw_pattern_has_all_bf16_and_fp32_special_bits(self):
        raw=probe.raw_payload(131136,0,0)
        self.assertEqual(set(struct.unpack('<65536H',raw[64:])),set(range(65536)))
        self.assertIn(struct.pack('<I',0x7fc00001),raw[:64]);self.assertIn(struct.pack('<I',0x80000000),raw[:64])
        for rank in range(4):
            for epoch in range(4):self.assertEqual(len(probe.raw_payload(1001,rank,epoch)),1001)
        self.assertNotEqual(probe.raw_payload(1001,0,0),probe.raw_payload(1001,0,1))
    def test_admission_edges_included_for_tuning(self):
        sizes=probe.case_sizes({'min_gather_bytes':3*1048576+3,'max_gather_bytes':4*1048576})
        self.assertIn(3*1048576+16,sizes);self.assertIn(4*1048576,sizes);self.assertIn(33554432,sizes)
    def test_actual_hc_drafter_sampler_and_split_prefill_geometries(self):
        sizes=probe.case_sizes({'min_gather_bytes':1,'max_gather_bytes':16777216})
        self.assertTrue({896,7168,16384,32768,65536,131072,262144,524288,
                         1048576,4194304,16777216,33554432}.issubset(sizes))
        self.assertEqual(len(sizes)*len(probe.DTYPES)*2,120)
    def test_origin_qps_require_real_payload_on_both_paths(self):
        old=paths();new=copy.deepcopy(old)
        for row in new:row.update(payload_bytes=1000,payload_writes=1,physical_hop_payload_bytes=1000)
        self.assertEqual(len(probe.path_delta(old,new,0,HCAS)),6)
        for field in ('payload_bytes','payload_writes'):
            bad=copy.deepcopy(new);bad[1][field]=0
            with self.assertRaises(RuntimeError):probe.path_delta(old,bad,0,HCAS)
    def test_tiny_one_pack_exemption_does_not_claim_second_payload(self):
        old=paths();new=copy.deepcopy(old)
        for row in new:
            if row['path_index']==0:row.update(payload_bytes=16,payload_writes=1,physical_hop_payload_bytes=16)
        rows=probe.path_delta(old,new,0,HCAS,require_both=False)
        self.assertEqual(sum(row['payload_passed'] for row in rows),3)
        with self.assertRaises(RuntimeError):probe.path_delta(old,new,0,HCAS)
    def test_origin_qp_errors_resets_and_identity_changes_fail(self):
        old=paths();good=copy.deepcopy(old)
        for row in good:row.update(payload_bytes=1000,payload_writes=1,physical_hop_payload_bytes=1000)
        for field,value in (('completion_errors',1),('payload_bytes',-1),('qp_number',999)):
            bad=copy.deepcopy(good);bad[0][field]=value
            with self.assertRaises(RuntimeError):probe.path_delta(old,bad,0,HCAS)
        with self.assertRaises(RuntimeError):probe.path_delta(old,good[:-1],0,HCAS)
    def test_function_gate_requires_tx_rx_all_four_and_zero_errors(self):
        old,new=counters(0),counters(1000000)
        self.assertEqual(len(probe.function_delta(old,new,16384,128)),4)
        for field,value in (('tx_vport_rdma_unicast_bytes',0),('rx_vport_rdma_unicast_bytes',0)):
            bad=copy.deepcopy(new);bad['devices']['a']['function_rdma'][field]=value
            with self.assertRaises(RuntimeError):probe.function_delta(old,bad,16384,128)
        bad=copy.deepcopy(new);bad['devices']['d']['hw_requests_and_errors']['req_cqe_error']=1
        with self.assertRaises(RuntimeError):probe.function_delta(old,bad,16384,128)
    def test_native_health_rejects_stalls_errors_and_incomplete_sequences(self):
        good={'native_active':True,'failure':None,'native_stats':{'error_seq':0,'abort_seq':0,'wait_stopped':False,'ctrl_seq':12,'completed_seq':12},'path_counters':[{'completion_errors':0} for _ in range(6)]}
        self.assertTrue(probe.native_healthy(good))
        for field,value in (('error_seq',1),('abort_seq',1),('wait_stopped',True),('completed_seq',11)):
            bad=copy.deepcopy(good);bad['native_stats'][field]=value;self.assertFalse(probe.native_healthy(bad))
        bad=copy.deepcopy(good);bad['path_counters'][2]['completion_errors']=1;self.assertFalse(probe.native_healthy(bad))
    def test_owned_flat_qps_marker_coexistence_and_exact_native_numbers(self):
        native=paths()
        rows=[dict(ifname=row['device'],pid=22,lqpn=row['qp_number']) for row in native]
        rows.append(dict(ifname='a',pid=11,lqpn=200))
        rows.append(dict(ifname='a',pid=None,type='GSI',comm='ib_core'))
        self.assertTrue(qualifier.qp_ownership(rows,HCAS,{22},{11},native)['passed'])
        with self.assertRaises(RuntimeError):qualifier.qp_ownership(rows,HCAS,{22},set(),native)
        with self.assertRaises(RuntimeError):qualifier.qp_ownership(rows,HCAS,{22},{11},[{**native[0],'qp_number':999}])
        with self.assertRaises(RuntimeError):qualifier.qp_ownership(rows+[dict(ifname='d',pid=33,lqpn=999)],HCAS,{22},{11},native)
    def test_nested_rdma_rows_preserve_parent_device(self):
        flat=qualifier.flatten_qps([{'dev':'a','qp':[{'pid':1,'lqpn':2}]}])
        self.assertEqual(flat,[{'dev':'a','pid':1,'lqpn':2}])
    def test_incomplete_or_forged_reports_do_not_qualify(self):
        for report in (None,{}, {'rank':0,'world':4,'library_sha256':'x','required_hcas':HCAS}):
            self.assertFalse(qualifier.validate_report(report,0,HCAS,'x'))
            self.assertFalse(qualifier.validate_rocenante_report(report,0,HCAS,'x',{'min_gather_bytes':1,'max_gather_bytes':16777216}))
    def test_source_pins_are_current_version(self):
        self.assertEqual(qualifier.BASE_COMMIT,'12df1a11afad322be5a204a2db890161cbf8131d')
        self.assertEqual(qualifier.PATCHED_TREE,'e4ae48e878d15a5ba52d91a18c6b5e79b50a7db0')
    def test_late_exception_invalidates_previously_successful_summary(self):
        # Execute the actual orchestration exception handler, including the case
        # where symmetric payloads succeeded before an asymmetric/TC failure.
        tree=ast.parse((ROOT/'scripts/qualify.py').read_text())
        handler=next(node for node in ast.walk(tree) if isinstance(node,ast.ExceptHandler)
                     and any(isinstance(child,ast.Constant) and child.value=='qualification-failed'
                             for child in ast.walk(node)))
        code=compile(ast.fix_missing_locations(ast.Module(body=handler.body,type_ignores=[])),'qualifier-exception','exec')
        for prior in (False,True):
            summary={'qualified':prior,'correctness_passed':True}
            events=[]
            namespace={'summary':summary,'exc':RuntimeError('late TC payload failure'),
                       'event':lambda *a,**k:events.append((a,k))}
            exec(code,namespace)
            self.assertIs(summary['qualified'],False)
            self.assertEqual(summary['failure'],'late TC payload failure')
            self.assertEqual(events[0][0],('qualification-failed',))
    def test_diagonal_tc_payload_requires_all_eight_hardware_redirects(self):
        def guard(value):
            rule={'in_hw':True,'skip_sw':True,'software_counter_available':True,'rule_identity':{'exact':'test-rule'},
                  'actions_stats':[{'hw_bytes':value,'hw_packets':value//64,'sw_packets':0,'drops':0} for _ in range(3)]}
            return {'plan_id':'pin','network_boot_ids':{str(i):'boot' for i in range(4)},
                    'hosts':{str(i):{'rules':[copy.deepcopy(rule),copy.deepcopy(rule)]} for i in range(4)}}
        old,new=guard(0),guard(1024)
        self.assertEqual(qualifier.hardware_mesh_delta(old,new)['rule_count'],8)
        for field,value in (('hw_bytes',0),('hw_packets',0),('sw_packets',1),('drops',1)):
            bad=copy.deepcopy(new)
            for stats in bad['hosts']['3']['rules'][1]['actions_stats']:stats[field]=value
            with self.subTest(field=field),self.assertRaises(RuntimeError):qualifier.hardware_mesh_delta(old,bad)
        bad=copy.deepcopy(new);bad['hosts']['2']['rules'][0]['skip_sw']=False
        with self.assertRaises(RuntimeError):qualifier.hardware_mesh_delta(old,bad)
        old2,new2=copy.deepcopy(old),copy.deepcopy(new)
        for obj in (old2,new2):
            for host in obj['hosts'].values():
                for rule in host['rules']:
                    for stats in rule['actions_stats']:
                        stats['bytes']=stats.pop('hw_bytes');stats['packets']=stats.pop('hw_packets')
        self.assertEqual(qualifier.hardware_mesh_delta(old2,new2)['rule_count'],8)


if __name__=='__main__':unittest.main()
