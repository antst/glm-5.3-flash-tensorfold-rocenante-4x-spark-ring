"""CPU-only planning/ownership contract checks; never contact a host."""
import copy
import ast
import json
import os
import hashlib
import socket
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from common import load_config
from network import make_plan,hardware_rule_attestation,hardware_payload_delta,preservation,validate_inventory,fabric_users,mesh_worker,firewall_policy_sha256,project_owned_removed_routes,operate,digest


class NetworkTests(unittest.TestCase):
    def config(self):
        c=load_config(ROOT/'config.example.json');c['shared_root']=str(ROOT)
        for rank,node in enumerate(c['nodes']):
            for p in node['fabric']:p['mac']='02:00:00:%02x:%02x:%02x'%(rank,p['physical_port'],p['rail'])
        return c
    def test_offline_complete_plan_has_only_required_diagonals(self):
        c=self.config()
        with patch('subprocess.run',side_effect=AssertionError('SSH forbidden')),patch.object(Path,'write_text',side_effect=AssertionError('Writes forbidden')):
            p=make_plan(c)
        self.assertTrue(p['complete'])
        self.assertEqual(p['neighbors'],[])
        self.assertEqual(p['software_forwarding_changes'],[])
        for n in p['nodes']:
            self.assertEqual((len(n['routes']),len(n['rules']),len(n['markers'])),(2,2,2))
            for r in n['routes']:
                self.assertFalse(r['object']['permanent_final_neighbor'])
                self.assertIn('via',r['add']);self.assertTrue(r['add'][3].endswith('/32'))
            for r in n['rules']:self.assertIn('skip_sw',r['add'])
    def test_missing_macs_reports_inventory_without_fabricated_plan(self):
        c=self.config()
        for n in c['nodes']:
            for p in n['fabric']:del p['mac']
        p=make_plan(c);self.assertFalse(p['complete']);self.assertEqual(len(p['requires_inventory']),16)
    def test_reversed_rank_direction_is_rejected(self):
        c=self.config();c['nodes'][0]['fabric'][0]['peer_rank']=3
        with self.assertRaisesRegex(ValueError,'f0=next'):make_plan(c)
    def test_pci_twin_names_must_match_prepared_backend(self):
        c=self.config();c['nodes'][0]['fabric'][0]['hca']='other'
        with self.assertRaisesRegex(ValueError,'canonical HCA'):make_plan(c)
    def sample_rule(self):
        o=make_plan(self.config())['nodes'][0]['rules'][0]['object']
        mac=bytes.fromhex(o['rewrite_ethernet_destination'].replace(':',''))
        edit=lambda offset,value,mask:{'offset':offset,'val':hex(value),'mask':hex(mask),'htype':'eth','cmd':'set'}
        row={'pref':o['preference'],'kind':'flower','options':{'handle':hex(o['handle']),'skip_sw':True,'in_hw':True,
             'keys':{'src_mac':o['match_ethernet_source'],'dst_mac':o['match_ethernet_destination'],'eth_type':'0x88b5'},
             'actions':[{'kind':'pedit','keys':[edit(12,0x08000000,0xffff)],'stats':{'sw_packets':0,'drops':0}},
             {'kind':'pedit','keys':[edit(0,int.from_bytes(mac[:4],'big'),0),edit(4,int.from_bytes(mac[4:]+b'\0\0','big'),0xffff)],'stats':{'sw_packets':0,'drops':0}},
             {'kind':'mirred','to_dev':o['egress_netdev'],'mirred_action':'redirect','stats':{'sw_packets':0,'drops':0}}]}}
        return o,row
    def test_hardware_rule_exact_bytes_and_zero_software_transit(self):
        o,row=self.sample_rule();self.assertTrue(hardware_rule_attestation(o,[row])['in_hw'])
        for field in ('skip_sw','in_hw'):
            bad=copy.deepcopy(row);bad['options'][field]=False
            with self.subTest(field=field),self.assertRaises(RuntimeError):hardware_rule_attestation(o,[bad])
        for counter in ('sw_packets','drops'):
            bad=copy.deepcopy(row);bad['options']['actions'][0]['stats'][counter]=1
            with self.subTest(counter=counter),self.assertRaises(RuntimeError):hardware_rule_attestation(o,[bad])
        bad=copy.deepcopy(row);bad['options']['actions'][1]['keys'][0]['val']='0x00'
        with self.assertRaisesRegex(RuntimeError,'rewrite bytes'):hardware_rule_attestation(o,[bad])
    def test_actual_tc_json_bare_hex_pedit_words_and_integer_values(self):
        o,row=self.sample_rule();row['options']['handle']=o['handle'];row['options']['keys']['eth_type']='88b5'
        for action in row['options']['actions'][:2]:
            for edit in action['keys']:
                for field in ('val','mask'):edit[field]=format(int(edit[field],0),'08x')
        self.assertTrue(hardware_rule_attestation(o,[row])['in_hw'])
        for action in row['options']['actions'][:2]:
            for edit in action['keys']:
                for field in ('val','mask'):edit[field]=int(edit[field],16)
        self.assertTrue(hardware_rule_attestation(o,[row])['in_hw'])
    def test_route_firewall_forwarding_profile_preservation(self):
        before={'configuration_sha256':{'/etc/netplan/file':'abc'},'sysctls':{'ip_forward':'1'},'firewall_sha256':{'v4':'hash'},
                'routes4':[{'dst':'default','gateway':'192.0.2.1','dev':'management'}],'routes6':[],'fabric':{}}
        after=copy.deepcopy(before);after['routes4'].append({'dst':'10.77.34.3/32','dev':'fabric','gateway':'10.77.41.4'})
        allowed=[{'object':{'destination_ipv4':'10.77.34.3'}}]
        self.assertTrue(preservation(before,after,planned_routes=allowed))
        # Installed iproute2 renders newly added IPv4 /32 routes as bare hosts.
        after['routes4'][-1]['dst']='10.77.34.3'
        self.assertTrue(preservation(before,after,planned_routes=allowed))
        for destination in ('10.77.34.4','10.77.34.0/24'):
            bad=copy.deepcopy(after);bad['routes4'][-1]['dst']=destination
            with self.subTest(destination=destination),self.assertRaisesRegex(RuntimeError,'route'):
                preservation(before,bad,planned_routes=allowed)
        for field in ('configuration_sha256','sysctls','firewall_sha256'):
            bad=copy.deepcopy(after);bad[field]['unplanned']='changed'
            with self.subTest(field=field),self.assertRaises(RuntimeError):preservation(before,bad,planned_routes=allowed)
        bad=copy.deepcopy(after);bad['routes4'][0]['gateway']='192.0.2.254'
        with self.assertRaisesRegex(RuntimeError,'route'):preservation(before,bad,planned_routes=allowed)
    def test_firewall_policy_hash_ignores_live_chain_counters_and_save_timestamps_only(self):
        before='# Generated at first capture\n*filter\n:INPUT ACCEPT [12:345]\n:OUTPUT ACCEPT [9:100]\n-A INPUT -s 192.0.2.1 -j ACCEPT\nCOMMIT\n'
        after=before.replace('first capture','later capture').replace('[12:345]','[999:99999]').replace('[9:100]','[10:120]')
        self.assertEqual(firewall_policy_sha256(before),firewall_policy_sha256(after))
        for change in (before.replace(':INPUT ACCEPT',':INPUT DROP'),before.replace('192.0.2.1','192.0.2.2'),
                       before.replace('COMMIT','-A INPUT -m comment --comment "[12:345]" -j ACCEPT\nCOMMIT')):
            self.assertNotEqual(firewall_policy_sha256(before),firewall_policy_sha256(change))
        quoted=before.replace('COMMIT','-A INPUT -m comment --comment "[12:345]" -j ACCEPT\nCOMMIT')
        self.assertNotEqual(firewall_policy_sha256(quoted),firewall_policy_sha256(quoted.replace('"[12:345]"','"[13:346]"')))
    def test_only_exact_connected_fabric_metric_after_one_successful_reload_is_admitted(self):
        pci='pci/0000:01:00.0'
        before={'configuration_sha256':{},'sysctls':{},'firewall_sha256':{},'routes6':[],
                'routes4':[{'dst':'10.77.12.0/24','dev':'fabric','prefsrc':'10.77.12.1','protocol':'kernel','scope':'link','metric':105},
                           {'dst':'default','dev':'management','gateway':'192.0.2.1','metric':100}],
                'fabric':{'hca':{'mac':'mac','interface':'fabric','hca':'hca','pci':pci[4:],'associated_netdevs':['fabric'],'mtu':9000,
                                 'ipv4':[{'address':'10.77.12.1','prefixlen':24}],'nm_connection':{'stdout':'active-uuid'}}}}
        stats=lambda count,failed=False:{'stdout':json.dumps({'dev':{pci:{'reload_failed':failed,'stats':{'reload':{'driver_reinit':{'unspecified':count}}}}}})}
        before['devlink_statistics']=stats(2);after=copy.deepcopy(before);after['devlink_statistics']=stats(3);after['routes4'][0]['metric']=111
        proof=preservation(before,after,allow_driver_reload=True)
        self.assertEqual(len(proof['connected_metric_changes']),1)
        with self.assertRaises(RuntimeError):preservation(before,after)
        for mutation in ('no_reload','two_reloads','failed_reload','gateway','default_metric','nonkernel','scope','subnet','source','device'):
            bad=copy.deepcopy(after)
            if mutation=='no_reload':bad['devlink_statistics']=stats(2)
            elif mutation=='two_reloads':bad['devlink_statistics']=stats(4)
            elif mutation=='failed_reload':bad['devlink_statistics']=stats(3,True)
            elif mutation=='gateway':bad['routes4'][0]['gateway']='10.77.12.2'
            elif mutation=='default_metric':bad['routes4'][1]['metric']=101
            elif mutation=='nonkernel':bad['routes4'][0]['protocol']='static'
            elif mutation=='scope':bad['routes4'][0]['scope']='global'
            elif mutation=='subnet':bad['routes4'][0]['dst']='10.77.13.0/24'
            elif mutation=='source':bad['routes4'][0]['prefsrc']='10.77.12.2'
            else:bad['routes4'][0]['dev']='other'
            with self.subTest(mutation=mutation),self.assertRaises(RuntimeError):preservation(before,bad,allow_driver_reload=True)
    def test_flat_rdma_inventory_rejects_foreign_users_and_allows_only_exact_markers(self):
        rows=[{'ifname':'rocep1s0f0','lqpn':10,'pid':42,'type':'RAW_PACKET'},
              {'ifname':'rocep1s0f0','lqpn':1,'comm':'ib_core','type':'GSI'},
              {'ifname':'other','pid':99,'type':'RC'}]
        self.assertEqual(len(fabric_users(rows,['rocep1s0f0'])),1)
        self.assertEqual(fabric_users(rows,['rocep1s0f0'],[42]),[])
        rows.append({'ifname':'rocep1s0f0','pid':43,'type':'RC'})
        self.assertEqual([r['pid'] for r in fabric_users(rows,['rocep1s0f0'],[42])],[43])
    def worker_fixture(self,temporary,journal_boot,current_boot):
        directory=Path(temporary)/'mesh'/'test-owner'
        directory.mkdir(mode=0o700,parents=True)
        directory.parent.chmod(0o700)
        plan=make_plan(self.config())
        nodeplan=plan['nodes'][0]
        owner={'recipe_id':'test-owner','plan_id':plan['plan_id'],'rank':0,'boot_id':journal_boot}
        (directory/'owner.json').write_text(json.dumps(owner))
        journal={'boot_id':journal_boot,'resources':[],'markers':[],'driver_changes':[],'guardian':None,'retired_boots':[]}
        proc=Path(temporary)/'proc';(proc/'sys/kernel/random').mkdir(parents=True)
        (proc/'sys/kernel/random/boot_id').write_text(current_boot)
        data={'nodeplan':nodeplan,'recipe_id':'test-owner','plan_id':plan['plan_id'],'marker_sha256':'not-read',
              'restore_driver':False}
        return directory,proc,journal,data
    def invoke_mocked_worker(self,temporary,data,action,extra_query=None):
        real_path=Path;original_stat=Path.stat
        def paths(value,*parts):
            value=str(value)
            if value=='/var/tmp/tensorfold-rocenante-mesh':value=str(real_path(temporary)/'mesh')
            elif value=='/proc' or value.startswith('/proc/'):value=str(real_path(temporary)/'proc')+value[5:]
            return real_path(value,*parts)
        def root_stat(path,*args,**kwargs):
            result=list(original_stat(path,*args,**kwargs));result[4]=0
            return os.stat_result(result)
        def run(argv,**kwargs):
            if argv[0]=='nvidia-smi':return SimpleNamespace(stdout='',stderr='',returncode=0)
            if argv[:4]==['rdma','-j','resource','show']:return SimpleNamespace(stdout='[]',stderr='',returncode=0)
            if extra_query is not None:return extra_query(argv)
            raise AssertionError('Unexpected system query/mutation: '+repr(argv))
        with patch('pathlib.Path',side_effect=paths),patch.object(real_path,'stat',root_stat),patch('os.geteuid',return_value=0),\
             patch('subprocess.run',side_effect=run),patch('os.kill',side_effect=AssertionError('Must not signal a process')):
            return mesh_worker(data,action)
    def test_previous_boot_rollback_retires_records_without_signaling_reused_pids(self):
        old='00000000-0000-0000-0000-000000000001';new='00000000-0000-0000-0000-000000000002'
        with tempfile.TemporaryDirectory() as temp:
            directory,proc,journal,data=self.worker_fixture(temp,old,new)
            journal['markers']=[{'pid':4242,'start_ticks':1,'boot_id':old,'argv':['old-marker']}]
            journal['guardian']={'pid':4243,'start_ticks':2,'boot_id':old,'argv':['old-guardian']}
            journal['resources']=[{'kind':'route','pending':False,'item':{'delete':['must-never-run']}}]
            (directory/'journal.json').write_text(json.dumps(journal))
            (directory/'before.json').write_text('original-backup-retained')
            (proc/'4242').mkdir();(proc/'4242/cmdline').write_bytes(b'unrelated-process\0')
            result=self.invoke_mocked_worker(temp,data,'rollback')
            self.assertTrue(result['rolled_back'])
            self.assertEqual(result['removed_routes'],[])
            after=json.loads((directory/'journal.json').read_text())
            self.assertEqual(after['boot_id'],new);self.assertEqual(after['markers'],[]);self.assertIsNone(after['guardian'])
            self.assertEqual(after['retired_boots'][0]['resources'],journal['resources'])
            self.assertEqual((directory/'before.json').read_text(),'original-backup-retained')
            self.assertEqual(json.loads((directory/'owner.json').read_text())['boot_id'],old)
    def test_same_boot_rollback_refuses_reused_guardian_pid(self):
        boot='00000000-0000-0000-0000-000000000001'
        with tempfile.TemporaryDirectory() as temp:
            directory,proc,journal,data=self.worker_fixture(temp,boot,boot)
            journal['guardian']={'pid':4242,'start_ticks':1,'boot_id':boot,'argv':['expected-guardian']}
            (directory/'journal.json').write_text(json.dumps(journal));(proc/'4242').mkdir()
            # stat fields after the process name: state is0, starttime is19.
            (proc/'4242/stat').write_text('4242 (unrelated) '+' '.join(['S']+['0']*18+['999']))
            (proc/'4242/cmdline').write_bytes(b'unrelated-process\0')
            with self.assertRaisesRegex(RuntimeError,'Guardian PID identity changed'):
                self.invoke_mocked_worker(temp,data,'rollback')
            self.assertIsNotNone(json.loads((directory/'journal.json').read_text())['guardian'])
    def route_removal(self,cfg,plan,rank,boot):
        item=plan['nodes'][rank]['routes'][0];o=item['object']
        row={'dst':o['destination_ipv4'],'dev':o['source_netdev'],'gateway':o['gateway_ipv4'],
             'prefsrc':o['source_ipv4'],'protocol':'boot','flags':[]}
        resource={'kind':'route','pending':False,'item':copy.deepcopy(item)}
        removal={'node':cfg['nodes'][rank]['name'],'owner':{'recipe_id':cfg['recipe_id'],'plan_id':plan['plan_id'],'rank':rank},
                 'boot_id':boot,'route_identity':row,'resource':resource,'resource_sha256':digest(resource)}
        return row,removal
    def route_facts(self,boot,rows=()):
        return {'boot_id':boot,'configuration_sha256':{'/etc/netplan/management.yaml':'unchanged'},
                'sysctls':{'ip_forward':'1','rp_filter':'2'},'firewall_sha256':{'v4':'policy'},
                'routes4':[{'dst':'default','dev':'management','gateway':'192.0.2.1','metric':100},*copy.deepcopy(rows)],
                'routes6':[],'fabric':{}}
    def test_worker_attests_only_full_current_boot_route_actually_deleted(self):
        boot='00000000-0000-0000-0000-000000000001'
        with tempfile.TemporaryDirectory() as temp:
            directory,proc,journal,data=self.worker_fixture(temp,boot,boot)
            cfg=self.config();cfg['recipe_id']='test-owner';plan=make_plan(cfg)
            row,removal=self.route_removal(cfg,plan,0,boot)
            journal['resources']=[removal['resource']];(directory/'journal.json').write_text(json.dumps(journal))
            present=[True];commands=[]
            def query(argv):
                commands.append(argv)
                if argv==['ip','-j','-4','route','show','exact',row['dst']+'/32']:
                    value=[row] if present[0] else []
                elif argv==removal['resource']['item']['delete']:
                    present[0]=False;value={}
                else:raise AssertionError('Unexpected route operation '+repr(argv))
                return SimpleNamespace(stdout=json.dumps(value),stderr='',returncode=0)
            result=self.invoke_mocked_worker(temp,data,'rollback',query)
            self.assertTrue(result['rolled_back']);self.assertFalse(present[0])
            self.assertEqual(result['removed_routes'],[removal])
            self.assertEqual(json.loads((directory/'journal.json').read_text())['resources'],[])
            self.assertEqual(commands.count(removal['resource']['item']['delete']),1)
    def test_route_projection_requires_exact_owned_full_identity_and_after_absence(self):
        cfg=self.config();plan=make_plan(cfg);boot='00000000-0000-0000-0000-000000000001'
        row,removal=self.route_removal(cfg,plan,0,boot)
        before=self.route_facts(boot,[row]);after=self.route_facts(boot)
        projected,proof=project_owned_removed_routes(before,after,[removal],plan['nodes'][0],cfg['recipe_id'],plan['plan_id'])
        self.assertEqual(proof,[removal]);self.assertTrue(preservation(projected,after)['passed'])
        self.assertEqual(before['routes4'][-1],row)
        mutations=('owner','node','boot','pending','item','digest','row','duplicate','still_present','different_boot','duplicate_before')
        for mutation in mutations:
            b=copy.deepcopy(before);a=copy.deepcopy(after);r=copy.deepcopy(removal);removals=[r]
            if mutation=='owner':r['owner']['rank']=1
            elif mutation=='node':r['node']='other'
            elif mutation=='boot':r['boot_id']='old'
            elif mutation=='pending':r['resource']['pending']=True;r['resource_sha256']=digest(r['resource'])
            elif mutation=='item':r['resource']['item']['delete']=['foreign'];r['resource_sha256']=digest(r['resource'])
            elif mutation=='digest':r['resource_sha256']='forged'
            elif mutation=='row':r['route_identity']['metric']=50
            elif mutation=='duplicate':removals.append(copy.deepcopy(r))
            elif mutation=='still_present':a['routes4'].append(row)
            elif mutation=='different_boot':a['boot_id']='different'
            else:b['routes4'].append(row)
            with self.subTest(mutation=mutation),self.assertRaises(RuntimeError):
                project_owned_removed_routes(b,a,removals,plan['nodes'][0],cfg['recipe_id'],plan['plan_id'])
    def test_rollback_projection_preserves_foreign_same_destination_and_all_other_policy(self):
        cfg=self.config();plan=make_plan(cfg);boot='00000000-0000-0000-0000-000000000002'
        row,removal=self.route_removal(cfg,plan,0,boot)
        foreign={**row,'metric':500,'protocol':'static'}
        before=self.route_facts(boot,[foreign]);after=copy.deepcopy(before)
        projected,_=project_owned_removed_routes(before,after,[],plan['nodes'][0],cfg['recipe_id'],plan['plan_id'])
        self.assertTrue(preservation(projected,after)['passed'])
        for mutation in ('foreign_removed','default_metric','default_gateway','configuration_sha256','sysctls','firewall_sha256'):
            bad=copy.deepcopy(after)
            if mutation=='foreign_removed':bad['routes4'].pop()
            elif mutation=='default_metric':bad['routes4'][0]['metric']=101
            elif mutation=='default_gateway':bad['routes4'][0]['gateway']='192.0.2.254'
            else:bad[mutation]['unplanned']='changed'
            with self.subTest(mutation=mutation),self.assertRaises(RuntimeError):preservation(projected,bad)
    def rollback_orchestration(self,partial=False,all_current=False,drift=False,preflight_failure=False):
        cfg=self.config();plan=make_plan(cfg);old='00000000-0000-0000-0000-000000000001';new='00000000-0000-0000-0000-000000000002'
        before={'hosts':{}};after={'hosts':{}};results={};calls=[];saves=[]
        for rank,node in enumerate(cfg['nodes']):
            current=all_current or (partial and rank>=2);boot=old if current else new
            row,removal=self.route_removal(cfg,plan,rank,boot)
            removals=[removal]
            if current:
                second=copy.deepcopy(removal);second['resource']['item']=plan['nodes'][rank]['routes'][1]
                o=second['resource']['item']['object'];second['route_identity']={**row,'dst':o['destination_ipv4'],
                    'dev':o['source_netdev'],'gateway':o['gateway_ipv4'],'prefsrc':o['source_ipv4']}
                second['resource_sha256']=digest(second['resource']);removals.append(second)
            # Rebooted hosts retain a foreign route at this same planned destination.
            rows=[r['route_identity'] for r in removals] if current else [{**row,'protocol':'static','metric':500}]
            before['hosts'][node['name']]=self.route_facts(boot,rows)
            after['hosts'][node['name']]=self.route_facts(boot,[] if current else rows)
            results[node['name']]={'rolled_back':True,'removed_routes':removals if current else [],'boot_id':boot}
        if drift:after['hosts'][cfg['nodes'][0]['name']]['routes4'][0]['metric']=101
        stale=copy.deepcopy(before)
        for facts in stale['hosts'].values():facts['boot_id']=old;facts['firewall_sha256']={'v4':'stale'};facts['routes4']=[]
        class FakeOperator:
            def __init__(self,config):self.config=config
            def event(self,*args,**kwargs):pass
            def parallel(self,function):return {n['name']:function(r) for r,n in enumerate(self.config['nodes'])}
            def remote_python(self,rank,code,**kwargs):
                tree=ast.parse(code)
                payload=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='payload' for t in n.targets))
                action=next(ast.literal_eval(n.args[1]) for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='mesh_worker')
                calls.append((action,rank));assert payload['before']==before['hosts'][cfg['nodes'][rank]['name']]
                if action=='rollback-preflight':
                    if preflight_failure and rank==3:raise RuntimeError('Unowned GPU workload')
                    return json.dumps({'idle':True})
                assert action=='rollback';return json.dumps(results[cfg['nodes'][rank]['name']])
        with tempfile.TemporaryDirectory() as temp:
            cfg['state_root']=temp
            with patch('network.Operator',FakeOperator),patch('network.marker_identity',return_value=('/unused','sha')),\
                 patch('network.collect',side_effect=[before,after]),patch('network.save',side_effect=lambda path,value:saves.append((path,copy.deepcopy(value)))):
                if drift or preflight_failure:
                    with self.assertRaises(RuntimeError):operate(cfg,plan,stale,'rollback')
                else:operate(cfg,plan,stale,'rollback')
        self.assertEqual([rank for action,rank in calls if action=='rollback-preflight'],[0,1,2,3])
        self.assertEqual([rank for action,rank in calls if action=='rollback'],[] if preflight_failure else [0,1,2,3])
        if not preflight_failure:self.assertEqual([a for a,r in calls[:4]],['rollback-preflight']*4)
        self.assertTrue(all(v['firewall_sha256']=={'v4':'stale'} for v in stale['hosts'].values()))
        if not drift and not preflight_failure:
            saved=next(value for path,value in saves if path.name=='network-rollback.json')
            self.assertTrue(all(v['preservation']['passed'] for v in saved['hosts'].values()))
            self.assertEqual(sum(len(v['preservation']['owned_removed_routes']) for v in saved['hosts'].values()),8 if all_current else 4 if partial else 0)
    def test_same_boot_rollback_projects_exactly_all_eight_owned_routes(self):self.rollback_orchestration(all_current=True)
    def test_full_reboot_rollback_uses_fresh_facts_and_preserves_unowned_routes(self):self.rollback_orchestration()
    def test_partial_reboot_rollback_projects_only_current_boot_owned_removals(self):self.rollback_orchestration(partial=True)
    def test_rollback_fresh_snapshot_rejects_unrelated_drift_and_checks_all_idle_before_mutation(self):
        self.rollback_orchestration(partial=True,drift=True)
        self.rollback_orchestration(partial=True,preflight_failure=True)
    def test_driver_rollback_uses_supported_global_parameter_inventory_and_restores_value(self):
        boot='00000000-0000-0000-0000-000000000001';pci='pci/0002:01:00.1'
        with tempfile.TemporaryDirectory() as temp:
            directory,proc,journal,data=self.worker_fixture(temp,boot,boot);data['restore_driver']=True
            journal['driver_changes']=[{'pci':pci,'name':'hairpin_queue_size','cmode':'driverinit','before':1024,'after':8192,'pending':False}]
            (directory/'journal.json').write_text(json.dumps(journal));state={'value':8192,'reload':1};commands=[]
            def query(argv):
                commands.append(argv)
                if argv==['devlink','-j','dev','param','show']:
                    value={'param':{pci:[{'name':'hairpin_queue_size','values':[{'cmode':'driverinit','value':state['value']}]}]}}
                elif argv==['devlink','-s','-j','dev','show',pci]:
                    value={'dev':{pci:{'stats':{'reload':{'driver_reinit':{'unspecified':state['reload']}}}}}}
                elif argv==['devlink','dev','param','set',pci,'name','hairpin_queue_size','value','1024','cmode','driverinit']:
                    state['value']=1024;value={}
                elif argv==['devlink','dev','reload',pci,'action','driver_reinit']:
                    state['reload']+=1;value={}
                else:raise AssertionError('Unsupported installed devlink CLI: '+repr(argv))
                return SimpleNamespace(stdout=json.dumps(value),stderr='',returncode=0)
            result=self.invoke_mocked_worker(temp,data,'rollback',query)
            self.assertTrue(result['rolled_back']);self.assertEqual(state,{'value':1024,'reload':2})
            self.assertEqual(json.loads((directory/'journal.json').read_text())['driver_changes'],[])
            self.assertIn(['devlink','-j','dev','param','show'],commands)
    def test_health_preflight_reuses_departed_server_time_wait_but_rejects_live_listener(self):
        boot='00000000-0000-0000-0000-000000000001'
        with tempfile.TemporaryDirectory() as temp:
            directory,proc,journal,data=self.worker_fixture(temp,boot,boot)
            (directory/'journal.json').write_text(json.dumps(journal))
            binary=Path(temp)/'marker';binary.write_bytes(b'test marker')
            data.update(marker_binary=str(binary),marker_sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
            data['nodeplan']['node']['management_address']='127.0.0.1'
            def query(argv):
                if argv[0] not in ('ip','tc'):raise AssertionError('Unexpected preflight command '+repr(argv))
                return SimpleNamespace(stdout='[]',stderr='',returncode=0)
            with socket.socket() as server:
                server.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
                server.bind(('127.0.0.1',0));server.listen(1);server.settimeout(5)
                data['health_port']=server.getsockname()[1]
                with self.assertRaises(OSError):self.invoke_mocked_worker(temp,data,'preflight',query)
                with socket.create_connection(server.getsockname(),timeout=5) as client:
                    accepted,_=server.accept();accepted.close()
                    self.assertEqual(client.recv(1),b'')
            # Server closed its accepted connection first: local endpoint is
            # TIME_WAIT. An ordinary bind fails; guardian-compatible reuse works.
            with socket.socket() as ordinary:
                with self.assertRaises(OSError):ordinary.bind(('127.0.0.1',data['health_port']))
            self.assertTrue(self.invoke_mocked_worker(temp,data,'preflight',query)['preflight'])
    def test_all_eight_rules_require_positive_hardware_transit_without_action_double_counting(self):
        before={'plan_id':'plan','network_boot_ids':{'spark'+str(i):str(i) for i in range(1,5)},'hosts':{}}
        for host in before['network_boot_ids']:
            before['hosts'][host]={'rules':[{'rule_identity':{'host':host,'rule':i},'skip_sw':True,'in_hw':True,
                'actions_stats':[{'packets':10,'bytes':640,'sw_packets':0,'sw_bytes':0,'drops':0} for j in range(3)]} for i in range(2)]}
        after=copy.deepcopy(before)
        for host in after['hosts'].values():
            for rule in host['rules']:
                for stats in rule['actions_stats']:stats.update(packets=20,bytes=1280)
        result=hardware_payload_delta(before,after)
        self.assertTrue(result['passed']);self.assertEqual(result['rule_count'],8)
        self.assertTrue(all(row['hardware_packets']==10 and row['hardware_bytes']==640 for row in result['rules']))
        # Drivers may place action statistics on only the first action.
        for host in after['hosts'].values():
            for rule in host['rules']:
                for stats in rule['actions_stats'][1:]:stats.update(packets=10,bytes=640)
        self.assertTrue(hardware_payload_delta(before,after)['passed'])
        for field in ('sw_packets','sw_bytes','drops'):
            bad=copy.deepcopy(after);bad['hosts']['spark2']['rules'][1]['actions_stats'][2][field]=1
            with self.subTest(field=field),self.assertRaises(RuntimeError):hardware_payload_delta(before,bad)
        for mutation in ('no_payload','identity','skip_sw','counter_reset','boot'):
            bad=copy.deepcopy(after);rule=bad['hosts']['spark2']['rules'][1]
            if mutation=='no_payload':rule['actions_stats']=copy.deepcopy(before['hosts']['spark2']['rules'][1]['actions_stats'])
            elif mutation=='identity':rule['rule_identity']['rule']=99
            elif mutation=='skip_sw':rule['skip_sw']=False
            elif mutation=='counter_reset':rule['actions_stats'][0]['packets']=9
            else:bad['network_boot_ids']['spark2']='new-boot'
            with self.subTest(mutation=mutation),self.assertRaises(RuntimeError):hardware_payload_delta(before,bad)


if __name__=='__main__':unittest.main()
