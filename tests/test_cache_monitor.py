"""Offline monitor tests: no SSH, HTTP, inference or real monitor starts."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

module_path = Path(__file__).with_name('cache_monitor.py')
if not module_path.exists():
    module_path = Path(__file__).with_name('cache-monitor.py')
if not module_path.exists():
    module_path = Path(__file__).resolve().parents[1] / 'scripts' / 'cache-monitor.py'
spec = importlib.util.spec_from_file_location('cache_monitor', module_path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
BOOT = '11111111-1111-1111-1111-111111111111'
OTHER_BOOT = '22222222-2222-2222-2222-222222222222'


def config():
    return {'nodes': [{'name': 'spark' + str(i), 'management_address': '192.0.2.' + str(i),
                       'ssh': 'admin@192.0.2.' + str(i), 'host_key_alias': 'spark' + str(i)} for i in range(1, 5)],
            'api_port': 8888, 'shared_root': '/example/recipe', 'profile':
            {'context': 1048576, 'parallel': 6, 'multi_window': 64}, 'image': 'public/image@' + m.GLM_MIA107_DIGEST}


def raw_health():
    d = dict.fromkeys(m.HEALTH_FIELDS, 0)
    d.update(ok=True, backend='tensorfold', context_length=1048576, pool_tokens=6821888,
             pool_free_tokens=530432, kept_bytes=270 << 20, kept_bytes_cap=4 * m.GIB,
             kept_prompts=6, requests_running=6,
             streams={'max': 6, 'decoding': 2, 'filling': 4, 'paused': 0},
             kept_mix={'own': 6, 'shared': 0, 'chats': 6, 'by_chat': {'SECRET': 'DO NOT LOG'}},
             prompt='DO NOT LOG', messages=['DO NOT LOG'])
    d.update(dict.fromkeys(m.SERVICE_COUNTERS, 0))
    return d


def raw_host():
    return {'boot_id': BOOT, 'memory': {'MemAvailable': 22 * m.GIB, 'MemFree': 18 * m.GIB,
                                     'SwapFree': 0, 'SwapTotal': 0},
            'vmstat': {'pgmajfault': 100, 'pswpin': 0, 'pswpout': 0}}


def sample(second=0):
    t = '2026-10-13T19:00:%02d+00:00' % second
    return {'schema': 'tensorfold-cache-monitor/v1', 'config_sha256': 'same', 'utc': t,
            'complete': True, 'health': {'ok': True, 'observed_utc': t, 'data': m.clean_health(raw_health())},
            'hosts': {n['name']: {'ok': True, 'observed_utc': t, 'data': m.clean_host(raw_host())}
                      for n in config()['nodes']}}


class DataTests(unittest.TestCase):
    def test_config_paths_resolve_relative_to_config_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'config.json'
            c = config()
            c['ssh_options'] = {'identity_file': 'key', 'known_hosts_file': 'hosts'}
            path.write_text(json.dumps(c))
            loaded, _ = m.load_config(path)
            self.assertEqual(loaded['ssh_options']['identity_file'], str(Path(tmp) / 'key'))
            self.assertEqual(loaded['ssh_options']['known_hosts_file'], str(Path(tmp) / 'hosts'))

    def test_config_rejects_ssh_destination_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'config.json'
            c = config()
            c['nodes'][0]['ssh'] = 'admin@192.0.2.99'
            path.write_text(json.dumps(c))
            with self.assertRaises(ValueError):
                m.load_config(path)

    def test_health_keeps_only_aggregate_allowlist(self):
        h = m.clean_health(raw_health())
        self.assertNotIn('DO NOT LOG', json.dumps(h))
        self.assertNotIn('by_chat', h['kept_mix'])
        self.assertEqual(h['pool_used_tokens'], 6291456)

    def test_missing_counter_is_error_not_zero(self):
        h = raw_health()
        del h['cached_tokens_total']
        with self.assertRaises(ValueError):
            m.clean_health(h)

    def test_false_nan_and_negative_values_rejected(self):
        for bad in [False, float('nan'), -1]:
            h = raw_health()
            h['cached_tokens_total'] = bad
            with self.assertRaises(ValueError):
                m.clean_health(h)

    def test_bad_memory_and_boot_rejected(self):
        for key, value in [('boot_id', 'no'), ('memory', {})]:
            h = raw_host()
            h[key] = value
            with self.assertRaises((ValueError, KeyError)):
                m.clean_host(h)

    def test_ssh_is_readonly_strict_and_configured(self):
        c = config()
        c['ssh_options'] = {'identity_file': '/key', 'known_hosts_file': '/hosts'}
        argv = m.ssh_command(c['nodes'][0], c)
        self.assertIn('StrictHostKeyChecking=yes', argv)
        self.assertIn('BatchMode=yes', argv)
        self.assertIn('UserKnownHostsFile=/hosts', argv)
        self.assertIn('/key', argv)
        self.assertIn('admin@192.0.2.1', argv)
        self.assertNotIn('sudo', argv[-1])
        self.assertIn('/proc/vmstat', argv[-1])

    def test_command_timeout_is_bounded(self):
        with patch.object(m.subprocess, 'run', side_effect=subprocess.TimeoutExpired(['ssh'], 10)) as run:
            result = m.attempt(lambda: m.command_json(['ssh']))
        self.assertFalse(result['ok'])
        self.assertEqual(run.call_args.kwargs['timeout'], 10)
        self.assertNotIn('data', result)

    def test_remote_stderr_is_not_persisted(self):
        result = subprocess.CompletedProcess([], 1, '', 'SECRET')
        with patch.object(m.subprocess, 'run', return_value=result):
            record = m.attempt(lambda: m.command_json(['ssh']))
        self.assertNotIn('SECRET', json.dumps(record))

    def test_partial_collection_keeps_good_hosts(self):
        def fake(argv):
            if 'ssh' == argv[0]:
                if 'admin@192.0.2.2' in argv:
                    raise RuntimeError('simulated unreachable')
                return raw_host()
            return raw_health()
        with patch.object(m, 'command_json', side_effect=fake):
            value = m.collect(config(), 'same')
        self.assertFalse(value['complete'])
        self.assertTrue(value['health']['ok'])
        self.assertFalse(value['hosts']['spark2']['ok'])
        self.assertTrue(value['hosts']['spark1']['ok'])

    def test_minima_exclude_failures(self):
        s = m.update_summary(None, sample(), config())
        broken = sample(30)
        broken['complete'] = False
        broken['hosts']['spark1'] = {'ok': False, 'observed_utc': broken['utc'], 'error': {'type': 'OSError'}}
        s = m.update_summary(s, broken, config())
        self.assertEqual(s['hosts']['spark1']['MemAvailable_minimum_sampled_bytes'], 22 * m.GIB)
        self.assertEqual(s['hosts']['spark1']['errors'], 1)
        self.assertEqual(s['failed_samples'], 1)
        self.assertIsNone(s['p8_estimate']['minimum_current_mem_available_bytes'])

    def test_vmstat_deltas_and_boot_reset(self):
        s = m.update_summary(None, sample(), config())
        second = sample(30)
        second['hosts']['spark1']['data']['vmstat']['pswpout'] = 10
        s = m.update_summary(s, second, config())
        self.assertEqual(s['hosts']['spark1']['vmstat_delta_totals']['pswpout'], 10)
        third = sample(59)
        third['hosts']['spark1']['data']['boot_id'] = OTHER_BOOT
        s = m.update_summary(s, third, config())
        self.assertIsNone(s['hosts']['spark1']['last_counter_delta'])
        self.assertEqual(s['hosts']['spark1']['counter_resets'], 1)
        self.assertEqual(s['hosts']['spark1']['vmstat_delta_totals']['pswpout'], 10)
        self.assertFalse(s['service_interval']['valid'])

    def test_vmstat_decrease_is_reset_not_negative_delta(self):
        s = m.update_summary(None, sample(), config())
        second = sample(30)
        second['hosts']['spark1']['data']['vmstat']['pgmajfault'] = 0
        s = m.update_summary(s, second, config())
        self.assertEqual(s['hosts']['spark1']['counter_resets'], 1)
        self.assertIsNone(s['hosts']['spark1']['last_counter_delta'])

    def test_zero_requests_is_not_100_percent_cache(self):
        s = m.update_summary(None, sample(), config())
        s = m.update_summary(s, sample(30), config())
        self.assertTrue(s['service_interval']['valid'])
        self.assertIsNone(s['service_interval']['cached_prompt_fraction'])

    def test_cache_fraction_and_throughput(self):
        s = m.update_summary(None, sample(), config())
        second = sample(30)
        second['health']['data']['counters'].update(requests_total=2, prompt_tokens_total=1000,
                                                   cached_tokens_total=900, completion_tokens_total=60)
        s = m.update_summary(s, second, config())
        self.assertEqual(s['service_interval']['cached_prompt_fraction'], .9)
        self.assertEqual(s['service_interval']['completion_tokens_per_second'], 2)

    def test_service_reset_discards_interval(self):
        first = sample()
        first['health']['data']['counters']['requests_total'] = 100
        s = m.update_summary(None, first, config())
        s = m.update_summary(s, sample(30), config())
        self.assertFalse(s['service_interval']['valid'])
        self.assertEqual(s['service_counter_resets'], 1)

    def test_gap_discards_first_recovered_interval(self):
        s = m.update_summary(None, sample(), config())
        broken = sample(30)
        broken['complete'] = False
        broken['health'] = {'ok': False, 'observed_utc': broken['utc'], 'error': {'type': 'TimeoutError'}}
        s = m.update_summary(s, broken, config())
        s = m.update_summary(s, sample(59), config())
        self.assertFalse(s['service_interval']['valid'])
        self.assertEqual(s['health_errors'], 1)

    def test_config_change_resets_watermarks(self):
        s = m.update_summary(None, sample(), config())
        second = sample(30)
        second['config_sha256'] = 'different'
        s = m.update_summary(s, second, config())
        self.assertEqual(s['samples'], 1)

    def test_p8_estimate_cannot_qualify_anything(self):
        e = m.p8_estimate(sample(), config())
        self.assertEqual(e['estimated_additional_arena_and_row_scratch_bytes'], 2097152 * 6780)
        self.assertFalse(e['eight_lanes_qualified'])
        self.assertIsNone(e['additional_p8_fixed_buffer_bytes'])
        self.assertEqual(e['decision'], 'hardware_qualification_required')
        c = config()
        c['image'] = 'different-model'
        self.assertEqual(m.p8_estimate(sample(), c)['status'], 'unsupported_profile')

    def test_p8_admission_still_does_not_qualify_eight_full_histories(self):
        c = config()
        c['profile']['parallel'] = 8
        value = sample()
        value['health']['data']['streams']['max'] = 8
        estimate = m.p8_estimate(value, c)
        self.assertEqual(estimate['configured_admission_lanes'], 8)
        self.assertFalse(estimate['eight_lanes_qualified'])
        self.assertFalse(estimate['eight_near_1m_histories_qualified'])
        self.assertEqual(estimate['decision'], 'hardware_qualification_required')
        value['health']['data']['streams']['max'] = 6
        self.assertEqual(m.p8_estimate(value, c)['status'], 'unsupported_profile')


class StorageAndOwnershipTests(unittest.TestCase):
    def test_rotation_total_bytes_and_age(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = m.Store(tmp, max_bytes=14000, segment_bytes=4000, age=100)
            for i in range(10):
                store.append(sample(), config(), now=1000 + i)
            files = list(Path(tmp).glob('samples-*.jsonl'))
            self.assertLessEqual(sum(p.stat().st_size for p in files), 14000)
            self.assertTrue(files)
            store.prune(1200)
            self.assertEqual(list(Path(tmp).glob('samples-*.jsonl')), [])
            self.assertTrue((Path(tmp) / 'summary.json').exists())

    def test_hourly_rotation_even_with_recent_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = m.Store(tmp)
            store.append(sample(), config(), now=1000)
            store.append(sample(30), config(), now=4700)
            self.assertEqual(len(list(Path(tmp).glob('samples-*.jsonl'))), 2)

    def test_state_missing_status_is_stale_not_healthy(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = m.status(Path(tmp), ['not-a-monitor'])
            self.assertFalse(state['running'])
            self.assertTrue(state['telemetry_stale'])
            self.assertIsNone(state['latest'])

    def test_flock_prevents_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            with m.flock(Path(tmp) / 'daemon.lock'):
                with self.assertRaises(BlockingIOError):
                    m.flock(Path(tmp) / 'daemon.lock')

    def test_pid_identity_rejects_reuse_other_boot_other_argv(self):
        good = {'pid': 123, 'boot_id': BOOT, 'start_ticks': 100, 'argv': ['monitor']}
        self.assertTrue(m.identity_matches(good, ['monitor'], lambda _: good))
        for key, value in [('start_ticks', 101), ('boot_id', OTHER_BOOT), ('argv', ['unrelated'])]:
            changed = {**good, key: value}
            self.assertFalse(m.identity_matches(good, ['monitor'], lambda _: changed))

    def test_exit_before_pidfd_open_needs_no_signal(self):
        with patch.object(m.os, 'pidfd_open', side_effect=ProcessLookupError), \
             patch.object(m.signal, 'pidfd_send_signal') as send:
            self.assertFalse(m.safe_signal({'pid': 123}, ['monitor']))
            send.assert_not_called()

    def test_optional_receipt_can_disappear(self):
        with patch.object(m, 'read_json', side_effect=FileNotFoundError):
            self.assertIsNone(m.optional_json('/missing'))

    def test_signal_refuses_foreign_identity(self):
        with patch.object(m.os, 'pidfd_open', return_value=99), patch.object(m.os, 'close'), \
             patch.object(m, 'identity_matches', return_value=False), patch.object(m.signal, 'pidfd_send_signal') as send:
            with self.assertRaises(RuntimeError):
                m.safe_signal({'pid': 123}, ['monitor'])
            send.assert_not_called()

    def test_pidfd_stops_only_our_dummy_child(self):
        # Harmless owned sleeper, not a telemetry daemon and no network calls.
        child = subprocess.Popen([sys.executable, '-B', '-c', 'import time; time.sleep(60)'])
        try:
            record = m.proc_identity(child.pid)
            m.safe_signal(record, record['argv'])
            child.wait(timeout=5)
            self.assertNotEqual(child.returncode, 0)
        finally:
            if child.poll() is None:
                child.kill()
                child.wait()

    def test_oversized_record_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            value = sample()
            value['extra'] = 'x' * 65536
            with self.assertRaises(ValueError):
                m.Store(tmp).append(value, config())


if __name__ == '__main__':
    unittest.main()
