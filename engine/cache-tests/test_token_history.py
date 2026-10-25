import array
import importlib.util
import json
import os
from pathlib import Path
import random
import tempfile
import unittest

ROOT = Path(os.environ.get('TENSORFOLD_CACHE_TEST_ROOT', Path(__file__).resolve().parents[1]))
MODULE = ROOT / 'tensorfold/cuda/token_history.py'
import sys
spec = importlib.util.spec_from_file_location('token_history_under_test', MODULE)
history = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = history
spec.loader.exec_module(history)


def tokens(values):
    return array.array('q', values)


class TokenHistoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'history.sqlite'
        self.h = history.TokenHistories(self.path)
        self.addCleanup(self.h.close)

    def test_append_writes_only_additional_token_payload(self):
        base = list(range(4096))
        a = self.h.put('a', tokens(base))
        before = self.h.written_payload_bytes
        b = self.h.put('b', tokens(base + list(range(4096, 4116))))
        self.assertEqual(self.h.written_payload_bytes - before, 20 * 8)
        self.assertEqual(self.h.read(a), tokens(base).tobytes())
        self.assertEqual(self.h.read(b), tokens(range(4116)).tobytes())
        self.assertTrue(self.h.matcher(b).matches(a))

    def test_partial_tail_is_replaced_without_rewriting_complete_prefix(self):
        a = self.h.put('a', tokens(range(4090)))
        before = self.h.written_payload_bytes
        b = self.h.put('b', tokens(range(4100)))
        self.assertEqual(self.h.written_payload_bytes - before, (2048 + 4) * 8)
        self.assertTrue(self.h.matcher(b).matches(a))
        self.assertEqual(self.h.read(a), tokens(range(4090)).tobytes())

    def test_branches_share_prefix_but_not_different_tails(self):
        base = list(range(4096))
        a = self.h.put('a', tokens(base + [9, 10]))
        b = self.h.put('b', tokens(base + [11, 12]))
        prefix = self.h.put('prefix', tokens(base))
        self.assertTrue(self.h.matcher(a).matches(prefix))
        self.assertTrue(self.h.matcher(b).matches(prefix))
        self.assertFalse(self.h.matcher(a).matches(b))

    def test_removing_old_checkpoint_keeps_shared_ancestry(self):
        a = self.h.put('a', tokens(range(4096)))
        b = self.h.put('b', tokens(range(6144)))
        self.h.remove('a')
        self.assertEqual(self.h.read(b), tokens(range(6144)).tobytes())
        self.assertTrue(self.h.matcher(b).matches(a))
        self.h.remove('b')
        self.assertEqual(self.h.info()['nodes'], 0)
        self.assertEqual(self.h.info()['payload_bytes'], 0)

    def test_same_owner_update_does_not_leak_or_remove_shared_nodes(self):
        self.h.put('a', tokens(range(4096)))
        b = self.h.put('a', tokens(range(6144)))
        self.h.put('a', tokens(range(6144)))
        self.assertEqual(self.h.info()['roots'], 1)
        self.assertEqual(self.h.info()['nodes'], 3)
        self.assertEqual(self.h.read(b), tokens(range(6144)).tobytes())
        self.h.remove('a')
        self.assertEqual(self.h.info()['nodes'], 0)

    def test_recovery_reconciles_unpublished_roots(self):
        keep = self.h.put('manifest-published', tokens(range(2048)))
        self.h.put('crashed-before-publication', tokens(range(4096)))
        self.h.reconcile({'manifest-published'})
        self.assertEqual(self.h.info()['roots'], 1)
        self.assertEqual(self.h.info()['nodes'], 1)
        self.assertEqual(self.h.read(keep), tokens(range(2048)).tobytes())

    def test_disk_reopen_recovers_exact_history(self):
        ref = self.h.put('a', tokens(range(8200)))
        other = history.TokenHistories(self.path)
        try:
            self.assertEqual(other.bind('a', ref.to_json()), ref)
            self.assertEqual(other.read(ref), tokens(range(8200)).tobytes())
        finally:
            other.close()

    def test_payload_corruption_is_detected(self):
        ref = self.h.put('a', tokens(range(2048)))
        with self.h.db:
            self.h.db.execute('UPDATE payloads SET data=?', (bytes(16384),))
        with self.assertRaisesRegex(ValueError, 'corrupt token-history payload'):
            self.h.read(ref)

    def test_node_length_corruption_is_detected(self):
        ref = self.h.put('a', tokens(range(2048)))
        with self.h.db:
            self.h.db.execute('UPDATE nodes SET n=n+1')
        with self.assertRaisesRegex(ValueError, 'inconsistent token-history node'):
            self.h.read(ref)

    def test_checkpoint_binding_cannot_select_another_history(self):
        a = self.h.put('a', tokens(range(2048)))
        b = self.h.put('b', tokens(range(4096)))
        with self.assertRaisesRegex(ValueError, 'root mismatch'):
            self.h.bind('a', b.to_json())
        self.assertEqual(self.h.bind('a', a.to_json()), a)

    def test_signed_keyed_tokens_preserve_exact_bits(self):
        data = [-(1 << 63), (1 << 63) - 1, -8, 0, 123]
        ref = self.h.put('image-keys', tokens(data))
        self.assertEqual(self.h.read(ref), tokens(data).tobytes())

    def test_prompt_matcher_handles_every_boundary_and_tail(self):
        data = tokens(range(8200))
        matcher = history.PrefixMatcher.from_buffer(data)
        for n in (1, 2047, 2048, 2049, 4095, 4096, 4097, 8192, 8200):
            ref = self.h.put(str(n), data[:n])
            self.assertTrue(matcher.matches(ref), n)
        bad = self.h.put('different', tokens([9] + list(range(1, 8200))))
        self.assertFalse(matcher.matches(bad))

    def test_random_fork_and_delete_sequences_preserve_every_live_root(self):
        rng = random.Random(42)
        live = {}
        for i in range(120):
            base = list(rng.choice(list(live.values()))) if live and rng.random() < .7 else list(range(2048))
            data = base[:rng.randrange(1, len(base) + 1)] + [rng.randrange(10000) for _ in range(9)]
            owner = str(i); self.h.put(owner, tokens(data)); live[owner] = data
            if len(live) > 12:
                victim = rng.choice(list(live)); self.h.remove(victim); del live[victim]
            for owner, expected in live.items():
                row = self.h.db.execute('SELECT head,n FROM roots WHERE owner=?', (owner,)).fetchone()
                self.assertEqual(self.h.read(history.HistoryRef(*row)), tokens(expected).tobytes())
        self.h.reconcile(set())
        self.assertEqual(self.h.info()['nodes'], 0)

    def test_ram_manifest_eviction_does_not_delete_disk_checkpoints(self):
        directory = Path(self.tmp.name) / 'points'; directory.mkdir()
        cache = history.ManifestCache(directory, cap_bytes=1500)
        for i in range(30):
            (directory / f'{i}.json').write_text(json.dumps({'n': i, 'blocks': ['a' * 32] * 10}))
            self.assertEqual(cache.get(str(i))['n'], i)
            self.assertLessEqual(cache.bytes, cache.cap_bytes)
        self.assertEqual(len(list(directory.glob('*.json'))), 30)
        self.assertEqual(cache.get('0')['n'], 0)

    def test_large_manifest_is_not_retained_in_ram(self):
        directory = Path(self.tmp.name) / 'points'; directory.mkdir()
        (directory / 'large.json').write_text(json.dumps({'x': 'a' * 2000}))
        cache = history.ManifestCache(directory, cap_bytes=500)
        self.assertEqual(len(cache.get('large')['x']), 2000)
        self.assertEqual(cache.bytes, 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
