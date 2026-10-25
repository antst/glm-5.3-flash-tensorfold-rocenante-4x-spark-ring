"""Full production BlockStore execution with CPU tensors in the pinned runtime."""
import json
import ast
import os
from pathlib import Path
import tempfile
import time
import types
import unittest
from unittest.mock import patch
import sqlite3
import hashlib

import numpy as np
import torch
from tensorfold.cuda import spill_blocks as sb
from tensorfold.cuda.spill import SpillConfig

os.environ['TF_GLM_SPILL_HOST_MIB'] = '512'
os.environ['TF_GLM_SPILL_READ_HOST_MIB'] = '256'
os.environ['TF_GLM_SPILL_STAGE_MIB'] = '1'
os.environ['TF_GLM_SPILL_READ_BLOCKS'] = '8'


class BlockStoreCpuTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cfg = SpillConfig(root=self.tmp.name, gib=.25, min_tokens=1, min_free_gib=0,
                               keep_builds=10, direct=False, writers=1)
        self.plane = types.SimpleNamespace(tensor=torch.arange(131072, dtype=torch.float32).reshape(-1, 1), div=1, pad=0, spanned=False)
        self.extent = types.SimpleNamespace(base=0)
        self.stores = []
        self.store = self.make_store()

    def make_store(self):
        s = sb.BlockStore(self.cfg, planes=[self.plane], block=2048, device=torch.device('cpu'),
                          signature='cpu-regression', weights='cpu-weights', quiet=True,
                          classes=['types:SimpleNamespace'])
        self.stores.append(s)
        return s

    def save(self, n, generation, *, store=None, ids=None, tokens=None, value=1):
        s = store or self.store
        ids = np.arange(n, dtype=np.int64) if ids is None else np.asarray(ids, dtype=np.int64)
        obj = types.SimpleNamespace(rec=torch.full((16,), value, dtype=torch.float32),
                                    head=torch.ones(8, dtype=torch.float32))
        result = s.persist(ids, obj, extent=self.extent, lineage='computation-1',
                           generation=generation, tokens=tokens, head=True)
        self.assertEqual(result, 'queued')
        self.assertTrue(s.drain(timeout=10))
        key = sb.snapshot_key(ids, generation)
        self.assertTrue(s.status(key)['durable'], s.status(key))
        return key

    def ram_hit(self, key):
        """Execute the deployed engine's exact RAM-hit method, with the real store."""
        source = Path(sb.__file__).parents[1] / 'families/glm5_next/cuda/multi.py'
        method = next(n for n in ast.walk(ast.parse(source.read_text()))
                      if isinstance(n, ast.FunctionDef) and n.name == '_touch')
        scope = {}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(source), 'exec'), scope)
        checkpoint = types.SimpleNamespace(spill_key=key)
        decoder = types.SimpleNamespace(kept=[checkpoint], disk=self.store)
        scope['_touch'](decoder, checkpoint)

    def test_daily_ram_hit_protects_old_disk_checkpoint_before_and_after_restart(self):
        hot = self.save(4096, 'popular')
        cold = self.save(4096, 'cold', ids=np.arange(4096) + 50000)
        month_ago = time.time() - 30 * 86400
        yesterday = time.time() - 86400
        for key, used in ((hot, month_ago), (cold, yesterday)):
            self.store.index[key].used = used
            os.utime(self.store._point_path(key, '.json'), (used, used))
        self.assertEqual(self.store._victims()[0][0], hot)
        with patch.object(self.store.accesses, '_persist', wraps=self.store.accesses._persist) as persist:
            self.ram_hit(hot)
            self.assertEqual(self.store._victims()[0][0], cold)
            persist.assert_not_called()
        self.assertTrue(self.store.drain(timeout=10))
        recovered = self.make_store()
        self.assertGreater(recovered.index[hot].used, recovered.index[cold].used)
        self.assertEqual(recovered._victims()[0][0], cold)
        self.assertEqual(set(recovered.index), {hot, cold})

    def test_ram_hit_updates_neither_kv_files_nor_checkpoint_manifest(self):
        key = self.save(4096, 'popular')
        path = self.store._point_path(key, '.json')
        before = Path(path).read_bytes(), os.stat(path).st_mtime_ns, self.store.stats['written_blocks']
        for _ in range(100):
            self.ram_hit(key)
        self.assertEqual(self.store.accesses.info()['dirty_entries'], 1)
        self.assertTrue(self.store.drain(timeout=10))
        self.assertEqual((Path(path).read_bytes(), os.stat(path).st_mtime_ns, self.store.stats['written_blocks']), before)

    def test_failed_shutdown_recency_flush_does_not_lose_kv(self):
        key = self.save(4096, 'popular')
        self.ram_hit(key)
        with patch.object(self.store.accesses, '_persist', side_effect=sqlite3.OperationalError('disk full')):
            self.assertFalse(self.store.drain(timeout=10))
        self.assertTrue(self.store.has(key))
        self.assertEqual(self.store.accesses.info()['dirty_entries'], 1)
        self.assertTrue(self.store.drain(timeout=10))
        self.assertIn(key, self.make_store().index)

    def test_new_checkpoint_and_disk_restore_have_exact_generation_recency_keys(self):
        key = self.save(4096, 'popular')
        self.ram_hit('same-prompt-wrong-generation')
        self.assertEqual(self.store.accesses.info()['dirty_entries'], 0)
        job = self.store.load_async(key, types.SimpleNamespace(base=32768))
        self.store.finish_load(job, timeout=10)
        self.store.release_load_state(job)
        self.assertEqual(self.store.accesses.info()['dirty_entries'], 1)
        self.assertTrue(self.store.drain(timeout=10))
        self.assertGreater(self.make_store().index[key].used, 0)

    def test_metadata_only_release_reuses_existing_kv_compatibility(self):
        from tensorfold.cuda.spill import build_id
        root = Path(sb.__file__).parents[2]
        manifest = root / 'source-manifest.json'
        raw = manifest.read_bytes()
        identity = json.loads(raw)['cache_source_compatibility']
        env = {'SPARK_RING_ENGINE_MANIFEST': str(manifest),
               'SPARK_RING_ENGINE_MANIFEST_SHA256': hashlib.sha256(raw).hexdigest()}
        with patch('tensorfold.families.detect', return_value=types.SimpleNamespace(module=identity['family'])):
            with patch.dict(os.environ, env):
                current = build_id('/fake-model')
            self.assertIn('sources=' + identity['compatible_sources_sha256'][:16], current)
            with patch.dict(os.environ, {**env, 'SPARK_RING_ENGINE_MANIFEST_SHA256': '0' * 64}):
                with self.assertRaisesRegex(ValueError, 'engine manifest changed'):
                    build_id('/fake-model')
            edited = json.loads(raw)
            edited['cache_source_compatibility']['current_sources_sha256'] = '0' * 64
            other = Path(self.tmp.name) / 'other-manifest.json'
            other.write_text(json.dumps(edited))
            with patch.dict(os.environ, {'SPARK_RING_ENGINE_MANIFEST': str(other),
                                        'SPARK_RING_ENGINE_MANIFEST_SHA256': hashlib.sha256(other.read_bytes()).hexdigest()}):
                unmatched = build_id('/fake-model')
            self.assertIn('sources=' + identity['current_sources_sha256'][:16], unmatched)
            self.assertNotEqual(current, unmatched)

    def test_incremental_kv_and_token_history_with_fixed_state_preserved(self):
        a = self.save(4096, 'epoch_1', value=1)
        blocks = self.store.stats['written_blocks']
        token_bytes = self.store.histories.written_payload_bytes
        b = self.save(6144, 'epoch_2', value=2)
        self.assertEqual(self.store.stats['written_blocks'] - blocks, 1)
        self.assertEqual(self.store.histories.written_payload_bytes - token_bytes, 2048 * 8)
        for key, n, value in ((a, 4096, 1), (b, 6144, 2)):
            pt = self.store.index[key]
            self.assertIsNone(pt._ids)
            self.assertIsNone(pt._blocks)
            np.testing.assert_array_equal(pt.ids, np.arange(n))
            self.assertFalse(Path(self.store._point_path(key, '.ids')).exists())
            where = types.SimpleNamespace(base=32768)
            loaded = self.store.load_async(key, where)
            obj, recovered = self.store.finish_load(loaded, timeout=10)
            self.assertTrue(torch.equal(obj.rec, torch.full((16,), value, dtype=torch.float32)))
            self.assertTrue(torch.equal(self.plane.tensor[32768:32768+n], self.plane.tensor[:n]))
            self.store.release_load_state(loaded)

    def test_candidates_select_exact_prefix_and_generation(self):
        a = self.save(4096, 'epoch_1')
        b = self.save(6144, 'epoch_2')
        self.assertEqual([p.key for p in self.store.candidates(np.arange(7000))], [b, a])
        changed = np.arange(7000); changed[8] = -1
        self.assertEqual(self.store.candidates(changed), [])
        self.assertLess(self.store.index_host_bytes, 32 << 20)

    def test_recovery_keeps_both_checkpoints_without_retaining_token_arrays(self):
        a = self.save(4096, 'epoch_1')
        b = self.save(6144, 'epoch_2')
        recovered = self.make_store()
        self.assertEqual(set(recovered.index), {a, b})
        for p in recovered.index.values():
            self.assertIsNone(p._ids)
            self.assertIsNone(p._blocks)
            np.testing.assert_array_equal(p.ids, np.arange(p.n))
        self.assertTrue(recovered._extends(recovered.index[b], recovered.index[a]))

    def test_image_key_history_and_original_tokens_both_survive(self):
        ids = -np.arange(1, 4097, dtype=np.int64)
        original = np.arange(4096, dtype=np.int64)
        key = self.save(4096, 'image_epoch', ids=ids, tokens=original)
        p = self.make_store().index[key]
        np.testing.assert_array_equal(p.ids, ids)
        np.testing.assert_array_equal(p.tokens, original)

    def test_ram_manifest_eviction_does_not_prune_checkpoints(self):
        a = self.save(4096, 'epoch_1'); b = self.save(6144, 'epoch_2')
        self.store.manifests.cap_bytes = 1
        self.store.manifests.forget(a); self.store.manifests.forget(b)
        for p in self.store.index.values():
            self.assertTrue(p.blocks)
        self.assertEqual(self.store.manifests.bytes, 0)
        self.assertEqual(set(self.store.index), {a, b})
        self.assertTrue(self.store.has(a)); self.assertTrue(self.store.has(b))

    def test_removed_checkpoint_does_not_delete_shared_token_history(self):
        a = self.save(4096, 'epoch_1'); b = self.save(6144, 'epoch_2')
        with self.store.lock:
            self.store._remove(a, 'pruned_turns')
        np.testing.assert_array_equal(self.store.index[b].ids, np.arange(6144))
        self.assertEqual(self.store.histories.info()['roots'], 1)
        with self.store.lock:
            self.store._remove(b, 'pruned_turns')
        self.assertEqual(self.store.histories.info()['nodes'], 0)

    def test_prepublication_write_failure_cleans_history_and_block_references(self):
        real_put = sb._put
        def fail_state(path, *a, **kw):
            if str(path).endswith('.st'):
                raise OSError('injected state write refusal')
            return real_put(path, *a, **kw)
        with patch.object(sb, '_put', side_effect=fail_state):
            result = self.store.persist(np.arange(4096), types.SimpleNamespace(rec=torch.ones(16)),
                                        extent=self.extent, lineage='failure', generation='failed_epoch')
            self.assertEqual(result, 'queued')
            self.assertTrue(self.store.drain(timeout=10))
        self.assertFalse(self.store.index)
        self.assertEqual(self.store.histories.info()['roots'], 0)
        self.assertEqual(self.store.host_pending, 0)
        self.assertEqual(self.store.stats['write_failed'], 1)

    def test_recovery_discards_orphan_history_without_damaging_valid_checkpoint(self):
        key = self.save(4096, 'epoch_1')
        self.store.histories.put('unpublished/ids', np.arange(8192))
        recovered = self.make_store()
        self.assertEqual(set(recovered.index), {key})
        self.assertEqual(recovered.histories.info()['roots'], 1)

    def test_thousand_long_checkpoints_survive_beyond_old_ram_limit(self):
        self.store.turns = 0  # stress disk capacity, not the independent per-conversation retention rule
        started = time.monotonic()
        for i in range(1000):
            self.save(100000 + 8 * i, 'growth_' + str(i))
        info = self.store.histories.info()
        self.assertEqual(len(self.store.index), 1000)
        self.assertGreater(info['logical_token_bytes'], 512 << 20)
        self.assertLess(info['payload_bytes'], info['logical_token_bytes'] // 20)
        self.assertLess(self.store.index_host_bytes, 32 << 20)
        self.assertLessEqual(self.store.manifests.bytes, self.store.manifests.cap_bytes)
        self.assertTrue(all(p._ids is None and p._blocks is None for p in self.store.index.values()))
        # Recovery must retain every on-disk checkpoint, even with an intentionally tiny hot manifest cache.
        recovered = self.make_store()
        self.assertEqual(len(recovered.index), 1000)
        self.assertLess(recovered.index_host_bytes, 32 << 20)
        self.assertLessEqual(recovered.manifests.bytes, recovered.manifests.cap_bytes)
        print('SHARED_HISTORY_GROWTH ' + json.dumps({
            'checkpoints': 1000, 'logicalTokenBytesWithoutSharing': info['logical_token_bytes'],
            'uniqueTokenPayloadBytes': info['payload_bytes'], 'metadataBudgetBytes': self.store.host_cap,
            'indexAccountingBytes': self.store.index_host_bytes, 'historySqliteCacheBytes': info['sqlite_cache_cap_bytes'],
            'manifestCacheBytes': self.store.manifests.bytes, 'retainedAfterRecovery': len(recovered.index),
            'seconds': round(time.monotonic() - started, 3)}), flush=True)


if __name__ == '__main__':
    unittest.main(verbosity=2)
