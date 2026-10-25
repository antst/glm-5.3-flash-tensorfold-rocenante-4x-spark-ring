import ast
import hashlib
import importlib.util
import json
import os
import pathlib
import sqlite3
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = pathlib.Path(os.environ.get('TENSORFOLD_CACHE_TEST_ROOT', pathlib.Path(__file__).resolve().parents[1]))
module_path = ROOT / 'tensorfold/cuda/token_history.py'
spec = importlib.util.spec_from_file_location('recency_under_test', module_path)
history = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = history
spec.loader.exec_module(history)


class RecencyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = pathlib.Path(self.tmp.name) / 'recency.sqlite'
        self.now = 0.0
        self.access = history.AccessTimes(self.path, clock=lambda: self.now)
        self.addCleanup(self.access.db.close)

    def test_many_hits_coalesce_and_write_nothing_before_two_hours(self):
        with patch.object(self.access, '_persist', wraps=self.access._persist) as persist:
            for i in range(10000):
                self.access.mark('daily-prompt', 1000.0 + i)
                self.assertTrue(self.access.flush())
            self.assertEqual(len(self.access.dirty), 1)
            self.now = 7199.9
            self.assertTrue(self.access.flush())
            persist.assert_not_called()
            self.now = 7200.0
            self.assertTrue(self.access.flush())
            self.assertEqual(persist.call_count, 1)
            self.assertEqual(self.access.saved('daily-prompt', 0), 10999.0)
            self.assertEqual(self.access.info()['dirty_entries'], 0)

    def test_clean_shutdown_force_flush_survives_reopen(self):
        self.access.mark('popular', 5678.0)
        self.assertTrue(self.access.flush(force=True))
        reopened = history.AccessTimes(self.path)
        self.addCleanup(reopened.db.close)
        self.assertEqual(reopened.saved('popular', 10.0), 5678.0)

    def test_abrupt_exit_loses_only_unflushed_recency(self):
        self.access.mark('popular', 1000.0)
        self.access.flush(force=True)
        self.access.mark('popular', 2000.0)
        reopened = history.AccessTimes(self.path)
        self.addCleanup(reopened.db.close)
        self.assertEqual(reopened.saved('popular', 10.0), 1000.0)
        self.assertEqual(self.access.dirty['popular'], 2000.0)

    def test_failed_batch_is_retained_and_retried(self):
        self.access.mark('popular', 2000.0)
        with patch.object(self.access, '_persist', side_effect=sqlite3.OperationalError('disk full')):
            self.assertFalse(self.access.flush(force=True))
        self.assertEqual(self.access.info()['flush_errors'], 1)
        self.assertEqual(self.access.dirty['popular'], 2000.0)
        self.assertTrue(self.access.flush(force=True))
        self.assertEqual(self.access.saved('popular', 0), 2000.0)

    def test_hit_during_flush_is_not_lost(self):
        self.access.mark('popular', 1000.0)
        original = self.access._persist
        def during(batch):
            self.access.mark('popular', 2000.0)
            original(batch)
        with patch.object(self.access, '_persist', side_effect=during):
            self.assertTrue(self.access.flush(force=True))
        self.assertEqual(self.access.dirty['popular'], 2000.0)
        self.access.flush(force=True)
        self.assertEqual(self.access.saved('popular', 0), 2000.0)

    def test_deleted_unpersisted_entries_do_not_accumulate(self):
        for i in range(10000):
            key = str(i)
            self.access.mark(key, float(i))
            self.access.forget(key)
        self.assertEqual(self.access.dirty, {})

    def test_delete_during_flush_is_eventually_persisted(self):
        self.access.mark('removed', 1000.0)
        original = self.access._persist
        def during(batch):
            self.access.forget('removed')
            original(batch)
        with patch.object(self.access, '_persist', side_effect=during):
            self.assertTrue(self.access.flush(force=True))
        self.assertIsNone(self.access.dirty['removed'])
        self.access.flush(force=True)
        self.assertEqual(self.access.saved('removed', 0), 0)

    def test_recovery_reconciles_deleted_checkpoint_records(self):
        self.access.mark('keep', 1000.0)
        self.access.mark('gone', 2000.0)
        self.access.flush(force=True)
        self.access.reconcile({'keep'})
        self.assertEqual(self.access.saved('keep', 0), 1000.0)
        self.assertEqual(self.access.saved('gone', 0), 0)
        self.assertEqual(self.access.persisted, {'keep'})

    def test_clean_empty_flush_performs_no_transaction(self):
        with patch.object(self.access, '_persist') as persist:
            self.assertTrue(self.access.flush(force=True))
            persist.assert_not_called()

    def test_saved_timestamp_never_moves_backwards(self):
        self.access.mark('popular', 2000.0)
        self.access.flush(force=True)
        self.access.mark('popular', 1000.0)
        self.access.flush(force=True)
        self.assertEqual(self.access.saved('popular', 0), 2000.0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
