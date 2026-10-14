"""Offline relay receipt rejection tests; no CUDA or transport qualification."""
import copy
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
spec = importlib.util.spec_from_file_location('ring_qualifier', ROOT / 'scripts/qualify.py')
qualify = importlib.util.module_from_spec(spec); spec.loader.exec_module(qualify)
SHA = 'a' * 64


def state(calls):
    return dict(enabled=True, source_sha256=SHA, route='adjacent-two-round', rank=0, world=4,
                neighbors=[1, 3], calls=calls, rounds=calls*2, scratch_bytes=32,
                peak_growth_bytes=32, max_scratch_bytes=64 << 20, max_growth_bytes=128 << 20,
                scratch_by_dtype={'torch.float32': dict(buffers=2, elements_each=4, bytes=32)})


def report():
    def controller(calls, native=0, graphs=0):
        return dict(ring_relay=state(calls), eager_p2p_calls={'exchange_all': calls},
                    eager_native_calls=native, guarded_replays=graphs)
    return dict(rank=0, mixed_p2p_passed=True, ring_relay_edges_passed=True,
                mixed_p2p=dict(passed=True, before=controller(0), after=controller(3, 6, 3),
                    rows=[dict(epoch=e, correct=True, send_bytes_per_tensor=65536) for e in (41,42,43)]),
                ring_relay_edges=dict(passed=True, before=controller(3,6,3), after=controller(5,6,3),
                    rows=[dict(elements=1,dtype='float32',correct=True), dict(elements=7,dtype='bfloat16',correct=True)]),
                native_final=controller(5,6,3))


class RingReceiptTests(unittest.TestCase):
    def test_complete_mixed_and_edge_proof(self):
        self.assertTrue(qualify.validate_mixed_p2p(report(), relay_sha=SHA))
        legacy = report(); del legacy['ring_relay_edges']
        self.assertTrue(qualify.validate_mixed_p2p(legacy))
        self.assertFalse(qualify.validate_mixed_p2p(legacy, relay_sha=SHA))

    def test_wrong_identity_route_accounting_and_caps_fail(self):
        updates = [dict(enabled=False), dict(source_sha256='b'*64), dict(rank=1), dict(world=8),
                   dict(neighbors=[1,2,3]), dict(route='direct'), dict(rounds=7), dict(calls=True),
                   dict(max_scratch_bytes=128<<20), dict(peak_growth_bytes=(128<<20)+1),
                   dict(scratch_bytes=33), dict(scratch_by_dtype={})]
        for update in updates:
            with self.subTest(update=update):
                r = report(); r['mixed_p2p']['after']['ring_relay'].update(update)
                self.assertFalse(qualify.validate_mixed_p2p(r, relay_sha=SHA))

    def test_missing_epochs_edges_stream_counts_and_final_drift_fail(self):
        mutations = [lambda r: r['mixed_p2p']['rows'].pop(),
                     lambda r: r['mixed_p2p']['after'].update(guarded_replays=2),
                     lambda r: r['mixed_p2p']['after'].update(eager_native_calls=5),
                     lambda r: r.update(ring_relay_edges_passed=False),
                     lambda r: r['ring_relay_edges']['rows'][0].update(elements=2),
                     lambda r: r['ring_relay_edges']['before'].update(ring_relay=state(4)),
                     lambda r: r['native_final'].update(ring_relay=state(6))]
        for mutation in mutations:
            r = report(); mutation(r)
            self.assertFalse(qualify.validate_mixed_p2p(r, relay_sha=SHA))


if __name__ == '__main__':unittest.main()
