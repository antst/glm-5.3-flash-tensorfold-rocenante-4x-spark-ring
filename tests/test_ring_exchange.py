"""Four-rank byte simulator: actual relay code, no Torch/CUDA/network imports."""
import importlib.util
import itertools
from pathlib import Path
import threading
import types
import unittest

PATH = Path(__file__).resolve().parents[1] / 'scripts/ring_exchange.py'
spec = importlib.util.spec_from_file_location('ring_exchange_under_test', PATH)
ring = importlib.util.module_from_spec(spec); spec.loader.exec_module(ring)
WIDTH = {'torch.float32': 4, 'torch.bfloat16': 2, 'torch.int32': 4, 'torch.int64': 8}
ADDRESSES = itertools.count(1)


class Tensor:
    def __init__(self, n, dtype='torch.float32', *, shape=None, device='cpu', storage=None, offset=0, address=None):
        self.dtype, self.device = dtype, device
        self.shape = (n,) if shape is None else shape
        self.n, self.offset = n, offset
        self.storage = bytearray(n * WIDTH[dtype]) if storage is None else storage
        self.address = next(ADDRESSES) * 1000000000 if address is None else address
        self.contiguous = True
        self.recorded = []

    @property
    def is_cuda(self): return self.device == 'cuda:0'
    def record_stream(self, stream): self.recorded.append(stream)
    def numel(self): return self.n
    def element_size(self): return WIDTH[self.dtype]
    def is_contiguous(self): return self.contiguous
    def data_ptr(self): return self.address + self.offset
    def raw(self): return bytes(self.storage[self.offset:self.offset + self.n * self.element_size()])
    def reshape(self, *shape):
        assert shape == (-1,)
        return Tensor(self.n, self.dtype, device=self.device, storage=self.storage, offset=self.offset, address=self.address)
    def __getitem__(self, sl):
        a, b, step = sl.indices(self.n)
        assert step == 1
        return Tensor(max(0, b-a), self.dtype, device=self.device, storage=self.storage,
                      offset=self.offset+a*self.element_size(), address=self.address)
    def zero_(self):
        self.storage[self.offset:self.offset+self.n*self.element_size()] = bytes(self.n*self.element_size())
        return self
    def copy_(self, other):
        assert self.n == other.n and self.dtype == other.dtype
        self.storage[self.offset:self.offset+self.n*self.element_size()] = other.raw()
        return self


class Stream:
    def __init__(self, name): self.name, self.syncs = name, 0
    def synchronize(self): self.syncs += 1


class Torch:
    def __init__(self):
        self.stream = Stream('initial')
        self.capturing = False
        self.allocated = []
        self.cuda = types.SimpleNamespace(current_stream=lambda: self.stream,
                                          is_current_stream_capturing=lambda: self.capturing)
    def empty(self, n, dtype, device):
        t = Tensor(n, dtype, device=device)
        self.allocated.append(t)
        return t


def inputs(rank, geometry, epoch=0, device='cpu'):
    sends, recvs = {}, {}
    for peer in range(4):
        if peer == rank: continue
        sends[peer], recvs[peer] = [], []
        for index, (n, dtype) in enumerate(geometry):
            a, b = Tensor(n, dtype, device=device), Tensor(n, dtype, device=device)
            # Raw payloads include every byte value, NaN patterns etc.; never
            # decode floats. Each source/destination/list index/epoch differs.
            a.storage[:] = bytes((j + rank*71 + peer*29 + index*13 + epoch*17) % 256 for j in range(n*WIDTH[dtype]))
            b.storage[:] = b'\xa5' * len(b.storage)
            sends[peer].append(a); recvs[peer].append(b)
    return sends, recvs


class Collective:
    """Enforce the original wrapper's paired counts and simultaneous raw copies."""
    def __init__(self):
        self.barrier = threading.Barrier(4, timeout=3)
        self.lock = threading.Lock()
        self.steps = [0]*4
        self.pending = {}
        self.records = []
        self.error = None

    def __call__(self, comm, sends, recvs):
        rank = comm.rank
        try:
            assert list(sends) == sorted(((rank-1)%4, (rank+1)%4))
            assert set(sends) == set(recvs)
            for p in sends:
                assert len(sends[p]) == len(recvs[p])
                for s, r in zip(sends[p], recvs[p]):
                    assert s.numel() == r.numel() and s.dtype == r.dtype
                    assert s.is_contiguous() and r.is_contiguous()
            with self.lock:
                phase = self.steps[rank]; self.steps[rank] += 1
                self.pending.setdefault(phase, {})[rank] = (sends, recvs)
            self.barrier.wait()
            if rank == 0:
                group = self.pending[phase]
                copied = []
                for source, (out, _) in group.items():
                    for dest, tensors in out.items():
                        targets = group[dest][1][source]
                        assert len(tensors) == len(targets)
                        for s, t in zip(tensors, targets):
                            assert s.numel() == t.numel() and s.dtype == t.dtype
                            copied.append((t, s.raw()))
                for t, raw in copied:
                    t.storage[t.offset:t.offset+len(raw)] = raw
                self.records.append({r: {'peers': list(out), 'tx': sum(t.numel()*t.element_size() for v in out.values() for t in v)}
                                     for r, (out, _) in group.items()})
            self.barrier.wait()
        except BaseException as exc:
            self.error = exc
            self.barrier.abort()
            raise


class RelayTests(unittest.TestCase):
    def make(self, **kwargs):
        collective = Collective()
        torches = [Torch() for _ in range(4)]
        relays = [ring.RingExchange(types.SimpleNamespace(rank=r, world=4), collective, torches[r], **kwargs) for r in range(4)]
        return collective, torches, relays

    def wave(self, collective, relays, geometry, epoch=0, device='cpu'):
        banks = [inputs(r, geometry, epoch, device) for r in range(4)]
        expected = {(r, p, i): banks[p][0][r][i].raw() for r in range(4) for p in range(4) if p != r for i in range(len(geometry))}
        errors = []
        def run(r):
            try: relays[r].exchange_all(*banks[r])
            except BaseException as exc:
                errors.append(exc); collective.barrier.abort()
        threads = [threading.Thread(target=run, args=(r,)) for r in range(4)]
        for t in threads: t.start()
        for t in threads: t.join(5)
        self.assertFalse(any(t.is_alive() for t in threads))
        self.assertEqual(errors, [], repr(collective.error))
        for r in range(4):
            for p, slots in banks[r][1].items():
                for i, t in enumerate(slots): self.assertEqual(t.raw(), expected[r,p,i], (r,p,i))
        return banks

    def test_all_source_destination_slots_even_odd_one_zero_and_multiple_dtypes(self):
        for geometry in ([(16,'torch.float32')], [(7,'torch.bfloat16')], [(1,'torch.float32')],
                         [(0,'torch.int64')], [],
                         [(0,'torch.float32'),(9,'torch.float32'),(5,'torch.bfloat16'),(8,'torch.float32'),(1,'torch.int64')]):
            with self.subTest(geometry=geometry):
                collective, torches, relays = self.make()
                self.wave(collective, relays, geometry)
                b = sum(n*WIDTH[d] for n,d in geometry)
                pad = sum(2*WIDTH[d] for n,d in geometry if n%2)
                self.assertEqual(len(collective.records), 2 if b else 0)
                for r in range(4):
                    self.assertEqual(sum(v[r]['tx'] for v in collective.records), 4*b+pad)
                    self.assertEqual(relays[r].report()['tx_bytes'], 4*b+pad)
                    self.assertLessEqual(len(relays[r]._buffers), len(ring.SUPPORTED_DTYPES))
                    self.assertTrue(all(len(v)==2 for v in relays[r]._buffers.values()))

    def test_reuse_growth_peak_and_cross_stream_allocator_registration(self):
        collective, torches, relays = self.make()
        self.wave(collective, relays, [(16,'torch.float32')], device='cuda:0')
        initial = [tuple(id(t) for t in x._buffers['torch.float32']) for x in relays]
        for torch in torches: torch.stream = Stream('second')
        self.wave(collective, relays, [(8,'torch.float32')], epoch=1, device='cuda:0')
        for r, relay in enumerate(relays):
            self.assertEqual(initial[r], tuple(id(t) for t in relay._buffers['torch.float32']))
            self.assertEqual(len(torches[r].allocated), 2)
            for t in relay._buffers['torch.float32']:
                self.assertEqual([s.name for s in t.recorded], ['initial','initial','second','second'])
        self.wave(collective, relays, [(32,'torch.float32')], epoch=2, device='cuda:0')
        for r, relay in enumerate(relays):
            self.assertEqual(relay.report()['scratch_bytes'], 128)
            self.assertEqual(relay.report()['peak_growth_bytes'], 64+128)
            self.assertEqual(relay.report()['growth_syncs'], 1)
            self.assertEqual(torches[r].stream.syncs, 1)
            self.assertEqual(len(relay._buffers['torch.float32']), 2)

    def test_scratch_limits_refuse_before_allocation_or_collective(self):
        calls = []
        torch = Torch()
        relay = ring.RingExchange(types.SimpleNamespace(rank=0,world=4), lambda *a: calls.append(a), torch,
                                  max_scratch_bytes=63)
        with self.assertRaisesRegex(ValueError, 'retained scratch'):
            relay.exchange_all(*inputs(0,[(16,'torch.float32')]))
        self.assertEqual(torch.allocated, []); self.assertEqual(calls, [])
        # Pre-reserve old geometry without submitting to a simulated collective.
        relay.max_scratch_bytes=128; relay._reserve({'torch.float32':(8,4)}, 'cpu')
        relay.max_growth_bytes=191
        with self.assertRaisesRegex(ValueError, 'old-plus-new'):
            relay.exchange_all(*inputs(0,[(32,'torch.float32')]))
        self.assertEqual(len(torch.allocated), 2); self.assertEqual(calls, [])
        self.assertEqual(torch.stream.syncs, 0)

    def test_invalid_geometry_never_falls_back_to_diagonal_or_submits(self):
        def cases(s, r):
            yield lambda: s.pop(2)
            yield lambda: r[1].append(Tensor(8))
            yield lambda: s[2].__setitem__(0, Tensor(7))
            yield lambda: r[3].__setitem__(0, Tensor(8,'torch.bfloat16'))
            yield lambda: setattr(r[1][0], 'shape', (2,4))
            yield lambda: setattr(s[3][0], 'contiguous', False)
            yield lambda: setattr(r[2][0], 'device', 'cuda:0')
            yield lambda: r[2].__setitem__(0, s[3][0])
            yield lambda: r[3].__setitem__(0, r[2][0])
        for index in range(9):
            calls=[]; torch=Torch()
            relay=ring.RingExchange(types.SimpleNamespace(rank=0,world=4),lambda *a:calls.append(a),torch)
            s,r=inputs(0,[(8,'torch.float32')]); list(cases(s,r))[index]()
            with self.subTest(index=index), self.assertRaises(ValueError): relay.exchange_all(s,r)
            self.assertEqual(calls, []); self.assertEqual(torch.allocated, [])

    def test_capture_world_and_device_changes_refused(self):
        for world in (1,2,3,5,True,4.0):
            with self.assertRaises(ValueError): ring.RingExchange(types.SimpleNamespace(rank=0,world=world),lambda *a:None,Torch())
        torch=Torch(); calls=[]
        relay=ring.RingExchange(types.SimpleNamespace(rank=0,world=4),lambda *a:calls.append(a),torch)
        torch.capturing=True
        with self.assertRaisesRegex(RuntimeError, 'eager-only'): relay.exchange_all(*inputs(0,[(8,'torch.float32')]))
        self.assertEqual(calls, []); self.assertEqual(torch.allocated, [])
        torch.capturing=False; relay._reserve({'torch.float32':(4,4)}, 'cpu')
        with self.assertRaisesRegex(ValueError, 'change devices'): relay.exchange_all(*inputs(0,[(8,'torch.float32')],device='cuda:0'))

    def test_failed_second_round_still_records_external_cuda_lifetime(self):
        torch=Torch(); calls=[]
        def fail(comm,sends,recvs):
            calls.append((sends,recvs))
            if len(calls)==2: raise RuntimeError('second group failed')
        relay=ring.RingExchange(types.SimpleNamespace(rank=0,world=4),fail,torch)
        with self.assertRaisesRegex(RuntimeError, 'second group failed'):
            relay.exchange_all(*inputs(0,[(3,'torch.float32')],device='cuda:0'))
        self.assertEqual(relay.calls,0)
        self.assertTrue(all(len(t.recorded)==2 for t in relay._buffers['torch.float32']))
        self.assertTrue(all(set(s)=={1,3} for s,r in calls))


if __name__ == '__main__': unittest.main()
