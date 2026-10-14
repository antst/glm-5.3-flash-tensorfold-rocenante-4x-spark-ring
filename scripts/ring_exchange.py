"""Four-rank eager all-to-all via adjacent NCCL peers only.

Rank r has neighbors L=(r-1)%4, R=(r+1)%4 and opposite O=(r+2)%4.
Round 1 sends the neighbors their direct messages and routes O's prefix via R,
suffix via L. Round 2 forwards the received prefix from L to R and suffix from
R to L. Receives retain their ORIGINAL source-rank slots. No reduction, dtype
conversion, diagonal NCCL call, or topology/driver modification is performed.

The caller holds its Controller lock, performs its ordering wait BEFORE this
call, and records its event AFTER it. This serializes scratch reuse across
streams and native gathers/graph replays. Captured invocation is refused.
The saved, unbound original_exchange_all(comm, sends, recvs) must be passed;
passing the intercepted wrapper would recurse.

Scratch: at most two retained tensors per supported dtype, each sized to the
largest aggregate relay requirement observed for that dtype; no shape cache.
For even counts the two together equal one peer's payload. Odd counts use
ceil(n/2)-element relay messages: the original NCCL wrapper requires each local
send and receive to have equal counts. Padding lives inside the same two
scratch tensors and is never exposed as payload. TX is 4*B for even counts,
4*B+2*element_size per odd tensor otherwise (B = one peer's payload).

Growth synchronizes the already-ordered current stream before allocating a
replacement pair, so prior scratch users have finished. Old and replacement
pairs overlap during allocation; report() accounts that requested live-storage
peak. CUDA allocator reserves/caches and NCCL workspace are separate, NOT
bounded by this storage report. Prefer pre-sizing with representative maximum
geometry during the owned warmup. An optional max_scratch_bytes caps retained
storage, and max_growth_bytes caps the old+replacement allocation peak.
"""
from __future__ import annotations

SUPPORTED_DTYPES = frozenset(('torch.float32', 'torch.bfloat16', 'torch.int32', 'torch.int64'))


class RingExchange:
    def __init__(self, comm, original_exchange_all, torch, *, max_scratch_bytes=None, max_growth_bytes=None):
        if type(comm.world) is not int or comm.world != 4 or type(comm.rank) is not int or comm.rank not in range(4):
            raise ValueError('ring relay requires exactly four ranks')
        if not callable(original_exchange_all):
            raise TypeError('saved original_exchange_all must be callable')
        for value in (max_scratch_bytes, max_growth_bytes):
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError('scratch limits must be nonnegative integer bytes or None')
        self.comm, self.original, self.torch = comm, original_exchange_all, torch
        self.rank = comm.rank
        self.left, self.right, self.opposite = (self.rank - 1) % 4, (self.rank + 1) % 4, (self.rank + 2) % 4
        self.max_scratch_bytes, self.max_growth_bytes = max_scratch_bytes, max_growth_bytes
        self._buffers = {}       # dtype -> (left/prefix storage, right/suffix storage)
        self._sizes = {}         # dtype -> (elements per tensor, bytes per element)
        self._device = None
        self.calls = self.rounds = self.tx_bytes = self.padding_tx_bytes = 0
        self.growths = self.growth_syncs = self.peak_growth_bytes = 0

    def _validate(self, sends, recvs):
        peers = set(range(4)) - {self.rank}
        if (not isinstance(sends, dict) or not isinstance(recvs, dict)
                or any(type(p) is not int for p in (*sends, *recvs))
                or set(sends) != peers or set(recvs) != peers):
            raise ValueError('ring relay requires all three other peer keys in sends and recvs')
        if any(not isinstance(v, (list, tuple)) for v in (*sends.values(), *recvs.values())):
            raise ValueError('each peer needs a tensor list')
        count = len(sends[self.left])
        if any(len(v) != count for v in (*sends.values(), *recvs.values())):
            raise ValueError('ring relay requires identical tensor-list lengths')
        layout, required, device = [], {}, None
        send_ranges, recv_ranges = [], []
        for i in range(count):
            first = sends[self.left][i]
            dtype, n, width, shape = first.dtype, first.numel(), first.element_size(), tuple(first.shape)
            if str(dtype) not in SUPPORTED_DTYPES or type(n) is not int or n < 0:
                raise ValueError('unsupported relay dtype/count')
            if device is None:
                device = first.device
            for bank, ranges in ((sends, send_ranges), (recvs, recv_ranges)):
                for peer in sorted(peers):
                    t = bank[peer][i]
                    if (t.dtype != dtype or t.numel() != n or tuple(t.shape) != shape
                            or t.element_size() != width or not t.is_contiguous() or t.device != device):
                        raise ValueError('ring relay requires uniform peer shapes/counts/dtypes, contiguous on one device')
                    if n:
                        ranges.append((t.data_ptr(), t.data_ptr() + n * width))
            half = (n + 1) // 2
            offset = required.get(dtype, (0, width))[0]
            # An odd message needs a second half-sized slice for outgoing suffix
            # padding (left buffer) / the incoming opposite suffix (right buffer).
            span = half * (2 if n % 2 else 1)
            required[dtype] = (offset + span, width)
            layout.append((i, dtype, n, half, offset))
        if any(a < d and c < b for a, b in recv_ranges for c, d in send_ranges):
            raise ValueError('relay send/receive storage must not overlap')
        if any(a < d and c < b for j, (a, b) in enumerate(recv_ranges) for c, d in recv_ranges[j + 1:]):
            raise ValueError('relay receive slots must not overlap')
        if self._device is not None and device is not None and self._device != device:
            raise ValueError('one relay instance cannot change devices')
        return layout, required, device

    def _reserve(self, required, device):
        growth = {dtype: spec for dtype, spec in required.items()
                  if spec[0] > self._sizes.get(dtype, (0, spec[1]))[0]}
        retained = sum(2 * n * width for n, width in self._sizes.values())
        replaced = sum(2 * self._sizes.get(d, (0, width))[0] * width for d, (_, width) in growth.items())
        additional = sum(2 * n * width for n, width in growth.values())
        final, peak = retained - replaced + additional, retained + additional
        if self.max_scratch_bytes is not None and final > self.max_scratch_bytes:
            raise ValueError('relay retained scratch exceeds bound')
        if self.max_growth_bytes is not None and peak > self.max_growth_bytes:
            raise ValueError('relay old-plus-new scratch exceeds growth bound')
        if not growth:
            return
        if self._buffers:
            # Controller._order has already made this stream wait on all earlier
            # users, even when the previous call was on another CUDA stream.
            self.torch.cuda.current_stream().synchronize()
            self.growth_syncs += 1
        self.peak_growth_bytes = max(self.peak_growth_bytes, peak)
        replacements = {dtype: (self.torch.empty(n, dtype=dtype, device=device),
                                self.torch.empty(n, dtype=dtype, device=device))
                        for dtype, (n, _) in growth.items()}
        self._buffers.update(replacements)
        self._sizes.update(growth)
        self._device = device
        self.growths += 1

    def exchange_all(self, sends, recvs):
        """Validate everything before allocation/submission; enqueue two rounds.

        Every rank must call with matching geometry and order. This local check
        cannot prove remote agreement; that remains the enclosing TP contract.
        """
        if self.torch.cuda.is_current_stream_capturing():
            raise RuntimeError('ring relay is eager-only')
        if self.comm.rank != self.rank or self.comm.world != 4:
            raise RuntimeError('relay communicator identity changed')
        layout, required, device = self._validate(sends, recvs)
        self._reserve(required, device)
        if not any(n for _, _, n, _, _ in layout):
            return
        current = self.torch.cuda.current_stream()
        touched = [t for dtype, (n, _) in required.items() if n for t in self._buffers[dtype]]
        # CUDA's allocator cannot see external NCCL accesses. Controller events
        # order work but do not replace allocator stream-lifetime bookkeeping.
        for t in touched:
            if t.is_cuda:
                t.record_stream(current)
        neighbors = sorted((self.left, self.right))
        round1_s = {p: list(sends[p]) for p in neighbors}
        round1_r = {p: list(recvs[p]) for p in neighbors}
        round2_s = {p: [] for p in neighbors}
        round2_r = {p: [] for p in neighbors}
        copies = []
        payload = padding = 0
        for i, dtype, n, half, offset in layout:
            # Zero-sized positions still occupy their matching list position.
            if n == 0:
                continue
            left, right = self._buffers[dtype]
            prefix = left[offset:offset + half]
            suffix = right[offset:offset + half]
            outgoing = sends[self.opposite][i].reshape(-1)
            incoming = recvs[self.opposite][i].reshape(-1)
            send_suffix, recv_suffix = outgoing[half:], incoming[half:]
            if n % 2:
                send_suffix = left[offset + half:offset + 2 * half]
                recv_suffix = right[offset + half:offset + 2 * half]
                send_suffix.zero_()
                send_suffix[:n - half].copy_(outgoing[half:])
                copies.append((incoming[half:], recv_suffix[:n - half]))
                padding += 2 * required[dtype][1]
            round1_s[self.left].append(send_suffix)
            round1_s[self.right].append(outgoing[:half])
            round1_r[self.left].append(prefix)
            round1_r[self.right].append(suffix)
            round2_s[self.left].append(suffix)
            round2_s[self.right].append(prefix)
            round2_r[self.left].append(incoming[:half])
            round2_r[self.right].append(recv_suffix)
            payload += n * required[dtype][1]
        try:
            self.original(self.comm, round1_s, round1_r)
            self.original(self.comm, round2_s, round2_r)
            for dst, src in copies:
                dst.copy_(src)
        finally:
            for t in touched:
                if t.is_cuda:
                    t.record_stream(current)
        self.calls += 1
        self.rounds += 2
        self.tx_bytes += 4 * payload + padding
        self.padding_tx_bytes += padding

    def report(self):
        return {'schema': 'tensorfold-adjacent-ring-exchange/v1', 'rank': self.rank, 'world': 4,
                'neighbors': sorted((self.left, self.right)), 'calls': self.calls, 'rounds': self.rounds,
                'tx_bytes': self.tx_bytes, 'padding_tx_bytes': self.padding_tx_bytes,
                'scratch_bytes': sum(2 * n * w for n, w in self._sizes.values()),
                'scratch_by_dtype': {str(d): {'buffers': 2, 'elements_each': n, 'bytes': 2 * n * w}
                                     for d, (n, w) in self._sizes.items()},
                'peak_growth_bytes': self.peak_growth_bytes, 'growths': self.growths,
                'growth_syncs': self.growth_syncs, 'max_scratch_bytes': self.max_scratch_bytes,
                'max_growth_bytes': self.max_growth_bytes,
                'accounting': 'Requested live tensor storage; excludes allocator reserves and NCCL workspace'}
