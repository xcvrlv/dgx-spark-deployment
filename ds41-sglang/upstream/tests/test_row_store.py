"""CPU parity, eviction and simultaneous callers on an unaligned fixture."""
import concurrent.futures
import ctypes as C
import json
import os
from pathlib import Path
import random
import tempfile

lib = C.CDLL(str(Path(__file__).resolve().parents[1]/'adapter/librow_store.so'))
U = C.c_uint64
P = C.c_void_p
lib.row_store_open.argtypes = [C.c_char_p, U, U, U, U]
lib.row_store_open.restype = P
lib.row_store_close.argtypes = [P]
lib.row_store_stats.argtypes = [P, C.POINTER(U)]

class Work(C.Structure):
    _fields_ = [('store', P), ('ids', P), ('weights', P), ('scales', P), ('count', U)]

lib.row_store_lookup.argtypes = [C.POINTER(Work)]
lib.row_store_range.argtypes = [P, U, U]
rows = 4099
rng = random.Random(413)
weights, scales = rng.randbytes(rows * 256), rng.randbytes(rows * 8)
offset = 777
with tempfile.NamedTemporaryFile() as f:
    f.write(bytes(offset) + weights + scales)
    f.flush()
    # (budget, owned range): a narrowed range zeroes unowned rows and pins the
    # owned scale shard, so the pinned and unpinned miss paths both get covered.
    cases = [(0, None), (272 * 17, None), (272 * rows, None),
             (272 * rows, (0, rows)), (1088 * 64, (1000, 3000))]
    for budget, owned in cases:
        store = lib.row_store_open(f.name.encode(), rows, offset, offset + len(weights), budget)
        assert store
        lo, hi = owned if owned else (0, rows)
        if owned:
            lib.row_store_range(store, lo, hi)
        def expected(i, table, width):
            if not (lo <= i < hi):
                return bytes(width)
            return table[i*width:(i+1)*width]
        def check(seed):
            r = random.Random(seed)
            ids = [0, rows - 1, 15, 16, 17, 0, 17] + [r.randrange(rows) for _ in range(1000)]
            indices = (C.c_int64 * len(ids))(*ids)
            w, s = C.create_string_buffer(len(ids) * 256), C.create_string_buffer(len(ids) * 8)
            work = Work(store, C.addressof(indices), C.addressof(w), C.addressof(s), len(ids))
            lib.row_store_lookup(C.byref(work))
            assert w.raw == b''.join(expected(i, weights, 256) for i in ids)
            assert s.raw == b''.join(expected(i, scales, 8) for i in ids)
            return sum(1 for i in ids if lo <= i < hi)
        with concurrent.futures.ThreadPoolExecutor(4) as pool:
            checked = sum(pool.map(check, range(12)))
        stats = (U * 9)()
        lib.row_store_stats(store, stats)
        assert stats[0] + stats[1] == checked
        assert stats[3] <= budget
        pinned = os.environ.get('DSV41_RESIDENT_SCALES', '1') not in ('0', 'off', 'false')
        if owned and pinned:
            assert stats[5] == (hi - lo) * 8
        print(json.dumps(dict(budget=budget, owned=owned, checked=checked, hits=stats[0],
                              misses=stats[1], reads=stats[2], slots=stats[4],
                              scale_bytes=stats[5], ways=stats[6], threads=stats[7],
                              passed=True)))
        lib.row_store_close(store)
