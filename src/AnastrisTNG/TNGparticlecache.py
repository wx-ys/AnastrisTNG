"""Persistent particle arrays indexed by their rows in an official snapshot."""

import json
import sqlite3
import zlib
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, contextmanager
from functools import partial

import numpy as np


def missing_rows(start, stop, covered):
    """Yield the portions of [start, stop) outside sorted, disjoint intervals."""
    for first, last in covered:
        if last <= start:
            continue
        if first >= stop:
            break
        if first > start:
            yield start, first
        start = last
    if start < stop:
        yield start, stop


def _copy_byte_lanes(target, source):
    # Each NumPy copy processes one byte position across an entire block.
    # A generic transpose copy walks the short, strided axis one value at a time.
    for destination, lane in zip(target, source):
        destination[:] = lane


class ParticleRowCache:
    """One SQLite file per snapshot, with bounded array blocks and no object keys."""

    block_bytes = 2 * 1024**2
    _where = """
        FROM particle_rows
        WHERE family = :family AND field = :field
          AND start >= COALESCE((
              SELECT MAX(start) FROM particle_rows
              WHERE family = :family AND field = :field AND start <= :first
          ), :first)
          AND start < :last AND (start >= :first OR stop > :first)
        ORDER BY start
    """

    def __init__(self, path, validate=None):
        self.path = path
        self.validate = validate

    @staticmethod
    def _params(key, start, stop):
        return dict(family=key[0], field=key[1], first=start, last=stop)

    @staticmethod
    def _pack(values):
        array = np.ascontiguousarray(values)
        # Group corresponding bytes of adjacent values before lossless compression.
        width = array.dtype.itemsize
        lanes = array.view(np.uint8).reshape(-1, width).T
        shuffled = np.empty(lanes.shape, dtype=np.uint8)
        _copy_byte_lanes(shuffled, lanes)
        # Even a tiny or incompressible block needs the zlib checksum. The
        # existing compressed encoding is readable by earlier cache readers.
        return 1, zlib.compress(shuffled, level=1)

    @staticmethod
    def _unpack(dtype, shape, count, compressed, payload):
        if compressed == 1:
            try:
                payload = zlib.decompress(payload)
            except zlib.error as exc:
                raise ValueError("Invalid compressed particle cache block") from exc
            lanes = np.frombuffer(payload, dtype=np.uint8).reshape(dtype.itemsize, -1)
            raw = np.empty(lanes.T.shape, dtype=np.uint8)
            _copy_byte_lanes(raw.T, lanes)
            array = raw.view(dtype)
        elif compressed == 0:
            # Older caches may hold raw blocks with no on-disk checksum.
            array = np.frombuffer(payload, dtype=dtype).copy()
        else:
            raise ValueError("Unknown particle cache encoding")
        return array.reshape((count, *shape))

    def _decode(self, key, row):
        first, last, dtype, shape, compressed, payload = row
        try:
            dtype = np.dtype(dtype)
            shape = json.loads(shape)
            count = last - first
            if (count <= 0 or dtype.kind not in "biufc" or not isinstance(shape, list)
                    or any(type(size) is not int or size <= 0 for size in shape)
                    or (self.validate and not self.validate(key, (count, *shape), dtype))):
                return None
            return self._unpack(dtype, shape, count, compressed, payload)
        except (ValueError, TypeError, OverflowError):
            return None

    @contextmanager
    def _reader(self):
        if not self.path.is_file():
            yield None
            return
        uri = self.path.resolve().as_uri() + "?mode=ro"
        with closing(sqlite3.connect(uri, uri=True, timeout=30)) as db:
            schema = db.execute("SELECT 1 FROM sqlite_master WHERE name = 'particle_rows'").fetchone()
            yield db if schema else None  # A writer can still be initializing the file.

    def has_fields(self, keys):
        """Can any requested column already have rows, before its offsets are known?"""
        with self._reader() as db:
            return db is not None and any(db.execute(
                "SELECT 1 FROM particle_rows WHERE family=? AND field=? LIMIT 1", key
            ).fetchone() for key in keys)

    def read(self, bounds, *, damaged=None):
        return self.read_intervals({key: [interval] for key, interval in bounds.items()}, damaged=damaged)

    def _decoded_blocks(self, cursor, key, workers):
        decode = partial(self._decode, key)
        # Keep at most four compressed blocks in flight. SQLite stays on
        # this thread; zlib and NumPy can decode independent blocks in parallel.
        while batch := cursor.fetchmany(4):
            large = sum(encoding == 1 and isinstance(payload, bytes) and len(payload) >= 128 * 1024
                        for *_, encoding, payload in batch)
            arrays = workers.map(decode, batch) if large >= 2 else map(decode, batch)
            for row, values in zip(batch, arrays):
                yield row[0], row[1], values

    def read_intervals(self, selections, *, damaged=None):
        result = {key: [] for key in selections}
        columns = "SELECT start, stop, dtype, shape, compressed, payload "
        # ThreadPoolExecutor starts its worker threads only on first submission.
        with self._reader() as db, ThreadPoolExecutor(max_workers=4) as workers:
            if db is None:
                return result
            for key, intervals in selections.items():
                for start, stop in intervals:
                    cursor = db.execute(columns + self._where, self._params(key, start, stop))
                    for first, last, values in self._decoded_blocks(cursor, key, workers):
                        if values is None:
                            if damaged is not None:
                                # A corrupt stop can claim zero rows while the
                                # primary key still occupies its starting row.
                                end = last if isinstance(last, int) and last > first else first + 1
                                damaged.add((key, first, end))
                            continue
                        lo, hi = max(first, start), min(last, stop)
                        result[key].append((lo, values[lo-first:hi-first]))
        return result

    def store(self, blocks, *, damaged=()):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path, timeout=30)) as db, db:
            db.execute("""
                CREATE TABLE IF NOT EXISTS particle_rows (
                    family TEXT NOT NULL,
                    field TEXT NOT NULL,
                    start INTEGER NOT NULL,
                    stop INTEGER NOT NULL,
                    dtype TEXT NOT NULL,
                    shape TEXT NOT NULL,
                    compressed INTEGER NOT NULL,
                    payload BLOB NOT NULL,
                    PRIMARY KEY (family, field, start)
                ) WITHOUT ROWID
            """)
            # Recheck coverage under the writer lock. Overlapping downloads can
            # finish concurrently, but a row is only stored once.
            db.execute("BEGIN IMMEDIATE")
            for key, first, _ in damaged:
                identity = (*key, first)
                where = "FROM particle_rows WHERE family=? AND field=? AND start=?"
                row = db.execute("SELECT start, stop, dtype, shape, compressed, payload " + where,
                                 identity).fetchone()
                # A concurrent repair may already have replaced this block.
                if row is not None and self._decode(key, row) is None:
                    db.execute("DELETE " + where, identity)
            for key, pieces in blocks.items():
                for start, values in pieces:
                    if self.validate and not self.validate(key, values.shape, values.dtype):
                        raise ValueError(f"Invalid particle data for {key[0]}.{key[1]}")
                    if not len(values):
                        continue
                    stop = start + len(values)
                    step = max(1, self.block_bytes // max(1, values.nbytes // len(values)))
                    dtype, shape = values.dtype.str, json.dumps(values.shape[1:])
                    covered = db.execute("SELECT start, stop " + self._where, self._params(key, start, stop))
                    for first, last in missing_rows(start, stop, covered.fetchall()):
                        for lo in range(first, last, step):
                            hi = min(lo + step, last)
                            compressed, payload = self._pack(values[lo-start:hi-start])
                            db.execute("INSERT INTO particle_rows VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                                       (*key, lo, hi, dtype, shape, compressed, payload))
