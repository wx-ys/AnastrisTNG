"""On-demand Illustris and TNG data backed by a shared snapshot cache."""
import errno
import json
import os
import re
import sqlite3
import sys
import tempfile
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, closing, contextmanager, nullcontext
from email.utils import parsedate_to_datetime
from http.client import HTTPException, IncompleteRead
from io import BytesIO, RawIOBase
from pathlib import Path
from socket import EAI_AGAIN
from threading import Event, RLock, Thread
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urljoin, urlsplit
from urllib.request import Request, urlopen

import h5py
import numpy as np
from filelock import FileLock
from dotenv import dotenv_values
from tqdm import tqdm

_IN_NOTEBOOK = "ipykernel.zmqshell" in sys.modules
if _IN_NOTEBOOK:
    from tqdm.notebook import IProgress
    if IProgress is not None:
        from tqdm.auto import tqdm

from .illustris_python.groupcat import gcPath, offsetPath, loadSingle
from .illustris_python.sublink import loadTree, treePath
from .illustris_python.snapshot import getSnapOffsets, snapPath
from pynbody.array import SimArray
from pynbody.family import get_family

from .TNGunits import groupcat_units, parameter_all_Description, snapshot_pa_name, snapshot_units
from .TNGgroupcat import local_snapshot_header
from .TNGload import concatenate_particle_arrays
from .TNGparticlecache import ParticleRowCache, missing_rows

PART_TYPE = {"gas": 0, "dm": 1, "star": 4, "bh": 5}
ZOOM_PART_TYPE = {**PART_TYPE, "dmlowres": 2, "tracer": 3}
API_TYPE = {"gas": "gas", "dm": "dm", "star": "stars", "bh": "bhs"}
CATALOG = {"Halo": "Group", "Subhalo": "Subhalo"}
_CATALOG_RANGE_WORKERS = 3
_CATALOG_RANGE_LIMIT = 1024**2
_CATALOG_COUNTS_AND_INDICES = {"GroupFirstSub", "GroupLen", "GroupLenType", "GroupNsubs",
                               "SubhaloGrNr", "SubhaloLen", "SubhaloLenType", "SubhaloParent"}
_CLUSTER_CATALOG_DTYPES = {
    "GroupContaminationFracByMass": "f4", "GroupContaminationFracByNumPart": "f4",
    "GroupOrigHaloID": "i4", "GroupPrimaryZoomTarget": "i4", "SubhaloOrigHaloID": "i4",
    "TracerLengthType": "i4", "SubhaloLengthType": "i4",
    "GroupOffsetType": "i8", "SubhaloOffsetType": "i8", "TracerOffsetType": "i8",
}
_PARTICLE_COMPONENTS = {
    "Coordinates": 3, "Velocities": 3, "BirthPos": 3, "BirthVel": 3, "CenterOfMass": 3,
    "MagneticField": 3, "GFM_MetalsTagged": 6, "GFM_StellarPhotometrics": 8, "BH_WindTimes": 5,
}
_PARTICLE_INTEGERS = {"ParticleIDs", "BH_Progs", "NumTracers", "AllowRefinement", "BH_WindCount"}
_OFFSET_RANGE = "Offsets via Range"
FULL_SNAPS = {2, 3, 4, 6, 8, 11, 13, 17, 21, 25, 33, 40, 50, 59, 67, 72, 78, 84, 91, 99}

# Published full/mini field lists
# https://www.tng-project.org/data/docs/specifications/#sec1
COMMON = {"Coordinates", "ParticleIDs", "Velocities"}
SUBFIND = {"SubfindDMDensity", "SubfindDensity", "SubfindHsml", "SubfindVelDisp"}
MINI = {
    "gas": COMMON | set("Density ElectronAbundance GFM_Metallicity InternalEnergy InternalEnergyOld Masses StarFormationRate".split()),
    "dm": COMMON,
    "star": COMMON | set("GFM_InitialMass GFM_Metallicity GFM_StellarFormationTime Masses StellarHsml".split()),
    "bh": COMMON | set("""Masses Potential BH_BPressure BH_CumEgyInjection_QM BH_CumEgyInjection_RM
        BH_CumMassGrowth_QM BH_CumMassGrowth_RM BH_Density BH_HostHaloMass BH_Hsml BH_Mass
        BH_Mdot BH_MdotBondi BH_MdotEddington BH_Pressure BH_Progs BH_U""".split()),
}
FULL_EXTRA = {
    "gas": set("""CenterOfMass EnergyDissipation GFM_AGNRadiation GFM_CoolingRate
        GFM_Metals GFM_MetalsTagged GFM_WindDMVelDisp GFM_WindHostHaloMass Machnumber MagneticField
        MagneticFieldDivergence NeutralHydrogenAbundance Potential""".split()),
    "dm": {"Potential"},
    "star": set("BirthPos BirthVel GFM_Metals GFM_MetalsTagged GFM_StellarPhotometrics Potential".split()),
    "bh": set(),
}

# Original Illustris snapshots use a single schema rather than TNG's full/mini split.
ILLUSTRIS_COMMON = COMMON | {"Potential", "SubfindDensity", "SubfindHsml", "SubfindVelDisp"}
ILLUSTRIS_FIELDS = {
    "gas": ILLUSTRIS_COMMON | set("""Density ElectronAbundance GFM_AGNRadiation GFM_CoolingRate GFM_Metallicity
        GFM_Metals GFM_WindDMVelDisp InternalEnergy Masses NeutralHydrogenAbundance NumTracers
        SmoothingLength StarFormationRate Volume""".split()),
    "dm": ILLUSTRIS_COMMON,
    "star": ILLUSTRIS_COMMON | set("""GFM_InitialMass GFM_Metallicity GFM_Metals GFM_StellarFormationTime
        GFM_StellarPhotometrics Masses NumTracers StellarHsml""".split()),
    "bh": ILLUSTRIS_COMMON | set("""BH_CumEgyInjection_QM BH_CumMassGrowth_QM BH_Density BH_Hsml BH_Mass
        BH_Mass_bubbles BH_Mass_ini BH_Mdot BH_Pressure BH_Progs BH_U HostHaloMass Masses NumTracers""".split()),
}


class RemoteDataError(RuntimeError):
    """A requested TNG resource could not be retrieved or validated."""


class _RangeCacheMiss(RemoteDataError):
    """A cache-only lookup did not cover the requested raw bytes."""


class _IncompleteTransfer(HTTPException):
    """The response body ended before the declared byte count."""


def _set_byte_total(bar, total):
    bar.total = total
    if _IN_NOTEBOOK and hasattr(bar, "container"):
        # tqdm's notebook widget keeps its original maximum when .total
        # changes. Hide that gauge while only a byte count and rate are known.
        indicator = bar.container.children[1]
        indicator.layout.display = "none" if total is None else None
        if total is not None:
            indicator.max = max(1, total)
            indicator.layout.width = None
            indicator.bar_style = ""


class _ParticleProgress:
    """Track all wire bytes for one particle selection."""

    def __init__(self, source, stack, description):
        self.source, self.description = source, description
        self.bar = None
        self.transports = set()
        self.expected = self.received = 0
        self.unknown_length = False
        self.pending = 0
        self.stage = ""
        self._lock = RLock()
        self._bars = ExitStack()
        self._monitor = None
        self._closed = Event()
        stack.push(self._close)

    def start(self):
        with self._lock:
            if self._closed.is_set() or self.bar is not None or self.source.progress is False:
                return
            self.bar = self._bars.enter_context(
                self.source._progress(self.description, total=None, unit="B", unit_scale=True))
            if self.stage:
                self.bar.set_postfix_str(self.stage)

    def phase(self, stage):
        """Show slow cache work as well as transfers, keeping quick cache hits quiet."""
        if self.source.progress is False:
            return
        with self._lock:
            if self._closed.is_set() or self.stage == stage:
                return
            self.stage = stage
            if self.bar is not None:
                # The previous transfer may be at 100% while validation or a
                # database write still has to finish. reserve() sets the next
                # known transfer size, complete() marks the whole operation.
                _set_byte_total(self.bar, None)
                self.bar.set_postfix_str(stage)
            if self._monitor is None and (self.source.progress is True or _IN_NOTEBOOK or sys.stderr.isatty()):
                self._monitor = Thread(target=self._refresh_slow, daemon=True)
                self._monitor.start()

    def _refresh_slow(self):
        if self._closed.wait(0.5):
            return
        self.start()
        while not self._closed.wait(1.0):
            with self._lock:
                if self.bar is not None and not self._closed.is_set():
                    self.bar.refresh()

    @contextmanager
    def waiting(self, resource):
        with self._lock:
            self.start()
            self.pending += 1
            if self.bar is not None:
                _set_byte_total(self.bar, None)
            self.phase(f"Waiting for {resource}")
        try:
            yield
        finally:
            with self._lock:
                self.pending -= 1

    def reserve(self, size):
        if self.source.progress is False:
            return
        with self._lock:
            if size is None:
                self.unknown_length = True
            else:
                self.expected += size
            if self.bar is not None:
                _set_byte_total(self.bar, None if self.unknown_length or self.pending else self.expected)
                self.bar.refresh()

    def update(self, size):
        with self._lock:
            if self.bar is not None:
                self.received += size
                self.bar.update(size)

    def mark_transport(self, transport):
        with self._lock:
            self.start()
            if self.bar is None or transport in self.transports:
                return
            self.transports.add(transport)
            label = "Cutout | offsets via Range" if len(self.transports) > 1 else transport
            self.bar.set_description_str(f"[{label}] {self.description}")

    def complete(self):
        with self._lock:
            if self.bar is not None:
                _set_byte_total(self.bar, None if self.unknown_length or not self.received else self.received)
                self.bar.set_postfix_str("Complete")

    def _close(self, exc_type, exc, traceback):
        with self._lock:
            self._closed.set()
            if exc is not None and self.bar is not None:
                self.bar.set_postfix_str("Interrupted" if exc_type is KeyboardInterrupt else "Failed")
            self._bars.__exit__(exc_type, exc, traceback)
            if exc is not None and _IN_NOTEBOOK and hasattr(self.bar, "container"):
                self.bar.container.children[1].bar_style = "danger"


class _RangeFile(RawIOBase):
    """Seekable, read-only view of an official static HDF5 file for h5py."""

    block_size = 64 * 1024

    def __init__(self, source, endpoint, bar, cache=None, lock=None, *, cache_only=False):
        self.source, self.endpoint, self.bar = source, endpoint, bar
        self.cache, self.lock = cache, lock
        self.cache_only = cache_only
        self.position = 0
        self.size = None
        self.validator = None
        self.blocks = OrderedDict()
        self.target = None
        if cache is None:
            self.blocks[0] = self.fetch(0, self.block_size)
        elif not self._load_file_info():
            self.blocks[0] = self._read_cached(0, self.block_size)

    def _load_file_info(self):
        info = self.cache.execute("SELECT size, validator FROM file_info WHERE id=1").fetchone()
        if info is not None:
            self.size, self.validator = info
        return info is not None

    def _read_cached(self, start, stop):
        def lookup():
            row = self.cache.execute("SELECT data FROM blocks WHERE start=?", (start,)).fetchone()
            if row is not None:
                if self.size is None:
                    self._load_file_info()
                if self.size is None or len(row[0]) != min(self.block_size, self.size - start):
                    raise RemoteDataError(f"Invalid cached block for {self.endpoint} at {start}")
                return row[0]
        data = lookup()
        if data is not None:
            return data
        self._check_available(start)
        with self.lock:
            data = lookup()
            if data is not None:
                return data
            self._load_file_info()
            # Batch adjacent misses, stopping before a block already on disk.
            stop = ((stop + self.block_size - 1) // self.block_size) * self.block_size
            if self.size is not None:
                stop = min(stop, self.size)
            next_hit = self.cache.execute(
                "SELECT min(start) FROM blocks WHERE start>? AND start<?", (start, stop)
            ).fetchone()[0]
            end = stop if next_hit is None else next_hit
            data = self.fetch(start, end)
            with self.cache:
                self._store_blocks(start, data)
            return data

    def _store_blocks(self, start, data):
        self.cache.execute("INSERT OR REPLACE INTO file_info VALUES (1, ?, ?)", (self.size, self.validator))
        self.cache.executemany("INSERT INTO blocks VALUES (?, ?)",
                               ((start + i, memoryview(data)[i:i+self.block_size])
                                for i in range(0, len(data), self.block_size)))

    def _check_available(self, start):
        if self.cache_only:
            raise _RangeCacheMiss(f"Uncached block for {self.endpoint} at {start}")
        self.source._check_online(f"{self.endpoint} bytes starting at {start}")

    def cached_row_slices(self, byte_offset, row_size, first, rows):
        byte_start = byte_offset + rows.start * row_size
        byte_stop = byte_offset + rows.stop * row_size
        first_block = byte_start // self.block_size * self.block_size
        blocks = self.cache.execute("SELECT start FROM blocks WHERE start>=? AND start<? ORDER BY start",
                                    (first_block, byte_stop))
        coverage = ((start, start + self.block_size) for (start,) in blocks)
        # Exclude any row touching a missing byte block, even if only part of
        # the row lies in that block.
        gaps = [((lo - byte_offset) // row_size, (hi - byte_offset + row_size - 1) // row_size)
                for lo, hi in missing_rows(byte_start, byte_stop, coverage)]
        return [(first + lo - rows.start, slice(lo, hi))
                for lo, hi in missing_rows(rows.start, rows.stop, gaps)]

    def fetch(self, start, stop):
        return self._accept_range(self._request_range(start, stop, self.bar))

    def _request_range(self, start, stop, bar):
        progress = bar if isinstance(bar, _ParticleProgress) else None
        if progress is not None:
            progress.mark_transport(_OFFSET_RANGE)
        headers = {"Range": f"bytes={start}-{stop - 1}"}
        if self.validator is not None:
            headers["If-Range"] = self.validator

        def read(response):
            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
            if response.status != 206 or not match:
                raise RemoteDataError(f"Server did not honor HTTP Range for {self.endpoint}")
            first, last, size = map(int, match.groups())
            if first != start or last + 1 != min(stop, size) or self.size not in (None, size):
                raise RemoteDataError(f"Unexpected byte range for {self.endpoint}")
            validator = response.headers.get("ETag")
            if not validator or validator.startswith("W/"):
                validator = response.headers.get("Last-Modified")
            if self.validator is not None and validator is not None and validator != self.validator:
                raise RemoteDataError(f"Source file changed while reading {self.endpoint}")
            if progress is not None:
                progress.phase("Reading offsets")
            data = bytearray()
            _copy_response(response, data.extend, bar, last + 1 - first,
                           f"Incomplete byte range for {self.endpoint}")
            location = response.geturl()
            return data, size, validator, location

        return self.source._open(self.endpoint, self.endpoint, headers=headers, read=read,
                                 target=self.target, progress=progress)

    def _accept_range(self, response):
        data, size, validator, location = response
        # Concurrent catalogue requests can each be the first to observe a
        # validator. Check again before their blocks enter the shared cache.
        if self.size not in (None, size) or self.validator and validator and self.validator != validator:
            raise RemoteDataError(f"Source file changed while reading {self.endpoint}")
        self.size = size
        if self.cache is not None:
            self.validator = self.validator or validator
        self.target = location if location != urljoin(self.source.url, self.endpoint) else None
        return data

    def _prefetch_catalog_blocks(self, blocks, pool):
        """Combine adjacent cache misses and fetch up to three disjoint spans at once."""
        if len(blocks) < 2:
            return
        with self.lock:
            self._load_file_info()
            cached = self.cache.execute("SELECT start FROM blocks WHERE start BETWEEN ? AND ?",
                                        (min(blocks), max(blocks)))
            blocks.difference_update(start for (start,) in cached)
            if len(blocks) < 2:
                return
            spans = []
            for start in sorted(blocks):
                stop = min(start + self.block_size, self.size)
                if spans and start == spans[-1][1] and stop - spans[-1][0] <= _CATALOG_RANGE_LIMIT:
                    spans[-1] = (spans[-1][0], stop)
                else:
                    spans.append((start, stop))
            for first in range(0, len(spans), _CATALOG_RANGE_WORKERS):
                batch = spans[first:first + _CATALOG_RANGE_WORKERS]
                futures = [pool.submit(self._request_range, start, stop, None) for start, stop in batch]
                responses = [future.result() for future in futures]
                # HDF5 and SQLite stay in the caller. Publish each batch only
                # after all responses have arrived and passed the validator checks.
                with self.cache:
                    for (start, _), response in zip(batch, responses):
                        data = self._accept_range(response)
                        self._store_blocks(start, data)
                        self.bar.update(len(data))

    def catalog_row_datasets(self, group, row):
        """Prime only the HDF5 blocks needed for one row of contiguous catalogue columns."""
        if self.cache_only or self.source.offline or self.size <= self.block_size or len(group) < 4:
            return group.items()
        block = self.block_size
        with ThreadPoolExecutor(max_workers=_CATALOG_RANGE_WORKERS) as pool:
            if group.file.userblock_size == 0:
                # Hard links in a native HDF5 file contain the object header address.
                headers = set()
                for name in group:
                    encoded = name.encode("utf8") if isinstance(name, str) else name
                    link = group.id.links.get_info(encoded)
                    if link.type == h5py.h5l.TYPE_HARD and 0 <= link.u < self.size:
                        headers.add(link.u // block * block)
                self._prefetch_catalog_blocks(headers, pool)
            datasets = list(group.items())
            blocks = set()
            file_id = group.file.id
            for _, dataset in datasets:
                if (not isinstance(dataset, h5py.Dataset) or not dataset.shape or dataset.dtype.hasobject
                        or dataset.file.id != file_id):
                    continue
                offset = dataset.id.get_offset()
                if offset is None or row >= len(dataset):
                    continue
                row_size = dataset.dtype.itemsize * int(np.prod(dataset.shape[1:]))
                start = offset + row * row_size
                if 0 < row_size <= _CATALOG_RANGE_LIMIT and 0 <= start < start + row_size <= self.size:
                    blocks.update(range(start // block * block, start + row_size, block))
            self._prefetch_catalog_blocks(blocks, pool)
        return datasets

    def seek(self, offset, whence=0):
        position = (0, self.position, self.size)[whence] + offset
        if position < 0:
            raise ValueError("Cannot seek before start of file")
        self.position = position
        return self.position

    def tell(self):
        return self.position

    def readinto(self, buffer):
        size = max(0, min(len(buffer), self.size - self.position))
        view = memoryview(buffer).cast("B")
        done = 0
        while done < size:
            start = self.position // self.block_size * self.block_size
            if start not in self.blocks:
                stop = min(self.size, start + 8 * 1024**2,
                           max(start + self.block_size, self.position + size - done))
                data = self.fetch(start, stop) if self.cache is None else self._read_cached(start, stop)
                if len(data) <= self.block_size:
                    self.blocks[start] = data
                    if len(self.blocks) > 64:
                        self.blocks.popitem(last=False)
            else:
                data = self.blocks[start]
            offset = self.position - start
            n = min(size - done, len(data) - offset)
            view[done:done+n] = memoryview(data)[offset:offset+n]
            self.position += n
            done += n
        return done


class _LocalTreeCacheMiss(RemoteDataError):
    pass


class _LocalTreeRangeFile(_RangeFile):
    """Copy requested SubLink blocks from a local mirror into a separate cache."""

    def __init__(self, source, filename, db, lock, stream):
        self.stream = stream
        if stream is not None:
            self.disk_size = os.fstat(stream.fileno()).st_size
        super().__init__(source, filename, None, db, lock)

    def _check_available(self, start):
        if self.stream is None:
            raise _LocalTreeCacheMiss(f"Uncached local SubLink block for {self.endpoint} at {start}")

    def fetch(self, start, stop):
        self.stream.seek(start)
        size = min(stop, self.disk_size) - start
        data = self.stream.read(size)
        if len(data) != size:
            raise RemoteDataError(f"Incomplete local SubLink block for {self.endpoint}")
        self.size = self.disk_size
        return data


def _download_forbidden(error):
    """Distinguish a data-server refusal after redirect from an API authorization error."""
    if not isinstance(error, HTTPError) or error.code != 403:
        return False
    target = urlsplit(error.geturl())
    host = target.hostname or ""
    return (target.scheme == "https" and host.endswith((".tng-project.org", ".illustris-project.org"))
            and (host.startswith("data-") or host.startswith("data.")))


def _transient_get_error(error):
    if isinstance(error, HTTPError):
        return error.code in (408, 429) or 500 <= error.code < 600
    if isinstance(error, URLError):
        error = error.reason
    return (isinstance(error, (TimeoutError, ConnectionError, IncompleteRead, _IncompleteTransfer))
            or isinstance(error, OSError) and error.errno in (EAI_AGAIN, errno.EHOSTUNREACH, errno.ENETUNREACH))


def _catalog_fallback_error(error):
    if isinstance(error, HTTPError) and error.code == 429:
        return False
    return _transient_get_error(error) or _download_forbidden(error)


def _retry_delay(error, attempt):
    header = error.headers.get("Retry-After") if isinstance(error, HTTPError) and error.headers else None
    if header:
        try:
            return max(0, int(header))
        except ValueError:
            try:
                return max(0, parsedate_to_datetime(header).timestamp() - time.time())
            except (TypeError, ValueError, OverflowError):
                pass
    return 0.25 * 2 ** attempt


def _response_length(response, resource):
    header = response.headers.get("Content-Length")
    if header is None:
        return None
    if not re.fullmatch(r"[0-9]+", str(header).strip()):
        raise RemoteDataError(f"Invalid download length for {resource}")
    return int(header)


def _copy_response(response, write, bar, total, incomplete_message):
    progress = bar if isinstance(bar, _ParticleProgress) else None
    if progress is not None:
        progress.reserve(total)
    written = 0
    try:
        # HTTPResponse.read1 returns available bytes without waiting for an entire block.
        read = getattr(response, "read1", response.read)
        while block := read(1024 * 1024):
            write(block)
            written += len(block)
            if bar is not None:
                bar.update(len(block))
        if total is not None and written != total:
            raise _IncompleteTransfer(f"{incomplete_message} ({written} of {total} bytes)")
    finally:
        if progress is not None and total is not None and written != total:
            progress.reserve(written - total)


@contextmanager
def temporary_path(destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=destination.parent, suffix=".part")
    os.close(fd)
    path = Path(name)
    try:
        yield path
    finally:
        path.unlink(missing_ok=True)


def write_json(path, value):
    with temporary_path(path) as tmp:
        tmp.write_text(json.dumps(value), encoding="utf8")
        os.replace(tmp, path)


def align_rows(ids, wanted):
    """Return exact integer row indices, allowing a subset in arbitrary order."""
    # A mixed int64/uint64 search can promote to float64 and lose low ID bits.
    ids = np.asarray(ids, dtype=np.uint64)
    wanted = np.asarray(wanted, dtype=np.uint64)
    if np.array_equal(ids, wanted):
        return slice(None)
    order = np.argsort(ids)
    rows = np.searchsorted(ids, wanted, sorter=order)
    if np.any(rows == len(ids)) or not np.array_equal(ids[order[rows]], wanted):
        raise RemoteDataError("ParticleIDs do not match the selection")
    return order[rows]


def _json_catalog_array(run, name, value, known_field):
    """Recover the public HDF5 dtype erased by info.json and older JSON row caches."""
    try:
        array = np.array(value)
    except ValueError:
        return np.array(value, dtype=object)  # Preserve ragged supplementary data.
    if isinstance(value, list) and array.dtype.kind == "f" and np.any(array >= 2**63):
        # NumPy can infer float64 for a JSON list mixing small and large integers.
        exact = np.array(value, dtype=object)
        if all(isinstance(x, int) for x in exact.flat):
            array = exact if exact.min() < 0 or exact.max() >= 2**64 else exact.astype("u8")
    if not known_field or array.dtype.kind not in "biuf":
        return array
    original = run.startswith("Illustris-")
    if name in _CATALOG_COUNTS_AND_INDICES:
        dtype = "u4" if original else "i4"
        if name == "GroupFirstSub" and isinstance(value, int) and value in (-1, 2**32 - 1, 2**64 - 1):
            return np.array(2**32 - 1 if original else -1, dtype=dtype)
    elif name in _CLUSTER_CATALOG_DTYPES:
        if run != "TNG-Cluster":
            return array
        dtype = _CLUSTER_CATALOG_DTYPES[name]
    elif name == "SubhaloIDMostbound":
        dtype = "u8"
    elif name == "SubhaloFlag":
        dtype = "?"
    elif original and name.startswith("SubhaloGasMetallicity"):
        dtype = "f8"
    else:
        dtype = "f4"
    dtype = np.dtype(dtype)
    if dtype.kind in "biu":
        low, high = (0, 1) if dtype.kind == "b" else (np.iinfo(dtype).min, np.iinfo(dtype).max)
        if array.dtype.kind not in "biu" or (array.size and (int(array.min()) < low or int(array.max()) > high)):
            return array  # Keep unknown sentinels or wider data intact.
    with np.errstate(over="ignore"):
        native = array.astype(dtype, copy=False)
    if dtype.kind == "f" and np.any(np.isfinite(array) & ~np.isfinite(native)):
        return array
    return native


class ApiSource:
    def __init__(self, run, snap, cache_dir, local_dir, api_key, progress, offline):
        self.original_illustris = re.fullmatch(r"Illustris-[1-3]", run) is not None
        if not self.original_illustris and not re.fullmatch(r"TNG(?:50-[1-4]|100-[1-3]|300-[1-3]|-Cluster)", run):
            raise ValueError("Expected Illustris-1 through Illustris-3 or a supported full-physics TNG run")
        last = 135 if self.original_illustris else 99
        if not isinstance(snap, (int, np.integer)) or not 0 <= snap <= last:
            raise ValueError(f"Expected a {run} snapshot number from 0 to {last}")
        if progress not in ("auto", True, False):
            raise ValueError("progress must be 'auto', True or False")
        self.run, self.snap = run, int(snap)
        self.url = f"https://www.tng-project.org/api/{run}/snapshots/{snap}/"
        self.root = Path(cache_dir).expanduser() / run / f"snap_{snap:03d}"
        self.local = os.fspath(Path(local_dir).expanduser()) if local_dir is not None else None
        self._local_header = local_snapshot_header(self.local, self.snap, expected_run=run) if self.local else {}
        cached_header = self.root / "header.json"
        if self._local_header and cached_header.is_file():
            self._check_mirror_counts(json.loads(cached_header.read_text(encoding="utf8")))
        self.key = api_key
        self.progress, self.offline = progress, offline
        schema = ILLUSTRIS_FIELDS if self.original_illustris else MINI
        self.fields = {fam: set(names) for fam, names in schema.items()}
        if not self.original_illustris and snap in FULL_SNAPS:
            for fam in self.fields:
                self.fields[fam].update(SUBFIND, FULL_EXTRA[fam])
        self.catalog_fields = {"halo": {}, "subhalo": {}}
        if run == "TNG-Cluster":
            self.fields["bh"].update(("BH_WindTimes", "BH_WindCount", "BH_MPB_CumEgyLow", "BH_MPB_CumEgyHigh"))
            if snap in FULL_SNAPS:
                self.fields["gas"].add("AllowRefinement")
        self._catalogs = {}
        self._archive_catalogs = set()
        self._ranges_via_api = set()
        self._particle_files = None

    def _check_online(self, resource):
        if self.offline:
            raise RemoteDataError(f"Offline cache miss: {self.run} / {self.snap} / {resource}")

    def _open(self, suffix, resource, headers=None, *, read=None, target=None, progress=None, catalog_fallback=False):
        """Perform an idempotent GET, retrying its body reader along with the request."""
        self._check_online(resource)
        key = self.key or os.environ.get("TNG_API_KEY")
        if not key:
            key = dotenv_values(Path.cwd() / ".env", encoding="utf-8-sig",
                                interpolate=False).get("TNG_API_KEY")
        if not key:
            raise RemoteDataError(f"Set TNG_API_KEY or pass api_key to fetch {resource}")
        req = Request(urljoin(self.url, suffix), headers=headers or {})
        if ".hdf5" not in suffix and not suffix.startswith("../../files/groupcat-"):
            req.add_header("Accept", "application/json")
        # Authenticate only the original API request. urllib drops this header on redirects.
        req.add_unredirected_header("api-key", key)
        if target and suffix in self._ranges_via_api:
            target = None
        refreshed = False
        retries = 0
        wait_budget = 10.0
        started = time.monotonic()
        while True:
            body_started = False
            try:
                request = Request(target, headers=headers or {}) if target else req
                with progress.waiting(resource) if progress is not None else nullcontext():
                    # The exact object catalogue is generated on demand. A healthy
                    # response can take over 30 seconds before its headers arrive.
                    response = urlopen(request, timeout=60 if catalog_fallback else 120)
                body_started = True
                if read is None:
                    return response
                with response:
                    return read(response)
            except (OSError, HTTPException) as exc:
                if isinstance(exc, HTTPError):
                    exc.close()
                    if not refreshed and exc.code == 403 and (target or _download_forbidden(exc)):
                        # Only the original API is authenticated. Renew an expired
                        # download URL without forwarding that key to the data host.
                        if target:
                            # Reusing redirects from this file can cost a failed
                            # round trip on every block. Let the API issue each one.
                            self._ranges_via_api.add(suffix)
                        refreshed = True
                        target = None
                        req.add_header("Cache-Control", "no-cache")
                        continue
                # info.json has a targeted HDF5 fallback. A slow gateway need
                # not time out repeatedly before we use the static archive.
                delay = _retry_delay(exc, retries)
                elapsed = time.monotonic() - started
                # A slow gateway can spend minutes preparing a cutout and then
                # return 504. Restarting that same wait compounds the outage.
                # A body interrupted during transfer can still be retried.
                retry_in_time = body_started or delay <= 30 - elapsed
                if (retry_in_time and _transient_get_error(exc) and retries < 2 and delay <= wait_budget
                        and not (catalog_fallback and _catalog_fallback_error(exc))):
                    retries += 1
                    wait_budget -= delay
                    if progress is not None:
                        progress.phase(f"Retrying {resource} in {delay:g} s")
                    time.sleep(delay)
                    continue
                host = f" at {urlsplit(exc.geturl()).hostname}" if isinstance(exc, HTTPError) else ""
                detail = f" (retry requested after {delay} s)" if delay > wait_budget else ""
                if not body_started and elapsed >= 30:
                    detail += f" (waited {elapsed:.1f} s)"
                raise RemoteDataError(f"TNG request failed for {resource}{host}: {exc}{detail}") from exc

    def _json(self, directory, filename, endpoint, resource, validate):
        path = directory / filename
        if path.is_file():
            return json.loads(path.read_text(encoding="utf8"))
        self._check_online(resource)
        directory.mkdir(parents=True, exist_ok=True)
        with FileLock(str(path) + ".lock"):
            if path.is_file():
                return json.loads(path.read_text(encoding="utf8"))
            value = self._fetch_json(endpoint, resource, validate)
            write_json(path, value)
            return value

    def _fetch_json(self, endpoint, resource, validate, *, catalog_fallback=False):
        def read(response):
            total = _response_length(response, resource)
            data = response.read()
            if total is not None and len(data) != total:
                raise _IncompleteTransfer(f"Incomplete download for {resource} ({len(data)} of {total} bytes)")
            return json.loads(data)

        with self._progress(f"Waiting for {resource}", total=1, unit="request") as bar:
            try:
                data = self._open(endpoint, resource, read=read, catalog_fallback=catalog_fallback)
            except RemoteDataError as exc:
                cause = exc.__cause__
                if catalog_fallback and _catalog_fallback_error(cause):
                    reason = f"HTTP {cause.code}" if isinstance(cause, HTTPError) else str(cause)
                    bar.set_postfix_str(reason)
                raise
            value = validate(data)
            bar.update(1)
        return value

    def _object_value(self, kind, number, name):
        path = self.root / "objects.sqlite3"
        if path.is_file():
            uri = path.resolve().as_uri() + "?mode=ro"
            with closing(sqlite3.connect(uri, uri=True)) as db:
                row = db.execute("SELECT value FROM metadata WHERE kind=? AND number=? AND name=?",
                                 (kind, int(number), name)).fetchone()
                if row is not None:
                    return row[0]

    def _store_object_value(self, kind, number, name, value, *, local=False):
        if self.offline and not local:
            return
        path = self.root / "objects.sqlite3"
        if not path.is_file():
            self.root.mkdir(parents=True, exist_ok=True)
            with FileLock(str(path) + ".lock"):
                if not path.is_file():
                    with temporary_path(path) as tmp:
                        with closing(sqlite3.connect(tmp)) as db, db:
                            db.execute("""CREATE TABLE metadata (
                                kind TEXT, number INTEGER, name TEXT, value BLOB NOT NULL,
                                PRIMARY KEY (kind, number, name)
                            ) WITHOUT ROWID""")
                        os.replace(tmp, path)
        with closing(sqlite3.connect(path, timeout=30)) as db, db:
            db.execute("INSERT OR IGNORE INTO metadata VALUES (?, ?, ?, ?)",
                       (kind, int(number), name, value))

    def _object_json(self, kind, number, name, read=None, *, local=False):
        cached = self._object_value(kind, number, name)
        if cached is not None:
            return json.loads(cached)
        if read is not None:
            value = read()
            self._store_object_value(kind, number, name, json.dumps(value), local=local)
            return value

    def _check_mirror_counts(self, header):
        for field in ("Ngroups_Total", "Nsubgroups_Total"):
            if field in self._local_header and field in header and self._local_header[field] != header[field]:
                raise ValueError(f"local_dir {field}={self._local_header[field]} differs from "
                                 f"{self.run} snapshot {self.snap} catalog count {header[field]}")

    def header(self):
        local = self._local_header

        def convert(meta):
            if meta["number"] != self.snap:
                raise RemoteDataError("Snapshot metadata has the wrong snapshot number")
            return dict(Time=1.0 / (1 + meta["redshift"]),
                        Ngroups_Total=meta["num_groups_fof"], Nsubgroups_Total=meta["num_groups_subfind"])

        def run_parameters(meta):
            if meta["name"] != self.run:
                raise RemoteDataError("Simulation metadata has the wrong run name")
            if not np.isfinite(meta["mass_dm"]) or meta["mass_dm"] <= 0:
                raise RemoteDataError(f"Invalid mass_dm for {self.run}")
            return dict(MassDM=meta["mass_dm"], HubbleParam=meta["hubble"], Omega0=meta["omega_0"],
                        OmegaLambda=meta["omega_L"], OmegaBaryon=meta["omega_B"], BoxSize=meta["boxsize"],
                        SofteningDMComoving=meta["softening_dm_comoving"],
                        SofteningDMMaxPhys=meta["softening_dm_max_phys"])

        jobs = []
        if not {"Time", "Ngroups_Total", "Nsubgroups_Total"} <= local.keys():
            jobs.append((self.root, "header.json", "", "snapshot metadata", convert))
        run_fields = {"BoxSize", "Omega0", "OmegaLambda", "HubbleParam", "OmegaBaryon",
                      "SofteningDMComoving", "SofteningDMMaxPhys"}
        need_run = not ({"MassTable", "MassDM"} & local.keys()) or not run_fields <= local.keys()
        directory = self.root.parent
        if need_run and (not self.offline or (directory / "simulation.json").is_file()):
            jobs.append((directory, "simulation.json", "../../", "simulation metadata", run_parameters))
        header = {}
        if len(jobs) == 2 and not self.offline and all(not (job[0] / job[1]).is_file() for job in jobs):
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(self._json, *job) for job in jobs]
                for future in futures:
                    header.update(future.result())
        else:
            for job in jobs:
                header.update(self._json(*job))
        self._check_mirror_counts(header)
        # Published HDF5 cosmology uses float64 scalars. Python floats from JSON
        # preserve float32 in NumPy 2 array arithmetic during unit conversion.
        for field in ("Time", "HubbleParam", "Omega0", "OmegaLambda", "OmegaBaryon"):
            if field in header:
                header[field] = np.float64(header[field])
        header.update(local)
        # Older offline caches store the run cosmology in the snapshot header.
        if need_run and not {"HubbleParam", "Omega0", "OmegaLambda"} <= header.keys():
            self._check_online("simulation metadata")
        return header

    @contextmanager
    def _range_cache(self, filename, *, particles=False, cache_only=False):
        metadata = re.fullmatch(r"(?:groupcat-\d+\.\d+|(?:sublink|offsets)\.\d+)\.hdf5", filename)
        if not particles and not metadata:
            if cache_only:
                raise _RangeCacheMiss(f"No cached blocks for {filename}")
            self._check_online(filename)
            yield None, None
            return
        directory = self.root / "particles" if particles else self.root.parent / "hdf5-blocks"
        path = directory / (filename + ".sqlite3")
        if not path.is_file():
            if particles or cache_only:
                raise _RangeCacheMiss(f"No cached blocks for {filename}")
            self._check_online(filename)
        with self._block_database(path, writable=not (self.offline or particles or cache_only)) as result:
            yield result

    @staticmethod
    @contextmanager
    def _block_database(path, *, writable):
        lock = FileLock(str(path) + ".lock")
        if writable and not path.is_file():
            path.parent.mkdir(parents=True, exist_ok=True)
            with lock:
                if not path.is_file():
                    with temporary_path(path) as tmp:
                        with closing(sqlite3.connect(tmp)) as db:
                            db.execute("CREATE TABLE file_info (id INTEGER PRIMARY KEY, size INTEGER, validator TEXT)")
                            db.execute("CREATE TABLE blocks (start INTEGER PRIMARY KEY, data BLOB NOT NULL)")
                            db.commit()
                        os.replace(tmp, path)
        uri = path.resolve().as_uri() + ("?mode=rw" if writable else "?mode=ro")
        with closing(sqlite3.connect(uri, uri=True)) as db:
            yield db, lock

    @contextmanager
    def _file_with_ranges(self, filename, local=None, *, particles=False, progress=None, cache_only=False):
        """Read raw HDF5 metadata by range, or replay previously cached particle ranges."""
        cached = self.root / "groupcat" / filename
        path = next((p for p in (local, cached) if p and Path(p).is_file()), None)
        if path:
            with h5py.File(path, "r") as file:
                yield file, None
            return
        with self._range_cache(filename, particles=particles, cache_only=cache_only) as (db, lock):
            with (nullcontext(progress) if progress is not None or cache_only else
                  self._progress(f"Reading {filename}", total=None, unit="B", unit_scale=True)) as bar:
                stream = _RangeFile(self, "../../files/" + filename, bar, db, lock, cache_only=particles or cache_only)
                with h5py.File(stream, "r") as file:
                    yield file, stream

    @contextmanager
    def _raw_file(self, filename, local=None, *, progress=None, cache_only=False):
        with self._file_with_ranges(filename, local, progress=progress, cache_only=cache_only) as (file, _):
            yield file

    def _offset_file(self, progress=None, *, cache_only=False):
        local = offsetPath(self.local, self.snap) if self.local else None
        return self._raw_file(f"offsets.{self.snap}.hdf5", local, progress=progress, cache_only=cache_only)

    def _groupcat_with_ranges(self, chunk, progress=None, *, cache_only=False):
        local = gcPath(self.local, self.snap, chunk) if self.local else None
        return self._file_with_ranges(f"groupcat-{self.snap}.{chunk}.hdf5", local,
                                      progress=progress, cache_only=cache_only)

    @contextmanager
    def _raw_groupcat(self, chunk, progress=None, *, cache_only=False):
        with self._groupcat_with_ranges(chunk, progress, cache_only=cache_only) as (file, _):
            yield file

    def _file_offsets(self, group, progress=None, *, cache_only=False):
        if self.original_illustris:
            with self._raw_groupcat(0, progress, cache_only=cache_only) as file:
                return file["Header"].attrs["FileOffsets_" + group]
        with self._offset_file(progress, cache_only=cache_only) as file:
            return file["FileOffsets/" + group][:]

    def _catalog_location(self, group, number, progress=None, *, cache_only=False):
        starts = self._file_offsets(group, progress, cache_only=cache_only)
        chunk = int(np.searchsorted(starts, number, side="right")) - 1
        if chunk < 0:
            raise RemoteDataError(f"Invalid file offsets for {group} {number} in {self.run} snapshot {self.snap}")
        return [chunk, int(number - starts[chunk])]

    def _layout(self, path, read):
        if not path.is_file():
            path.parent.mkdir(parents=True, exist_ok=True)
            with FileLock(str(path) + ".lock"):
                if not path.is_file():
                    write_json(path, read())
        return json.loads(path.read_text(encoding="utf8"))

    def _column_file(self, path, endpoint, dataset, count, *, raw=False):
        def validate(file):
            if dataset not in file or file[dataset].shape[:1] != (count,):
                raise RemoteDataError(f"Missing or incomplete {dataset} in {path.name}")
        if not path.is_file():
            self._check_online(dataset)
            path.parent.mkdir(parents=True, exist_ok=True)
            with FileLock(str(path) + ".lock"):
                if not path.is_file():
                    with temporary_path(path) as tmp:
                        if raw:
                            with self._raw_file(endpoint) as source, h5py.File(tmp, "w") as dest:
                                validate(source)
                                column = source[dataset]
                                target = dest.create_dataset(dataset, shape=column.shape, dtype=column.dtype)
                                step = max(1, (8 * 1024**2) // (column.dtype.itemsize * int(np.prod(column.shape[1:]))))
                                for start in range(0, count, step):
                                    target[start:start+step] = column[start:start+step]
                        else:
                            self._stream(endpoint, dataset, tmp)
                        with h5py.File(tmp, "r") as file:
                            validate(file)
                        os.replace(tmp, path)
        with h5py.File(path, "r") as file:
            validate(file)
            return file[dataset][:]

    @staticmethod
    def _field_names(fields):
        if any(not isinstance(k, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", k) for k in fields):
            raise ValueError("Expected HDF5 field names")
        return list(dict.fromkeys(fields))

    def catalog_columns(self, kind, fields):
        group = CATALOG[kind]
        name = "groups" if kind == "Halo" else "subgroups"
        count = self.header()["N" + name + "_Total"]
        result = {"count": np.int64(count)}
        if not count:
            return result
        def chunk_path(i):
            if self.local:
                path = Path(gcPath(self.local, self.snap, i))
                if path.is_file():
                    return path
            return self.root / "groupcat" / f"groupcat-{self.snap}.{i}.hdf5"

        chunks = []
        if chunk_path(0).is_file():
            with h5py.File(chunk_path(0), "r") as first:
                if not fields and group in first:
                    fields = list(first[group])
                chunks = [chunk_path(i) for i in range(first["Header"].attrs["NumFiles"])]
            if not all(path.is_file() for path in chunks):
                chunks = []
        def available_fields():
            # The HDF5 schema includes simulation-specific and supplementary columns.
            def names_in_chunk(i):
                filename = f"groupcat-{self.snap}.{i}.hdf5"
                with self._raw_file(filename, chunk_path(i)) as file:
                    return list(file[group]) if group in file else []
            names = names_in_chunk(0)
            if not names:
                chunk, _ = self._catalog_location(group, 0)
                names = names_in_chunk(chunk)
            if not names:
                raise RemoteDataError(f"No {group} datasets in a non-empty catalog")
            return names

        if not fields:
            fields = self._layout(self.root / "groupcat" / group / "schema.json", available_fields)
        single = len(fields) == 1
        fields = self._field_names(fields)
        pieces = {field: [] for field in fields}
        for chunk in chunks:
            with h5py.File(chunk, "r") as file:
                catalogue = file.get(group)
                if catalogue is not None:
                    for field in fields:
                        if field in catalogue:
                            pieces[field].append(catalogue[field][:])

        result.update(dict.fromkeys(fields))  # Keep the requested column order.
        paths = {}
        for field in fields:
            parts = pieces.pop(field)
            if parts and sum(map(len, parts)) == count:
                result[field] = np.concatenate(parts) if len(parts) > 1 else parts[0]
            else:
                paths[field] = self.root / "groupcat" / group / (field + ".hdf5")

        def read_column(field):
            endpoint = f"../../files/groupcat-{self.snap}/?" + urlencode({group: field})
            return self._column_file(paths[field], endpoint, f"{group}/{field}", count)

        # The archive serves one column per GET. Overlap independent misses,
        # leaving local reads and single-column requests on the caller's thread.
        concurrent = not self.offline and sum(not path.is_file() for path in paths.values()) > 1
        with (ThreadPoolExecutor(max_workers=2) if concurrent else nullcontext()) as pool:
            values = map(read_column, paths) if pool is None else pool.map(read_column, paths)
            result.update(zip(paths, values))
        return result[fields[0]] if single else result

    def merger_tree(self, number, fields, *, only_mpb=False, optional=False):
        if not isinstance(number, (int, np.integer)) or number < 0:
            raise ValueError("Expected a non-negative integer subhalo ID")
        fields = self._field_names(fields)
        # Older local catalogs store SubLink offsets inside the group catalog.
        if not self.original_illustris and self.local and Path(gcPath(self.local, self.snap)).name.startswith("groups_"):
            try:
                treePath(self.local, "SubLink", 0)
            except ValueError:
                pass
            else:
                try:
                    local_fields = fields
                    if optional:
                        with h5py.File(treePath(self.local, "SubLink", 0), "r") as file:
                            local_fields = [k for k in fields if k in file]
                        if not local_fields:
                            return {}
                    tree = loadTree(self.local, self.snap, number, fields=local_fields, onlyMPB=only_mpb)
                except (FileNotFoundError, KeyError):
                    pass
                else:
                    if tree is None:
                        raise ValueError(f"Subhalo {number} has no SubLink tree at snapshot {self.snap}")
                    return {local_fields[0]: tree} if isinstance(tree, np.ndarray) else tree
        def locate():
            if self.original_illustris:
                group_chunk, index = self._catalog_location("Subhalo", number)
                raw = self._raw_groupcat(group_chunk)
                prefix = "Offsets/Subhalo_Sublink"
            else:
                raw, index = self._offset_file(), number
                prefix = "Subhalo/SubLink/"
            with raw as file:
                row = int(file[prefix + "RowNum"][index])
                if row < 0:
                    raise ValueError(f"Subhalo {number} has no SubLink tree at snapshot {self.snap}")
                root = int(file[prefix + "SubhaloID"][index])
                length = int(file[prefix + "LastProgenitorID"][index]) - root + 1
                starts = (file["Header"].attrs["FileOffsets_SubLink"] if self.original_illustris
                          else file["FileOffsets/SubLink"][:])
            chunk = int(np.searchsorted(starts, row, side="right")) - 1
            if chunk < 0 or length < 1:
                raise RemoteDataError("Invalid SubLink offsets")
            return [chunk, row - int(starts[chunk]), length, root]

        chunk, row, length, root = self._object_json("Subhalo", number, "sublink-location", locate,
                                                    local=self.local is not None)
        filename = f"sublink.{chunk}.hdf5"
        local = None
        if self.local:
            try:
                local = treePath(self.local, "SubLink", chunk)
            except ValueError:
                pass
        def read(file):
            if int(file["SubhaloID"][row]) != root:
                raise RemoteDataError("SubLink offsets do not match tree rows")
            size = length
            if only_mpb:
                size = int(file["MainLeafProgenitorID"][row]) - root + 1
                if not 1 <= size <= length:
                    raise RemoteDataError("Invalid SubLink main progenitor branch length")
            result = {field: file[field][row:row+size] for field in fields if not optional or field in file}
            if any(len(value) != size for value in result.values()):
                raise RemoteDataError("Incomplete SubLink columns")
            return result

        def read_cached(path, stream=None):
            with self._block_database(path, writable=stream is not None) as (db, lock):
                reader = _LocalTreeRangeFile(self, filename, db, lock, stream)
                with h5py.File(reader, "r") as file:
                    return read(file)

        # Keep byte ranges from distinct HDF5 layouts in separate caches.
        directory = self.root.parent / "local-sublink-blocks"
        if local:
            with open(local, "rb") as stream:
                stat = os.fstat(stream.fileno())
                identity = f"{stat.st_dev:x}-{stat.st_ino:x}-{stat.st_mtime_ns:x}-{stat.st_size:x}"
                return read_cached(directory / f"{filename}.{identity}.sqlite3", stream)
        for path in directory.glob(f"{filename}.*.sqlite3"):
            try:
                return read_cached(path)
            except _LocalTreeCacheMiss:
                pass
        with self._raw_file(filename) as file:
            return read(file)

    def at_snapshot(self, snap):
        if snap == self.snap:
            return self
        return ApiSource(self.run, int(snap), self.root.parent.parent, self.local,
                         self.key, self.progress, self.offline)

    def halo_values(self, number, fields, *, fetch=False):
        """Read one FoF row from existing row/column caches, then fill requested cells."""
        fields = self._field_names(fields)
        if not isinstance(number, (int, np.integer)) or number < 0:
            raise ValueError("Expected a non-negative FoF halo ID")
        row = self._local_catalog("Halo", number)
        if row is None:
            row = self._catalogs.get(("Halo", number))
        if row is None:
            row = self._object_json("Halo", number, "catalog")
        if row is None:
            row = self._catalog_row("Group", number)
        result = {k: self._catalog_array(k, row[k])[()] for k in fields if row is not None and k in row}
        for k in fields:
            if k in result:
                continue
            cell = self._cached_catalog_cell("Halo", number, k)
            if cell is not None:
                result[k] = cell
        if fetch and set(fields) - result.keys():
            chunk, index = self._object_json("Halo", number, "catalog-location",
                                             lambda: self._catalog_location("Group", number))
            with self._raw_groupcat(chunk) as file:
                for k in fields:
                    if k not in result:
                        result[k] = file["Group"][k][index]
                        self._store_catalog_cell("Halo", number, k, result[k])
        return result

    def _cached_catalog_cell(self, kind, number, name):
        group = CATALOG[kind]
        column = self.root / "groupcat" / group / (name + ".hdf5")
        if column.is_file():
            with h5py.File(column, "r") as file:
                return file[group][name][number]
        cell = self._object_value(kind, number, "catalog:" + name)
        if cell is not None:
            return np.load(BytesIO(cell), allow_pickle=False)[()]

    def _store_catalog_cell(self, kind, number, name, value):
        with BytesIO() as stream:
            np.save(stream, value, allow_pickle=False)
            self._store_object_value(kind, number, "catalog:" + name, stream.getvalue())

    def history_header(self, snap):
        local = local_snapshot_header(self.local, int(snap), expected_run=self.run) if self.local else {}
        header = self.header()
        if "Time" in local:
            return {**header, **local}
        if snap != self.snap:
            header["Time"] = np.float64(self._scale_factors()[str(int(snap))])
        return header

    def history_scales(self, numbers, header):
        """Resolve a branch's epochs with one lookup of the run-wide redshift table."""
        times = {self.snap: header["Time"]}
        missing = set(map(int, numbers)) - times.keys()
        if self.local:
            for i in missing:
                local = local_snapshot_header(self.local, i, expected_run=self.run)
                if "Time" in local:
                    times[i] = local["Time"]
        if missing - times.keys():
            table = self._scale_factors()
            times.update({i: table[str(i)] for i in missing - times.keys()})
        return np.asarray([times[int(i)] for i in numbers])

    def _scale_factors(self):
        return self._json(self.root.parent, "scale_factors.json", "../", "snapshot redshifts",
                          lambda rows: {str(r["number"]): 1 / (1 + r["redshift"]) for r in rows})

    def _zoom_chunks(self, number):
        if self.run != "TNG-Cluster":
            raise ValueError("Zoom selections require TNG-Cluster")
        def locate():
            with self._offset_file() as file:
                zoom = file["OriginalZooms"]
                return [file["FileOffsets/Group"][:].tolist(), zoom["GroupsTotalLengthByType"][:].tolist(),
                        zoom["OuterFuzzTotalLengthByType"][:].tolist()]
        starts, groups, fuzz = self._layout(self.root / "zoom-layout.json", locate)
        if number >= self.header()["Ngroups_Total"]:
            raise ValueError(f"No FoF halo {number} in snapshot {self.snap}")
        i = int(np.searchsorted(starts, number, side="right")) - 1
        if not 0 <= i < len(groups) or len(fuzz) != len(groups):
            raise RemoteDataError("Invalid original zoom offsets")
        return [(i, np.asarray(groups[i])), (i + len(groups), np.asarray(fuzz[i]))]

    def _zoom_particles(self, number, fields):
        chunks = self._zoom_chunks(number)
        result = {}
        for fam, names in fields.items():
            pt = ZOOM_PART_TYPE[fam]
            for name in self._field_names(names):
                pieces = []
                for chunk, counts in chunks:
                    count = counts[pt]
                    if not count:
                        continue
                    dataset = f"PartType{pt}/{name}"
                    local = snapPath(self.local, self.snap, chunk) if self.local else None
                    if local and Path(local).is_file():
                        with h5py.File(local, "r") as file:
                            if dataset in file and len(file[dataset]) == count:
                                pieces.append(file[dataset][:])
                                continue
                    path = self.root / "zoom-chunks" / str(chunk) / fam / (name + ".hdf5")
                    filename = f"snapshot-{self.snap}.{chunk}.hdf5"
                    if fam in API_TYPE:
                        query = urlencode({API_TYPE[fam]: name})
                        endpoint = f"../../files/{filename}?{query}"
                        pieces.append(self._column_file(path, endpoint, dataset, count))
                    else:
                        pieces.append(self._column_file(path, filename, dataset, count, raw=True))
                if pieces:
                    result.setdefault(fam, {})[name] = (
                        concatenate_particle_arrays(name, pieces) if len(pieces) > 1 else pieces[0])
        return result

    def _catalog_row(self, group, number, download=False, *, field=None):
        """Read a catalogue cell or row through local files and HDF5 range caches."""
        kind = "Halo" if group == "Group" else "Subhalo"

        def read_chunk(chunk, row, *, cache_only):
            with self._groupcat_with_ranges(chunk, cache_only=cache_only) as (file, stream):
                columns = file.get(group)
                if columns is None:
                    return None
                if field is not None and field in columns:
                    dataset = columns[field]
                    return {field: dataset[row]} if row < len(dataset) else None
                if row >= len(columns[group + "LenType"]):
                    return None
                if field is not None:
                    return {}
                datasets = (stream.catalog_row_datasets(columns, row)
                            if stream is not None and not cache_only else columns.items())
                return {name: dataset[row] for name, dataset in datasets}

        if not download:
            saved = self._object_value(kind, number, "catalog-hdf5")
            if saved is not None:
                with h5py.File(BytesIO(saved), "r") as record:
                    if field is not None:
                        return {field: record[field][()]} if field in record else {}
                    return {name: dataset[()] for name, dataset in record.items()}
            # A legacy cache can contain the complete first chunk without the offsets file.
            try:
                value = read_chunk(0, number, cache_only=True)
                if value is not None:
                    return value
            except _RangeCacheMiss:
                pass
        try:
            chunk, row = self._catalog_location(group, number, cache_only=not download)
            value = read_chunk(chunk, row, cache_only=not download)
        except _RangeCacheMiss:
            return None
        if value is None:
            raise RemoteDataError(f"No {group} {number} in {self.run} snapshot {self.snap}")
        if download and self.local:
            # Either the offsets or the target shard may have come from the mirror.
            # Keep the typed row replayable even if the mirror is absent next time.
            if field is None:
                with BytesIO() as stream:
                    with h5py.File(stream, "w") as record:
                        for name, cell in value.items():
                            record[name] = cell
                    self._store_object_value(kind, number, "catalog-hdf5", stream.getvalue())
            elif field in value:
                self._store_catalog_cell(kind, number, field, value[field])
        return value

    def _local_catalog(self, kind, number):
        if self.local:
            try:
                args = {"haloID" if kind == "Halo" else "subhaloID": number}
                return loadSingle(self.local, self.snap, **args)
            except (FileNotFoundError, KeyError):
                pass

    def _describe_catalog_fields(self, kind, names):
        fields = self.catalog_fields[kind.lower()]
        for name in set(names) - fields.keys():
            fields[name] = parameter_all_Description("groupcatalogs", kind.lower(), name)

    def _remember_catalog(self, kind, number, value):
        self._describe_catalog_fields(kind, value)
        self._catalogs[kind, number] = value
        return value

    def _download_catalog(self, kind, number, field):
        group = CATALOG[kind]
        obj = f"{kind.lower()}s/{number}"
        self._check_online(f"{obj} / {field or 'catalog'}")
        endpoint = obj + "/info.json"

        def extract(meta):
            data = meta.get(group)
            if data is None:
                data = {k: v for k, v in meta.items() if k.startswith(group)
                        or (self.run == "TNG-Cluster" and k in ("TracerLengthType", "TracerOffsetType"))}
            counts = np.asarray(data[group + "LenType"])
            if counts.shape != (6,) or counts.dtype.kind not in "iu" or np.any(counts < 0):
                raise RemoteDataError(f"Invalid {group}LenType for {obj}")
            return data

        # Once the gateway has failed for this object, a second field can go
        # straight to the same shard. The JSON service can still rescue a failed range.
        key = kind, number
        sources = ("hdf5", "json") if key in self._archive_catalogs else ("json", "hdf5")
        errors = {}
        for source in sources:
            if source == "json":
                try:
                    row = self._object_json(kind, number, "catalog", lambda: self._fetch_json(
                        endpoint, endpoint, extract, catalog_fallback=True))
                except RemoteDataError as exc:
                    if not _catalog_fallback_error(exc.__cause__) and "hdf5" not in errors:
                        raise
                    errors[source] = exc
                else:
                    self._archive_catalogs.discard(key)
                    return row, True
            else:
                try:
                    row = self._catalog_row(group, int(number), download=True, field=field)
                except (RemoteDataError, OSError, KeyError, ValueError) as exc:
                    cause = exc.__cause__
                    if (not errors and isinstance(cause, HTTPError) and cause.code in (401, 403, 429)
                            and not _download_forbidden(cause)):
                        raise
                    errors[source] = exc
                else:
                    if field is not None:
                        self._archive_catalogs.add(key)
                    return row, field is None
        raise RemoteDataError(
            f"{self.run} snapshot {self.snap} {endpoint} failed ({errors['json']}). "
            f"HDF5 catalogue fallback failed ({errors['hdf5']})."
        ) from errors["hdf5"]

    def catalog(self, kind, number, *, field=None):
        """Return the full native catalogue row, or one native value when a field is given."""
        if kind not in CATALOG or not isinstance(number, (int, np.integer)) or number < 0:
            raise ValueError("Expected Halo or Subhalo and a non-negative integer ID")
        if field is not None and (not isinstance(field, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", field)):
            raise KeyError(field)
        if (kind, number) in self._catalogs:
            row = self._catalogs[kind, number]
            return row if field is None else row[field]
        complete = True
        value = self._local_catalog(kind, number)
        if value is None:
            value = self._object_json(kind, number, "catalog")
        if value is None and field is not None:
            cell = self._cached_catalog_cell(kind, number, field)
            if cell is not None:
                return cell
        if value is None:
            value = self._catalog_row(CATALOG[kind], int(number), field=field)
            complete = field is None
        if value is None:
            value, complete = self._download_catalog(kind, number, field)
        if complete:
            self._remember_catalog(kind, number, value)
        return value if field is None else value[field]

    def particle_counts(self, kind, number):
        """Use the database-backed subhalo summary when only the layout is needed."""
        if not isinstance(number, (int, np.integer)) or number < 0:
            raise ValueError("Expected a non-negative integer ID")
        if kind == "Zoom":
            return sum(counts for _, counts in self._zoom_chunks(number))
        obj = f"{kind.lower()}s/{number}"
        if kind == "Subhalo" and (kind, number) not in self._catalogs:
            local = self._local_catalog(kind, number)
            if local is not None:
                return self._remember_catalog(kind, number, local)["SubhaloLenType"]
            value = self._object_json(kind, number, "catalog")
            if value is not None:
                return value["SubhaloLenType"]

            def extract(meta):
                counts = [0] * 6
                if meta["id"] != number:
                    raise ValueError("Unexpected subhalo ID")
                for fam, pt in PART_TYPE.items():
                    count = meta["len_" + API_TYPE[fam]]
                    if type(count) is not int or count < 0:
                        raise ValueError("Invalid subhalo particle count")
                    counts[pt] = count
                return counts

            counts = self._object_json(kind, number, "counts")
            if counts is not None:
                return counts
            cached = self._catalog_row("Subhalo", int(number))
            if cached is not None:
                return cached["SubhaloLenType"]
            if not self.offline:
                try:
                    return self._object_json(kind, number, "counts",
                                             lambda: self._fetch_json(obj + "/", obj + " counts", extract))
                except RemoteDataError as exc:
                    cause = exc.__cause__
                    if not (_catalog_fallback_error(cause) or isinstance(cause, HTTPError) and cause.code == 404):
                        raise
                except (KeyError, ValueError, TimeoutError):
                    pass  # Use the full catalog when the summary is unavailable.
        return self.catalog(kind, number)[CATALOG[kind] + "LenType"]

    def _particle_offsets(self, kind, number, progress=None):
        group = CATALOG[kind]

        def locate():
            if self.local:
                try:
                    subset = getSnapOffsets(self.local, self.snap, number, group)
                except (FileNotFoundError, KeyError):
                    pass
                else:
                    if self._particle_files is None:
                        self._particle_files = subset["snapOffsets"]
                        self._store_object_value("Snapshot", 0, "particle-files",
                                                 json.dumps(self._particle_files.tolist()))
                    return subset["offsetType"].tolist()
            if self.original_illustris:
                chunk, row = self._catalog_location(group, number, progress)
                with self._raw_groupcat(chunk, progress) as file:
                    return file[f"Offsets/{group}_SnapByType"][row].tolist()
            with self._offset_file(progress) as file:
                return file[f"{group}/SnapByType"][number].tolist()

        return self._object_json(kind, number, "particle-offsets",
                                 lambda: self._particle_type_vector(locate(), "offsets").tolist())

    @staticmethod
    def _particle_type_vector(values, label):
        values = np.asarray(values)
        if values.shape != (6,) or values.dtype.kind not in "iu" or np.any(values < 0):
            raise RemoteDataError(f"Invalid snapshot particle {label}")
        return values

    @staticmethod
    def _offset_service_unavailable(exc):
        cause = exc.__cause__ or exc
        return (_transient_get_error(cause) or _download_forbidden(cause)
                or isinstance(cause, HTTPError) and cause.code == 404)

    def _valid_particle_field(self, key, shape, dtype):
        fam, name = key
        if not shape or dtype.kind not in "biufc":
            return False
        if name not in self.fields[fam]:
            # User-configured extra numeric datasets can have their own schema.
            return all(size > 0 for size in shape[1:])
        width = _PARTICLE_COMPONENTS.get(name)
        if name == "GFM_Metals":
            width = 9 if self.original_illustris else 10
        kinds = "iu" if name in _PARTICLE_INTEGERS else "f"
        return shape[1:] == ((width,) if width else ()) and dtype.kind in kinds

    def _has_particle_shards(self):
        # A partial mirror can start at any chunk. Even one legacy range can
        # supply a subset of a column once the object offsets are known.
        if self.local:
            directory = Path(self.local) / f"snapdir_{self.snap:03d}"
            if any(directory.glob(f"snap*_{self.snap:03d}.*.hdf5")):
                return True
        return (any((self.root / "particles").glob(f"snapshot-{self.snap}.*.hdf5.sqlite3"))
                or any((self.root / "groupcat").glob(f"snapshot-{self.snap}.*.hdf5")))

    def _particle_file_starts(self, progress=None):
        if self._particle_files is None:
            def read():
                if self.original_illustris:
                    with self._raw_groupcat(0, progress) as file:
                        return file["Header"].attrs["FileOffsets_Snap"].tolist()
                with self._offset_file(progress) as file:
                    return file["FileOffsets/SnapByType"][:].T.tolist()

            self._particle_files = np.asarray(self._object_json("Snapshot", 0, "particle-files", read))
        return self._particle_files

    def _cached_particle_chunk(self, chunk, selections, stack, progress):
        datasets = {}

        def collect(file, fields):
            for key in sorted(fields):
                fam, name = key
                path = f"PartType{PART_TYPE[fam]}/{name}"
                try:
                    if path in file:
                        dataset = file[path]
                        if (isinstance(dataset, h5py.Dataset)
                                and self._valid_particle_field(key, dataset.shape, dataset.dtype)
                                and len(dataset) >= max(rows.stop for _, rows in selections[key])):
                            datasets[key] = dataset
                except (OSError, RemoteDataError):
                    continue

        local = snapPath(self.local, self.snap, chunk) if self.local else None
        if local and Path(local).is_file():
            try:
                collect(stack.enter_context(h5py.File(local, "r")), selections.keys())
            except OSError:
                pass
        cached_fields = selections.keys() - datasets.keys()
        if not cached_fields:
            return datasets

        filename = f"snapshot-{self.snap}.{chunk}.hdf5"
        try:
            file, stream = stack.enter_context(self._file_with_ranges(filename, particles=True, progress=progress))
        except (OSError, RemoteDataError):
            return datasets
        collect(file, cached_fields)
        if stream is not None:
            for key in cached_fields & datasets.keys():
                dataset = datasets[key]
                offset = dataset.id.get_offset()
                row_size = dataset.dtype.itemsize * int(np.prod(dataset.shape[1:]))
                if offset is not None and row_size and not dataset.dtype.hasobject:
                    selections[key] = [part for first, rows in selections[key]
                                       for part in stream.cached_row_slices(offset, row_size, first, rows)]
        return datasets

    def _read_cached_particle_chunks(self, selections, progress):
        pieces = {key: [] for key in selections}
        chunks = {}
        file_starts = self._particle_file_starts(progress)
        if (file_starts.ndim != 2 or file_starts.shape[0] != 6 or not file_starts.shape[1]
                or file_starts.dtype.kind not in "iu" or np.any(file_starts[:, 0] != 0)
                or np.any(file_starts[:, 1:] < file_starts[:, :-1])):
            raise RemoteDataError("Invalid snapshot file offsets")
        for key, intervals in selections.items():
            starts = file_starts[PART_TYPE[key[0]]]
            for start, stop in intervals:
                first = int(np.searchsorted(starts, start, side="right")) - 1
                last = int(np.searchsorted(starts, stop, side="left"))
                if first < 0 or last <= first:
                    raise RemoteDataError(f"Invalid particle interval for {key[0]}")
                for chunk in range(first, last):
                    end = min(stop, int(starts[chunk+1])) if chunk+1 < len(starts) else stop
                    begin = int(starts[chunk])
                    lo = max(begin, start)
                    if end > lo:
                        rows = slice(lo - begin, end - begin)
                        chunks.setdefault(chunk, {}).setdefault(key, []).append((lo, rows))

        for chunk, slices in sorted(chunks.items()):
            with ExitStack() as stack:
                datasets = self._cached_particle_chunk(chunk, slices, stack, progress)
                for key, dataset in datasets.items():
                    for first, rows in slices[key]:
                        try:
                            values = dataset[rows]
                        except (OSError, RemoteDataError):
                            continue
                        pieces[key].append((first, values))
        return pieces

    @staticmethod
    def _missing_particle_rows(bounds, pieces):
        pending = {}
        for key, (start, stop) in bounds.items():
            covered = sorted((first, first + len(values)) for first, values in pieces[key])
            gaps = list(missing_rows(start, stop, covered))
            if gaps:
                pending[key] = gaps
        return pending

    @staticmethod
    def _assemble_particle_rows(key, bounds, pieces):
        next_row, stop = bounds
        arrays = []
        for first, values in sorted(pieces, key=lambda part: part[0]):
            if first != next_row:
                raise RemoteDataError(f"Incomplete rows for {key[0]}.{key[1]}")
            next_row += len(values)
            arrays.append(values)
        if next_row != stop:
            raise RemoteDataError(f"Wrong particle count for {key[0]}.{key[1]}")
        if len(arrays) == 1:
            return arrays[0]
        dtype = arrays[0].dtype
        if all(array.dtype == dtype for array in arrays):
            result = np.empty((stop - bounds[0], *arrays[0].shape[1:]), dtype=dtype)
            return np.concatenate(arrays, out=result)
        return concatenate_particle_arrays(key[1], arrays)

    def _download_particle_cutout(self, obj, keys, path, progress):
        fields = {}
        for fam, name in keys:
            fields.setdefault(fam, set()).add(name)
        query = urlencode({API_TYPE[fam]: ",".join(sorted(names)) for fam, names in fields.items()})
        self._stream(obj + "/cutout.hdf5?" + query, obj, path, progress=progress)

    def _cutout_rows(self, path, bounds, missing, progress):
        """Validate full object extents and take only the rows still required."""
        progress.phase("Validating cutout")
        result = {}
        with h5py.File(path, "r") as file:
            for key, intervals in missing.items():
                fam, name = key
                start, stop = bounds[key]
                dataset = file.get(f"PartType{PART_TYPE[fam]}/{name}")
                if (not isinstance(dataset, h5py.Dataset)
                        or not self._valid_particle_field(key, dataset.shape, dataset.dtype)
                        or dataset.shape[0] != stop - start):
                    raise RemoteDataError(f"Invalid cutout data for {fam}.{name}")
                result[key] = [(lo, dataset[lo-start:hi-start]) for lo, hi in intervals]
        return result

    @staticmethod
    def _add_particle_rows(target, rows):
        for key, spans in rows.items():
            target.setdefault(key, []).extend(spans)

    def _load_particle_rows(self, obj, bounds, cache, lock, stack, progress, prefetched):
        damaged = set()
        updates = {}
        pieces = {key: [] for key in bounds}
        if cache is not None and prefetched is None:
            progress.phase("Reading particle cache")
            pieces = cache.read(bounds, damaged=damaged)
        missing = self._missing_particle_rows(bounds, pieces)
        if missing and cache is not None and prefetched is None and self._has_particle_shards():
            progress.phase("Reading local snapshot rows")
            try:
                local = self._read_cached_particle_chunks(missing, progress)
            except (RemoteDataError, OSError, HTTPException) as exc:
                if not self._offset_service_unavailable(exc):
                    raise
                local = {}  # A cutout can still supply the missing rows.
            self._add_particle_rows(pieces, local)
            for key, start, stop in damaged:
                for first, values in local.get(key, ()):
                    lo, hi = max(first, start), min(first + len(values), stop)
                    if lo < hi:
                        updates.setdefault(key, []).append((lo, values[lo-first:hi-first]))
            missing = self._missing_particle_rows(bounds, pieces)

        if missing and self.offline:
            details = ", ".join(f"{fam}.{name} rows " + ", ".join(f"[{lo}, {hi})" for lo, hi in intervals)
                                for (fam, name), intervals in missing.items())
            self._check_online(f"{obj} / {details}")
        if missing and not lock.is_locked:
            progress.phase("Waiting for particle cache")
            self.root.mkdir(parents=True, exist_ok=True)
            stack.enter_context(lock)
            if cache is not None:
                # Another reader may have filled these gaps. Keep the valid
                # arrays we already decoded; only query outstanding intervals.
                self._add_particle_rows(pieces, cache.read_intervals(missing, damaged=damaged))
                missing = self._missing_particle_rows(bounds, pieces)
        if missing:
            if prefetched is None:
                path = stack.enter_context(temporary_path(self.root / "cutout.hdf5"))
                self._download_particle_cutout(obj, missing, path, progress)
            else:
                path, transfer = prefetched
                transfer.result()
            rows = self._cutout_rows(path, bounds, missing, progress)
            self._add_particle_rows(updates, rows)
            self._add_particle_rows(pieces, rows)
        if cache is not None and updates:
            progress.phase("Writing particle cache")
            cache.store(updates, damaged=damaged)
        return pieces

    def _catalog_array(self, name, value, *, with_units=False):
        try:
            unit = groupcat_units(name)
        except KeyError:
            unit = None
        if isinstance(value, (np.generic, np.ndarray)):
            array = np.array(value, copy=True)  # Preserve native HDF5 dtypes, including custom columns.
        else:
            array = _json_catalog_array(self.run, name, value, unit is not None)
        return SimArray(array, unit) if with_units and unit is not None else array

    def catalog_arrays(self, kind, number):
        return {name: self._catalog_array(name, value, with_units=True)
                for name, value in self.catalog(kind, number).items()}

    def catalog_array(self, kind, number, field):
        if field is None:
            raise KeyError(field)
        value = self.catalog(kind, number, field=field)
        self._describe_catalog_fields(kind, [field])
        return self._catalog_array(field, value, with_units=True)

    def _progress(self, description, **options):
        disable = self.progress is False
        if self.progress == "auto" and not _IN_NOTEBOOK:
            disable = None  # Let tqdm detect interactive terminals.
        bar = tqdm(desc=description, disable=disable, **options)
        if options.get("unit") == "B" and options.get("total") is None:
            _set_byte_total(bar, None)
        return bar

    def _stream(self, endpoint, resource, path, *, progress=None):
        if progress is not None:
            progress.mark_transport("Cutout")
        with (nullcontext(progress) if progress is not None else
              self._progress(f"Waiting for {resource}", total=None, unit="B", unit_scale=True)) as bar:
            def save(response):
                total = _response_length(response, resource)
                if progress is None:
                    _set_byte_total(bar, None if total is None else bar.n + total)
                    bar.set_description_str(resource)
                else:
                    progress.phase("Downloading cutout")
                # A retry starts a fresh body in the same unpublished temporary file.
                with path.open("wb") as stream:
                    _copy_response(response, stream.write, bar, total, f"Incomplete download for {resource}")

            self._open(endpoint, resource, read=save, progress=progress)

    def particles(self, kind, number, fields, *, counts=None):
        if kind == "Zoom":
            return self._zoom_particles(number, fields)
        if kind not in CATALOG or not isinstance(number, (int, np.integer)) or number < 0:
            raise ValueError("Expected Halo or Subhalo and a non-negative integer ID")
        if counts is None:
            counts = self.particle_counts(kind, number)
        counts = self._particle_type_vector(counts, "counts")
        request = {fam: set(self._field_names(names)) for fam, names in fields.items()
                   if names and counts[PART_TYPE[fam]]}
        if not request:
            return {}

        obj = f"{kind.lower()}s/{number}"
        details = ", ".join(f"{fam}.{name}" for fam, names in request.items() for name in sorted(names))
        try:
            with ExitStack() as stack:
                label = f"Loading {self.run}/{self.snap:03d}/{kind} {number} / {details}"
                progress = _ParticleProgress(self, stack, label)
                progress.phase("Checking particle offsets and cache")
                cache = ParticleRowCache(self.root / "particle-rows.sqlite3", self._valid_particle_field)
                lengths = {(fam, name): int(counts[PART_TYPE[fam]]) for fam, names in request.items()
                           for name in names}
                slot = (2 * int(number) + (kind == "Subhalo")) % 64
                lock = FileLock(str(self.root / f"particle-cutout-{slot}.lock"))
                offsets = self._object_json(kind, number, "particle-offsets")
                prefetched = None
                if offsets is None and not self.offline and not self._has_particle_shards():
                    self.root.mkdir(parents=True, exist_ok=True)
                    stack.enter_context(lock)
                    offsets = self._object_json(kind, number, "particle-offsets")
                    if offsets is None and not cache.has_fields(lengths):
                        # No local source can satisfy even one requested field.
                        # Transfer to a private file while the offsets are read.
                        path = stack.enter_context(temporary_path(self.root / "cutout.hdf5"))
                        pool = stack.enter_context(ThreadPoolExecutor(max_workers=1))
                        prefetched = path, pool.submit(self._download_particle_cutout, obj, lengths, path, progress)
                if offsets is None:
                    try:
                        offsets = self._particle_offsets(kind, number, progress)
                    except (RemoteDataError, OSError, HTTPException) as exc:
                        if self.offline:
                            rows = ", ".join(f"{fam}.{name} object rows [0, {count})"
                                             for (fam, name), count in lengths.items())
                            raise RemoteDataError(f"Cannot locate {rows}: {exc}") from exc
                        if not self._offset_service_unavailable(exc):
                            raise
                        # Full cutouts have their own object-relative row order.
                        # Without official offsets, no rows may enter the snapshot cache.
                        cache = None
                        offsets = [0] * 6
                offsets = self._particle_type_vector(offsets, "offsets")
                bounds = {}
                for key, length in lengths.items():
                    start = int(offsets[PART_TYPE[key[0]]])
                    bounds[key] = (start, start + length)
                pieces = self._load_particle_rows(obj, bounds, cache, lock, stack, progress, prefetched)
                progress.phase("Assembling particle arrays")
                result = {fam: {} for fam in request}
                for (fam, name), interval in bounds.items():
                    result[fam][name] = self._assemble_particle_rows((fam, name), interval, pieces[fam, name])
                progress.complete()
                return result
        except (RemoteDataError, OSError, HTTPException, KeyError, IndexError, ValueError) as exc:
            # pynbody treats OSError as an invitation to derive an array.
            raise RemoteDataError(f"Could not read {obj} / {details}: {exc}") from exc


class ApiLazyContext:
    """_loaded_index tracks the source object of each row across merges and masks."""
    def __init__(self, owner, source):
        self.owner, self.source = owner, source
        self.names = {fam: {snapshot_pa_name(k): k for k in keys} for fam, keys in source.fields.items()}

    def bind_to(self, owner):
        return ApiLazyContext(owner, self.source)

    def loadable_keys(self, fam=None):
        if fam is None:
            return sorted(set().union(*self.names.values()))
        return sorted(self.names.get(getattr(fam, "name", fam), {}))

    def load_array(self, name, fam=None):
        families = self.owner.families() if fam is None else [get_family(getattr(fam, "name", fam))]
        if not families or any(name not in self.names[f.name] for f in families):
            raise OSError(f"No particle field {name} for this selection")
        loaded = []
        for family in families:
            key = self.names[family.name][name]
            origins = self.owner._loaded_index[family.name]
            ids = np.asarray(self.owner[family]["iord"])
            result = None
            for token in np.unique(origins):
                kind, number = ("Halo", token) if token >= 0 else ("Subhalo", -1 - token)
                data = self.source.particles(kind, number, {family.name: [key, "ParticleIDs"]})[family.name]
                mask = origins == token
                rows = align_rows(data["ParticleIDs"], ids[mask])
                if result is None:
                    result = np.empty((len(ids),) + data[key].shape[1:], dtype=data[key].dtype)
                result[mask] = data[key][rows]
            if result is not None:
                loaded.append((family, SimArray(result, snapshot_units(key))))
        for family, array in loaded:
            self.owner[family][name] = array


class ApiObjectContext(ApiLazyContext):
    """A standalone object whose row counts are known before any particles are read."""

    @classmethod
    def new_snapshot(cls, source, kind, number, families, order):
        from .TNGload import _LazySnap

        counts = source.particle_counts(kind, number)
        # pynbody.new allocates zero-filled pos, vel and mass. They would mask
        # _load_array and return zeros on first access, so build only the layout.
        owner = _LazySnap()
        offset = 0
        for fam in order.split(","):
            end = offset + (int(counts[PART_TYPE[fam]]) if fam in families else 0)
            owner._family_slice[get_family(fam)] = slice(offset, end)
            offset = end
        owner._num_particles = offset
        owner._decorate()
        ctx = cls(owner, source)
        ctx.kind, ctx.number = kind, int(number)
        ctx.counts = counts
        ctx._batch_fields = None
        ctx._batch_lock = getattr(owner, "_get_array_lock", None) or RLock()
        for names in ctx.names.values():
            names.update(HaloID="HaloID", SubhaloID="SubhaloID")
            if kind == "Zoom":
                names["ZoomID"] = "ZoomID"
        ctx.names["dm"]["mass"] = "Masses"  # Synthesized from Mdm.
        owner._lazy_ctx = ctx
        return owner

    def load_array(self, name, fam=None):
        families = self.owner.families() if fam is None else [get_family(getattr(fam, "name", fam))]
        if not families or any(name not in self.names[f.name] for f in families):
            raise OSError(f"No particle field {name} for this selection")
        with self._batch_lock:
            self.read_fields(self._batch_fields or self._missing_fields((name,), families))

    def _missing_fields(self, names, families):
        fields = {}
        for family in families:
            available = set(self.owner[family].keys())
            missing = [self.names[family.name][name] for name in names if name not in available]
            if missing:
                fields[family.name] = missing
        return fields

    def prefetch_fields(self, names, families):
        with self._batch_lock:
            fields = self._missing_fields(names, families)
            if fields:
                # One ordinary pynbody lookup installs the whole batch. Its normal
                # postprocessing applies units and transformations to every new array.
                fam, keys = next(iter(fields.items()))
                self._batch_fields = fields
                try:
                    self.owner[get_family(fam)][snapshot_pa_name(keys[0])]
                finally:
                    self._batch_fields = None

    def read_fields(self, fields):
        request = {}
        for fam, keys in fields.items():
            remote = set(keys) - {"HaloID", "SubhaloID", "ZoomID"}
            if fam == "dm":
                remote.discard("Masses")
            if remote:
                request[fam] = remote
        data = self.source.particles(self.kind, self.number, request, counts=self.counts) if request else {}
        for fam, keys in fields.items():
            particles = self.owner[get_family(fam)]
            if not len(particles):
                continue
            for key in keys:
                if key in ("HaloID", "SubhaloID", "ZoomID"):
                    particles[key] = np.full(len(particles), self.number if key == self.kind + "ID" else -1,
                                             dtype=np.int32)
                else:
                    unit = snapshot_units(key)
                    if fam == "dm" and key == "Masses":
                        values = np.full(len(particles), self.owner.properties["Mdm"].in_units(unit))
                    else:
                        values = data[fam][key]
                    particles[snapshot_pa_name(key)] = SimArray(values, unit)
