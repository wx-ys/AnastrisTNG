"""On-demand Illustris and TNG data backed by a shared snapshot cache."""
import json
import os
import re
import sqlite3
import sys
import tempfile
from collections import OrderedDict
from contextlib import closing, contextmanager
from http.client import HTTPException
from io import BytesIO, RawIOBase
from pathlib import Path
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

PART_TYPE = {"gas": 0, "dm": 1, "star": 4, "bh": 5}
ZOOM_PART_TYPE = {**PART_TYPE, "dmlowres": 2, "tracer": 3}
API_TYPE = {"gas": "gas", "dm": "dm", "star": "stars", "bh": "bhs"}
CATALOG = {"Halo": "Group", "Subhalo": "Subhalo"}
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


class _RangeFile(RawIOBase):
    """Seekable, read-only view of an official static HDF5 file for h5py."""

    block_size = 64 * 1024

    def __init__(self, source, endpoint, bar, cache=None, lock=None):
        self.source, self.endpoint, self.bar = source, endpoint, bar
        self.cache, self.lock = cache, lock
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
        self.source._check_online(f"{self.endpoint} bytes starting at {start}")
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
            data = self.fetch(start, stop if next_hit is None else next_hit)
            with self.cache:
                self.cache.execute("INSERT OR REPLACE INTO file_info VALUES (1, ?, ?)",
                                   (self.size, self.validator))
                self.cache.executemany("INSERT INTO blocks VALUES (?, ?)",
                                       ((start + i, memoryview(data)[i:i+self.block_size])
                                        for i in range(0, len(data), self.block_size)))
            return data

    def fetch(self, start, stop):
        headers = {"Range": f"bytes={start}-{stop - 1}"}
        if self.validator is not None:
            headers["If-Range"] = self.validator
        if self.target:
            try:
                response = urlopen(Request(self.target, headers=headers), timeout=120)
            except HTTPError as exc:
                exc.close()
                if exc.code != 403:
                    raise RemoteDataError(f"Range request failed for {self.endpoint}: HTTP {exc.code}") from None
                self.target = None
        if not self.target:
            response = self.source._open(self.endpoint, self.endpoint, headers=headers)
        with response:
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
            data = response.read()
            if len(data) != last + 1 - first:
                raise RemoteDataError(f"Incomplete byte range for {self.endpoint}")
            self.size = size
            if self.cache is not None:
                self.validator = self.validator or validator
            if response.geturl() != urljoin(self.source.url, self.endpoint):
                self.target = response.geturl()
        self.bar.update(len(data))
        return data

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


def _download_forbidden(error):
    """Distinguish a data-server refusal after redirect from an API authorization error."""
    if not isinstance(error, HTTPError) or error.code != 403:
        return False
    target = urlsplit(error.geturl())
    host = target.hostname or ""
    return (target.scheme == "https" and host.endswith((".tng-project.org", ".illustris-project.org"))
            and (host.startswith("data-") or host.startswith("data.")))


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
        self.local = os.fspath(local_dir) if local_dir is not None else None
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
        self._particle_files = None

    def _check_online(self, resource):
        if self.offline:
            raise RemoteDataError(f"Offline cache miss: {self.run} / {self.snap} / {resource}")

    def _open(self, suffix, resource, headers=None):
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
        for attempt in range(2):
            try:
                return urlopen(req, timeout=120)
            except HTTPError as exc:
                exc.close()
                if attempt == 0 and _download_forbidden(exc):
                    # Request a fresh download location through the authenticated API.
                    req.add_header("Cache-Control", "no-cache")
                    continue
                host = urlsplit(exc.geturl()).hostname
                raise RemoteDataError(f"TNG request failed for {resource} at {host}: {exc}") from exc
            except URLError as exc:
                raise RemoteDataError(f"TNG request failed for {resource}: {exc}") from exc

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

    def _fetch_json(self, endpoint, resource, validate):
        with self._progress(f"Reading {resource}", total=1, unit="request") as bar:
            with self._open(endpoint, resource) as response:
                value = validate(json.load(response))
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

    def _store_object_value(self, kind, number, name, value):
        if self.offline:
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

    def _object_json(self, kind, number, name, read=None):
        cached = self._object_value(kind, number, name)
        if cached is not None:
            return json.loads(cached)
        if read is not None:
            value = read()
            self._store_object_value(kind, number, name, json.dumps(value))
            return value

    def header(self):
        local = local_snapshot_header(self.local, self.snap) if self.local else {}

        def convert(meta):
            if meta["number"] != self.snap:
                raise RemoteDataError("Snapshot metadata has the wrong snapshot number")
            return dict(Time=1.0 / (1 + meta["redshift"]),
                        Ngroups_Total=meta["num_groups_fof"], Nsubgroups_Total=meta["num_groups_subfind"])

        header = {}
        if not {"Time", "Ngroups_Total", "Nsubgroups_Total"} <= local.keys():
            header = self._json(self.root, "header.json", "", "snapshot metadata", convert)

        def run_parameters(meta):
            if meta["name"] != self.run:
                raise RemoteDataError("Simulation metadata has the wrong run name")
            if not np.isfinite(meta["mass_dm"]) or meta["mass_dm"] <= 0:
                raise RemoteDataError(f"Invalid mass_dm for {self.run}")
            return dict(MassDM=meta["mass_dm"], HubbleParam=meta["hubble"], Omega0=meta["omega_0"],
                        OmegaLambda=meta["omega_L"], OmegaBaryon=meta["omega_B"], BoxSize=meta["boxsize"],
                        SofteningDMComoving=meta["softening_dm_comoving"],
                        SofteningDMMaxPhys=meta["softening_dm_max_phys"])

        run_fields = {"BoxSize", "Omega0", "OmegaLambda", "HubbleParam", "OmegaBaryon",
                      "SofteningDMComoving", "SofteningDMMaxPhys"}
        if "MassTable" not in local or not run_fields <= local.keys():
            directory = self.root.parent
            # Old caches contain the original cosmology in header.json. Keep
            # them readable offline and acquire precise run parameters online.
            if not self.offline or (directory / "simulation.json").is_file():
                header.update(self._json(directory, "simulation.json", "../../", "simulation metadata", run_parameters))
            elif not {"HubbleParam", "Omega0", "OmegaLambda"} <= (header.keys() | local.keys()):
                self._check_online("simulation metadata")
        header.update(local)
        return header

    @contextmanager
    def _range_cache(self, filename, *, particles=False):
        legacy_catalog = self.original_illustris and re.fullmatch(r"groupcat-\d+\.\d+\.hdf5", filename)
        if not particles and not legacy_catalog and not re.fullmatch(r"(?:sublink|offsets)\.\d+\.hdf5", filename):
            self._check_online(filename)
            yield None, None
            return
        directory = self.root / "particles" if particles else self.root.parent / "hdf5-blocks"
        path = directory / (filename + ".sqlite3")
        lock = FileLock(str(path) + ".lock")
        if not path.is_file():
            self._check_online(filename)
            path.parent.mkdir(parents=True, exist_ok=True)
            with lock:
                if not path.is_file():
                    with temporary_path(path) as tmp:
                        with closing(sqlite3.connect(tmp)) as db:
                            db.execute("CREATE TABLE file_info (id INTEGER PRIMARY KEY, size INTEGER, validator TEXT)")
                            db.execute("CREATE TABLE blocks (start INTEGER PRIMARY KEY, data BLOB NOT NULL)")
                            db.commit()
                        os.replace(tmp, path)
        uri = path.resolve().as_uri() + ("?mode=ro" if self.offline else "?mode=rw")
        with closing(sqlite3.connect(uri, uri=True)) as db:
            yield db, lock

    @contextmanager
    def _raw_file(self, filename, local=None, *, particles=False):
        """Use a local raw file or read HDF5 metadata and datasets by byte range."""
        cached = self.root / "groupcat" / filename
        path = next((p for p in (local, cached) if p and Path(p).is_file()), None)
        if path:
            with h5py.File(path, "r") as file:
                yield file
            return
        with self._range_cache(filename, particles=particles) as (db, lock):
            with self._progress(f"Reading {filename}", total=None, unit="B", unit_scale=True) as bar:
                stream = _RangeFile(self, "../../files/" + filename, bar, db, lock)
                with h5py.File(stream, "r") as file:
                    yield file

    def _offset_file(self):
        local = offsetPath(self.local, self.snap) if self.local else None
        return self._raw_file(f"offsets.{self.snap}.hdf5", local)

    def _raw_groupcat(self, chunk):
        local = gcPath(self.local, self.snap, chunk) if self.local else None
        return self._raw_file(f"groupcat-{self.snap}.{chunk}.hdf5", local)

    def _file_offsets(self, group):
        if self.original_illustris:
            with self._raw_groupcat(0) as file:
                return file["Header"].attrs["FileOffsets_" + group]
        with self._offset_file() as file:
            return file["FileOffsets/" + group][:]

    def _catalog_location(self, group, number):
        starts = self._file_offsets(group)
        chunk = int(np.searchsorted(starts, number, side="right")) - 1
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
        for field in fields:
            dataset = f"{group}/{field}"
            pieces = []
            for chunk in chunks:
                with h5py.File(chunk, "r") as file:
                    if dataset in file:
                        pieces.append(file[dataset][:])
            if pieces and sum(map(len, pieces)) == count:
                result[field] = np.concatenate(pieces) if len(pieces) > 1 else pieces[0]
                continue
            path = self.root / "groupcat" / group / (field + ".hdf5")
            endpoint = f"../../files/groupcat-{self.snap}/?" + urlencode({group: field})
            result[field] = self._column_file(path, endpoint, dataset, count)
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

        chunk, row, length, root = self._object_json("Subhalo", number, "sublink-location", locate)
        local = None
        if self.local:
            try:
                local = treePath(self.local, "SubLink", chunk)
            except ValueError:
                pass
        with self._raw_file(f"sublink.{chunk}.hdf5", local) as file:
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
        result = {k: row[k] for k in fields if row is not None and k in row}
        for k in fields:
            if k in result:
                continue
            column = self.root / "groupcat" / "Group" / (k + ".hdf5")
            if column.is_file():
                with h5py.File(column, "r") as file:
                    result[k] = file["Group"][k][number]
            else:
                cell = self._object_value("Halo", number, "catalog:" + k)
                if cell is not None:
                    result[k] = np.load(BytesIO(cell), allow_pickle=False)[()]
        if fetch and set(fields) - result.keys():
            chunk, index = self._object_json("Halo", number, "catalog-location",
                                             lambda: self._catalog_location("Group", number))
            with self._raw_groupcat(chunk) as file:
                for k in fields:
                    if k not in result:
                        result[k] = file["Group"][k][index]
                        with BytesIO() as stream:
                            np.save(stream, result[k], allow_pickle=False)
                            self._store_object_value("Halo", number, "catalog:" + k, stream.getvalue())
        return result

    def history_header(self, snap):
        local = local_snapshot_header(self.local, int(snap)) if self.local else {}
        header = self.header()
        if "Time" in local:
            return {**header, **local}
        if snap != self.snap:
            header["Time"] = self._scale_factors()[str(int(snap))]
        return header

    def history_scales(self, numbers, header):
        """Resolve a branch's epochs with one lookup of the run-wide redshift table."""
        times = {self.snap: header["Time"]}
        missing = set(map(int, numbers)) - times.keys()
        if self.local:
            for i in missing:
                local = local_snapshot_header(self.local, i)
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
                    result.setdefault(fam, {})[name] = np.concatenate(pieces) if len(pieces) > 1 else pieces[0]
        return result

    def _catalog_file(self, filename, required, download):
        path = self.root / "groupcat" / filename
        if path.is_file():
            return path
        if not download:
            return None
        path.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(str(path) + ".lock"):
            if not path.exists():
                with temporary_path(path) as tmp:
                    self._stream("../../files/" + filename, filename, tmp)
                    with h5py.File(tmp, "r") as file:
                        if required not in file:
                            raise RemoteDataError(f"Missing {required} in {filename}")
                    os.replace(tmp, path)
        return path

    def _catalog_row(self, group, number, download=False):
        """Resolve one raw FoF/Subfind row. Low IDs often need only chunk zero."""
        if self.original_illustris and download:
            chunk, index = self._catalog_location(group, number)
            with self._raw_groupcat(chunk) as file:
                return {k: dataset[index].tolist() for k, dataset in file[group].items()}
        root = self.root / "groupcat"
        offsets_name = f"offsets.{self.snap}.hdf5"
        if not (root / offsets_name).is_file():
            first = self._catalog_file(f"groupcat-{self.snap}.0.hdf5", "Header", download)
            if first is None:
                return None
            with h5py.File(first, "r") as file:
                if group in file and number < len(file[group][group + "LenType"]):
                    return {k: v[number] for k, v in file[group].items()}
        if self.original_illustris:
            chunk, row = self._catalog_location(group, number)
        else:
            offsets = self._catalog_file(offsets_name, "FileOffsets/" + group, download)
            if offsets is None:
                return None
            with h5py.File(offsets, "r") as file:
                starts = file["FileOffsets/" + group][:]
                chunk = int(np.searchsorted(starts, number, side="right")) - 1
                row = number - int(starts[chunk])
        path = self._catalog_file(f"groupcat-{self.snap}.{chunk}.hdf5", "Header", download)
        if path is None:
            return None
        with h5py.File(path, "r") as file:
            if group not in file or row >= len(file[group][group + "LenType"]):
                raise RemoteDataError(f"No {group} {number} in {self.run} snapshot {self.snap}")
            return {k: v[row] for k, v in file[group].items()}

    def _local_catalog(self, kind, number):
        if self.local:
            try:
                args = {"haloID" if kind == "Halo" else "subhaloID": number}
                return loadSingle(self.local, self.snap, **args)
            except (FileNotFoundError, KeyError):
                pass

    def _remember_catalog(self, kind, number, value):
        fields = self.catalog_fields[kind.lower()]
        for name in value.keys() - fields.keys():
            fields[name] = parameter_all_Description("groupcatalogs", kind.lower(), name)
        self._catalogs[kind, number] = value
        return value

    def catalog(self, kind, number):
        if kind not in CATALOG or not isinstance(number, (int, np.integer)) or number < 0:
            raise ValueError("Expected Halo or Subhalo and a non-negative integer ID")
        if (kind, number) in self._catalogs:
            return self._catalogs[kind, number]
        group = CATALOG[kind]
        value = self._local_catalog(kind, number)
        if value is None:
            obj = f"{kind.lower()}s/{number}"

            def extract(meta):
                data = meta.get(group)
                if data is None:
                    data = {k: v for k, v in meta.items() if k.startswith(group)}
                counts = np.asarray(data[group + "LenType"])
                if counts.shape != (6,) or counts.dtype.kind not in "iu" or np.any(counts < 0):
                    raise RemoteDataError(f"Invalid {group}LenType for {obj}")
                return data

            value = self._object_json(kind, number, "catalog")
            if value is None:
                value = self._catalog_row(group, int(number))
            if value is None:
                try:
                    value = self._object_json(kind, number, "catalog",
                                              lambda: self._fetch_json(obj + "/info.json", obj, extract))
                except (RemoteDataError, TimeoutError) as exc:
                    cause = exc.__cause__
                    timed_out = isinstance(exc, TimeoutError) or isinstance(getattr(cause, "reason", None), TimeoutError)
                    server_error = isinstance(cause, HTTPError) and cause.code in (500, 502, 503, 504)
                    if not (timed_out or server_error or _download_forbidden(cause)):
                        raise
                    value = self._catalog_row(group, int(number), download=True)
                    if self.original_illustris:
                        self._store_object_value(kind, number, "catalog", json.dumps(value))
        return self._remember_catalog(kind, number, value)

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
                    timed_out = isinstance(getattr(cause, "reason", None), TimeoutError)
                    unavailable = isinstance(cause, HTTPError) and cause.code in (404, 500, 502, 503, 504)
                    if not (timed_out or unavailable):
                        raise
                except (KeyError, ValueError, TimeoutError):
                    pass  # Use the full catalog when the summary is unavailable.
        return self.catalog(kind, number)[CATALOG[kind] + "LenType"]

    def _particle_offsets(self, kind, number):
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
                chunk, row = self._catalog_location(group, number)
                with self._raw_groupcat(chunk) as file:
                    return file[f"Offsets/{group}_SnapByType"][row].tolist()
            with self._offset_file() as file:
                return file[f"{group}/SnapByType"][number].tolist()

        return self._object_json(kind, number, "particle-offsets", locate)

    def _particle_file_starts(self):
        if self._particle_files is None:
            def read():
                if self.original_illustris:
                    with self._raw_groupcat(0) as file:
                        return file["Header"].attrs["FileOffsets_Snap"].tolist()
                with self._offset_file() as file:
                    return file["FileOffsets/SnapByType"][:].T.tolist()

            self._particle_files = np.asarray(self._object_json("Snapshot", 0, "particle-files", read))
        return self._particle_files

    def _snapshot_particle_chunk(self, chunk, selections):
        data = {fam: {} for fam in selections}

        def read(file):
            for fam, (rows, fields) in selections.items():
                for name in sorted(fields - data[fam].keys()):
                    dataset = f"PartType{PART_TYPE[fam]}/{name}"
                    if dataset in file:
                        values = file[dataset][rows]
                        if len(values) != rows.stop - rows.start:
                            raise RemoteDataError(f"Incomplete snapshot data for {fam}.{name} in chunk {chunk}")
                        data[fam][name] = values

        local = snapPath(self.local, self.snap, chunk) if self.local else None
        if local and Path(local).is_file():
            with h5py.File(local, "r") as file:
                read(file)
        if any(fields - data[fam].keys() for fam, (_, fields) in selections.items()):
            filename = f"snapshot-{self.snap}.{chunk}.hdf5"
            with self._raw_file(filename, particles=True) as file:
                read(file)
            for fam, (_, fields) in selections.items():
                missing = fields - data[fam].keys()
                if missing:
                    raise RemoteDataError(f"Missing snapshot fields for {fam} in {filename}: {', '.join(sorted(missing))}")
        return data

    def catalog_arrays(self, kind, number):
        result = {}
        for name, value in self.catalog(kind, number).items():
            array = np.array(value, copy=True)
            try:
                unit = groupcat_units(name)
            except KeyError:
                # Preserve supplementary catalog columns whose units are unknown.
                result[name] = array
            else:
                result[name] = SimArray(array, unit)
        return result

    def _progress(self, description, **options):
        disable = self.progress is False
        if self.progress == "auto" and not _IN_NOTEBOOK:
            disable = None  # Let tqdm detect interactive terminals.
        return tqdm(desc=description, disable=disable, **options)

    def _stream(self, endpoint, resource, path):
        with self._progress(f"Waiting for {resource}", total=None, unit="B", unit_scale=True) as bar:
            with self._open(endpoint, resource) as response, path.open("wb") as stream:
                length = response.headers.get("Content-Length")
                total = int(length) if length else None
                bar.total = total
                bar.set_description_str(resource)
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    stream.write(block)
                    bar.update(len(block))
            if total is not None and path.stat().st_size != total:
                raise RemoteDataError(f"Incomplete download for {resource}")

    def particles(self, kind, number, fields, *, counts=None):
        if kind == "Zoom":
            return self._zoom_particles(number, fields)
        if counts is None:
            counts = self.particle_counts(kind, number)
        request = {fam: set(self._field_names(names)) for fam, names in fields.items()
                   if names and counts[PART_TYPE[fam]]}
        if not request:
            return {}

        obj = f"{kind.lower()}s/{number}"
        details = ", ".join(f"{fam}.{name}" for fam, names in request.items() for name in sorted(names))
        try:
            offsets = self._particle_offsets(kind, number)
            file_starts = self._particle_file_starts()
            if len(offsets) != 6 or file_starts.ndim != 2 or file_starts.shape[0] != 6:
                raise RemoteDataError("Invalid snapshot offsets")
            chunks = {}
            for fam, names in request.items():
                part_type = PART_TYPE[fam]
                start = int(offsets[part_type])
                stop = start + int(counts[part_type])
                starts = file_starts[part_type]
                first = int(np.searchsorted(starts, start, side="right")) - 1
                last = int(np.searchsorted(starts, stop, side="left"))
                if first < 0 or last <= first:
                    raise RemoteDataError(f"Invalid particle interval for {fam}")
                for chunk in range(first, last):
                    end = min(stop, starts[chunk + 1]) if chunk + 1 < len(starts) else stop
                    begin = int(starts[chunk])
                    if end > max(begin, start):
                        rows = slice(max(0, start - begin), int(end - begin))
                        chunks.setdefault(chunk, {})[fam] = rows, names

            pieces = {fam: {name: [] for name in names} for fam, names in request.items()}
            for chunk, selections in sorted(chunks.items()):
                for fam, values in self._snapshot_particle_chunk(chunk, selections).items():
                    for name, array in values.items():
                        pieces[fam][name].append(array)

            result = {}
            for fam, values in pieces.items():
                result[fam] = {}
                for name, arrays in values.items():
                    if sum(map(len, arrays)) != counts[PART_TYPE[fam]]:
                        raise RemoteDataError(f"Wrong particle count for {fam}.{name}")
                    result[fam][name] = np.concatenate(arrays) if len(arrays) > 1 else arrays[0]
            return result
        except (RemoteDataError, OSError, HTTPException, KeyError, IndexError) as exc:
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
        self.read_fields({f.name: [self.names[f.name][name]] for f in families
                          if name not in self.owner[f].keys()})

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
