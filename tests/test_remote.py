"""Compare API reads with local snapshots and public reference cutouts."""
import io
import json
import os
import shutil
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlsplit

import h5py
import numpy as np
from dotenv import dotenv_values

from AnastrisTNG.TNGapi import API_TYPE, PART_TYPE, RemoteDataError
from AnastrisTNG.TNGparticlecache import ParticleRowCache
from AnastrisTNG.TNGsimulation import Snapshot


class Response(io.BytesIO):
    def __init__(self, data):
        super().__init__(data)
        self.headers = {"Content-Length": str(len(data))}


class RemoteSnapshotTest(unittest.TestCase):
    def setUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        root = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        self.local = root / "TNG50-1" / "output"
        self.cache = root / "cache"
        self.calls = []
        self.bad_cutout_schema = None
        self.bad_cutout_length = False
        self.dm_mass = 3.07367708626465e-5
        self.samples = {}
        for pt in [0, 1, 4, 5]:
            row = np.arange(5, dtype=np.float32) + 10 * pt
            self.samples[pt] = {
                "Coordinates": np.column_stack([row, row + 1, row + 2]),
                "Velocities": np.column_stack([row, -row, row]),
                "ParticleIDs": np.arange(5, dtype=np.uint64) + np.uint64(2**60 + 10*pt),
                "Potential": -row,
            }
            if pt != 1:
                self.samples[pt]["Masses"] = row + 1
            if pt == 0:
                self.samples[pt]["Density"] = row + 0.5
        group = self.local / "groups_099" / "groups_099.0.hdf5"
        group.parent.mkdir(parents=True)
        self.members = {"Group": [slice(0, 3), slice(3, 5)], "Subhalo": [slice(0, 2), slice(3, 4)]}
        self.catalog = {}
        with h5py.File(group, "w") as file:
            header = file.create_group("Header").attrs
            header.update(Time=1., HubbleParam=.6774, Omega0=.3089, OmegaLambda=.6911,
                          BoxSize=35000., Ngroups_Total=2, Nsubgroups_Total=2,
                          NumFiles=1, Ngroups_ThisFile=2, Nsubgroups_ThisFile=2,
                          FileOffsets_Group=[0], FileOffsets_Subhalo=[0],
                          FileOffsets_Snap=np.array([[0, 2] if i in self.samples else [0, 0] for i in range(6)]))
            for kind, members in self.members.items():
                fields = {kind + "LenType": [[s.stop-s.start if i in self.samples else 0
                                               for i in range(6)] for s in members]}
                if kind == "Group":
                    fields.update(GroupFirstSub=[0, 1], GroupNsubs=[1, 1])
                else:
                    fields.update(SubhaloGrNr=[0, 1])
                mass_types = []
                for rows in members:
                    masses = [0.] * 6
                    masses[1] = (rows.stop - rows.start) * self.dm_mass
                    for pt in (0, 4, 5):
                        masses[pt] = float(self.samples[pt]["Masses"][rows].sum())
                    mass_types.append(masses)
                fields[kind + "MassType"] = mass_types
                fields[kind + "Mass"] = [sum(row) for row in mass_types]
                self.catalog[kind] = fields
                file["Offsets/" + kind + "_SnapByType"] = [
                    [s.start if i in self.samples else 0 for i in range(6)] for s in members]
                for name, values in fields.items():
                    file[kind + "/" + name] = values
        for chunk, rows in enumerate([slice(0, 2), slice(2, 5)]):
            path = self.local / "snapdir_099" / f"snap_099.{chunk}.hdf5"
            path.parent.mkdir(exist_ok=True)
            with h5py.File(path, "w") as file:
                file.create_group("Header").attrs.update(
                    MassTable=[0, self.dm_mass, 0, 0, 0, 0], OmegaBaryon=.0486,
                    NumPart_Total=[5 if i in self.samples else 0 for i in range(6)],
                    NumPart_ThisFile=[rows.stop-rows.start if i in self.samples else 0 for i in range(6)])
                file.create_group("Parameters").attrs.update(SofteningTypeOfPartType1=2,
                                                            SofteningComovingType2=.39, SofteningMaxPhysType2=.195)
                for pt, fields in self.samples.items():
                    for name, values in fields.items():
                        file[f"PartType{pt}/{name}"] = values[rows]
        self.archive = {f"snapshot-99.{chunk}.hdf5":
                        (self.local / f"snapdir_099/snap_099.{chunk}.hdf5").read_bytes()
                        for chunk in range(2)}
        with io.BytesIO() as stream:
            with h5py.File(stream, "w") as file:
                file["FileOffsets/SnapByType"] = np.array(
                    [[0, 2] if pt in self.samples else [0, 0] for pt in range(6)]).T
                for kind, members in self.members.items():
                    file[f"FileOffsets/{kind}"] = [0]
                    file[f"{kind}/SnapByType"] = [
                        [s.start if pt in self.samples else 0 for pt in range(6)] for s in members]
            self.archive["offsets.99.hdf5"] = stream.getvalue()
        stack.enter_context(patch("AnastrisTNG.TNGapi.urlopen", side_effect=self.respond))

    def respond(self, request, timeout):
        url = urlsplit(request.full_url)
        if url.path == "/api/TNG50-1/":
            self.calls.append(("simulation", {}))
            return Response(json.dumps(dict(name="TNG50-1", mass_dm=self.dm_mass, boxsize=35000.,
                                            omega_0=.3089, omega_L=.6911, omega_B=.0486, hubble=.6774,
                                            softening_dm_comoving=.39, softening_dm_max_phys=.195)).encode())
        if "/files/" in url.path:
            filename = url.path.rsplit("/", 1)[-1]
            content = self.archive[filename]
            start, end = map(int, request.get_header("Range")[6:].split("-"))
            stop = min(end + 1, len(content))
            self.calls.append((filename, (start, stop)))
            response = Response(content[start:stop])
            response.status = 206
            response.headers["Content-Range"] = f"bytes {start}-{stop-1}/{len(content)}"
            response.geturl = lambda: request.full_url
            return response
        path = url.path.split("/snapshots/99/")[1]
        query = parse_qs(url.query)
        self.calls.append((path, query))
        if not path:
            return Response(json.dumps(dict(number=99, redshift=0., num_groups_fof=2,
                                            num_groups_subfind=2)).encode())
        if path == "subhalos/":
            return Response(json.dumps({"results": [{"id": 0}]}).encode())
        group, num, endpoint = path.split("/")
        kind = "Group" if group == "halos" else "Subhalo"
        idx = int(num)
        if endpoint == "" and kind == "Subhalo":
            counts = self.catalog[kind]["SubhaloLenType"][idx]
            return Response(json.dumps(dict(id=idx, len_gas=counts[0], len_dm=counts[1],
                                            len_stars=counts[4], len_bhs=counts[5])).encode())
        if endpoint == "info.json":
            fields = {name: values[idx] for name, values in self.catalog[kind].items()}
            return Response(json.dumps({kind: fields}).encode())
        if endpoint == "cutout.hdf5":
            with io.BytesIO() as stream:
                with h5py.File(stream, "w") as file:
                    for family, particle_type in PART_TYPE.items():
                        for field in query.get(API_TYPE[family], [""])[0].split(","):
                            if not field:
                                continue
                            values = self.samples[particle_type][field][self.members[kind][idx]]
                            if field == "Coordinates":
                                if self.bad_cutout_schema == "shape":
                                    values = values[:, 0]
                                if self.bad_cutout_schema == "dtype":
                                    values = values.astype("int32")
                            file[f"PartType{particle_type}/{field}"] = values
                response = Response(stream.getvalue())
                if self.bad_cutout_length:
                    length = response.headers["Content-Length"]
                    response.headers["Content-Length"] = f"{length}, {length}"
                return response
        self.fail(f"Unexpected object endpoint: {path}")

    def remote(self, **kwargs):
        return Snapshot.from_api("TNG50-1", 99, cache_dir=self.cache, progress=False, **kwargs)

    def seed_range_cache(self, source, filename, *, omitted=()):
        content = self.archive[filename]
        path = source.root / "particles" / (filename + ".sqlite3")
        block_size = 64 * 1024
        with source._block_database(path, writable=True) as (db, _), db:
            db.execute("INSERT INTO file_info VALUES (1, ?, NULL)", (len(content),))
            db.executemany("INSERT INTO blocks VALUES (?, ?)", [
                (start, content[start:start+block_size])
                for start in range(0, len(content), block_size) if start not in omitted])
        return path

    def test_default_cache_created_on_first_online_write(self):
        home = self.cache.parent / "home"
        expected = home / ".cache/AnastrisTNG/TNG50-1/snap_099"
        with patch("pathlib.Path.home", return_value=home):
            with self.assertRaisesRegex(RemoteDataError, "Offline cache miss"):
                Snapshot.from_api("TNG50-1", 99, offline=True, progress=False)
            self.assertFalse(expected.exists())
            online = Snapshot.from_api("TNG50-1", 99, api_key="test", progress=False)
            self.assertEqual(online._source.root, expected)
            self.assertTrue((expected / "header.json").is_file())
            self.assertTrue((expected.parent / "simulation.json").is_file())

    def test_small_subhalo_field_and_partial_host_miss_use_cutout(self):
        snap = self.remote(api_key="test")
        sub = snap.read_particle_field(0, "gas", "rho")
        np.testing.assert_array_equal(sub, self.samples[0]["Density"][:2])
        self.assertEqual([call for call in self.calls if call[0].endswith("cutout.hdf5")],
                         [("subhalos/0/cutout.hdf5", {"gas": ["Density"]})])
        self.assertFalse(any(path.startswith("snapshot-") for path, _ in self.calls))
        self.calls.clear()
        halo = snap.read_particle_field(0, "gas", "rho", groupType="Halo")
        np.testing.assert_array_equal(halo, self.samples[0]["Density"][:3])
        self.assertEqual([call for call in self.calls if call[0].endswith("cutout.hdf5")],
                         [("halos/0/cutout.hdf5", {"gas": ["Density"]})])
        self.assertFalse(any(path.startswith("snapshot-") for path, _ in self.calls))
        self.calls.clear()
        offline = self.remote(offline=True)
        np.testing.assert_array_equal(offline.read_particle_field(0, "gas", "rho", groupType="Halo"), halo)
        np.testing.assert_array_equal(offline.read_particle_field(0, "gas", "rho"), sub)
        self.assertFalse(self.calls)
        self.assertFalse((snap._source.root / "halos").exists())
        self.assertFalse((snap._source.root / "subhalos").exists())

    def test_cutout_rows_are_shared_with_host_offline(self):
        source = self.remote(api_key="test")._source
        fields = {"star": ["Coordinates", "Masses"]}
        counts = source.particle_counts("Subhalo", 0)
        source.progress = True
        with patch("AnastrisTNG.TNGapi.tqdm") as progress:
            sub = source.particles("Subhalo", 0, fields, counts=counts)
            progress.assert_called_once()
            bar = progress.return_value.__enter__.return_value
            self.assertEqual(bar.total, sum(call.args[0] for call in bar.update.call_args_list))
            self.assertTrue(bar.set_description_str.call_args.args[0].startswith("[Cutout | offsets via Range] "))
        source.progress = False
        cutouts = [(path, query) for path, query in self.calls if path.endswith("cutout.hdf5")]
        self.assertEqual(cutouts, [("subhalos/0/cutout.hdf5",
                                    {"stars": ["Coordinates,Masses"]})])
        for field in fields["star"]:
            np.testing.assert_array_equal(sub["star"][field], self.samples[4][field][:2])

        # Proof that the cold read did not fetch reference IDs or snapshot headers.
        self.assertFalse(any(path.startswith("snapshot-") for path, _ in self.calls))
        self.assertFalse((source.root / "particles").exists())
        self.calls.clear()
        host = source.particles("Halo", 0, fields)
        self.assertEqual([call for call in self.calls if call[0].endswith("cutout.hdf5")],
                         [("halos/0/cutout.hdf5", {"stars": ["Coordinates,Masses"]})])
        self.assertFalse(any(path.startswith("snapshot-") for path, _ in self.calls))
        self.calls.clear()
        offline = self.remote(offline=True)._source
        for kind, size in (("Subhalo", 2), ("Halo", 3)):
            result = offline.particles(kind, 0, fields)
            for field in fields["star"]:
                np.testing.assert_array_equal(result["star"][field], self.samples[4][field][:size])
        self.assertFalse(self.calls)
        self.assertFalse(list(source.root.rglob("*.part")))

    def test_invalid_cutout_shape_can_retry_without_poisoning_cache(self):
        self.bad_cutout_schema = "shape"
        snap = self.remote(api_key="test")
        with self.assertRaisesRegex(RemoteDataError, "Invalid cutout data"):
            snap.read_particle_field(0, "star", "pos")
        self.assertFalse((snap._source.root / "particle-rows.sqlite3").exists())
        self.bad_cutout_schema = None
        data = snap.read_particle_field(0, "star", "pos")
        np.testing.assert_array_equal(data, self.samples[4]["Coordinates"][:2])
        self.assertEqual(sum(path.endswith("cutout.hdf5") for path, _ in self.calls), 2)
        self.assertFalse(any(path.startswith("snapshot-") for path, _ in self.calls))

    def test_invalid_cutout_dtype_never_fetches_raw_snapshot(self):
        self.bad_cutout_schema = "dtype"
        snap = self.remote(api_key="test")
        with self.assertRaisesRegex(RemoteDataError, "Invalid cutout data"):
            snap.read_particle_field(0, "star", "pos")
        self.assertFalse((snap._source.root / "particle-rows.sqlite3").exists())
        self.assertFalse(any(path.startswith("snapshot-") for path, _ in self.calls))

    def test_malformed_cutout_content_length_never_fetches_raw_snapshot(self):
        self.bad_cutout_length = True
        snap = self.remote(api_key="test")
        with self.assertRaisesRegex(RemoteDataError, "Invalid download length"):
            snap.read_particle_field(0, "star", "pos")
        self.assertFalse((snap._source.root / "particle-rows.sqlite3").exists())
        self.assertFalse(any(path.startswith("snapshot-") for path, _ in self.calls))

    def test_existing_range_cache_replays_without_redownloading(self):
        online = self.remote(api_key="test")
        online._source.particle_counts("Subhalo", 0)
        online._source._particle_offsets("Subhalo", 0)
        online._source._particle_file_starts()
        cache_path = self.seed_range_cache(online._source, "snapshot-99.0.hdf5")
        self.calls.clear()
        for snap in (online, self.remote(offline=True)):
            data = snap.read_particle_field(0, "star", "pos")
            np.testing.assert_array_equal(data, self.samples[4]["Coordinates"][:2])
        self.assertFalse(self.calls)
        self.assertTrue(cache_path.is_file())
        self.assertFalse((online._source.root / "particle-rows.sqlite3").exists())

    def test_progress_reports_cutout_only_when_offsets_are_cached(self):
        source = self.remote(api_key="test")._source
        source.particle_counts("Subhalo", 0)
        source._particle_offsets("Subhalo", 0)
        self.calls.clear()
        source.progress = True
        with patch("AnastrisTNG.TNGapi.tqdm") as progress:
            source.particles("Subhalo", 0, {"star": ["Coordinates"]})
            progress.assert_called_once()
            bar = progress.return_value.__enter__.return_value
            self.assertTrue(bar.set_description_str.call_args.args[0].startswith("[Cutout] "))
        self.assertEqual(self.calls, [("subhalos/0/cutout.hdf5", {"stars": ["Coordinates"]})])
        self.calls.clear()
        source.progress = False
        data = source.particles("Subhalo", 0, {"star": ["ParticleIDs"]})
        self.assertEqual(self.calls, [("subhalos/0/cutout.hdf5", {"stars": ["ParticleIDs"]})])
        np.testing.assert_array_equal(data["star"]["ParticleIDs"], self.samples[4]["ParticleIDs"][:2])

    def test_neutral_hydrogen_batches_missing_gas_fields(self):
        metals = np.zeros((5, 10), dtype="f4")
        metals[:, 0] = 0.76
        extras = dict(GFM_Metals=metals, ElectronAbundance=np.full(5, 0.1, dtype="f4"),
                      InternalEnergy=np.full(5, 100, dtype="f4"),
                      NeutralHydrogenAbundance=np.full(5, 0.8, dtype="f4"),
                      StarFormationRate=np.array([1, 0, 1, 0, 0], dtype="f4"))
        self.samples[0].update(extras)
        shard = self.local / "snapdir_099/snap_099.0.hdf5"
        with h5py.File(shard, "r+") as file:
            for field, values in extras.items():
                file[f"PartType0/{field}"] = values[:2]
        self.archive["snapshot-99.0.hdf5"] = shard.read_bytes()
        local = Snapshot(str(self.local), 99).load_particle(0)
        online = self.remote(api_key="test").load_particle(0)
        online.ancestor.gas["mass"] = local.gas["mass"].copy()

        actual = online.gas["mHI"]
        fields = "Density,ElectronAbundance,GFM_Metals,InternalEnergy,NeutralHydrogenAbundance,StarFormationRate"
        cutouts = [(url, query) for url, query in self.calls if url.endswith("cutout.hdf5")]
        self.assertEqual(cutouts, [("subhalos/0/cutout.hdf5", {"gas": [fields]})])
        self.assertFalse(any(path.startswith("snapshot-") for path, _ in self.calls))
        np.testing.assert_array_equal(actual, local.gas["mHI"])
        before = len(self.calls)
        np.testing.assert_array_equal(online.gas["mH2"], local.gas["mH2"])
        physical = self.remote(offline=True).load_particle(0)
        physical.ancestor.gas["mass"] = local.gas["mass"].astype(np.float32)
        physical.physical_units(persistent=True)
        self.assertEqual(physical.gas["mass"].dtype, np.dtype("float32"))
        self.assertEqual(str(physical.gas["mass"].units), "Msol")
        # These finite masses in Msol exceed the float32 range in grams.
        with np.errstate(over="raise"):
            for suffix in ("", "_BR06", "_SH03"):
                for species in ("mHI", "mH2"):
                    field = species + suffix
                    expected = local.gas[field]
                    self.assertTrue(np.isfinite(expected).all())
                    np.testing.assert_allclose(physical.gas[field], expected, rtol=1e-6)
        self.assertEqual(str(physical.gas["rho"].units), "Msol kpc**-3")
        self.assertEqual(len(self.calls), before)

    def test_partially_cached_raw_shard_uses_cutout_and_stores_only_missing_rows(self):
        host_count = 20000
        fields = {"star": ["Coordinates"]}
        self.samples[4] = {"Coordinates": np.arange(3*host_count, dtype="f4").reshape(host_count, 3)}
        self.members["Group"][0] = slice(0, host_count)
        with io.BytesIO(self.archive["offsets.99.hdf5"]) as content:
            with h5py.File(content, "r+") as file:
                file["FileOffsets/SnapByType"][1, 4] = host_count
            self.archive["offsets.99.hdf5"] = content.getvalue()
        with io.BytesIO() as content:
            with h5py.File(content, "w", alignment_threshold=1, alignment_interval=65536) as file:
                dataset = file.create_dataset("PartType4/Coordinates", data=self.samples[4]["Coordinates"])
                tail_block = dataset.id.get_offset() + 3 * 65536
            self.archive["snapshot-99.0.hdf5"] = content.getvalue()

        source = self.remote(api_key="test")._source
        self.seed_range_cache(source, "snapshot-99.0.hdf5", omitted=(tail_block,))
        counts = [0, 0, 0, 0, host_count, 0]
        data = source.particles("Halo", 0, fields, counts=counts)
        np.testing.assert_array_equal(data["star"]["Coordinates"], self.samples[4]["Coordinates"])
        self.assertEqual([call for call in self.calls if call[0].endswith("cutout.hdf5")],
                         [("halos/0/cutout.hdf5", {"stars": ["Coordinates"]})])
        self.assertFalse(any(path.startswith("snapshot-") for path, _ in self.calls))
        key = ("star", "Coordinates")
        row_cache = ParticleRowCache(source.root / "particle-rows.sqlite3")
        pieces = row_cache.read({key: (0, host_count)})[key]
        self.assertEqual(pieces[0][0], 3 * 65536 // 12)
        self.assertEqual(sum(len(rows) for _, rows in pieces), host_count - pieces[0][0])
        self.calls.clear()
        offline = self.remote(offline=True)._source.particles("Halo", 0, fields, counts=counts)
        np.testing.assert_array_equal(offline["star"]["Coordinates"], self.samples[4]["Coordinates"])
        self.assertFalse(self.calls)

    def test_particle_row_cache_deduplicates_concurrent_overlapping_writes(self):
        cache = ParticleRowCache(self.cache / "particle-rows.sqlite3")
        cache.block_bytes = 128
        key = ("star", "Coordinates")
        array = np.arange(72, dtype=">f4").reshape(24, 3)
        self.assertEqual(cache._pack(array[:10])[0], 1)
        requests = [{key: [(0, array[:15])]}, {key: [(8, array[8:])]}]
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(cache.store, requests))
        pieces = cache.read({key: (3, 20)})[key]
        self.assertEqual(sum(len(part) for _, part in pieces), 17)
        source = self.remote(api_key="test")._source
        actual = source._assemble_particle_rows(key, (3, 20), pieces)
        self.assertEqual(actual.dtype, array.dtype)
        np.testing.assert_array_equal(actual, array[3:20])

    def test_cutout_checks_scalar_and_multicomponent_field_schemas(self):
        source = self.remote(api_key="test")._source
        counts = self.catalog["Subhalo"]["SubhaloLenType"][0]
        for name, dims, dtype in [
            ("Density", (5, 2), "f4"), ("Density", (5,), "i4"),
            ("Density", (5,), "c8"), ("MagneticField", (5, 2), "f4"),
            ("GFM_Metals", (5, 9), "f4"), ("GFM_MetalsTagged", (5, 7), "f4")
        ]:
            with self.subTest(name=name, dims=dims, dtype=dtype):
                self.samples[0][name] = np.ones(dims, dtype=dtype)
                with self.assertRaisesRegex(RemoteDataError, "Invalid cutout data for gas." + name):
                    source.particles("Subhalo", 0, {"gas": [name]}, counts=counts)
                self.assertFalse((source.root / "particle-rows.sqlite3").exists())
        self.assertFalse(any(path.startswith("snapshot-") for path, _ in self.calls))

    def test_cutout_accepts_run_specific_particle_fields(self):
        from AnastrisTNG.TNGapi import ApiSource
        counts = self.catalog["Subhalo"]["SubhaloLenType"][0]
        for run, fam, name, width in [
            ("Illustris-3", "star", "GFM_Metals", 9),
            ("TNG50-1", "gas", "GFM_Metals", 10),
            ("TNG50-1", "gas", "GFM_MetalsTagged", 6),
            ("TNG50-1", "star", "GFM_StellarPhotometrics", 8),
            ("TNG-Cluster", "bh", "BH_WindTimes", 5)
        ]:
            with self.subTest(run=run, name=name):
                source = ApiSource(run, 99, self.cache, None, "test", False, False)
                source._store_object_value("Subhalo", 0, "particle-offsets", json.dumps([0] * 6))
                values = np.arange(width * 5, dtype="f4").reshape(5, width)
                self.samples[PART_TYPE[fam]][name] = values
                result = source.particles("Subhalo", 0, {fam: [name]}, counts=counts)
                np.testing.assert_array_equal(result[fam][name], values[:2])
                source.offline = True
                replay = source.particles("Subhalo", 0, {fam: [name]}, counts=counts)
                np.testing.assert_array_equal(replay[fam][name], values[:2])

    def test_partial_particle_hit_decompresses_existing_rows_once(self):
        source = self.remote(api_key="test")._source
        fields = {"gas": ["Density"]}
        source.particles("Subhalo", 0, fields)
        with patch.object(ParticleRowCache, "_unpack", wraps=ParticleRowCache._unpack) as unpack:
            result = source.particles("Halo", 0, fields)
        np.testing.assert_array_equal(result["gas"]["Density"], self.samples[0]["Density"][:3])
        self.assertEqual(unpack.call_count, 1)

    def test_large_particle_cache_decodes_independent_blocks_concurrently(self):
        from threading import Barrier
        cache = ParticleRowCache(self.cache / "particle-rows.sqlite3")
        cache.block_bytes = 256 * 1024
        key = ("star", "Coordinates")
        coords = np.random.default_rng(3).normal(size=(40000, 3)).astype(">f4")
        cache.store({key: [(0, coords)]})
        rendezvous = Barrier(2)
        decode = cache._unpack

        def decode_block(*args):
            rendezvous.wait(timeout=2)
            return decode(*args)

        with patch.object(cache, "_unpack", side_effect=decode_block):
            pieces = cache.read({key: (5, len(coords)-3)})[key]
        source = self.remote(api_key="test")._source
        result = source._assemble_particle_rows(key, (5, len(coords)-3), pieces)
        self.assertEqual(result.dtype, coords.dtype)
        np.testing.assert_array_equal(result, coords[5:-3])

    def test_stale_damage_report_preserves_another_readers_repaired_rows(self):
        import sqlite3
        from contextlib import closing
        cache = ParticleRowCache(self.cache / "particle-rows.sqlite3")
        key = ("gas", "Density")
        values = np.arange(10, dtype="f4")
        cache.store({key: [(0, values)]})
        with closing(sqlite3.connect(cache.path)) as db, db:
            db.execute("UPDATE particle_rows SET compressed=1, payload=x'00'")
        damaged = set()
        self.assertEqual(cache.read({key: (4, 7)}, damaged=damaged)[key], [])
        # The second reader acquired its report before the first transaction committed.
        cache.store({key: [(0, values)]}, damaged=damaged)
        cache.store({key: [(4, values[4:7])]}, damaged=damaged)
        pieces = cache.read({key: (0, len(values))})[key]
        self.assertEqual(len(pieces), 1)
        np.testing.assert_array_equal(pieces[0][1], values)

    def test_partial_host_keeps_integer_ids_from_unsigned_mirror_and_signed_cutout(self):
        # The public snapshot uses uint64; some official cutouts encode IDs as int64.
        path = self.local / "snapdir_099/snap_099.1.hdf5"
        with h5py.File(path, "r+") as file:
            del file["PartType4/ParticleIDs"]
        self.samples[4]["ParticleIDs"] = self.samples[4]["ParticleIDs"].astype("i8")
        snap = self.remote(local_dir=self.local, api_key="test")
        ids = snap.read_particle_field(0, "star", "iord", groupType="Halo")
        self.assertEqual(ids.dtype, np.dtype("uint64"))
        np.testing.assert_array_equal(ids, self.samples[4]["ParticleIDs"][:3].astype("u8"))

    def test_same_length_corruption_is_detected_and_legacy_raw_rows_still_read(self):
        import sqlite3
        from contextlib import closing

        source = self.remote(api_key="test")._source
        fields = {"gas": ["Density"]}
        values = source.particles("Subhalo", 1, fields)["gas"]["Density"]
        db_path = source.root / "particle-rows.sqlite3"
        with closing(sqlite3.connect(db_path)) as db, db:
            payload = db.execute("SELECT payload FROM particle_rows").fetchone()[0]
            db.execute("UPDATE particle_rows SET payload=?", (payload[:-1] + bytes([payload[-1] ^ 1]),))
        offline = self.remote(offline=True)._source
        with self.assertRaisesRegex(RemoteDataError, "gas.Density"):
            offline.particles("Subhalo", 1, fields)
        np.testing.assert_array_equal(source.particles("Subhalo", 1, fields)["gas"]["Density"], values)
        with closing(sqlite3.connect(db_path)) as db, db:
            db.execute("UPDATE particle_rows SET compressed=0, payload=?", (values.tobytes(),))
        np.testing.assert_array_equal(offline.particles("Subhalo", 1, fields)["gas"]["Density"], values)

    def test_damaged_particle_blocks_repair_from_cutout_and_mirror(self):
        import sqlite3
        from contextlib import closing
        source = self.remote(api_key="test")._source
        fields = {"gas": ["Density"]}
        with patch.object(ParticleRowCache, "block_bytes", 4):
            source.particles("Halo", 0, fields)
        db_path = source.root / "particle-rows.sqlite3"
        good_sql = "SELECT * FROM particle_rows WHERE start != 1 ORDER BY start"
        with closing(sqlite3.connect(db_path)) as db, db:
            adjacent = db.execute(good_sql).fetchall()
            self.assertEqual(len(adjacent), 2)
            # Older releases accepted a two-component density if it had the right row count.
            db.execute("UPDATE particle_rows SET shape='[2]', compressed=0, payload=? WHERE start=1",
                       (np.array([-1., -2.], dtype="f4").tobytes(),))
        self.calls.clear()
        offline = self.remote(offline=True)._source
        with self.assertRaisesRegex(RemoteDataError, r"gas\.Density.*\[1, ?2\)"):
            offline.particles("Halo", 0, fields)
        self.assertFalse(self.calls)
        online = source.particles("Halo", 0, fields)
        np.testing.assert_array_equal(online["gas"]["Density"], self.samples[0]["Density"][:3])
        self.assertEqual(self.calls, [("halos/0/cutout.hdf5", {"gas": ["Density"]})])
        with closing(sqlite3.connect(db_path)) as db, db:
            self.assertEqual(db.execute(good_sql).fetchall(), adjacent)
            db.execute("UPDATE particle_rows SET compressed=1, payload=x'00' WHERE start=1")
        self.calls.clear()
        mirror = self.remote(local_dir=self.local, offline=True)._source
        np.testing.assert_array_equal(mirror.particles("Halo", 0, fields)["gas"]["Density"],
                                      self.samples[0]["Density"][:3])
        replay = offline.particles("Halo", 0, fields)
        np.testing.assert_array_equal(replay["gas"]["Density"], self.samples[0]["Density"][:3])
        with closing(sqlite3.connect(db_path)) as db:
            self.assertEqual(db.execute(good_sql).fetchall(), adjacent)
        self.assertFalse(self.calls)
        with closing(sqlite3.connect(db_path)) as db, db:
            db.execute("UPDATE particle_rows SET stop=start WHERE start=0")
        repaired = source.particles("Halo", 0, fields)
        np.testing.assert_array_equal(repaired["gas"]["Density"], self.samples[0]["Density"][:3])
        np.testing.assert_array_equal(offline.particles("Halo", 0, fields)["gas"]["Density"],
                                      repaired["gas"]["Density"])

    def test_cold_cutout_overlaps_offsets_and_concurrent_reads_share_transfer(self):
        from threading import Event
        source = self.remote(api_key="test")._source
        counts = source.particle_counts("Subhalo", 0)
        offset_started, cutout_started = Event(), Event()

        def wait_for_peer(request, timeout):
            path = urlsplit(request.full_url).path
            if path.endswith("/cutout.hdf5"):
                cutout_started.set()
                self.assertTrue(offset_started.wait(3), "offset request serialized behind the cutout")
            elif path.endswith("/offsets.99.hdf5"):
                offset_started.set()
                self.assertTrue(cutout_started.wait(3), "cutout serialized behind offsets")
            return self.respond(request, timeout)

        with patch("AnastrisTNG.TNGapi.urlopen", side_effect=wait_for_peer), ThreadPoolExecutor(2) as pool:
            reads = [pool.submit(source.particles, "Subhalo", 0, {"gas": ["Density"]}, counts=counts)
                     for _ in range(2)]
            for read in reads:
                np.testing.assert_array_equal(read.result()["gas"]["Density"], self.samples[0]["Density"][:2])
        self.assertEqual(sum(path.endswith("cutout.hdf5") for path, _ in self.calls), 1)
        self.assertEqual(sum(path == "offsets.99.hdf5" for path, _ in self.calls), 1)
        self.assertFalse(any(path.startswith("snapshot-") for path, _ in self.calls))

    def test_standalone_cutout_survives_unavailable_offsets_without_shared_rows(self):
        source = self.remote(api_key="test")._source
        counts = source.particle_counts("Subhalo", 0)
        fields = {"gas": ["Density"]}

        def unavailable(request, timeout):
            if urlsplit(request.full_url).path.endswith("/offsets.99.hdf5"):
                raise HTTPError(request.full_url, 503, "Offsets unavailable", {}, None)
            return self.respond(request, timeout)

        with patch("AnastrisTNG.TNGapi.urlopen", side_effect=unavailable):
            result = source.particles("Subhalo", 0, fields, counts=counts)
        np.testing.assert_array_equal(result["gas"]["Density"], self.samples[0]["Density"][:2])
        self.assertFalse((source.root / "particle-rows.sqlite3").exists())
        self.assertIsNone(source._object_json("Subhalo", 0, "particle-offsets"))
        self.assertFalse(any(path.startswith("snapshot-") for path, _ in self.calls))
        self.assertFalse(list(source.root.rglob("*.part")))

    def test_offset_authentication_and_bad_metadata_still_surface(self):
        source = self.remote(api_key="test")._source
        counts = source.particle_counts("Subhalo", 0)
        fields = {"gas": ["Density"]}

        def deny(request, timeout):
            if urlsplit(request.full_url).path.endswith("/offsets.99.hdf5"):
                raise HTTPError(request.full_url, 403, "Forbidden", {}, None)
            return self.respond(request, timeout)

        with patch("AnastrisTNG.TNGapi.urlopen", side_effect=deny):
            with self.assertRaisesRegex(RemoteDataError, "HTTP Error 403"):
                source.particles("Subhalo", 0, fields, counts=counts)
        with io.BytesIO(self.archive["offsets.99.hdf5"]) as content:
            with h5py.File(content, "r+") as file:
                file["Subhalo/SnapByType"][0, 0] = -1
            self.archive["offsets.99.hdf5"] = content.getvalue()
        with self.assertRaisesRegex(RemoteDataError, "Invalid snapshot particle offsets"):
            source.particles("Subhalo", 0, fields, counts=counts)
        self.assertIsNone(source._object_json("Subhalo", 0, "particle-offsets"))
        self.assertFalse((source.root / "particle-rows.sqlite3").exists())

    def test_uncached_subhalo_offsets_still_reuse_host_columns(self):
        source = self.remote(api_key="test")._source
        source.particles("Halo", 0, {"gas": ["Density"]})
        self.calls.clear()
        result = source.particles("Subhalo", 0, {"gas": ["Density", "Potential"]})
        for name in ("Density", "Potential"):
            np.testing.assert_array_equal(result["gas"][name], self.samples[0][name][:2])
        self.assertEqual([call for call in self.calls if call[0].endswith("cutout.hdf5")],
                         [("subhalos/0/cutout.hdf5", {"gas": ["Potential"]})])

    def test_offline_missing_offsets_identify_object_relative_rows(self):
        source = self.remote(api_key="test")._source
        counts = source.particle_counts("Subhalo", 0)
        self.calls.clear()
        source.offline = True
        with self.assertRaisesRegex(RemoteDataError, r"gas\.Density object rows \[0, 2\)"):
            source.particles("Subhalo", 0, {"gas": ["Density"]}, counts=counts)
        self.assertFalse(self.calls)

    def test_invalid_mirror_column_can_be_filled_from_cutout(self):
        path = self.local / "snapdir_099/snap_099.0.hdf5"
        with h5py.File(path, "r+") as file:
            del file["PartType0/Density"]
            file["PartType0/Density"] = np.ones((2, 2), dtype="f4")
        snap = self.remote(local_dir=self.local, api_key="test")
        np.testing.assert_array_equal(snap.read_particle_field(0, "gas", "rho"), self.samples[0]["Density"][:2])
        self.assertEqual(self.calls, [("subhalos/0/cutout.hdf5", {"gas": ["Density"]})])

    def test_particle_progress_reports_one_object_and_all_transfers(self):
        snap = self.remote(api_key="test")
        source = snap._source
        source.catalog("Halo", 0)
        source.progress = True
        self.calls.clear()
        with patch("AnastrisTNG.TNGapi.tqdm") as progress:
            data = source.particles("Halo", 0, {"star": ["Coordinates"], "dm": ["Coordinates"]})
            progress.assert_called_once()
            kwargs = progress.call_args.kwargs
            for label in ["TNG50-1", "099", "Halo 0", "dm.Coordinates", "star.Coordinates"]:
                self.assertIn(label, kwargs["desc"])
            bar = progress.return_value.__enter__.return_value
            self.assertEqual(bar.total, sum(call.args[0] for call in bar.update.call_args_list))
            self.assertTrue(bar.set_description_str.call_args.args[0].startswith("[Cutout | offsets via Range] "))
        np.testing.assert_array_equal(data["star"]["Coordinates"], self.samples[4]["Coordinates"][:3])
        with patch("AnastrisTNG.TNGapi.tqdm") as progress:
            source.particles("Halo", 0, {"star": ["Coordinates"], "dm": ["Coordinates"]})
            progress.assert_not_called()

    def test_cross_shard_halo_field_uses_one_cutout(self):
        snap = self.remote(api_key="test")
        data = snap.read_particle_field(0, "star", "pos", groupType="Halo")
        np.testing.assert_array_equal(data, self.samples[4]["Coordinates"][:3])
        self.assertEqual([call for call in self.calls if call[0].endswith("cutout.hdf5")],
                         [("halos/0/cutout.hdf5", {"stars": ["Coordinates"]})])
        self.assertFalse(any(path.startswith("snapshot-") for path, _ in self.calls))

    def test_snapshot_header_overrides_catalog_and_run_defaults(self):
        path = self.local / "snapdir_099/snap_099.0.hdf5"
        with h5py.File(path, "r+") as f:
            f["Header"].attrs.update(Time=.25, HubbleParam=.72, Omega0=.33, OmegaLambda=.67,
                                     OmegaBaryon=.052, BoxSize=33000., MassTable=[0, .0123456789, 0, 0, 0, 0])
            f["Parameters"].attrs.update(Sigma8=.89, PrimordialIndex=.97, SofteningComovingType2=1.2,
                                         SofteningMaxPhysType2=.4)
        for snap in (Snapshot(str(self.local), 99), self.remote(local_dir=self.local, offline=True)):
            p = snap.properties
            self.assertEqual([p[k] for k in ("a", "h", "omegaM0", "omegaL0", "omegaB0", "sigma8", "ns")],
                             [.25, .72, .33, .67, .052, .89, .97])
            self.assertAlmostEqual(float(p["boxsize"].in_units("a kpc h^-1")), 33000.)
            self.assertAlmostEqual(float(p["eps"].in_units("kpc h^-1")), .3)
            np.testing.assert_array_equal(snap.load_particle(0).dm["mass"], [.0123456789] * 2)
            self.assertEqual(p["Halos_total"], 2)
        self.assertFalse(self.calls)

    def test_mirror_run_is_checked_before_using_the_cache(self):
        with self.assertRaisesRegex(ValueError, "TNG100-1"):
            Snapshot.from_api("TNG100-1", 99, local_dir=self.local, cache_dir=self.cache, progress=False)
        anonymous = self.local.parent.rename(self.local.parent.with_name("mirror")) / "output"
        # The DM mass identifies the wrong resolution even in a renamed mirror with the same box size.
        with self.assertRaisesRegex(ValueError, "TNG50-2"):
            Snapshot.from_api("TNG50-2", 99, local_dir=anonymous, cache_dir=self.cache, progress=False)
        snap = self.remote(local_dir=anonymous, offline=True)
        self.assertEqual(float(snap.properties["Mdm"]), self.dm_mass)
        shutil.rmtree(anonymous / "snapdir_099")
        group = anonymous / "groups_099/groups_099.0.hdf5"
        group.rename(group.with_name("groups_099.1.hdf5"))
        # A group catalog alone identifies the box, independently of particle headers.
        with self.assertRaisesRegex(ValueError, "TNG100-1"):
            Snapshot.from_api("TNG100-1", 99, local_dir=anonymous, cache_dir=self.cache, progress=False)
        with self.assertWarnsRegex(RuntimeWarning, "published run-table mass"):
            partial = self.remote(local_dir=anonymous, offline=True)
        self.assertEqual(partial.properties["Halos_total"], 2)
        self.assertFalse(self.calls)
        self.assertFalse(self.cache.exists())

    def test_local_snapshot_identity_and_cached_catalog_counts_are_checked(self):
        group = self.local / "groups_099/groups_099.0.hdf5"
        with h5py.File(group, "r+") as file:
            file["Header"].attrs["SnapNum"] = 98
        with self.assertRaisesRegex(ValueError, "98"):
            self.remote(local_dir=self.local, offline=True)
        with h5py.File(group, "r+") as file:
            del file["Header"].attrs["SnapNum"]
            file["Header"].attrs["Ngroups_Total"] = 3
        self.remote(api_key="test")
        self.calls.clear()
        with self.assertRaisesRegex(ValueError, "Ngroups_Total"):
            self.remote(local_dir=self.local, offline=True)
        self.assertFalse(self.calls)

    def test_cold_headers_are_concurrent_and_replay_with_hdf5_precision(self):
        from threading import Barrier, BrokenBarrierError

        both_requests = Barrier(2)
        def respond(request, timeout):
            try:
                both_requests.wait(timeout=5)
            except BrokenBarrierError:
                pass
            return self.respond(request, timeout)

        with patch("AnastrisTNG.TNGapi.urlopen", side_effect=respond):
            online = self.remote(api_key="test")
        self.assertFalse(both_requests.broken, "the two cold metadata requests ran sequentially")
        self.calls.clear()
        local = Snapshot(str(self.local), 99)
        for snap in (online, self.remote(offline=True)):
            for name in ("a", "h"):
                self.assertIs(type(snap.properties[name]), type(local.properties[name]))
        self.assertFalse(self.calls)

    def test_partial_mirror_and_legacy_cache_use_run_metadata(self):
        old = dict(Time=1., HubbleParam=.5, Omega0=.4, OmegaLambda=.6, BoxSize=1.,
                   Ngroups_Total=2, Nsubgroups_Total=2)
        directory = self.cache / "TNG50-1/snap_099"
        directory.mkdir(parents=True)
        (directory / "header.json").write_text(json.dumps(old))
        with self.assertWarnsRegex(RuntimeWarning, "published run-table mass"):
            legacy = self.remote(offline=True)
        self.assertEqual(float(legacy.properties["Mdm"]), 0.0000307367708626464)
        snap = self.remote(api_key="test")
        self.assertEqual(float(snap.properties["Mdm"]), self.dm_mass)
        self.assertAlmostEqual(snap.properties["h"], .6774)
        with h5py.File(self.local / "snapdir_099/snap_099.0.hdf5", "r+") as f:
            del f["Header"].attrs["MassTable"]
            f["Header"].attrs["OmegaBaryon"] = .05
        mixed = self.remote(local_dir=self.local, offline=True)
        self.assertEqual(mixed.properties["omegaB0"], .05)
        np.testing.assert_array_equal(mixed.read_particle_field(0, "dm", "mass"), [self.dm_mass] * 2)
        # Run parameters remain available from any local snapshot chunk.
        (self.local / "snapdir_099/snap_099.0.hdf5").unlink()
        with h5py.File(self.local / "snapdir_099/snap_099.1.hdf5", "r+") as f:
            f["Header"].attrs["MassTable"] = [0, .1234, 0, 0, 0, 0]
        self.assertEqual(float(self.remote(local_dir=self.local, offline=True).properties["Mdm"]), .1234)
        self.assertEqual(self.calls, [("simulation", {})])

    def test_missing_metadata_for_illustris_and_cluster(self):
        from types import SimpleNamespace
        from pynbody.simdict import SimDict
        from AnastrisTNG.TNGgroupcat import get_eps_Mdm

        for run, box, mass, eps_at_z0 in [
                ("Illustris-1", 75000, 0.000440896524361096, 1.0),
                ("TNG-Cluster", 680000, 0.00413211373831018, 1.0)]:
            with self.subTest(run=run):
                props = SimDict()
                props["run"] = run
                props["read_Snap_properties"] = dict(Time=1., HubbleParam=.7, Omega0=.3,
                                                     OmegaLambda=.7, Ngroups_Total=1, Nsubgroups_Total=1)
                self.assertEqual(float(props["boxsize"].in_units("a kpc h^-1")), box)
                with self.assertWarnsRegex(RuntimeWarning, "published run-table mass"):
                    eps, mdm = get_eps_Mdm(SimpleNamespace(properties=props))
                self.assertEqual(float(eps.in_units("kpc h^-1")), eps_at_z0)
                self.assertEqual(float(mdm.in_units("1e10 Msol h^-1")), mass)

    def test_original_illustris_embedded_offsets(self):
        from AnastrisTNG.TNGapi import ApiSource
        for run in ("Illustris-1", "Illustris-2", "Illustris-3"):
            source = ApiSource(run, 135, self.cache, None, None, False, True)
            self.assertIn("Volume", source.fields["gas"])
            self.assertNotIn("MagneticField", source.fields["gas"])
        for run, snap in [("Illustris-3", 136), ("TNG50-4", 100), ("Illustris-4", 135)]:
            with self.assertRaises(ValueError):
                ApiSource(run, snap, self.cache, None, None, False, True)
        raw = source.root / "groupcat"
        raw.mkdir(parents=True)
        for chunk in (0, 1):
            with h5py.File(raw / f"groupcat-135.{chunk}.hdf5", "w") as file:
                file.create_group("Header").attrs.update(
                    FileOffsets_Group=[0, 2], FileOffsets_Subhalo=[0, 3], FileOffsets_SubLink=[0, 10],
                    FileOffsets_Snap=np.tile([0, 2], (6, 1)))
                file["Group/GroupLenType"] = np.ones((2, 6), dtype=int)
                file["Group/GroupMass"] = [10 + chunk, 20 + chunk]
                if chunk == 1:
                    file["Offsets/Group_SnapByType"] = np.tile([[2], [3]], (1, 6))
                    file["Offsets/Subhalo_SublinkSubhaloID"] = [99, 200]
                    file["Offsets/Subhalo_SublinkRowNum"] = [-1, 12]
                    file["Offsets/Subhalo_SublinkLastProgenitorID"] = [99, 202]
        with h5py.File(raw / "sublink.1.hdf5", "w") as file:
            file["SubhaloID"] = [0, 1, 200, 201, 202]
            file["MainLeafProgenitorID"] = [1, 1, 201, 201, 202]
            file["SnapNum"] = [135, 131, 135, 127, 129]
        self.assertEqual(source._catalog_row("Group", 3)["GroupMass"], 21)
        fallback = source._catalog_row("Group", 3, download=True)
        self.assertEqual(fallback["GroupMass"], 21)
        np.testing.assert_array_equal(source.catalog("Halo", 3)["GroupLenType"], [1] * 6)
        shutil.copyfile(self.local / "snapdir_099/snap_099.1.hdf5", raw / "snapshot-135.1.hdf5")
        ids = source.particles("Halo", 3, {"dm": ["ParticleIDs"]})["dm"]["ParticleIDs"]
        np.testing.assert_array_equal(ids, self.samples[1]["ParticleIDs"][3:4])
        np.testing.assert_array_equal(source.merger_tree(4, ["SnapNum"], only_mpb=True)["SnapNum"], [135, 127])
        np.testing.assert_array_equal(source.merger_tree(4, ["SnapNum"])["SnapNum"], [135, 127, 129])
        with self.assertRaisesRegex(ValueError, "no SubLink tree"):
            source.merger_tree(3, ["SnapNum"])

    def test_catalog_gateway_timeout_and_offline_reopen(self):
        requests = []
        with io.BytesIO(self.archive["offsets.99.hdf5"]) as content:
            with h5py.File(content, "r+") as file:
                del file["FileOffsets/Subhalo"]
                file["FileOffsets/Subhalo"] = [0, 1]
            self.archive["offsets.99.hdf5"] = content.getvalue()
        with io.BytesIO() as content:
            with h5py.File(content, "w") as file:
                for key, values in self.catalog["Subhalo"].items():
                    file["Subhalo/" + key] = [values[1]]
                file["UnrelatedData"] = np.zeros(256 * 1024, dtype="i4")
            self.archive["groupcat-99.1.hdf5"] = content.getvalue()

        def gateway(request, timeout):
            path = urlsplit(request.full_url).path
            requests.append(path)
            if path.endswith("/info.json"):
                raise HTTPError(request.full_url, 504, "Gateway Time-out", {}, None)
            if "/files/" in path:
                self.assertIsNotNone(request.get_header("Range"), path)
            return self.respond(request, timeout)

        with patch("AnastrisTNG.TNGapi.urlopen", side_effect=gateway):
            snap = self.remote(api_key="test")
            sub = snap.load_particle(1, groupType="Subhalo", decorate=True)
            ids = np.array(sub.gas["iord"])
            density = sub.gas["rho"].copy()
            sub.physical_units()
            self.assertEqual(sub.GC["SubhaloGrNr"], 1)
            self.assertEqual(sum(path.endswith("info.json") for path in requests), 1)
            files = {path.rsplit("/", 1)[-1] for path in requests if "/files/" in path}
            self.assertEqual(files, {"offsets.99.hdf5", "groupcat-99.1.hdf5"})
            transferred = sum(span[1] - span[0] for name, span in self.calls if name == "groupcat-99.1.hdf5")
            self.assertLess(transferred, len(self.archive["groupcat-99.1.hdf5"]))
        with patch("AnastrisTNG.TNGapi.urlopen", side_effect=AssertionError("unexpected HTTP")):
            replay = self.remote(offline=True).load_particle(1)
            np.testing.assert_array_equal(replay.gas["iord"], ids)
            np.testing.assert_array_equal(replay.gas["rho"], density)
            self.assertEqual(replay.GC["SubhaloGrNr"], 1)

    def test_deferred_catalog_reads_only_the_requested_hdf5_cell(self):
        sub = self.remote(api_key="test").load_particle(1)
        sub.physical_units()
        spin = np.array([[4, 2, 1], [25.448586, 178.45984, 331.6787]], dtype="f4")
        with h5py.File(self.local / "groups_099/groups_099.0.hdf5", "r+") as file:
            file["Subhalo/SubhaloSpin"] = spin
        reference = Snapshot(str(self.local), 99).load_particle(1)
        reference.physical_units()
        self.catalog["Subhalo"]["SubhaloSpin"] = spin.tolist()
        with io.BytesIO() as data:
            with h5py.File(data, "w") as file:
                for name, values in self.catalog["Subhalo"].items():
                    file["Subhalo/" + name] = spin if name == "SubhaloSpin" else values
                file["Padding"] = np.zeros(131072, dtype="i4")
                unrelated = file.create_dataset("Subhalo/SubhaloExtra", data=np.zeros((2, 65536), dtype="i4"))
                unused_offset = unrelated.id.get_offset() + unrelated.shape[1] * unrelated.dtype.itemsize
            self.archive["groupcat-99.0.hdf5"] = data.getvalue()

        failed_json = []
        def gateway(request, timeout):
            if urlsplit(request.full_url).path.endswith("info.json"):
                failed_json.append(request.full_url)
                raise HTTPError(request.full_url, 504, "Gateway Timeout", {}, None)
            return self.respond(request, timeout)

        self.calls.clear()
        with patch("AnastrisTNG.TNGapi.urlopen", side_effect=gateway):
            catalog = sub.GC
            self.assertFalse(self.calls)
            result = catalog["SubhaloSpin"]
            self.assertIs(sub.SubhaloSpin, result)
            self.assertEqual(result.dtype, spin.dtype)
            self.assertEqual(str(result.units), "kpc km s**-1")
            np.testing.assert_array_equal(result, reference.GC["SubhaloSpin"])
            self.assertTrue(sub._gc_pending)
            ranges = [span for name, span in self.calls if name == "groupcat-99.0.hdf5"]
            self.assertTrue(ranges)
            self.assertFalse(any(start <= unused_offset < stop for start, stop in ranges))

        with patch("AnastrisTNG.TNGapi.urlopen", side_effect=AssertionError("unexpected HTTP")):
            replay = self.remote(offline=True).load_particle(1)
            replay.physical_units()
            np.testing.assert_array_equal(replay.GC["SubhaloSpin"], result)
            with self.assertRaisesRegex(RemoteDataError, "Offline cache miss"):
                _ = replay.GC["SubhaloExtra"]

        with patch("AnastrisTNG.TNGapi.urlopen", side_effect=gateway):
            result.convert_units("pc km s^-1")
            scaled = result.copy()
            self.assertIn("SubhaloExtra", dict(catalog))
            self.assertIs(catalog["SubhaloSpin"], result)
            self.assertEqual(result.units, scaled.units)
            np.testing.assert_array_equal(result, scaled)
            self.assertFalse(sub._gc_pending)
        self.assertEqual(len(failed_json), 1)

    def test_slow_exact_catalog_response_is_used_without_an_archive_download(self):
        sub = self.remote(api_key="test").load_particle(0)
        self.calls.clear()
        def slow_info(request, timeout):
            if not request.full_url.endswith("subhalos/0/info.json"):
                self.fail("Unexpected request for an archive block")
            if timeout < 32:
                raise TimeoutError("Generated catalogue needs 32 seconds")
            return self.respond(request, timeout)
        with patch("AnastrisTNG.TNGapi.urlopen", side_effect=slow_info):
            self.assertEqual(sub.GC["SubhaloMass"].dtype, np.dtype("float32"))
            sub.GC = sub.GC
            self.assertEqual(dict(sub.GC)["SubhaloGrNr"], 0)
        self.assertEqual(self.calls, [("subhalos/0/info.json", {})])

    def test_rate_limited_archive_does_not_retry_the_json_service(self):
        source = self.remote(api_key="test")._source
        source._archive_catalogs.add(("Subhalo", 1))
        limited = HTTPError(source.url, 429, "Too Many Requests", {"Retry-After": "120"}, None)
        with patch("AnastrisTNG.TNGapi.urlopen", side_effect=limited) as get:
            with self.assertRaisesRegex(RemoteDataError, "429"):
                source.catalog_array("Subhalo", 1, "SubhaloSpin")
        get.assert_called_once()

    def test_catalog_failure_reports_both_services_and_preserves_snapshot(self):
        snap = self.remote(api_key="test")

        def refused(request, timeout):
            path = urlsplit(request.full_url).path
            if path.endswith("/info.json"):
                raise HTTPError(request.full_url, 504, "Gateway Timeout", {}, None)
            if "groupcat-99." in path:
                raise HTTPError("https://data-eu.tng-project.org/download", 403, "Forbidden", {}, None)
            return self.respond(request, timeout)

        with patch("AnastrisTNG.TNGapi.urlopen", side_effect=refused):
            with self.assertRaisesRegex(RemoteDataError, r"subhalos/0.*info.json.*504.*HDF5.*403") as failed:
                snap.load_subhalo(0)
        self.assertIsInstance(failed.exception.__cause__, RemoteDataError)
        self.assertEqual(len(snap), 0)
        snap.load_subhalo(0)
        self.assertEqual(len(np.unique(snap["iord"])), 8)

    def test_mixed_catalog_fallback_replays_without_partial_mirror(self):
        with io.BytesIO() as data:
            with h5py.File(data, "w") as file:
                file["FileOffsets/Subhalo"] = [0, 1]
            self.archive["offsets.99.hdf5"] = data.getvalue()
        with io.BytesIO() as data:
            with h5py.File(data, "w") as file:
                file.create_group("Header")
                file["Subhalo/SubhaloLenType"] = np.zeros((1, 6), dtype="i4")
                file["Subhalo/SubhaloIDMostbound"] = np.array([2**63 + 17], dtype="u8")
                file["Subhalo/SubhaloMass"] = np.array([.5], dtype="f4")
            self.archive["groupcat-99.1.hdf5"] = data.getvalue()

        def gateway(request, timeout):
            if urlsplit(request.full_url).path.endswith("/info.json"):
                raise HTTPError(request.full_url, 504, "Gateway timeout", {}, None)
            return self.respond(request, timeout)

        for product, mirror_path in (("offsets.99.hdf5", "../postprocessing/offsets/offsets_099.hdf5"),
                                     ("groupcat-99.1.hdf5", "groups_099/fof_subhalo_tab_099.1.hdf5")):
            with self.subTest(product=product):
                mirror = self.cache.parent / product / "TNG50-1/output"
                mirror.mkdir(parents=True)
                local_file = mirror / mirror_path
                local_file.parent.mkdir(parents=True, exist_ok=True)
                local_file.write_bytes(self.archive[product])
                options = dict(cache_dir=self.cache / product, progress=False)
                with patch("AnastrisTNG.TNGapi.urlopen", side_effect=gateway):
                    source = Snapshot.from_api("TNG50-1", 99, local_dir=mirror, api_key="test", **options)._source
                    row = source.catalog("Subhalo", 1)
                with patch("AnastrisTNG.TNGapi.urlopen", side_effect=AssertionError("unexpected HTTP")):
                    offline = Snapshot.from_api("TNG50-1", 99, offline=True, **options)._source
                    copy = offline.catalog("Subhalo", 1)
                self.assertEqual(copy["SubhaloIDMostbound"], 2**63 + 17)
                for key in row:
                    self.assertEqual(copy[key].dtype, row[key].dtype)
                    np.testing.assert_array_equal(copy[key], row[key])

    def test_full_catalog_row_batches_required_ranges_and_reuses_cached_blocks(self):
        from threading import Lock
        from time import sleep
        from AnastrisTNG.TNGapi import ApiSource

        count, number = 16384, 8192
        filename = "groupcat-99.0.hdf5"
        with io.BytesIO() as data:
            with h5py.File(data, "w") as file:
                file["Subhalo/SubhaloLenType"] = np.ones((count, 6), dtype="i4")
                for batch in range(6):
                    for slot in range(4):
                        file[f"Subhalo/SubhaloTest{batch}{slot}"] = np.arange(count, dtype="u4") + batch * 10 + slot
                    file[f"OtherData/{batch}"] = np.zeros(4 * count, dtype="i4")
            self.archive[filename] = data.getvalue()
        active = peak = 0
        ranges = []
        guard = Lock()
        change_version = changed = False

        def serve(request, timeout):
            nonlocal active, peak, changed
            path = urlsplit(request.full_url).path
            if path.endswith("info.json"):
                raise HTTPError(request.full_url, 504, "Unavailable", {}, None)
            if not path.endswith(filename):
                return self.respond(request, timeout)
            with guard:
                active += 1
                peak = max(peak, active)
                if change_version and active > 1:
                    changed = True
            try:
                sleep(.008)
                response = self.respond(request, timeout)
                first, last = map(int, request.get_header("Range")[6:].split("-"))
                ranges.append((first, min(last + 1, len(self.archive[filename]))))
                if changed:
                    response.headers["ETag"] = f'"revision-{first}"'
                return response
            finally:
                with guard:
                    active -= 1

        def source(directory, offline=False):
            return ApiSource("TNG50-1", 99, self.cache / directory, None, "test", False, offline)

        def blocks(spans):
            return [offset for start, stop in spans for offset in range(start, stop, 65536)]

        with patch("AnastrisTNG.TNGapi.urlopen", side_effect=serve):
            with source("serial")._raw_groupcat(0) as file:
                expected = {name: dataset[number] for name, dataset in file["Subhalo"].items()}
            baseline = list(ranges)
            ranges.clear()
            online = source("shared")
            with online._raw_groupcat(0) as file:
                file["Subhalo/SubhaloTest00"][number]
            initial = list(ranges)
            ranges.clear()
            result = online.catalog_arrays("Subhalo", number)
        self.assertGreaterEqual(peak, 2)
        self.assertLessEqual(peak, 3)
        self.assertLess(len(initial) + len(ranges), len(baseline))
        self.assertCountEqual(blocks(initial + ranges), blocks(baseline))
        with patch("AnastrisTNG.TNGapi.urlopen", side_effect=AssertionError("unexpected HTTP")):
            offline = source("shared", offline=True).catalog_arrays("Subhalo", number)
        for key, value in expected.items():
            for returned in (result, offline):
                self.assertEqual(returned[key].dtype, value.dtype)
                np.testing.assert_array_equal(returned[key], value)
        # The first validators can arrive together, after an untagged header.
        # A batch from conflicting revisions must never become a cached row.
        change_version = True
        with patch("AnastrisTNG.TNGapi.urlopen", side_effect=serve):
            with self.assertRaisesRegex(RemoteDataError, "Source file changed"):
                source("revision").catalog_arrays("Subhalo", number)
        with patch("AnastrisTNG.TNGapi.urlopen", side_effect=AssertionError("unexpected HTTP")):
            with self.assertRaisesRegex(RemoteDataError, "Offline cache miss"):
                source("revision", offline=True).catalog_arrays("Subhalo", number)

    def test_flat_catalog_and_invalid_api_key(self):
        snap = self.remote(api_key="test")
        fields = {k: values[0] for k, values in self.catalog["Group"].items()}
        fields["GroupExtraColumn"] = [1, 2]
        with patch("AnastrisTNG.TNGapi.urlopen", return_value=Response(json.dumps(fields).encode())):
            self.assertEqual(snap._source.catalog("Halo", 0), fields)
            raw = snap._source.catalog_arrays("Halo", 0)["GroupExtraColumn"]
            self.assertIs(type(raw), np.ndarray)
            np.testing.assert_array_equal(raw, [1, 2])
        error = HTTPError("https://www.tng-project.org/", 403, "Forbidden", {}, None)
        with patch("AnastrisTNG.TNGapi.urlopen", side_effect=error) as get:
            with self.assertRaisesRegex(RemoteDataError, "403"):
                snap._source.catalog("Subhalo", 0)
            get.assert_called_once()

    def test_json_catalog_native_dtypes_live_and_cached(self):
        high_id = 2**63 + 17
        self.catalog["Subhalo"].update(SubhaloIDMostbound=[high_id, 1], SubhaloFlag=[1, 0],
                                       SubhaloPos=[[.125, .25, .5]] * 2, SubhaloExtra=[2**39, 3])
        source = self.remote(api_key="test")
        for snap in (source, self.remote(offline=True)):
            sub = snap.load_particle(0)
            for name, expected in (("SubhaloIDMostbound", "uint64"), ("SubhaloFlag", "bool"),
                                   ("SubhaloGrNr", "int32"), ("SubhaloLenType", "int32"),
                                   ("SubhaloMass", "float32"), ("SubhaloPos", "float32"),
                                   ("SubhaloExtra", "int64")):
                with self.subTest(field=name):
                    self.assertEqual(sub.GC[name].dtype, np.dtype(expected))
            self.assertEqual(int(sub.GC["SubhaloIDMostbound"]), high_id)
            halo = snap.load_particle(0, groupType="Halo")
            self.assertEqual(halo.GC["GroupFirstSub"].dtype, np.dtype("int32"))
            row = snap._source.halo_values(0, ["GroupLenType", "GroupMass"])
            self.assertEqual(np.asarray(row["GroupLenType"]).dtype, np.dtype("int32"))
            self.assertEqual(np.asarray(row["GroupMass"]).dtype, np.dtype("float32"))
        # Native HDF5 data, including deliberate differences from the public schema, keeps its storage dtype.
        mirrored = self.remote(local_dir=self.local, offline=True).load_particle(0)
        self.assertEqual(mirrored.GC["SubhaloMass"].dtype, np.dtype("float64"))

    def test_original_and_cluster_catalog_schemas_and_sentinels(self):
        from AnastrisTNG.TNGapi import ApiSource

        # Existing row caches contain JSON without a dtype table.
        cases = [
            ("Illustris-3", -1, "uint32", 2**32 - 1, "float64"),
            ("TNG50-1", 2**32 - 1, "int32", -1, "float32"),
            ("TNG-Cluster", -1, "int32", -1, "float32"),
        ]
        for run, sentinel, counter_type, stored_sentinel, metal_type in cases:
            source = ApiSource(run, 99, self.cache, None, None, False, True)
            group = dict(GroupLenType=[0] * 6, GroupFirstSub=sentinel)
            if run == "TNG-Cluster":
                group.update(GroupPrimaryZoomTarget=2, GroupOffsetType=[2**39] * 6)
            source._store_object_value("Halo", 0, "catalog", json.dumps(group), local=True)
            sub = dict(SubhaloGasMetallicityHalfRad=.1234567890123, SubhaloLenType=[0] * 6,
                       SubhaloIDMostbound=2**53 + 7, SubhaloExtraIDs=[2**63 + 9, 1])
            source._store_object_value("Subhalo", 0, "catalog", json.dumps(sub), local=True)
            with self.subTest(run=run):
                halo = source.catalog_arrays("Halo", 0)
                self.assertEqual(halo["GroupFirstSub"].dtype, np.dtype(counter_type))
                self.assertEqual(int(halo["GroupFirstSub"]), stored_sentinel)
                if run == "TNG-Cluster":
                    self.assertEqual(halo["GroupPrimaryZoomTarget"].dtype, np.dtype("int32"))
                    self.assertEqual(halo["GroupPrimaryZoomTarget"], 2)
                    self.assertEqual(halo["GroupOffsetType"].dtype, np.dtype("int64"))
                sub = source.catalog_arrays("Subhalo", 0)
                self.assertEqual(sub["SubhaloGasMetallicityHalfRad"].dtype, np.dtype(metal_type))
                self.assertEqual(sub["SubhaloLenType"].dtype, np.dtype(counter_type))
                self.assertEqual(sub["SubhaloIDMostbound"].dtype, np.dtype("uint64"))
                self.assertEqual(int(sub["SubhaloIDMostbound"]), 2**53 + 7)
                self.assertEqual(int(sub["SubhaloExtraIDs"][0]), 2**63 + 9)
        self.assertFalse(self.calls)
        cluster = ApiSource("TNG-Cluster", 99, self.cache, None, "test", False, False)
        flat = dict(GroupLenType=[0] * 6, TracerLengthType=[3] * 6, TracerOffsetType=[2**40] * 6)
        with patch("AnastrisTNG.TNGapi.urlopen", return_value=Response(json.dumps(flat).encode())):
            halo = cluster.catalog_arrays("Halo", 1)
        self.assertEqual(halo["TracerLengthType"].dtype, np.dtype("int32"))
        self.assertEqual(halo["TracerOffsetType"].dtype, np.dtype("int64"))

    def test_redirected_403_reacquires_location_from_api(self):
        snap = self.remote(api_key="test")
        original = []
        def expired_once(req, timeout):
            original.append((req.full_url, req.get_header("Cache-control")))
            if len(original) == 1:
                raise HTTPError("https://data-eu.tng-project.org/download", 403, "Forbidden", {}, None)
            return self.respond(req, timeout)
        with patch("AnastrisTNG.TNGapi.urlopen", side_effect=expired_once):
            _ = snap._source.catalog("Subhalo", 0)
            sub = snap.load_particle(0)
            self.assertGreater(len(sub.gas["rho"]), 0)
        endpoint = snap._source.url + "subhalos/0/info.json"
        self.assertEqual(original[:2], [(endpoint, None), (endpoint, "no-cache")])

    def test_slow_gateway_does_not_restart_a_long_cutout_wait(self):
        source = self.remote(api_key="test")._source
        elapsed = [0.0]

        def timeout(request, timeout):
            elapsed[0] += 65
            raise HTTPError("https://data-eu.tng-project.org/download", 504, "Gateway Timeout", {}, None)

        with patch("AnastrisTNG.TNGapi.time.monotonic", side_effect=lambda: elapsed[0]), \
                patch("time.sleep") as sleep, patch("AnastrisTNG.TNGapi.urlopen", side_effect=timeout) as get:
            with self.assertRaisesRegex(RemoteDataError, "star mass.*504"):
                source._open("subhalos/0/cutout.hdf5?stars=Masses", "star mass")
        self.assertEqual(get.call_count, 1)
        sleep.assert_not_called()

    def test_transient_metadata_retries_and_rate_limit_budget(self):
        source = self.remote(api_key="test")._source
        for failure in (HTTPError(source.url, 429, "Busy", {"Retry-After": "1"}, None),
                        HTTPError(source.url, 503, "Unavailable", {}, None),
                        URLError(TimeoutError("read timeout")), ConnectionResetError("reset")):
            with self.subTest(failure=failure), patch("time.sleep") as pause:
                with patch("AnastrisTNG.TNGapi.urlopen",
                           side_effect=[failure, Response(b'{"ok": 1}')]) as get:
                    result = source._fetch_json("", "test snapshot", lambda value: value["ok"])
                self.assertEqual(result, 1)
                self.assertEqual(get.call_count, 2)
                pause.assert_called_once()
                if isinstance(failure, HTTPError) and failure.code == 429:
                    self.assertEqual(pause.call_args.args, (1,))
        limited = HTTPError(source.url, 429, "Busy", {"Retry-After": "3600"}, None)
        with patch("AnastrisTNG.TNGapi.urlopen", side_effect=limited) as get, patch("time.sleep") as pause:
            with self.assertRaisesRegex(RemoteDataError, "test snapshot.*429") as result:
                source._fetch_json("", "test snapshot", lambda value: value)
            self.assertIs(result.exception.__cause__, limited)
            get.assert_called_once()
            pause.assert_not_called()

        partial = Response(b'{"ok":')
        partial.headers["Content-Length"] = "100"
        with patch("AnastrisTNG.TNGapi.urlopen", side_effect=[partial, Response(b'{"ok": 1}')]), patch("time.sleep"):
            self.assertEqual(source._fetch_json("", "test snapshot", lambda data: data["ok"]), 1)

    def test_interrupted_signed_range_retries_and_reauthenticates(self):
        source = self.remote(api_key="test")._source
        filename = "sublink.37.hdf5"
        with io.BytesIO() as content:
            with h5py.File(content, "w") as file:
                file["SubhaloID"] = np.arange(150000, dtype="i8")
            self.archive[filename] = content.getvalue()
        signed = "https://data-eu.tng-project.org/download/test"
        direct = []
        api = []

        def serve(request, timeout):
            span = request.get_header("Range")
            if request.full_url == signed:
                self.assertIsNone(request.get_header("Api-key"))
                direct.append(span)
                if len(direct) == 1:
                    raise ConnectionResetError("interrupted data host")
                raise HTTPError(signed, 403, "Expired", {}, None)
            api.append((span, request.get_header("Cache-control")))
            self.assertEqual(request.get_header("Api-key"), "test")
            response = self.respond(request, timeout)
            response.geturl = lambda: request.full_url if direct else signed
            return response

        with patch("AnastrisTNG.TNGapi.urlopen", side_effect=serve), patch("time.sleep") as pause:
            with source._raw_file(filename) as file:
                self.assertEqual(file["SubhaloID"][100000], 100000)
        self.assertEqual(len(direct), 2)
        self.assertEqual(direct[0], direct[1])
        self.assertIn((direct[0], "no-cache"), api)
        pause.assert_called_once()

    def test_repeated_download_host_403_uses_authenticated_groupcat(self):
        snap = self.remote(api_key="test")
        paths = []
        self.archive["groupcat-99.0.hdf5"] = (self.local / "groups_099/groups_099.0.hdf5").read_bytes()
        def download_refused(req, timeout):
            path = urlsplit(req.full_url).path
            paths.append(path)
            if path.endswith("info.json"):
                raise HTTPError("https://data-eu.tng-project.org/download", 403, "Forbidden", {}, None)
            if path.endswith("/files/groupcat-99.0.hdf5"):
                self.assertEqual(req.get_header("Api-key"), "test")
                self.assertIsNotNone(req.get_header("Range"))
            return self.respond(req, timeout)
        with patch("AnastrisTNG.TNGapi.urlopen", side_effect=download_refused):
            sub = snap.load_particle(0)
            expected = np.array(sub.gas["rho"])
            sub.physical_units()
            self.assertEqual(sub.GC["SubhaloGrNr"], 0)
        self.assertEqual(sum(p.endswith("info.json") for p in paths), 2)
        self.assertEqual(sum("/files/groupcat-" in p for p in paths), 1)
        with patch("AnastrisTNG.TNGapi.urlopen", side_effect=AssertionError("unexpected HTTP")):
            np.testing.assert_array_equal(self.remote(offline=True).load_particle(0).gas["rho"], expected)

    def test_rejected_signed_ranges_use_official_endpoint_for_remaining_blocks(self):
        source = self.remote(api_key="test")._source
        filename = "groupcat-99.0.hdf5"
        self.archive[filename] = bytes(4 * 65536)
        issued, rejected = [], []

        def serve(request, timeout):
            if urlsplit(request.full_url).hostname == "data-eu.tng-project.org":
                self.assertIsNone(request.get_header("Api-key"))
                rejected.append(request.full_url)
                raise HTTPError(request.full_url, 403, "Forbidden", {}, None)
            self.assertEqual(request.get_header("Api-key"), "test")
            result = self.respond(request, timeout)
            issued.append(request.get_header("Range"))
            signed = f"https://data-eu.tng-project.org/file/test/{len(issued)}"
            result.geturl = lambda: signed
            return result

        from AnastrisTNG.TNGapi import _RangeFile
        with patch("AnastrisTNG.TNGapi.urlopen", side_effect=serve):
            with source._range_cache(filename) as (db, lock):
                with _RangeFile(source, "../../files/" + filename, None, db, lock) as stream:
                    for block in range(4):
                        stream.seek(65536 * block)
                        self.assertEqual(stream.read(1), b"\0")
        self.assertEqual(len(issued), 4)
        self.assertEqual(len(rejected), 1)

    def compare(self, local, remote):
        for fam in ("gas", "dm", "star", "bh"):
            a, b = getattr(local, fam), getattr(remote, fam)
            for name in ["iord", "HaloID", "SubhaloID", "pos", "mass", "pot"] + (["rho"] if fam == "gas" else []):
                x, y = a[name], b[name]
                self.assertEqual(x.units, y.units)
                self.assertEqual(x.dtype, y.dtype)
                np.testing.assert_array_equal(x, y)

    def test_subhalo_then_host_and_offline_reopen(self):
        local = Snapshot(str(self.local), 99)
        online = self.remote(api_key="test")
        self.assertCountEqual(self.calls, [("", {}), ("simulation", {})])
        for snap in (local, online):
            snap.load_subhalo(1)  # Keep this unrelated object across the masked merge.
            snap.load_subhalo(0)
            snap.load_halo(0)
            np.testing.assert_array_equal(snap.gas["SubhaloID"], [1, 0, 0, -1])
            np.testing.assert_array_equal(snap.gas["iord"], self.samples[0]["ParticleIDs"][[3, 0, 1, 2]])
        self.compare(local, online)
        self.assertFalse(any(path.startswith("snapshot-") for path, _ in self.calls))
        self.assertTrue(any(path == "halos/0/cutout.hdf5" for path, _ in self.calls))
        count = len(self.calls)
        for offline in (False, True):
            reused = self.remote(offline=offline)
            reused.load_subhalo(1)
            reused.load_subhalo(0)
            reused.load_halo(0)
            self.compare(local, reused)
        self.assertEqual(len(self.calls), count)

    def test_decorated_host_first_and_offline_miss(self):
        snap = self.remote(api_key="test")
        sub = snap.load_particle(0)
        self.assertEqual(sub.GC["SubhaloGrNr"], 0)
        np.testing.assert_array_equal(sub.gas["rho"], self.samples[0]["Density"][:2])
        snap._source.particle_counts("Subhalo", 1)
        offline = self.remote(offline=True).load_particle(1)
        with self.assertRaisesRegex(RemoteDataError, "subhalos/1.*gas.Potential.*Offline cache miss"):
            _ = offline.gas["pot"]
        self.assertNotIn("pot", offline.gas.keys())
        snap.load_halo(0)
        snap.load_subhalo(0)
        np.testing.assert_array_equal(snap.gas["SubhaloID"], [0, 0, -1])

    def test_failed_catalog_leaves_snapshot_ready_for_retry(self):
        snap = self.remote(api_key="test")
        snap.load_subhalo(1)
        before = snap["iord"].copy()
        fetch = snap._source._fetch_json

        def unavailable(endpoint, *args, **kwargs):
            if endpoint == "subhalos/0/info.json":
                raise RemoteDataError("catalogue unavailable")
            return fetch(endpoint, *args, **kwargs)

        with patch.object(snap._source, "_fetch_json", side_effect=unavailable):
            with self.assertRaisesRegex(RemoteDataError, "catalogue unavailable"):
                snap.load_subhalo(0)
        np.testing.assert_array_equal(snap["iord"], before)
        self.assertEqual(set(snap.PT_loaded_Subhalo), {1})
        snap.load_subhalo(0)
        self.assertEqual(len(snap), 12)
        self.assertEqual(len(np.unique(snap["iord"])), len(snap))
        self.assertNotIn(("subhalos/0/", {}), self.calls)

    def test_merge_preserves_mixed_unsigned_and_signed_particle_ids(self):
        snap = self.remote(api_key="test")
        for sample in self.samples.values():
            sample["ParticleIDs"] += np.uint64(2**63)
        ids = self.samples[0]["ParticleIDs"].copy()
        snap.load_subhalo(1)
        for sample in self.samples.values():
            sample["ParticleIDs"] = sample["ParticleIDs"].view(np.int64)
        snap.load_subhalo(0)
        self.assertEqual(snap.gas["iord"].dtype, np.dtype("uint64"))
        np.testing.assert_array_equal(snap.gas["iord"], ids[[3, 0, 1]])
        snap.load_halo(0)
        np.testing.assert_array_equal(snap.gas["iord"], ids[[3, 0, 1, 2]])
        np.testing.assert_array_equal(snap.gas["SubhaloID"], [1, 0, 0, -1])

    def test_retry_host_replacement_after_membership_read_fails(self):
        snap = self.remote(api_key="test")
        snap.load_subhalo(1)
        snap.load_subhalo(0)
        before = snap["iord"].copy(), snap["SubhaloID"].copy()
        read = snap._source.particles

        def unavailable(kind, number, fields, **kwargs):
            if kind == "Subhalo" and number == 0:
                raise RemoteDataError("membership IDs unavailable")
            return read(kind, number, fields, **kwargs)

        with patch.object(snap._source, "particles", side_effect=unavailable):
            with self.assertRaisesRegex(RemoteDataError, "membership IDs unavailable"):
                snap.load_halo(0)
        np.testing.assert_array_equal(snap["iord"], before[0])
        np.testing.assert_array_equal(snap["SubhaloID"], before[1])
        self.assertNotIn(0, snap.PT_loaded_Halo)
        snap.load_halo(0)
        np.testing.assert_array_equal(snap.gas["SubhaloID"], [1, 0, 0, -1])
        self.assertEqual(len(np.unique(snap["iord"])), len(snap))

    def test_merging_rebinds_other_views_without_fetching_their_catalogues(self):
        for snap in (Snapshot(str(self.local), 99), self.remote(api_key="test")):
            with self.subTest(source=type(snap._source).__name__):
                self.assertEqual(len(snap.halos[0]), 0)
                snap.load_subhalo(0)
                self.assertEqual(len(snap.halos[0]), 8)
                self.assertNotIn(("halos/0/info.json", {}), self.calls)
                self.assertEqual(snap.halos[0].GC["GroupFirstSub"], 0)

    def test_object_array_preserves_offline_error(self):
        self.remote(api_key="test").load_particle(0)
        sub = self.remote(offline=True).load_particle(0)
        self.assertEqual(sub["Snapshot"], 99)
        with self.assertRaisesRegex(RemoteDataError, "Offline cache miss"):
            sub["pos"]
        sub.ancestor.gas["gas_only"] = np.ones(len(sub.gas))
        with self.assertRaisesRegex(KeyError, "family-level"):
            sub["gas_only"]

    def test_physical_units_default_applies_to_late_arrays_on_both_sources(self):
        for snap in (Snapshot(str(self.local), 99), self.remote(api_key="test")):
            with self.subTest(source=type(snap._source).__name__):
                sub = snap.load_particle(0)
                sub.physical_units()
                self.assertEqual(str(sub.gas["mass"].units), "Msol")
                self.assertEqual(str(sub.gas["rho"].units), "Msol kpc**-3")
                np.testing.assert_allclose(sub.gas["mass"], self.samples[0]["Masses"][:2] * (1e10 / .6774))
                temporary = snap.load_particle(0)
                temporary.physical_units(persistent=False)
                self.assertEqual(temporary.gas["rho"].units, snap.read_particle_field(0, "gas", "rho").units)

    def test_batched_radius_uses_pynbody_physical_units(self):
        local = Snapshot(str(self.local), 99).load_particle(0)
        sub = self.remote(api_key="test").load_particle(0)
        for obj in (local, sub):
            obj.physical_units(persistent=True)
        expected, actual = local.R(calfor="gas"), sub.R(calfor="gas")
        self.assertEqual(actual.units, expected.units)
        np.testing.assert_allclose(actual, expected)
        self.assertEqual(str(sub.gas["pos"].units), "kpc")
        self.assertEqual(str(sub.gas["mass"].units), "Msol")
        self.assertEqual([q for p, q in self.calls if p.endswith("cutout.hdf5")],
                         [{"gas": ["Coordinates,Masses"]}])

    def test_physical_units_and_family_selection(self):
        local = Snapshot(str(self.local), 99)
        online = self.remote(api_key="test")
        subs = []
        for snap in (local, online):
            snap.load_particle_para["particle_field"] = "gas,star"
            sub = snap.load_particle(0, order="gas,dm,star,bh")
            self.assertEqual([f.name for f in sub.families()], ["gas", "star"])
            sub.physical_units(persistent=True)
            subs.append(sub)
        x, y = (sub.gas["rho"] for sub in subs)
        self.assertEqual(x.units, y.units)
        self.assertEqual(str(y.units), "Msol kpc**-3")
        np.testing.assert_allclose(x, y, rtol=np.finfo(x.dtype).eps)
        self.assertEqual([call for call in self.calls if call[0].endswith("cutout.hdf5")],
                         [("subhalos/0/cutout.hdf5", {"gas": ["Density"]})])

    def test_partial_local_output(self):
        snap = self.remote(local_dir=self.local, offline=True)
        snap.load_halo(0)
        self.assertFalse(self.calls)
        # Only one field is missing in one of the required local chunks.
        with h5py.File(self.local / "snapdir_099/snap_099.1.hdf5", "r+") as file:
            del file["PartType0/Density"]
        snap = self.remote(local_dir=self.local, api_key="test")
        snap.load_halo(0)
        np.testing.assert_array_equal(snap.gas["rho"], self.samples[0]["Density"][:3])
        self.assertEqual(self.calls, [("halos/0/cutout.hdf5", {"gas": ["Density"]})])

    def test_concurrent_field_fill(self):
        first = self.remote(api_key="test").load_particle(0)
        second = self.remote(api_key="test").load_particle(0)
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda sub: sub.gas["rho"], [first, second]))
        for result in results:
            np.testing.assert_array_equal(result, self.samples[0]["Density"][:2])
        self.assertEqual(sum(path == "subhalos/0/cutout.hdf5" for path, _ in self.calls), 1)
        self.assertFalse(any(path.startswith("snapshot-") for path, _ in self.calls))

    def test_catalog_validation_workflow(self):
        online = self.remote(api_key="test")
        offline = self.remote(offline=True)
        sub_id = select_gas_subhalo(online)
        validate_catalogs(self, online, offline, sub_id, verbose=False)

    def test_truncated_cutout_and_progress_without_length(self):
        source = self.remote(api_key="test")
        sub = source.load_particle(0)
        source._source._particle_offsets("Subhalo", 0)
        def truncated(request, timeout):
            response = self.respond(request, timeout)
            if urlsplit(request.full_url).path.endswith("/cutout.hdf5"):
                response.truncate(len(response.getvalue()) - 10)
            return response
        with patch("AnastrisTNG.TNGapi.urlopen", side_effect=truncated) as get, patch("time.sleep"):
            with self.assertRaisesRegex(RemoteDataError, "Incomplete download"):
                _ = sub.gas["rho"]
            self.assertEqual(get.call_count, 3)
        self.assertFalse(list(self.cache.rglob("*.part")))

        def no_length(request, timeout):
            response = self.respond(request, timeout)
            response.headers.pop("Content-Length")
            return response
        with patch("AnastrisTNG.TNGapi.urlopen", side_effect=no_length), patch("AnastrisTNG.TNGapi.tqdm") as bar:
            reopened = Snapshot.from_api("TNG50-1", 99, cache_dir=self.cache, api_key="test", progress=True)
            sub = reopened.load_particle(0)
            bar.assert_not_called()
            np.testing.assert_array_equal(sub.gas["rho"], self.samples[0]["Density"][:2])
            self.assertIsNone(bar.return_value.__enter__.return_value.total)
            self.assertFalse(bar.call_args.kwargs["disable"])
            meter = bar.return_value.__enter__.return_value
            meter.update.assert_called()
            stages = [call.args[0] for call in meter.set_postfix_str.call_args_list]
            self.assertTrue(any("Waiting" in stage for stage in stages))
            for stage in ("Downloading cutout", "Validating cutout", "Writing particle cache", "Assembling particle arrays"):
                self.assertIn(stage, stages)
            self.assertEqual(stages[-1], "Complete")

    def test_slow_cache_read_displays_stage_before_it_finishes(self):
        from threading import Event

        source = self.remote(api_key="test")._source
        counts = source.particle_counts("Subhalo", 0)
        fields = {"star": ["Masses"]}
        first = source.particles("Subhalo", 0, fields, counts=counts)
        source.progress = True
        displayed = Event()
        read = ParticleRowCache.read

        def wait_for_display(cache, *args, **kwargs):
            self.assertTrue(displayed.wait(3), "cache read did not report progress while blocked")
            return read(cache, *args, **kwargs)

        def observe(stage):
            if stage == "Reading particle cache":
                displayed.set()

        with patch("AnastrisTNG.TNGapi.tqdm") as meter, patch.object(ParticleRowCache, "read", wait_for_display):
            bar = meter.return_value.__enter__.return_value
            bar.set_postfix_str.side_effect = observe
            result = source.particles("Subhalo", 0, fields, counts=counts)
            self.assertIsNone(bar.total)
            self.assertEqual(bar.set_postfix_str.call_args.args, ("Complete",))
        np.testing.assert_array_equal(result["star"]["Masses"], first["star"]["Masses"])

    def test_notebook_progress_matches_known_and_unknown_response_lengths(self):
        from tqdm.notebook import IProgress, tqdm as notebook_bar
        if IProgress is None:
            self.skipTest("ipywidgets is optional")
        source = self.remote(api_key="test")._source
        counts = source.particle_counts("Subhalo", 0)
        source._particle_offsets("Subhalo", 0)
        source.progress = True
        meters = []

        def create(**kwargs):
            meter = notebook_bar(display=False, **kwargs)
            meters.append(meter)
            return meter

        def without_length(request, timeout):
            response = self.respond(request, timeout)
            response.headers.pop("Content-Length")
            return response

        store = ParticleRowCache.store

        def check_write(cache, *args, **kwargs):
            self.assertIsNone(meters[-1].total, "cache writes still have work to do after download")
            self.assertEqual(meters[-1].container.children[1].layout.display, "none")
            return store(cache, *args, **kwargs)

        with ExitStack() as stack:
            stack.enter_context(patch("AnastrisTNG.TNGapi._IN_NOTEBOOK", True))
            stack.enter_context(patch("AnastrisTNG.TNGapi.tqdm", side_effect=create))
            stack.enter_context(patch.object(ParticleRowCache, "store", check_write))
            source.particles("Subhalo", 0, {"star": ["Masses"]}, counts=counts)
            with patch("AnastrisTNG.TNGapi.urlopen", side_effect=without_length):
                source.particles("Subhalo", 0, {"star": ["Coordinates"]}, counts=counts)
        known, unknown = meters
        self.assertEqual(known.container.children[1].max, known.n)
        self.assertEqual(known.total, known.n)
        self.assertIsNone(unknown.total)
        self.assertEqual(unknown.container.children[1].layout.display, "none")

    def test_standalone_lazy_fields(self):
        snap = self.remote(api_key="test")
        sub = snap.load_particle(0)
        self.assertCountEqual(self.calls[:2], [("", {}), ("simulation", {})])
        self.assertEqual(self.calls[2:], [("subhalos/0/", {})])
        for key in ["pos", "vel", "iord", "mass"]:
            self.assertNotIn(key, sub.keys())
        np.testing.assert_array_equal(sub.dm["mass"], np.full(2, self.dm_mass))
        density = sub.gas["rho"]
        np.testing.assert_array_equal(density, self.samples[0]["Density"][:2])
        np.testing.assert_array_equal(sub.gas["iord"], self.samples[0]["ParticleIDs"][:2])
        self.assertNotIn("pos", sub.keys())
        pos = sub["pos"]
        for fam, pt in PART_TYPE.items():
            np.testing.assert_array_equal(getattr(sub, fam)["pos"], self.samples[pt]["Coordinates"][:2])
        self.assertEqual(len(pos), len(sub))
        before = len(self.calls)
        ids = sub["iord"]
        self.assertEqual(len(ids), len(sub))
        self.assertEqual(ids.dtype, np.uint64)
        queries = {}
        for path, fields in self.calls[before:]:
            self.assertEqual(path, "subhalos/0/cutout.hdf5")
            queries.update(fields)
        self.assertEqual(queries, {family: ["ParticleIDs"] for family in ("dm", "stars", "bhs")})
        self.assertNotIn("vel", sub.keys())

    def test_standalone_configured_fields_and_eager_opt_in(self):
        snap = self.remote(api_key="test")
        snap.load_particle_para["star_fields"] = ["Potential"]
        halo = snap.load_particle(0, groupType="Halo", decorate=False)
        self.assertEqual([call for call in self.calls if call[0].endswith("cutout.hdf5")],
                         [("halos/0/cutout.hdf5", {"stars": ["Potential"]})])
        self.assertNotIn("pos", halo.keys())
        np.testing.assert_array_equal(halo.star["pot"], self.samples[4]["Potential"][:3])
        full = snap.load_particle(1, lazy=False)
        self.assertIn("pos", full.keys())
        self.assertIn("vel", full.keys())
        np.testing.assert_array_equal(full.dm["iord"], self.samples[1]["ParticleIDs"][3:4])

    def test_independent_field_omits_ids_and_reopens_offline(self):
        online = self.remote(api_key="test")
        mass = online.read_particle_field(0, "star", "mass")
        self.assertEqual([call for call in self.calls if call[0].endswith("cutout.hdf5")],
                         [("subhalos/0/cutout.hdf5", {"stars": ["Masses"]})])
        self.assertFalse(any(path.startswith("snapshot-") for path, _ in self.calls))
        local = Snapshot(str(self.local), 99).read_particle_field(0, "star", "Masses")
        self.assertEqual(mass.units, local.units)
        np.testing.assert_array_equal(mass, local)
        before = len(self.calls)
        replay = self.remote(offline=True).read_particle_field(0, "star", "Masses")
        np.testing.assert_array_equal(mass, replay)
        self.assertEqual(before, len(self.calls))

        sub = online.load_particle(0)
        aligned = sub.star["mass"]
        self.assertEqual(before, len(self.calls))
        rows = np.argsort(sub.star["iord"])
        np.testing.assert_array_equal(aligned[rows], mass)

        # An independent read can reuse an already aligned field.
        pos = sub.star["pos"]
        before = len(self.calls)
        raw = online.read_particle_field(0, "star", "pos")
        np.testing.assert_array_equal(raw, pos)
        self.assertEqual(before, len(self.calls))

    def test_independent_field_partial_local(self):
        with h5py.File(self.local / "snapdir_099/snap_099.0.hdf5", "r+") as f:
            del f["PartType4/ParticleIDs"]
        local = self.remote(local_dir=self.local, offline=True)
        np.testing.assert_array_equal(local.read_particle_field(0, "star", "mass"),
                                      self.samples[4]["Masses"][:2])
        self.assertFalse(self.calls)

    def test_cached_fields_then_merge(self):
        snap = self.remote(api_key="test")
        sub = snap.load_particle(0)
        mass = sub.star["mass"].copy()
        rho = sub.gas["rho"].copy()
        replay = self.remote(offline=True).load_particle(0)
        np.testing.assert_array_equal(replay.star["mass"], mass)
        snap.load_subhalo(0)
        gas_ids = snap.gas["iord"].copy()
        snap.load_halo(0)
        for merged in (snap, self.remote(offline=True)):
            if merged is not snap:
                merged.load_subhalo(0)
                merged.load_halo(0)
            member = merged.gas["SubhaloID"] == 0
            np.testing.assert_array_equal(merged.gas["iord"][member], gas_ids)
            np.testing.assert_array_equal(merged.gas["rho"][member], rho)

    def test_summary_only_cache_and_deferred_catalog_units(self):
        snap = self.remote(api_key="test")
        sub = snap.load_particle(0)
        rho = sub.gas["rho"]
        sub.physical_units()
        self.assertIn("SubhaloID=0", repr(sub))
        self.assertCountEqual([path for path, q in self.calls],
                              ["", "simulation", "subhalos/0/", "offsets.99.hdf5", "subhalos/0/cutout.hdf5"])
        offline = self.remote(offline=True).load_particle(0)
        offline.physical_units(persistent=True)
        np.testing.assert_array_equal(offline.gas["rho"], rho)
        with self.assertRaisesRegex(RemoteDataError, "Offline cache miss"):
            dict(offline.GC)
        reference = Snapshot(str(self.local), 99).load_particle(0)
        reference.physical_units()
        for obj in (sub, offline):
            self.assertEqual(obj.GC["SubhaloMass"].units, reference.GC["SubhaloMass"].units)
            np.testing.assert_allclose(obj.GC["SubhaloMass"], reference.GC["SubhaloMass"], rtol=1e-6)
        self.assertEqual([path for path, q in self.calls].count("subhalos/0/info.json"), 1)

    def test_subhalo_repr_and_notebook_display_defer_catalog(self):
        try:
            from IPython.core.formatters import DisplayFormatter
        except ImportError:
            DisplayFormatter = None

        online = self.remote(api_key="test").load_particle(0)
        offline = self.remote(offline=True).load_particle(0)
        self.calls.clear()
        for sub in (online, offline):
            expected = repr(sub)
            self.assertIn("SubhaloID=0", expected)
            self.assertFalse(self.calls)
            if DisplayFormatter is not None:
                data, _ = DisplayFormatter().format(sub)
                self.assertEqual(data["text/plain"], expected)
                self.assertFalse(self.calls)
            self.assertFalse(hasattr(sub, "_ipython_canary_method_should_not_exist_"))
            self.assertFalse(self.calls)
            self.assertTrue(sub._gc_pending)

        self.assertEqual(online.SubhaloMass, online.GC["SubhaloMass"])
        self.assertEqual(self.calls, [("subhalos/0/info.json", {})])

    def test_incomplete_summary_uses_full_catalog(self):
        snap = self.remote(api_key="test")
        def incomplete(request, timeout):
            if urlsplit(request.full_url).path.endswith("subhalos/0/"):
                return Response(b'{"id":0}')
            return self.respond(request, timeout)
        with patch("AnastrisTNG.TNGapi.urlopen", side_effect=incomplete):
            sub = snap.load_particle(0)
        self.assertEqual(len(sub), 8)
        self.assertIsNone(snap._source._object_json("Subhalo", 0, "counts"))
        with patch("AnastrisTNG.TNGapi.urlopen", side_effect=AssertionError("unexpected HTTP")):
            self.assertEqual(len(self.remote(offline=True).load_particle(0)), len(sub))

    def test_summary_cache_with_incomplete_local_mirror(self):
        online = self.remote(api_key="test")
        mass = online.load_particle(0).star["mass"].copy()
        # A local chunk without its group catalog/offsets cannot supply object counts.
        (self.local / "groups_099/groups_099.0.hdf5").unlink()
        with patch("AnastrisTNG.TNGapi.urlopen", side_effect=AssertionError("unexpected HTTP")):
            sub = self.remote(local_dir=self.local, offline=True).load_particle(0)
            np.testing.assert_array_equal(sub.star["mass"], mass)

    def test_uncached_evolution_offline(self):
        snap = self.remote(api_key="test")
        snap._source.offline = True
        for method, arg in [("galaxy_evolution", 0), ("halo_evolution", 0)]:
            with self.subTest(method=method), self.assertRaisesRegex(RemoteDataError, "Offline cache miss"):
                getattr(snap, method)(arg, fields=[])
        hybrid = self.remote(local_dir=self.local, offline=True)
        np.testing.assert_allclose(hybrid.halos_GC("GroupMass"), self.catalog["Group"]["GroupMass"])

    def test_catalog_columns_overlap_cold_downloads_and_open_each_local_shard_once(self):
        from threading import Barrier

        snap = self.remote(api_key="test")
        fields = ["GroupMass", "GroupLenType"]
        both = Barrier(2)

        def serve(endpoint, resource, path):
            field = parse_qs(urlsplit(endpoint).query)["Group"][0]
            both.wait(timeout=3)
            with h5py.File(path, "w") as file:
                file["Group/" + field] = self.catalog["Group"][field]

        with patch.object(snap._source, "_stream", side_effect=serve):
            columns = snap.halos_GC(fields)
        self.assertEqual(list(columns), ["count", *fields])
        mirror = self.remote(local_dir=self.local, offline=True)
        with patch("AnastrisTNG.TNGapi.h5py.File", wraps=h5py.File) as opened:
            local = mirror.halos_GC(fields)
        for field in fields:
            np.testing.assert_array_equal(columns[field], local[field])
        # One header inspection plus one data pass per shard for this multi-column read.
        paths = [Path(call.args[0]) for call in opened.call_args_list]
        self.assertLessEqual(paths.count(self.local / "groups_099/groups_099.0.hdf5"), 2)

    def test_archive_columns_retry_and_offline(self):
        snap = self.remote(api_key="test")
        requests = []
        def serve(endpoint, resource, path):
            field = parse_qs(urlsplit(endpoint).query)["Group"][0]
            requests.append(field)
            values = self.catalog["Group"][field]
            with h5py.File(path, "w") as f:
                f["Group/" + field] = values[:1] if len(requests) == 1 else values
        with patch.object(snap._source, "_stream", side_effect=serve):
            with self.assertRaisesRegex(RemoteDataError, "incomplete"):
                snap.halos_GC("GroupMass")
            snap.halos_GC("GroupMass")
            columns = snap.halos_GC(["GroupMass", "GroupLenType"])
            self.assertEqual(requests, ["GroupMass", "GroupMass", "GroupLenType"])
        self.assertEqual(columns["count"], 2)
        offline = self.remote(offline=True)
        np.testing.assert_array_equal(columns["GroupMass"], offline.halos_GC("GroupMass"))
        with self.assertRaisesRegex(RemoteDataError, "GroupFirstSub"):
            offline.halos_GC("GroupFirstSub")

    def test_zoom_extra_types_and_shared_chunks(self):
        snap = self.remote(api_key="test")
        snap._source.run = snap.properties["run"] = "TNG-Cluster"
        root = snap._source.root
        counts = [[0, 0, 2, 3, 0, 0]]
        (root / "zoom-layout.json").write_text(json.dumps([[0], counts, counts]))
        products = {}
        calls = []
        for chunk in (0, 1):
            output = io.BytesIO()
            with h5py.File(output, "w") as f:
                f["PartType2/Masses"] = np.asarray([1, 2], dtype="f4") + chunk
                ids = np.arange(3, dtype="uint64") + (2**60 + 3 * chunk)
                f["PartType3/TracerID"] = ids.view("int64") if chunk else ids
            products[f"snapshot-99.{chunk}.hdf5"] = output.getvalue()
        def respond(request, timeout):
            filename = urlsplit(request.full_url).path.split("/")[-1]
            content = products[filename]
            calls.append(filename)
            self.assertEqual(request.get_header("Range"), "bytes=0-65535")
            response = Response(content)
            response.status = 206
            response.geturl = lambda: request.full_url
            response.headers["Content-Range"] = f"bytes 0-{len(content)-1}/{len(content)}"
            return response
        with patch("AnastrisTNG.TNGapi.urlopen", side_effect=respond):
            mass = snap.read_particle_field(0, "dmlowres", "mass", "Zoom")
            ids = snap.read_particle_field(0, "tracer", "iord", "Zoom")
            np.testing.assert_equal(np.asarray(mass), [1, 2, 2, 3])
            self.assertEqual(ids.dtype, np.dtype("uint64"))
            np.testing.assert_equal(np.asarray(ids), np.arange(6, dtype="uint64") + 2**60)
            snap._source.offline = True
            np.testing.assert_array_equal(ids, snap.read_particle_field(1, "tracer", "TracerID", "Zoom"))
            self.assertEqual(len(calls), 4)

    def test_sublink_raw_blocks_shared_across_snapshots(self):
        source = self.remote(api_key="test")._source
        filename = "sublink.3.hdf5"
        output = io.BytesIO()
        values = np.arange(200000, dtype="i8")
        with h5py.File(output, "w") as file:
            file["SubhaloID"] = values
        content = output.getvalue()
        requests = []
        etag = '"original"'
        truncate = False

        def respond(request, timeout):
            start, end = map(int, request.get_header("Range")[6:].split("-"))
            stop = min(end + 1, len(content))
            requests.append((start, stop, request.get_header("If-range")))
            response = Response(content[start:stop-int(truncate)])
            response.status = 206
            response.headers.update({"Content-Range": f"bytes {start}-{stop-1}/{len(content)}", "ETag": etag})
            response.geturl = lambda: request.full_url
            return response

        def check(src, rows):
            with src._raw_file(filename) as file:
                np.testing.assert_array_equal(file["SubhaloID"][rows], values[rows])

        with patch("AnastrisTNG.TNGapi.urlopen", side_effect=respond):
            check(source, slice(40000, 80000))
            first = list(requests)
            self.assertTrue(any(stop - start > 65536 for start, stop, _ in first))
            other = source.at_snapshot(50)
            other.offline = True
            check(other, slice(60000, 78000))
            new_rows = slice(140000, 140004)
            with self.assertRaisesRegex(RemoteDataError, "Offline cache miss"):
                check(other, new_rows)
            self.assertEqual(requests, first)
            etag = '"changed"'
            with self.assertRaisesRegex(RemoteDataError, "changed"):
                check(source, new_rows)
            etag = '"original"'
            truncate = True
            with self.assertRaisesRegex(RemoteDataError, "Incomplete"):
                check(source, new_rows)
            truncate = False
            check(source, new_rows)
            self.assertEqual(requests[-1][2], etag)
            check(other, new_rows)

    def test_history_from_local_shard_with_nonroot_row(self):
        # An intermediate-snapshot object can begin in the middle of a SubLink shard.
        snap = self.remote(api_key="test")
        base = self.local.parent / "postprocessing"
        offset = base / "offsets/offsets_099.hdf5"
        offset.parent.mkdir(parents=True)
        with h5py.File(offset, "w") as f:
            f["FileOffsets/SubLink"] = [0, 40]
            for field, value in {"RowNum": 41, "SubhaloID": 301, "LastProgenitorID": 303}.items():
                f["Subhalo/SubLink/" + field] = [value]
        path = base / "trees/SubLink/tree_extended.1.hdf5"
        path.parent.mkdir(parents=True)
        with h5py.File(path, "w", alignment_threshold=1, alignment_interval=65536) as f:
            f["SubhaloID"] = [300, 301, 302, 303, 304]
            f["FirstProgenitorID"] = [301, 302, -1, -1, -1]
            f["NextProgenitorID"] = [-1, -1, 303, -1, -1]
            f["MainLeafProgenitorID"] = [302, 302, 302, 303, 304]
            f["SubfindID"] = [0, 0, 1, 2, 0]
            f["SnapNum"] = [99, 99, 98, 98, 98]
            f["SubhaloMassType"] = np.ones((5, 6))
            f["SubhaloPos"] = np.arange(15).reshape(5, 3).astype("f4")
            f["UncachedColumn"] = np.arange(5)
        snap._source.local = str(self.local)
        (snap._source.root.parent / "scale_factors.json").write_text('{"98":0.9}')
        snap._source.offline = True
        history = snap.merger_history(0, fields=["SubhaloPos"], minMassRatio=.1)
        self.assertEqual(history["numMergers"], 1)
        np.testing.assert_allclose(history["First-SubhaloPos"][0], np.arange(6, 9) * .9 / .6774)
        np.testing.assert_allclose(history["Next-a"], [.9])
        reopened = self.remote(offline=True)
        again = reopened.merger_history(0, fields=["SubhaloPos"], minMassRatio=.1)
        for key in history:
            np.testing.assert_array_equal(history[key], again[key])
        with patch("AnastrisTNG.TNGapi.urlopen", side_effect=AssertionError("unexpected HTTP")):
            with self.assertRaisesRegex(RemoteDataError, "Offline cache miss"):
                reopened._source.merger_tree(0, ["UncachedColumn"])
            copied = self.local / "trees/SubLink/tree_extended.1.hdf5"
            copied.parent.mkdir(parents=True)
            shutil.copyfile(path, copied)
            np.testing.assert_array_equal(
                snap._source.merger_tree(0, ["UncachedColumn"])["UncachedColumn"], [1, 2, 3])
            np.testing.assert_array_equal(
                reopened._source.merger_tree(0, ["UncachedColumn"])["UncachedColumn"], [1, 2, 3])
            gal = snap.galaxy_evolution(0, fields="SubhaloPos")
            np.testing.assert_array_equal(gal["SnapNum"], [99, 98])
            np.testing.assert_allclose(gal["SubhaloPos"][1], history["First-SubhaloPos"][0])
            self.assertEqual(gal["count"], 2)
        self.assertFalse((snap._source.root / "subhalos").exists())

    def test_evolution_reuses_branch_and_catalog_columns(self):
        snap = self.remote(api_key="test")
        source = snap._source
        source._store_object_value("Subhalo", 0, "sublink-location", "[0, 1, 3, 100]")
        (source.root.parent / "scale_factors.json").write_text('{"95":0.8}')
        shard = source.root / "groupcat/sublink.0.hdf5"
        shard.parent.mkdir()
        with h5py.File(shard, "w") as file:
            file["SubhaloID"] = [99, 100, 101, 102]
            file["MainLeafProgenitorID"] = [101, 101, 101, 102]
            file["SnapNum"] = [99, 99, 95, 95]
            file["SubhaloGrNr"] = [0, 1, 0, 1]
            file["SubfindID"] = [1, 0, 1, 2]
            file["GroupPos"] = np.arange(12, dtype="f4").reshape(4, 3)
        self.archive["sublink.0.hdf5"] = shard.read_bytes()
        shard.unlink()
        def catalog_column(epoch, field, values):
            path = source.at_snapshot(epoch).root / "groupcat/Group" / (field + ".hdf5")
            path.parent.mkdir(parents=True, exist_ok=True)
            with h5py.File(path, "w") as file:
                file["Group/" + field] = values
        catalog_column(99, "GroupFirstSub", [-1, 0])
        for epoch in (99, 95):
            catalog_column(epoch, "GroupMass", np.array([3, 4], dtype="f4"))
        source.offline = True
        with self.assertRaisesRegex(ValueError, "no central"):
            snap.halo_evolution(0, fields="GroupMass")
        source.offline = False
        halo = snap.halo_evolution(1, False, fields=["GroupMass", "GroupPos"])
        np.testing.assert_array_equal(halo["HaloID"], [1, 0])
        np.testing.assert_array_equal(halo["GroupMass"], [4, 3])
        np.testing.assert_array_equal(halo["GroupPos"], np.arange(3, 9).reshape(2, 3))
        self.assertFalse((source.root / "subhalos").exists())
        # Exercise the same positional calls against the original local file layout.
        local_tree = self.local / "trees/SubLink/tree_extended.0.hdf5"
        local_tree.parent.mkdir(parents=True)
        local_tree.write_bytes(self.archive["sublink.0.hdf5"])
        group = self.local / "groups_099/groups_099.0.hdf5"
        with h5py.File(group, "a") as file:
            for k, v in {"RowNum": 1, "LastProgenitorID": 102, "SubhaloID": 100}.items():
                file["Offsets/Subhalo_Sublink" + k] = [v, -1]
            file["Group/GroupFirstSub"][:] = [-1, 0]
            file["Group/GroupMass"][:] = [3, 4]
            file["Group/GroupPos"] = np.arange(6).reshape(2, 3)
        other = self.local / "groups_095/groups_095.0.hdf5"
        other.parent.mkdir()
        shutil.copyfile(group, other)
        with h5py.File(other, "a") as file:
            file["Header"].attrs["Time"] = .8
            file["Group/GroupPos"][0] = [6, 7, 8]
        local = Snapshot(str(self.local), 99)
        expected = local.halo_evolution(1, False)
        for field in halo:
            np.testing.assert_array_equal(expected[field], halo[field])
        gal = local.galaxy_evolution(0, ["GroupPos"], False)
        remote_gal = snap.galaxy_evolution(0, ["GroupPos"], False)
        for field in gal:
            np.testing.assert_array_equal(gal[field], remote_gal[field])
        full = source.merger_tree(0, ["GroupPos"])
        np.testing.assert_array_equal(full["GroupPos"], np.arange(3, 12).reshape(3, 3))
        self.assertEqual(sum(path == "sublink.0.hdf5" for path, _ in self.calls), 1)
        again = self.remote(offline=True).halo_evolution(1, fields=["GroupMass", "GroupPos"])
        np.testing.assert_allclose(again["GroupPos"], np.asarray(halo["GroupPos"]) * [[1], [.8]] / .6774)


def select_gas_subhalo(snap):
    """Consult catalog metadata before downloading any particles."""
    source = snap._source
    query = "subhalos/?mass_gas__gt=0.01&mass__lt=5&order_by=-id&limit=50"
    with source._open(query, "gas-bearing subhalo candidates") as response:
        candidates = json.load(response)["results"]
    for candidate in candidates:
        number = candidate["id"]
        sub = source.catalog("Subhalo", number)
        host = source.catalog("Halo", sub["SubhaloGrNr"])
        counts = host["GroupLenType"]
        if (0 < sub["SubhaloLenType"][0] < counts[0]
                and sum(counts[pt] for pt in PART_TYPE.values()) <= 100_000):
            return number
    raise ValueError("No small gas-bearing host found in the candidates. Set TNG_TEST_SUBHALO_ID in .env.")


def validate_catalogs(test, online, offline, sub_id, verbose):
    def check_mass(particles, catalog, prefix, number):
        reference = catalog[prefix + "Mass"]
        by_type = np.asarray(catalog[prefix + "MassType"].in_units(reference.units))
        measured = np.zeros(6)
        for family, pt in PART_TYPE.items():
            group = getattr(particles, family)
            test.assertEqual(len(group), catalog[prefix + "LenType"][pt])
            if len(group):
                measured[pt] = np.asarray(group["mass"].in_units(reference.units)).sum(dtype=np.float64)
        total = measured.sum()
        if verbose:
            print(f"\n{prefix} {number}: particle mass={total:.12g}, catalog={float(reference):.12g}, "
                  f"difference={total - float(reference):+.9g} [{reference.units}], "
                  f"relative difference={total / float(reference) - 1:+.9%}")
        # Gas + stars keeps wind particles together across the catalog's type convention.
        # Check the fixed DM mass independently of the variable baryonic masses.
        for label, value, expected in [
            ("total", total, float(reference)), ("DM", measured[1], by_type[1]),
            ("gas + stars", measured[0] + measured[4], by_type[0] + by_type[4]),
            ("BH", measured[5], by_type[5]),
        ]:
            with test.subTest(object=prefix, id=number, mass=label):
                np.testing.assert_allclose(value, expected, rtol=1e-4, atol=0)

    online.load_subhalo(sub_id)
    sub = online.subhalos[sub_id]
    host_id = int(sub.GC["SubhaloGrNr"])
    test.assertGreater(len(sub.gas), 0, "Select a subhalo containing gas")
    check_mass(sub, sub.GC, "Subhalo", sub_id)
    # Independent cutouts. Compare the same cells by integer ID after the host replaces the subhalo.
    ids = np.asarray(sub.gas["iord"]).copy()
    density = sub.gas["rho"].copy()
    online.load_halo(host_id)
    check_mass(online, online.halos[host_id].GC, "Group", host_id)
    member = np.asarray(online.gas["SubhaloID"]) == sub_id
    host_ids = np.asarray(online.gas["iord"])[member]
    host_density = online.gas["rho"]
    test.assertEqual(density.units, host_density.units)
    test.assertEqual(density.dtype, host_density.dtype)
    np.testing.assert_array_equal(np.sort(ids), np.sort(host_ids))
    np.testing.assert_array_equal(density[np.argsort(ids)], host_density[member][np.argsort(host_ids)])
    offline.load_subhalo(sub_id)
    offline.load_halo(host_id)
    for key in ("iord", "SubhaloID", "rho", "mass"):
        np.testing.assert_array_equal(online.gas[key], offline.gas[key])
    if verbose:
        print(f"Matched {len(ids)} gas-cell IDs and densities, replayed host {host_id} offline.")


class PublicRowOrderValidation(unittest.TestCase):
    def test_stellar_fields_then_ids_and_offline_reopen(self):
        key = os.getenv("TNG_API_KEY") or dotenv_values(Path(__file__).resolve().parents[1] / ".env",
                                                        encoding="utf-8-sig").get("TNG_API_KEY")
        if not key:
            self.skipTest("Requires a TNG API key")
        with tempfile.TemporaryDirectory() as cache:
            options = dict(cache_dir=cache, progress=False)
            online = Snapshot.from_api("TNG50-1", 99, api_key=key, **options)
            sub = online.load_particle(12)
            fields = [("mass", "Masses"), ("aform", "GFM_StellarFormationTime"), ("iord", "ParticleIDs")]
            values = {}
            for name, _ in fields:
                values[name] = sub.star[name].copy()
            self.assertFalse((online._source.root / "subhalos").exists())
            query = "subhalos/12/cutout.hdf5?stars=Masses,GFM_StellarFormationTime,ParticleIDs"
            with online._source._open(query, "reference stellar cutout") as response:
                with h5py.File(io.BytesIO(response.read()), "r") as ref:
                    for name, raw in fields:
                        expected = ref["PartType4"][raw][:]
                        self.assertEqual(values[name].dtype, expected.dtype)
                        np.testing.assert_array_equal(values[name], expected)
            with patch("AnastrisTNG.TNGapi.urlopen", side_effect=AssertionError("unexpected HTTP")):
                offline = Snapshot.from_api("TNG50-1", 99, offline=True, **options).load_particle(12)
                for name, _ in reversed(fields):
                    self.assertEqual(offline.star[name].units, values[name].units)
                    np.testing.assert_array_equal(offline.star[name], values[name])
            print(f"Verified all three arrays in file order and offline for {len(values['mass'])} stars.")


class PublicCatalogValidation(unittest.TestCase):
    def test_real_api_mass_and_lazy_density(self):
        settings = dotenv_values(Path(__file__).resolve().parents[1] / ".env",
                                 encoding="utf-8-sig", interpolate=False)
        settings.update(os.environ)
        key = settings.get("TNG_API_KEY")
        if not key:
            self.skipTest("Fill in TNG_API_KEY in the project .env to run the real API validation")
        run = settings.get("TNG_TEST_RUN") or "TNG50-4"
        snap = int(settings.get("TNG_TEST_SNAP") or 99)
        with tempfile.TemporaryDirectory() as cache:
            online = Snapshot.from_api(run, snap, cache_dir=cache, api_key=key)
            value = settings.get("TNG_TEST_SUBHALO_ID")
            number = int(value) if value else select_gas_subhalo(online)
            print(f"\nValidating {run} snapshot {snap}, subhalo {number}, "
                  f"Mdm={float(online.properties['Mdm']):.15g} [{online.properties['Mdm'].units}]", flush=True)
            offline = Snapshot.from_api(run, snap, cache_dir=cache, offline=True)
            self.assertEqual(float(offline.properties["Mdm"]), float(online.properties["Mdm"]))
            validate_catalogs(self, online, offline, number, verbose=True)


class PublicArchiveValidation(unittest.TestCase):
    """Compare public methods with independent official products and replay without HTTP."""

    def setUp(self):
        self.key = os.getenv("TNG_API_KEY") or dotenv_values(".env", encoding="utf-8-sig").get("TNG_API_KEY")
        if not self.key:
            self.skipTest("Requires a TNG API key")

    def open(self, run, offline=False):
        return Snapshot.from_api(run, 99, api_key=self.key, progress=False, offline=offline)

    def test_original_illustris(self):
        from AnastrisTNG.illustris_python.sublink import numMergers
        def open_snapshot(offline=False):
            return Snapshot.from_api("Illustris-3", 135, api_key=self.key, progress=False, offline=offline)
        snap = open_snapshot()
        source = snap._source
        number = 10

        def reference(endpoint):
            with source._open(endpoint, "independent Illustris reference") as response:
                return h5py.File(io.BytesIO(response.read()), "r")

        sub = snap.load_particle(number, groupType="Subhalo")
        with reference(f"subhalos/{number}/cutout.hdf5?gas=Density,Volume,ParticleIDs&stars=Masses") as ref:
            for alias, field in (("rho", "Density"), ("Volume", "Volume"), ("iord", "ParticleIDs")):
                np.testing.assert_array_equal(sub.gas[alias], ref["PartType0"][field][:])
            np.testing.assert_array_equal(sub.star["mass"], ref["PartType4/Masses"][:])
        group_mass = snap.halos_GC("GroupMass")
        submass = snap.subhalos_GC("SubhaloMass")
        host = int(sub.GC["SubhaloGrNr"])
        with source._raw_groupcat(0) as raw:
            chunks = int(raw["Header"].attrs["NumFiles"])
            self.assertEqual(len(group_mass), raw["Header"].attrs["Ngroups_Total"])
            self.assertEqual(len(submass), raw["Header"].attrs["Nsubgroups_Total"])
            central = int(raw["Group/GroupFirstSub"][host])
        group_start = sub_start = 0
        for chunk in range(chunks):
            with source._raw_groupcat(chunk) as raw:
                groups = raw["Group/GroupMass"][:]
                subs = raw["Subhalo/SubhaloMass"][:]
                np.testing.assert_array_equal(group_mass[group_start:group_start+len(groups)], groups)
                np.testing.assert_array_equal(submass[sub_start:sub_start+len(subs)], subs)
                group_start += len(groups)
                sub_start += len(subs)
        self.assertEqual((group_start, sub_start), (len(group_mass), len(submass)))
        small_halo = 10000
        dm = snap.load_particle(small_halo, groupType="Halo").dm
        with reference(f"halos/{small_halo}/cutout.hdf5?dm=ParticleIDs") as ref:
            np.testing.assert_array_equal(dm["iord"], ref["PartType1/ParticleIDs"][:])
            self.assertGreater(len(dm), 0)
            np.testing.assert_array_equal(dm["mass"], np.full(len(dm), float(snap.properties["Mdm"])))
        galaxy = snap.galaxy_evolution(number, fields="SubhaloMass", physical_units=False)
        halo = snap.halo_evolution(host, fields="Group_M_Crit200", physical_units=False)
        for obj, track, pairs in [
                (number, galaxy, [("SubfindID", "SubfindID"), ("SubhaloMass", "SubhaloMass")]),
                (central, halo, [("HaloID", "SubhaloGrNr"), ("Group_M_Crit200", "Group_M_Crit200")])]:
            with reference(f"subhalos/{obj}/sublink/mpb.hdf5") as ref:
                np.testing.assert_array_equal(track["SnapNum"], ref["SnapNum"][:])
                for output, field in pairs:
                    np.testing.assert_array_equal(track[output], ref[field][:])
        opts = dict(subID=number, massPartType="dm", minMassRatio=0.01)
        history = snap.merger_history(**opts)
        with reference(f"subhalos/{number}/sublink/full.hdf5") as ref:
            keys = ["SubhaloID", "FirstProgenitorID", "NextProgenitorID", "MainLeafProgenitorID", "SubhaloMassType"]
            self.assertEqual(history["numMergers"], numMergers({k: ref[k][:] for k in keys}, 0.01, "dm"))
            self.assertEqual(len(history["MergerEvents"]), 136)
        with patch("AnastrisTNG.TNGapi.urlopen", side_effect=AssertionError("unexpected HTTP")):
            offline = open_snapshot(True)
            replay = offline.load_particle(number)
            for alias in ("rho", "Volume", "iord"):
                np.testing.assert_array_equal(replay.gas[alias], sub.gas[alias])
            np.testing.assert_array_equal(replay.star["mass"], sub.star["mass"])
            dm_again = offline.load_particle(small_halo, groupType="Halo").dm
            np.testing.assert_array_equal(dm_again["iord"], dm["iord"])
            np.testing.assert_array_equal(dm_again["mass"], dm["mass"])
            np.testing.assert_array_equal(offline.halos_GC("GroupMass"), group_mass)
            np.testing.assert_array_equal(offline.subhalos_GC("SubhaloMass"), submass)
            for result, again in [(history, offline.merger_history(**opts)),
                    (galaxy, offline.galaxy_evolution(number, fields="SubhaloMass", physical_units=False)),
                    (halo, offline.halo_evolution(host, fields="Group_M_Crit200", physical_units=False))]:
                for k in result:
                    np.testing.assert_array_equal(again[k], result[k])
        print(f"Illustris-3/135 subhalo {number}, {len(sub.gas)} gas cells; {len(group_mass)} halos, "
              f"{len(submass)} subhalos; galaxy {galaxy['count']} epochs, host {host} "
              f"{halo['count']} epochs, {history['numMergers']} mergers.", flush=True)

    def test_catalog_columns(self):
        snap = self.open("TNG50-4")
        group = snap.halos_GC(["GroupMass", "GroupLenType"])
        sub = snap.subhalos_GC("SubhaloMass")
        group_start = sub_start = chunk = 0
        chunks = 1
        while chunk < chunks:
            endpoint = f"../../files/groupcat-99.{chunk}.hdf5"
            with snap._source._open(endpoint, "raw catalog reference") as r:
                with h5py.File(io.BytesIO(r.read())) as f:
                    header = f["Header"].attrs
                    chunks = header["NumFiles"]
                    group_end = group_start + header["Ngroups_ThisFile"]
                    sub_end = sub_start + header["Nsubgroups_ThisFile"]
                    if group_end > group_start:
                        for name in ("GroupMass", "GroupLenType"):
                            np.testing.assert_array_equal(group[name][group_start:group_end], f["Group"][name][:])
                    if sub_end > sub_start:
                        np.testing.assert_array_equal(sub[sub_start:sub_end], f["Subhalo/SubhaloMass"][:])
                    group_start, sub_start = group_end, sub_end
            chunk += 1
        self.assertEqual(len(sub), sub_start)
        self.assertEqual(group["count"], group_start)
        with patch("AnastrisTNG.TNGapi.urlopen", side_effect=AssertionError("unexpected HTTP")):
            offline = self.open("TNG50-4", True)
            for name in ("GroupMass", "GroupLenType"):
                np.testing.assert_array_equal(offline.halos_GC(name), group[name])
            np.testing.assert_array_equal(offline.subhalos_GC("SubhaloMass"), sub)
        print(f"Catalogs: {len(sub)} subhalos and {len(group['GroupMass'])} groups", flush=True)

    def test_galaxy_and_host_evolution(self):
        snap = self.open("TNG50-4")
        merger_options = dict(subID=10, fields=["SubfindID", "SubhaloMassType", "SubhaloPos"],
                              massPartType="dm", minMassRatio=0.01)
        history = snap.merger_history(**merger_options)
        def reference(endpoint):
            with snap._source._open(endpoint, "official evolution reference") as response:
                return h5py.File(io.BytesIO(response.read()), "r")
        def compare(track, branch, id_name, ref_id, prefix):
            self.assertEqual(track["count"], len(branch["SnapNum"]))
            np.testing.assert_array_equal(track["SnapNum"], branch["SnapNum"][:])
            np.testing.assert_array_equal(track[id_name], branch[ref_id][:])
            for suffix in ("Mass", "Pos"):
                name = prefix + suffix
                np.testing.assert_array_equal(track[name], branch[name][:])

        with reference("subhalos/10/sublink/mpb.hdf5") as galaxy_ref:
            galaxy_first = snap.galaxy_evolution(10, fields="SubhaloMass", physical_units=False)
            galaxy_fields = ["SubhaloMass", "SubhaloPos", "SubhaloMassType", "SubhaloGrNr"]
            galaxy = snap.galaxy_evolution(10, fields=galaxy_fields, physical_units=False)
            compare(galaxy, galaxy_ref, "SubfindID", "SubfindID", "Subhalo")
            for field in ("SubhaloMassType", "SubhaloGrNr"):
                np.testing.assert_array_equal(galaxy[field], galaxy_ref[field][:])
            host = int(galaxy["SubhaloGrNr"][0])
            self.assertEqual(host, 0)
            np.testing.assert_array_equal(galaxy_first["SubhaloMass"], galaxy["SubhaloMass"])
        with reference("../../files/groupcat-99/?Group=GroupFirstSub") as root:
            central = int(root["Group/GroupFirstSub"][host])
        with reference(f"subhalos/{central}/sublink/mpb.hdf5") as host_ref:
            first = snap.halo_evolution(host, False, fields="GroupMass")
            halo = snap.halo_evolution(host, False, fields=["GroupMass", "GroupPos"])
            compare(halo, host_ref, "HaloID", "SubhaloGrNr", "Group")
            np.testing.assert_array_equal(first["GroupMass"], halo["GroupMass"])
            # Independent FoF catalogs at the start, middle and end of the track.
            for j in (0, len(halo["a"]) // 2, -1):
                n, i = int(halo["SnapNum"][j]), int(halo["HaloID"][j])
                for field in ("GroupMass", "GroupPos"):
                    with reference(f"../../files/groupcat-{n}/?Group={field}") as f:
                        np.testing.assert_array_equal(halo[field][j], f["Group"][field][i])
        with patch("AnastrisTNG.TNGapi.urlopen", side_effect=AssertionError("unexpected HTTP")):
            offline = self.open("TNG50-4", True)
            replay_history = offline.merger_history(**merger_options)
            for k in history:
                np.testing.assert_array_equal(history[k], replay_history[k])
            for raw, method, number, prefix in ((galaxy, offline.galaxy_evolution, 10, "Subhalo"),
                                                (halo, offline.halo_evolution, host, "Group")):
                fields = galaxy_fields if prefix == "Subhalo" else ["GroupMass", "GroupPos"]
                replay = method(number, fields=fields, physical_units=False)
                for k in raw:
                    np.testing.assert_array_equal(raw[k], replay[k])
                physical = method(number, fields=fields)
                expected = raw[fields[1]].view(np.ndarray) * raw["a"][:, None] / snap.properties["h"]
                np.testing.assert_allclose(physical[fields[1]], expected, rtol=2e-7)
                np.testing.assert_allclose(physical[fields[0]], np.asarray(raw[fields[0]]) / snap.properties["h"])
        print(f"Subhalo 10: {galaxy['count']} epochs; host {host}, central {central}: {halo['count']} epochs.", flush=True)

    def test_merger_history(self):
        from AnastrisTNG.illustris_python.sublink import numMergers
        snap = self.open("TNG50-4")
        number = 10
        options = dict(subID=number, massPartType="dm", minMassRatio=0.01)
        first = snap.merger_history(**options)
        fields = ["SubfindID", "SubhaloMassType", "SubhaloPos"]
        history = snap.merger_history(fields=fields, **options)
        self.assertEqual(first["numMergers"], history["numMergers"])
        with snap._source._open(f"subhalos/{number}/sublink/full.hdf5", "full tree reference") as r:
            with h5py.File(io.BytesIO(r.read())) as f:
                tree = snap._source.merger_tree(number, fields + ["SubhaloID", "NextProgenitorID",
                    "MainLeafProgenitorID", "FirstProgenitorID", "SnapNum"])
                row = np.flatnonzero((f["SnapNum"][:] == 99) & (f["SubfindID"][:] == number)).item()
                rows = slice(row, row + len(tree["SubhaloID"]))
                for k, col in tree.items():
                    np.testing.assert_array_equal(col, f[k][rows])
                ref = {k: f[k][rows] for k in tree}
                self.assertEqual(history["numMergers"], numMergers(ref, 0.01, "dm"))
                for branch in ("First", "Next"):
                    for i, sfid in enumerate(history[branch + "-SubfindID"]):
                        node = ((ref["SnapNum"] == history[branch + "-SnapNum"][i])
                                & (ref["SubfindID"] == sfid))
                        pos = ref["SubhaloPos"][np.flatnonzero(node).item()]
                        expected = pos * np.float64(history[branch + "-a"][i] / snap.properties["h"])
                        np.testing.assert_allclose(history[branch + "-SubhaloPos"][i], expected, rtol=2e-7)
        self.assertGreater(history["numMergers"], 0)
        with patch("AnastrisTNG.TNGapi.urlopen", side_effect=AssertionError("unexpected HTTP")):
            again = self.open("TNG50-4", True).merger_history(fields=fields, **options)
            for k in history:
                np.testing.assert_array_equal(again[k], history[k])
        print(f"Subhalo {number}: {len(tree['SnapNum'])} tree rows, {history['numMergers']} mergers", flush=True)

    def test_cluster_zoom(self):
        snap = self.open("TNG-Cluster")
        zoom = snap.load_particle(0, groupType="Zoom")
        self.assertTrue(zoom._gc_pending)
        ids, masses = zoom.bh["iord"], zoom.bh["mass"]
        self.assertTrue(zoom._gc_pending)
        catalog_mass = zoom.GC["GroupMass"]
        pos = 0
        for chunk in (0, 352):
            endpoint = f"../../files/snapshot-99.{chunk}.hdf5?bhs=ParticleIDs,Masses"
            with snap._source._open(endpoint, "reference zoom BHs") as r:
                with h5py.File(io.BytesIO(r.read())) as f:
                    ref = f["PartType5"]
                    n = len(ref["Masses"])
                    np.testing.assert_array_equal(ids[pos:pos+n], ref["ParticleIDs"][:])
                    np.testing.assert_array_equal(masses[pos:pos+n], ref["Masses"][:])
                    pos += n
                    print(f"Cluster chunk {chunk}: {n} black holes", flush=True)
        self.assertEqual(pos, len(zoom.bh))
        with patch("AnastrisTNG.TNGapi.urlopen", side_effect=AssertionError("unexpected HTTP")):
            offline = self.open("TNG-Cluster", True)
            again = offline.load_particle(0, groupType="Zoom")
            np.testing.assert_array_equal(again.bh["mass"], masses)
            np.testing.assert_array_equal(again.bh["iord"], ids)
            np.testing.assert_array_equal(again.GC["GroupMass"], catalog_mass)
            same_zoom = offline.load_particle(1, groupType="Zoom")
            np.testing.assert_array_equal(same_zoom.bh["iord"], ids)


class PublicApiComparison(unittest.TestCase):
    @unittest.skipUnless(all(os.getenv(name) for name in
                             ("TNG_API_KEY", "TNG_TEST_LOCAL_DIR", "TNG_TEST_SUBHALO_ID")),
                         "requires a TNG API key, local output and a selected gas-bearing subhalo")
    def test_public_cutouts_against_raw_snapshot(self):
        run = os.getenv("TNG_TEST_RUN", "TNG50-1")
        number = int(os.getenv("TNG_TEST_SNAP", "99"))
        sub_id = int(os.environ["TNG_TEST_SUBHALO_ID"])
        local = Snapshot(os.environ["TNG_TEST_LOCAL_DIR"], number)
        local.load_subhalo(sub_id)
        host = int(local.subhalos[sub_id].GC["SubhaloGrNr"])
        local.load_halo(host)
        self.assertGreater(len(local.gas), 0)
        with tempfile.TemporaryDirectory() as cache:
            for offline in (False, True):
                api = Snapshot.from_api(run, number, cache_dir=cache, offline=offline, progress=False)
                api.load_subhalo(sub_id)
                api.load_halo(host)
                for fam in local.families():
                    for key in ["iord", "SubhaloID", "pos", "mass"] + (["rho"] if fam.name == "gas" else []):
                        x, y = local[fam][key], api[fam][key]
                        self.assertEqual(x.units, y.units)
                        self.assertEqual(x.dtype, y.dtype)
                        if fam.name == "dm" and key == "mass":
                            # API mass_dm reports 15 significant digits for the
                            # constant also stored as a binary double in MassTable.
                            np.testing.assert_allclose(x, y, rtol=5e-15, atol=0)
                        else:
                            np.testing.assert_array_equal(x, y)
