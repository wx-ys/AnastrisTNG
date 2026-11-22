"""Exercise the public API against the same particles in raw files and API cutouts."""
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
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlsplit

import h5py
import numpy as np
from dotenv import dotenv_values

from AnastrisTNG.TNGapi import PART_TYPE, RemoteDataError
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
        self.bad_ids = False
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
        stack.enter_context(patch("AnastrisTNG.TNGapi.urlopen", side_effect=self.respond))

    def respond(self, request, timeout):
        url = urlsplit(request.full_url)
        if url.path == "/api/TNG50-1/":
            self.calls.append(("simulation", {}))
            return Response(json.dumps(dict(name="TNG50-1", mass_dm=self.dm_mass, boxsize=35000.,
                                            omega_0=.3089, omega_L=.6911, omega_B=.0486, hubble=.6774,
                                            softening_dm_comoving=.39, softening_dm_max_phys=.195)).encode())
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
        self.assertEqual(endpoint, "cutout.hdf5")
        out = io.BytesIO()
        with h5py.File(out, "w") as file:
            for fam, names in query.items():
                pt = {"gas": 0, "dm": 1, "stars": 4, "bhs": 5}[fam]
                for name in names[0].split(","):
                    values = self.samples[pt][name][self.members[kind][idx]]
                    if self.bad_ids and name == "ParticleIDs":
                        values = np.full_like(values, values[0])
                    file[f"PartType{pt}/{name}"] = values
        return Response(out.getvalue())

    def remote(self, **kwargs):
        return Snapshot.from_api("TNG50-1", 99, cache_dir=self.cache, progress=False, **kwargs)

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
                    FileOffsets_Group=[0, 2], FileOffsets_Subhalo=[0, 3], FileOffsets_SubLink=[0, 10])
                file["Group/GroupLenType"] = np.ones((2, 6), dtype=int)
                file["Group/GroupMass"] = [10 + chunk, 20 + chunk]
                if chunk == 1:
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
        self.assertEqual(source.catalog("Halo", 3)["GroupLenType"], [1] * 6)
        np.testing.assert_array_equal(source.merger_tree(4, ["SnapNum"], only_mpb=True)["SnapNum"], [135, 127])
        np.testing.assert_array_equal(source.merger_tree(4, ["SnapNum"])["SnapNum"], [135, 127, 129])
        with self.assertRaisesRegex(ValueError, "no SubLink tree"):
            source.merger_tree(3, ["SnapNum"])

    def test_catalog_gateway_timeout_and_offline_reopen(self):
        requests = []

        def gateway(request, timeout):
            path = urlsplit(request.full_url).path
            requests.append(path)
            if path.endswith("/info.json"):
                raise HTTPError(request.full_url, 504, "Gateway Time-out", {}, None)
            if "/files/" in path:
                filename = path.split("/")[-1]
                content = io.BytesIO()
                with h5py.File(content, "w") as file:
                    if filename == "offsets.99.hdf5":
                        file["FileOffsets/Subhalo"] = [0, 1]
                    else:
                        self.assertIn(filename, ("groupcat-99.0.hdf5", "groupcat-99.1.hdf5"))
                        i = int(filename.split(".")[1])
                        file.create_group("Header")
                        for key, values in self.catalog["Subhalo"].items():
                            file["Subhalo/" + key] = [values[i]]
                return Response(content.getvalue())
            return self.respond(request, timeout)

        with patch("AnastrisTNG.TNGapi.urlopen", side_effect=gateway):
            snap = self.remote(api_key="test")
            sub = snap.load_particle(1, groupType="Subhalo", decorate=True)
            ids = np.array(sub.gas["iord"])
            density = sub.gas["rho"].copy()
            sub.physical_units()
            self.assertEqual(sub.GC["SubhaloGrNr"], 1)
            self.assertEqual(sum(path.endswith("info.json") for path in requests), 1)
            self.assertEqual(sum("/files/" in path for path in requests), 3)
        with patch("AnastrisTNG.TNGapi.urlopen", side_effect=AssertionError("unexpected HTTP")):
            replay = self.remote(offline=True).load_particle(1)
            np.testing.assert_array_equal(replay.gas["iord"], ids)
            np.testing.assert_array_equal(replay.gas["rho"], density)

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

    def test_repeated_download_host_403_uses_authenticated_groupcat(self):
        snap = self.remote(api_key="test")
        paths = []
        def download_refused(req, timeout):
            path = urlsplit(req.full_url).path
            paths.append(path)
            if path.endswith("info.json"):
                raise HTTPError("https://data-eu.tng-project.org/download", 403, "Forbidden", {}, None)
            if path.endswith("/files/groupcat-99.0.hdf5"):
                self.assertEqual(req.get_header("Api-key"), "test")
                return Response((self.local / "groups_099/groups_099.0.hdf5").read_bytes())
            return self.respond(req, timeout)
        with patch("AnastrisTNG.TNGapi.urlopen", side_effect=download_refused):
            sub = snap.load_particle(0)
            expected = np.array(sub.gas["rho"])
            sub.physical_units()
            self.assertEqual(sub.GC["SubhaloGrNr"], 0)
        self.assertEqual(sum(p.endswith("info.json") for p in paths), 2)
        self.assertEqual(sum("/files/" in p for p in paths), 1)
        with patch("AnastrisTNG.TNGapi.urlopen", side_effect=AssertionError("unexpected HTTP")):
            np.testing.assert_array_equal(self.remote(offline=True).load_particle(0).gas["rho"], expected)

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
        self.assertEqual(self.calls, [("", {}), ("simulation", {})])
        for snap in (local, online):
            snap.load_subhalo(1)  # Keep this unrelated object across the masked merge.
            snap.load_subhalo(0)
            snap.load_halo(0)
            np.testing.assert_array_equal(snap.gas["SubhaloID"], [1, 0, 0, -1])
            np.testing.assert_array_equal(snap.gas["iord"], self.samples[0]["ParticleIDs"][[3, 0, 1, 2]])
        self.compare(local, online)
        supplemental = [q for path, q in self.calls if path == "halos/0/cutout.hdf5"
                        and "Density" in q.get("gas", [""])[0]]
        self.assertEqual(supplemental, [{"gas": ["Density"]}])
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
        snap.load_halo(0)
        snap.load_subhalo(0)
        np.testing.assert_array_equal(snap.gas["SubhaloID"], [0, 0, -1])
        offline = self.remote(offline=True).load_particle(0)
        with self.assertRaisesRegex(RemoteDataError, "subhalos/0.*gas.Potential"):
            _ = offline.gas["pot"]
        self.assertNotIn("pot", offline.gas.keys())

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
        queries = [q for path, q in self.calls if path.endswith("cutout.hdf5")]
        self.assertEqual(queries, [{"gas": ["Density"]}])

    def test_inconsistent_cutout_can_retry(self):
        snap = self.remote(api_key="test").load_particle(0)
        _ = snap.gas["rho"]
        self.bad_ids = True
        with self.assertRaisesRegex(RemoteDataError, "ParticleIDs"):
            _ = snap.gas["iord"]
        self.bad_ids = False
        np.testing.assert_array_equal(snap.gas["iord"], self.samples[0]["ParticleIDs"][:2])
        self.assertFalse(list(self.cache.rglob("*.part")))

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
        self.assertEqual(sum("Density" in str(q) for _, q in self.calls), 1)

    def test_catalog_validation_workflow(self):
        online = self.remote(api_key="test")
        offline = self.remote(offline=True)
        sub_id = select_gas_subhalo(online)
        validate_catalogs(self, online, offline, sub_id, verbose=False)

    def test_truncated_download_and_progress_without_length(self):
        snap = self.remote(api_key="test").load_particle(0)
        _ = snap.gas["iord"]
        def truncated(request, timeout):
            response = self.respond(request, timeout)
            response.headers["Content-Length"] = str(len(response.getvalue()) + 100)
            return response
        with patch("AnastrisTNG.TNGapi.urlopen", side_effect=truncated):
            with self.assertRaisesRegex(RemoteDataError, "Incomplete download"):
                _ = snap.gas["rho"]
        self.assertFalse(list(self.cache.rglob("*.part")))

        def no_length(request, timeout):
            response = self.respond(request, timeout)
            response.headers.clear()
            return response
        with patch("AnastrisTNG.TNGapi.urlopen", side_effect=no_length), patch("AnastrisTNG.TNGapi.tqdm") as bar:
            reopened = Snapshot.from_api("TNG50-1", 99, cache_dir=self.cache, api_key="test", progress=True)
            sub = reopened.load_particle(0)
            bar.assert_not_called()
            np.testing.assert_array_equal(sub.gas["rho"], self.samples[0]["Density"][:2])
            self.assertIsNone(bar.call_args.kwargs["total"])
            self.assertFalse(bar.call_args.kwargs["disable"])
            bar.return_value.__enter__.return_value.update.assert_called()

    def test_standalone_lazy_fields(self):
        snap = self.remote(api_key="test")
        sub = snap.load_particle(0)
        self.assertEqual([path for path, _ in self.calls], ["", "simulation", "subhalos/0/"])
        for key in ["pos", "vel", "iord", "mass"]:
            self.assertNotIn(key, sub.keys())
        np.testing.assert_array_equal(sub.dm["mass"], np.full(2, self.dm_mass))
        density = sub.gas["rho"]
        np.testing.assert_array_equal(density, self.samples[0]["Density"][:2])
        np.testing.assert_array_equal(sub.gas["iord"], self.samples[0]["ParticleIDs"][:2])
        self.assertEqual([q for path, q in self.calls if path.endswith(".hdf5")],
                         [{"gas": ["Density"]}, {"gas": ["ParticleIDs"]}])
        pos = sub["pos"]
        self.assertEqual(set(self.calls[-1][1]), {"gas", "stars", "dm", "bhs"})
        self.assertEqual(set(v[0] for v in self.calls[-1][1].values()), {"Coordinates"})
        for fam, pt in PART_TYPE.items():
            np.testing.assert_array_equal(getattr(sub, fam)["pos"], self.samples[pt]["Coordinates"][:2])
        self.assertEqual(len(pos), len(sub))
        before = len(self.calls)
        ids = sub["iord"]
        self.assertEqual(len(ids), len(sub))
        self.assertEqual(ids.dtype, np.uint64)
        self.assertEqual({fam for _, q in self.calls[before:] for fam in q}, {"dm", "stars", "bhs"})
        self.assertTrue(all(names == ["ParticleIDs"] for _, q in self.calls[before:] for names in q.values()))
        self.assertNotIn("vel", sub.keys())

    def test_standalone_configured_fields_and_eager_opt_in(self):
        snap = self.remote(api_key="test")
        snap.load_particle_para["star_fields"] = ["Potential"]
        halo = snap.load_particle(0, groupType="Halo", decorate=False)
        self.assertEqual(self.calls[-1][1], {"stars": ["Potential"]})
        self.assertNotIn("pos", halo.keys())
        np.testing.assert_array_equal(halo.star["pot"], self.samples[4]["Potential"][:3])
        full = snap.load_particle(1, lazy=False)
        self.assertIn("pos", full.keys())
        self.assertIn("vel", full.keys())
        self.assertIn("ParticleIDs", self.calls[-1][1]["dm"][0])

    def test_independent_field_omits_ids_and_reopens_offline(self):
        online = self.remote(api_key="test")
        mass = online.read_particle_field(0, "star", "mass")
        self.assertEqual(self.calls[-1][1], {"stars": ["Masses"]})
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

    def test_independent_field_partial_local_and_failed_download(self):
        with h5py.File(self.local / "snapdir_099/snap_099.0.hdf5", "r+") as f:
            del f["PartType4/ParticleIDs"]
        local = self.remote(local_dir=self.local, offline=True)
        np.testing.assert_array_equal(local.read_particle_field(0, "star", "mass"),
                                      self.samples[4]["Masses"][:2])
        self.assertFalse(self.calls)
        remote = self.remote(api_key="test")
        remote._source.particle_counts("Subhalo", 0)
        out = io.BytesIO()
        with h5py.File(out, "w") as f:
            f["PartType4/Masses"] = [1.]
        with patch("AnastrisTNG.TNGapi.urlopen", return_value=Response(out.getvalue())):
            with self.assertRaisesRegex(RemoteDataError, "incomplete field"):
                remote.read_particle_field(0, "star", "mass")
        self.assertFalse(list(self.cache.rglob("Masses.hdf5")))
        self.assertFalse(list(self.cache.rglob("*.part")))
        np.testing.assert_array_equal(remote.read_particle_field(0, "star", "mass"),
                                      self.samples[4]["Masses"][:2])

    def test_cached_fields_without_ids_then_merge(self):
        snap = self.remote(api_key="test")
        sub = snap.load_particle(0)
        mass = sub.star["mass"].copy()
        rho = sub.gas["rho"].copy()
        replay = self.remote(offline=True).load_particle(0)
        np.testing.assert_array_equal(replay.star["mass"], mass)
        with self.assertRaisesRegex(RemoteDataError, "star.ParticleIDs"):
            _ = replay.star["iord"]
        snap.load_subhalo(0)
        stars = self.calls[-1][1]["stars"][0].split(",")
        self.assertIn("ParticleIDs", stars)
        self.assertNotIn("Masses", stars)
        gas_ids = snap.gas["iord"].copy()
        snap.load_halo(0)
        for merged in (snap, self.remote(offline=True)):
            if merged is not snap:
                merged.load_subhalo(0)
                merged.load_halo(0)
            member = merged.gas["SubhaloID"] == 0
            np.testing.assert_array_equal(merged.gas["iord"][member], gas_ids)
            np.testing.assert_array_equal(merged.gas["rho"][member], rho)

    def test_legacy_cutouts_and_single_field_cache(self):
        snap = self.remote(api_key="test")
        sub = snap.load_particle(0)
        directory = snap._source.root / "subhalos/0"
        ids = self.samples[4]["ParticleIDs"][:2]
        mass = self.samples[4]["Masses"][:2]
        with h5py.File(directory / "first.hdf5", "w") as f:
            f["PartType4/ParticleIDs"] = ids
        with h5py.File(directory / "supplement.hdf5", "w") as f:
            f["PartType4/ParticleIDs"] = ids[::-1]
            f["PartType4/Masses"] = mass[::-1]
        (directory / "fields.json").write_text(json.dumps(
            {"star": {"ParticleIDs": "first.hdf5", "Masses": "supplement.hdf5"}}))
        legacy = directory / "single_fields/star"
        legacy.mkdir(parents=True)
        with h5py.File(legacy / "Coordinates.hdf5", "w") as f:
            f["PartType4/Coordinates"] = self.samples[4]["Coordinates"][:2]
        for obj in (sub, self.remote(offline=True).load_particle(0)):
            np.testing.assert_array_equal(obj.star["pos"], self.samples[4]["Coordinates"][:2])
            np.testing.assert_array_equal(obj.star["mass"], mass)
            np.testing.assert_array_equal(obj.star["iord"], ids)
        self.assertFalse(any(path.endswith("cutout.hdf5") for path, _ in self.calls))

    def test_summary_only_cache_and_deferred_catalog_units(self):
        snap = self.remote(api_key="test")
        sub = snap.load_particle(0)
        rho = sub.gas["rho"]
        sub.physical_units()
        self.assertIn("SubhaloID=0", repr(sub))
        self.assertEqual([path for path, q in self.calls],
                         ["", "simulation", "subhalos/0/", "subhalos/0/cutout.hdf5"])
        offline = self.remote(offline=True).load_particle(0)
        offline.physical_units(persistent=True)
        np.testing.assert_array_equal(offline.gas["rho"], rho)
        with self.assertRaisesRegex(RemoteDataError, "Offline cache miss"):
            _ = offline.GC
        reference = Snapshot(str(self.local), 99).load_particle(0)
        reference.physical_units()
        for obj in (sub, offline):
            self.assertEqual(obj.GC["SubhaloMass"].units, reference.GC["SubhaloMass"].units)
            np.testing.assert_allclose(obj.GC["SubhaloMass"], reference.GC["SubhaloMass"], rtol=1e-6)
        self.assertEqual([path for path, q in self.calls].count("subhalos/0/info.json"), 1)

    def test_incomplete_summary_uses_full_catalog(self):
        snap = self.remote(api_key="test")
        def incomplete(request, timeout):
            if urlsplit(request.full_url).path.endswith("subhalos/0/"):
                return Response(b'{"id":0}')
            return self.respond(request, timeout)
        with patch("AnastrisTNG.TNGapi.urlopen", side_effect=incomplete):
            sub = snap.load_particle(0)
        self.assertEqual(len(sub), 8)
        self.assertFalse(list(self.cache.rglob("counts.json")))
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
                f["PartType3/TracerID"] = np.arange(3, dtype="uint64") + (2**60 + 3 * chunk)
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
        with h5py.File(path, "w") as f:
            f["SubhaloID"] = [300, 301, 302, 303, 304]
            f["FirstProgenitorID"] = [301, 302, -1, -1, -1]
            f["NextProgenitorID"] = [-1, -1, 303, -1, -1]
            f["MainLeafProgenitorID"] = [302, 302, 302, 303, 304]
            f["SubfindID"] = [0, 0, 1, 2, 0]
            f["SnapNum"] = [99, 99, 98, 98, 98]
            f["SubhaloMassType"] = np.ones((5, 6))
            f["SubhaloPos"] = np.arange(15).reshape(5, 3).astype("f4")
        snap._source.local = str(self.local)
        (snap._source.root.parent / "scale_factors.json").write_text('{"98":0.9}')
        snap._source.offline = True
        history = snap.merger_history(0, fields=["SubhaloPos"], minMassRatio=.1)
        self.assertEqual(history["numMergers"], 1)
        np.testing.assert_allclose(history["First-SubhaloPos"][0], np.arange(6, 9) * .9 / .6774)
        np.testing.assert_allclose(history["Next-a"], [.9])
        snap._source.local = None
        again = snap.merger_history(0, fields=["SubhaloPos"], minMassRatio=.1)
        for key in history:
            np.testing.assert_array_equal(history[key], again[key])
        with patch("AnastrisTNG.TNGapi.urlopen", side_effect=AssertionError("unexpected HTTP")):
            gal = snap.galaxy_evolution(0, fields="SubhaloPos")
            np.testing.assert_array_equal(gal["SnapNum"], [99, 98])
            np.testing.assert_allclose(gal["SubhaloPos"][1], history["First-SubhaloPos"][0])
            self.assertEqual(gal["count"], 2)

    def test_evolution_reuses_branch_and_catalog_columns(self):
        snap = self.remote(api_key="test")
        source = snap._source
        directory = source.root / "subhalos/0/sublink"
        directory.mkdir(parents=True)
        (directory / "layout.json").write_text("[0, 1, 3, 100]")
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
        halo = snap.halo_evolution(1, False, fields=["GroupMass", "GroupPos"])
        np.testing.assert_array_equal(halo["HaloID"], [1, 0])
        np.testing.assert_array_equal(halo["GroupMass"], [4, 3])
        self.assertFalse((directory / "mpb/GroupMass.npy").exists())
        np.testing.assert_array_equal(np.load(directory / "mpb/GroupPos.npy"), halo["GroupPos"])
        # Exercise the same positional calls against the original local file layout.
        local_tree = self.local / "trees/SubLink/tree_extended.0.hdf5"
        local_tree.parent.mkdir(parents=True)
        shutil.copyfile(shard, local_tree)
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
        shard.unlink()
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
            with patch.object(online._source, "_stream", wraps=online._source._stream) as download:
                for name, raw in fields:
                    values[name] = sub.star[name].copy()
                    self.assertEqual(download.call_args.args[0], "subhalos/12/cutout.hdf5?stars=" + raw)
                self.assertEqual(download.call_count, len(fields))
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
                        np.testing.assert_array_equal(x, y)
