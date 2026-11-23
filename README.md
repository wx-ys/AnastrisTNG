# AnastrisTNG

English · [简体中文](README_CN.md)

AnastrisTNG loads Illustris and IllustrisTNG particles, group catalogs and SubLink merger trees into pynbody for galaxy and halo analysis.

**Local files** — Open a run’s downloaded snapshot files and group catalogs with `Snapshot(output_dir, snap_number)`. For merger histories, also download the SubLink trees.

**Remote API** — Use `Snapshot.from_api(run, snap_number)` to read particle data, catalog fields and merger histories directly from the TNG servers. The reader downloads data on demand and caches it locally. Sign in to the TNG website to get your key from your [TNG account profile](https://www.tng-project.org/users/profile/).

Once the snapshot is open, both sources use the same analysis calls.

| Task | Entry point |
| --- | --- |
| Read particles and the catalog row for one halo or subhalo | `snap.load_particle(...)`, `sub.GC` |
| Load a subhalo and its host into one snapshot | `snap.load_subhalo(...)`, `snap.load_halo(...)` |
| Read columns from the full group catalog | `snap.halos_GC(...)`, `snap.subhalos_GC(...)` |
| Find mergers along a galaxy’s main branch | `snap.merger_history(...)` |
| Trace galaxy or halo evolution | `snap.galaxy_evolution(...)`, `snap.halo_evolution(...)` |
| Select a complete TNG-Cluster zoom | `snap.load_particle(..., groupType="Zoom")` |

Supported runs include Illustris-1 through Illustris-3, TNG50-1 through TNG50-4, TNG100-1 through TNG100-3, TNG300-1 through TNG300-3 and TNG-Cluster. The redshift-zero snapshot is **135 for original Illustris** and **99 for TNG**.

## Installation

Install from the repository with Python 3.8 or later.

```shell
git clone https://github.com/wx-ys/AnastrisTNG.git
cd AnastrisTNG
python -m pip install -e .
```

## Open a snapshot

**Local files.** Pass the run’s `output` directory containing `snapdir_099/` and `groups_099/`. For the TNG public file layout, place the matching offsets file in `../postprocessing/offsets/`.

```python
from AnastrisTNG import Snapshot

snap = Snapshot("/data/TNG50-4/output", 99)
```

**Remote API.** Set the environment variable `TNG_API_KEY`, or put this line in a `.env` file in the working directory of your Python session or notebook kernel.

```dotenv
TNG_API_KEY=your_key_here
```

```python
from AnastrisTNG import Snapshot

snap = Snapshot.from_api("TNG50-4", 99)
```

For original Illustris at redshift zero, use `Snapshot.from_api("Illustris-3", 135)`. If you already have some files, pass `local_dir="/data/TNG50-4/output"` to use them and fetch missing data from the API.

## Examples

### Read a subhalo’s gas density

```python
sub = snap.load_particle(22842)
sub.physical_units()
rho = sub.gas["rho"]
```

Accessing `rho` reads the gas `Density` rows for this subhalo. For local files and API reads, pynbody applies `physical_units()` to arrays loaded later as well. Use `persistent=False` to convert only the arrays already in memory. Access `sub.gas["iord"]` when you need gas particle IDs.

### Load a subhalo and its host

```python
snap.load_subhalo(22842)
host_id = int(snap.subhalos[22842].GC["SubhaloGrNr"])
snap.load_halo(host_id)
rho_in_host = snap.gas["rho"][snap.gas["SubhaloID"] == 22842]
```

### Read catalog columns and trace evolution

```python
halo_mass = snap.halos_GC("GroupMass")
host_ids = snap.subhalos_GC("SubhaloGrNr")
host_id = int(host_ids[10])

galaxy = snap.galaxy_evolution(10, fields=["SubhaloMass"])
host = snap.halo_evolution(host_id, fields=["GroupMass"])
mergers = snap.merger_history(10, massPartType="dm", minMassRatio=0.01)
```

The galaxy history follows subhalo 10. The host history follows the main branch of the host’s central subhalo. Use `SnapNum` to match epochs.

### Select a full TNG-Cluster zoom

Pass a FoF halo ID from the chosen snapshot to select its entire zoom region.

```python
cluster = Snapshot.from_api("TNG-Cluster", 99)
zoom = cluster.load_particle(0, groupType="Zoom")
bh_mass = zoom.bh["mass"]
```

### Reopen cached data offline

After reading the gas density online, reopen the same run and snapshot. The default cache is `~/.cache/AnastrisTNG`, created automatically on the first cache write. If you set `cache_dir`, use the same directory in both sessions.

```python
offline = Snapshot.from_api("TNG50-4", 99, offline=True)
rho = offline.load_particle(22842).gas["rho"]
```

Halo and subhalo particle reads share losslessly compressed rows by official snapshot offsets. The reader combines valid cached rows and local snapshot data, then requests missing fields in one cutout. On a cold read it can start the cutout while looking up offsets. If the offset service is temporarily unavailable, an online read can still return the cutout. Sharing those rows across objects or reopening them offline requires a successful offset lookup.

The progress display reports waits, transferred bytes, cache work and array assembly. For transfers of unknown size it shows bytes and speed. Fast cache hits stay quiet. Set `progress=False` in `from_api` to hide progress.

New blocks in the shared particle row cache include a checksum. The reader can repair damaged rows from a local mirror or cutout and reuse the other blocks. Older uncompressed blocks remain readable and can be checked for shape and byte length.

Catalog and SubLink data also persist across sessions. A lookup such as `sub.GC["SubhaloSpin"]` first checks for an exact cached value. On a miss, the official `info.json` endpoint returns the complete record in one response and makes later fields available from cache. If that service times out, the reader fetches only the HDF5 blocks needed to locate and read the requested field. Use `dict(sub.GC)` or `sub.load_GC()` to read the whole record. If both sources fail, the exception includes both causes. For data previously held in legacy per-object cutouts, catalog or count files, or SubLink arrays, populate the shared cache online before reopening offline.

## Analysis notebooks

| Topic | Notebooks |
| --- | --- |
| Getting started | [Quick start](examples/AnastrisTNG_quick_start-cn.ipynb) · [Galaxy properties](examples/AnastrisTNG_galaxy_func-cn.ipynb) |
| Structure and imaging | [Face-on galaxy](examples/AnastrisTNG_galaxy_face_on-cn.ipynb) · [Radial and vertical profiles](examples/AnastrisTNG_profile-cn.ipynb) |
| Evolution | [Merger trees](examples/AnastrisTNG_evolution_history-cn.ipynb) · [Particle tracing](examples/AnastrisTNG_trace_particles.ipynb) |
| Extensions | [GalaxyPose star formation histories](examples/AnastrisTNG_galpos_SFH-cn.ipynb) · [pynbody-extras](examples/AnastrisTNG_pynext_example-cn.ipynb) |

<img src="images/radial_profile.png" alt="Radial profile" height="210"> <img src="images/vertical_profile.png" alt="Vertical profile" height="210">

Install the optional GalaxyPose integration with `python -m pip install galpos`. For questions about the original star formation history analysis, contact lushuai@stu.xmu.edu.cn. [GalaxyPose](https://github.com/GalaxySimAnalytics/GalaxyPose)

![Spatially resolved star formation history for a TNG50 galaxy](images/TNG50_SFH_Subhalo_424289.png)

## Tests

Run the reader tests locally. The second command compares particles, catalogs and evolution with public Illustris-3 data and requires an API key.

```shell
python -m unittest discover -s tests -p test_remote.py -k RemoteSnapshotTest -v
python -m unittest discover -s tests -p test_remote.py -k PublicArchiveValidation.test_original_illustris -v
```

## Project

Maintained by [Shuai Lu](https://github.com/wx-ys) under the [MIT license](LICENSE). Built on [pynbody](https://github.com/pynbody/pynbody), [illustris_python](https://github.com/illustristng/illustris_python) and [pytreegrav](https://github.com/mikegrudic/pytreegrav).
