# AnastrisTNG

English · [简体中文](README_CN.md)

AnastrisTNG brings particles, group catalogs and SubLink merger trees from Illustris, IllustrisTNG and TNG-Cluster into pynbody. You can use the same analysis code with local files or the public TNG API.

## Installation

Use Python 3.8 or later.

```shell
git clone https://github.com/wx-ys/AnastrisTNG.git
cd AnastrisTNG
python -m pip install -e .
```

## Read your first subhalo

For API access, set `TNG_API_KEY` in your environment or put `TNG_API_KEY=your_key_here` in a `.env` file in your working directory. [Get your key from your TNG account](https://www.tng-project.org/users/profile/).

This example reads subhalo 12 at redshift zero in TNG50-1. Use snapshot 99 for TNG and 135 for the original Illustris runs at this redshift.

```python
from AnastrisTNG import Snapshot

snap = Snapshot.from_api("TNG50-1", 99)
sub = snap.load_particle(12)
sub.physical_units()

spin = sub.GC["SubhaloSpin"]
print(sub.gas["mHI"].sum())
```

`sub.gas["mHI"]` gives the atomic hydrogen mass per gas cell. For subhalo 12 the sum is about 2.02 × 10⁸ solar masses. Calling `physical_units()` also applies to particle fields you load later. Use `sub.GC` for one object's catalog properties, or `snap.subhalos_GC("SubhaloGrNr")` to read a column for all subhalos.

The full example also orients the galaxy face-on and measures the fraction of stellar kinetic energy in rotation within 30 kpc. Replace `Your_API_Key_Here` in the notebook with your key or remove the `api_key` argument to use your configured key. To hide download progress, pass `progress=False` to `from_api`. [Open the API usage notebook](examples/AnastrisTNG_api_usage.ipynb).

## Other ways to open the data

- Open **local files** with `snap = Snapshot("/data/TNG50-1/output", 99)`. The `output/` directory contains snapshots and group catalogs. For public TNG downloads, put the matching offsets file in `../postprocessing/offsets/` relative to `output/`. Local merger histories also require the SubLink trees.
- To **use a partial download**, pass `local_dir="/data/TNG50-1/output"` to `Snapshot.from_api(...)`. Missing data are fetched from the API.
- To **work offline**, reopen previously read data with `Snapshot.from_api("TNG50-1", 99, offline=True)`. The default cache is `~/.cache/AnastrisTNG`. If you chose another `cache_dir`, reuse it.

## Analysis notebooks

| What you want to do | Notebook |
| --- | --- |
| Load halos, subhalos or a TNG-Cluster zoom | [Quick start](examples/AnastrisTNG_quick_start-cn.ipynb) |
| Measure galaxy properties and orient a galaxy face-on | [Galaxy properties](examples/AnastrisTNG_galaxy_func-cn.ipynb) · [Face-on views](examples/AnastrisTNG_galaxy_face_on-cn.ipynb) |
| Measure radial and vertical profiles | [Profiles](examples/AnastrisTNG_profile-cn.ipynb) |
| Follow a merger tree or trace particles across snapshots | [Merger histories](examples/AnastrisTNG_evolution_history-cn.ipynb) · [Particle tracing](examples/AnastrisTNG_trace_particles.ipynb) |
| Use optional analysis packages | [GalaxyPose star formation histories](examples/AnastrisTNG_galpos_SFH-cn.ipynb) · [pynbody-extras](examples/AnastrisTNG_pynext_example-cn.ipynb) |

For the GalaxyPose notebook, install the optional package with `python -m pip install galpos`. [GalaxyPose](https://github.com/GalaxySimAnalytics/GalaxyPose)

Maintained by [Shuai Lu](https://github.com/wx-ys) under the [MIT license](LICENSE). Built on [pynbody](https://github.com/pynbody/pynbody), [illustris_python](https://github.com/illustristng/illustris_python) and [pytreegrav](https://github.com/mikegrudic/pytreegrav).
