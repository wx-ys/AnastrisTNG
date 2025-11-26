## Introduction
AnastrisTNG is a python package for processing and analyzing the cosmological simulation [IllustrisTNG](https://www.tng-project.org/).
It supports Illustris, TNG50, TNG100, TNG300, and **TNG-Cluster (recently released)**.
 

>[!NOTE]
>analysis utilities are being migrated and updated in the [pynbody-extras](https://github.com/wx-ys/pynbody-extras) project: https://github.com/wx-ys/pynbody-extras.
>Usage example: `examples/AnastrisTNG_pynext-cn.ipynb`
>
>spatially-resolved star formation history tools in the [GalaxyPose](https://github.com/GalaxySimAnalytics/GalaxyPose) project: https://github.com/GalaxySimAnalytics/GalaxyPose
>Usage example: `examples/AnastrisTNG_galpos_SFH-cn.ipynb`

## Installation

Python version >= 3.8:
```
git clone https://github.com/wx-ys/AnastrisTNG.git
```

Install this in editable mode.
```
cd AnastrisTNG
pip install -e .
```

Recommended dependencies (these are installed automatically when possible):
- `numpy`, `scipy`, `h5py`, `tqdm`, `six`, `numba`
- `pynbody >= 1.4.0`


>[!NOTE]
>for Python 3.8 users:
>Direct `pip install pynbody` may fail on some systems. If you encounter issues, run:
>
> ```bash
> pip install --upgrade pip setuptools wheel
> pip install "numpy<1.26" "cython<3.0"
> pip install --no-build-isolation pynbody
> ```
> Then install AnastrisTNG:
> ```bash
> pip install -e .
> ```


## Features


* __Supports  Illustris, TNG50, TNG100, TNG300, and **TNG-Cluster (recently released)**, with units handled.__

* __Rapid exploration of galaxy and halo evolution, including merger trees and histories.__

* __Coordinate-consistent analysis of interacting galaxies.__

* __GroupCatalog analysis and particle tracing (including gas).__

* __Radial and vertical profile tools for 3D properties and 2D projections.__
<center>
      <img src="./images/radial_profile.png"  height = "300">
      <img src="./images/vertical_profile.png" height = "300">
</center>


* __Fast (tens of seconds) inspection of spatially-resolved star formation histories (SFH). (not public; robust version at [GalaxyPose](https://github.com/GalaxySimAnalytics/GalaxyPose)). If interested, contact lushuai@stu.xmu.edu.cn.__
![image](./images/TNG50_SFH_Subhalo_424289.png)

## Usage

Basic example:
```python
from AnastrisTNG import TNGsimulation 
BasePath = 'filepath'  # Path to simulation data
snap=99                # Snapshot number

Snapshot=TNGsimulation.Snapshot(BasePath,snap) # use snapshot 99

Snapshot.load_halo(400)     # load halo with ID 400
Snapshot.load_subhalo(8)    # load subhalo with ID 8

# load a single subhalo/galaxy (decorate converts to helper object)
sub = Snapshot.load_particle(ID=10, groupType='Subhalo', decorate=True)
sub.physical_units() #in physical units
sub.face_on(alignwith='star',rmax=8) # Align face-on by stellar angular momentum within 8 kpc
```

See [examples](examples) for more:
- [quick_start](examples/AnastrisTNG_quick_start-cn.ipynb): Quick start
- [galaxy_face_on](examples/AnastrisTNG_galaxy_face_on-cn.ipynb): Extract, align, and image a galaxy
- [galaxy_func](examples/AnastrisTNG_galaxy_func-cn.ipynb): Utility functions
- [galaxy_profile](examples/AnastrisTNG_galaxy_profile-cn.ipynb): Radial profile analysis
- [galaxy_evolution](examples/AnastrisTNG_galaxy_evolution-cn.ipynb): Galaxy evolution and merger history
- [galpos_SFH](examples/AnastrisTNG_galpos_SFH-cn.ipynb): Spatially-resolved star formation history with GalaxyPose
- [pynbody-extras](examples/AnastrisTNG_pynext-cn.ipynb): Example using `pynbody-extras` utilities

## Maintainers

[@wx-ys](https://github.com/wx-ys).


## License

[MIT](LICENSE) © Shuai Lu

## Acknowledgments
* [illustris_python](https://github.com/illustristng/illustris_python)
* [pytreegrav](https://github.com/mikegrudic/pytreegrav)
* [pynbody](https://github.com/pynbody/pynbody)