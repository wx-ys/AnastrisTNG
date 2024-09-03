## Introduction
AnastrisTNG is a python package for processing and analyzing the cosmological simulation [IllustrisTNG](https://www.tng-project.org/).

## Installation

```
git clone https://github.com/wx-ys/AnastrisTNG.git
cd AnastrisTNG
pip install .
```

## Usage


```python
from AnastrisTNG import TNGsimulation 
BasePath = 'filepath'       
snap=99  #snapshot

Snapshot=TNGsimulation.Snapshot(BasePath,snap)
Snapshot.load_halo(400)    #load a halo(id=400) data
```

## Maintainers

[@wx-ys](https://github.com/wx-ys).


## License

[MIT](LICENSE) © Shuai Lu

## Acknowledgments
* [illustris_python](https://github.com/illustristng/illustris_python)
* [pytreegrav](https://github.com/mikegrudic/pytreegrav)
* [pynbody](https://github.com/pynbody/pynbody)