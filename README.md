## Introduction
Anastris is a package for processing and analyzing data from the cosmological simulation IllustrisTNG.

## Installation

clone the repo(https://github.com/wx-ys/AnastrisTNG.git) and run ```python setup.py install``` from the repo directory.

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