import numpy as np
from AnastrisTNG.TNGload import _split_runs, pynbody_to_hdf


def test_split_runs_single_contiguous():
    idx = np.arange(254750400, 254750400 + 68)  # 一段连续(如一个 halo 的 gas)
    runs = _split_runs(idx)
    assert len(runs) == 1
    np.testing.assert_array_equal(runs[0], idx)


def test_split_runs_two_blocks():
    a = np.arange(1000, 1010)
    b = np.arange(5000, 5005)
    idx = np.concatenate([a, b])             # 两段不相邻,保序
    runs = _split_runs(idx)
    assert len(runs) == 2
    np.testing.assert_array_equal(runs[0], a)
    np.testing.assert_array_equal(runs[1], b)


def test_split_runs_empty():
    assert _split_runs(np.array([], dtype=np.int64)) == []


def test_pynbody_to_hdf_maps_known_name():
    # 从一组真实 HDF 字段名反推 pynbody→HDF
    keys = ['Coordinates', 'Velocities', 'Masses', 'ParticleIDs', 'Density', 'Potential']
    m = pynbody_to_hdf(keys)
    assert m['pos'] == 'Coordinates'
    assert m['vel'] == 'Velocities'
    assert m['rho'] == 'Density'
    assert m['pot'] == 'Potential'
    # 未映射字段保持不变
    keys2 = ['ElectronAbundance']
    m2 = pynbody_to_hdf(keys2)
    assert m2['ElectronAbundance'] == 'ElectronAbundance'
