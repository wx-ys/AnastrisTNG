import os
import numpy as np
import pytest

BP = '/home/yxi/Simulation/sims/TNG50-1/output'
SNAP = 99
SMALL_GAS_SUBHALO = 3052   # gas=68 dm=390 star=487
DM_SUBHALO = 1000          # dm=3603 star=2


@pytest.fixture
def need_data():
    if not os.path.isdir(BP + '/snapdir_%03d' % SNAP):
        pytest.skip("TNG data not available at " + BP)
    return True


def test_loadable_keys_include_rho_and_pot(need_data):
    from AnastrisTNG.TNGsimulation import Snapshot
    snap = Snapshot(BP, SNAP)
    gkeys = snap.loadable_keys('gas')
    assert 'rho' in gkeys          # Density
    assert 'pot' in gkeys          # Potential
    dkeys = snap.loadable_keys('dm')
    assert 'pot' in dkeys          # dm 也有 Potential


def test_load_particle_returns_lazy_snap(need_data):
    from AnastrisTNG.TNGsimulation import Snapshot
    from AnastrisTNG.TNGload import _LazySnap
    from AnastrisTNG.illustris_python.snapshot import getSnapOffsets, loadSubset
    snap = Snapshot(BP, SNAP)
    f = snap.load_particle(SMALL_GAS_SUBHALO, groupType='Subhalo', decorate=False)
    assert isinstance(f, _LazySnap)
    assert len(f._loaded_index['gas']) == 68
    # 参考:直接读该子结构 gas Density
    sub = getSnapOffsets(BP, SNAP, SMALL_GAS_SUBHALO, 'Subhalo')
    ref = loadSubset(BP, SNAP, 'gas', ['Density'], subset=sub)['Density']
    # lazy:访问 f.g['rho'] 触发按需读取
    lazy = f.g['rho'].view(np.ndarray)
    assert lazy.shape == ref.shape
    np.testing.assert_allclose(lazy, ref)


def test_load_particle_lazy_dm_pot(need_data):
    from AnastrisTNG.TNGsimulation import Snapshot
    from AnastrisTNG.illustris_python.snapshot import getSnapOffsets, loadSubset
    snap = Snapshot(BP, SNAP)
    f = snap.load_particle(DM_SUBHALO, groupType='Subhalo', decorate=False)
    sub = getSnapOffsets(BP, SNAP, DM_SUBHALO, 'Subhalo')
    ref = loadSubset(BP, SNAP, 'dm', ['Potential'], subset=sub)['Potential']
    lazy = f.dm['pot'].view(np.ndarray)
    np.testing.assert_allclose(lazy, ref)
