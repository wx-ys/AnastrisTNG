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
