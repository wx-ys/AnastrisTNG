'''
Load groupcatalog for Halo or Subhalo
'''

from pynbody import simdict
from pynbody.array import SimArray

from AnastrisTNG.illustris_python.groupcat import loadSingle, loadHalos, loadSubhalos
from AnastrisTNG.TNGunits import groupcat_units, halo_pa_name, subhalo_pa_name
from AnastrisTNG.TNGsnapshot import get_Snapshot_property


def get_Subhalo_property(BasePath, Snap, subhaloID):

    single = subhaloproperties(BasePath, Snap, subhaloID)
    Subhalo = simdict.SimDict()

    for i in single:
        Subhalo[subhalo_pa_name(i)] = single[i]
    Subhalo['ID'] = subhaloID
    snapshot = get_Snapshot_property(BasePath, Snap)
    for i in snapshot:
        Subhalo[i] = snapshot[i]
    return Subhalo


def get_groupcatalogs_pa(Base, Snap):
    groupcatalog = {}
    groupcatalog['halo'] = _get_Halo_pa(Base, Snap)
    groupcatalog['subhalo'] = _get_Subhalo_pa(Base, Snap)
    return groupcatalog


def _get_Subhalo_pa(BasePath, Snap):

    return list(loadSingle(BasePath, Snap, subhaloID=1).keys())


def _get_Halo_pa(BasePath, Snap):

    return list(loadSingle(BasePath, Snap, HaloID=1).keys())


def get_Halo_property(BasePath, Snap, haloID):

    single = haloproperties(BasePath, Snap, haloID)
    Halo1 = simdict.SimDict()

    for i in single:
        Halo1[halo_pa_name(i)] = single[i]
    Halo1['ID'] = haloID
    snapshot = get_Snapshot_property(BasePath, Snap)
    for i in snapshot:
        Halo1[i] = snapshot[i]
    return Halo1


def subhalosproperty(BasePath, Snap, fields):
    subhalos = loadSubhalos(BasePath, Snap, fields=fields)
    for i in subhalos:
        try:
            subhalos[i] = SimArray(subhalos[i], groupcat_units(i))
        except:
            continue
    return subhalos


def halosproperty(BasePath, Snap, fields):

    halos = loadHalos(BasePath, Snap, fields=fields)
    return halos


def subhaloproperties(BasePath, Snap, subhaloID):

    single = loadSingle(BasePath, Snap, subhaloID=subhaloID)
    for i in single:
        single[i] = SimArray(single[i], groupcat_units(i))

    return single


def haloproperties(BasePath, Snap, haloID):

    single = loadSingle(BasePath, Snap, haloID=haloID)
    for i in single:
        single[i] = SimArray(single[i], groupcat_units(i))

    return single
