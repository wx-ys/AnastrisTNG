'''
Load groupcatalog for Halo or Subhalo
'''
import numpy as np
from pynbody import simdict, family, units
from pynbody.array import SimArray
from pynbody.snapshot import new

from AnastrisTNG.illustris_python.groupcat import loadSingle, loadHalos, loadSubhalos, loadHeader
from AnastrisTNG.TNGunits import groupcat_units, halo_pa_name, subhalo_pa_name


##merge two simsnap and cover
def simsnap_cover(f1, f2):
    """
    Overwrites the data in simsnap f1 with data from simsnap f2.

    Parameters:
    -----------
    f1 : simsnap
        The target simsnap to be overwritten.
    f2 : simsnap
        The source simsnap providing the data.
    """
    for i in f1:
        del f1[i]
    f1._num_particles = len(f2)
    if len(f2.dm) > 0:
        f1._family_slice[family.get_family('dm')] = f2._family_slice[
            family.get_family('dm')
        ]
        for i in f1.dm:
            del f1.dm[i]
    if len(f2.s) > 0:
        f1._family_slice[family.get_family('star')] = f2._family_slice[
            family.get_family('star')
        ]
        for i in f1.s:
            del f1.s[i]
    if len(f2.g) > 0:
        f1._family_slice[family.get_family('gas')] = f2._family_slice[
            family.get_family('gas')
        ]
        for i in f1.g:
            del f1.g[i]
    if len(f2.bh) > 0:
        f1._family_slice[family.get_family('bh')] = f2._family_slice[
            family.get_family('bh')
        ]
        for i in f1.bh:
            del f1.bh[i]

    f1._create_arrays(["pos", "vel"], 3)
    f1._create_arrays(["mass"], 1)
    f1._decorate()
    if len(f1.dm) > 0:
        for i in f2.dm:
            f1.dm[i] = f2.dm[i]
    if len(f1.s) > 0:
        for i in f2.s:
            f1.s[i] = f2.s[i]
    if len(f1.g) > 0:
        for i in f2.g:
            f1.g[i] = f2.g[i]
    if len(f1.bh) > 0:
        for i in f2.bh:
            f1.bh[i] = f2.bh[i]


def simsnap_merge(f1, f2):
    """
    Merges twosimsnap f1 and f2 into a new simsnap f3.

    Parameters:
    -----------
    f1 : simsnap
        The first simsnap.
    f2 : simsnap
        The second simsnap.
    Returns:
    --------
    f3 : simsnap
        The new simsnap containing merged data from f1 and f2.
    """
    f3 = new(
        star=len(f1.s) + len(f2.s),
        gas=len(f1.g) + len(f2.g),
        dm=len(f1.dm) + len(f2.dm),
        bh=len(f1.bh) + len(f2.bh),
        order='dm,star,gas,bh',
    )
    if len(f3.s) > 0:
        if len(f1.s) == 0:
            for i in f2.s:
                f3.s[i] = f2.s[i]
        elif len(f2.s) == 0:
            for i in f1.s:
                f3.s[i] = f1.s[i]
        else:
            for i in f2.s:
                f3.s[i] = SimArray(np.append(f1.s[i], f2.s[i], axis=0), f2.s[i].units)

    if len(f3.dm) > 0:
        if len(f1.dm) == 0:
            for i in f2.dm:
                f3.dm[i] = f2.dm[i]
        elif len(f2.dm) == 0:
            for i in f1.dm:
                f3.dm[i] = f1.dm[i]
        else:
            for i in f2.dm:
                f3.dm[i] = SimArray(
                    np.append(f1.dm[i], f2.dm[i], axis=0), f2.dm[i].units
                )

    if len(f3.g) > 0:
        if len(f1.g) == 0:
            for i in f2.g:
                f3.g[i] = f2.g[i]
        elif len(f2.g) == 0:
            for i in f1.g:
                f3.g[i] = f1.g[i]
        else:
            for i in f2.g:
                f3.g[i] = SimArray(np.append(f1.g[i], f2.g[i], axis=0), f2.g[i].units)

    if len(f3.bh) > 0:
        if len(f1.bh) == 0:
            for i in f2.bh:
                f3.bh[i] = f2.bh[i]
        elif len(f2.bh) == 0:
            for i in f1.bh:
                f3.bh[i] = f1.bh[i]
        else:
            for i in f2.bh:
                f3.bh[i] = SimArray(
                    np.append(f1.bh[i], f2.bh[i], axis=0), f2.bh[i].units
                )

    return f3


def get_parttype(particle_field):
    particle_typeload = ''

    if ('dm' in particle_field) or ('darkmatter' in particle_field):
        if len(particle_typeload) > 0:
            particle_typeload += ',dm'
        else:
            particle_typeload += 'dm'

    if (
        ('star' in particle_field)
        or ('stars' in particle_field)
        or ('stellar' in particle_field)
    ):
        if len(particle_typeload) > 0:
            particle_typeload += ',star'
        else:
            particle_typeload += 'star'

    if (
        ('gas' in particle_field)
        or ('g' in particle_field)
        or ('cells' in particle_field)
    ):
        if len(particle_typeload) > 0:
            particle_typeload += ',gas'
        else:
            particle_typeload += 'gas'

    if (
        ('bh' in particle_field)
        or ('bhs' in particle_field)
        or ('blackhole' in particle_field)
        or ('blackholes' in particle_field)
    ):
        if len(particle_typeload) > 0:
            particle_typeload += ',bh'
        else:
            particle_typeload += 'bh'
    return particle_typeload


def get_Snapshot_property(BasePath: str, Snap: int) -> simdict.SimDict:
    """
    Retrieves properties of a specific snapshot.

    Parameters:
    -----------
    BasePath : str
        The base path to the directory containing snapshot files.
    Snap : int
        The identifier of the snapshot to be loaded.

    Returns:
    --------
    Snapshot : simdict.SimDict
        A SimDict object containing the properties of the specified snapshot.
    """
    SnapshotHeader = loadHeader(BasePath, Snap)
    Snapshot = simdict.SimDict()
    Snapshot['filepath'] = BasePath
    Snapshot['read_Snap_properties'] = SnapshotHeader
    Snapshot['Snapshot'] = Snap
    return Snapshot


def get_eps_Mdm(Snapshot):
    """
    Retrieves the gravitational softenings for stars and dark matter (DM) based on the simulation run and redshift.

    Parameters:
    -----------
    Snapshot : object
        An object containing snapshot properties, including `z` (redshift) and `run` (simulation run).

    Returns:
    --------
    eps_star : SimArray
        The gravitational softening length for stars.
    eps_dm : SimArray
        The gravitational softening length for dark matter.
    ------
    'Gravitational softenings for stars and DM are in comoving kpc until z=1,
    after which they are fixed to their z=1 values.' -- Dylan Nelson.
    Data is sourced from https://www.tng-project.org/data/docs/background/.
    """
    MatchRun = {
        'TNG50-1': [0.39, 3.1e5 / 1e10],
        'TNG50-2': [0.78, 2.5e6 / 1e10],
        'TNG50-3': [1.56, 2e7 / 1e10],
        'TNG50-4': [3.12, 1.6e8 / 1e10],
        'TNG100-1': [1., 5.1e6 / 1e10],
        'TNG100-2': [2., 4e7 / 1e10],
        'TNG100-3': [4., 3.2e8 / 1e10],
        'TNG300-1': [2., 4e7 / 1e10],
        'TNG300-1': [4., 3.2e8 / 1e10],
        'TNG300-1': [8., 2.5e9 / 1e10],
    }

    if Snapshot.properties['z'] > 1:
        return SimArray(
            MatchRun[Snapshot.properties['run']][0], units.a * units.kpc / units.h
        ), SimArray(
            MatchRun[Snapshot.properties['run']][1], 1e10 * units.Msol / units.h
        )
    else:
        return SimArray(
            MatchRun[Snapshot.properties['run']][0] / 2., units.kpc / units.h
        ), SimArray(
            MatchRun[Snapshot.properties['run']][1], 1e10 * units.Msol / units.h
        )

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
