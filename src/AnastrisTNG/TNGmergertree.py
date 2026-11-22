'''
Evolution of galaxies: galaxy_evolution()
History of galaxy mergers: merger_history()
'''

from typing import List
from functools import lru_cache
from fractions import Fraction

from pynbody.array import SimArray
from pynbody import units

import numpy as np

from AnastrisTNG.illustris_python.sublink import loadTree, maxPastMass
from AnastrisTNG.illustris_python.groupcat import loadHeader, loadSingle
from AnastrisTNG.TNGunits import groupcat_units
from AnastrisTNG.TNGsnapshot import get_t


# Refer to illustris_python.sublink.numMergers, This is a modified version.
def merger_history(
    BasePath: str,
    snap: int = 99,
    subID: int = 10,
    fields: List[str] = ['SubfindID', 'SubhaloMassType', 'SnapNum'],
    minMassRatio: float = 1e-10,
    massPartType: str = 'stars',
    physical_units: bool = True,
    *, _source=None,
) -> dict:
    """
    This function queries the merger history of a subhalo (galaxy).

    Parameters:
    -----------
    BasePath : str
        The directory where the simulation data is stored.

    snap : int, optional
        The snapshot number.
        Default is 99.

    subID : int,
        The ID of the subhalo being queried.

    Needfields : list, optional
        The list of subhalo parameters to be queried. Default includes:
        ['SubfindID', 'SubhaloMassType', 'SnapNum'].

    minMassRatio : float, optional
        The minimum mass ratio for considering a merger event. Default is 1e-10.

    massPartType : str, optional
        The type of particle mass to base the mass ratio on. It can be 'stars' or
        'darkmatter'. Default is 'stars'.

    Returns:
    --------
    dict
        A dictionary containing the queried subhalo's merger history based on the
        specified parameters.
    """
    index = 0
    MergerHistory = {}
    original_illustris = _source.original_illustris if _source is not None else "Illustris-" in str(BasePath)
    MergerHistory['MergerEvents'] = np.zeros(136 if original_illustris or snap > 99 else 100)
    MergerHistory['MassRatio'] = []
    reqFields = [
        'SubhaloID',
        'NextProgenitorID',
        'MainLeafProgenitorID',
        'FirstProgenitorID',
        'SubhaloMassType',
        'SnapNum',
        'SubfindID',
    ]
    fields = [fields] if isinstance(fields, str) else list(fields)
    allfields = list(dict.fromkeys(reqFields + fields))
    tree = (_source.merger_tree(subID, allfields) if _source is not None else
            loadTree(basePath=BasePath, snapNum=snap, id=subID, fields=allfields, onlyMPB=False))
    @lru_cache(maxsize=None)
    def header(number):
        return _source.history_header(number) if _source is not None else loadHeader(BasePath, number)
    fields = list(dict.fromkeys(fields + ['SnapNum']))
    for i in fields:
        MergerHistory['First-' + i] = []
        MergerHistory['Next-' + i] = []

    """ Calculate the number of mergers in this sub-tree (optionally above some mass ratio threshold). """
    # verify the input sub-tree has the required fields

    if not set(reqFields).issubset(tree.keys()):
        raise Exception(
            'Error: Input tree needs to have loaded fields: ' + ', '.join(reqFields)
        )

    numMergers = 0
    invMassRatio = 1.0 / minMassRatio

    # walk back main progenitor branch
    rootID = tree['SubhaloID'][index]
    fpID = tree['FirstProgenitorID'][index]

    while fpID != -1:
        fpIndex = index + (fpID - rootID)
        fpMass = maxPastMass(tree, fpIndex, massPartType)

        # explore breadth
        npID = tree['NextProgenitorID'][fpIndex]

        while npID != -1:
            npIndex = index + (npID - rootID)
            npMass = maxPastMass(tree, npIndex, massPartType)

            # count if both masses are non-zero, and ratio exceeds threshold
            if fpMass > 0.0 and npMass > 0.0:
                ratio = npMass / fpMass

                if ratio >= minMassRatio and ratio <= invMassRatio:
                    numMergers += 1
                    for key in fields:
                        A1 = tree[key][fpIndex]
                        A2 = tree[key][npIndex]
                        try:
                            unit = groupcat_units(key)
                        except KeyError:
                            pass  # SubLink IDs and other tree-only fields retain their native dtype.
                        else:
                            A1 = SimArray(A1, units=unit)
                            A2 = SimArray(A2, units=unit)
                            if physical_units:
                                A1 = _physical_unit_mer(A1, BasePath, tree['SnapNum'][fpIndex], header)
                                A2 = _physical_unit_mer(A2, BasePath, tree['SnapNum'][npIndex], header)
                        MergerHistory['First-' + key].append(A1)
                        MergerHistory['Next-' + key].append(A2)
                    MergerHistory['MassRatio'].append(ratio)

                    MergerHistory['MergerEvents'][tree['SnapNum'][fpIndex]] += 1
                    # Snap.append([tree['SnapNum'][fpIndex],tree['SnapNum'][npIndex]])
            npID = tree['NextProgenitorID'][npIndex]
        fpID = tree['FirstProgenitorID'][fpIndex]
    MergerHistory['numMergers'] = numMergers
    if numMergers > 0:

        F_scalefactor = [
            header(int(i))['Time'] for i in MergerHistory['First-SnapNum']
        ]
        N_scalefactor = [
            header(int(i))['Time'] for i in MergerHistory['Next-SnapNum']
        ]
        omega_m = header(snap)['Omega0']
        H0_kmsMpc = (
            100.0
            * header(snap)['HubbleParam']
            * units.km
            / units.s
            / units.Mpc
        )
        F_time = [get_t(omega_m, (1 / i - 1), H0_kmsMpc) for i in F_scalefactor]
        N_time = [get_t(omega_m, (1 / i - 1), H0_kmsMpc) for i in N_scalefactor]
        MergerHistory['First-a'] = np.array(F_scalefactor)
        MergerHistory['Next-a'] = np.array(N_scalefactor)
        MergerHistory['First-t'] = np.array(F_time)
        MergerHistory['Next-t'] = np.array(N_time)
    return MergerHistory


def _finish_evolution(tree, base_path, snap, source, physical_units):
    """Attach catalog units and the cosmic time at each actual branch snapshot."""
    numbers = tree['SnapNum']
    tree['count'] = len(numbers)
    if source is None:
        cosmology = loadHeader(base_path, snap)
        scale = np.array([loadHeader(base_path, int(i))['Time'] for i in numbers])
    else:
        cosmology = source.header()
        scale = source.history_scales(numbers, cosmology)
    hubble = np.float64(cosmology['HubbleParam'])
    for name in tree:
        try:
            unit = groupcat_units(name)
        except KeyError:
            continue
        values = np.asarray(tree[name])
        if physical_units:
            for token in str(unit).split():
                symbol, _, exponent = token.partition("**")
                if symbol not in ("a", "h"):
                    continue
                power = Fraction(exponent or "1")
                factor = hubble if symbol == "h" else scale.reshape((-1,) + (1,) * (values.ndim - 1))
                values = values * factor ** float(power)
                unit = unit / units.Unit(symbol) ** power
        tree[name] = SimArray(values, unit)
    tree['a'] = scale
    h0 = 100 * hubble * units.km / units.s / units.Mpc
    tree['t'] = SimArray([get_t(cosmology['Omega0'], 1/a - 1, h0).in_units('Gyr') for a in scale], units.Gyr)
    return tree


def galaxy_evolution(basePath: str, snap: int, subID: int,
                     fields=('SnapNum', 'SubfindID'), physical_units: bool = True, *, _source=None) -> dict:
    """Return the SubLink main progenitor branch, starting at the selected subhalo."""
    fields = [fields] if isinstance(fields, str) else list(fields)
    fields = list(dict.fromkeys(['SnapNum', 'SubfindID'] + fields))
    tree = (loadTree(basePath, snap, subID, fields, onlyMPB=True) if _source is None else
            _source.merger_tree(subID, fields, only_mpb=True))
    if tree is None:
        raise ValueError(f"Subhalo {subID} has no SubLink tree at snapshot {snap}")
    return _finish_evolution(tree, basePath, snap, _source, physical_units)


def halo_evolution(basePath: str, snap: int, haloID: int, physical_units: bool = True,
                   *, fields=None, _source=None) -> dict:
    """Follow the FoF hosts of the central subhalo's main progenitors.

    With fields omitted, return all columns of the starting FoF catalog row.
    """
    if fields is not None:
        fields = [fields] if isinstance(fields, str) else list(dict.fromkeys(fields))
    if _source is None:
        root = loadSingle(basePath, snap, haloID=haloID)
    elif fields is None:
        root = _source.catalog("Halo", haloID)
    else:
        root = _source.halo_values(haloID, ['GroupFirstSub'], fetch=True)
    central = int(root['GroupFirstSub'])
    if central in (-1, 2**32 - 1, 2**64 - 1):
        raise ValueError(f"FoF halo {haloID} has no central subhalo at snapshot {snap}")
    if fields is None:
        fields = list(root)
    links = ['SnapNum', 'SubhaloGrNr']
    if _source is None:
        branch = loadTree(basePath, snap, central, links, onlyMPB=True)
        if branch is None:
            raise ValueError(f"Central subhalo {central} has no SubLink tree at snapshot {snap}")
    else:
        branch = _source.merger_tree(central, links, only_mpb=True)
    snapshots, hosts = branch['SnapNum'], branch['SubhaloGrNr']
    columns = {}
    if _source is None:
        rows = [root if i == snap and host == haloID else loadSingle(basePath, int(i), haloID=int(host))
                for i, host in zip(snapshots, hosts)]
    else:
        sources = [_source.at_snapshot(i) for i in snapshots]
        rows = [src.halo_values(int(host), fields) for src, host in zip(sources, hosts)]
        missing = [k for k in fields if k.startswith('Group') and any(k not in row for row in rows)]
        # SubLink duplicates FoF catalog columns at each node. One branch slice
        # supplies that column across all epochs. Run-specific extras can live
        # solely in the group catalog.
        columns = _source.merger_tree(central, missing, only_mpb=True, optional=True) if missing else {}
        for index, (src, host, row) in enumerate(zip(sources, hosts, rows)):
            for k in missing:
                if k not in row and k in columns:
                    row[k] = columns[k][index]
            remaining = [k for k in fields if k not in row]
            if remaining:
                row.update(src.halo_values(int(host), remaining, fetch=True))
    tree = {k: np.asarray([row[k] for row in rows], dtype=columns[k].dtype if k in columns else None)
            for k in fields}
    tree.update(HaloID=SimArray(hosts, groupcat_units('SubhaloGrNr')), SnapNum=snapshots)
    return _finish_evolution(tree, basePath, snap, _source, physical_units)


def _physical_unit_mer(array: SimArray, basepath: str, snap: int, header=None) -> SimArray:
    if header is None:
        header = lambda number: loadHeader(basepath, number)
    if isinstance(array, SimArray):
        if 'h' in str(array.units):
            units_strlist = str(array.units).split(' ')
            for i in units_strlist:
                if 'h' in i:
                    h_strlist = i.split('**')
                    if len(h_strlist) == 1:
                        h_pow = 1
                    else:
                        if '/' in h_strlist[1]:
                            h_pow = float(h_strlist[1].split('/')[0]) / float(
                                h_strlist[1].split('/')[1]
                            )
                        else:
                            h_pow = float(h_strlist[1])
                    h_this = np.float64(header(snap)['HubbleParam'])
                    array = array * h_this ** (h_pow)
                    units_strlist.remove(i)
                    array.units = units.Unit(" ".join(units_strlist))
                    break
        if 'a' in str(array.units) and 'mag' not in str(array.units):
            units_strlist = str(array.units).split(' ')
            for i in units_strlist:
                if 'a' in i:
                    a_strlist = i.split('**')
                    if len(a_strlist) == 1:
                        a_pow = 1
                    else:
                        if '/' in a_strlist[1]:
                            a_pow = float(a_strlist[1].split('/')[0]) / float(
                                a_strlist[1].split('/')[1]
                            )
                        else:
                            a_pow = float(a_strlist[1])
                    a_this = np.float64(header(snap)['Time'])
                    array = array * a_this ** (a_pow)
                    units_strlist.remove(i)
                    array.units = units.Unit(" ".join(units_strlist))
                    break
    return array
