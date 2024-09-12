'''
Evolution of galaxies: galaxy_evolution()
History of galaxy mergers: merger_history()
'''

from typing import List

from pynbody.array import SimArray
from pynbody import units

import numpy as np

from AnastrisTNG.illustris_python.sublink import loadTree, maxPastMass
from AnastrisTNG.illustris_python.groupcat import loadHeader
from AnastrisTNG.TNGunits import groupcat_units
from AnastrisTNG.TNGsnapshot import get_t



# Refer to illustris_python.sublink.numMergers, This is a modified version.
def merger_history(BasePath: str, 
                   snap: int = 99, 
                   subID: int = 10,
                   fields: List[str] = ['SubfindID','SubhaloMassType','SnapNum'],
                   minMassRatio: float = 1e-10, 
                   massPartType: str = 'stars',
                   physical_units: bool = True,
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
    index=0
    MergerHistory={}
    MergerHistory['MergerEvents']=np.zeros(100)
    MergerHistory['MassRatio']=[]
    reqFields = ['SubhaloID', 'NextProgenitorID', 'MainLeafProgenitorID',
                 'FirstProgenitorID', 'SubhaloMassType','SnapNum','SubfindID']
    allfields=list(set(reqFields+fields))
    tree = loadTree(basePath=BasePath,snapNum=snap,id=subID,fields=allfields,onlyMPB=False)
    fields=list(set(fields+['SnapNum']))
    for i in fields:
        MergerHistory['First-'+i]=[]
        MergerHistory['Next-'+i]=[]

    """ Calculate the number of mergers in this sub-tree (optionally above some mass ratio threshold). """
    # verify the input sub-tree has the required fields

    if not set(reqFields).issubset(tree.keys()):
        raise Exception('Error: Input tree needs to have loaded fields: '+', '.join(reqFields))

    numMergers   = 0
    invMassRatio = 1.0 / minMassRatio

    # walk back main progenitor branch
    rootID = tree['SubhaloID'][index]
    fpID   = tree['FirstProgenitorID'][index]

    while fpID != -1:
        fpIndex = index + (fpID - rootID)
        fpMass  = maxPastMass(tree, fpIndex, massPartType)

        # explore breadth
        npID = tree['NextProgenitorID'][fpIndex]

        while npID != -1:
            npIndex = index + (npID - rootID)
            npMass  = maxPastMass(tree, npIndex, massPartType)

            # count if both masses are non-zero, and ratio exceeds threshold
            if fpMass > 0.0 and npMass > 0.0:
                ratio = npMass / fpMass

                if ratio >= minMassRatio and ratio <= invMassRatio:
                    numMergers += 1
                    for key in fields:
                        try:
                            A1_snap=tree['SnapNum'][fpIndex]
                            A2_snap=tree['SnapNum'][npIndex]
                            A1=SimArray(tree[key][fpIndex],units=groupcat_units(key))
                            A2=SimArray(tree[key][npIndex],units=groupcat_units(key))
                            if physical_units:
                                A1=_physical_unit_mer(A1,BasePath,A1_snap)
                                A2=_physical_unit_mer(A2,BasePath,A2_snap)
                            MergerHistory['First-'+key].append(A1)
                            MergerHistory['Next-'+key].append(A2)
                        except: 
                            MergerHistory['First-'+key].append(tree[key][fpIndex])
                            MergerHistory['Next-'+key].append(tree[key][npIndex])
                    MergerHistory['MassRatio'].append(ratio)

                    MergerHistory['MergerEvents'][tree['SnapNum'][fpIndex]]+=1
                    #Snap.append([tree['SnapNum'][fpIndex],tree['SnapNum'][npIndex]])
                  #  MergerHistory['MergerEvents']=np.zeros(100)
            npID = tree['NextProgenitorID'][npIndex]
        fpID = tree['FirstProgenitorID'][fpIndex]
    MergerHistory['numMergers']=numMergers
    if numMergers>0:
        
        F_scalefactor=[loadHeader(BasePath,i)['Time'] for i in MergerHistory['First-SnapNum']]
        N_scalefactor=[loadHeader(BasePath,i)['Time'] for i in MergerHistory['Next-SnapNum']]
        omega_m = loadHeader(BasePath,99)['Omega0']
        H0_kmsMpc = 100.0 * loadHeader(BasePath,99)['HubbleParam']*units.km/units.s/units.Mpc
        F_time=[get_t(omega_m,(1/i-1),H0_kmsMpc) for i in F_scalefactor]
        N_time=[get_t(omega_m,(1/i-1),H0_kmsMpc) for i in N_scalefactor]
        MergerHistory['First-a']=np.array(F_scalefactor)
        MergerHistory['Next-a']=np.array(N_scalefactor)
        MergerHistory['First-t']=np.array(F_time)
        MergerHistory['Next-t']=np.array(N_time)
    return MergerHistory

def galaxy_evolution(basePath: str,
                     snap: int,
                     subID: int,
                     fields: List[str] = ['SnapNum','SubfindID'],
                     physical_units: bool = True,
                     ) -> dict:
    """
    The evolution of the galaxy.
    """
    basefields=['SnapNum','SubfindID']
    allfields=list(set(basefields+fields))
    tree = loadTree(basePath,snap,subID,allfields,onlyMPB=True)
    for i in tree:
        try:
            tree[i]=SimArray(tree[i],units=groupcat_units(i))
        except:
            continue
    if tree['count']>0 and physical_units:
        tree=_physical_unit_evo(tree,basePath)
    if tree['count']>0:
        scalefactor=[loadHeader(basePath,i)['Time'] for i in tree['SnapNum']]
        tree['a']=np.array(scalefactor)
        omega_m = loadHeader(basePath,99)['Omega0']
        H0_kmsMpc = 100.0 * loadHeader(basePath,99)['HubbleParam']*units.km/units.s/units.Mpc
        
        time=[get_t(omega_m,(1/i-1),H0_kmsMpc).in_units('Gyr') for i in scalefactor]
        tree['t']=SimArray(np.array(time),units.Gyr)
    return tree

def _physical_unit_mer(array: SimArray, basepath: str, snap: int) -> SimArray:
    if isinstance(array,SimArray):
        if 'h' in str(array.units):
            units_strlist=str(array.units).split(' ')
            for i in units_strlist:
                if 'h' in i:
                    h_strlist=i.split('**')
                    if len(h_strlist)==1:
                        h_pow=1
                    else:
                        if '/' in h_strlist[1]:
                            h_pow=float(h_strlist[1].split('/')[0])/float(h_strlist[1].split('/')[1])
                        else:
                            h_pow=float(h_strlist[1])
                    h_this=loadHeader(basepath,snap)['HubbleParam']
                    array=array*h_this**(h_pow)
                    units_strlist.remove(i)
                    array.units=units.Unit(" ".join(units_strlist))
                    break
        if 'a' in str(array.units) and 'mag' not in str(array.units):
            units_strlist=str(array.units).split(' ')
            for i in units_strlist:
                if 'a' in i:
                    a_strlist=i.split('**')
                    if len(a_strlist)==1:
                        a_pow=1
                    else:
                        if '/' in a_strlist[1]:
                            a_pow=float(a_strlist[1].split('/')[0])/float(a_strlist[1].split('/')[1])
                        else:
                            a_pow=float(a_strlist[1])
                    a_this=loadHeader(basepath,snap)['Time']
                    array=array*a_this**(a_pow)
                    units_strlist.remove(i)
                    array.units=units.Unit(" ".join(units_strlist))
                    break
    return array

def _physical_unit_evo(tree: dict, basepath: str) -> dict:
    for para in tree:
        if isinstance(tree[para],SimArray):
            if 'h' in str(tree[para].units):
                units_strlist=str(tree[para].units).split(' ')
                for i in units_strlist:
                    if 'h' in i:
                        h_strlist=i.split('**')
                        if len(h_strlist)==1:
                            h_pow=1
                        else:
                            if '/' in h_strlist[1]:
                                h_pow=float(h_strlist[1].split('/')[0])/float(h_strlist[1].split('/')[1])
                            else:
                                h_pow=float(h_strlist[1])
                        h_this=loadHeader(basepath,99)['HubbleParam']
                        tree[para]=tree[para]*h_this**(h_pow)
                        units_strlist.remove(i)
                        tree[para].units=units.Unit(" ".join(units_strlist))
                        break
            if 'a' in str(tree[para].units) and 'mag' not in str(tree[para].units):
                units_strlist=str(tree[para].units).split(' ')
                for i in units_strlist:
                    if 'a' in i:
                        a_strlist=i.split('**')
                        if len(a_strlist)==1:
                            a_pow=1
                        else:
                            if '/' in a_strlist[1]:
                                a_pow=float(a_strlist[1].split('/')[0])/float(a_strlist[1].split('/')[1])
                            else:
                                a_pow=float(a_strlist[1])
                        for Snum in range(tree['count']):
                            a_this=loadHeader(basepath,tree['SnapNum'][Snum])['Time']
                            tree[para][Snum]=tree[para][Snum]*a_this**(a_pow)
                        units_strlist.remove(i)
                        tree[para].units=units.Unit(" ".join(units_strlist))
                        break
    return tree