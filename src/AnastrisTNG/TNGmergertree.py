from AnastrisTNG.illustris_python.sublink import loadTree, maxPastMass
from AnastrisTNG.TNGunits import groupcat_units
from pynbody.array import SimArray
import numpy as np
from typing import List

# Refer to illustris_python.sublink.numMergers, This is a modified version.
def merger_history(BasePath: str, 
                   snap: int = 99, 
                   subID: int = 10,
                   Needfields: List[str] = ['SubfindID','SubhaloMassType','SnapNum'],
                   minMassRatio: float = 1e-10, 
                   massPartType: str = 'stars'
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
    allfields=list(set(reqFields+Needfields))
    tree = loadTree(basePath=BasePath,snapNum=snap,id=subID,fields=allfields,onlyMPB=False)
    for i in Needfields:
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
                    for key in Needfields:
                        try:
                            MergerHistory['First-'+key].append(SimArray(tree[key][fpIndex],units=groupcat_units(key)))
                            MergerHistory['Next-'+key].append(SimArray(tree[key][npIndex]),units=groupcat_units(key))
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
    return MergerHistory

def galaxy_evolution(basePath: str,
                     snap: int,
                     subID: int,
                     fields: List[str],
                     ) -> dict:
    """
    The galaxy evolutionary properties.
    """
    
    
    tree = loadTree(basePath,snap,subID,fields,onlyMPB=True)
    for i in tree.keys():
        try:
            tree[i]=SimArray(tree[i],units=groupcat_units(i))
        except:
            continue
    return tree