

from AnastrisTNG.illustris_python.groupcat import loadSingle

from pynbody.array import SimArray 
from AnastrisTNG.TNGunits import groupcat_units,halo_pa_name,subhalo_pa_name
from pynbody import simdict
from AnastrisTNG.TNGsnapshot import get_Snapshot_property

def get_Subhalo_property(BasePath,Snap,subhaloID):

    single=subhaloproperties(BasePath,Snap,subhaloID)
    Subhalo=simdict.SimDict()

    for i in single:
        Subhalo[subhalo_pa_name(i)]=single[i]
    Subhalo['ID']=subhaloID
    snapshot=get_Snapshot_property(BasePath,Snap)
    for i in snapshot:
        Subhalo[i]=snapshot[i]
    return Subhalo

def get_groupcatalogs_pa(Base, Snap):
    groupcatalog={}
    groupcatalog['halo']=_get_Halo_pa(Base,Snap)
    groupcatalog['subhalo']=_get_Subhalo_pa(Base,Snap)
    return groupcatalog    

def _get_Subhalo_pa(Base,Snap):
    
    return list(loadSingle(BasePath,Snap,subhaloID=1).keys())

def _get_Halo_pa(Base,Snap):
    
    return list(loadSingle(BasePath,Snap,HaloID=1).keys())

def get_Halo_property(BasePath,Snap,haloID):

    single=haloproperties(BasePath,Snap,haloID)
    Halo1=simdict.SimDict()

    for i in single:
        Halo1[halo_pa_name(i)]=single[i]
    Halo1['ID']=haloID
    snapshot=get_Snapshot_property(BasePath,Snap)
    for i in snapshot:
        Halo1[i]=snapshot[i]
    return Halo1


def subhaloproperties(BasePath,Snap,subhaloID):

    single=loadSingle(BasePath,Snap,subhaloID=subhaloID)
    for i in single:
        single[i]=SimArray(single[i],groupcat_units(i))

    return single


def haloproperties(BasePath,Snap,haloID):

    single=loadSingle(BasePath,Snap,haloID=haloID)
    for i in single:
        single[i]=SimArray(single[i],groupcat_units(i))

    return single








