

from AnastrisTNG.illustris_python.groupcat import loadSingle

from pynbody.array import SimArray 
from AnastrisTNG.TNGunits import GroupcatUnits,HaloPaName,SubhaloPaName
from pynbody import simdict
from AnastrisTNG.TNGsnapshot import get_Snapshot_property

def get_Subhalo_property(BasePath,Snap,subhaloID):

    single=subhaloproperties(BasePath,Snap,subhaloID)
    Subhalo=simdict.SimDict()

    for i in single.keys():
        Subhalo[SubhaloPaName(i)]=single[i]
    Subhalo['ID']=subhaloID
    snapshot=get_Snapshot_property(BasePath,Snap)
    for i in snapshot.keys():
        Subhalo[i]=snapshot[i]
    return Subhalo


def get_Halo_property(BasePath,Snap,haloID):

    single=haloproperties(BasePath,Snap,haloID)
    Halo1=simdict.SimDict()

    for i in single.keys():
        Halo1[HaloPaName(i)]=single[i]
    Halo1['ID']=haloID
    snapshot=get_Snapshot_property(BasePath,Snap)
    for i in snapshot.keys():
        Halo1[i]=snapshot[i]
    return Halo1


def subhaloproperties(BasePath,Snap,subhaloID):

    single=loadSingle(BasePath,Snap,subhaloID=subhaloID)
    for i in single.keys():
        single[i]=SimArray(single[i],GroupcatUnits(i))

    return single


def haloproperties(BasePath,Snap,haloID):

    single=loadSingle(BasePath,Snap,haloID=haloID)
    for i in single.keys():
        single[i]=SimArray(single[i],GroupcatUnits(i))

    return single








