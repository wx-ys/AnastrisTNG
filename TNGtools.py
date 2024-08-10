import numpy as np
from AnastrisTNG.illustris_python.snapshot import *
import h5py


'''
def inverseMapPartIndicesToSubhaloIDs(sP, indsType, ptName, debug=False, flagFuzz=True,
                                     ):
   #  SubhaloLenType, SnapOffsetsSubhalo
    """ For a particle type ptName and snapshot indices for that type indsType, compute the
        subhalo ID to which each particle index belongs. 
        If flagFuzz is True (default), particles in FoF fuzz are marked as outside any subhalo,
        otherwise they are attributed to the closest (prior) subhalo.
    """
    gcLenType = SubhaloLenType[:,sP.ptNum(ptName)]
    gcOffsetsType = SnapOffsetsSubhalo[:,sP.ptNum(ptName)][:-1]

    # val gives the indices of gcOffsetsType such that, if each indsType was inserted
    # into gcOffsetsType just -before- its index, the order of gcOffsetsType is unchanged
    # note 1: (gcOffsetsType-1) so that the case of the particle index equaling the
    # subhalo offset (i.e. first particle) works correctly
    # note 2: np.ss()-1 to shift to the previous subhalo, since we want to know the
    # subhalo offset index -after- which the particle should be inserted
    val = np.searchsorted( gcOffsetsType - 1, indsType ) - 1
    val = val.astype('int32')

    # search and flag all matches where the indices exceed the length of the
    # subhalo they have been assigned to, e.g. either in fof fuzz, in subhalos with
    # no particles of this type, or not in any subhalo at the end of the file
    if flagFuzz:
        gcOffsetsMax = gcOffsetsType + gcLenType - 1
        ww = np.where( indsType > gcOffsetsMax[val] )[0]

        if len(ww):
            val[ww] = -1

    if debug:
        # for all inds we identified in subhalos, verify parents directly
        for i in range(len(indsType)):
            if val[i] < 0:
                continue
            assert indsType[i] >= gcOffsetsType[val[i]]
            if flagFuzz:
                assert indsType[i] < gcOffsetsType[val[i]]+gcLenType[val[i]]
                assert gcLenType[val[i]] != 0

    return val
'''





def findtracer(basePath,snapNum,findID=None,istracerid=False,):
    
    """ 
        Find the tracers of the specified IDs (ParentIDs or TracerIDs)
        Note: it will work for all 100 snapshots for TNG300 and TNG50, but only the (20) full snapshots for TNG100.

        Input:
            findID: 1d array of the specified IDs
            istracerid: Ture for matching TracerIDs;
                        False for matching ParentIDs;

        Output: 
            result: dict, keys: 'ParentID','TracerID'.
            Note: As a parent has no, or multiple, tracers:
                if matching ParentIDs, the number of tracers found is likely to be different from the len(findID).
                if matching TracerIDs, the number of tracers found must be the same as the len(findID).

        Example1:
            findID=np.array([ID1,ID2,ID3...,IDi]) # IDi could be a gas cell, star, wind phase cell, or BH IDs
            Tracer=findtracer(basePath,snapNum,findID=findID,istracerid=False,)

        Example2:
            findID=np.array([ID1,ID2,ID3...,IDi]) # IDi should be tracer IDs 
            Tracer=findtracer(basePath,snapNum,findID=findID,istracerid=True,)

        Example3: find the progenitor gas ParticleIDs of star ParticleIDs
            findID=np.array([ID1,ID2,ID3...,IDi]) # IDi is the current star ParticleIDs(ParentID)
            Tracernow=findtracer(basePath,snapNumNow,findID=findID,istracerid=False,) # link the current ParticleIDs(ParentID) to TracerID
            Trecerbefore=findtracer(basePath,snapNumbefore,findID=Tracernow['TracerID'],istracerid=True,) # link the TracerID to progenitor ParticleIDs(ParentID)
            #Trecerbefore['ParentID'] is the progenitor ParticleIDs # could be gas, or star..        
    """
    result = {}
    result['ParentID']=np.array([])
    result['TracerID']=np.array([])
    
    # PartType3, tracer
    ptNum = 3
    gName = "PartType" + str(ptNum)
    
    # Apart from ParentID and TracerID, there is also FluidQuantities in TNG100
    fields=['ParentID','TracerID']

    findIDset=set(findID)

    # load header from first chunk
    with h5py.File(snapPath(basePath, snapNum), 'r') as f:
        header = dict(f['Header'].attrs.items())
        nPart = getNumPart(header)


        fileNum = 0
        fileOff = 0
        numToRead = nPart[ptNum]
        
        if not numToRead:

            return result
 
        i = 1
        while gName not in f:
            f = h5py.File(snapPath(basePath, snapNum, i), 'r')
            i += 1

        if not fields:
            fields = list(f[gName].keys())

    wOffset = 0
    origNumToRead = numToRead
    


    while numToRead:
        f = h5py.File(snapPath(basePath, snapNum, fileNum), 'r')

        if gName not in f:
            f.close()
            fileNum += 1
            fileOff  = 0
            continue

        numTypeLocal = f['Header'].attrs['NumPart_ThisFile'][ptNum]
        numToReadLocal = numToRead

        if fileOff + numToReadLocal > numTypeLocal:
            numToReadLocal = numTypeLocal - fileOff

        if istracerid:
            findresult=findIDset.isdisjoint(f['PartType3']['TracerID'][:])  # time complexity O( min(len(set1),len(set2)) )
        else:
            findresult=findIDset.isdisjoint(f['PartType3']['ParentID'][:])

        if findresult==False:
            ParentID=np.array(f['PartType3']['ParentID'])
            TracerID=np.array(f['PartType3']['TracerID'])
            if istracerid:
                Findepatticle=np.isin(TracerID,findID)                     # time complexity O( len(array1)*len(array2) ) 
            else:
                Findepatticle=np.isin(ParentID,findID)
            result['TracerID']=np.append(result['TracerID'],TracerID[Findepatticle])
            result['ParentID']=np.append(result['ParentID'],ParentID[Findepatticle])
            print('Number of tracers that have been matched: ',len(result['TracerID']))
        wOffset   += numToReadLocal
        numToRead -= numToReadLocal
        fileNum   += 1
        fileOff    = 0  

        f.close()
        
        # if matching TracerIDs, the number of tracers found must be the same as the len(findID).
        if istracerid and len(result['TracerID'])==len(findID):   
            break
    result['TracerID']=result['TracerID'].astype(int)
    result['ParentID']=result['ParentID'].astype(int)
    return result