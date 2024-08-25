import numpy as np
from AnastrisTNG.illustris_python.snapshot import *
import h5py
from tqdm import tqdm
import multiprocessing as mp
'''
# form https://www.tng-project.org/data/forum/topic/274/match-snapshot-particles-with-their-halosubhalo/
# Careful memory usage
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
def process_file(file_info):
    """
    Process a single file to find tracers of specified IDs (ParentIDs or TracerIDs).
    This function is used by the `findtracer_MP` function to distribute tasks among multiple processes.

    Parameters:
    ----------
    file_info : tuple
        A tuple containing the following elements:
            - basePath : str
                The base directory path where simulation data is stored.
            - snapNum : int
                Snapshot number to search within.
            - fileNum : int
                The file number within the snapshot to process.
            - findIDset : set
                Set of specified IDs (ParentIDs or TracerIDs) to find.
            - istracerid : bool
                If True, match TracerIDs; if False, match ParentIDs.

    Returns:
    -------
    dict
        A dictionary with keys:
            - 'ParentID': List of matched ParentIDs.
            - 'TracerID': List of matched TracerIDs.
        Note:
            - When `istracerid` is True, the dictionary contains tracers that match the IDs in `findIDset`.
            - When `istracerid` is False, the dictionary contains parents that match the IDs in `findIDset`.

    Notes:
    -----
    - This function is designed to be used with multiprocessing to improve performance when searching through large datasets.
    - It reads a specific file within a snapshot and checks for the presence of IDs in the dataset.

    Example:
    --------
    To use with `findtracer_MP`:
        file_info = (basePath, snapNum, fileNum, findIDset, istracerid)
        result_local = process_file(file_info)
    """
    basePath, snapNum, fileNum, findIDset, istracerid = file_info
    result_local = {'ParentID': [], 'TracerID': []}
    
    gName = "PartType3"
    fields = ['ParentID', 'TracerID']
    
    with h5py.File(snapPath(basePath, snapNum, fileNum), 'r') as f:
        if gName not in f:
            return result_local
        
        if istracerid:
            findresult=findIDset.isdisjoint(f['PartType3']['TracerID'][:])  # time complexity O( min(len(set1),len(set2)) )
        else:
            findresult=findIDset.isdisjoint(f['PartType3']['ParentID'][:])
        
        if findresult==False:
            ParentID = np.array(f[gName]['ParentID'])
            TracerID = np.array(f[gName]['TracerID'])
            
            if istracerid:
                Findepatticle = np.isin(TracerID, findIDset)
            else:
                Findepatticle = np.isin(ParentID, findIDset)
            
            result_local['TracerID'] = TracerID[Findepatticle]
            result_local['ParentID'] = ParentID[Findepatticle]
    
    return result_local


def findtracer_MP(basePath, snapNum, findID=None, istracerid=False,NP=6):
    """
    Find the tracers of specified IDs (ParentIDs or TracerIDs) using multiprocessing to speed up the search.

    Note:
        This function works for all snapshots in TNG300 and TNG50, but only the 20 full snapshots for TNG100.
        Using multiprocessing with the parameter NP can improve performance, but be mindful of available memory.

    Parameters:
    ----------
    basePath : str
        The base directory path where simulation data is stored.
    snapNum : int
        Snapshot number to search within.
    findID : list 
        1D array of the specified IDs (ParentIDs or TracerIDs) to find. Default is None.
    istracerid : bool, optional
        If True, match TracerIDs; if False, match ParentIDs. Default is False.
    NP : int, optional
        Number of multiprocessing processes to use. Default is 6. More processes can speed up the search but require more memory.

    Returns:
    -------
    dict
        A dictionary with keys:
            - 'ParentID': Array of matched ParentIDs.
            - 'TracerID': Array of matched TracerIDs.
        Note:
            - When matching ParentIDs, the number of tracers found may differ from the length of `findID` since a parent can have no or multiple tracers.
            - When matching TracerIDs, the number of tracers found must match the length of `findID`.

    Examples:
    --------
    Example 1:
        findID = np.array([ID1, ID2, ID3, ..., IDi])  # IDi can be gas cell, star, wind phase cell, or BH IDs
        Tracer = findtracer_MP(basePath, snapNum, findID=findID, istracerid=False)

    Example 2:
        findID = np.array([ID1, ID2, ID3, ..., IDi])  # IDi should be tracer IDs
        Tracer = findtracer_MP(basePath, snapNum, findID=findID, istracerid=True)

    Example 3: Find the progenitor gas ParticleIDs of star ParticleIDs
        findID = np.array([ID1, ID2, ID3, ..., IDi])  # IDi are the current star ParticleIDs (ParentID)
        Tracernow = findtracer_MP(basePath, snapNumNow, findID=findID, istracerid=False)  # Link current ParticleIDs (ParentID) to TracerID
        Trecerbefore = findtracer_MP(basePath, snapNumbefore, findID=Tracernow['TracerID'], istracerid=True)  # Link TracerID to progenitor ParticleIDs (ParentID)
        # Trecerbefore['ParentID'] contains the progenitor ParticleIDs (could be gas or star)
    """
    
    result = {'ParentID': np.array([]), 'TracerID': np.array([])}
    findIDset = set(findID)
    
    # Load header to determine number of particles
    with h5py.File(snapPath(basePath, snapNum), 'r') as f:
        header = dict(f['Header'].attrs.items())
        nPart = getNumPart(header)
        numToRead = nPart[3]  #trecer num
        
        if not numToRead:
            return result
        
        # file num
        file_numbers = []
        i = 1
        while True:
            try:
                with h5py.File(snapPath(basePath, snapNum, i), 'r') as f:
                    if "PartType3" in f:
                        file_numbers.append(i)
                        i += 1
                    else:
                        break
            except FileNotFoundError:
                break
    # mutiprocesses
    with mp.Pool(processes=NP) as pool:
        # date
        file_infos = [(basePath, snapNum, fileNum, findIDset, istracerid) for fileNum in file_numbers]
        
        # progressing bar
        with tqdm(total=len(file_infos)) as pbar:
            # Use imap to process files and update the progress bar
            for result_local in pool.imap(process_file, file_infos):
                result['TracerID'] = np.append(result['TracerID'], result_local['TracerID'])
                result['ParentID'] = np.append(result['ParentID'], result_local['ParentID'])
                pbar.update(1)
    
    # Convert to integer type
    result['TracerID'] = result['TracerID'].astype(int)
    result['ParentID'] = result['ParentID'].astype(int)
    
    return result



def findtracer(basePath : str,snapNum : int, findID : list =None,
               istracerid : bool = False,) -> dict:
    """ 
    Find the tracers of specified IDs (ParentIDs or TracerIDs) in the simulation data.

    Note:
        This function works for all 100 snapshots for TNG300 and TNG50, but only the 20 full snapshots for TNG100.

    Parameters:
    ----------
    basePath : str
        The base directory path where simulation data is stored.
    snapNum : int
        Snapshot number to search within.
    findID : list or array
        1D array of the specified IDs (ParentIDs or TracerIDs) to find. Default is None.
    istracerid : bool, optional
        If True, match TracerIDs; if False, match ParentIDs. Default is False.

    Returns:
    -------
    dict
        A dictionary with keys:
            - 'ParentID': Array of matched ParentIDs.
            - 'TracerID': Array of matched TracerIDs.
        Note: 
            - When matching ParentIDs, the number of tracers found may differ from the length of `findID` since a parent can have no or multiple tracers.
            - When matching TracerIDs, the number of tracers found must match the length of `findID`.

    Examples:
    --------
    Example 1:
        findID = np.array([ID1, ID2, ID3, ..., IDi])  # IDi can be gas cell, star, wind phase cell, or BH IDs
        Tracer = findtracer(basePath, snapNum, findID=findID, istracerid=False)

    Example 2:
        findID = np.array([ID1, ID2, ID3, ..., IDi])  # IDi should be tracer IDs
        Tracer = findtracer(basePath, snapNum, findID=findID, istracerid=True)

    Example 3: Find the progenitor gas ParticleIDs of star ParticleIDs
        findID = np.array([ID1, ID2, ID3, ..., IDi])  # IDi are the current star ParticleIDs (ParentID)
        Tracernow = findtracer(basePath, snapNumNow, findID=findID, istracerid=False)  # Link current ParticleIDs (ParentID) to TracerID
        Trecerbefore = findtracer(basePath, snapNumbefore, findID=Tracernow['TracerID'], istracerid=True)  # Link TracerID to progenitor ParticleIDs (ParentID)
        # Trecerbefore['ParentID'] contains the progenitor ParticleIDs (could be gas or star)
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
    


    # progress bar
    with tqdm(total=numToRead) as pbar:
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
            pbar.update(numToReadLocal)
            
            # if matching TracerIDs, the number of tracers found must be the same as the len(findID).
            if istracerid and len(result['TracerID'])==len(findID):   
                break
    result['TracerID']=result['TracerID'].astype(int)
    result['ParentID']=result['ParentID'].astype(int)
    return result