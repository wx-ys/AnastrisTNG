'''
Some useful tools
find tracers: findtracer_MP(), findtracer(), Function.
star both pos: Star_birth(), Class.
potential: cal_potential, cal_acceleration, Function.
galaxy profile: Profile_1D(). Class.
...
'''

from typing import List
import multiprocessing as mp

import numpy as np
import h5py
from tqdm import tqdm
from pynbody import units
from pynbody.array import SimArray
from pynbody.analysis.profile import Profile as _Profile

from AnastrisTNG.illustris_python.snapshot import *
from AnastrisTNG.Anatools import Orbit
from AnastrisTNG.pytreegrav import PotentialTarget, AccelTarget


def cal_potential(sim, targetpos):
    """
    Calculates the gravitational potential at target positions.

    Parameters:
    -----------
    sim : object
        The simulation data object containing particle positions and masses.
    targetpos : array-like
        The positions where the gravitational potential needs to be calculated.

    Returns:
    --------
    phi : SimArray
        The gravitational potential at the target positions.
    """

    try:
        eps = sim.properties.get('eps', 0)
    except:
        eps = 0
    if eps == 0:
        print('Calculate the gravity without softening length')
    pot = PotentialTarget(
        targetpos,
        sim['pos'].view(np.ndarray),
        sim['mass'].view(np.ndarray),
        np.repeat(eps, len(targetpos)).view(np.ndarray),
    )
    phi = SimArray(pot, units.G * sim['mass'].units / sim['pos'].units)
    phi.sim = sim
    return phi


def cal_acceleration(sim, targetpos):
    """
    Calculates the gravitational acceleration at specified target positions.

    Parameters:
    -----------
    sim : object
        The simulation data object containing particle positions and masses.
    targetpos : array-like
        The positions where the gravitational acceleration needs to be calculated.

    Returns:
    --------
    acc : SimArray
        The gravitational acceleration at the target positions.
    """
    try:
        eps = sim.properties.get('eps', 0)
    except:
        eps = 0
    if eps == 0:
        print('Calculate the gravity without softening length')
    accelr = AccelTarget(
        targetpos,
        sim['pos'].view(np.ndarray),
        sim['mass'].view(np.ndarray),
        np.repeat(eps, len(targetpos)).view(np.ndarray),
    )
    acc = SimArray(
        accelr, units.G * sim['mass'].units / sim['pos'].units / sim['pos'].units
    )
    acc.sim = sim
    return acc


class Profile_1D:
    def __init__(
        self, sim, ndim=2, type='lin', nbins=100, rmin=0.1, rmax=100.0, **kwargs
    ):
        """
        Initializes the profile object for different types of particles in the simulation.

        Parameters:
        -----------
        sim : object
            The simulation data object containing particles of different types (e.g., stars, gas, dark matter).
        ndim : int, optional
            The number of dimensions for the profile (default is 2).
        type : str, optional
            The type of profile ('lin' for linear or other types as needed, default is 'lin').
        nbins : int, optional
            The number of bins to use in the profile (default is 100).
        rmin : float, optional
            The minimum radius for the profile (default is 0.1).
        rmax : float, optional
            The maximum radius for the profile (default is 100.0).
        **kwargs : additional keyword arguments
            Additional parameters to pass to the Profile initialization.
        """
        print(
            "Profile_1D -- assumes it's already at the center, and the disk is in the x-y plane"
        )
        print("If not, please use face_on()")
        self.__Pall = _Profile(
            sim, ndim=ndim, type=type, nbins=nbins, rmin=rmin, rmax=rmax, **kwargs
        )
        self.__Pstar = _Profile(
            sim.s, ndim=ndim, type=type, nbins=nbins, rmin=rmin, rmax=rmax, **kwargs
        )
        self.__Pgas = _Profile(
            sim.g, ndim=ndim, type=type, nbins=nbins, rmin=rmin, rmax=rmax, **kwargs
        )
        self.__Pdm = _Profile(
            sim.dm, ndim=ndim, type=type, nbins=nbins, rmin=rmin, rmax=rmax, **kwargs
        )

        self.__properties = {}
        self.__properties['Qgas'] = self.Qgas
        self.__properties['Qstar'] = self.Qstar
        self.__properties['Q2ws'] = self.Q2ws
        self.__properties['Q2thin'] = self.Q2thin
        self.__properties['Q2thick'] = self.Q2thick

        def v_circ(p, grav_sim=None):
            """Circular velocity, i.e. rotation curve. Calculated by computing the gravity
            in the midplane - can be expensive"""
            # print("Profile v_circ -- this routine assumes the disk is in the x-y plane")
            grav_sim = grav_sim or p.sim
            cal_2 = np.sqrt(2) / 2
            basearray = np.array(
                [
                    (1, 0, 0),
                    (0, 1, 0),
                    (-1, 0, 0),
                    (0, -1, 0),
                    (cal_2, cal_2, 0),
                    (-cal_2, cal_2, 0),
                    (cal_2, -cal_2, 0),
                    (-cal_2, -cal_2, 0),
                ]
            )
            R = p['rbins'].in_units('kpc').copy()
            POS = np.array([(0, 0, 0)])
            for j in R:
                binsr = basearray * j
                POS = np.concatenate((POS, binsr), axis=0)
            POS = SimArray(POS, R.units)
            ac = cal_acceleration(grav_sim, POS)
            ac.convert_units('kpc Gyr**-2')
            POS.convert_units('kpc')
            velall = np.diag(np.dot(ac - ac[0], -POS.T))
            if 'units' in dir(velall):
                velall.units = units.kpc**2 / units.Gyr**2
            else:
                velall = SimArray(velall, units.kpc**2 / units.Gyr**2)
            velTrue = np.zeros(len(R))
            for i in range(len(R)):
                velTrue[i] = np.mean(velall[i + 1 : 8 * (i + 1) + 1])
            velTrue[velTrue < 0] = 0
            velTrue = np.sqrt(velTrue)
            velTrue = SimArray(velTrue, units.kpc / units.Gyr)
            velTrue.convert_units('km s**-1')
            velTrue.sim = grav_sim.ancestor
            return velTrue

        def pot(p, grav_sim=None):
            grav_sim = grav_sim or p.sim
            cal_2 = np.sqrt(2) / 2
            basearray = np.array(
                [
                    (1, 0, 0),
                    (0, 1, 0),
                    (-1, 0, 0),
                    (0, -1, 0),
                    (cal_2, cal_2, 0),
                    (-cal_2, cal_2, 0),
                    (cal_2, -cal_2, 0),
                    (-cal_2, -cal_2, 0),
                ]
            )
            R = p['rbins'].in_units('kpc').copy()
            POS = np.array([(0, 0, 0)])
            for j in R:
                binsr = basearray * j
                POS = np.concatenate((POS, binsr), axis=0)
            POS = SimArray(POS, R.units)
            po = cal_potential(grav_sim, POS)
            po.conver_units('km**2 s**-2')
            poall = np.zeros(len(R))
            for i in range(len(R)):
                poall[i] = np.mean(po[i + 1 : 8 * (i + 1) + 1])

            poall = SimArray(poall, po.units)
            poall.sim = grav_sim.ancestor
            return poall

        def omega(p):
            """Circular frequency Omega = v_circ/radius (see Binney & Tremaine Sect. 3.2)"""
            prof = p['v_circ'] / p['rbins']
            prof.convert_units('km s**-1 kpc**-1')
            return prof

        self.__Pall._profile_registry[v_circ.__name__] = v_circ
        self.__Pall._profile_registry[omega.__name__] = omega
        self.__Pall._profile_registry[pot.__name__] = pot

        self.__Pstar._profile_registry[v_circ.__name__] = v_circ
        self.__Pstar._profile_registry[omega.__name__] = omega
        self.__Pstar._profile_registry[pot.__name__] = pot

        self.__Pgas._profile_registry[v_circ.__name__] = v_circ
        self.__Pgas._profile_registry[omega.__name__] = omega
        self.__Pgas._profile_registry[pot.__name__] = pot

        self.__Pdm._profile_registry[v_circ.__name__] = v_circ
        self.__Pdm._profile_registry[omega.__name__] = omega
        self.__Pdm._profile_registry[pot.__name__] = pot

    def __getitem__(self, key):

        if isinstance(key, str):
            ks = key.split('-')
            if len(ks) > 1:
                if set(['star', 's', 'Star']) & set(ks):
                    return self.__Pstar[ks[0]]
                if set(['gas', 'g', 'Gas']) & set(ks):
                    return self.__Pgas[ks[0]]
                if set(['dm', 'darkmatter', 'DM']) & set(ks):
                    return self.__Pdm[ks[0]]
                if set(['all', 'ALL']) & set(ks):
                    return self.__Pall[ks[0]]
            else:
                if key in self.__properties:
                    return self.__properties[key]()
                else:
                    return self.__Pall[key]
        else:
            print('Type error, should input a str')
            return

    def Qgas(self):
        '''
        Toomre-Q for gas
        '''
        return (
            self.__Pall['kappa']
            * self.__Pgas['vr_disp']
            / (np.pi * self.__Pgas['density'] * units.G)
        ).in_units("")

    def Qstar(self):
        '''
        Toomre-Q parameter
        '''
        return (
            self.__Pall['kappa']
            * self.__Pstar['vr_disp']
            / (3.36 * self.__Pstar['density'] * units.G)
        ).in_units("")

    def Q2ws(self):
        '''
        Toomre Q of two component. Wang & Silk (1994)
        '''
        Qs = (
            self.__Pall['kappa']
            * self.__Pstar['vr_disp']
            / (np.pi * self.__Pstar['density'] * units.G)
        ).in_units("")
        Qg = (
            self.__Pall['kappa']
            * self.__Pgas['vr_disp']
            / (np.pi * self.__Pgas['density'] * units.G)
        ).in_units("")
        return (Qs * Qg) / (Qs + Qg)

    def Q2thin(self):
        '''
        The effective Q of two component thin disk. Romeo & Wiegert (2011) eq. 6.
        '''
        w = (
            2
            * self.__Pstar['vr_disp']
            * self.__Pgas['vr_disp']
            / ((self.__Pstar['vr_disp']) ** 2 + self.__Pgas['vr_disp'] ** 2)
        ).in_units("")
        Qs = (
            self.__Pall['kappa']
            * self.__Pstar['vr_disp']
            / (np.pi * self.__Pstar['density'] * units.G)
        ).in_units("")
        Qg = (
            self.__Pall['kappa']
            * self.__Pgas['vr_disp']
            / (np.pi * self.__Pgas['density'] * units.G)
        ).in_units("")

        q = [Qs * Qg / (Qs + w * Qg)]
        return [
            (
                Qs[i] * Qg[i] / (Qs[i] + w[i] * Qg[i])
                if Qs[i] > Qg[i]
                else Qs[i] * Qg[i] / (w[i] * Qs[i] + Qg[i])
            )
            for i in range(len(w))
        ]

    def Q2thick(self):
        '''
        The effective Q of two component thick disk. Romeo & Wiegert (2011) eq. 9.
        '''
        w = (
            2
            * self.__Pstar['vr_disp']
            * self.__Pgas['vr_disp']
            / ((self.__Pstar['vr_disp']) ** 2 + self.__Pgas['vr_disp'] ** 2)
        ).in_units("")
        Ts = 0.8 + 0.7 * (self.__Pstar['vz_disp'] / self.__Pstar['vr_disp']).in_units(
            ""
        )
        Tg = 0.8 + 0.7 * (self.__Pgas['vz_disp'] / self.__Pgas['vr_disp']).in_units("")
        Qs = (
            self.__Pall['kappa']
            * self.__Pstar['vr_disp']
            / (np.pi * self.__Pstar['density'] * units.G)
        ).in_units("")
        Qg = (
            self.__Pall['kappa']
            * self.__Pgas['vr_disp']
            / (np.pi * self.__Pgas['density'] * units.G)
        ).in_units("")
        Qs = Qs * Ts
        Qg = Qg * Tg
        return [
            (
                Qs[i] * Qg[i] / (Qs[i] + w[i] * Qg[i])
                if Qs[i] > Qg[i]
                else Qs[i] * Qg[i] / (w[i] * Qs[i] + Qg[i])
            )
            for i in range(len(w))
        ]

from AnastrisTNG.TNGsnapshot import Basehalo
class Star_birth(Basehalo):
    '''the pos when the star form according to the host galaxy position'''
    
    #TODO face on matrix; on the edge of boxsize

    def __init__(self, Snap, subID, usebirthvel = True, usebirthmass = True):
        '''
        input:
        Snap,
        subID,
        '''

        originfield = Snap.load_particle_para['star_fields'].copy()
        Snap.load_particle_para['star_fields'] = ['Coordinates', 'Velocities', 'Masses', 'ParticleIDs',
                                                  'GFM_StellarFormationTime', 'GFM_InitialMass', 'BirthPos', 'BirthVel']
        PT = Snap.load_particle(subID, decorate = False, order = 'star',)
        Basehalo.__init__(self, PT)

        evo = Snap.galaxy_evolution(
            subID, ['SubhaloPos', 'SubhaloVel', 'SubhaloSpin'], physical_units=False
        )
        pos_ckpc = (evo['SubhaloPos']).view(np.ndarray) / self.h

        vel_ckpcGyr = (
            evo['SubhaloVel'].in_units('kpc Gyr**-1').view(np.ndarray).T / evo['a']
        ).T
        time_Gyr = evo['t'].in_units('Gyr')
        self.orbit = Orbit(pos_ckpc, vel_ckpcGyr, time_Gyr)
        self.s['BirthPos'].convert_units('a kpc')
        
        Birthpos = self.s['BirthPos'][
            (self.s['tform'] > self.orbit.tmin) & (self.s['tform'] < self.orbit.tmax)
        ]
        Birthvel = self.s['BirthVel'][
            (self.s['tform'] > self.orbit.tmin) & (self.s['tform'] < self.orbit.tmax)
        ]
        Birtha = self.s['aform'][
            (self.s['tform'] > self.orbit.tmin) & (self.s['tform'] < self.orbit.tmax)
        ]
        pos, vel = self.orbit.get(self.s['tform'].view(np.ndarray))
        galapos = pos[
            (self.s['tform'] > self.orbit.tmin) & (self.s['tform'] < self.orbit.tmax)
        ]
        galavel = vel[
            (self.s['tform'] > self.orbit.tmin) & (self.s['tform'] < self.orbit.tmax)
        ]
        
        distance = Birthpos - galapos
        
        # deal with periodic boundary
        boxsize = Snap.boxsize.in_units('a kpc').view(np.ndarray)
        
        distance[distance < -boxsize/2] = distance[distance < -boxsize/2] +boxsize
        distance[distance > boxsize/2] = distance[distance > boxsize/2] -boxsize
        
        distance_in_kpc = (distance.T * Birtha).T
        colVel = (Birthvel.in_units('kpc Gyr**-1 a**1/2').view(np.ndarray).T * np.sqrt(Birtha)).T - (galavel.T * Birtha).T   # unit: kpc / Gyr
        
        
        
        self.s['pos'].convert_units('a kpc')
        self.s['pos'] = self.s['pos'] - self.orbit.get(t=self.orbit.tmax)[0]
        self.s['pos'].convert_units('kpc')
        
        self.s['vel'].convert_units('a kpc Gyr**-1')
        self.s['vel'] = self.s['vel'] - self.orbit.get(t=self.orbit.tmax)[1]
        self.s['vel'].convert_units('km s**-1')
        
        
        self.s['pos'][
            (self.s['tform'] > self.orbit.tmin) & (self.s['tform'] < self.orbit.tmax)
        ] = distance_in_kpc
        
        if usebirthvel:
            self.s['vel'].convert_units('kpc Gyr**-1')
            self.s['vel'][
            (self.s['tform'] > self.orbit.tmin) & (self.s['tform'] < self.orbit.tmax)
            ] = colVel
            self.s['vel'].convert_units('km s**-1')
        if usebirthmass:
            self.s['mass'] = self.s['GFM_InitialMass']   
        self.s['mass'].convert_units('Msol')
        Snap.load_particle_para['star_fields'] = originfield            #recover
        
    def wrap(self):
        pass

def _process_file(file_info):
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
    """
    basePath, snapNum, fileNum, findIDset, istracerid = file_info
    result_local = {'ParentID': [], 'TracerID': []}

    gName = "PartType3"
    fields = ['ParentID', 'TracerID']

    with h5py.File(snapPath(basePath, snapNum, fileNum), 'r') as f:
        # print('open file')
        if gName not in f:
            print('skip', fileNum)
            return result_local
        #  print(len(f['PartType3']['TracerID'][:]))
        if istracerid:
            findresult = findIDset.isdisjoint(
                f['PartType3']['TracerID'][:]
            )  # time complexity O( min(len(set1),len(set2)) )
        else:
            findresult = findIDset.isdisjoint(f['PartType3']['ParentID'][:])

        if not findresult:

            ParentID = np.array(f[gName]['ParentID'])
            TracerID = np.array(f[gName]['TracerID'])

            if istracerid:
                Findepatticle = np.isin(TracerID, list(findIDset))
            else:
                Findepatticle = np.isin(ParentID, list(findIDset))

            result_local['TracerID'] = TracerID[Findepatticle]
            result_local['ParentID'] = ParentID[Findepatticle]

    return result_local


def findtracer_MP(
    basePath: str,
    snapNum: int,
    findID: List[int],
    *,
    istracerid: bool = False,
    NP: int = 6,
) -> dict:
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
    findID : list [int]
        1D array of the specified IDs (ParentIDs or TracerIDs) to find.
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
        numToRead = nPart[3]  # trecer num

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
        file_infos = [
            (basePath, snapNum, fileNum, findIDset, istracerid)
            for fileNum in file_numbers
        ]

        # progressing bar
        with tqdm(total=len(file_infos)) as pbar:
            # Use imap to process files and update the progress bar
            for result_local in pool.imap_unordered(_process_file, file_infos):
                result['TracerID'] = np.append(
                    result['TracerID'], result_local['TracerID']
                )
                result['ParentID'] = np.append(
                    result['ParentID'], result_local['ParentID']
                )
                # print(len(result_local['ParentID']),len(result['ParentID']))
                pbar.update(1)

    # Convert to integer type
    result['TracerID'] = result['TracerID'].astype(int)
    result['ParentID'] = result['ParentID'].astype(int)

    return result


def findtracer(
    basePath: str,
    snapNum: int,
    findID: List[int],
    *,
    istracerid: bool = False,
) -> dict:
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
    result['ParentID'] = np.array([])
    result['TracerID'] = np.array([])

    # PartType3, tracer
    ptNum = 3
    gName = "PartType" + str(ptNum)

    # Apart from ParentID and TracerID, there is also FluidQuantities in TNG100
    fields = ['ParentID', 'TracerID']

    findIDset = set(findID)

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
                fileOff = 0
                continue

            numTypeLocal = f['Header'].attrs['NumPart_ThisFile'][ptNum]
            numToReadLocal = numToRead

            if fileOff + numToReadLocal > numTypeLocal:
                numToReadLocal = numTypeLocal - fileOff

            if istracerid:
                findresult = findIDset.isdisjoint(
                    f['PartType3']['TracerID'][:]
                )  # time complexity O( min(len(set1),len(set2)) )
            else:
                findresult = findIDset.isdisjoint(f['PartType3']['ParentID'][:])

            if findresult == False:
                ParentID = np.array(f['PartType3']['ParentID'])
                TracerID = np.array(f['PartType3']['TracerID'])
                if istracerid:
                    Findepatticle = np.isin(
                        TracerID, findID
                    )  # time complexity O( len(array1)*len(array2) )
                else:
                    Findepatticle = np.isin(ParentID, findID)
                result['TracerID'] = np.append(
                    result['TracerID'], TracerID[Findepatticle]
                )
                result['ParentID'] = np.append(
                    result['ParentID'], ParentID[Findepatticle]
                )
                print(
                    'Number of tracers that have been matched: ',
                    len(result['TracerID']),
                )

            wOffset += numToReadLocal
            numToRead -= numToReadLocal
            fileNum += 1
            fileOff = 0

            f.close()
            pbar.update(numToReadLocal)

            # if matching TracerIDs, the number of tracers found must be the same as the len(findID).
            if istracerid and len(result['TracerID']) == len(findID):
                break
    result['TracerID'] = result['TracerID'].astype(int)
    result['ParentID'] = result['ParentID'].astype(int)
    return result


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
