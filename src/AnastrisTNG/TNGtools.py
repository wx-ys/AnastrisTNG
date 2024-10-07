'''
Some useful tools
find tracers: findtracer_MP(), findtracer(), Function.
star both pos: Star_birth(), Class.
potential: cal_potential, cal_acceleration, Function.
galaxy profile: single: profile(), all: Profile_1D(). Class.
...
'''

from typing import List
import multiprocessing as mp
import re
import math

import numpy as np
import h5py
from tqdm import tqdm
from pynbody import units, filt
from pynbody.array import SimArray
from pynbody.analysis.profile import Profile 

from AnastrisTNG.illustris_python.snapshot import *
from AnastrisTNG.Anatools import Orbit
from AnastrisTNG.pytreegrav import PotentialTarget, AccelTarget
from AnastrisTNG.TNGsnapshot import Basehalo


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

class profile(Profile):
    
    def _calculate_x(self, sim):
        if self.zmax:
            return SimArray(np.abs(sim['z']), sim['z'].units)
        else:
            return ((sim['pos'][:, 0:self.ndim] ** 2).sum(axis=1)) ** (1, 2)
    
    def __init__(self, sim, rmin = 0.1, rmax = 30, 
                 nbins=100, ndim=2, type='lin', weight_by='mass', calc_x=None, **kwargs):
        
        zmax = kwargs.get('zmax', None)
        self.zmax = zmax
        if isinstance(zmax, str):
            zmax = units.Unit(zmax)
        
        if self.zmax:
            if isinstance(rmin, str):
                rmin = units.Unit(rmin)
            if isinstance(rmax, str):
                rmax = units.Unit(rmax)
            self.rmin = rmin
            self.rmax = rmax

            assert ndim in [2, 3]
            if ndim == 3:
                sub_sim = sim[
                    filt.Disc(rmax, zmax) & ~filt.Disc(rmin, zmax)]
            else:
                sub_sim = sim[(filt.BandPass('x', rmin, rmax) |
                            filt.BandPass('x', -rmax, -rmin)) &
                            filt.BandPass('z', -zmax, zmax)]

            Profile.__init__(
                self, sub_sim, nbins=nbins, weight_by=weight_by, 
                ndim=ndim, type=type, **kwargs)
        else:
            Profile.__init__(
                self, sim, rmin=rmin, rmax=rmax, nbins=nbins, weight_by=weight_by, 
                ndim=ndim, type=type, **kwargs)
    
        
    def _setup_bins(self):
        Profile._setup_bins(self)
        if self.zmax:
            dr = self.rmax - self.rmin

            if self.ndim == 2:
                self._binsize = (
                    self['bin_edges'][1:] - self['bin_edges'][:-1]) * dr
            else:
                area = SimArray(
                    np.pi * (self.rmax ** 2 - self.rmin ** 2), "kpc^2")
                self._binsize = (
                    self['bin_edges'][1:] - self['bin_edges'][:-1]) * area
    def _get_profile(self, name):
        """Return the profile of a given kind"""
        x = name.split(",")
        find = re.search(r'_\d+',name)
        if name in self._profiles:
            return self._profiles[name]

        elif x[0] in Profile._profile_registry:
            args = x[1:]
            self._profiles[name] = Profile._profile_registry[x[0]](self, *args)
            try:
                self._profiles[name].sim = self.sim
            except AttributeError:
                pass
            return self._profiles[name]

        elif name in list(self.sim.keys()) or name in self.sim.all_keys():
            self._profiles[name] = self._auto_profile(name)
            self._profiles[name].sim = self.sim
            return self._profiles[name]

        elif name[-5:] == "_disp" and (name[:-5] in list(self.sim.keys()) or name[:-5] in self.sim.all_keys()):
            self._profiles[name] = self._auto_profile(
                name[:-5], dispersion=True)
            self._profiles[name].sim = self.sim
            return self._profiles[name]

        elif name[-4:] == "_rms" and (name[:-4] in list(self.sim.keys()) or name[:-4] in self.sim.all_keys()):
            self._profiles[name] = self._auto_profile(name[:-4], rms=True)
            self._profiles[name].sim = self.sim
            return self._profiles[name]

        elif name[-4:] == "_med" and (name[:-4] in list(self.sim.keys()) or name[:-4] in self.sim.all_keys()):
            self._profiles[name] = self._auto_profile(name[:-4], median=True)
            self._profiles[name].sim = self.sim
            return self._profiles[name]

        elif name[0:2] == "d_" and (name[2:] in list(self.keys()) or name[2:] in self.derivable_keys() or name[2:] in self.sim.all_keys()):
            #            if np.diff(self['dr']).all() < 1e-13 :
            self._profiles[name] = np.gradient(self[name[2:]], self['dr'][0])
            self._profiles[name] = self._profiles[name] / self['dr'].units
            return self._profiles[name]
            # else :
            #    raise RuntimeError, "Derivatives only possible for profiles of fixed bin width."
        elif find and (name[:find.start()] in list(self.sim.keys()) or name[:find.start()] in self.sim.all_keys()):
            self._profiles[name] = self._auto_profile(name[:find.start()], q = float(name[find.start()+1:]))
            self._profiles[name].sim = self.sim
            return self._profiles[name]
            
        else:
            raise KeyError(name + " is not a valid profile")

    def _auto_profile(self, name, dispersion=False, rms=False, median=False, q=None ):
        result = np.zeros(self.nbins)

        # force derivation of array if necessary:
        self.sim[name]

        for i in range(self.nbins):
            subs = self.sim[self.binind[i]]
            name_array = subs[name].view(np.ndarray)
            mass_array = subs[self._weight_by].view(np.ndarray)

            if dispersion:
                sq_mean = (name_array ** 2 * mass_array).sum() / \
                    self['weight_fn'][i]
                mean_sq = (
                    (name_array * mass_array).sum() / self['weight_fn'][i]) ** 2
                try:
                    result[i] = math.sqrt(sq_mean - mean_sq)
                except ValueError:
                    # sq_mean<mean_sq occasionally from numerical roundoff
                    result[i] = 0

            elif rms:
                result[i] = np.sqrt(
                    (name_array ** 2 * mass_array).sum() / self['weight_fn'][i])
            elif median:
                if len(subs) == 0:
                    result[i] = np.nan
                else:
                    sorted_name = sorted(name_array)
                    result[i] = sorted_name[int(np.floor(0.5 * len(subs)))]
            elif q:
                if len(subs) == 0:
                    result[i] = np.nan
                else:
                    sorted_name = sorted(name_array)
                    weight_array = mass_array[np.argsort(name_array)]
                    cumw = np.cumsum(weight_array) / np.sum(weight_array)
                    imin = min(
                            np.arange(len(sorted_name)), key=lambda x: abs(cumw[x] - q/100))
                    inc = q/100 - cumw[imin]
                    lowval = sorted_name[imin]
                    if inc > 0:
                        nextval = sorted_name[imin + 1]
                    else:
                        if imin == 0:
                            nextval = lowval
                        else:
                            nextval = sorted_name[imin - 1]

                    result[i] = lowval + inc * (nextval - lowval)
                    #result[i] = sorted_name[cumw*100>q].min()+sorted_name[cumw*100<q].max()
            else:
                result[i] = (name_array * mass_array).sum() / self['weight_fn'][i]

        result = result.view(SimArray)
        result.units = self.sim[name].units
        result.sim = self.sim
        return result
    
@Profile.profile_property
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
@Profile.profile_property
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
    po.convert_units('km**2 s**-2')
    poall = np.zeros(len(R))
    for i in range(len(R)):
        poall[i] = np.mean(po[i + 1 : 8 * (i + 1) + 1])

    poall = SimArray(poall, po.units)
    poall.sim = grav_sim.ancestor
    return poall

@Profile.profile_property
def omega(p):
    """Circular frequency Omega = v_circ/radius (see Binney & Tremaine Sect. 3.2)"""
    prof = p['v_circ'] / p['rbins']
    prof.convert_units('km s**-1 kpc**-1')
    return prof

class Profile_1D:
    _properties={}
    def __init__(
        self, sim, rmin=0.1, rmax=100.0, zmax = 5.,nbins=100, type='lin', **kwargs
    ):
        """
        Initializes the profile object for different types of particles in the simulation.

        Parameters:
        -----------
        sim : object
            The simulation data object containing particles of different types (e.g., stars, gas, dark matter).
        rmin : float, optional
            The minimum radius for the profile (default is 0.1).
        rmax : float, optional
            The maximum radius for the profile (default is 100.0).
        zmax : float, optional
            maximum height to consider (default is 5.0).
        nbins : int, optional
            The number of bins to use in the profile (default is 100).
        type : str, optional
            The type of profile ('lin' for linear or other types as needed, default is 'lin').

        **kwargs : additional keyword arguments
            Additional parameters to pass to the Profile initialization.
            
        Usage: str like 'A-B-C'
                A: the parameter key,  d_A, derivatives, A_disp, A_med, A_rms, A_30 ...
                B: family, 'star', 'gas', 'dm', 'all'
                C: direction and dims, 'z', 'Z', 'r', 'R'; 'z' vertical and 3 dims, 'Z' 2dims ... 
            examples : 'vr-star-R'
        """
        print(
            "Profile_1D -- assumes it's already at the center, and the disk is in the x-y plane"
        )
        print("If not, please use face_on()")
        self.__P={'all':{}, 'star':{}, 'gas':{}, 'dm':{}}
        self.__P['all']['r']=profile(sim, rmin=rmin, rmax=rmax, nbins=nbins,ndim=3, type=type, **kwargs)
        self.__P['all']['R']=profile(sim, rmin=rmin, rmax=rmax, nbins=nbins, ndim=2, type=type, **kwargs)
        self.__P['all']['Z']=profile(sim, rmin=rmin, rmax=rmax, nbins=nbins, ndim=2, type=type, zmax = zmax, **kwargs)
        self.__P['all']['z']=profile(sim, rmin=rmin, rmax=rmax, nbins=nbins, ndim=3, type=type, zmax = zmax,**kwargs)
        
        self.__P['star']['r']=profile(sim.s, rmin=rmin, rmax=rmax, nbins=nbins,ndim=3, type=type, **kwargs)
        self.__P['star']['R']=profile(sim.s, rmin=rmin, rmax=rmax, nbins=nbins, ndim=2, type=type, **kwargs)
        self.__P['star']['Z']=profile(sim.s, rmin=rmin, rmax=rmax, nbins=nbins, ndim=2, type=type, zmax = zmax, **kwargs)
        self.__P['star']['z']=profile(sim, rmin=rmin, rmax=rmax, nbins=nbins, ndim=3, type=type, zmax = zmax,**kwargs)
        try:
            self.__P['gas']['r']=profile(sim.g, rmin=rmin, rmax=rmax, nbins=nbins,ndim=3, type=type, **kwargs)
        except:
            print('No gas r')
        try:
            self.__P['gas']['R']=profile(sim.g, rmin=rmin, rmax=rmax, nbins=nbins, ndim=2, type=type, **kwargs)
        except:
            print('No gas R')
        try:
            self.__P['gas']['Z']=profile(sim.g, rmin=rmin, rmax=rmax, nbins=nbins, ndim=2, type=type, zmax = zmax, **kwargs)
        except:
            print('No gas Z')
        try:
            self.__P['gas']['z']=profile(sim.g, rmin=rmin, rmax=rmax, nbins=nbins, ndim=3, type=type, zmax = zmax,**kwargs)
        except:
            print('No gas z')
        
        self.__P['dm']['r']=profile(sim.dm, rmin=rmin, rmax=rmax, nbins=nbins, ndim=3, type=type, **kwargs)
        self.__P['dm']['R']=profile(sim.dm, rmin=rmin, rmax=rmax, nbins=nbins, ndim=2, type=type, **kwargs)
        self.__P['dm']['Z']=profile(sim.dm, rmin=rmin, rmax=rmax, nbins=nbins, ndim=2, type=type, zmax = zmax, **kwargs)
        self.__P['dm']['z']=profile(sim.dm, rmin=rmin, rmax=rmax, nbins=nbins, ndim=3, type=type, zmax = zmax,**kwargs)

    
    def _util_fa(self, ks):
        if set(['star', 's', 'Star']) & set(ks):
            return 'star'
        if set(['gas', 'g', 'Gas']) & set(ks):
            return 'gas'
        if set(['dm', 'DM']) & set(ks):
            return 'dm'
        if set(['all', 'ALL']) & set(ks):
            return 'all'
        return 'all'
    
    def _util_pr(self, ks):
        if set(['r']) & set(ks):
            return 'r'
        if set(['z']) & set(ks):
            return 'z'
        if set(['R']) & set(ks):
            return 'R'
        if set(['Z']) & set(ks):
            return 'Z'
        return 'R'   

    def __getitem__(self, key):

        if isinstance(key, str):
            ks = key.split('-')
            if len(ks) > 1:
                return self.__P[self._util_fa(ks)][self._util_pr(ks)][ks[0]]
            else:
                if key in self._properties:
                    return self._properties[key](self)
                else:
                    return self.__P['all']['R'][key]
        else:
            print('Type error, should input a str')
            return
    @staticmethod
    def profile_property(fn):
        Profile_1D._properties[fn.__name__] = fn
        return fn
    
@Profile_1D.profile_property    
def Qgas(self):
    '''
    Toomre-Q for gas
    '''
    return (
        self['kappa-all-R']
        * self['vrxy_disp-gas-R']
        / (np.pi * self['density-gas-R'] * units.G)
    ).in_units("")
    
@Profile_1D.profile_property  
def Qstar(self):
    '''
    Toomre-Q parameter
    '''
    return (
        self['kappa-all-R']
        * self['vrxy_disp-star-R']
        / (3.36 * self['density-star-R'] * units.G)
    ).in_units("")
    
@Profile_1D.profile_property  
def Qs(self):
    '''
    Toomre-Q parameter
    '''
    return (
        self['kappa-all-R']
        * self['vrxy_disp-star-R']
        / (np.pi * self['density-star-R'] * units.G)
    ).in_units("")
    
@Profile_1D.profile_property  
def Q2ws(self):
    '''
    Toomre Q of two component. Wang & Silk (1994)
    '''
    Qs = self['Qs']
    Qg = self['Qgas']
    return (Qs * Qg) / (Qs + Qg)

@Profile_1D.profile_property  
def Q2thin(self):
    '''
    The effective Q of two component thin disk. Romeo & Wiegert (2011) eq. 6.
    '''
    w = (
        2
        * self['vrxy_disp-star-R']
        * self['vrxy_disp-gas-R']
        / ((self['vrxy_disp-star-R']) ** 2 + self['vrxy_disp-gas-R'] ** 2)
    ).in_units("")
    Qs = self['Qs']
    Qg = self['Qgas']
    q = [Qs * Qg / (Qs + w * Qg)]
    return [
        (
            Qs[i] * Qg[i] / (Qs[i] + w[i] * Qg[i])
            if Qs[i] > Qg[i]
            else Qs[i] * Qg[i] / (w[i] * Qs[i] + Qg[i])
        )
        for i in range(len(w))
    ]
    
@Profile_1D.profile_property  
def Q2thick(self):
    '''
    The effective Q of two component thick disk. Romeo & Wiegert (2011) eq. 9.
    '''
    w = (
        2
        * self['vrxy_disp-star-R']
        * self['vrxy_disp-gas-R']
        / ((self['vrxy_disp-star-R']) ** 2 + self['vrxy_disp-gas-R'] ** 2)
    ).in_units("")
    Ts = 0.8 + 0.7 * (self['vz_disp-star-R'] / self['vrxy_disp-star-R']).in_units(
        ""
    )
    Tg = 0.8 + 0.7 * (self['vz_disp-gas-R'] / self['vrxy_disp-gas-R']).in_units("")
    Qs = self['Qs']
    Qg = self['Qgas']
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


class Star_birth(Basehalo):
    '''the pos when the star form according to the host galaxy position'''
    
    def __init__(self, Snap, ID, issubhalo = True, usebirthvel = True, usebirthmass = True):
        '''
        input:
        Snap,
        ID,
        '''

        originfield = Snap.load_particle_para['star_fields'].copy()
        Snap.load_particle_para['star_fields'] = ['Coordinates', 'Velocities', 'Masses', 'ParticleIDs',
                                                  'GFM_StellarFormationTime', 'GFM_InitialMass', 'BirthPos', 'BirthVel']
        if issubhalo:
            PT = Snap.load_particle(ID, groupType = 'Subhalo', decorate = False, order = 'star',)
        else:
            PT = Snap.load_particle(ID, groupType = 'Halo',decorate = False, order = 'star',)
        Basehalo.__init__(self, PT)
        if issubhalo:
            evo = Snap.galaxy_evolution(
                ID, ['SubhaloPos', 'SubhaloVel', 'SubhaloSpin'], physical_units=False
            )
            pos_ckpc = (evo['SubhaloPos']).view(np.ndarray) / self.h

            vel_ckpcGyr = (
                evo['SubhaloVel'].in_units('kpc Gyr**-1').view(np.ndarray).T / evo['a']
            ).T
        else:
            evo = Snap.halo_evolution(
                ID, physical_units=False
            )
            pos_ckpc = (evo['GroupPos']).view(np.ndarray) / self.h

            vel_ckpcGyr = (
                evo['GroupVel'].in_units('kpc Gyr**-1 a**-1').view(np.ndarray).T / evo['a'] /evo['a']
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
