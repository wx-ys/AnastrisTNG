'''
Some useful tools
find tracers: findtracer_MP(), findtracer(), Function.
potential: cal_potential, cal_acceleration, Function.
galaxy profile: single: profile(), all: Profile_1D(). Class.
...
'''

from typing import Any, Dict, List, Optional, Tuple
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
        
        elif name[-4:] == "_sum" and (name[:-4] in list(self.sim.keys()) or name[:-4] in self.sim.all_keys()):
            self._profiles[name] = self._auto_profile(name[:-4], sum=True)
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

    def _auto_profile(self, name, dispersion=False, rms=False, median=False,sum=False, q=None ):
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
            elif sum:
                result[i] = name_array.sum()
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

# a fix for pynbody kappa
@Profile.profile_property
def kappa(pro):
    """Radial frequency kappa = sqrt(R dOmega^2/dR + 4 Omega^2) (see Binney & Tremaine Sect. 3.2) in the z=0 plane"""
    dOmega2dR = (np.gradient(pro['omega'] ** 2) / np.gradient(pro['rbins'])).view(SimArray)  
    dOmega2dR.sim = pro.sim
    dOmega2dR.units = pro['omega'].units ** 2 / pro['rbins'].units
    return np.sqrt(pro['rbins'] * dOmega2dR + 4 * pro['omega'] ** 2)

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
    
    def __init__(self, Snap, ID, issubhalo = True, usebirthvel = True, usebirthmass = True, useCM = False):
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
                ID, ['SubhaloPos', 'SubhaloVel', 'SubhaloSpin','SubhaloCM'], physical_units=False
            )
            if useCM:
                pos_ckpc = (evo['SubhaloCM']).view(np.ndarray) / self.h
            else:
                pos_ckpc = (evo['SubhaloPos']).view(np.ndarray) / self.h

            vel_ckpcGyr = (
                evo['SubhaloVel'].in_units('kpc Gyr**-1').view(np.ndarray).T / evo['a']
            ).T
        else:
            evo = Snap.halo_evolution(
                ID, physical_units=False
            )
            if useCM:
                pos_ckpc = (evo['GroupCM']).view(np.ndarray) / self.h
            else:
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
        boxsize = Snap.boxsize.in_units('a kpc', **Snap.conversion_context()).view(np.ndarray)
        
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

_IDFINDER_FIND_IDS: Optional[np.ndarray] = None
Task = Tuple[str, int, int, str, List[str]]


def _idfinder_init_worker(find_ids: np.ndarray) -> None:
    global _IDFINDER_FIND_IDS
    _IDFINDER_FIND_IDS = find_ids


def _idfinder_worker_range(
    args: Task
) -> Tuple[Dict[str, np.ndarray], int]:
    file_path, start, end, id_field, return_fields = args

    if _IDFINDER_FIND_IDS is None:
        raise RuntimeError("IDFinder worker was not initialized with find IDs")

    return IDFinder._worker_range_impl(
        file_path=file_path,
        start=start,
        end=end,
        findID=_IDFINDER_FIND_IDS,
        id_field=id_field,
        return_fields=return_fields,
    )
class IDFinder:
    """
    Find particles in TNG snapshot files or a single HDF5 file by matching IDs.

    Parameters
    ----------
    basePath : str
        Base directory path of the simulation, or the path to a single HDF5
        snapshot chunk file when *snapNum* is omitted.
    snapNum : int, optional
        Snapshot number. When omitted, *basePath* is treated as a concrete
        HDF5 file path and only that file is scanned.

    Examples
    --------
    General particle lookup::

    >>> finder = IDFinder(basePath, snapNum)
    >>> result = finder.find(
    ...     ids,
    ...     id_field="PartType4/ParticleIDs",
    ...     return_fields=["PartType4/Coordinates", "PartType4/Masses"],
    ... )

    Tracer lookup::

    >>> finder = IDFinder(basePath, snapNum)
    >>> tracers = finder.find_tracers(star_ids, istracerid=False)
    >>> # tracers['TracerID'] -> tracer IDs attached to those stars

    Chaining tracers across snapshots::

    >>> now    = IDFinder(basePath, snapNumNow).find_tracers(star_ids)
    >>> before = IDFinder(basePath, snapNumBefore).find_tracers(
    ...              now['TracerID'], istracerid=True)
    >>> # before['ParentID'] -> progenitor gas/star ParticleIDs

    Notes
    -----
    - In sequential mode, results follow snapshot scan order, not input-ID order.
    - In multiprocessing mode, results are merged as worker tasks finish; result
      order is therefore undefined.
    - If a requested return field is missing from a matched file, a KeyError is raised.
    """

    TRACER_PARENT_FIELD = 'PartType3/ParentID'
    TRACER_ID_FIELD     = 'PartType3/TracerID'

    def __init__(self, basePath: str, snapNum: Optional[int] = None):
        self.basePath = basePath
        self.snapNum  = snapNum
        self._files = self._resolve_files()
        self.snapPath = self._files[0]

        with h5py.File(self.snapPath, 'r') as f:
            header = dict(f['Header'].attrs.items())
            self._nPart = getNumPart(header)

        self._numFiles = len(self._files)

    def _resolve_files(self) -> List[str]:
        """Resolve the concrete HDF5 file list to scan."""
        if self.snapNum is None:
            return [self.basePath]

        first_file = snapPath(self.basePath, self.snapNum, 0)
        with h5py.File(first_file, 'r') as f:
            num_files = int(f['Header'].attrs.get('NumFilesPerSnapshot', 1))

        return [snapPath(self.basePath, self.snapNum, fileNum) for fileNum in range(num_files)]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_field_path(field_path: str) -> Optional[int]:
        """
        Parse an HDF5 dataset path.

        Returns
        -------
        pt_num : int or None
            Particle type number, taken from the first path component that
            matches ``"PartTypeN"`` (N is an integer).  ``None`` for paths
            that contain no such component.

        Examples
        --------
        ``"PartType3/TracerID"``         -> 3
        ``"PartType3/SubGroup/Data"``    -> 3
        ``"particles/PartType4/Masses"`` -> 4
        ``"Header/NumPart"``             -> None
        """
        for part in field_path.split('/'):
            if part.startswith('PartType') and part[len('PartType'):].isdigit():
                return int(part[len('PartType'):])
        return None

    @staticmethod
    def _normalize_ids(findID: Any) -> np.ndarray:
        """
        Normalize IDs into a 1D numpy array.

        Accepts scalar IDs or array-like inputs.
        """
        ids = np.asarray(findID)
        if ids.ndim == 0:
            ids = ids.reshape(1)
        else:
            ids = ids.ravel()

        return ids

    @staticmethod
    def _empty_result(return_fields: List[str]) -> Dict[str, np.ndarray]:
        """Generate an empty result dict with the correct keys and empty arrays."""
        return {field: np.array([]) for field in return_fields}

    @staticmethod
    def _merge_chunks(
        chunks: Dict[str, List[np.ndarray]], 
        return_fields: List[str]
        ) -> Dict[str, np.ndarray]:
        """Merge lists of arrays from multiple chunks into single arrays for each field."""
        return {
            field: np.concatenate(chunks[field]) if chunks[field] else np.array([])
            for field in return_fields
        }

    @staticmethod
    def _validate_fields_in_file(
        f: h5py.File,
        id_field: str,
        return_fields: List[str],
    ) -> None:
        """
        Ensure the search field and all requested return fields exist.

        We only call this for files that contain the id_field dataset.
        """
        if id_field not in f:
            raise KeyError("Missing search field: %s" % id_field)

        missing_fields = [field for field in return_fields if field not in f]
        if missing_fields:
            raise KeyError(
                "Missing return field(s) in file %s: %s"
                % (getattr(f, 'filename', '<unknown>'), ', '.join(missing_fields))
            )

    @staticmethod
    def _iter_ranges(n_rows: int, max_rows_per_chunk: Optional[int]):
        if max_rows_per_chunk is None or max_rows_per_chunk <= 0:
            max_rows_per_chunk = n_rows

        for start in range(0, n_rows, max_rows_per_chunk):
            end = min(start + max_rows_per_chunk, n_rows)
            yield start, end

    @staticmethod
    def _worker_range_impl(
        file_path: str,
        start: int,
        end: int,
        findID: np.ndarray,
        id_field: str,
        return_fields: List[str],
    ) -> Tuple[Dict[str, np.ndarray], int]:
        result = IDFinder._empty_result(return_fields)

        try:
            with h5py.File(file_path, 'r') as f:
                if id_field not in f:
                    return result, 0

                IDFinder._validate_fields_in_file(f, id_field, return_fields)

                key_arr = f[id_field][start:end]
                #mask = np.isin(key_arr, findID)
                idx = np.searchsorted(findID, key_arr)
                valid = idx < findID.size
                mask = np.zeros_like(key_arr, dtype=bool)
                mask[valid] = findID[idx[valid]] == key_arr[valid]

                if mask.any():
                    for field in return_fields:
                        result[field] = f[field][start:end][mask]
        except OSError:
            return result, 0

        return result, end - start


    def _build_file_task_groups(
        self,
        id_field: str,
        return_fields: List[str],
        max_rows_per_chunk: Optional[int],
    ) -> Tuple[List[List[Task]], int]:
        task_groups: List[List[Task]] = []
        total_rows = 0

        for file_path in self._files:
            with h5py.File(file_path, 'r') as f:
                if id_field not in f:
                    continue

                IDFinder._validate_fields_in_file(f, id_field, return_fields)

                n_rows = int(f[id_field].shape[0])
                total_rows += n_rows

                file_tasks: List[Task] = []
                for start, end in self._iter_ranges(n_rows, max_rows_per_chunk):
                    file_tasks.append((file_path, start, end, id_field, return_fields))

                if file_tasks:
                    task_groups.append(file_tasks)
        return task_groups, total_rows

    @staticmethod
    def _flatten_task_groups(task_groups: List[List[Task]]) -> List[Task]:
        tasks: List[Task] = []
        for group in task_groups:
            tasks.extend(group)
        return tasks

    @staticmethod
    def _round_robin_task_groups(task_groups: List[List[Task]]) -> List[Task]:
        tasks: List[Task] = []
        max_group_len = max((len(group) for group in task_groups), default=0)

        for index in range(max_group_len):
            for group in task_groups:
                if index < len(group):
                    tasks.append(group[index])

        return tasks

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def find(
        self,
        findID,
        id_field: str,
        return_fields: List[str],
        *,
        stop_early: bool = False,
        stop_when_found: int = 0,
        NP: int = 1,
        max_rows_per_chunk: Optional[int] = 10_000_000,
    ) -> Dict[str, np.ndarray]:
        """
        Search snapshot chunk files for rows whose *id_field* value is in *findID*.

        Parameters
        ----------
        findID : array-like
            IDs to match against *id_field*.
        id_field : str
            HDF5 dataset path used as the search key,
            e.g. ``"PartType3/ParentID"`` or ``"PartType3/SubGroup/ID"``.
        return_fields : list of str
            HDF5 dataset paths to collect for matching rows.
            The returned dict always includes *id_field* as well, even if it is
            not explicitly listed here.
        stop_early : bool, optional
            When ``True``, stop scanning once all unique requested IDs are found.
            This is only meaningful for one-to-one mappings and is ignored when ``NP > 1``.
        stop_when_found : int, optional
            Advanced override: stop after exactly this many cumulative matches.
            ``0`` (default) defers to *stop_early*. Ignored when ``NP > 1``.
        NP : int, optional
            Number of worker processes. ``NP <= 1`` runs sequentially.
        max_rows_per_chunk : int or None, optional
            Maximum number of rows to read per chunk.

        Returns
        -------
        dict
            Mapping from field path to numpy array. The returned dict always
            contains *id_field* and all requested *return_fields*.

        Examples
        --------
        Sequential — find star masses by ParticleID::

            result = IDFinder(basePath, snapNum).find(
                ids,
                id_field="PartType4/ParticleIDs",
                return_fields=["PartType4/Coordinates", "PartType4/Masses"],
            )

        Parallel with 8 processes::

            result = IDFinder(basePath, snapNum).find(
                ids,
                id_field="PartType4/ParticleIDs",
                return_fields=["PartType4/Masses"],
                NP=8,
            )

        Stop as soon as all TracerIDs are found (sequential only)::

            result = IDFinder(basePath, snapNum).find(
                tracer_ids,
                id_field="PartType3/TracerID",
                return_fields=["PartType3/ParentID"],
                stop_early=True,
            )
        Notes
        -----
        - Sequential mode returns rows in snapshot scan order, not input-ID order.
        - Multiprocessing mode does not guarantee a stable output order.
        - Duplicate input IDs are deduplicated before the search.
        - Multiprocessing scans chunk ranges in parallel.
        - stop_early / stop_when_found are only honored in sequential mode
        """
        # --- sequential path ---
        # Resolve the effective early-stop count:
        #   stop_when_found > 0  -> use it directly (advanced override)
        #   stop_early=True      -> stop once all requested IDs are found
        #   otherwise            -> scan every chunk file
        pt_num = self._parse_field_path(id_field)
        return_fields = list(return_fields)
        if id_field not in return_fields:
            return_fields.insert(0, id_field)

        if pt_num is not None and not self._nPart[pt_num]:
            return self._empty_result(return_fields)

        findID = self._normalize_ids(findID)
        if findID.size == 0:
            return self._empty_result(return_fields)

        # Reduce redundant comparisons when the caller passes duplicated IDs.
        findID = np.unique(findID)

        stop_target = stop_when_found if stop_when_found > 0 else (len(findID) if stop_early else 0)

        task_groups, total_rows = self._build_file_task_groups(
            id_field,
            return_fields,
            max_rows_per_chunk,
        )

        if not task_groups:
            return self._empty_result(return_fields)

        if NP > 1:
            tasks = self._round_robin_task_groups(task_groups)
            return self._find_mp(
                findID,
                id_field,
                return_fields,
                NP=NP,
                max_rows_per_chunk=max_rows_per_chunk,
                tasks=tasks,
                total_rows=total_rows,
            )

        tasks = self._flatten_task_groups(task_groups)
        chunks: Dict[str, List[np.ndarray]] = {field: [] for field in return_fields}
        total_matched = 0

        with tqdm(total=total_rows) as pbar:
            for file_path, start, end, _, _ in tasks:
                result_local, scanned_rows = IDFinder._worker_range_impl(
                    file_path=file_path,
                    start=start,
                    end=end,
                    findID=findID,
                    id_field=id_field,
                    return_fields=return_fields,
                )

                for field in return_fields:
                    if len(result_local[field]):
                        chunks[field].append(result_local[field])

                if len(result_local[id_field]):
                    total_matched += len(result_local[id_field])

                pbar.update(scanned_rows)

                if stop_target and total_matched >= stop_target:
                    break

        return self._merge_chunks(chunks, return_fields)

    def _find_mp(
        self,
        findID: np.ndarray,
        id_field: str,
        return_fields: List[str],
        *,
        NP: int,
        max_rows_per_chunk: Optional[int],
        tasks: Optional[List[Task]] = None,
        total_rows: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Internal multiprocessing backend used by find when NP > 1.

        Notes
        -----
        Results are merged in task-completion order, so the output ordering is
        undefined and may differ between runs.
        """
        pt_num = self._parse_field_path(id_field)
        return_fields = list(return_fields)
        if id_field not in return_fields:
            return_fields.insert(0, id_field)

        if pt_num is not None and not self._nPart[pt_num]:
            return self._empty_result(return_fields)

        if tasks is None or total_rows is None:
            task_groups, total_rows = self._build_file_task_groups(
                id_field,
                return_fields,
                max_rows_per_chunk,
            )
            tasks = self._round_robin_task_groups(task_groups)

        if not tasks:
            return self._empty_result(return_fields)

        chunks: Dict[str, List[np.ndarray]] = {field: [] for field in return_fields}

        with mp.Pool(
            processes=min(NP, len(tasks)),
            initializer=_idfinder_init_worker,
            initargs=(findID,),
        ) as pool:
            with tqdm(total=total_rows) as pbar:
                for result_local, scanned_rows in pool.imap_unordered(_idfinder_worker_range, tasks):
                    for field in return_fields:
                        if len(result_local[field]):
                            chunks[field].append(result_local[field])
                    pbar.update(scanned_rows)

        return self._merge_chunks(chunks, return_fields)
        
        
        

    def find_tracers(
        self,
        findID,
        *,
        istracerid: bool = False,
        NP: int = 1,
        max_rows_per_chunk: Optional[int] = 10_000_000,
    ) -> dict:
        """
        Find Monte-Carlo tracers by ParentID or TracerID.

        Parameters
        ----------
        findID : array-like or scalar
            IDs to search for.
        istracerid : bool, optional
            If ``True``, match TracerIDs; otherwise match ParentIDs.
            Default ``False``.
        NP : int, optional
            Number of worker processes.  ``NP <= 1`` (default) runs
            sequentially; ``NP > 1`` uses a multiprocessing pool.
        max_rows_per_chunk : int or None, optional
            Maximum number of rows to read per chunk when scanning files.
            Default is 10 million.

        Returns
        -------
        dict
            ``{'ParentID': np.ndarray, 'TracerID': np.ndarray}``

        Notes
        -----
        Works for all snapshots in TNG50 and TNG300, but only the 20 full
        snapshots for TNG100.

        When ``NP > 1`` the result order is undefined.
        When matching ParentIDs the result count may differ from ``len(findID)``
        because one parent can have zero or multiple tracers.
        When matching TracerIDs the result count should match the number of
        unique input tracer IDs if the snapshot data are complete.
        """
        id_field      = self.TRACER_ID_FIELD if istracerid else self.TRACER_PARENT_FIELD
        return_fields = [self.TRACER_PARENT_FIELD, self.TRACER_ID_FIELD]
        # istracerid=True is 1:1 -> safe to stop early (sequential only; NP>1 ignores it)
        raw = self.find(
            findID, id_field, return_fields, 
            stop_early=istracerid, NP=NP,max_rows_per_chunk=max_rows_per_chunk)
        return {
            'ParentID': raw[self.TRACER_PARENT_FIELD].astype(int),
            'TracerID': raw[self.TRACER_ID_FIELD].astype(int),
        }


# ---------------------------------------------------------------------------
# Module-level convenience wrappers (backward compatibility)
# ---------------------------------------------------------------------------

def find_by_id(
    basePath: str,
    snapNum: int,
    findID,
    id_field: str,
    return_fields: List[str],
    *,
    stop_early: bool = False,
    stop_when_found: int = 0,
    NP: int = 1,
    max_rows_per_chunk: Optional[int] = 10_000_000,
) -> Dict[str, np.ndarray]:
    """
    Search snapshot chunk files by matching IDs.

    Thin wrapper around :meth:`IDFinder.find`.

    Notes
    -----
    The returned dict always includes *id_field*, even if it is not explicitly
    included in *return_fields*. When ``NP > 1`` the result order is undefined.
    """
    return IDFinder(basePath, snapNum).find(
        findID,
        id_field,
        return_fields,
        stop_early=stop_early,
        stop_when_found=stop_when_found,
        NP=NP,
        max_rows_per_chunk=max_rows_per_chunk,
    )

def findtracer(
    basePath: str,
    snapNum: int,
    findID,
    *,
    istracerid: bool = False,
    NP: int = 1,
    max_rows_per_chunk: Optional[int] = 10_000_000,
) -> Dict[str, np.ndarray]:
    """
    Find MC tracers by ParentID or TracerID.

    Thin wrapper around :meth:`IDFinder.find_tracers`.

    Notes
    -----
    When ``NP > 1`` the result order is undefined.
    """
    return IDFinder(basePath, snapNum).find_tracers(
        findID,
        istracerid=istracerid,
        NP=NP,
        max_rows_per_chunk=max_rows_per_chunk,
    )



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
