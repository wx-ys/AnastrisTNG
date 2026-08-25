'''
Basehalo for subhalo and halo
Derived array for some particle types
'''

from functools import reduce
from typing import List

import numpy as np
from pynbody import units, filt, derived_array, config
from pynbody.family import get_family
from pynbody.simdict import SimDict
from pynbody.array import SimArray
from pynbody.snapshot import SubSnap
from pynbody.analysis.angmom import calc_faceon_matrix
        
from AnastrisTNG.TNGunits import illustrisTNGruns, NotneedtransGCPa
from AnastrisTNG.pytreegrav import Potential, Accel
from AnastrisTNG.Anatools import ang_mom, fit_krotmax, MoI_shape


def gravity_parallel() -> bool:
    """ Check if gravity calculations should be run in parallel based on the pynbody configuration."""
    return config["threading"] in ['True', True, 'true', '1', 1]

class Basehalo(SubSnap):
    """
    Represents a single halo in the simulation.

    This class contains information about the particles of the halo and its corresponding group catalog data.
    It also includes functions to compute properties specific to this halo.

    Attributes:
    ----------
    GC : SimDict
        The group catalog for this halo. Detailed information about this can be found at
        https://www.tng-project.org/data/docs/specifications/#sec2.

    Parameters:
    ----------
    simarray : SimArray
        An object containing the particle data for the halo.

    """

    def __init__(self, simarray):
        """
        Initializes the Halo object.

        Parameters:
        -----------
        simarray : object
            An object that contains halo particles.
        """
        SubSnap.__init__(self, simarray, slice(len(simarray)))
        self.GC = SimDict()
        self.GC.update(simarray.properties)


    def physical_units(self, persistent: bool = False):
        """
        Convert the units of the simulation arrays and properties to physical units.
            the conversion is temporary (default is False).
        """
        if (len(self) != len(self.ancestor)) or (hasattr(self.ancestor, '_canloadPT')):
            self.ancestor.physical_units(persistent=persistent)
        else:
            dims = self.properties['baseunits'] + [units.a, units.h]
            urc = len(dims) - 2
            all = list(self.ancestor._arrays.values())
            for x in self.ancestor._family_arrays:
                if x in self.properties.get('staunit', []):
                    continue
                else:
                    all += list(self.ancestor._family_arrays[x].values())

            for ar in all:
                if ar.units is not units.no_unit:
                    self._autoconvert_array_unit(ar.ancestor, dims, urc)

            for k in list(self.properties):
                v = self.properties[k]
                if isinstance(v, units.UnitBase):
                    try:
                        new_unit = v.dimensional_project(dims)
                    except units.UnitsException:
                        continue
                    new_unit = reduce(
                        lambda x, y: x * y, [a**b for a, b in zip(dims, new_unit[:urc])]
                    )
                    new_unit *= v.ratio(new_unit, **self.conversion_context())
                    self.properties[k] = new_unit
                if isinstance(v, SimArray):
                    if (v.units is not None) and (v.units is not units.no_unit):
                        try:
                            d = v.units.dimensional_project(dims)
                        except units.UnitsException:
                            return
                        new_unit = reduce(
                            lambda x, y: x * y, [a**b for a, b in zip(dims, d[:urc])]
                        )
                        if new_unit != v.units:
                            self.properties[k].convert_units(new_unit)
            self.GC_physical_units()
            if persistent:
                self._autoconvert = dims
            else:
                self._autoconvert = None

    def GC_physical_units(self, distance='kpc', velocity='km s^-1', mass='Msol'):
        """
        Converts the units of the group catalog (GC) properties to physical units.

        This method updates the `GC` attribute of the `Subhalo` instance to use physical units
        for its properties, based on predefined unit conversions and the current unit context.

        Conversion is applied only to properties that are not listed in `NotneedtransGCPa`.

        Notes:
        -----
        - `self.ancestor.properties['baseunits']` provides the base units for dimensional analysis.
        - The dimensional projection and conversion are handled using the `units` library.
        - Properties listed in `NotneedtransGCPa` are skipped during the conversion process.
        """
        dims = self.ancestor.properties['baseunits'] + [units.a, units.h]
        urc = len(dims) - 2
        for k in list(self.GC):
            if k in NotneedtransGCPa:
                continue
            v = self.GC[k]
            if isinstance(v, units.UnitBase):
                try:
                    new_unit = v.dimensional_project(dims)
                except units.UnitsException:
                    continue
                new_unit = reduce(
                    lambda x, y: x * y, [a**b for a, b in zip(dims, new_unit[:urc])]
                )
                new_unit *= v.ratio(new_unit, **self.conversion_context())
                self.GC[k] = new_unit
            if isinstance(v, SimArray):
                if (v.units is not None) and (v.units is not units.no_unit):
                    try:
                        d = v.units.dimensional_project(dims)
                    except units.UnitsException:
                        return
                    new_unit = reduce(
                        lambda x, y: x * y, [a**b for a, b in zip(dims, d[:urc])]
                    )
                    if new_unit != v.units:
                        self.GC[k].convert_units(new_unit)

    def vel_center(self, mode='ssc', pos=None, r_cal='2 kpc'):
        '''
        The center velocity.
        Refer from https://pynbody.readthedocs.io/latest/_modules/pynbody/analysis/halo.html#vel_center

        ``mode`` used to cal center pos see ``center``
        ``pos``  Specified position.
        ``r_cal`` The size of the sphere to use for the velocity calculate

        '''
        if self.__check_paticles():
            print('No particles loaded in this Halo')
            return

        if pos is None:
            pos = self.center(mode)

        cen = self.s[filt.Sphere(r_cal, pos)]
        if len(cen) < 5:
            # fall-back to DM
            cen = self.dm[filt.Sphere(r_cal, pos)]
        if len(cen) < 5:
            # fall-back to gas
            cen = self.g[filt.Sphere(r_cal, pos)]
        if len(cen) < 5:
            cen = self[filt.Sphere(r_cal, pos)]
        if len(cen) < 5:
            # very weird snapshot, or mis-centering!
            raise ValueError("Insufficient particles around center to get velocity")

        vcen = (cen['vel'].transpose() * cen['mass']).sum(axis=1) / cen['mass'].sum()
        vcen.units = cen['vel'].units

        return vcen

    def center(self, mode='ssc'):
        '''
        The position center of this snapshot
        Refer from https://pynbody.readthedocs.io/latest/_modules/pynbody/analysis/halo.html#center

        The centering scheme is determined by the ``mode`` keyword. As well as the
        The following centring modes are available:

        *  *pot*: potential minimum

        *  *com*: center of mass

        *  *ssc*: shrink sphere center

        *  *hyb*: for most halos, returns the same as ssc,
                but works faster by starting iteration near potential minimum

        Before the main centring routine is called, the snapshot is translated so that the
        halo is already near the origin. The box is then wrapped so that halos on the edge
        of the box are handled correctly.
        '''
        if self.__check_paticles():
            print('No particles loaded in this Halo')
            return
        if mode == 'pot':
            #   if 'phi' not in self.keys():
            #      phi=self['phi']
            i = self["phi"].argmin()
            return self["pos"][i].copy()
        if mode == 'com':
            return self.mean_by_mass('pos')
        if mode == 'ssc':
            from pynbody.analysis.halo import shrink_sphere_center

            return shrink_sphere_center(self)
        if mode == 'hyb':
            #    if 'phi' not in self.keys():
            #       phi=self['phi']
            from pynbody.analysis.halo import hybrid_center

            return hybrid_center(self)
        print('No such mode')

        return

    def ang_mom_vec(self, alignwith: str = 'all', rmax=None, **kwargs):

        filtbyr = self._sele_family(alignwith, rmax=rmax, **kwargs)
        angmom = ang_mom(filtbyr)
        return angmom
    
    def to_cen(self, mode='ssc',cen=None,vel=None):
        self.check_boundary()
        if cen is None:
            pos_cen = self.center(mode=mode)
        else:
            pos_cen = cen
        if vel is None:
            vel_cen = self.vel_center(pos=pos_cen)
        else:
            vel_cen = vel
        self.shift(pos=pos_cen, vel=vel_cen)

    def face_on(self, **kwargs)-> List:
        """
        Transforms the halo's coordinate system to a 'face-on' view.

        This method aligns the halo such that the selected component's angular momentum
        is aligned with the z-axis. It optionally shifts the halo to the center of the coordinate system.

        Parameters:
        -----------
        mode : str, optional
            Determines how to center the halo. Default is 'ssc'. Other options might include 'virial' or 'custom'.
        alignwith : str, optional
            Specifies which component to use for alignment. Options include:
            - 'all' or 'total': Uses the combined angular momentum of all components.
            - 'dm', 'darkmatter': Uses the angular momentum of dark matter.
            - 'star', 's': Uses the angular momentum of stars.
            - 'gas', 'g': Uses the angular momentum of gas.
            - 'baryon', 'baryonic': Uses the combined angular momentum of stars and gas.
        shift : bool, optional
            If True, shifts the halo to its center of mass and adjusts the coordinate system. Default is True.
        retmatrix, retpos, retvel: bool, default True
            return rotation matrix, center pos and vel.
        """
    
        mode = kwargs.get('mode', 'ssc')
        shift = kwargs.get('shift', True)
        alignwith = kwargs.get('alignwith', 'all')
        alignmode = kwargs.get('alignmode', 'jz')
        
        retmatrix = kwargs.get('retmatrix',True)
        retpos = kwargs.get('retpos',True)
        retvel = kwargs.get('retvel',True)

        self.check_boundary()
        pos_center = self.center(mode=mode)
        vel_center = self.vel_center(mode=mode)

        if alignmode == 'jz':
            self.shift(pos=pos_center, vel=vel_center)
            angmom = self.ang_mom_vec( **kwargs)
            trans = calc_faceon_matrix(angmom)
        elif alignmode == 'krot':
            self.shift(pos=pos_center, vel=vel_center)
            resul = self.krot(calmode = 'max', calfor = alignwith,**kwargs)
            if resul:
                trans = resul['krotmat']
            else:
                self.shift(pos=-pos_center, vel=-vel_center)
                return
        elif alignmode == 'moi':
            self.shift(pos=pos_center, vel=vel_center)
            trans = self.moi_shape(calfor = alignwith, calpa = 'mass', nbins=1)['rotations']
        else:
            print('No such alignmode')
            return
        
        if shift:
            self._transform(trans)
        else:
            self.shift(pos=-pos_center, vel=-vel_center)
            self._transform(trans)
        ret = []
        if retpos:
            ret.append(pos_center)
        if retvel:
            ret.append(vel_center)
        if retmatrix:
            ret.append(trans)
        return ret

    def R_vir(self, overden: float = 178, cen=None, rho_def='critical') -> SimArray:
        """
        the virial radius of the halo.

        Parameters:
        -----------
        overden : float, default is 178
            The overdensity criterion.
        cen : array-like, default is the cen derived from self.center(mode='ssc')
            The center position to use.
        """
        from pynbody.analysis.halo import virial_radius
        
        if isinstance(cen, type(None)):
            cen = self.center(mode='ssc')
        try:
            R = virial_radius(self, cen=cen, overden=overden, rho_def=rho_def, r_max=self['r'].max())
        except:
            print(r'It is so weird. Rvir > r_max, use 5xr_max to calculate it')
            R = virial_radius(self, cen=cen, overden=overden, rho_def=rho_def, r_max=5*self['r'].max())
        return R

    def moi_shape(self, calfor: str = 'all', calpa: str = 'mass', **kwargs):
        '''
        Returns
        -------
        rbin : SimArray
            The radial bins used for the fitting

        axis_lengths : SimArray
            A nbins x ndim array containing the axis lengths of the ellipsoids in each shell

        num_particles : np.ndarray
            The number of particles within each bin

        rotation_matrices : np.ndarray
            The rotation matrices for each shell
        '''
        filtbyr = self._sele_family(calfor, **kwargs)
        return MoI_shape(filtbyr, calpa = calpa, **kwargs)
    
    def krot(self, rmax: float = None, calfor: str = 'star', **kwargs) -> np.ndarray:

        filtbyr = self._sele_family(calfor, rmax=rmax, **kwargs)

        calmode = kwargs.get('calmode', 'now')

        if calmode == 'now':
            return np.array(
                np.sum((0.5 * filtbyr['mass'] * (filtbyr['vcxy'] ** 2)))
                / np.sum(filtbyr['mass'] * filtbyr['ke'])
            )
        if calmode == 'max':
            fitmethod = kwargs.get('fitmethod', 'BFGS')
            result = fit_krotmax(
                filtbyr['pos'].view(np.ndarray),
                filtbyr['vel'].view(np.ndarray),
                filtbyr['mass'].view(np.ndarray),
                method=fitmethod,
            )
            return result
        print('No such calmode')
        return

    def sfh(self, **kwargs) -> dict:
        nbins = kwargs.get('nbins', 200)
        massmode = kwargs.get('massmode', 'now')
        if massmode == 'now':
            weight = self.s['mass']
        elif massmode == 'birth':
            weight = self.s['GFM_InitialMass']
        else:
            print('No such massmode')
            return
        mass_h, evo_t = np.histogram(
            self.s['tform'],
            bins=np.linspace(
                self.s['tform'].min().in_units('Gyr'), self.t.in_units('Gyr'), nbins
            ),
            weights=weight,
        )
        mass_h = SimArray(mass_h, weight.units)
        evo_t = SimArray(evo_t, units.Gyr)
        t_inter = np.diff(evo_t)

        SFR = (mass_h / (t_inter)).in_units('Msol yr**-1')
        mass_cumsum = mass_h.cumsum()

        result = {
            't': evo_t[1:],
            'sfr': SFR,
            'mass': mass_cumsum,
        }
        return result
    '''
    def profile(self, ndim: int = 2, type: str = 'lin', nbins: int = 100, rmin: float = 0.1, rmax: float = 100, **kwargs):
        return
        #return Profile_1D(self, ndim, type, nbins, rmin, rmax, **kwargs)
    '''
    def star_t(self, tmax: float, **kwargs):
        if tmax > self.t.in_units('Gyr'):
            print('tmax should be less than', self.t.in_units('Gyr'))
            return
        tmin = kwargs.get('tmin', 0)
        if tmin > tmax:
            print('tmin should be smaller than tmax. 0 is recommended')
            return
        massmode = kwargs.get('massmode', 'now')
        
        if massmode == 'now':
            return self.s['mass'][
            (self.s['tform'].in_units('Gyr') < tmax)
            & (self.s['tform'].in_units('Gyr') > tmin)
        ].sum()
        elif massmode == 'birth':
            return self.s['GFM_InitialMass'][
            (self.s['tform'].in_units('Gyr') < tmax)
            & (self.s['tform'].in_units('Gyr') > tmin)
        ].sum()
        else:
            print('No such massmode')
            return

    def t_star(self, frac: float = 0.5, **kwargs):
        if (frac > 1) or (frac <= 0):
            print('frac should range from 0-1')
            return
        massmode = kwargs.get('massmode', 'now')
        tform_sort = self.s['tform'][self.s['tform'].argsort()].in_units('Gyr')
        if massmode == 'now':
            mass_sort = self.s['mass'][self.s['tform'].argsort()]

        elif massmode == 'birth':
            mass_sort = self.s['GFM_InitialMass'][self.s['tform'].argsort()]
        else:
            print('No such massmode')
            return
        masscrit = frac * mass_sort[tform_sort < self.t.in_units('Gyr')].sum()
        mass_cumsum = mass_sort.cumsum()
        return (
            tform_sort[mass_cumsum > masscrit].min()
            + tform_sort[mass_cumsum < masscrit].max()
        ) / 2

    def R(self, frac: float = 0.5, calfor: str = 'star', calpa: str = 'mass', **kwargs) -> SimArray:
        '''projected radius that contain specific fraction of target something'''
        return self.__call_r('rxy', frac, calfor, calpa, **kwargs)

    def r(self, frac: float = 0.5, calfor: str = 'star', calpa: str = 'mass', **kwargs) -> SimArray:
        '''spherical radius that contain specific fraction of target something'''
        return self.__call_r('r', frac, calfor, calpa, **kwargs)
    
    def rho(self, rmax: float, calfor: str = 'star', calpa: str = 'mass', **kwargs) -> SimArray:
        '''Volume density'''
        rmin = kwargs.get('rmin', None)
        filtbyr = self._sele_family(calfor, rmax=rmax, rmin=rmin)
        pasum = np.array(filtbyr[calpa].sum())
        pavolume = 4/3*np.pi*(rmax**3 - rmin**3) if rmin else 4/3*np.pi*rmax**3
        
        return SimArray(pasum/pavolume, filtbyr[calpa].units/filtbyr['r'].units**3)
    
    def Sigma(self, Rmax: float, calfor: str = 'star', calpa: str = 'mass', **kwargs) -> SimArray:
        '''Surface density'''
        Rmin = kwargs.get('Rmin', None)
        zmax = kwargs.get('zmax', None)
        filtbyr = self._sele_family(calfor, Rmax=Rmax, Rmin=Rmin, zmax=zmax)
        pasum = np.array(filtbyr[calpa].sum())
        paarea = np.pi*(Rmax**2 - Rmin**2) if Rmin else np.pi*Rmax**2
        
        return SimArray(pasum/paarea, filtbyr[calpa].units/filtbyr['rxy'].units**2)
    
    def sum(self, calfor: str= 'star', calpa: str = 'mass', **kwargs) -> SimArray:
        '''sum of something'''
        return np.sum(self._sele_family(calfor, **kwargs)[calpa])
    
    def check_boundary(self) -> bool:
        """
        Check if any particle lay on the edge of the box.
        """
        boxsize = self.properties['boxsize'].in_units(self['pos'].units, **self.conversion_context())
        if (len(self) != len(self.ancestor)) or (hasattr(self.ancestor, '_canloadPT')):
            return self.ancestor.check_boundary()
        if (self['x'].max() - self['x'].min()) > (boxsize / 2):
            print('On the edge of the box, move to center')
            self.wrap()
            return True
        if (self['y'].max() - self['y'].min()) > (boxsize / 2):
            print('On the edge of the box, move to center')
            self.wrap()
            return True
        if (self['z'].max() - self['z'].min()) > (boxsize / 2):
            print('On the edge of the box, move to center')
            self.wrap()
            return True
        return False

    def shift(self, pos: SimArray = None, vel: SimArray = None, phi: SimArray = None):
        '''
        shift to the specific position
        then set its pos, vel, phi, acc to 0.
        '''
        if (len(self) != len(self.ancestor)) or (hasattr(self.ancestor, '_canloadPT')):
            self.ancestor.shift(pos, vel, phi)
        else:
            if pos is not None:
                self['pos'] -= pos
            if vel is not None:
                self['vel'] -= vel
            if (phi is not None) and ('phi' in self):
                self['phi'] -= phi

    def _sele_family(self, family, **kwargs):
        rmax = kwargs.get('rmax', None)
        rmin = kwargs.get('rmin', None)
        Rmax = kwargs.get('Rmax', None)
        Rmin = kwargs.get('Rmin', None)
        zmax = kwargs.get('zmax', None)
        sele = kwargs.get('sele', None)
        
        if set(['star', 's']) & set([family.lower()]):
            selfam = self.s
        elif set(['gas', 'g']) & set([family.lower()]):
            selfam = self.g
        elif set(['dm', 'darkmatter']) & set([family.lower()]):
            selfam = self.dm
        elif set(['total', 'all']) & set([family.lower()]):
            selfam = self
        elif set(['baryon']) & set([family.lower()]):
            slice1 = self._get_family_slice(get_family('s'))
            slice2 = self._get_family_slice(get_family('g'))
            selfam = self[
                np.append(
                    np.arange(len(self))[slice1], np.arange(len(self))[slice2]
                ).astype(np.int64)
            ]
        else:
            print('calfor wrong !!!')
            return
        if sele is not None:
            selfam = selfam[sele]
        if rmax:
            selfam = selfam[filt.LowPass('r', rmax)]
        if rmin:
            selfam = selfam[filt.HighPass('r', rmin)]
        if Rmax:
            selfam = selfam[filt.LowPass('rxy', Rmax)]
        if Rmin:
            selfam = selfam[filt.HighPass('rxy', Rmin)]
        if zmax:
            selfam = selfam[filt.BandPass('z', -zmax, zmax)]

        return selfam

    @property
    def _filename(self):
        if self._descriptor in self.base._filename:
            return self.base._filename
        else:
            return self.base._filename + ":" + self._descriptor
        
    @property
    def Re(self):
        return self.R()

    @property
    def re(self):
        return self.r()

    def wrap(self, boxsize=None, convention='center'):
        if (len(self) != len(self.ancestor)) or (hasattr(self.ancestor, '_canloadPT')):
            self.ancestor.wrap(boxsize, convention)
        else:
            super().wrap(boxsize, convention)

    def rotate_x(self, angle):
        if (len(self) != len(self.ancestor)) or (hasattr(self.ancestor, '_canloadPT')):
            return self.ancestor.rotate_x(angle)
        else:
            return super().rotate_x(angle)

    def rotate_y(self, angle):
        if (len(self) != len(self.ancestor)) or (hasattr(self.ancestor, '_canloadPT')):
            return self.ancestor.rotate_y(angle)
        else:
            return super().rotate_y(angle)

    def rotate_z(self, angle):
        if (len(self) != len(self.ancestor)) or (hasattr(self.ancestor, '_canloadPT')):
            return self.ancestor.rotate_z(angle)
        else:
            return super().rotate_z(angle)

    def transform(self, matrix):
        if (len(self) != len(self.ancestor)) or (hasattr(self.ancestor, '_canloadPT')):
            try:
                return self.ancestor._transform(matrix)
            except:
                return self.ancestor.transform(matrix)
        else:
            try:
                return super()._transform(matrix)
            except:
                return super().transform(matrix)
            
    def __call_r(
        self, callkeys: str = 'r', frac: float = 0.5, calfor: str = 'star', calpa: str ='mass', **kwargs
    ) -> SimArray:
        '''
        Sort particles by callkeys, and then cumsum calpa, 
        return callkeys where the cumsum of calpa is equal to frac * the sum of calpa
        '''
        calfam = self._sele_family(calfor, **kwargs)

        call_pa = calfam[calpa]
        
        
        callr = calfam[callkeys]
        args = np.argsort(callr)
        r_sort = callr[args]
        pa_sort = call_pa[args]
        pa_cumsum = pa_sort.cumsum()
        if hasattr(frac,'__iter__'):
            callpasum = call_pa.sum()
            pacrit = [i * callpasum for i in frac]
            Rcall = SimArray([(
                r_sort[pa_cumsum > i].min() + r_sort[pa_cumsum < i].max()
            ) / 2 for i in pacrit])
            Rcall.units = r_sort.units
            Rcall.sim = self
        else:
            pacrit = frac * call_pa.sum()
            Rcall = (
                r_sort[pa_cumsum > pacrit].min() + r_sort[pa_cumsum < pacrit].max()
            ) / 2

        return Rcall

    def __getitem__(self,i):
        try:
            return super().__getitem__(i)
        except:
            pass
        try:
            return self.properties[i]
        except:
            pass
        raise TypeError
    
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except:
            pass

        try:
            return self.properties[name]
        except:
            pass

        if name in self.GC:
            return self.GC[name]

        raise AttributeError(
            "%r object has no attribute %r" % (type(self).__name__, name)
        )

    def __check_paticles(self):
        if len(self) > 0:
            return False
        else:
            return True

    def _transform(self, matrix):
        if (len(self) != len(self.ancestor)) or (hasattr(self.ancestor, '_canloadPT')):
            try:
                self.ancestor._transform(matrix)
            except:
                self.ancestor.transform(matrix)
        else:
            try:
                super()._transform(matrix)
            except:
                super().transform(matrix)



# some deride_property

# all
@derived_array
def jy(sim):
    """y-component of the angular momentum"""
    return sim['j'][:,1]

# all
@derived_array
def jx(sim):
    """x-component of the angular momentum"""
    return sim['j'][:,0]

# all
@derived_array
def jc(sim):
    '''the maximum angular momentum'''
    return sim['j2']**(1,2)

# all
@derived_array
def circularity(sim):
    '''the circularity parameter'''
    return sim['jz']/sim['jc']

# all
@derived_array
def be(sim):
    '''binding energy normalized by the minimum value '''
    return sim['phi']/sim['phi'].abs().max()


# all
@derived_array
def phi(sim):
    """
    Calculate the gravitational potential for all particles
        https://github.com/mikegrudic/pytreegrav
    """
    if 'phi' not in sim:
        print('There is no phi in the keyword')
        if ('mass' in sim) and ('pos' in sim):
            print('Calculating gravity and it will take tens of seconds')
            if len(sim.ancestor['mass']) > 1000:
                print('Calculate by using Octree')
            else:
                print('Calculate by using brute force')
            try:
                eps = sim.ancestor.properties.get('eps', 0)
            except:
                eps = 0
            if eps == 0:
                print('Calculate the gravity without softening length')
            pot = Potential(
                sim.ancestor['pos'].view(np.ndarray),
                sim.ancestor['mass'].view(np.ndarray),
                np.repeat(eps, len(sim.ancestor['mass'])).view(np.ndarray),
                parallel = gravity_parallel(),
            )
            phi = SimArray(
                pot, units.G * sim.ancestor['mass'].units / sim.ancestor['pos'].units
            )
            sim.ancestor['phi'] = phi
            sim.ancestor['phi'].convert_units('km**2 s**-2')
            return sim['phi']
        else:
            print(
                '\'phi\' fails to be calculated. The keys \'mass\' and \'pos\' are required '
            )
            return
    return sim['phi']


# all
@derived_array
def acc(sim):
    """
    Calculate the acceleration for all particles.
        https://github.com/mikegrudic/pytreegrav
    """
    if 'acc' not in sim:
        print('There is no acc in the keyword')
        if ('mass' in sim) and ('pos' in sim):
            if len(sim.ancestor['mass']) > 1000:
                print('Calculate by using Octree')
            else:
                print('Calculate by using brute force')
            try:
                eps = sim.ancestor.properties.get('eps', 0)
            except:
                eps = 0
            if eps == 0:
                print('Calculate the gravity without softening length')
            accelr = Accel(
                sim.ancestor['pos'].view(np.ndarray),
                sim.ancestor['mass'].view(np.ndarray),
                np.repeat(eps, len(sim.ancestor['mass'])).view(np.ndarray),
                parallel = gravity_parallel(),
            )
            acc = SimArray(
                accelr,
                units.G
                * sim.ancestor['mass'].units
                / sim.ancestor['pos'].units
                / sim.ancestor['pos'].units,
            )
            sim.ancestor['acc'] = acc
            sim.ancestor['acc'].convert_units('km s^-1 Gyr^-1')
        else:
            print(
                '\'acc\' fails to be calculated. The keys \'mass\' and \'pos\' are required '
            )
            return
    return sim['acc']


acc.__stable__ = True
phi.__stable__ = True


# star
@derived_array
def tform(
    sim,
):
    """
    Calculates the stellar formation time based on the 'aform' array.

    Notes:
    ------
    The function uses the 'aform' array to compute the formation time, which is then converted to Gyr.
    The calculation requires cosmological parameters like `omegaM0` and `h` from the simulation properties.
    """
    try:
        sim['aform']
    except (KeyError, OSError):
        print('need aform to cal: GFM_StellarFormationTime')
    import numpy as np

    omega_m = sim.properties['omegaM0']
    a = sim['aform'].view(np.ndarray).copy()
    a[a < 0] = 0
    omega_fac = np.sqrt((1 - omega_m) / omega_m) * a ** (3 / 2)
    H0_kmsMpc = 100.0 * sim.ancestor.properties['h']
    t = SimArray(
        2.0 * np.arcsinh(omega_fac) / (H0_kmsMpc * 3 * np.sqrt(1 - omega_m)),
        units.Mpc / units.km * units.s,
    )
    t.convert_units('Gyr')
    t[t == 0] = 14.0
    return t

# star
@derived_array
def age(sim):
    """
    Calculates the age of stars based on their formation time.

    Notes:
    ------
    The age is computed as the difference between the current simulation time (`t`) and the stellar formation time (`tform`).
    Particles with a negative age are considered wind particles.
    """
    ag = sim.properties['t'] - sim['tform']
    ag.convert_units('Gyr')
    return ag

# star
@derived_array
def U_mag(sim):
    """
    Vega magnitudes
    see https://www.tng-project.org/data/docs/specifications/#parttype4 for details
    In detail, these are:
    Buser's X filter, where X=U,B3,V (Vega magnitudes),
    then IR K filter + Palomar 200 IR detectors + atmosphere.57 (Vega),
    then SDSS Camera X Response Function, airmass = 1.3 (June 2001), where X=g,r,i,z (AB magnitudes).
    They can be found in the filters.log file in the BC03 package.
    The details on the four SDSS filters can be found in Stoughton et al. 2002, section 3.2.1.
    """

    try:
        sim['GFM_StellarPhotometrics']
    except (KeyError, OSError):
        print("Need 'GFM_StellarPhotometrics' of star ")

    return sim['GFM_StellarPhotometrics'][:, 0]

@derived_array
def U_lum(sim):
    """3571  unit Lsun"""
    return 10.0 ** (-0.4*(sim.s['U_mag']-5.55))

# star
@derived_array
def B_mag(sim):
    """Vega magnitudes"""
    try:
        sim['GFM_StellarPhotometrics']
    except (KeyError, OSError):
        print("Need 'GFM_StellarPhotometrics' of star ")

    return sim['GFM_StellarPhotometrics'][:, 1]

@derived_array
def B_lum(sim):
    """4344 unit Lsun"""
    return 10.0 ** (-0.4*(sim.s['B_mag']-5.45))

# star
@derived_array
def V_mag(sim):
    """ Vega magnitudes """
    try:
        sim['GFM_StellarPhotometrics']
    except (KeyError, OSError):
        print("Need 'GFM_StellarPhotometrics' of star ")

    return sim['GFM_StellarPhotometrics'][:, 2]

@derived_array
def V_lum(sim):
    """5456 unit Lsun"""
    return 10.0 ** (-0.4*(sim.s['V_mag']-4.78))

# star
@derived_array
def K_mag(sim):
    """ Vega magnitudes"""
    try:
        sim['GFM_StellarPhotometrics']
    except (KeyError, OSError):
        print("Need 'GFM_StellarPhotometrics' of star ")

    return sim['GFM_StellarPhotometrics'][:, 3]

@derived_array
def K_lum(sim):
    """21603 unit Lsun"""
    return 10.0 ** (-0.4*(sim.s['K_mag']-3.29))

# star
@derived_array
def g_mag(sim):
    """AB magnitudes """
    try:
        sim['GFM_StellarPhotometrics']
    except (KeyError, OSError):
        print("Need 'GFM_StellarPhotometrics' of star ")

    return sim['GFM_StellarPhotometrics'][:, 4]

@derived_array
def g_lum(sim):
    """4670 unit Lsun"""
    return 10.0 ** (-0.4*(sim.s['g_mag']-5.12))

# star
@derived_array
def r_mag(sim):
    """AB magnitudes """
    try:
        sim['GFM_StellarPhotometrics']
    except (KeyError, OSError):
        print("Need 'GFM_StellarPhotometrics' of star ")

    return sim['GFM_StellarPhotometrics'][:, 5]

@derived_array
def r_lum(sim):
    """6156 unit Lsun"""
    return 10.0 ** (-0.4*(sim.s['r_mag']-4.64))

# star
@derived_array
def i_mag(sim):
    """AB magnitudes """
    try:
        sim['GFM_StellarPhotometrics']
    except (KeyError, OSError):
        print("Need 'GFM_StellarPhotometrics' of star ")

    return sim['GFM_StellarPhotometrics'][:, 6]

@derived_array
def i_lum(sim):
    """7472 unit Lsun"""
    return 10.0 ** (-0.4*(sim.s['i_mag']-4.53))

# star
@derived_array
def z_mag(sim):
    """AB magnitudes """
    try:
        sim['GFM_StellarPhotometrics']
    except (KeyError, OSError):
        print("Need 'GFM_StellarPhotometrics' of star ")

    return sim['GFM_StellarPhotometrics'][:, 7]

@derived_array
def z_lum(sim):
    """8917 unit Lsun"""
    return 10.0 ** (-0.4*(sim.s['z_mag']-4.51))

@derived_array
def metals(sim):
    """ """
    try:
        sim['GFM_Metals']
    except (KeyError, OSError):
        print("Need 'GFM_Metals'")

    return sim['GFM_Metals']

# Refer mostly https://pynbody.readthedocs.io/latest/
# gas
@derived_array
def temp(sim):
    """
    Calculates the gas temperature based on the internal energy.

    Notes:
    ------
    This function uses the two-phase ISM sub-grid model to calculate the gas temperature.
    The formula used is based on the internal energy and gas properties.
    For more information, refer to Sec.6 of the TNG FAQ:
    https://www.tng-project.org/data/docs/faq/
    """
    try:
        sim['u']
    except (KeyError, OSError):
        print('need gas InternalEnergy to cal: InternalEnergy')
    gamma = 5.0 / 3
    UnitEtoUnitM = ((units.kpc / units.Gyr).in_units('km s^-1')) ** 2
    T = (gamma - 1) / units.k * sim['mu'] * sim['u'] * UnitEtoUnitM

    T.convert_units('K')
    return T


# gas
@derived_array
def ne(sim):
    """
    Calculates the electron number density from the electron abundance and hydrogen number density.

    Notes:
    ------
    This function computes the electron number density using the electron abundance and the hydrogen number density.
    It assumes that `ElectronAbundance` and `nH` are available in the simulation object.

    Formula:
    --------
    n_e = ElectronAbundance * n_H
    where:
    - ElectronAbundance is the fraction of electrons per hydrogen atom.
    - n_H is the hydrogen number density in cm^-3.
    """
    n = sim['ElectronAbundance'] * sim['nH'].in_units('cm^-3')
    n.units = units.cm**-3
    return n


# gas
@derived_array
def em(sim):
    """
    Calculates the Emission Measure (n_e^2) per particle, which is used to be integrated along the line of sight (LoS).

    Formula:
    --------
    EM = n_e^2
    where:
    - n_e is the electron number density in cm^-3.
    """
    return (sim['ne'] * sim['ne']).in_units('cm^-6')


# gas
@derived_array
def p(sim):
    """
    Calculates the pressure in the gas.

    Notes:
    ------
    The pressure is calculated using the formula:
    P = (2 / 3) * u * rho
    where:
    - u is the internal energy per unit mass.  e.g., InternalEnergy in TNG
    - rho is the gas density in units of solar masses per cubic kiloparsec (Msol kpc^-3). e.g.,  Density in TNG
    """
    p = sim["u"] * sim["rho"].in_units('Msol kpc^-3') * (2.0 / 3)
    p.convert_units("Pa")
    return p


@derived_array
def cs(sim):
    """
    Calculates the sound speed in the gas.

    Notes:
    ------
    The sound speed is calculated using the formula:
    c_s = sqrt( (5/3) * (k_B * T) / μ )
    where:
    - k_B is the Boltzmann constant.
    - T is the gas temperature.
    - μ is the mean molecular weight.
    """
    return (np.sqrt(5.0 / 3.0 * units.k * sim['temp'] / sim['mu'])).in_units('km s^-1')


@derived_array
def c_s(self):
    """
    Calculates the sound speed of the gas based on pressure and density.

    ------
    The sound speed is calculated using the formula:
    c_s = sqrt( (5/3) * (p / rho) )
    where:
    - p is the gas pressure.
    - rho is the gas density.
    """
    # x = np.sqrt(5./3.*units.k*self['temp']*self['mu'])
    x = np.sqrt(5.0 / 3.0 * self['p'] / self['rho'].in_units('Msol kpc^-3'))
    x.convert_units('km s^-1')
    return x


# gas
@derived_array
def c_n_sq(sim):
    """
    Calculates the turbulent amplitude C_N^2 for use in spectral calculations,
    As in Eqn 20 of Macquart & Koay 2013 (ApJ 776 2).

    ------
    This calculation assumes a Kolmogorov spectrum of turbulence below the SPH resolution.

    The formula used is:
    C_N^2 = ((beta - 3) / (2 * (2 * π)^(4 - beta))) * L_min^(3 - beta) * EM

    Where:
    - beta = 11/3
    - L_min = 0.1 Mpc (minimum scale of turbulence)
    - EM = emission measure
    """

    ## Spectrum of turbulence below the SPH resolution, assume Kolmogorov
    beta = 11.0 / 3.0
    L_min = 0.1 * units.Mpc
    c_n_sq = (
        ((beta - 3.0) / ((2.0) * (2.0 * np.pi) ** (4.0 - beta)))
        * L_min ** (3.0 - beta)
        * sim["em"]
    )
    c_n_sq.units = units.m ** (-20, 3)

    return c_n_sq

# gas
@derived_array
def Halpha(sim):
    """
    Compute the H-alpha intensity for each gas particle based on the emission measure.

    References:
    - Draine, B. T. (2011). "Physics of the Interstellar and Intergalactic Medium".
    - For more details on the H-alpha intensity and its calculation, see:
      https://pynbody.readthedocs.io/latest/_modules/pynbody/snapshot/gadgethdf.html
    - Additional information can be found at:
      http://astro.berkeley.edu/~ay216/08/NOTES/Lecture08-08.pdf

    """
    # Define the H-alpha coefficient based on Planck's constant and the speed of light
    coeff = (
        (6.6260755e-27) * (299792458.0 / 656.281e-9) / (4.0 * np.pi)
    )  ## units : erg sr^-1

    # Compute the recombination coefficient for H-alpha
    alpha = coeff * 7.864e-14 * (1e4 / sim['temp'].in_units('K'))

    # Set units for the alpha coefficient
    alpha.units = (
        units.erg * units.cm ** (3) * units.s ** (-1) * units.sr ** (-1)
    )  ## intensity in erg cm^3 s^-1 sr^-1

    # Calculate and return the H-alpha intensity
    return (alpha * sim["em"]).in_units(
        'erg cm^-3 s^-1 sr^-1'
    )  # Flux erg cm^-3 s^-1 sr^-1

# gas
@derived_array
def nH(sim):
    """
    Calculate the total hydrogen number density for each gas particle.

    The hydrogen number density is computed using the following formula:
    - Total Hydrogen Number Density: X_H * (rho / m_p)
      where X_H is the hydrogen mass fraction, rho is the gas density, and m_p is the proton mass.
    """
    nh = sim['XH'] * (sim['rho'].in_units('g cm^-3') / units.m_p).in_units('cm^-3')
    nh.units = units.cm**-3
    return nh

# gas
@derived_array
def XH(sim):
    """
    Calculate the hydrogen mass fraction for each gas particle.

    If the 'GFM_Metals' data is available in the simulation, the hydrogen mass fraction is extracted
    from this data. If 'GFM_Metals' is not present, a default value of 0.76 is used.
    """
    try:
        Xh = sim['GFM_Metals'].view(np.ndarray).T[0]
        return SimArray(Xh)
    except (KeyError, OSError):
        print('No GFM_Metals, use hydrogen mass fraction XH=0.76')
        return SimArray(0.76 * np.ones(len(sim)))


# gas
@derived_array
def mu(sim):
    """
    Calculate the mean molecular weight of the gas.

    The mean molecular weight is computed using the hydrogen mass fraction (XH) and the electron
    abundance. The formula used is:
        μ = 4 / (1 + 3 * XH + 4 * XH * ElectronAbundance)
    """
    try:
        sim['ElectronAbundance']
    except (KeyError, OSError):
        print('need gas ElectronAbundance to cal: ElectronAbundance')
    muu = SimArray(
        4
        / (1 + 3 * sim['XH'] + 4 * sim['XH'] * sim['ElectronAbundance']).astype(
            np.float64
        ),
        units.m_p,
    )
    return muu.in_units('m_p')


# gas
def _hi_h2_masses(sim):
    """
    Per-particle atomic (HI) and molecular (H2) hydrogen masses, in ``Msol``.

    Follows the same prescription as Martini's ``TNGSource`` (see
    ``martini/sources/tng_source.py``), i.e. Marinacci et al. 2017, Diemer et al.
    2018 and Leroy et al. 2008:

    - the neutral hydrogen fraction ``fneutral`` is taken from the TNG
      ``NeutralHydrogenAbundance`` field (``nH0/nH``) and is overridden for
      star-forming cells by the effective-temperature two-phase ISM expression
      of Springel & Hernquist 2003 (the tabulated abundance there is based on the
      effective, not physical, temperature).  For mini snapshots, which lack
      ``NeutralHydrogenAbundance``, the same two-phase expression is used for all
      gas as an approximation.
    - the atomic fraction ``fatomic`` of the *neutral* hydrogen follows the
      pressure-based partition of Leroy et al. 2008, ``fatomic = 1 / (1 +
      (P/1.7e4 K cm^-3)^0.8)``, where ``P`` is the partial thermal pressure of the
      neutral gas (Marinacci et al. 2017 / Diemer et al. 2018 eq. 6).

    Then, with gas mass ``m`` and hydrogen mass fraction ``X_H``::

        mHI = m * X_H * fneutral * fatomic
        mH2 = m * X_H * fneutral * (1 - fatomic)
    """
    gamma = 5.0 / 3.0
    k_B = units.k.ratio('erg K^-1')
    m_p = units.m_p.ratio('g')

    XH = sim['XH'].view(np.ndarray).astype(np.float64)
    xe = sim['ElectronAbundance'].view(np.ndarray)
    u = sim['u'].in_units('cm^2 s^-2').view(np.ndarray)  # specific energy, erg g^-1
    rho = sim['rho'].in_units('g cm^-3').view(np.ndarray)  # mass density, g cm^-3
    m_cgs = sim['mass'].in_units('g').view(np.ndarray)  # gas mass, g

    # mean molecular weight in proton masses, from the TNG FAQ
    mu = 4.0 / (1.0 + 3.0 * XH + 4.0 * XH * xe)
    # hydrogen number density, cm^-3
    nH = rho * XH / m_p

    # effective-temperature two-phase ISM (Springel & Hernquist 2003; Stevens 19):
    # cold, fully neutral
    mu_c = 4.0 / (1.0 + 3.0 * XH) * m_p
    u_c = k_B * 1e3 / (mu_c * (gamma - 1.0))  # erg g^-1, T_c = 1e3 K
    # hot, He fully ionised
    mu_h = 4.0 / (3.0 + 5.0 * XH) * m_p
    T_h = 1e3 + 5.73e7 / (1.0 + 573.0 * np.maximum(1.0, nH / 0.13) ** -0.8)  # K
    u_h = k_B * T_h / (mu_h * (gamma - 1.0))  # erg g^-1
    fneutral_coldhot = np.clip((u_h - u) / (u_h - u_c), 0.0, 1.0)

    # neutral hydrogen fraction
    try:
        fneutral = sim['NeutralHydrogenAbundance'].view(np.ndarray).copy()
        try:
            sfr = sim['sfr'].view(np.ndarray)
        except (KeyError, OSError):
            sfr = np.zeros_like(XH)
        fneutral[sfr > 0] = fneutral_coldhot[sfr > 0]
    except (KeyError, OSError):
        from warnings import warn

        warn(
            "NeutralHydrogenAbundance not available for mini snapshots,"
            " approximating the neutral fraction with the effective-temperature"
            " two-phase ISM (Springel & Hernquist 2003) - to avoid this use a"
            " full snapshot instead.",
            UserWarning,
        )
        fneutral = fneutral_coldhot

    # partial thermal pressure of the neutral gas, K cm^-3
    P = (gamma - 1.0) * u * fneutral * rho / k_B
    # atomic fraction of the neutral hydrogen (Leroy et al. 2008)
    fatomic = 1.0 / (1.0 + (P / 1.7e4) ** 0.8)

    Msun = units.Msol.ratio('g')
    mHI = SimArray(m_cgs / Msun * XH * fneutral * fatomic, units.Msol)
    mHI.sim = sim
    mH2 = SimArray(m_cgs / Msun * XH * fneutral * (1.0 - fatomic), units.Msol)
    mH2.sim = sim
    return mHI, mH2


# gas
@derived_array
def mHI(sim):
    """Atomic hydrogen (HI) mass per gas particle, in ``Msol``.

    Computed as m * X_H * fneutral * fatomic following :class:`martini.sources.tng_source.TNGSource`.
    """
    return _hi_h2_masses(sim)[0]


# gas
@derived_array
def mH2(sim):
    """Molecular hydrogen (H2) mass per gas particle, in ``Msol``.

    Computed as m * X_H * fneutral * (1 - fatomic) following :class:`martini.sources.tng_source.TNGSource`.
    """
    return _hi_h2_masses(sim)[1]


###############################################################################
# Variant HI/H2 prescriptions
# -----------------------------------------------------------------------------
# The default ``mHI``/``mH2`` above reproduce Martini's ``TNGSource`` (Leroy et
# al. 2008 pressure partition + Martini neutral fraction).  The derived arrays
# below add the alternative recipes implemented by ``galcalc.py`` (Dirty-AstroPy;
# arhstevens, Stevens et al. 2019) and by ``HI_mass_simulation.py``, each with a
# ``_suffix`` to distinguish it from the default:
#
#   ``_BR06``  Blitz & Rosolowsky (2006), parameterised by Leroy et al. (2008).
#              The atomic/molecular split of the *neutral* hydrogen follows the
#              pressure law ``f_atomic = 1/(1 + (P/P0)**alpha)`` with
#              ``P0 = 1.7e4 K cm^-3`` and ``alpha = 0.8``; ``P`` is the partial
#              thermal pressure of the neutral gas.  The neutral fraction for
#              star-forming cells uses the effective-temperature two-phase ISM
#              (Springel & Hernquist 2003) as coded in
#              ``galcalc.neutralFraction_SFcells``.  Identical to the default
#              except for the tiny difference in the mean-molecular weight used
#              for the cold ISM phase.
#   ``_SH03``  As ``_BR06``, but the star-forming-cell neutral fraction uses the
#              Springel & Hernquist (2003) two-phase model with the SH03
#              feedback parameters (SN heating temperature 1e8 K and ``A0 =
#              1e3``).  The critical density ``n_H,th`` is held at the SH03
#              canonical value 0.13 cm^-3: the Katz et al. (1996) cooling
#              function needed to solve it self-consistently is not available, so
#              this variant differs from ``_BR06`` in the SH03 feedback
#              constants rather than in a solved ``n_H,th`` (documented
#              approximation).  Because the effective-temperature two-phase
#              fraction saturates to ~1 for the cool ISM, ``_SH03`` and ``_BR06``
#              agree closely; the recipes differ mainly in the molecular
#              partition (``_GK11``/``_KMT13`` vs ``_BR06``).
#   ``_GK11``  Gnedin & Kravtsov (2011), eq. 10 (their ``method=2``).  An
#              iterative molecular-fraction fit that depends on the interstellar
#              radiation field and metallicity.
#   ``_KMT13`` Krumholz et al. (2013), eq. 10 (their ``method=4``).  An iterative
#              fit that additionally depends on the local (dark-matter + star)
#              density ``rho_sd``.
#
# Approximations
# --------------
# This is *not* the full Dirty-AstroPy ``galcalc`` pipeline.  Two inputs that
# require external data are replaced as follows:
#
#   * The ultraviolet background radiation field, which enters ``_GK11`` and
#     ``_KMT13`` through the dimensionless ISMF ``G0``, is approximated by a
#     constant floor (``ISRF = 1``, in units of the Milky Way Draine 1978 field).
#     The FG09/HM12 redshift tables are not available in this environment.
#   * For ``_KMT13`` the local ``rho_sd`` is approximated by the local dark-matter
#     density field ``SubfindDMDensity`` when present (converted to M_sun pc^-3),
#     otherwise by the gas density as a documented proxy for the total local
#     density.
#
# ``galcalc.u2temp``/``temp2u`` (which operate on specific energy in J kg^-1)
# are reimplemented here in cgs (erg g^-1) so they plug straight into the
# pressure/neutral-fraction equations used throughout this module.
###############################################################################

_P0_BR06 = 1.7e4  # K cm^-3, pressure at which H2/HI = 1 (Blitz & Rosolowsky 2006; Leroy et al. 2008)
_ALPHA_BR06 = 0.8  # exponent of the BR06 pressure law
_ISRF_FLOOR = 1.0  # UV background floor in MW-field units (approx. FG09, z ~ 0)
_RHO_CGS_TO_MSUN_PC3 = units.pc.ratio('cm') ** 3 / units.Msol.ratio('g')  # g cm^-3  ->  M_sun pc^-3

# galcalc internal units (M_sun, pc, yr), used by the _GK11/_KMT13 iterative fits.
_KG_PER_MSUN = 1.989e30
_M_PER_PC = 3.0857e16
_S_PER_YR = 60 * 60 * 24 * 365.24


def _galcalc_constants():
    """Physical constants in galcalc's internal units (M_sun, pc, yr)."""
    m_p = 1.6726219e-27 / _KG_PER_MSUN  # proton mass, M_sun
    G = 6.67408e-11 * _KG_PER_MSUN * _S_PER_YR ** 2 / _M_PER_PC ** 3
    k_B = 1.38064852e-23 * _S_PER_YR ** 2 / _M_PER_PC ** 2 / _KG_PER_MSUN
    const_ratio = k_B / (m_p * G)
    return m_p, G, k_B, const_ratio, _M_PER_PC


def _temp2u(temp, mu, gamma=5.0 / 3.0):
    """Temperature (K) -> specific energy (erg g^-1), for mean molecular weight ``mu``."""
    k_B = units.k.ratio('erg K^-1')
    m_p = units.m_p.ratio('g')
    return temp * k_B / (mu * m_p * (gamma - 1.0))


def _u2temp(u, mu, gamma=5.0 / 3.0):
    """Specific energy (erg g^-1) -> temperature (K), for mean molecular weight ``mu``."""
    k_B = units.k.ratio('erg K^-1')
    m_p = units.m_p.ratio('g')
    return u * mu * m_p * (gamma - 1.0) / k_B


def _two_phase_neutral(u, nH, f_H, T_SN, A0, n_H_th, gamma=5.0 / 3.0, T_cold=1.0e3):
    """Neutral fraction from the effective-temperature two-phase ISM.

    ``u`` is the specific internal energy (erg g^-1), ``nH`` the hydrogen number
    density (cm^-3), ``f_H`` the hydrogen mass fraction.  This implements the
    ``galcalc.neutralFraction_SFcells`` / ``neutralFraction_SFcells_SH03`` formula.
    The cold phase is fully neutral (mu_c = 4/(1 + 3 f_H)) and the hot phase has
    helium fully ionised (mu_h = 4/(8 - 5(1 - f_H))).
    """
    mu_c = 4.0 / (1.0 + 3.0 * f_H)
    mu_h = 4.0 / (8.0 - 5.0 * (1.0 - f_H))
    u_cold = _temp2u(T_cold, mu_c, gamma)
    u_SN = _temp2u(T_SN, mu_h, gamma)
    A = A0 * (nH / n_H_th) ** (-0.8)
    u_hot = u_SN / (1.0 + A) + u_cold
    return np.clip((u_hot - u) / (u_hot - u_cold), 0.0, 1.0)


def _gas_neutral_fraction(sim, sf_kind, gamma=5.0 / 3.0):
    """Neutral hydrogen mass fraction ``fneutral`` per gas cell.

    Non-star-forming cells use ``NeutralHydrogenAbundance`` when available,
    otherwise fall back to the effective-temperature two-phase ISM.  Star-forming
    cells use a two-phase fraction chosen by ``sf_kind``:
      * ``'SFcells'`` - ``galcalc.neutralFraction_SFcells`` (T_SN = 5.73e7, A0=573, n_H,th = 0.13)
      * ``'SH03'``    - ``galcalc.neutralFraction_SFcells_SH03`` feedback constants
                        (T_SN = 1e8, A0=1e3; n_H,th held at the SH03 value 0.13 cm^-3)
    """
    XH = sim['XH'].view(np.ndarray).astype(np.float64)
    u = sim['u'].in_units('cm^2 s^-2').view(np.ndarray)
    rho = sim['rho'].in_units('g cm^-3').view(np.ndarray)
    nH = rho * XH / units.m_p.ratio('g')

    if sf_kind == 'SH03':
        # SH03 two-phase constants (T_SN = 1e8 K, A0 = 1e3); n_H,th is kept at the
        # canonical SH03 value 0.13 cm^-3 because the Katz+96 cooling-function
        # table required to solve it self-consistently is not available (approx.).
        n_H_th = 0.13
        T_SN, A0 = 1.0e8, 1.0e3
    else:
        n_H_th = 0.13
        T_SN, A0 = 5.73e7, 573.0
    fneutral_two = _two_phase_neutral(u, nH, XH, T_SN, A0, n_H_th, gamma)

    try:
        fneutral = sim['NeutralHydrogenAbundance'].view(np.ndarray).copy()
        try:
            sfr = sim['sfr'].view(np.ndarray)
        except (KeyError, OSError):
            sfr = np.zeros_like(XH)
        fneutral[sfr > 0] = fneutral_two[sfr > 0]
    except (KeyError, OSError):
        from warnings import warn

        warn(
            f"NeutralHydrogenAbundance not available for mini snapshots,"
            f" approximating the neutral fraction (sf_kind={sf_kind!r}) for all gas.",
            UserWarning,
        )
        fneutral = fneutral_two
    return fneutral


def _metallicity(sim):
    """Total metal mass fraction ``Z`` (excluding H and He) per gas cell."""
    try:
        metals = sim['GFM_Metals'].view(np.ndarray)
        Z = metals[:, 2:].sum(axis=1).astype(np.float64)
    except (KeyError, OSError):
        from warnings import warn

        warn("GFM_Metals not available; assuming solar metallicity Z = 0.0127.", UserWarning)
        Z = np.full(len(sim['mass']), 0.0127, dtype=np.float64)
    Z[Z < 1e-5] = 1e-5  # floor (BBN); mirrors galcalc
    return Z


def _br06_partition(fneutral, P_K_cm3, P0=_P0_BR06, alpha=_ALPHA_BR06):
    """Atomic fraction of the neutral hydrogen, Blitz & Rosolowsky (2006) / Leroy et al. (2008)."""
    R_mol = (fneutral * P_K_cm3 / P0) ** alpha
    return 1.0 / (1.0 + R_mol)


def _molecular_fraction_gk11(mass, sfr, Z, X, rho, temp, fneutral,
                             sigma_sfr0=1e-9, f_esc=0.15, isrf_floor=_ISRF_FLOOR,
                             it_max=300, rtol=5e-3):
    """H2/(HI+H2) from Gnedin & Kravtsov (2011) eq. 10 (galcalc ``method=2``).

    Inputs use galcalc's internal units: ``mass`` M_sun, ``rho`` M_sun pc^-3,
    ``temp`` K, ``sfr`` M_sun yr^-1; ``X`` is the hydrogen mass fraction (used
    in place of galcalc's metallicity-to-X fitting function).
    """
    m_p, G, k_B, const_ratio, m_per_pc = _galcalc_constants()
    m_per_pc_cm = m_per_pc * 100.0
    denom = m_p * m_per_pc_cm ** 3  # M_sun per cm^3 of volume, for n_H
    f_th = 1.0
    Y = 1.0 - X - Z
    n_H = X * rho / denom

    gamma = 5.0 / 3.0
    mu = (X + 4.0 * Y) / ((2.0 - fneutral) * (X + Y))
    fzero = fneutral <= 0
    fneutral = np.where(fzero, 1e-6, fneutral)

    D_MW = Z / 0.0127
    f_H2_old = np.zeros_like(fneutral)

    for it in range(it_max):
        f_mol = X * fneutral * f_H2_old / (X + Y)
        gamma = (5.0 / 3.0) * (1.0 - f_mol) + 1.4 * f_mol
        mu = (X + 4.0 * Y) * (1.0 + (1.0 - fneutral) / fneutral) / (
            (X + Y) * (1.0 + 2.0 * (1.0 - fneutral) / fneutral - f_H2_old / 2.0)
        )
        Sigma = np.sqrt(gamma * const_ratio * f_th * rho * temp / mu)  # M_sun pc^-2
        Sigma_n = fneutral * X * Sigma
        area = mass / Sigma
        Sigma_SFR = sfr / area
        G0 = np.maximum(isrf_floor, f_esc * Sigma_SFR / sigma_sfr0)
        D_star = 1.5e-3 * np.log(1.0 + (3.0 * G0) ** 1.7)
        alpha = 2.5 * G0 / (1.0 + (0.5 * G0) ** 2.0)
        s = 0.04 / (D_star + D_MW)
        g = (1.0 + alpha * s + s * s) / (1.0 + s)
        Lambda = np.log(1.0 + g * D_MW ** (3.0 / 7.0) * (G0 / 15.0) ** (4.0 / 7.0))
        Sigma_c = 20.0 * Lambda ** (4.0 / 7.0) / (D_MW * np.sqrt(1.0 + G0 * D_MW ** 2.0))
        f_H2 = (1.0 + Sigma_c / Sigma_n) ** (-2.0)
        if np.allclose(f_H2[~fzero], f_H2_old[~fzero], rtol=rtol):
            break
        f_H2_old = f_H2.copy()

    f_H2 = np.where(fzero, 0.0, f_H2)
    return f_H2


def _molecular_fraction_kmt13(mass, sfr, Z, X, rho, temp, fneutral, rho_sd,
                              sigma_sfr0=1e-9, f_esc=0.15, isrf_floor=_ISRF_FLOOR,
                              it_max=300, rtol=5e-3):
    """H2/(HI+H2) from Krumholz et al. (2013) eq. 10 (galcalc ``method=4``).

    Inputs use galcalc's internal units; ``rho_sd`` is the local (DM + star)
    density in M_sun pc^-3.
    """
    m_p, G, k_B, const_ratio, m_per_pc = _galcalc_constants()
    m_per_pc_cm = m_per_pc * 100.0
    denom = m_p * m_per_pc_cm ** 3
    f_th = 1.0
    Y = 1.0 - X - Z
    n_H = X * rho / denom

    gamma = 5.0 / 3.0
    mu = (X + 4.0 * Y) / ((2.0 - fneutral) * (X + Y))
    fzero = fneutral <= 0
    fneutral = np.where(fzero, 1e-6, fneutral)

    D_MW = Z / 0.0127
    f_H2_old = np.zeros_like(fneutral)

    f_c = 5.0  # clumping factor
    alpha = 5.0  # turbulence/magnetic-to-thermal pressure
    zeta_d = 0.33
    f_w = 0.5
    c_w = 8e3 / m_per_pc * _S_PER_YR  # warm-medium sound speed, internal units
    T_CNMmax = 243.0  # K, max CNM temperature

    for it in range(it_max):
        f_mol = X * fneutral * f_H2_old / (X + Y)
        gamma = (5.0 / 3.0) * (1.0 - f_mol) + 1.4 * f_mol
        mu = (X + 4.0 * Y) * (1.0 + (1.0 - fneutral) / fneutral) / (
            (X + Y) * (1.0 + 2.0 * (1.0 - fneutral) / fneutral - f_H2_old / 2.0)
        )
        Sigma = np.sqrt(gamma * const_ratio * f_th * rho * temp / mu)
        Sigma_n = fneutral * X * Sigma
        G0 = np.maximum(isrf_floor, f_esc * sfr / mass * Sigma / sigma_sfr0)

        n_CNM2p = 23.0 * G0 * 4.1 / (1.0 + 3.1 * D_MW ** 0.365)
        R_H2 = f_H2_old / np.maximum(1.0 - f_H2_old, 1e-12)
        Sigma_HI = np.maximum((1.0 - f_H2_old) * Sigma_n, 1e-12)
        frac = 32.0 * zeta_d * alpha * f_w * c_w * c_w * rho_sd / (np.pi * G * Sigma_HI ** 2.0)
        P_th = np.pi * G * Sigma_HI ** 2.0 / (4.0 * alpha) * (
            1.0 + 2.0 * R_H2 + np.sqrt((1.0 + 2.0 * R_H2) ** 2.0 + frac)
        )
        n_CNMhydro = P_th / (1.1 * k_B * T_CNMmax) / m_per_pc_cm ** 3.0
        n_CNM = np.maximum(n_CNM2p, n_CNMhydro)
        chi = 7.2 * G0 / (0.1 * n_CNM)
        tau_c = 0.066 * f_c * D_MW * Sigma_n
        s = np.log(1.0 + 0.6 * chi + 0.01 * chi * chi) / (0.6 * tau_c)
        f_H2 = np.zeros_like(fneutral)
        mask = s < 2.0
        f_H2[mask] = 1.0 - 0.75 * s[mask] / (1.0 + 0.25 * s[mask])
        if np.allclose(f_H2[~fzero], f_H2_old[~fzero], rtol=rtol):
            break
        f_H2_old = f_H2.copy()

    f_H2 = np.where(fzero, 0.0, f_H2)
    return f_H2


def _rhosd(sim):
    """Local (dark-matter + star) density ``rho_sd`` in M_sun pc^-3.

    Prefers the ``SubfindDMDensity`` local dark-matter field; falls back to the
    gas density (documented proxy) when it is not loaded, since no cheap per-cell
    dark-matter+star density is available in a gas-only sub-snapshot.
    """
    try:
        return sim['SubfindDMDensity'].in_units('Msol pc^-3').view(np.ndarray)
    except (KeyError, OSError):
        from warnings import warn

        warn(
            "SubfindDMDensity not available; approximating rho_sd with the gas density.",
            UserWarning,
        )
        return sim['rho'].in_units('g cm^-3').view(np.ndarray) * _RHO_CGS_TO_MSUN_PC3


def _variant_fields(sim):
    """Shared physical arrays (cgs) needed by all the HI/H2 variants."""
    XH = sim['XH'].view(np.ndarray).astype(np.float64)
    u = sim['u'].in_units('cm^2 s^-2').view(np.ndarray)  # erg g^-1
    rho = sim['rho'].in_units('g cm^-3').view(np.ndarray)  # g cm^-3
    mascgs = sim['mass'].in_units('g').view(np.ndarray)  # g
    masmsun = sim['mass'].in_units('Msol').view(np.ndarray)  # M_sun
    try:
        sfr = sim['sfr'].in_units('Msol yr^-1').view(np.ndarray)
    except (KeyError, OSError):
        sfr = np.zeros_like(XH)
    return XH, u, rho, mascgs, masmsun, sfr


def _variant_msun_pc3(sim):
    return sim['rho'].in_units('g cm^-3').view(np.ndarray) * _RHO_CGS_TO_MSUN_PC3


def _variant_temp(sim, u_cgs, gamma=5.0 / 3.0):
    """Temperature (K) from specific energy, using mu = the fitting default of 1.0."""
    return _u2temp(u_cgs, 1.0, gamma)


def _br06_hih2(sim):
    XH, u, rho, mascgs, masmsun, sfr = _variant_fields(sim)
    k_B = units.k.ratio('erg K^-1')
    gamma = 5.0 / 3.0
    fneutral = _gas_neutral_fraction(sim, 'SFcells')
    P_cgs = (gamma - 1.0) * rho * u  # erg cm^-3 (total thermal)
    P_K_cm3 = P_cgs / k_B  # K cm^-3
    fatomic = _br06_partition(fneutral, P_K_cm3)
    Msun = units.Msol.ratio('g')
    mHI = SimArray(mascgs / Msun * XH * fneutral * fatomic, units.Msol)
    mHI.sim = sim
    mH2 = SimArray(mascgs / Msun * XH * fneutral * (1.0 - fatomic), units.Msol)
    mH2.sim = sim
    return mHI, mH2


def _sh03_hih2(sim):
    XH, u, rho, mascgs, masmsun, sfr = _variant_fields(sim)
    k_B = units.k.ratio('erg K^-1')
    gamma = 5.0 / 3.0
    fneutral = _gas_neutral_fraction(sim, 'SH03')
    P_cgs = (gamma - 1.0) * rho * u
    P_K_cm3 = P_cgs / k_B
    fatomic = _br06_partition(fneutral, P_K_cm3)
    Msun = units.Msol.ratio('g')
    mHI = SimArray(mascgs / Msun * XH * fneutral * fatomic, units.Msol)
    mHI.sim = sim
    mH2 = SimArray(mascgs / Msun * XH * fneutral * (1.0 - fatomic), units.Msol)
    mH2.sim = sim
    return mHI, mH2


def _gk11_hih2(sim):
    XH, u, rho, mascgs, masmsun, sfr = _variant_fields(sim)
    gamma = 5.0 / 3.0
    fneutral = _gas_neutral_fraction(sim, 'SFcells')
    Z = _metallicity(sim)
    rho_pc3 = _variant_msun_pc3(sim)
    temp = _variant_temp(sim, u)
    f_H2 = _molecular_fraction_gk11(masmsun, sfr, Z, XH, rho_pc3, temp, fneutral)
    mHI = SimArray(masmsun * XH * fneutral * (1.0 - f_H2), units.Msol)
    mHI.sim = sim
    mH2 = SimArray(masmsun * XH * fneutral * f_H2, units.Msol)
    mH2.sim = sim
    return mHI, mH2


def _kmt13_hih2(sim):
    XH, u, rho, mascgs, masmsun, sfr = _variant_fields(sim)
    gamma = 5.0 / 3.0
    fneutral = _gas_neutral_fraction(sim, 'SFcells')
    Z = _metallicity(sim)
    rho_pc3 = _variant_msun_pc3(sim)
    temp = _variant_temp(sim, u)
    rho_sd = _rhosd(sim)
    f_H2 = _molecular_fraction_kmt13(masmsun, sfr, Z, XH, rho_pc3, temp, fneutral, rho_sd)
    mHI = SimArray(masmsun * XH * fneutral * (1.0 - f_H2), units.Msol)
    mHI.sim = sim
    mH2 = SimArray(masmsun * XH * fneutral * f_H2, units.Msol)
    mH2.sim = sim
    return mHI, mH2


@derived_array
def mHI_BR06(sim):
    """Atomic hydrogen (HI) mass, ``_BR06`` (Blitz & Rosolowsky 2006 / Leroy et al. 2008)."""
    return _br06_hih2(sim)[0]


@derived_array
def mH2_BR06(sim):
    """Molecular hydrogen (H2) mass, ``_BR06`` (Blitz & Rosolowsky 2006 / Leroy et al. 2008)."""
    return _br06_hih2(sim)[1]


@derived_array
def mHI_SH03(sim):
    """Atomic hydrogen (HI) mass, ``_SH03`` (Springel & Hernquist 2003 two-phase + BR06 partition)."""
    return _sh03_hih2(sim)[0]


@derived_array
def mH2_SH03(sim):
    """Molecular hydrogen (H2) mass, ``_SH03`` (Springel & Hernquist 2003 two-phase + BR06 partition)."""
    return _sh03_hih2(sim)[1]


@derived_array
def mHI_GK11(sim):
    """Atomic hydrogen (HI) mass, ``_GK11`` (Gnedin & Kravtsov 2011 eq. 10)."""
    return _gk11_hih2(sim)[0]


@derived_array
def mH2_GK11(sim):
    """Molecular hydrogen (H2) mass, ``_GK11`` (Gnedin & Kravtsov 2011 eq. 10)."""
    return _gk11_hih2(sim)[1]


@derived_array
def mHI_KMT13(sim):
    """Atomic hydrogen (HI) mass, ``_KMT13`` (Krumholz et al. 2013 eq. 10)."""
    return _kmt13_hih2(sim)[0]


@derived_array
def mH2_KMT13(sim):
    """Molecular hydrogen (H2) mass, ``_KMT13`` (Krumholz et al. 2013 eq. 10)."""
    return _kmt13_hih2(sim)[1]


@SimDict.setter
def read_Snap_properties(f, SnapshotHeader):
    """
    Set cosmological and simulation properties for a given snapshot.

    Parameters:
    -----------
    f : SimDict
        The simulation dictionary to be updated.
    SnapshotHeader : dict
        A dictionary containing header information for the snapshot, including cosmological parameters
        and box size.

    Cosmological Model (TNG runs):
    -------------------------------
    - Standard ΛCDM model based on Planck 2015 results:
      - omegaL0 (Dark Energy density parameter): 0.6911
      - omegaM0 (Matter density parameter): 0.3089
      - omegaB0 (Baryon density parameter): 0.0486
      - sigma8 (Amplitude of matter density fluctuations): 0.8159
      - ns (Spectral index of primordial fluctuations): 0.9667
      - h (Hubble parameter): 0.6774

    Cosmological Model (Illustris runs):
    -------------------------------
    - Standard ΛCDM model based on Planck 2013 results:
      - omegaL0 (Dark Energy density parameter): 0.7274
      - omegaM0 (Matter density parameter): 0.2726
      - omegaB0 (Baryon density parameter): 0.0456
      - sigma8 (Amplitude of matter density fluctuations): 0.809
      - ns (Spectral index of primordial fluctuations): 0.963
      - h (Hubble parameter): 0.704
    # from https://arxiv.org/abs/1405.1418
    """

    f['a'] = SnapshotHeader['Time']                 # Scale factor (time)
    f['z'] = (1 / SnapshotHeader['Time']) - 1       # Redshift
    if "TNG" in f['run']:
        f['h'] = SnapshotHeader['HubbleParam']          # Hubble parameter.
        f['omegaM0'] = SnapshotHeader['Omega0']         # Matter density parameter.
        f['omegaL0'] = SnapshotHeader['OmegaLambda']    # Dark energy density parameter.
        f['omegaB0'] = 0.0486                           # Baryon density parameter (fixed value).
        f['sigma8'] = 0.8159                            # Amplitude of matter density fluctuations (fixed value).
        f['ns'] = 0.9667                                # Spectral index (fixed value).
    elif "Illustris" in f['run']:
        f['h'] = 0.704
        f['omegaM0'] = 0.2726
        f['omegaL0'] = 0.7274
        f['omegaB0'] = 0.0456                           # Baryon density parameter (fixed value).
        f['sigma8'] = 0.809                          # Amplitude of matter density fluctuations (fixed value).
        f['ns'] = 0.963                               # Spectral index (fixed value).
    else:
        raise ValueError("Unknown run type in 'run' property")
    f['boxsize'] = SnapshotHeader['BoxSize'] * units.kpc * units.a / units.h                        # Size of the simulation box (in kpc)
    f['Halos_total'] = SnapshotHeader['Ngroups_Total']          # Total number of halos in the snapshot.
    f['Subhalos_total'] = SnapshotHeader['Nsubgroups_Total']    # Total number of subhalos in the snapshot.

@SimDict.setter
def filepath(f, BasePath):
    """
    Set the file path for the simulation data.

    Parameters:
    -----------
    f : SimDict
        The simulation dictionary to be updated.
    BasePath : str
        The base directory path where the simulation data files are located.
    """
    f['filedir'] = BasePath
    for i in illustrisTNGruns:
        if i in BasePath:
            f['run'] = i
            break


@SimDict.getter
def t(d):
    """
    Calculate the age of the snapshot

    This function uses cosmological parameters and redshift to compute the age of the snapshot.
    The formula is derived from Peebles (p. 317, eq. 13.2).
    """
    import math

    omega_m = d['omegaM0']
    redshift = d['z']
    H0_kmsMpc = 100.0 * d['h'] * units.km / units.s / units.Mpc

    return get_t(omega_m, redshift, H0_kmsMpc)


@SimDict.getter
def rho_crit(d):
    z = d['z']
    omM = d['omegaM0']
    omL = d['omegaL0']
    h0 = d['h']
    a = d['a']
    omK = 1.0 - omM - omL
    _a_dot = h0 * a * np.sqrt(omM * (a**-3) + omK * (a**-2) + omL)
    H_z = _a_dot / a
    H_z = units.Unit("100 km s^-1 Mpc^-1") * H_z

    rho_crit = (3 * H_z**2) / (8 * np.pi * units.G)
    return rho_crit


@SimDict.getter
def tLB(d):
    """
    Calculate the lookback time.
    """
    import math

    omega_m = d['omegaM0']
    redshift = 0.0
    H0_kmsMpc = 100.0 * d['h'] * units.km / units.s / units.Mpc

    tlb = get_t(omega_m, redshift, H0_kmsMpc) - d['t']
    return tlb


@SimDict.getter
def cosmology(d):
    cos = {}
    cos['h'] = d.get('h')
    cos['omegaM0'] = d.get('omegaM0')
    cos['omegaL0'] = d.get('omegaL0')
    cos['omegaB0'] = d.get('omegaB0')
    cos['sigma8'] = d.get('sigma8')
    cos['ns'] = d.get('ns')
    return cos


def get_t(omega_m, redshift, H0_kmsMpc):
    import math

    omega_fac = math.sqrt((1 - omega_m) / omega_m) * pow(1 + redshift, -3.0 / 2.0)
    AGE = 2.0 * math.asinh(omega_fac) / (H0_kmsMpc * 3 * math.sqrt(1 - omega_m))
    return AGE.in_units('Gyr') * units.Gyr
