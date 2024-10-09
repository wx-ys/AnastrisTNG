'''
Basehalo for subhalo and halo
Derived array for some particle types
'''

import types
from functools import reduce

import numpy as np
from pynbody import units, filt, derived_array
from pynbody.family import get_family
from pynbody.simdict import SimDict
from pynbody.array import SimArray
from pynbody.snapshot import SubSnap
from pynbody.analysis.angmom import calc_faceon_matrix
        
from AnastrisTNG.TNGunits import illustrisTNGruns, NotneedtransGCPa
from AnastrisTNG.pytreegrav import Potential, Accel
from AnastrisTNG.Anatools import ang_mom, fit_krotmax, MoI_shape


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
                        lambda x, y: x * y, [a**b for a, b in zip(dims, new_unit[:])]
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

    def vel_center(self, mode='ssc', pos=None, r_cal='1 kpc'):
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

        if pos == None:
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

    def face_on(self, **kwargs):
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
        """
    
        mode = kwargs.get('mode', 'ssc')
        shift = kwargs.get('shift', True)
        alignwith = kwargs.get('alignwith', 'all')
        alignmode = kwargs.get('alignmode', 'jz')
        retmatrix = kwargs.get('retmatrix',False)

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
            trans = self.moi_shape(calfor = alignwith, calpa = 'mass', nbins=1)[3]
        else:
            print('No such alignmode')
            return
        
        if shift:
            self._transform(trans)
        else:
            self.shift(pos=-pos_center, vel=-vel_center)
            self._transform(trans)
        if retmatrix:
            return trans
        else:
            return

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
        R = virial_radius(self, cen=cen, overden=overden, rho_def=rho_def)
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

        return self.__call_r('rxy', frac, calfor, calpa, **kwargs)

    def r(self, frac: float = 0.5, calfor: str = 'star', calpa: str = 'mass', **kwargs) -> SimArray:

        return self.__call_r('r', frac, calfor, calpa, **kwargs)
    
    def rho(self, rmax: float, calfor: str = 'star', calpa: str = 'mass', **kwargs) -> SimArray:
        
        rmin = kwargs.get('rmin', None)
        filtbyr = self._sele_family(calfor, rmax=rmax, rmin=rmin)
        pasum = np.array(filtbyr[calpa].sum())
        pavolume = 4/3*np.pi*(rmax**3 - rmin**3) if rmin else 4/3*np.pi*rmax**3
        
        return SimArray(pasum/pavolume, filtbyr[calpa].units/filtbyr['r'].units**3)
    
    def Sigma(self, Rmax: float, calfor: str = 'star', calpa: str = 'mass', **kwargs) -> SimArray:
        
        Rmin = kwargs.get('Rmin', None)
        zmax = kwargs.get('zmax', None)
        filtbyr = self._sele_family(calfor, Rmax=Rmax, Rmin=Rmin, zmax=zmax)
        pasum = np.array(filtbyr[calpa].sum())
        paarea = np.pi*(Rmax**2 - Rmin**2) if Rmin else np.pi*Rmax**2
        
        return SimArray(pasum/paarea, filtbyr[calpa].units/filtbyr['rxy'].units**2)
    
    def check_boundary(self):
        """
        Check if any particle lay on the edge of the box.
        """
        if (len(self) != len(self.ancestor)) or (hasattr(self.ancestor, '_canloadPT')):
            self.ancestor.check_boundary()
            return
        if (self['x'].max() - self['x'].min()) > (self.boxsize / 2):
            print('On the edge of the box, move to center')
            self.wrap()
            return
        if (self['y'].max() - self['y'].min()) > (self.boxsize / 2):
            print('On the edge of the box, move to center')
            self.wrap()
            return
        if (self['z'].max() - self['z'].min()) > (self.boxsize / 2):
            print('On the edge of the box, move to center')
            self.wrap()
            return
        return

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
            self.ancestor.rotate_x(angle)
        else:
            super().rotate_x(angle)

    def rotate_y(self, angle):
        if (len(self) != len(self.ancestor)) or (hasattr(self.ancestor, '_canloadPT')):
            self.ancestor.rotate_y(angle)
        else:
            super().rotate_y(angle)

    def rotate_z(self, angle):
        if (len(self) != len(self.ancestor)) or (hasattr(self.ancestor, '_canloadPT')):
            self.ancestor.rotate_z(angle)
        else:
            super().rotate_z(angle)

    def transform(self, matrix):
        if (len(self) != len(self.ancestor)) or (hasattr(self.ancestor, '_canloadPT')):
            self.ancestor._transform(matrix)
        else:
            super()._transform(matrix)
            
    def __call_r(
        self, callkeys: str = 'r', frac: float = 0.5, calfor: str = 'star', calpa: str ='mass', **kwargs
    ) -> SimArray:
        '''
        Sort particles by callkeys, and then cumsum calpa, 
        return callkeys where the cumsum of calpa is equal to frac * the sum of calpa
        '''
        calfam = self._sele_family(calfor, **kwargs)

        call_pa = calfam[calpa]
        pacrit = frac * call_pa.sum()
        callr = calfam[callkeys]
        args = np.argsort(callr)
        r_sort = callr[args]
        pa_sort = call_pa[args]
        pa_cumsum = pa_sort.cumsum()
        Rcall = (
            r_sort[pa_cumsum > pacrit].min() + r_sort[pa_cumsum < pacrit].max()
        ) / 2

        return Rcall

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
            self.ancestor._transform(matrix)
        else:
            super()._transform(matrix)



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
    if 'aform' not in sim:
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
    see https://www.tng-project.org/data/docs/specifications/#parttype4 for details
    In detail, these are:
    Buser's X filter, where X=U,B3,V (Vega magnitudes),
    then IR K filter + Palomar 200 IR detectors + atmosphere.57 (Vega),
    then SDSS Camera X Response Function, airmass = 1.3 (June 2001), where X=g,r,i,z (AB magnitudes).
    They can be found in the filters.log file in the BC03 package.
    The details on the four SDSS filters can be found in Stoughton et al. 2002, section 3.2.1.
    """

    if 'GFM_StellarPhotometrics' not in sim:
        print("Need 'GFM_StellarPhotometrics' of star ")

    return sim['GFM_StellarPhotometrics'][:, 0]

# star
@derived_array
def B_mag(sim):
    """ """
    if 'GFM_StellarPhotometrics' not in sim:
        print("Need 'GFM_StellarPhotometrics' of star ")

    return sim['GFM_StellarPhotometrics'][:, 1]

# star
@derived_array
def V_mag(sim):
    """ """
    if 'GFM_StellarPhotometrics' not in sim:
        print("Need 'GFM_StellarPhotometrics' of star ")

    return sim['GFM_StellarPhotometrics'][:, 2]

# star
@derived_array
def K_mag(sim):
    """ """
    if 'GFM_StellarPhotometrics' not in sim:
        print("Need 'GFM_StellarPhotometrics' of star ")

    return sim['GFM_StellarPhotometrics'][:, 3]

# star
@derived_array
def g_mag(sim):
    """ """
    if 'GFM_StellarPhotometrics' not in sim:
        print("Need 'GFM_StellarPhotometrics' of star ")

    return sim['GFM_StellarPhotometrics'][:, 4]

# star
@derived_array
def r_mag(sim):
    """ """
    if 'GFM_StellarPhotometrics' not in sim:
        print("Need 'GFM_StellarPhotometrics' of star ")

    return sim['GFM_StellarPhotometrics'][:, 5]

# star
@derived_array
def i_mag(sim):
    """ """
    if 'GFM_StellarPhotometrics' not in sim:
        print("Need 'GFM_StellarPhotometrics' of star ")

    return sim['GFM_StellarPhotometrics'][:, 6]

# star
@derived_array
def z_mag(sim):
    """ """
    if 'GFM_StellarPhotometrics' not in sim:
        print("Need 'GFM_StellarPhotometrics' of star ")

    return sim['GFM_StellarPhotometrics'][:, 7]

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
    if 'u' not in sim:
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
    - u is the internal energy per unit mass.
    - rho is the gas density in units of solar masses per cubic kiloparsec (Msol kpc^-3).
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
    if 'GFM_Metals' in sim:
        Xh = sim['GFM_Metals'].view(np.ndarray).T[0]
        return SimArray(Xh)
    else:
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
    if 'ElectronAbundance' not in sim:
        print('need gas ElectronAbundance to cal: ElectronAbundance')
    muu = SimArray(
        4
        / (1 + 3 * sim['XH'] + 4 * sim['XH'] * sim['ElectronAbundance']).astype(
            np.float64
        ),
        units.m_p,
    )
    return muu.in_units('m_p')


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
    """

    f['a'] = SnapshotHeader['Time']                 # Scale factor (time)
    f['z'] = (1 / SnapshotHeader['Time']) - 1       # Redshift
    f['h'] = SnapshotHeader['HubbleParam']          # Hubble parameter.
    f['omegaM0'] = SnapshotHeader['Omega0']         # Matter density parameter.
    f['omegaL0'] = SnapshotHeader['OmegaLambda']    # Dark energy density parameter.
    f['omegaB0'] = 0.0486                           # Baryon density parameter (fixed value).
    f['sigma8'] = 0.8159                            # Amplitude of matter density fluctuations (fixed value).
    f['ns'] = 0.9667                                # Spectral index (fixed value).
    f['boxsize'] = SimArray(                        # Size of the simulation box (in kpc)
        1.0, SnapshotHeader['BoxSize'] * units.kpc * units.a / units.h
    )
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
