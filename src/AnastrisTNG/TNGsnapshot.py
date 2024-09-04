'''
Analysis of radial distributions 
Simsnap operations, overwrite and merge
Derived array for some particle types
'''
import types

import numpy as np
from pynbody import simdict,units,family,derived_array
from pynbody.snapshot import new
from pynbody.array import SimArray
from pynbody.analysis.profile import Profile as _Profile

from AnastrisTNG.illustris_python.groupcat import loadHeader
from AnastrisTNG.TNGunits import illustrisTNGruns
from AnastrisTNG.pytreegrav import Accel, Potential,PotentialTarget,AccelTarget

def cal_potential(sim,targetpos):
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
        eps=sim.properties.get('eps',0)
    except:
        eps=0
    if eps==0:
        print('Calculate the gravity without softening length')
    pot=PotentialTarget(targetpos,sim['pos'].view(np.ndarray),sim['mass'].view(np.ndarray),
                np.repeat(eps,len(targetpos)).view(np.ndarray))
    phi=SimArray(pot,units.G*sim['mass'].units/sim['pos'].units)
    phi.sim=sim
    return phi


def cal_acceleration(sim,targetpos):
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
        eps=sim.properties.get('eps',0)
    except:
        eps=0
    if eps==0:
        print('Calculate the gravity without softening length')
    accelr=AccelTarget(targetpos,sim['pos'].view(np.ndarray),sim['mass'].view(np.ndarray),
                np.repeat(eps,len(targetpos)).view(np.ndarray))
    acc=SimArray(accelr,units.G*sim['mass'].units/sim['pos'].units/sim['pos'].units)
    acc.sim=sim
    return acc

class Profile_1D:
    def __init__(self,sim,ndim=2,type='lin',nbins=100,rmin=0.1,rmax=100.,**kwargs):
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
        self.__Pall=_Profile(sim,ndim=ndim,type=type,nbins=nbins,rmin=rmin,rmax=rmax,**kwargs)
        self.__Pstar=_Profile(sim.s,ndim=ndim,type=type,nbins=nbins,rmin=rmin,rmax=rmax,**kwargs)
        self.__Pgas=_Profile(sim.g,ndim=ndim,type=type,nbins=nbins,rmin=rmin,rmax=rmax,**kwargs)
        self.__Pdm=_Profile(sim.dm,ndim=ndim,type=type,nbins=nbins,rmin=rmin,rmax=rmax,**kwargs)

        
        self.__properties={}
        self.__properties['Qgas']=self.Qgas
        self.__properties['Qstar']=self.Qstar
        self.__properties['Q2ws']=self.Q2ws
        self.__properties['Q2thin']=self.Q2thin
        self.__properties['Q2thick']=self.Q2thick



        def v_circ(p,grav_sim=None):
            """Circular velocity, i.e. rotation curve. Calculated by computing the gravity
            in the midplane - can be expensive"""
            #print("Profile v_circ -- this routine assumes the disk is in the x-y plane")
            grav_sim = grav_sim or p.sim
            cal_2=np.sqrt(2)/2
            basearray=np.array([(1,0,0),(0,1,0),(-1,0,0),(0,-1,0),
                                (cal_2,cal_2,0),(-cal_2,cal_2,0),
                                (cal_2,-cal_2,0),(-cal_2,-cal_2,0)])
            R=p['rbins'].in_units('kpc').copy()
            POS=np.array([(0,0,0)])
            for j in R:
                binsr=basearray*j
                POS=np.concatenate((POS,binsr),axis=0)
            POS=SimArray(POS,R.units)
            ac=cal_acceleration(grav_sim,POS)
            ac.convert_units('kpc Gyr**-2')
            POS.convert_units('kpc')
            velall=np.diag(np.dot(ac-ac[0],-POS.T))
            if 'units' in dir(velall):
                velall.units=units.kpc**2/units.Gyr**2
            else:
                velall=SimArray(velall,units.kpc**2/units.Gyr**2)
            velTrue=np.zeros(len(R))
            for i in range(len(R)):
                velTrue[i]=np.mean(velall[i+1:8*(i+1)+1])
            velTrue[velTrue<0]=0
            velTrue=np.sqrt(velTrue)
            velTrue=SimArray(velTrue,units.kpc/units.Gyr)
            velTrue.convert_units('km s**-1')
            velTrue.sim=grav_sim.ancestor
            return velTrue
        def pot(p,grav_sim=None):
            grav_sim = grav_sim or p.sim
            cal_2=np.sqrt(2)/2
            basearray=np.array([(1,0,0),(0,1,0),(-1,0,0),(0,-1,0),
                                (cal_2,cal_2,0),(-cal_2,cal_2,0),
                                (cal_2,-cal_2,0),(-cal_2,-cal_2,0)])
            R=p['rbins'].in_units('kpc').copy()
            POS=np.array([(0,0,0)])
            for j in R:
                binsr=basearray*j
                POS=np.concatenate((POS,binsr),axis=0)
            POS=SimArray(POS,R.units)
            po=cal_potential(grav_sim,POS)
            po.conver_units('km**2 s**-2')
            poall=np.zeros(len(R))
            for i in range(len(R)):
                poall[i]=np.mean(po[i+1:8*(i+1)+1])
            
            poall=SimArray(poall,po.units)
            poall.sim=grav_sim.ancestor
            return poall

        def omega(p):
            """Circular frequency Omega = v_circ/radius (see Binney & Tremaine Sect. 3.2)"""
            prof = p['v_circ'] / p['rbins']
            prof.convert_units('km s**-1 kpc**-1')
            return prof
        self.__Pall._profile_registry[v_circ.__name__]=v_circ
        self.__Pall._profile_registry[omega.__name__]=omega
        self.__Pall._profile_registry[pot.__name__]=pot

        self.__Pstar._profile_registry[v_circ.__name__]=v_circ
        self.__Pstar._profile_registry[omega.__name__]=omega
        self.__Pstar._profile_registry[pot.__name__]=pot

        self.__Pgas._profile_registry[v_circ.__name__]=v_circ
        self.__Pgas._profile_registry[omega.__name__]=omega
        self.__Pgas._profile_registry[pot.__name__]=pot

        self.__Pdm._profile_registry[v_circ.__name__]=v_circ
        self.__Pdm._profile_registry[omega.__name__]=omega
        self.__Pdm._profile_registry[pot.__name__]=pot
    
    def __getitem__(self, key):
        
        if isinstance(key,str):
            ks=key.split('-')
            if len(ks)>1:
                if set(['star','s','Star']) & set(ks):
                    return self.__Pstar[ks[0]]
                if set(['gas','g','Gas']) & set(ks):
                    return self.__Pgas[ks[0]]
                if set(['dm','darkmatter','DM']) & set(ks):
                    return self.__Pdm[ks[0]]
                if set(['all','ALL']) & set(ks):
                    return self.__Pstar[ks[0]]
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
        return (self.__Pall['kappa']*self.__Pgas['vr_disp']/(np.pi * self.__Pgas['density'] * units.G)).in_units("")
    def Qstar(self):
        '''
        Toomre-Q parameter
        '''
        return (self.__Pall['kappa']*self.__Pstar['vr_disp']/(3.36 * self.__Pstar['density'] * units.G)).in_units("")
    def Q2ws(self):
        '''
        Toomre Q of two component. Wang & Silk (1994)
        '''
        Qs=(self.__Pall['kappa']*self.__Pstar['vr_disp']/(np.pi * self.__Pstar['density'] * units.G)).in_units("")
        Qg=(self.__Pall['kappa']*self.__Pgas['vr_disp']/(np.pi * self.__Pgas['density'] * units.G)).in_units("")
        return (Qs*Qg)/(Qs+Qg)
    def Q2thin(self):
        '''
        The effective Q of two component thin disk. Romeo & Wiegert (2011) eq. 6.
        '''
        w=(2*self.__Pstar['vr_disp']*self.__Pgas['vr_disp']/((self.__Pstar['vr_disp'])**2+self.__Pgas['vr_disp']**2)).in_units("")
        Qs=(self.__Pall['kappa']*self.__Pstar['vr_disp']/(np.pi * self.__Pstar['density'] * units.G)).in_units("")
        Qg=(self.__Pall['kappa']*self.__Pgas['vr_disp']/(np.pi * self.__Pgas['density'] * units.G)).in_units("")

        q=[Qs*Qg/(Qs+w*Qg)]
        return [Qs[i]*Qg[i]/(Qs[i]+w[i]*Qg[i]) if Qs[i]>Qg[i] else Qs[i]*Qg[i]/(w[i]*Qs[i]+Qg[i]) for i in range(len(w))] 
    def Q2thick(self):
        '''
        The effective Q of two component thick disk. Romeo & Wiegert (2011) eq. 9. 
        '''
        w=(2*self.__Pstar['vr_disp']*self.__Pgas['vr_disp']/((self.__Pstar['vr_disp'])**2+self.__Pgas['vr_disp']**2)).in_units("")
        Ts=0.8+0.7*(self.__Pstar['vz_disp']/self.__Pstar['vr_disp']).in_units("")
        Tg=0.8+0.7*(self.__Pgas['vz_disp']/self.__Pgas['vr_disp']).in_units("")
        Qs=(self.__Pall['kappa']*self.__Pstar['vr_disp']/(np.pi * self.__Pstar['density'] * units.G)).in_units("")
        Qg=(self.__Pall['kappa']*self.__Pgas['vr_disp']/(np.pi * self.__Pgas['density'] * units.G)).in_units("")
        Qs=Qs*Ts
        Qg=Qg*Tg
        return [Qs[i]*Qg[i]/(Qs[i]+w[i]*Qg[i]) if Qs[i]>Qg[i] else Qs[i]*Qg[i]/(w[i]*Qs[i]+Qg[i]) for i in range(len(w))] 




##merge two simsnap and cover

def simsnap_cover(f1,f2):
    """
    Overwrites the data in simsnap f1 with data from simsnap f2.

    Parameters:
    -----------
    f1 : simsnap
        The target simsnap to be overwritten.
    f2 : simsnap
        The source simsnap providing the data.
    """
    for i in f1:
        del f1[i]
    f1._num_particles=len(f2)
    if len(f2.dm)>0:
        f1._family_slice[family.get_family('dm')]= f2._family_slice[family.get_family('dm')]
        for i in f1.dm:
            del f1.dm[i]
    if len(f2.s)>0:
        f1._family_slice[family.get_family('star')]= f2._family_slice[family.get_family('star')]
        for i in f1.s:
            del f1.s[i]
    if len(f2.g)>0:
        f1._family_slice[family.get_family('gas')]= f2._family_slice[family.get_family('gas')]
        for i in f1.g:
            del f1.g[i]
    if len(f2.bh)>0:
        f1._family_slice[family.get_family('bh')]= f2._family_slice[family.get_family('bh')]
        for i in f1.bh:
            del f1.bh[i]
            
    f1._create_arrays(["pos", "vel"], 3)
    f1._create_arrays(["mass"], 1)
    f1._decorate()
    if len(f1.dm)>0:
        for i in f2.dm:
            f1.dm[i]=f2.dm[i]
    if len(f1.s)>0:
        for i in f2.s:
            f1.s[i]=f2.s[i]
    if len(f1.g)>0:
        for i in f2.g:
            f1.g[i]=f2.g[i]
    if len(f1.bh)>0:
        for i in f2.bh:
            f1.bh[i]=f2.bh[i]



def simsnap_merge(f1,f2):
    """
    Merges twosimsnap f1 and f2 into a new simsnap f3.

    Parameters:
    -----------
    f1 : simsnap
        The first simsnap.
    f2 : simsnap
        The second simsnap.
    Returns:
    --------
    f3 : simsnap
        The new simsnap containing merged data from f1 and f2.
    """
    f3=new(star=len(f1.s)+len(f2.s),
            gas=len(f1.g)+len(f2.g),
            dm=len(f1.dm)+len(f2.dm),
            bh=len(f1.bh)+len(f2.bh),
            order='dm,star,gas,bh')
    if len(f3.s)>0:
        if len(f1.s)==0:
            for i in f2.s:
                f3.s[i]=f2.s[i]
        elif len(f2.s)==0:
            for i in f1.s:
                f3.s[i]=f1.s[i]
        else:
            for i in f2.s:
                f3.s[i]=SimArray(np.append(f1.s[i],f2.s[i],axis=0),f2.s[i].units)

    if len(f3.dm)>0:
        if len(f1.dm)==0:
            for i in f2.dm:
                f3.dm[i]=f2.dm[i]
        elif len(f2.dm)==0:
            for i in f1.dm:
                f3.dm[i]=f1.dm[i]
        else:
            for i in f2.dm:
                f3.dm[i]=SimArray(np.append(f1.dm[i],f2.dm[i],axis=0),f2.dm[i].units)

    if len(f3.g)>0:
        if len(f1.g)==0:
            for i in f2.g:
                f3.g[i]=f2.g[i]
        elif len(f2.g)==0:
            for i in f1.g:
                f3.g[i]=f1.g[i]
        else:
            for i in f2.g:
                f3.g[i]=SimArray(np.append(f1.g[i],f2.g[i],axis=0),f2.g[i].units)

    if len(f3.bh)>0:
        if len(f1.bh)==0:
            for i in f2.bh:
                f3.bh[i]=f2.bh[i]
        elif len(f2.bh)==0:
            for i in f1.bh:
                f3.bh[i]=f1.bh[i]
        else:
            for i in f2.bh:
                f3.bh[i]=SimArray(np.append(f1.bh[i],f2.bh[i],axis=0),f2.bh[i].units)

    return f3


def get_parttype(particle_field):
    particle_typeload=''

    if ('dm' in particle_field) or ('darkmatter' in particle_field):
        if len(particle_typeload)>0:
            particle_typeload+=',dm'
        else:
            particle_typeload+='dm'

    if ('star' in particle_field) or ('stars' in particle_field) or ('stellar' in particle_field):
        if len(particle_typeload)>0: 
            particle_typeload+=',star'
        else:
            particle_typeload+='star'


    if ('gas' in particle_field) or ('g' in particle_field) or ('cells' in particle_field):
        if len(particle_typeload)>0:
            particle_typeload+=',gas'
        else:
            particle_typeload+='gas'

    if (('bh' in particle_field) or ('bhs' in particle_field) or 
        ('blackhole' in particle_field) or ('blackholes' in particle_field)):
        if len(particle_typeload)>0:
            particle_typeload+=',bh'
        else:
            particle_typeload+='bh'
    return particle_typeload





def get_Snapshot_property(BasePath : str,Snap : int) ->simdict.SimDict:
    """
    Retrieves properties of a specific snapshot.

    Parameters:
    -----------
    BasePath : str
        The base path to the directory containing snapshot files.
    Snap : int
        The identifier of the snapshot to be loaded.

    Returns:
    --------
    Snapshot : simdict.SimDict
        A SimDict object containing the properties of the specified snapshot.
    """
    SnapshotHeader=loadHeader(BasePath,Snap)
    Snapshot=simdict.SimDict()
    Snapshot['filepath']=BasePath
    Snapshot['read_Snap_properties']=SnapshotHeader
    Snapshot['Snapshot']=Snap
    return Snapshot



def get_eps_Mdm(Snapshot):
    """
    Retrieves the gravitational softenings for stars and dark matter (DM) based on the simulation run and redshift.

    Parameters:
    -----------
    Snapshot : object
        An object containing snapshot properties, including `z` (redshift) and `run` (simulation run).

    Returns:
    --------
    eps_star : SimArray
        The gravitational softening length for stars.
    eps_dm : SimArray
        The gravitational softening length for dark matter.
    
    Notes:
    ------
    'Gravitational softenings for stars and DM are in comoving kpc until z=1, 
    after which they are fixed to their z=1 values.' -- Dylan Nelson.
    Data is sourced from https://www.tng-project.org/data/docs/background/.
    """
    MatchRun={
        'TNG50-1':[0.39,3.1e5/1e10],
        'TNG50-2':[0.78,2.5e6/1e10],
        'TNG50-3':[1.56,2e7/1e10],
        'TNG50-4':[3.12,1.6e8/1e10],

        'TNG100-1':[1,5.1e6/1e10],
        'TNG100-2':[2,4e7/1e10],
        'TNG100-3':[4,3.2e8/1e10],

        'TNG300-1':[2,4e7/1e10],
        'TNG300-1':[4,3.2e8/1e10],
        'TNG300-1':[8,2.5e9/1e10],
    }

    if Snapshot.z>1:
        return SimArray(MatchRun[Snapshot.run][0],units.a*units.kpc/units.h),SimArray(MatchRun[Snapshot.run][1],1e10*units.Msol/units.h)
    else:
        return SimArray(MatchRun[Snapshot.run][0]/2,units.kpc/units.h),SimArray(MatchRun[Snapshot.run][1],1e10*units.Msol/units.h)
    

#some deride_property

# all
@derived_array
def phi(sim) :
    """
    Calculate the gravitational potential for all particles

    Notes:
    ------
    This function checks if 'phi' is present in the simulation object. If not, it verifies if 'mass' and 'pos' are available
    and then calculates the potential using the `_PT_potential` method. If 'mass' and 'pos' are not available, it prints an error message.
    """
    if 'phi' not in sim:
        print('There is no phi in the keyword')
        if ('mass' in sim) and ('pos' in sim):
            sim.ancestor._PT_potential()
        else:
            print('\'phi\' fails to be calculated. The keys \'mass\' and \'pos\' are required ')
            return
    return sim['phi']


# all
@derived_array
def acc(sim) :
    """
    Calculate the acceleration for all particles.

    Notes:
    ------
    This function checks if 'acc' is present in the simulation object. If not, it verifies if 'mass' and 'pos' are available
    and then calculates the acceleration using the `_PT_acceleration` method. If 'mass' and 'pos' are not available, it prints an error message.
    """
    if 'acc' not in sim:
        print('There is no acc in the keyword')
        if ('mass' in sim) and ('pos' in sim):
            sim.ancestor._PT_acceleration()
        else:
            print('\'acc\' fails to be calculated. The keys \'mass\' and \'pos\' are required ')
            return
    return sim['acc']




# star
@derived_array
def tform(sim,):
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
    a=sim['aform'].view(np.ndarray)
    a[a<0]=0
    omega_fac = np.sqrt( (1-omega_m)/omega_m ) * a**(3/2)
    H0_kmsMpc = 100.0 * sim.ancestor.properties['h']
    t =SimArray(2.0 * np.arcsinh(omega_fac) / (H0_kmsMpc * 3 * np.sqrt(1-omega_m)),units.Mpc/units.km*units.s)
    t.convert_units('Gyr')
    t[t==0]=14.
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
    return sim.properties['t']-sim['tform']

#star
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
    
    return sim['GFM_StellarPhotometrics'][:,0]

#star
@derived_array
def B_mag(sim):
    """
    """
    if 'GFM_StellarPhotometrics' not in sim:
        print("Need 'GFM_StellarPhotometrics' of star ")
    
    return sim['GFM_StellarPhotometrics'][:,1]

#star
@derived_array
def V_mag(sim):
    """
    """
    if 'GFM_StellarPhotometrics' not in sim:
        print("Need 'GFM_StellarPhotometrics' of star ")
    
    return sim['GFM_StellarPhotometrics'][:,2]

#star
@derived_array
def K_mag(sim):
    """
    """
    if 'GFM_StellarPhotometrics' not in sim:
        print("Need 'GFM_StellarPhotometrics' of star ")
    
    return sim['GFM_StellarPhotometrics'][:,3]

#star
@derived_array
def g_mag(sim):
    """
    """
    if 'GFM_StellarPhotometrics' not in sim:
        print("Need 'GFM_StellarPhotometrics' of star ")
    
    return sim['GFM_StellarPhotometrics'][:,4]

#star
@derived_array
def r_mag(sim):
    """
    """
    if 'GFM_StellarPhotometrics' not in sim:
        print("Need 'GFM_StellarPhotometrics' of star ")
    
    return sim['GFM_StellarPhotometrics'][:,5]


#star
@derived_array
def i_mag(sim):
    """
    """
    if 'GFM_StellarPhotometrics' not in sim:
        print("Need 'GFM_StellarPhotometrics' of star ")
    
    return sim['GFM_StellarPhotometrics'][:,6]

#star
@derived_array
def z_mag(sim):
    """
    """
    if 'GFM_StellarPhotometrics' not in sim:
        print("Need 'GFM_StellarPhotometrics' of star ")
    
    return sim['GFM_StellarPhotometrics'][:,7]



#Refer mostly https://pynbody.readthedocs.io/latest/
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
    gamma = 5./3
    UnitEtoUnitM=((units.kpc/units.Gyr).in_units('km s^-1'))**2
    T=(gamma-1)/units.k*sim['mu']*sim['u']*UnitEtoUnitM

    T.convert_units('K')
    return T

#gas
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
    n=sim['ElectronAbundance']*sim['nH'].in_units('cm^-3')
    n.units=units.cm**-3
    return n

#gas
@derived_array
def em(sim) :
    """
    Calculates the Emission Measure (n_e^2) per particle, which is used to be integrated along the line of sight (LoS).

    Notes:
    ------
    The emission measure is calculated as the square of the electron number density (`ne`). 
    This value is used in astrophysical applications to estimate the emission from gas in a simulation, 
    often integrated along the line of sight.

    Formula:
    --------
    EM = n_e^2
    where:
    - n_e is the electron number density in cm^-3.
    """
    return (sim['ne']*sim['ne']).in_units('cm^-6')



#gas
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

    The result is converted to Pascals (Pa).

    Formula:
    --------
    P = (2 / 3) * u * rho
    """
    p = sim["u"] * sim["rho"].in_units('Msol kpc^-3') * (2. / 3)
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

    The result is converted to kilometers per second (km/s).

    Formula:
    --------
    c_s = sqrt( (5/3) * k * T / μ )
    """
    return (np.sqrt(5.0 / 3.0 * units.k* sim['temp'] / sim['mu'])).in_units('km s^-1')


@derived_array
def c_s(self):
    """
    Calculates the sound speed of the gas based on pressure and density.

    Notes:
    ------
    The sound speed is calculated using the formula:
    c_s = sqrt( (5/3) * (p / rho) )
    where:
    - p is the gas pressure.
    - rho is the gas density.

    The result is converted to kilometers per second (km/s).

    Formula:
    --------
    c_s = sqrt( (5/3) * p / rho )
    """
    #x = np.sqrt(5./3.*units.k*self['temp']*self['mu'])
    x = np.sqrt(5. / 3. * self['p'] / self['rho'].in_units('Msol kpc^-3'))
    x.convert_units('km s^-1')
    return x

#gas
@derived_array
def c_n_sq(sim) :
    """
    Calculates the turbulent amplitude C_N^2 for use in spectral calculations, 
    As in Eqn 20 of Macquart & Koay 2013 (ApJ 776 2).

    Notes:
    ------
    This calculation assumes a Kolmogorov spectrum of turbulence below the SPH resolution.
    
    The formula used is:
    C_N^2 = ((beta - 3) / (2 * (2 * π)^(4 - beta))) * L_min^(3 - beta) * EM
    
    Where:
    - beta = 11/3
    - L_min = 0.1 Mpc (minimum scale of turbulence)
    - EM = emission measure

    The result is returned in units of m^-20/3.

    Formula:
    --------
    C_N^2 = ((β - 3) / (2 * (2 * π)^(4 - β))) * L_min^(3 - β) * EM
    """

    ## Spectrum of turbulence below the SPH resolution, assume Kolmogorov
    beta = 11./3.
    L_min = 0.1*units.Mpc
    c_n_sq = ((beta - 3.)/((2.)*(2.*np.pi)**(4.-beta)))*L_min**(3.-beta)*sim["em"]
    c_n_sq.units = units.m**(-20,3)

    return c_n_sq


#gas
@derived_array
def Halpha(sim) :
    """
    Compute the H-alpha intensity for each gas particle based on the emission measure.

    The H-alpha emission is calculated using the following:
    - **Emission Measure (EM)**: \( \text{EM} = n_e^2 \), where \( n_e \) is the electron number density.
    - **H-alpha Intensity**: Computed using the Case B recombination coefficient and the emission measure.

    References:
    - Draine, B. T. (2011). "Physics of the Interstellar and Intergalactic Medium". 
    - For more details on the H-alpha intensity and its calculation, see: 
      https://pynbody.readthedocs.io/latest/_modules/pynbody/snapshot/gadgethdf.html
    - Additional information can be found at: 
      http://astro.berkeley.edu/~ay216/08/NOTES/Lecture08-08.pdf

    """
    # Define the H-alpha coefficient based on Planck's constant and the speed of light
    coeff = (6.6260755e-27) * (299792458. / 656.281e-9) / (4.*np.pi) ## units : erg sr^-1
    
    # Compute the recombination coefficient for H-alpha
    alpha = coeff * 7.864e-14 * (1e4 / sim['temp'].in_units('K'))

    # Set units for the alpha coefficient
    alpha.units = units.erg * units.cm**(3) * units.s**(-1) * units.sr**(-1) ## intensity in erg cm^3 s^-1 sr^-1

    # Calculate and return the H-alpha intensity
    return (alpha * sim["em"]).in_units('erg cm^-3 s^-1 sr^-1') # Flux erg cm^-3 s^-1 sr^-1

#gas
@derived_array
def nH(sim):
    """
    Calculate the total hydrogen number density for each gas particle.

    The hydrogen number density is computed using the following formula:
    - Total Hydrogen Number Density: X_H * (rho / m_p)
      where X_H is the hydrogen mass fraction, rho is the gas density, and m_p is the proton mass.

    """
    nh=sim['XH']*(sim['rho'].in_units('g cm^-3')/units.m_p).in_units('cm^-3')
    nh.units=units.cm**-3
    return nh


#gas
@derived_array
def XH(sim):
    """
    Calculate the hydrogen mass fraction for each gas particle.

    If the 'GFM_Metals' data is available in the simulation, the hydrogen mass fraction is extracted
    from this data. If 'GFM_Metals' is not present, a default value of 0.76 is used.
    """
    if 'GFM_Metals' in sim:
        Xh=sim['GFM_Metals'].view(np.ndarray).T[0]
        return SimArray(Xh)
    else:
        print('No GFM_Metals, use hydrogen mass fraction XH=0.76')
        return SimArray(0.76*np.ones(len(sim)))

    
# gas
@derived_array
def mu(sim):
    """
    Calculate the mean molecular weight of the gas.

    The mean molecular weight is computed using the hydrogen mass fraction (XH) and the electron
    abundance. The formula used is:
        μ = 4 / (1 + 3 * XH + 4 * XH * ElectronAbundance)

    Notes:
    ------
    - The 'ElectronAbundance' must be present in the simulation data to compute the mean molecular weight.
    """
    if 'ElectronAbundance' not in sim:
        print('need gas ElectronAbundance to cal: ElectronAbundance')
    muu=SimArray(4/(1+3*sim['XH']+4*sim['XH']*sim['ElectronAbundance']).astype(np.float64),units.m_p)
    return muu.in_units('m_p')





@simdict.SimDict.setter
def read_Snap_properties(f,SnapshotHeader):
    """
    Set cosmological and simulation properties for a given snapshot.

    This function initializes the simulation dictionary with various cosmological parameters and 
    snapshot-specific properties based on the provided SnapshotHeader.

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

    Snapshot-Specific Properties:
    ------------------------------
    - 'a': Scale factor (time) of the snapshot.
    - 'h': Hubble parameter.
    - 'omegaM0': Matter density parameter.
    - 'omegaL0': Dark energy density parameter.
    - 'omegaB0': Baryon density parameter (fixed value).
    - 'sigma8': Amplitude of matter density fluctuations (fixed value).
    - 'ns': Spectral index (fixed value).
    - 'boxsize': Size of the simulation box (in kpc), adjusted for the scale factor and Hubble parameter.
    - 'Halos_total': Total number of halos in the snapshot.
    - 'Subhalos_total': Total number of subhalos in the snapshot.
    """

    f['a']=SnapshotHeader['Time']
    f['h']=SnapshotHeader['HubbleParam']
    f['omegaM0']=SnapshotHeader['Omega0']
    f['omegaL0']=SnapshotHeader['OmegaLambda']
    f['omegaB0']=0.0486
    f['sigma8']=0.8159
    f['ns']=0.9667
    f['boxsize']=SimArray(1.,SnapshotHeader['BoxSize']*units.kpc*units.a/units.h)
    f['Halos_total']=SnapshotHeader['Ngroups_Total']
    f['Subhalos_total']=SnapshotHeader['Nsubgroups_Total']



@simdict.SimDict.setter
def filepath(f,BasePath):
    """
    Set the file path for the simulation data.

    This function updates the simulation dictionary with the base path of the data files and
    determines the simulation run identifier based on the base path.

    Parameters:
    -----------
    f : SimDict
        The simulation dictionary to be updated.
    BasePath : str
        The base directory path where the simulation data files are located.

    This function:
    - Sets the 'filedir' key to the provided BasePath.
    - Identifies the simulation run by checking which known run names are present in the BasePath.
    - Updates the 'run' key in the simulation dictionary with the identified run name.
    """
    f['filedir']=BasePath
    for i in illustrisTNGruns:
        if i in BasePath:
            f['run']=i
            break



@simdict.SimDict.getter
def t(d):
    """
    Calculate the age of the snapshot

    This function uses cosmological parameters and redshift to compute the age of the snapshot.
    The formula is derived from Peebles (p. 317, eq. 13.2).
    """
    import math
    omega_m = d['omegaM0']
    redshift=d['z']
    omega_fac = math.sqrt( (1-omega_m)/omega_m ) * pow(1+redshift,-3.0/2.0)
    H0_kmsMpc = 100.0 * d['h']*units.km/units.s/units.Mpc
    AGE = 2.0 * math.asinh(omega_fac) / (H0_kmsMpc * 3 * math.sqrt(1-omega_m))
    return AGE.in_units('Gyr')*units.Gyr

@simdict.SimDict.getter
def tLB(d):
    """
    Calculate the lookback time.
    """
    import math
    omega_m = d['omegaM0']
    redshift=0.
    omega_fac = math.sqrt( (1-omega_m)/omega_m ) * pow(1+redshift,-3.0/2.0)
    H0_kmsMpc = 100.0 * d['h']*units.km/units.s/units.Mpc
    AGE = 2.0 * math.asinh(omega_fac) / (H0_kmsMpc * 3 * math.sqrt(1-omega_m))
    return AGE.in_units('Gyr')*units.Gyr-d['t']



