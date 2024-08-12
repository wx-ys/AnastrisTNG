from pynbody import simdict,units,family,derived_array
from pynbody.snapshot import new
import types
from AnastrisTNG.illustris_python.groupcat import loadHeader
from pynbody.array import SimArray
from AnastrisTNG.TNGunits import illustrisTNGruns
import numpy as np


def Simsnap_cover(f1,f2):
    f1._num_particles=len(f2)
    if len(f2.dm)>0:
        f1._family_slice[family.get_family('dm')]= f2._family_slice[family.get_family('dm')]
    if len(f2.s)>0:
        f1._family_slice[family.get_family('star')]= f2._family_slice[family.get_family('star')]
    if len(f2.g)>0:
        f1._family_slice[family.get_family('gas')]= f2._family_slice[family.get_family('gas')]
    if len(f2.bh)>0:
        f1._family_slice[family.get_family('bh')]= f2._family_slice[family.get_family('bh')]
    for i in f1.keys():
        del f1[i]
    f1._create_arrays(["pos", "vel"], 3)
    f1._create_arrays(["mass"], 1)
    f1._decorate()
    if len(f1.dm)>0:
        for i in f2.dm.keys():
            f1.dm[i]=f2.dm[i]
    if len(f1.s)>0:
        for i in f2.s.keys():
            f1.s[i]=f2.s[i]
    if len(f1.g)>0:
        for i in f2.g.keys():
            f1.g[i]=f2.g[i]
    if len(f1.bh)>0:
        for i in f2.bh.keys():
            f1.bh[i]=f2.bh[i]



def Simsnap_merge(f1,f2):
    f3=new(star=len(f1.s)+len(f2.s),
            gas=len(f1.g)+len(f2.g),
            dm=len(f1.dm)+len(f2.dm),
            bh=len(f1.bh)+len(f2.bh),
            order='dm,star,gas,bh')
    if len(f3.s)>0:
        if len(f1.s)==0:
            for i in f2.s.keys():
                f3.s[i]=f2.s[i]
        elif len(f2.s)==0:
            for i in f1.s.keys():
                f3.s[i]=f1.s[i]
        else:
            for i in f2.s.keys():
                f3.s[i]=SimArray(np.append(f1.s[i],f2.s[i],axis=0),f2.s[i].units)

    if len(f3.dm)>0:
        if len(f1.dm)==0:
            for i in f2.dm.keys():
                f3.dm[i]=f2.dm[i]
        elif len(f2.dm)==0:
            for i in f1.dm.keys():
                f3.dm[i]=f1.dm[i]
        else:
            for i in f2.dm.keys():
                f3.dm[i]=SimArray(np.append(f1.dm[i],f2.dm[i],axis=0),f2.dm[i].units)

    if len(f3.g)>0:
        if len(f1.g)==0:
            for i in f2.g.keys():
                f3.g[i]=f2.g[i]
        elif len(f2.g)==0:
            for i in f1.g.keys():
                f3.g[i]=f1.g[i]
        else:
            for i in f2.g.keys():
                f3.g[i]=SimArray(np.append(f1.g[i],f2.g[i],axis=0),f2.g[i].units)

    if len(f3.bh)>0:
        if len(f1.bh)==0:
            for i in f2.bh.keys():
                f3.bh[i]=f2.bh[i]
        elif len(f2.bh)==0:
            for i in f1.bh.keys():
                f3.bh[i]=f1.bh[i]
        else:
            for i in f2.bh.keys():
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





def get_Snapshot_property(BasePath,Snap):
    SnapshotHeader=loadHeader(BasePath,Snap)
    Snapshot=simdict.SimDict()
    Snapshot['filepath']=BasePath
    Snapshot['read_Snap_properties']=SnapshotHeader
    Snapshot['Snapshot']=Snap
    return Snapshot

'''

def set_Snapshot_property(f,BasePath,Snap):
    SnapshotHeader=loadHeader(BasePath,Snap)
    Snapshot(f)
    f.Snapshot['read_Snap_properties']=SnapshotHeader
    for i in f.Snapshot.keys():
        if 'sim' in dir(f.Snapshot[i]):
            f.Snapshot[i].sim=f
    f.Snapshot['filepath']=BasePath
    f.Snapshot['Snapshot']=Snap
    f.properties=f.Snapshot



def Snapshot(f):
    if 'Snapshot' in dir(f):
        pass
    else:
        f.__setattr__('Snapshot',simdict.SimDict())
    return f.Snapshot
'''


def get_eps_Mdm(Snapshot):
    '''
    Gravitational softenings for stars and DM are in comoving kpc 
    until z=1 after which they are fixed to their z=1 values.
                                                    ---Dylan Nelson
    
    Data from https://www.tng-project.org/data/docs/background/
    '''
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
    




# all
@derived_array
def phi(sim) :
    if 'phi' not in sim.keys():
        print('There is no phi in the keyword')
        if ('mass' in sim.keys()) and ('pos' in sim.keys()):
            sim.ancestor._PT_potential()
        else:
            print('\'phi\' fails to be calculated. The keys \'mass\' and \'pos\' are required ')
            return
    return sim['phi']


# all
@derived_array
def acc(sim) :
    if 'acc' not in sim.keys():
        print('There is no acc in the keyword')
        if ('mass' in sim.keys()) and ('pos' in sim.keys()):
            sim.ancestor._PT_acceleration()
        else:
            print('\'acc\' fails to be calculated. The keys \'mass\' and \'pos\' are required ')
            return
    return sim['acc']




# star
@derived_array
def tform(sim,):
    if 'aform' not in sim.keys():
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
    
    return sim.properties['t']-sim['tform']





#Refer mostly https://pynbody.readthedocs.io/latest/
# gas
@derived_array
def temp(sim):
    '''
    TNG use two-phase ISM sub-grid model
    the gas temperature
    see Sec.6 of https://www.tng-project.org/data/docs/faq/  
    '''
    if 'u' not in sim.keys():
        print('need gas InternalEnergy to cal: InternalEnergy')
    gamma = 5./3
    UnitEtoUnitM=((units.kpc/units.Gyr).in_units('km s^-1'))**2
    T=(gamma-1)/units.k*sim['mu']*sim['u']*UnitEtoUnitM

    T.convert_units('K')
    return T

#gas
@derived_array
def ne(sim):
    '''
    the electron number density
    '''
    n=sim['ElectronAbundance']*sim['nH'].in_units('cm^-3')
    n.units=units.cm**-3
    return n

#gas
@derived_array
def em(sim) :
    """Emission Measure (n_e^2) per particle to be integrated along LoS"""

    return (sim['ne']*sim['ne']).in_units('cm^-6')



#gas
@derived_array
def p(sim):
    """Pressure"""
    p = sim["u"] * sim["rho"].in_units('Msol kpc^-3') * (2. / 3)
    p.convert_units("Pa")
    return p

@derived_array
def cs(sim):
    """Sound speed"""
    return (np.sqrt(5.0 / 3.0 * units.k* sim['temp'] / sim['mu'])).in_units('km s^-1')


@derived_array
def c_s(self):
    """Ideal gas sound speed based on pressure and density"""
    #x = np.sqrt(5./3.*units.k*self['temp']*self['mu'])
    x = np.sqrt(5. / 3. * self['p'] / self['rho'].in_units('Msol kpc^-3'))
    x.convert_units('km s^-1')
    return x

#gas
@derived_array
def c_n_sq(sim) :
    """Turbulent amplitude C_N^2 for use in SM calculations (e.g. Eqn 20 of Macquart & Koay 2013 ApJ 776 2) """

    ## Spectrum of turbulence below the SPH resolution, assume Kolmogorov
    beta = 11./3.
    L_min = 0.1*units.Mpc
    c_n_sq = ((beta - 3.)/((2.)*(2.*np.pi)**(4.-beta)))*L_min**(3.-beta)*sim["em"]
    c_n_sq.units = units.m**(-20,3)

    return c_n_sq


#gas
@derived_array
def Halpha(sim) :
    # Refer from https://pynbody.readthedocs.io/latest/_modules/pynbody/snapshot/gadgethdf.html#
    """H alpha intensity (based on Emission Measure n_e^2) per particle to be integrated along LoS"""

    ## Rate at which recombining electrons and protons produce Halpha photons.
    ## Case B recombination assumed from Draine (2011)
    #alpha = 2.54e-13 * (sim.g['temp'].in_units('K') / 1e4)**(-0.8163-0.0208*np.log(sim.g['temp'].in_units('K') / 1e4))
    #alpha.units = units.cm**(3) * units.s**(-1)

    ## H alpha intensity = coeff * EM
    ## where coeff is h (c / Lambda_Halpha) / 4Pi) and EM is int rho_e * rho_p * alpha
    ## alpha = 7.864e-14 T_1e4K from http://astro.berkeley.edu/~ay216/08/NOTES/Lecture08-08.pdf

    coeff = (6.6260755e-27) * (299792458. / 656.281e-9) / (4.*np.pi) ## units are erg sr^-1
    alpha = coeff * 7.864e-14 * (1e4 / sim['temp'].in_units('K'))

    alpha.units = units.erg * units.cm**(3) * units.s**(-1) * units.sr**(-1) ## It's intensity in erg cm^3 s^-1 sr^-1

    return (alpha * sim["em"]).in_units('erg cm^-3 s^-1 sr^-1') # Flux erg cm^-3 s^-1 sr^-1

#gas
@derived_array
def nH(sim):
    '''
    the total hydrogen number density
    '''
    nh=sim['XH']*(sim['rho'].in_units('g cm^-3')/units.m_p).in_units('cm^-3')
    nh.units=units.cm**-3
    return nh


#gas
@derived_array
def XH(sim):
    '''
    the hydrogen mass fraction
    '''
    if 'GFM_Metals' in sim.keys():
        Xh=sim['GFM_Metals'].view(np.ndarray).T[0]
        return SimArray(Xh)
    else:
        print('No GFM_Metals, use hydrogen mass fraction XH=0.76')
        return SimArray(0.76*np.ones(len(sim)))

    
# gas
@derived_array
def mu(sim):
    '''
    the mean molecular weight
    XH can be best estimated by GFM_Metals
    '''
    if 'ElectronAbundance' not in sim.keys():
        print('need gas ElectronAbundance to cal: ElectronAbundance')
   
    return SimArray(4/(1+3*sim['XH']+4*sim['XH']*sim['ElectronAbundance']).astype(np.float64),units.m_p)





@simdict.SimDict.setter
def read_Snap_properties(f,SnapshotHeader):
    '''
    TNG runs cosmologicaL model:
        Standard ΛCDM: Planck 2015 results
            omegaL0=0.6911
            omegaM0=0.3089
            omegaB0=0.0486
            sigama8=0.8159
            ns=0.9667
            h=0.6774
    The box side-length
    '''

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
    f['filedir']=BasePath
    for i in illustrisTNGruns:
        if i in BasePath:
            f['run']=i
            break



@simdict.SimDict.getter
def t(d):
    #(Peebles, p. 317, eq. 13.2)
    import math
    omega_m = d['omegaM0']
    redshift=d['z']
    omega_fac = math.sqrt( (1-omega_m)/omega_m ) * pow(1+redshift,-3.0/2.0)
    H0_kmsMpc = 100.0 * d['h']*units.km/units.s/units.Mpc
    AGE = 2.0 * math.asinh(omega_fac) / (H0_kmsMpc * 3 * math.sqrt(1-omega_m))
    return AGE.in_units('Gyr')*units.Gyr

@simdict.SimDict.getter
def tLB(d):
    import math
    omega_m = d['omegaM0']
    redshift=0.
    omega_fac = math.sqrt( (1-omega_m)/omega_m ) * pow(1+redshift,-3.0/2.0)
    H0_kmsMpc = 100.0 * d['h']*units.km/units.s/units.Mpc
    AGE = 2.0 * math.asinh(omega_fac) / (H0_kmsMpc * 3 * math.sqrt(1-omega_m))
    return AGE.in_units('Gyr')*units.Gyr-d['t']



