from pynbody.snapshot import SimSnap
from pynbody import derived_array,filt
from pynbody.analysis.profile import Profile
from AnastrisTNG.illustris_python.snapshot import getSnapOffsets,loadSubset,loadSubhalo
from AnastrisTNG.TNGsnapshot import *
from AnastrisTNG.TNGunits import *
from AnastrisTNG.TNGmergertree import *
from AnastrisTNG.TNGsubhalo import subhalos
from AnastrisTNG.TNGhalo import halos, calc_faceon_matrix
from functools import reduce
class Snapshot(SimSnap):





    def __init__(self,BasePath,Snap,):
        SimSnap.__init__(self)
        self._num_particles = 0
        self._filename = "<created>"
        self._create_arrays(["pos", "vel"], 3)
        self._create_arrays(["mass"], 1)
        self._family_slice[family.get_family('dm')]=slice(0,0)
        self._family_slice[family.get_family('star')]=slice(0,0)
        self._family_slice[family.get_family('gas')]=slice(0,0)
        self._family_slice[family.get_family('bh')]=slice(0,0)
        self._decorate()
        self.properties['filepath']=BasePath
        self._filename = self.properties['run']

        self.__set_Snapshot_property(BasePath,Snap)
        self.properties['eps'],self.properties['Mdm']=get_eps_Mdm(self)
        self.properties['baseunits']=[units.Unit(x) for x in ('kpc', 'km s^-1', 'Msol')]
        for i in self.properties.keys():
            if isinstance(self.properties[i],SimArray):
                self.properties[i].sim=self
        self._filename = self._filename+'_'+'snapshot'+str(self.snapshot)

        self.__set_load_particle()
        self.subhalos=subhalos(self)
        self.halos=halos(self)
        self.__canloadPT=True
        self.__PT_loaded={'Halo':set(),
                        'Subhalo':set()}
        
        self.__GC_loaded={'Halo':set(),
                            'Subhalo':set()}
        self.__pos=SimArray([0.,0.,0.],units.kpc)
        self.__pos.sim=self
        self.__vel=SimArray([0.,0.,0.],units.km/units.s)
        self.__vel.sim=self
        self.__phi=SimArray([0.,0.,0.],units.km**2/units.s**2)
        self.__phi.sim=self
        self.__acc=SimArray([0.,0.,0.],units.km/units.s**2)
        self.__acc.sim=self
        self.__init_stable_array()

    @staticmethod
    def profile(sim,ndim=2,type='lin',nbins=100,**kwargs):
        pr=Profile(sim,ndim=ndim,type=type,nbins=nbins,**kwargs)
        
        def test_something(self):

            return self['rbins']

        
        pr._profile_registry[test_something.__name__]=test_something
        return pr



    


    def physical_units(self, persistent=False):

        dims = self.properties['baseunits']+[units.a,units.h]
        urc=len(dims)-2
        all = list(self._arrays.values())
        for x in self._family_arrays:
            if x in ['nH','Halpha','em','ne','temp','mu','c_n_sq','p','cs','c_s','acc','phi']:
                continue
            else:
                all += list(self._family_arrays[x].values())

        for ar in all:
            self._autoconvert_array_unit(ar.ancestor, dims,urc)

        for k in list(self.properties.keys()):
            v = self.properties[k]
            if isinstance(v, units.UnitBase):
                try:
                    new_unit = v.dimensional_project(dims)
                except units.UnitsException:
                    continue
                new_unit = reduce(lambda x, y: x * y, [
                                  a ** b for a, b in zip(dims, new_unit[:])])
                new_unit *= v.ratio(new_unit, **self.conversion_context())
                self.properties[k] = new_unit
            if isinstance(v,SimArray):
                if (v.units is not None) and (v.units is not units.no_unit):
                    try:
                        d = v.units.dimensional_project(dims)
                    except units.UnitsException:
                        return
                    new_unit = reduce(lambda x, y: x * y, [
                              a ** b for a, b in zip(dims, d[:urc])])
                    if new_unit != v.units:
                        self.properties[k].convert_units(new_unit)

        self.subhalos.physical_units()
        self.halos.physical_units()
        if persistent:
            self._autoconvert = dims
        else:
            self._autoconvert = None
    def load_GC(self):
        self.subhalos._load_GC()
        for i in self.subhalos.keys():
            self.__GC_loaded['Subhalo'].add(int(i))
        self.halos._load_GC()
        for i in self.halos.keys():
            self.__GC_loaded['Halo'].add(int(i))

    def load_halo(self,haloID):
        if haloID in self.__PT_loaded['Halo']:
            print(haloID, ' was already loaded into this Snapshot')
            return
        if self.__canloadPT:
            self.load_particle_para['particle_field']=self.load_particle_para['particle_field'].lower()
            self.load_particle_para['particle_field']=get_parttype(self.load_particle_para['particle_field'])
            f=self.load_particle(ID=haloID,groupType='Halo')

            fmerge=Simsnap_merge(self,f)
            Simsnap_cover(self,fmerge)

            ind = np.empty((len(self),), dtype='int8')
            for i, f in enumerate(self.ancestor.families()):
                ind[self._get_family_slice(f)] = i

            self._family_index_cached = ind
            self.halos[haloID]._load_GC()
            self.subhalos.update()
            self.halos.update()
            self.__PT_loaded['Halo'].add(haloID)
            self.__GC_loaded['Halo'].add(haloID)
            if self.halos[haloID].GC['GroupFirstSub'] != -1:
                for i in range(self.halos[haloID].GC['GroupFirstSub'],
                            self.halos[haloID].GC['GroupFirstSub']+self.halos[haloID].GC['GroupNsubs']):
                    self.__PT_loaded['Subhalo'].add(i)
        else:
            print('The pos and vel of the snapshot particles') 
            print('are not in the coordinate system in the original box.')
            print('No new particles can be loaded')

    

    def match_subhalo(self,subhaloID: int):
        parID=np.array([])
        for ty in ['star','gas','dm','bh']:
            thiID=loadSubhalo(self.properties['filedir'],self.snapshot,subhaloID,ty,fields=['ParticleIDs'])
            if isinstance(thiID,dict):
                continue
            parID=np.append(parID,thiID)
        parID.astype(np.uint64)
        self['SubhaloID'][np.isin(self['iord'],parID)]=subhaloID
        #self.subhalos.update()


    def load_subhalo(self, subhaloID:int ):
        if not isinstance(subhaloID,int):
            raise TypeError("subhaloID should be int")
        if subhaloID in self.__PT_loaded['Subhalo']:
            print(subhaloID, ' was already loaded into this Snapshot')
            print('So here update subhalos about this subhalo')
            if subhaloID in self['SubhaloID']:
                self.subhalos.update()
            else:
                print('No particle has this SubhaloID.')
                print('So here match this subhalo particle and modify their SubhaloID')
                self.match_subhalo(subhaloID)
            return
        if self.__canloadPT:
            self.load_particle_para['particle_field']=self.load_particle_para['particle_field'].lower()
            self.load_particle_para['particle_field']=get_parttype(self.load_particle_para['particle_field'])
            f=self.load_particle(ID=subhaloID,groupType='Subhalo')


            fmerge=Simsnap_merge(self,f)
            Simsnap_cover(self,fmerge)

            ind = np.empty((len(self),), dtype='int8')
            for i, f in enumerate(self.ancestor.families()):
                ind[self._get_family_slice(f)] = i
            self._family_index_cached = ind

            self.subhalos[subhaloID]._load_GC()
            self['HaloID'][self['SubhaloID']==subhaloID]=self.subhalos[subhaloID].GC['SubhaloGrNr']
            self.subhalos.update()
            self.halos.update()
            self.__PT_loaded['Subhalo'].add(subhaloID)
            self.__GC_loaded['Subhalo'].add(subhaloID)
        else:
            print('The pos and vel of the snapshot particles') 
            print('are not in the coordinate system in the original box.')
            print('No new particles can be loaded')
    
    
    def load_particle(self,ID,groupType='Subhalo'):
        '''
        ID: int, halo or subhalo id
        groupType: str, 'Halo' or 'Subhalo'

        '''
        if groupType =='Halo':
            subset = getSnapOffsets(self.properties['filedir'], self.snapshot, ID,'Group')
        else:
            subset = getSnapOffsets(self.properties['filedir'], self.snapshot, ID,'Subhalo')
            
        lenType=subset['lenType']
        f=new(dm=int(lenType[1]),star=int(lenType[4]),gas=int(lenType[0]),bh=int(lenType[5]),
                       order='dm,star,gas,bh')
        
        for party in self.load_particle_para['particle_field'].split(","):
            if len(f[family.get_family(party)])>0 :
                if len(self.load_particle_para[party+'_fields'])>0:
                    self.load_particle_para[party+'_fields']=list(set(self.load_particle_para[party+'_fields']+
                                                                    self.load_particle_para['Basefields']))
                else:
                    self.load_particle_para[party+'_fields']=list.copy(self.load_particle_para['Basefields'])

                if party=='dm':
                    if 'Masses' in self.load_particle_para[party+'_fields']:
                        self.load_particle_para[party+'_fields'].remove('Masses')
                    loaddata=loadSubset(self.properties['filedir'], self.snapshot, party,
                                            self.load_particle_para[party+'_fields'],subset=subset)
                    for i in self.load_particle_para[party+'_fields']:
                        f.dm[SnapshotPaName(i)]=SimArray(loaddata[i],SnapshotsUnits(i))
                    if 'Masses' in self.load_particle_para['Basefields']:
                        f.dm['mass']=self.properties['Mdm'].in_units(SnapshotsUnits('Masses'))*np.ones(len(f.dm))
                        self.load_particle_para[party+'_fields'].append('Masses')
                    f.dm[groupType+'ID']=SimArray(ID*np.ones(len(f.dm)).astype(np.int32))
                    if groupType =='Halo':
                        f.dm['SubhaloID']=SimArray(-1*np.ones(len(f.dm)).astype(np.int32))
                    else:
                        f.dm['HaloID']=SimArray(-1*np.ones(len(f.dm)).astype(np.int32))

                if party=='star':
                    loaddata=loadSubset(self.properties['filedir'], self.snapshot, party,
                                            self.load_particle_para[party+'_fields'],subset=subset)
                    for i in self.load_particle_para[party+'_fields']:
                        f.s[SnapshotPaName(i)]=SimArray(loaddata[i],SnapshotsUnits(i))
                    f.s[groupType+'ID']=SimArray(ID*np.ones(len(f.s)).astype(np.int32))
                    if groupType =='Halo':
                        f.s['SubhaloID']=SimArray(-1*np.ones(len(f.s)).astype(np.int32))
                    else:
                        f.s['HaloID']=SimArray(-1*np.ones(len(f.s)).astype(np.int32))

                if party=='gas':
                    loaddata=loadSubset(self.properties['filedir'], self.snapshot, party,
                                            self.load_particle_para[party+'_fields'],subset=subset)
                    for i in self.load_particle_para[party+'_fields']:
                        f.g[SnapshotPaName(i)]=SimArray(loaddata[i],SnapshotsUnits(i))
                    f.g[groupType+'ID']=SimArray(ID*np.ones(len(f.g)).astype(np.int32))
                    if groupType =='Halo':
                        f.g['SubhaloID']=SimArray(-1*np.ones(len(f.g)).astype(np.int32))
                    else:
                        f.g['HaloID']=SimArray(-1*np.ones(len(f.g)).astype(np.int32))

                if party=='bh':
                    loaddata=loadSubset(self.properties['filedir'], self.snapshot, party,
                                            self.load_particle_para[party+'_fields'],subset=subset)
                    for i in self.load_particle_para[party+'_fields']:
                        f.bh[SnapshotPaName(i)]=SimArray(loaddata[i],SnapshotsUnits(i))
                    f.bh[groupType+'ID']=SimArray(ID*np.ones(len(f.bh)).astype(np.int32))
                    if groupType =='Halo':
                        f.bh['SubhaloID']=SimArray(-1*np.ones(len(f.bh)).astype(np.int32))
                    else:
                        f.bh['HaloID']=SimArray(-1*np.ones(len(f.bh)).astype(np.int32))
        f.properties=self.properties
        f._filename=self.filename+'_'+groupType+'_'+str(ID)
        return f
    


    def target_acceleration(self,targetpos):
        try:
            eps=self.properties.get('eps',0)
        except:
            eps=0
        if eps==0:
            print('Calculate the gravity without softening length')
        accelr=AccelTarget(targetpos,self['pos'].view(np.ndarray),self['mass'].view(np.ndarray),
                  np.repeat(eps,len(targetpos)).view(np.ndarray))
        acc=SimArray(accelr,units.G*self['mass'].units/self['pos'].units/self['pos'].units)
        acc.sim=self
        return acc

    
    def target_potential(self,targetpos):
        try:
            eps=self.properties.get('eps',0)
        except:
            eps=0
        if eps==0:
            print('Calculate the gravity without softening length')
        pot=PotentialTarget(targetpos,self['pos'].view(np.ndarray),self['mass'].view(np.ndarray),
                  np.repeat(eps,len(targetpos)).view(np.ndarray))
        phi=SimArray(pot,units.G*self['mass'].units/self['pos'].units)
        phi.sim=self
        return phi
    


    


    def wrap(self,boxsize=None, convention='center'):
        
        super().wrap(boxsize, convention)
        self.__canloadPT=False

        print('It involves a change of coordinates')
        print('Can\'t load new particles in this Snapshot')

    
    def check_boundary(self):
        if (self['x'].max()-self['x'].min())>(self.boxsize/2):
            print('On the edge of the box, move to center')
            self.wrap()
            return
        if (self['y'].max()-self['y'].min())>(self.boxsize/2):
            print('On the edge of the box, move to center')
            self.wrap()
            return
        if (self['z'].max()-self['z'].min())>(self.boxsize/2):
            print('On the edge of the box, move to center')
            self.wrap()
            return

        return

    def _PT_potential(self):
        '''
        Calculate the potential for each particle
        https://github.com/mikegrudic/pytreegrav
        '''
        self.check_boundary()
        print('Calculating gravity and it will take tens of seconds')
        if len(self['mass'])>1000:
            print('Calculate by using Octree')
        else:
            print('Calculate by using brute force')
        try:
            eps=self.properties.get('eps',0)
        except:
            eps=0
        if eps==0:
            print('Calculate the gravity without softening length')
       # self.physical_units()
        pot=Potential(self['pos'].view(np.ndarray),self['mass'].view(np.ndarray),
                  np.repeat(eps,len(self['mass'])).view(np.ndarray))
        phi=SimArray(pot,units.G*self['mass'].units/self['pos'].units)
        self['phi']=phi
        self['phi'].convert_units('km**2 s**-2')
        self.__canloadPT=False
        return self['phi']
    
    def _PT_acceleration(self):
        '''
        Calculate the acceleration for each particle
        https://github.com/mikegrudic/pytreegrav
        '''
        self.check_boundary()
        if len(self['mass'])>1000:
            print('Calculate by using Octree')
        else:
            print('Calculate by using brute force')
        try:
            eps=self.properties.get('eps',0)
        except:
            eps=0
        if eps==0:
            print('Calculate the gravity without softening length')
      #  self.physical_units()
        accelr=Accel(self['pos'].view(np.ndarray),self['mass'].view(np.ndarray),
                  np.repeat(eps,len(self['mass'])).view(np.ndarray))
        acc=SimArray(accelr,units.G*self['mass'].units/self['pos'].units/self['pos'].units)
        self['acc']=acc
        self.__canloadPT=False
        self['acc'].convert_units('km s^-1 Gyr^-1')
        return self['acc']

    def __init_stable_array(self):
        if '_derived_quantity_registry' in dir(self):
            self._derived_quantity_registry[phi.__name__]=phi
            self._derived_quantity_registry[phi.__name__] = phi
            phi.__stable__=True

            self._derived_quantity_registry[acc.__name__]=acc
            self._derived_quantity_registry[acc.__name__] = acc
            acc.__stable__=True
            return
        if '_derived_array_registry' in dir(self):
            self._derived_array_registry[phi.__name__]=phi
            self._derived_array_registry[phi.__name__] = phi
            phi.__stable__=True

            self._derived_array_registry[acc.__name__]=acc
            self._derived_array_registry[acc.__name__] = acc
            acc.__stable__=True

    def __repr__(self):
        return "<Snapshot \"" + self.filename + "\" len=" + str(len(self)) + ">"

    def __set_Snapshot_property(self,BasePath,Snap):
        SnapshotHeader=loadHeader(BasePath,Snap)
        self.properties=simdict.SimDict()
        self.properties['read_Snap_properties']=SnapshotHeader
        for i in self.properties.keys():
            if 'sim' in dir(self.properties[i]):
                self.properties[i].sim=self
        self.properties['filepath']=BasePath
        self.properties['Snapshot']=Snap




    def __set_load_particle(self):
        pa={}
        pa['particle_field']='dm,star,gas,bh'
        pa['Basefields']=['Coordinates','Velocities','Masses','ParticleIDs']
        pa['star_fields']=[]
        pa['gas_fields']=[]
        pa['dm_fields']=[]
        pa['bh_fields']=[]
        self.load_particle_para=pa

    def _autoconvert_array_unit(self, ar, dims=None, ucut=3):
        """Given an array ar, convert its units such that the new units span
        dims[:ucut]. dims[ucut:] are evaluated in the conversion (so will be things like
        a, h etc).

        If dims is None, use the internal autoconvert state to perform the conversion."""

        if dims is None:
            dims = self.ancestor._autoconvert
        if dims is None:
            return
        if (ar.units is not None) and (ar.units is not units.no_unit):
            try:
                d = ar.units.dimensional_project(dims)
            except units.UnitsException:
                return
            new_unit = reduce(lambda x, y: x * y, [
                              a ** b for a, b in zip(dims, d[:ucut])])
            if new_unit != ar.units:

                ar.convert_units(new_unit)


    def _a_dot(self):
        a=self.a
        h0=self.h
        om_m=self.omegaM0
        om_l=self.omegaL0
        om_k = 1.0 - om_m - om_l
        return  h0 * a * np.sqrt(om_m * (a ** -3) + om_k * (a ** -2) + om_l)
    

    

    @property
    def status_loadPT(self):
        '''
        Check the ability of this snapshot to load new particles
        '''

        if self.__canloadPT:
            return 'able'
        else:
            return 'locked'
        
    def shift(self,pos=None,vel=None,phi=None):
        '''
        shift to the specific position
        then set its pos, vel, phi, acc to 0.
        '''
        
        if pos is not None:
            self['pos']-=pos
            self.__pos.convert_units(self['pos'].units)
            self.__pos+=pos
        if vel is not None:
            self['vel']-=vel
            self.__vel.convert_units(self['vel'].units)
            self.__vel+=vel
        if (phi is not None) and ('phi' in self.keys()):
            self['phi']-=phi
            self.__phi.convert_units(self['phi'].units)
            self.__phi+=phi

        if 'acc' in self.keys():
            theacc=self.target_acceleration(np.array([[0,0,0],pos]))[1]
            self['acc']-=theacc
            self.__acc.convert_units(self['acc'].units)
            self.__acc+=theacc


    def get_origin_inbox(self):
        
        return self.__pos.copy(),self.__vel.copy(),self.__acc.copy(),self.__phi.copy()



    def vel_center(self,mode='ssc',pos=None,r_cal='1 kpc'):
        '''
        The center velocity.
        Refer from https://pynbody.readthedocs.io/latest/_modules/pynbody/analysis/halo.html#vel_center

        ``mode`` used to cal center pos see ``center``
        ``pos``  Specified position.
        ``r_cal`` The size of the sphere to use for the velocity calculate

        '''


        if pos==None:
            pos=self.center(mode)

        cen = self.s[filt.Sphere(r_cal,pos)]
        if len(cen) < 5:
            # fall-back to DM
            cen = self.dm[filt.Sphere(r_cal,pos)]
        if len(cen) < 5:
            # fall-back to gas
            cen = self.g[filt.Sphere(r_cal,pos)]
        if len(cen) < 5:
            # very weird snapshot, or mis-centering!
            raise ValueError("Insufficient particles around center to get velocity")

        vcen = (cen['vel'].transpose() * cen['mass']).sum(axis=1)/cen['mass'].sum()
        vcen.units = cen['vel'].units

        return vcen


    def center(self,mode='ssc'):
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
        if mode=='pot':
         #   if 'phi' not in self.keys():
          #      phi=self['phi']
            i = self["phi"].argmin()
            return self["pos"][i].copy()
        if mode=='com':
            return self.mean_by_mass('pos')
        if mode=='ssc':
            from pynbody.analysis.halo import shrink_sphere_center
            return shrink_sphere_center(self)
        if mode=='hyb':
        #    if 'phi' not in self.keys():
         #       phi=self['phi']
            from pynbody.analysis.halo import hybrid_center
            return hybrid_center(self)
        print('No such mode')

        return 
    
    def face_on(self,mode='ssc',alignwith='all',shift=True):
        pos_center=self.center(mode=mode)
        vel_center=self.vel_center(mode=mode)
        if alignwith in ['all','total','All','Total']:
            angmom = (self['mass'].reshape((len(self), 1)) *
              np.cross(self['pos']-pos_center, self['vel']-vel_center)).sum(axis=0).view(np.ndarray)
        elif alignwith in ['DM','dm','darkmatter','Darkmatter']:
            angmom = (self.dm['mass'].reshape((len(self.dm), 1)) *
              np.cross(self.dm['pos']-pos_center, self.dm['vel']-vel_center)).sum(axis=0).view(np.ndarray)
        elif alignwith in ['star','s','Star']:
            angmom = (self.s['mass'].reshape((len(self.s), 1)) *
              np.cross(self.s['pos']-pos_center, self.s['vel']-vel_center)).sum(axis=0).view(np.ndarray)
        elif alignwith in ['gas','g','Gas']:
            angmom = (self.g['mass'].reshape((len(self.g), 1)) *
              np.cross(self.g['pos']-pos_center, self.g['vel']-vel_center)).sum(axis=0).view(np.ndarray)
        elif alignwith in ['baryon','baryonic']:
            angmom1 = (self.g['mass'].reshape((len(self.g), 1)) *
              np.cross(self.g['pos']-pos_center, self.g['vel']-vel_center)).sum(axis=0).view(np.ndarray)
            angmo2 = (self.s['mass'].reshape((len(self.s), 1)) *
              np.cross(self.s['pos']-pos_center, self.s['vel']-vel_center)).sum(axis=0).view(np.ndarray)
            angmom=angmom1+angmo2

        trans =calc_faceon_matrix(angmom)
        if shift:
            self.ancestor.shift(pos=pos_center,vel=vel_center)
            self._transform(trans)
        else:
            self._transform(trans)

    
    @property
    def cosmology(self):
        cos={}
        cos['h']=self.properties.get('h')
        cos['omegaM0']=self.properties.get('omegaM0')
        cos['omegaL0']=self.properties.get('omegaL0')
        cos['omegaB0']=self.properties.get('omegaB0')
        cos['sigma8']=self.properties.get('sigma8')
        cos['ns']=self.properties.get('ns')
        return cos

    @property
    def GC_loaded_Subhalo(self):
        return np.sort(list(self.__GC_loaded['Subhalo'].copy()))
    
    @property
    def GC_loaded_Halo(self):
        return np.sort(list(self.__GC_loaded['Halo'].copy()))

    @property
    def PT_loaded_Halo(self):
        return np.sort(list(self.__PT_loaded['Halo'].copy()))
    
    @property
    def PT_loaded_Subhalo(self):
        return np.sort(list(self.__PT_loaded['Subhalo'].copy()))
    
    @property
    def eps(self):
        return self.properties['eps']
    
    @property
    def rho_crit(self):
        z = self.z
        omM = self.omegaM0
        omL = self.omegaL0
        h0 = self.h
        a = self.a
        H_z = self._a_dot() /a
        H_z = units.Unit("100 km s^-1 Mpc^-1") * H_z

        rho_crit = (3 * H_z ** 2) / (8 * np.pi * units.G)
        return rho_crit
    
    @property
    def snapshot(self):
        return self.properties['Snapshot']

    @property
    def run(self):
        return self.properties['run']

    @property
    def ns(self):
        return self.properties['ns']

    @property
    def sigma8(self):
        return self.properties['sigma8']

    @property
    def omegaB0(self):
        return self.properties['omegaB0']

    @property
    def omegaL0(self):
        return self.properties['omegaL0']

    @property
    def omegaM0(self):
        return self.properties['omegaM0']

    @property
    def boxsize(self):
        return self.properties['boxsize']

    @property
    def h(self):
        return self.properties['h']

    @property
    def t(self):
        return self.properties['t']
    
    @property
    def tLB(self):
        return self.properties['tLB']
    
    @property
    def a(self):
        return self.properties['a']
    
    @property
    def z(self):
        return self.properties['z']