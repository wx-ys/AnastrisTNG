import numpy as np
from AnastrisTNG.TNGgroupcat import haloproperties
from AnastrisTNG.TNGunits import NotneedtransGCPa
from pynbody.simdict import SimDict
from pynbody.array import SimArray
from pynbody import units,filt
from functools import reduce
from pynbody.analysis.angmom import calc_faceon_matrix
from pynbody.analysis.halo import virial_radius
class Halo:
    """
    Represents a single halo in the simulation.

    This class contains information about the particles of the halo and its corresponding group catalog data.
    It also includes functions to compute properties specific to this halo.

    Attributes:
    ----------
    PT : SimArray
        The particles of the halo. Detailed information about this can be found at
        https://www.tng-project.org/data/docs/specifications/#sec1.
    GC : SimDict
        The group catalog for this halo. Detailed information about this can be found at
        https://www.tng-project.org/data/docs/specifications/#sec2.

    Parameters:
    ----------
    simarray : SimArray
        An object containing the particle data for the halo.

    """
    
    
    def __init__(self,simarray):
        """
        Initializes the Halo object.

        Parameters:
        -----------
        simarray : object
            An object that contains halo particles.
        """
        self.PT=simarray
        self.GC=SimDict()
        self.GC.update(simarray.properties)
        self.GC['HaloID']=int(simarray.filename.split('_')[-1])

    def _load_GC(self):
        """
        Loads the group catalog data for this halo and updates its properties.
        """
        proper=haloproperties(self.GC['filedir'],
                                 self.GC['Snapshot'],
                                 self.GC['HaloID'])
        self.GC.update(proper)
        for i in self.GC.keys():
            if isinstance(self.GC[i],SimArray):
                self.GC[i].sim=self.PT.ancestor

    def GC_physical_units(self, distance='kpc', velocity='km s^-1', mass='Msol'):
        """
        Converts the units of the group catalog (GC) properties to physical units.
        
        This method updates the `GC` attribute of the `Subhalo` instance to use physical units
        for its properties, based on predefined unit conversions and the current unit context.

        Conversion is applied only to properties that are not listed in `NotneedtransGCPa`.

        Notes:
        -----
        - `self.PT.ancestor.properties['baseunits']` provides the base units for dimensional analysis.
        - The dimensional projection and conversion are handled using the `units` library.
        - Properties listed in `NotneedtransGCPa` are skipped during the conversion process.
        """
        dims = self.PT.ancestor.properties['baseunits']+[units.a,units.h]
        urc=len(dims)-2
        for k in list(self.GC.keys()):
            if k in NotneedtransGCPa:
                continue
            v = self.GC[k]
            if isinstance(v, units.UnitBase):
                try:
                    new_unit = v.dimensional_project(dims)
                except units.UnitsException:
                    continue
                new_unit = reduce(lambda x, y: x * y, [
                                  a ** b for a, b in zip(dims, new_unit[:urc])])
                new_unit *= v.ratio(new_unit, **self.conversion_context())
                self.GC[k] = new_unit
            if isinstance(v,SimArray):
                if (v.units is not None) and (v.units is not units.no_unit):
                    try:
                        d = v.units.dimensional_project(dims)
                    except units.UnitsException:
                        return
                    new_unit = reduce(lambda x, y: x * y, [
                              a ** b for a, b in zip(dims, d[:urc])])
                    if new_unit != v.units:
                        self.GC[k].convert_units(new_unit)

    def vel_center(self,mode='ssc',pos=None,r_cal='1 kpc'):
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

        if pos==None:
            pos=self.center(mode)

        cen = self.PT.s[filt.Sphere(r_cal,pos)]
        if len(cen) < 5:
            # fall-back to DM
            cen = self.PT.dm[filt.Sphere(r_cal,pos)]
        if len(cen) < 5:
            # fall-back to gas
            cen = self.PT.g[filt.Sphere(r_cal,pos)]
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
        if self.__check_paticles():
            print('No particles loaded in this Halo')
            return
        if mode=='pot':
         #   if 'phi' not in self.keys():
          #      phi=self['phi']
            i = self.PT["phi"].argmin()
            return self.PT["pos"][i].copy()
        if mode=='com':
            return self.PT.mean_by_mass('pos')
        if mode=='ssc':
            from pynbody.analysis.halo import shrink_sphere_center
            return shrink_sphere_center(self.PT)
        if mode=='hyb':
        #    if 'phi' not in self.keys():
         #       phi=self['phi']
            from pynbody.analysis.halo import hybrid_center
            return hybrid_center(self.PT)
        print('No such mode')

        return 
    
    def face_on(self,mode='ssc',alignwith='all',shift=True):
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
            - 'DM', 'dm', 'darkmatter', 'Darkmatter': Uses the angular momentum of dark matter.
            - 'star', 's', 'Star': Uses the angular momentum of stars.
            - 'gas', 'g', 'Gas': Uses the angular momentum of gas.
            - 'baryon', 'baryonic': Uses the combined angular momentum of stars and gas.
        shift : bool, optional
            If True, shifts the halo to its center of mass and adjusts the coordinate system. Default is True.

        Notes:
        ------
        - `self.center(mode=mode)` computes the center of the halo based on the specified mode.
        - `self.vel_center(mode=mode)` computes the velocity center of the halo.
        - `self.R_vir(cen=pos_center)` provides the virial radius based on the computed center.
        - `calc_faceon_matrix(angmom)` calculates the transformation matrix to align the halo to face-on view.
        - The method assumes the presence of `immediate_mode` for `innerpart` to enable efficient data access.
        """
        
        
        pos_center=self.center(mode=mode)
        vel_center=self.vel_center(mode=mode)
        Rvir=self.R_vir(cen=pos_center)
        innerpart=self.PT[filt.Sphere(radius=0.1*Rvir,cen=pos_center)]
        with innerpart.immediate_mode:
            if alignwith in ['all','total','All','Total']:
                angmom = (innerpart['mass'].reshape((len(innerpart), 1)) *
                np.cross(innerpart['pos']-pos_center, innerpart['vel']-vel_center)).sum(axis=0).view(np.ndarray)
            elif alignwith in ['DM','dm','darkmatter','Darkmatter']:
                angmom = (innerpart.dm['mass'].reshape((len(innerpart.dm), 1)) *
                np.cross(innerpart.dm['pos']-pos_center, innerpart.dm['vel']-vel_center)).sum(axis=0).view(np.ndarray)
            elif alignwith in ['star','s','Star']:
                angmom = (innerpart.s['mass'].reshape((len(innerpart.s), 1)) *
                np.cross(innerpart.s['pos']-pos_center, innerpart.s['vel']-vel_center)).sum(axis=0).view(np.ndarray)
            elif alignwith in ['gas','g','Gas']:
                angmom = (innerpart.g['mass'].reshape((len(innerpart.g), 1)) *
                np.cross(innerpart.g['pos']-pos_center, innerpart.g['vel']-vel_center)).sum(axis=0).view(np.ndarray)
            elif alignwith in ['baryon','baryonic']:
                angmom1 = (innerpart.g['mass'].reshape((len(innerpart.g), 1)) *
                np.cross(innerpart.g['pos']-pos_center, innerpart.g['vel']-vel_center)).sum(axis=0).view(np.ndarray)
                angmo2 = (innerpart.s['mass'].reshape((len(innerpart.s), 1)) *
                np.cross(innerpart.s['pos']-pos_center, innerpart.s['vel']-vel_center)).sum(axis=0).view(np.ndarray)
                angmom=angmom1+angmo2

        trans =calc_faceon_matrix(angmom)
        if shift:
            phimax=None
            if 'phi' in self.PT.keys():
                R200=self.R_vir(cen=pos_center,overden=200)
                phimax=self.PT[filt.Annulus(r1=R200,r2=Rvir,cen=pos_center,)]['phi'].mean()
            self.PT.ancestor.shift(pos=pos_center,vel=vel_center,phi=phimax)
            self._transform(trans)
        else:
            self._transform(trans)
    def R_vir(self,cen=None,overden=178):
        """
        Calculates the virial radius of the halo.

        Parameters:
        -----------
        cen : array-like 
            The center position to use.
        overden : float
            The overdensity criterion.

        Returns:
        --------
        R : simarray
            The virial radius.
        """
        R=virial_radius(self.PT,cen=cen,overden=overden,rho_def='critical')
        return R
    
    def wrap(self,boxsize=None, convention='center'):
        self.PT.ancestor.wrap(boxsize,convention)

    def rotate_x(self,angle):
        self.PT.ancestor.rotate_x(angle)

    def rotate_y(self,angle):
        self.PT.ancestor.rotate_y(angle)

    def rotate_z(self,angle):
        self.PT.ancestor.rotate_z(angle)

    def transform(self, matrix):
        self.PT.ancestor._transform(matrix)

    def __check_paticles(self):
        if len(self.PT)>0:
            return False
        else:
            return True

    def _transform(self, matrix):
        self.PT.ancestor._transform(matrix)

    def __repr__(self):
       return "<Halo \"" + self.PT.ancestor.filename + "\" HaloID=" + str(self.GC['HaloID']) + ">"

class halos:
    def __init__(self, snaps):
        """
        Initializes the halos object.

        Parameters:
        -----------
        snaps : object
            An object that contains snapshot properties.
        """
        self.__snaps = snaps  
        self._data = {}

    def keys(self):
        """
        Returns the keys of the halos dictionary.
        
        Returns:
        --------
        keys : list
            List of keys in the _data dictionary.
        """
        return self._data.keys()

    def clear(self):
        """
        Clears all halo data from the dictionary.
        """
        self._data.clear()

    def update(self):
        """
        Updates the PT attribute of all Halo objects in the _data dictionary.
        """
        for i in self._data.keys():
            self._data[i].PT=self._generate_value(i) 


    def GC(self,key):
        """
        Returns a combined SimArray of a specific parameter from all loaded halos.

        Parameters:
        -----------
        key : str
            The key in the group catalog.

        Returns:
        --------
        ku : SimArray
            A SimArray combining the values of the specified key from all halos.
        """
        k=[]
        for i in self.__snaps.GC_loaded_Halo:
            k.append(self[str(i)].GC[key])
        ku=SimArray(np.array(k),k[0].units)
        return ku

    def _load_GC(self):
        """
        Loads the group catalog data for all Halo objects in the _data dictionary.
        """
        for i in self._data.keys():
            self._data[i]._load_GC()

    
    def _generate_value(self, key):
        """
        Generates a Halo object from a given key.

        Parameters:
        -----------
        key : str
            The Halo ID.

        Returns:
        --------
        property_value : object
            The halo properties or None if the Halo ID is invalid.
        """
        if (int(key) < self.__snaps.properties['Halos_total']) and (int(key)> -1):
            if 'HaloID' in self.__snaps.keys():
                property_value = self.__snaps[np.where(self.__snaps['HaloID']==int(key))]
            else:
                property_value = self.__snaps[slice(0,0)]

            if len(property_value)==0:
                property_value = self.__snaps[slice(0,0)]
            
            property_value._descriptor='Halo'+'_'+key
            return property_value
        else:
            print('InputError: '+key+', HaloID should be a non-negative integer and '+'\n'+
                  'less than the total number of Halos in this snapshot :',self.__snaps.properties['Halos_total'])
            return None

    def __getitem__(self, key):
        """
        Retrieves a Halo object from the _data dictionary.

        Parameters:
        -----------
        key : int, str, list, or np.ndarray
            The key or list of keys to retrieve.

        Returns:
        --------
        Halo object or None
            The requested Halo object or None if not found.
        """
        if isinstance(key,list) or isinstance(key,np.ndarray):
            key=np.array(key).flatten()
            for j in key:
                el=j
                el = str(el)  
                if el not in self._data:
                    wrd=self._generate_value(el)
                    if wrd is not None:
                        self._data[el] = Halo(wrd)
            return 
        if isinstance(key, int):
            key = str(key)  
        if key not in self._data:
            self._data[key] = Halo(self._generate_value(key))
        if self._data[key] is None:
            del self._data[key]
            return None
        return self._data[key]

    def __setitem__(self, key, value):
        """
        Sets a Halo object in the _data dictionary.

        Parameters:
        -----------
        key : str
            The key for the Halo object.
        value : Halo
            The Halo object to set.
        """
        self._data[key] = value

    def __repr__(self):
        """
        Returns a string representation of the halos object.

        Returns:
        --------
        repr : str
            A string representation of the halos object.
        """
        return "<Halos \"" + self.__snaps.filename + "\" num=" + str(len(self._data)) + ">"
    def physical_units(self):
        """
        Converts the group catalog units of all Halo objects in the _data dictionary to physical units.
        """
        for i in self.keys():
            self._data[i].GC_physical_units()