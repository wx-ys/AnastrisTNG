import numpy as np
from TNGgroupcat import subhaloproperties
from pynbody.simdict import SimDict
from pynbody.array import SimArray
from pynbody import units
from functools import reduce




class Subhalo:
    def __init__(self,simarray):
        self.PT=simarray
        self.GC=SimDict()
        self.GC.update(simarray.properties)
        self.GC['SubhaloID']=int(simarray.filename.split('_')[-1])
        if self.GC['SubhaloID'] in simarray.ancestor.PT_loaded_Subhalo and len(simarray)==0:
            simarray.ancestor.match_subhalo(int(simarray.filename.split('_')[-1]))
            self.PT=simarray.ancestor[simarray.ancestor['SubhaloID']==self.GC['SubhaloID']]
            self.PT._descriptor='Subhalo'+'_'+simarray.filename.split('_')[-1]
        




    def _load_GC(self):
        proper=subhaloproperties(self.GC['filedir'],
                                 self.GC['Snapshot'],
                                 self.GC['SubhaloID'])
        self.GC.update(proper)
        for i in self.GC.keys():
            if isinstance(self.GC[i],SimArray):
                self.GC[i].sim=self.PT.ancestor

    def GC_physical_units(self,):


        dims =self.PT.ancestor.properties['baseunits']+[units.a,units.h]
        urc=len(dims)-2
        for k in list(self.GC.keys()):

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
                        
    def wrap(self,boxsize=None, convention='center'):
        self.PT.ancestor.wrap(boxsize,convention)

    def rotate_x(self,angle):
        self.PT.ancestor.rotate_x(angle)

    def rotate_y(self,angle):
        self.PT.ancestor.rotate_y(angle)

    def rotate_z(self,angle):
        self.PT.ancestor.rotate_z(angle)

    def transform(self, matrix):
        self.PT.ancestor.transform(matrix)

    def _transform(self, matrix):
        self.PT.ancestor._transform(matrix)
    def __repr__(self):
        return "<Subhalo \"" + self.PT.ancestor.filename + "\" SubhaloID=" + str(self.GC['SubhaloID']) + ">"



class subhalos:
    def __init__(self, snaps):
        self.__snaps = snaps.ancestor  
        self._data = {}

    def keys(self):
        return self._data.keys()

    def clear(self):
        self._data.clear()

    def update(self):
        for i in self._data.keys():
            self._data[i].PT=self._generate_value(i) 
    

    def _load_GC(self):
        for i in self._data.keys():
            self._data[i]._load_GC()

    def _generate_value(self, key):
        if (int(key) < self.__snaps.properties['Subhalos_total']) and (int(key)> -1):
            


            if 'SubhaloID' in self.__snaps.keys():
                property_value = self.__snaps[np.where(self.__snaps['SubhaloID']==int(key))]
            else:
                property_value = self.__snaps[slice(0,0)]

            if len(property_value)==0:
                property_value = self.__snaps[slice(0,0)]

            property_value._descriptor='Subhalo'+'_'+key
            return property_value
        else:
            print('InputError: '+key+', SubhaloID should be a non-negative integer and '+'\n'+
                  'less than the total number of Subhalos in this snapshot :',self.__snaps.properties['Subhalos_total'])
            return None

    def __getitem__(self, key):
        if isinstance(key,list) or isinstance(key,np.ndarray):
            key=np.array(key).flatten()
            for j in key:
                el=j
                el = str(el)  
                if el not in self._data:
                    wrd=self._generate_value(el)
                    if wrd is not None:
                        self._data[el] = Subhalo(wrd)

            return 



        if isinstance(key, int):
            key = str(key)  
        if key not in self._data:
            self._data[key] = Subhalo(self._generate_value(key))
        if self._data[key] is None:
            del self._data[key]
            return None
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value

    def __repr__(self):

        return "<Subhalos \"" + self.__snaps.filename + "\" num=" + str(len(self._data)) + ">"
    
    def physical_units(self):
        for i in list(self.keys()):
            self._data[i].GC_physical_units()