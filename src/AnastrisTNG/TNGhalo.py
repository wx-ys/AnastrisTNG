'''
Halo data processing
'''

from functools import reduce

import numpy as np
from pynbody.array import SimArray

from AnastrisTNG.TNGgroupcat import haloproperties
from AnastrisTNG.TNGsnapshot import Basehalo

class Halo(Basehalo):
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
        Basehalo.__init__(self, simarray)
        self._descriptor = 'Halo' + '_' + simarray.filename.split('_')[-1]
        self.GC['HaloID'] = int(simarray.filename.split('_')[-1])
        if len(simarray) > 0:
            self.load_GC()

    def load_GC(self):
        """
        Loads the group catalog data for this halo and updates its properties.
        """
        proper = haloproperties(
            self.GC['filedir'], self.GC['Snapshot'], self.GC['HaloID']
        )
        self.GC.update(proper)
        for i in self.GC:
            if isinstance(self.GC[i], SimArray):
                self.GC[i].sim = self.ancestor

    def __repr__(self):
        return (
            "<Halo \""
            + self.ancestor.filename
            + "\" HaloID="
            + str(self.GC['HaloID'])
            + ">"
        )


class Halos:
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
        for i in self._data:
            self._data[i] = Halo(self._generate_value(i))

    def GC(self, key, IDs=None):
        """
        Returns a combined SimArray of a specific parameter from all loaded halos.

        Parameters:
        -----------
        key : str
            The key in the group catalog.
        IDs :
            [haloid]

        Returns:
        --------
        ku : SimArray
            A SimArray combining the values of the specified key from all halos.
        """
        if IDs is None:
            k = [self[str(i)].GC[key] for i in self.__snaps.GC_loaded_Halo]
        else:
            k = [self[str(i)].GC[key] for i in IDs]
        ku = SimArray(np.array(k), k[0].units)
        ku.sim = self.__snaps
        return ku

    def load_GC(self):
        """
        Loads the group catalog data for all Halo objects in the _data dictionary.
        """
        for i in self._data:
            self._data[i].load_GC()

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
        if (int(key) < self.__snaps.properties['Halos_total']) and (int(key) > -1):
            if 'HaloID' in self.__snaps:
                property_value = self.__snaps[
                    np.where(self.__snaps['HaloID'] == int(key))
                ]
            else:
                property_value = self.__snaps[slice(0, 0)]

            if len(property_value) == 0:
                property_value = self.__snaps[slice(0, 0)]

            property_value._descriptor = 'Halo' + '_' + key
            return property_value
        else:
            print(
                'InputError: '
                + key
                + ', HaloID should be a non-negative integer and '
                + '\n'
                + 'less than the total number of Halos in this snapshot :',
                self.__snaps.properties['Halos_total'],
            )
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
        if isinstance(key, list) or isinstance(key, np.ndarray):
            key = np.array(key).flatten()
            for j in key:
                el = j
                el = str(el)
                if el not in self._data:
                    wrd = self._generate_value(el)
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
        return (
            "<Halos \"" + self.__snaps.filename + "\" num=" + str(len(self._data)) + ">"
        )

    def physical_units(self):
        """
        Converts the group catalog units of all Halo objects in the _data dictionary to physical units.
        """
        for i in self.keys():
            self._data[i].GC_physical_units()
