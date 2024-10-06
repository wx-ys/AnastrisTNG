'''
Subhalo data processing
'''

from functools import reduce

import numpy as np
from pynbody.array import SimArray

from AnastrisTNG.TNGgroupcat import subhaloproperties
from AnastrisTNG.TNGsnapshot import Basehalo

class Subhalo(Basehalo):
    """
    Represents a single subhalo in the simulation.

    This class contains information about the particles of the subhalo and its corresponding group catalog data.
    It also includes functions to compute properties specific to this subhalo.

    Attributes:
    ----------
    PT : SimArray
        The particles of the subhalo. Detailed information about this can be found at
        https://www.tng-project.org/data/docs/specifications/#sec1.
    GC : SimDict
        The group catalog for this subhalo. Detailed information about this can be found at
        https://www.tng-project.org/data/docs/specifications/#sec2.

    Parameters:
    ----------
    simarray : SimArray
        An object containing the particle data for the subhalo.

    """

    def __init__(self, simarray):
        SubhaloID = int(simarray.filename.split('_')[-1])

        if (
            (hasattr(simarray.ancestor, 'PT_loaded_Subhalo'))
            and (SubhaloID in simarray.ancestor.PT_loaded_Subhalo)
            and len(simarray) == 0
        ):
            simarray.ancestor.match_subhalo(int(simarray.filename.split('_')[-1]))
            Basehalo.__init__(
                self,
                simarray.ancestor[simarray.ancestor['SubhaloID'] == SubhaloID],
            )
        else:
            Basehalo.__init__(self, simarray)
        self._descriptor = 'Subhalo' + '_' + simarray.filename.split('_')[-1]
        self.GC['SubhaloID'] = SubhaloID
        if len(simarray) > 0:
            self.load_GC()

    def load_GC(self):
        """
        Loads the group catalog data for this halo and updates its properties.
        """
        proper = subhaloproperties(
            self.GC['filedir'], self.GC['Snapshot'], self.GC['SubhaloID']
        )
        self.GC.update(proper)
        for i in self.GC:
            if isinstance(self.GC[i], SimArray):
                self.GC[i].sim = self.ancestor

    def __repr__(self):
        return (
            "<Subhalo \""
            + self.ancestor.filename
            + "\" SubhaloID="
            + str(self.GC['SubhaloID'])
            + ">"
        )


class Subhalos:

    def __init__(self, snaps):
        """
        Initializes the subhalos object.

        Parameters:
        -----------
        snaps : object
            An object that contains snapshot properties.
        """
        self.__snaps = snaps.ancestor
        self._data = {}

    def keys(self):
        """
        Returns the keys of the stored subhalo data.

        Returns:
        --------
        dict_keys
            Keys representing the subhalo IDs stored in `_data`.
        """
        return self._data.keys()

    def clear(self):
        """
        Clears all stored subhalo data.
        """
        self._data.clear()

    def update(self):
        """
        Updates the properties of each stored subhalo.
        """
        for i in self._data:
            self._data[i] = Subhalo(self._generate_value(i))

    def GC(self, key):
        """
        Retrieves the group catalog data for a specific subhalo parameter.

        Parameters:
        -----------
        key : str
            The key to access in the group catalog data.

        Returns:
        --------
        SimArray
            A SimArray containing the group catalog data for the specified key.
        """
        k = [self[str(i)].GC[key] for i in self.__snaps.GC_loaded_Subhalo]
        ku = SimArray(np.array(k), k[0].units)
        ku.sim = self.__snaps
        return ku

    def load_GC(self):
        """
        Loads the group catalog data for all stored subhalos.
        """
        for i in self._data:
            self._data[i].load_GC()

    def _generate_value(self, key):
        """
        Generates a subhalo object for a given key.

        Parameters:
        -----------
        key : str
            The ID of the subhalo to generate.

        Returns:
        --------
        Subhalo or None
            The Subhalo object if the key is valid, otherwise None.
        """
        if (int(key) < self.__snaps.properties['Subhalos_total']) and (int(key) > -1):

            if 'SubhaloID' in self.__snaps:
                property_value = self.__snaps[
                    np.where(self.__snaps['SubhaloID'] == int(key))
                ]
            else:
                property_value = self.__snaps[slice(0, 0)]

            if len(property_value) == 0:
                property_value = self.__snaps[slice(0, 0)]

            property_value._descriptor = 'Subhalo' + '_' + key
            return property_value
        else:
            print(
                'InputError: '
                + key
                + ', SubhaloID should be a non-negative integer and '
                + '\n'
                + 'less than the total number of Subhalos in this snapshot :',
                self.__snaps.properties['Subhalos_total'],
            )
            return None

    def __getitem__(self, key):
        """
        Retrieves a subhalo object by its key.

        Parameters:
        -----------
        key : int, str, list, or np.ndarray
            The key(s) representing the subhalo ID(s).

        Returns:
        --------
        Subhalo or None
            The Subhalo object if found, otherwise None.
        """
        if isinstance(key, list) or isinstance(key, np.ndarray):
            key = np.array(key).flatten()
            for j in key:
                el = str(j)
                if el not in self._data:
                    wrd = self._generate_value(el)
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
        """
        Sets or updates a subhalo object in the data store.

        Parameters:
        -----------
        key : str
            The ID of the subhalo.
        value : Subhalo
            The Subhalo object to store.
        """
        self._data[key] = value

    def __repr__(self):
        """
        Represents the subhalos object as a string.

        Returns:
        --------
        str
            A string representation of the subhalos object.
        """

        return (
            "<Subhalos \""
            + self.__snaps.filename
            + "\" num="
            + str(len(self._data))
            + ">"
        )

    def physical_units(self):
        """
        Converts the group catalog data of each stored subhalo to physical units.
        """
        for i in list(self.keys()):
            self._data[i].GC_physical_units()
