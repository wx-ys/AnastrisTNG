'''
Halo data processing
'''

import numpy as np
from pynbody.array import SimArray

from AnastrisTNG.TNGgroupcat import haloproperties
from AnastrisTNG.TNGsnapshot import Basehalo, _CatalogueViews

class Halo(Basehalo):
    """
    Represents a single halo in the simulation.

    This class contains information about the particles of the halo and its corresponding group catalog data.
    It also includes functions to compute properties specific to this halo.

    Attributes:
    ----------
    GC : mutable mapping
        The group catalog for this halo. Detailed information about this can be found at
        https://www.tng-project.org/data/docs/specifications/#sec2.

    Parameters:
    ----------
    simarray : SimArray
        An object containing the particle data for the halo.

    """

    _catalog_kind = "Halo"

    def __init__(self, simarray, *, defer_catalog=False):
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
        if defer_catalog:
            self._gc_pending = True
        elif len(simarray) > 0:
            self.load_GC()

    def load_GC(self):
        """
        Loads the group catalog data for this halo and updates its properties.
        """
        source = getattr(self.ancestor, '_source', None)
        if source is not None:
            proper = source.catalog_arrays("Halo", self._GC["HaloID"])
        else:
            proper = haloproperties(
                self._GC['filedir'], self._GC['Snapshot'], self._GC['HaloID']
            )
        self._set_catalog(proper)

    def __repr__(self):
        return (
            "<Halo \""
            + self.ancestor.filename
            + "\" HaloID="
            + str(self._GC['HaloID'])
            + ">"
        )


class Halos(_CatalogueViews):
    _view_type = Halo
    _length_field = "GroupLenType"

    def __init__(self, snaps):
        """
        Initializes the halos object.

        Parameters:
        -----------
        snaps : object
            An object that contains snapshot properties.
        """
        self.__snaps = snaps
        super().__init__()

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
