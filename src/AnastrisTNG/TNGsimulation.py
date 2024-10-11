'''
Load illustrisTNG data and process it.
'''

from functools import reduce

from pynbody.snapshot import SimSnap, new
from pynbody import filt
import h5py

from AnastrisTNG.illustris_python.snapshot import (
    getSnapOffsets,
    loadSubset,
    loadSubhalo,
    snapPath,
)
from AnastrisTNG.TNGsnapshot import *
from AnastrisTNG.TNGunits import *
from AnastrisTNG.TNGmergertree import *
from AnastrisTNG.Anatools import ang_mom
from AnastrisTNG.TNGsubhalo import Subhalos, Subhalo
from AnastrisTNG.TNGhalo import Halos, Halo
from AnastrisTNG.TNGgroupcat import loadSingle, halosproperty, subhalosproperty, get_eps_Mdm, get_parttype,simsnap_cover, simsnap_merge
from AnastrisTNG.pytreegrav import Accel, Potential, PotentialTarget, AccelTarget


class Snapshot(SimSnap):
    """
    This class represents a snapshot of simulated cosmological data, containing information about
    the snapshot itself, as well as data on halos and subhalos.

    The Snapshot class inherits from SimSnap and provides additional functionality specific to
    handling snapshot data from simulations, including particle data , group catalopg data and cosmological parameters.

    Attributes:
    -----------
    properties : simdict
        Contains various properties of the snapshot, including file path, run information,
        and cosmological parameters.
    subhalos : Subhalos
        An instance representing the subhalos within the snapshot.
    halos : Halos
        An instance representing the halos within the snapshot.
    """

    def __init__(
        self,
        BasePath: str,
        Snap: int,
    ):
        """
        Initializes the Snapshot object with the provided base path and snapshot number.

        Parameters:
        -----------
        BasePath : str
            The base directory path where the snapshot data is stored.
        Snap : int
            The snapshot number corresponding to the desired snapshot in the simulation series.
        """
        SimSnap.__init__(self)
        self._num_particles = 0
        self._filename = "<created>"
        self._create_arrays(["pos", "vel"], 3)
        self._create_arrays(["mass"], 1)
        self._family_slice[get_family('dm')] = slice(0, 0)
        self._family_slice[get_family('star')] = slice(0, 0)
        self._family_slice[get_family('gas')] = slice(0, 0)
        self._family_slice[get_family('bh')] = slice(0, 0)
        self._decorate()
        self.__set_Snapshot_property(BasePath, Snap)
        self.properties['filedir'] = BasePath
        self.properties['filepath'] = BasePath
        self._filename = self.properties['run']

        self.properties['eps'], self.properties['Mdm'] = get_eps_Mdm(self)
        self.properties['baseunits'] = [
            units.Unit(x) for x in ('kpc', 'km s^-1', 'Msol')
        ]
        self.properties['staunit'] = [
            'nH',
            'Halpha',
            'em',
            'ne',
            'temp',
            'mu',
            'c_n_sq',
            'p',
            'cs',
            'c_s',
            'acc',
            'phi',
            'age',
            'tform',
            'SubhaloPos',
            'sfr',
        ]
        for i in self.properties:
            if isinstance(self.properties[i], SimArray):
                self.properties[i].sim = self
        self._filename = self._filename + '_' + 'snapshot' + str(self.snapshot)

        self.__set_load_particle()
        self.subhalos = Subhalos(self)
        self.halos = Halos(self)
        self._canloadPT = True
        self.__PT_loaded = {'Halo': set(), 'Subhalo': set()}

        self.__GC_loaded = {'Halo': set(), 'Subhalo': set()}

        self.__pos = SimArray([0.0, 0.0, 0.0], units.kpc)
        self.__pos.sim = self
        self.__vel = SimArray([0.0, 0.0, 0.0], units.km / units.s)
        self.__vel.sim = self
        self.__phi = SimArray([0.0, 0.0, 0.0], units.km**2 / units.s**2)
        self.__phi.sim = self
        self.__acc = SimArray([0.0, 0.0, 0.0], units.km / units.s**2)
        self.__acc.sim = self
        __file_pa = h5py.File(snapPath(BasePath, Snap), 'r')
        __halo_pa = list(loadSingle(BasePath, Snap, haloID=1).keys())
        __subhalo_pa = list(loadSingle(BasePath, Snap, subhaloID=1).keys())
        self.loadable_parameters = {
            'groupcatalogs': {
                'halo': {
                    x: parameter_all_Description('groupcatalogs', 'halo', x)
                    for x in __halo_pa
                },
                'subhalo': {
                    x: parameter_all_Description('groupcatalogs', 'subhalo', x)
                    for x in __subhalo_pa
                },
            },
            'snapshots': {
                'gas': {
                    x: parameter_all_Description('snapshots', 'gas', x)
                    for x in list(__file_pa['PartType0'].keys())
                },
                'star': {
                    x: parameter_all_Description('snapshots', 'star', x)
                    for x in list(__file_pa['PartType4'].keys())
                },
                'dm': {
                    x: parameter_all_Description('snapshots', 'dm', x)
                    for x in list(__file_pa['PartType1'].keys())
                },
                'bh': {
                    x: parameter_all_Description('snapshots', 'bh', x)
                    for x in list(__file_pa['PartType5'].keys())
                },
            },
        }
        __file_pa.close()

    @staticmethod
    def parameter_describe(table: str, contents: str, parameters: str) -> str:
        '''
        Input:
            table: str,
                'snapshots' or 'groupcatalogs'.
            contents: str,
                'halo' or 'subhalo' for table = 'groupcatalogs'.
                'Gas', 'DM', 'Stars', 'BH' for table = 'snapshots'.
            parameters: str,
                specified parameter.

        Output: str
            description of the parameter.
        '''

        return parameter_all_Description(table, contents, parameters)

    def physical_units(self, persistent: bool = False):
        """
        Convert the units of the simulation arrays and properties to physical units.

        This method adjusts the units of the simulation data arrays and properties based on the
        base units of the simulation. It ensures that the data is consistent with physical units
        (e.g., kpc, Msol, km/s) and converts the data if necessary.

        Parameters:
        -----------
        persistent : bool, optional
            If True, the conversion to physical units will be persistent, meaning that
            future calculations and accesses will use these units by default. If False,
            the conversion is temporary (default is False).
        """

        dims = self.properties['baseunits'] + [units.a, units.h]
        urc = len(dims) - 2
        all = list(self._arrays.values())
        for x in self._family_arrays:
            if x in self.properties.get('staunit', []):
                continue
            else:
                all += list(self._family_arrays[x].values())

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

        self.subhalos.physical_units()
        self.halos.physical_units()
        if persistent:
            self._autoconvert = dims
        else:
            self._autoconvert = None
        self._canloadPT = False

    def galaxy_evolution(
        self,
        subID,
        fields: List[str] = ['SnapNum', 'SubfindID'],
        physical_units: bool = True,
    ):

        return galaxy_evolution(
            self.properties['filedir'],
            self.properties['Snapshot'],
            subID,
            fields,
            physical_units,
        )
    def halo_evolution(
        self,
        haloID,
        physical_units: bool = True,
    ):

        return halo_evolution(
            self.properties['filedir'],
            self.properties['Snapshot'],
            haloID,
            physical_units,
        )

    def merger_history(
        self,
        subID,
        fields: List[str] = ['SubfindID', 'SubhaloMassType', 'SnapNum'],
        minMassRatio: float = 1e-10,
        massPartType: str = 'stars',
        physical_units: bool = True,
    ) -> dict:
        return merger_history(
            self.properties['filedir'],
            self.properties['Snapshot'],
            subID,
            fields,
            minMassRatio,
            massPartType,
            physical_units,
        )

    def halos_GC(self, fields: List[str]):
        if isinstance(fields, str):
            fields = list([fields])
        if isinstance(fields, list):
            halosGC = halosproperty(
                self.properties['filedir'], self.properties['Snapshot'], fields
            )
            for i in halosGC:
                try:
                    halosGC[i] = SimArray(halosGC[i], groupcat_units(i))
                    halosGC[i].sim = self
                except:
                    continue
            return halosGC
        else:
            print('fields must be a str or list of the parameter name!')
            return

    def subhalos_GC(self, fields: List[str]):
        if isinstance(fields, str):
            fields = list([fields])
        if isinstance(fields, list):
            subhaloGC = subhalosproperty(
                self.properties['filedir'], self.properties['Snapshot'], fields
            )
            for i in subhaloGC:
                try:
                    subhaloGC[i] = SimArray(subhaloGC[i], groupcat_units(i))
                    subhaloGC[i].sim = self
                except:
                    continue
            return subhaloGC
        else:
            print('fields must be a str or list of the parameter name!')
            return

    def load_GC(self):
        """
        Load the group catalog (GC) data for halos and subhalos into the snapshot.
        """

        self.subhalos.load_GC()
        for i in self.subhalos.keys():
            self.__GC_loaded['Subhalo'].add(int(i))
        self.halos.load_GC()
        for i in self.halos.keys():
            self.__GC_loaded['Halo'].add(int(i))

    def load_halo(self, haloID: int):
        """
        Load a specific halo's particles and group catalog (GC) data into the snapshot.
        """
        if not (isinstance(haloID, int) or np.issubdtype(haloID, np.integer)):
            raise TypeError("haloID should be int")
        if haloID in self.__PT_loaded['Halo']:
            print(haloID, ' was already loaded into this Snapshot')
            return
        if self._canloadPT:
            self.load_particle_para['particle_field'] = self.load_particle_para[
                'particle_field'
            ].lower()
            self.load_particle_para['particle_field'] = get_parttype(
                self.load_particle_para['particle_field']
            )
            f = self.load_particle(ID=haloID, groupType='Halo', decorate=False)
            # del self[self['HaloID']==haloID] del the loaded subhalo in this halo or overwrite it ?
            if 'HaloID' in self:
                subhaloIDover = set(self[self['HaloID'] == haloID]['SubhaloID'])
            else:
                subhaloIDover = set([])
            if -1 in subhaloIDover:
                subhaloIDover.remove(-1)
            if len(subhaloIDover) > 0:
                fmerge = simsnap_merge(self[self['HaloID'] != haloID], f)
            else:
                fmerge = simsnap_merge(self, f)
            simsnap_cover(self, fmerge)
            for i in subhaloIDover:
                self.match_subhalo(i)
            ind = np.empty((len(self),), dtype='int8')
            for i, f in enumerate(self.ancestor.families()):
                ind[self._get_family_slice(f)] = i

            self._family_index_cached = ind
            self.subhalos.update()
            self.halos.update()
            self.halos[haloID].load_GC()
            self.__PT_loaded['Halo'].add(haloID)
            self.__GC_loaded['Halo'].add(haloID)
            if self.halos[haloID].GC['GroupFirstSub'] != -1:
                for i in range(
                    self.halos[haloID].GC['GroupFirstSub'],
                    self.halos[haloID].GC['GroupFirstSub']
                    + self.halos[haloID].GC['GroupNsubs'],
                ):
                    self.__PT_loaded['Subhalo'].add(i)
        else:
            print('The pos and vel of the snapshot particles')
            print('are not in the coordinate system in the original box.')
            print('New particles can not be loaded')

    def match_subhalo(self, subhaloID: int):
        """
        Match particles to a specific subhalo based on the subhalo's ID.
        """

        parID = np.array([])
        for ty in ['star', 'gas', 'dm', 'bh']:
            thiID = loadSubhalo(
                self.properties['filedir'],
                self.snapshot,
                subhaloID,
                ty,
                fields=['ParticleIDs'],
            )
            if isinstance(thiID, dict):
                continue
            parID = np.append(parID, thiID)
        parID.astype(np.uint64)
        self['SubhaloID'][np.isin(self['iord'], parID)] = subhaloID
        # self.subhalos.update()

    def load_subhalo(self, subhaloID: int):
        """
        Load particles and properties associated with a specific subhalo into the snapshot.
        """

        if not (isinstance(subhaloID, int) or np.issubdtype(subhaloID, np.integer)):
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
        if self._canloadPT:
            self.load_particle_para['particle_field'] = self.load_particle_para[
                'particle_field'
            ].lower()
            self.load_particle_para['particle_field'] = get_parttype(
                self.load_particle_para['particle_field']
            )
            f = self.load_particle(ID=subhaloID, groupType='Subhalo', decorate=False)

            fmerge = simsnap_merge(self, f)
            simsnap_cover(self, fmerge)

            ind = np.empty((len(self),), dtype='int8')
            for i, f in enumerate(self.ancestor.families()):
                ind[self._get_family_slice(f)] = i
            self._family_index_cached = ind

            self.subhalos[subhaloID].load_GC()
            self['HaloID'][self['SubhaloID'] == subhaloID] = self.subhalos[
                subhaloID
            ].GC['SubhaloGrNr']
            self.subhalos.update()
            self.halos.update()
            self.__PT_loaded['Subhalo'].add(subhaloID)
            self.__GC_loaded['Subhalo'].add(subhaloID)
        else:
            print('The pos and vel of the snapshot particles')
            print('are not in the coordinate system in the original box.')
            print('New particles can not be loaded')

    def load_particle(
        self, ID: int, groupType: str = 'Subhalo', decorate=True, **kwargs
    ) -> SimSnap:
        '''
        ID: int, halo or subhalo id
        groupType: str, 'Halo' or 'Subhalo'

        '''
        if groupType == 'Halo':
            subset = getSnapOffsets(
                self.properties['filedir'], self.snapshot, ID, 'Group'
            )
        else:
            subset = getSnapOffsets(
                self.properties['filedir'], self.snapshot, ID, 'Subhalo'
            )

        lenType = subset['lenType']
        order = kwargs.get('order', self.load_particle_para['particle_field'])
        f = new(
            dm=int(lenType[1]),
            star=int(lenType[4]),
            gas=int(lenType[0]),
            bh=int(lenType[5]),
            order=order,
        )

        for party in self.load_particle_para['particle_field'].split(","):
            if len(f[get_family(party)]) > 0:
                if len(self.load_particle_para[party + '_fields']) > 0:
                    self.load_particle_para[party + '_fields'] = list(
                        set(
                            self.load_particle_para[party + '_fields']
                            + self.load_particle_para['Basefields']
                        )
                    )
                else:
                    self.load_particle_para[party + '_fields'] = list.copy(
                        self.load_particle_para['Basefields']
                    )

                if party == 'dm':
                    if 'Masses' in self.load_particle_para[party + '_fields']:
                        self.load_particle_para[party + '_fields'].remove('Masses')
                    loaddata = loadSubset(
                        self.properties['filedir'],
                        self.snapshot,
                        party,
                        self.load_particle_para[party + '_fields'],
                        subset=subset,
                    )
                    for i in self.load_particle_para[party + '_fields']:
                        f.dm[snapshot_pa_name(i)] = SimArray(
                            loaddata[i], snapshot_units(i)
                        )
                    if 'Masses' in self.load_particle_para['Basefields']:
                        f.dm['mass'] = self.properties['Mdm'].in_units(
                            snapshot_units('Masses')
                        ) * np.ones(len(f.dm))
                        self.load_particle_para[party + '_fields'].append('Masses')
                    f.dm[groupType + 'ID'] = SimArray(
                        ID * np.ones(len(f.dm)).astype(np.int32)
                    )
                    if groupType == 'Halo':
                        f.dm['SubhaloID'] = SimArray(
                            -1 * np.ones(len(f.dm)).astype(np.int32)
                        )
                    else:
                        f.dm['HaloID'] = SimArray(
                            -1 * np.ones(len(f.dm)).astype(np.int32)
                        )

                if party == 'star':
                    loaddata = loadSubset(
                        self.properties['filedir'],
                        self.snapshot,
                        party,
                        self.load_particle_para[party + '_fields'],
                        subset=subset,
                    )
                    for i in self.load_particle_para[party + '_fields']:
                        f.s[snapshot_pa_name(i)] = SimArray(
                            loaddata[i], snapshot_units(i)
                        )
                    f.s[groupType + 'ID'] = SimArray(
                        ID * np.ones(len(f.s)).astype(np.int32)
                    )
                    if groupType == 'Halo':
                        f.s['SubhaloID'] = SimArray(
                            -1 * np.ones(len(f.s)).astype(np.int32)
                        )
                    else:
                        f.s['HaloID'] = SimArray(
                            -1 * np.ones(len(f.s)).astype(np.int32)
                        )

                if party == 'gas':
                    loaddata = loadSubset(
                        self.properties['filedir'],
                        self.snapshot,
                        party,
                        self.load_particle_para[party + '_fields'],
                        subset=subset,
                    )
                    for i in self.load_particle_para[party + '_fields']:
                        f.g[snapshot_pa_name(i)] = SimArray(
                            loaddata[i], snapshot_units(i)
                        )
                    f.g[groupType + 'ID'] = SimArray(
                        ID * np.ones(len(f.g)).astype(np.int32)
                    )
                    if groupType == 'Halo':
                        f.g['SubhaloID'] = SimArray(
                            -1 * np.ones(len(f.g)).astype(np.int32)
                        )
                    else:
                        f.g['HaloID'] = SimArray(
                            -1 * np.ones(len(f.g)).astype(np.int32)
                        )

                if party == 'bh':
                    loaddata = loadSubset(
                        self.properties['filedir'],
                        self.snapshot,
                        party,
                        self.load_particle_para[party + '_fields'],
                        subset=subset,
                    )
                    for i in self.load_particle_para[party + '_fields']:
                        f.bh[snapshot_pa_name(i)] = SimArray(
                            loaddata[i], snapshot_units(i)
                        )
                    f.bh[groupType + 'ID'] = SimArray(
                        ID * np.ones(len(f.bh)).astype(np.int32)
                    )
                    if groupType == 'Halo':
                        f.bh['SubhaloID'] = SimArray(
                            -1 * np.ones(len(f.bh)).astype(np.int32)
                        )
                    else:
                        f.bh['HaloID'] = SimArray(
                            -1 * np.ones(len(f.bh)).astype(np.int32)
                        )
        f.properties = self.properties.copy()
        f._filename = self.filename + '_' + groupType + '_' + str(ID)
        if decorate:
            if groupType == 'Halo':
                return Halo(f)
            if groupType == 'Subhalo':
                return Subhalo(f)
        return f

    def target_acceleration(self, targetpos: np.ndarray) -> SimArray:
        """
        Calculate the acceleration of specific position.
        """

        try:
            eps = self.properties.get('eps', 0)
        except:
            eps = 0
        if eps == 0:
            print('Calculate the gravity without softening length')
        accelr = AccelTarget(
            targetpos,
            self['pos'].view(np.ndarray),
            self['mass'].view(np.ndarray),
            np.repeat(eps, len(targetpos)).view(np.ndarray),
        )
        acc = SimArray(
            accelr, units.G * self['mass'].units / self['pos'].units / self['pos'].units
        )
        acc.sim = self
        return acc

    def target_potential(self, targetpos: np.ndarray) -> SimArray:
        """
        Calculate the potential of specificc position.
        """

        try:
            eps = self.properties.get('eps', 0)
        except:
            eps = 0
        if eps == 0:
            print('Calculate the gravity without softening length')
        pot = PotentialTarget(
            targetpos,
            self['pos'].view(np.ndarray),
            self['mass'].view(np.ndarray),
            np.repeat(eps, len(targetpos)).view(np.ndarray),
        )
        phi = SimArray(pot, units.G * self['mass'].units / self['pos'].units)
        phi.sim = self
        return phi

    def wrap(self, boxsize=None, convention='center'):

        super().wrap(boxsize, convention)
        self._canloadPT = False

        print('It involves a change of coordinates')
        print('Can\'t load new particles in this Snapshot')

    def check_boundary(self):
        """
        Check if any particle lay on the edge of the box.
        """
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

    def __repr__(self):
        return "<Snapshot \"" + self.filename + "\" len=" + str(len(self)) + ">"

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except:
            pass

        try:
            return self.properties[name]
        except:
            pass

        raise AttributeError(
            "%r object has no attribute %r" % (type(self).__name__, name)
        )

    def __set_Snapshot_property(self, BasePath: str, Snap: int):
        """
        Init properties (Simdict) from Base Path and Snap.
        """

        SnapshotHeader = loadHeader(BasePath, Snap)
        self.properties = SimDict()
        self.properties['read_Snap_properties'] = SnapshotHeader
        for i in self.properties:
            if 'sim' in dir(self.properties[i]):
                self.properties[i].sim = self
        self.properties['filedir'] = BasePath
        self.properties['Snapshot'] = Snap

    def __set_load_particle(self):
        pa = {}
        pa['particle_field'] = 'dm,star,gas,bh'
        pa['Basefields'] = ['Coordinates', 'Velocities', 'Masses', 'ParticleIDs']
        pa['star_fields'] = []
        pa['gas_fields'] = []
        pa['dm_fields'] = []
        pa['bh_fields'] = []
        self.load_particle_para = pa

    @property
    def status_loadPT(self):
        '''
        Check the ability of this snapshot to load new particles
        '''

        if self._canloadPT:
            return 'able'
        else:
            return 'locked'

    def shift(self, pos: SimArray = None, vel: SimArray = None, phi: SimArray = None):
        '''
        shift to the specific position
        then set its pos, vel, phi, acc to 0.
        '''
        self._canloadPT = False
        if pos is not None:
            self['pos'] -= pos
            self.__pos.convert_units(self['pos'].units)
            self.__pos += pos
        if vel is not None:
            self['vel'] -= vel
            self.__vel.convert_units(self['vel'].units)
            self.__vel += vel
        if (phi is not None) and ('phi' in self):
            self['phi'] -= phi
            self.__phi.convert_units(self['phi'].units)
            self.__phi += phi

        if 'acc' in self:
            theacc = self.target_acceleration(np.array([[0, 0, 0], pos]))[1]
            self['acc'] -= theacc
            self.__acc.convert_units(self['acc'].units)
            self.__acc += theacc

    def get_origin_inbox(self):

        return (
            self.__pos.copy(),
            self.__vel.copy(),
            self.__acc.copy(),
            self.__phi.copy(),
        )

    def vel_center(
        self, mode: str = 'ssc', pos: SimArray = None, r_cal='1 kpc'
    ) -> SimArray:
        '''
        The center velocity.
        Refer from https://pynbody.readthedocs.io/latest/_modules/pynbody/analysis/halo.html#vel_center

        ``mode`` used to cal center pos see ``center``
        ``pos``  Specified position.
        ``r_cal`` The size of the sphere to use for the velocity calculate

        '''
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
            print('May mis-centering! Try using other mode to get the center pos')
            raise ValueError("Insufficient particles around center to get velocity")

        vcen = (cen['vel'].transpose() * cen['mass']).sum(axis=1) / cen['mass'].sum()
        vcen.units = cen['vel'].units

        return vcen

    def center(self, mode: str = 'ssc'):
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

    def ang_mom_vec(self, alignwith: str = 'all', rmax=None):
        alignwith = alignwith.lower()
        if rmax == None:
            callan = self
        else:
            callan = self[filt.Sphere(rmax)]

        if alignwith in ['all', 'total']:
            angmom = ang_mom(callan)
        elif alignwith in ['dm', 'darkmatter']:
            angmom = ang_mom(callan.dm)
        elif alignwith in ['star', 's']:
            angmom = ang_mom(callan.s)
        elif alignwith in ['gas', 'g']:
            angmom = ang_mom(callan.g)
        elif alignwith in ['baryon', 'baryonic']:
            angmom1 = ang_mom(callan.s)
            angmo2 = ang_mom(callan.g)
            angmom = angmom1 + angmo2
        return angmom

    def face_on(self, **kwargs):
        mode = kwargs.get('mode', 'ssc')
        alignwith = kwargs.get('alignwith', 'all')
        rmax = kwargs.get('rmax', None)
        shift = kwargs.get('shift', True)
        alignmode = kwargs.get('alignmode', 'jc')

        self.check_boundary()
        pos_center = self.center(mode=mode)
        vel_center = self.vel_center(mode=mode)

        self.ancestor.shift(pos=pos_center, vel=vel_center)
        if alignmode == 'jc':
            angmom = self.ang_mom_vec(alignwith=alignwith, rmax=rmax)
        elif alignmode == 'jabs':
            angmom = self.ang_mom_vec(alignwith=alignwith, rmax=rmax)
        else:
            angmom = self.ang_mom_vec(alignwith=alignwith, rmax=rmax)

        trans = calc_faceon_matrix(angmom)
        if shift:
            self._transform(trans)
        else:
            self.ancestor.shift(pos=-pos_center, vel=-vel_center)
            self._transform(trans)

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
    def snapshot(self):
        return self.properties['Snapshot']
