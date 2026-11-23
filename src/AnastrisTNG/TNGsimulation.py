'''
Load illustrisTNG data and process it.
'''

from copy import deepcopy

from pynbody.snapshot import SimSnap, new
from pynbody import filt
import h5py

from AnastrisTNG.illustris_python.snapshot import (
    getSnapOffsets,
    loadSubset,
    loadSubhalo,
    snapPath,
    loadOriginalZoom
)
from AnastrisTNG.TNGsnapshot import *
from AnastrisTNG.TNGunits import *
from AnastrisTNG.TNGmergertree import *
from AnastrisTNG.Anatools import ang_mom
from AnastrisTNG.TNGload import _TNGSimSnap, particle_id_dtype
from AnastrisTNG.TNGsubhalo import Subhalos, Subhalo
from AnastrisTNG.TNGhalo import Halos, Halo
from AnastrisTNG.TNGgroupcat import (
    get_eps_Mdm, get_parttype, haloproperties, halosproperty, loadSingle, local_snapshot_header,
    simsnap_cover, simsnap_merge, subhaloproperties, subhalosproperty,
)
from AnastrisTNG.pytreegrav import Accel, Potential, PotentialTarget, AccelTarget


_BASE_FIELDS = ("Coordinates", "Velocities", "Masses", "ParticleIDs")


class Snapshot(_TNGSimSnap):
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

    @classmethod
    def from_api(cls, run, snap, *, cache_dir=None, local_dir=None, api_key=None,
                 progress="auto", offline=False):
        """Open an on-demand Illustris, TNG or TNG-Cluster snapshot with a persistent cache.
        Set TNG_API_KEY for uncached online reads.
        local_dir may point to an existing simulation output directory.
        """
        from pathlib import Path
        from .TNGapi import ApiSource

        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "AnastrisTNG"
        source = ApiSource(run, snap, cache_dir, local_dir, api_key, progress, offline)
        return cls(source.local or str(source.root), snap, _source=source)

    def __init__(
        self,
        BasePath: str,
        Snap: int,
        *, _source=None,
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
        self._source = _source
        super().__init__()
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
        self._filename = self.properties['run']

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
        if _source is not None:
            from .TNGapi import ApiLazyContext
            self._loaded_index = {fam: np.empty(0, dtype=np.int64) for fam in _source.fields}
            self._lazy_ctx = ApiLazyContext(self, _source)
            self.loadable_parameters = {
                "snapshots": {fam: {k: parameter_all_Description("snapshots", fam, k) for k in keys}
                              for fam, keys in _source.fields.items()},
                "groupcatalogs": _source.catalog_fields,
            }
            return
        with h5py.File(snapPath(BasePath, Snap), 'r') as __file_pa:
            __halo_pa = list(loadSingle(BasePath, Snap, haloID=1).keys())
            __subhalo_pa = list(loadSingle(BasePath, Snap, subhaloID=1).keys())
            self.loadable_parameters = {
            'groupcatalogs': {
                'halo': {
                x: parameter_all_Description('groupcatalogs', 'halo', x)
                for x in __halo_pa
                },
                'subhalo': {
                # Retrieve a description of the parameter for the specified group catalog and subhalo.
                x: parameter_all_Description('groupcatalogs', 'subhalo', x)
                for x in __subhalo_pa
                },
            },
            'snapshots': {},
            }
            if 'PartType0' in __file_pa:
                self.loadable_parameters['snapshots']['gas'] = {
                x: parameter_all_Description('snapshots', 'gas', x)
                for x in list(__file_pa['PartType0'].keys())
            }
            if 'PartType4' in __file_pa:
                self.loadable_parameters['snapshots']['star'] = {
                x: parameter_all_Description('snapshots', 'star', x)
                for x in list(__file_pa['PartType4'].keys())
            }
            if 'PartType1' in __file_pa:
                self.loadable_parameters['snapshots']['dm'] = {
                x: parameter_all_Description('snapshots', 'dm', x)
                for x in list(__file_pa['PartType1'].keys())
            }
            if 'PartType5' in __file_pa:
                self.loadable_parameters['snapshots']['bh'] = {
                x: parameter_all_Description('snapshots', 'bh', x)
                for x in list(__file_pa['PartType5'].keys())
            }
            # ---- lazy-load 支持 ----
            from AnastrisTNG.TNGload import _LazyCtx
            self._loaded_index = {f: np.array([], dtype=np.int64) for f in ('dm', 'gas', 'star', 'bh')}
            self._lazy_ctx = _LazyCtx(self)
            # 每 family -> HDF 字段集合(pynbody 名)。与上面 loadable_parameters 扫描同源,复用 __file_pa。
            # 注意:一律用字符串 ('dm'/'gas'/'star'/'bh') 作 dict key(与 _loaded_index 一致)。
            fam_hdf_keys = {}
            fam_loadable = {}
            for fam, ptnum in (('gas', 0), ('star', 4), ('dm', 1), ('bh', 5)):
                group = 'PartType' + str(ptnum)
                if group not in __file_pa:
                    continue
                keys = list(__file_pa[group].keys())
                fam_hdf_keys[fam] = set(keys)
                fam_loadable[fam] = [snapshot_pa_name(k) for k in keys]
            try:
                self._snap_offsets = getSnapOffsets(BasePath, Snap, 1, 'Group')['snapOffsets']
            except Exception:
                # 读不到 offsets 时仍允许构造 Snapshot(仅失去 lazy 能力),不破坏非 lazy 用法。
                self._snap_offsets = None
            self._lazy_ctx.set_snapshot_meta(BasePath, Snap, self._snap_offsets, fam_loadable, fam_hdf_keys)
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

    def physical_units(self, persistent: bool = True):
        """Convert particles with pynbody and update the loaded TNG catalogues."""
        super().physical_units(*self.properties['baseunits'], persistent=persistent)
        self.subhalos.physical_units()
        self.halos.physical_units()
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
            _source=self._source,
        )
    def halo_evolution(
        self,
        haloID,
        physical_units: bool = True,
        *, fields=None,
    ):

        return halo_evolution(
            self.properties['filedir'],
            self.properties['Snapshot'],
            haloID,
            physical_units,
            fields=fields, _source=self._source,
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
            _source=self._source,
        )

    def halos_GC(self, fields: List[str]):
        if isinstance(fields, str):
            fields = list([fields])
        if isinstance(fields, list):
            halosGC = (self._source.catalog_columns("Halo", fields) if self._source is not None
                       else halosproperty(self.properties['filedir'], self.snapshot, fields))
            if len(fields) == 1:
                try:
                    hGC = SimArray(halosGC, groupcat_units(fields[0]))
                    hGC.sim =self
                    return hGC
                except:
                    return halosGC
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
            subhaloGC = (self._source.catalog_columns("Subhalo", fields) if self._source is not None
                         else subhalosproperty(self.properties['filedir'], self.snapshot, fields))
            if len(fields) == 1:
                try:
                    subGC = SimArray(subhaloGC, groupcat_units(fields[0]))
                    subGC.sim = self
                    return subGC
                except:
                    return subhaloGC
            
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

    def _catalog_for_merge(self, kind, number):
        total = self.properties["Halos_total" if kind == "Halo" else "Subhalos_total"]
        if not 0 <= number < total:
            raise ValueError(f"No {kind} {number} in snapshot {self.snapshot}")
        if self._source is not None:
            return self._source.catalog_arrays(kind, number)
        read = haloproperties if kind == "Halo" else subhaloproperties
        return read(self.properties["filedir"], self.snapshot, number)

    def _commit_particle_merge(self, merged):
        simsnap_cover(self, merged)
        ind = np.empty(len(self), dtype='int8')
        for index, family in enumerate(self.families()):
            ind[self._get_family_slice(family)] = index
        self._family_index_cached = ind
        self.subhalos.update()
        self.halos.update()

    def load_halo(self, haloID: int):
        """
        Load a specific halo's particles and group catalog (GC) data into the snapshot.
        """
        if not isinstance(haloID, (int, np.integer)):
            raise TypeError("haloID should be int")
        haloID = int(haloID)
        if haloID in self.__PT_loaded['Halo']:
            print(haloID, ' was already loaded into this Snapshot')
            return
        if self._canloadPT:
            catalog = self._catalog_for_merge("Halo", haloID)
            first_sub, n_sub = int(catalog["GroupFirstSub"]), int(catalog["GroupNsubs"])
            if n_sub < 0 or (n_sub and not 0 <= first_sub < first_sub + n_sub <= self.properties["Subhalos_total"]):
                raise ValueError(f"Invalid subhalo membership for Halo {haloID}")
            self.load_particle_para['particle_field'] = self.load_particle_para[
                'particle_field'
            ].lower()
            self.load_particle_para['particle_field'] = get_parttype(
                self.load_particle_para['particle_field']
            )
            f = self.load_particle(ID=haloID, groupType='Halo', decorate=False, lazy=False)
            # del self[self['HaloID']==haloID] del the loaded subhalo in this halo or overwrite it ?
            if 'HaloID' in self:
                subhaloIDover = set(self[self['HaloID'] == haloID]['SubhaloID'])
            else:
                subhaloIDover = set([])
            if -1 in subhaloIDover:
                subhaloIDover.remove(-1)
            if len(subhaloIDover) > 0:
                for sub_id in subhaloIDover:
                    self._match_subhalo(f, sub_id)
                keep_mask = self['HaloID'] != haloID
                f1 = self[keep_mask]
                # 掩码子视图不继承 _loaded_index;在合并前按掩码重排/重映射已加载粒子的原始行号,
                # 否则之前加载(非本 halo)粒子的行索引会在 simsnap_merge 中丢失。
                f1._loaded_index = {}
                for fam_obj in self.ancestor.families():
                    fam_str = getattr(fam_obj, 'name', fam_obj)
                    f1._loaded_index[fam_str] = np.asarray(
                        self._loaded_index.get(fam_str, np.array([], dtype=np.int64)),
                        dtype=np.int64,
                    )[keep_mask[self._get_family_slice(fam_obj)]]
                fmerge = simsnap_merge(f1, f)
            else:
                fmerge = simsnap_merge(self, f)
            self._commit_particle_merge(fmerge)
            self.halos._record_loaded(haloID, catalog)
            self.__PT_loaded['Halo'].add(haloID)
            self.__GC_loaded['Halo'].add(haloID)
            self.__PT_loaded['Subhalo'].update(range(first_sub, first_sub + n_sub))
        else:
            print('The pos and vel of the snapshot particles')
            print('are not in the coordinate system in the original box.')
            print('New particles can not be loaded')

    def match_subhalo(self, subhaloID: int):
        """
        Match particles to a specific subhalo based on the subhalo's ID.
        """

        self._match_subhalo(self, subhaloID)

    def _match_subhalo(self, target, subhaloID):
        families = [fam.name for fam in target.families()]
        if self._source is not None:
            data = self._source.particles("Subhalo", subhaloID,
                                          {fam: ["ParticleIDs"] for fam in families})
        else:
            data = {fam: loadSubhalo(self.properties['filedir'], self.snapshot,
                                     subhaloID, fam, fields=['ParticleIDs']) for fam in families}
        for fam, fields in data.items():
            if "ParticleIDs" in fields:
                particles = target[get_family(fam)]
                ids = (np.asarray(particles["iord"]), fields["ParticleIDs"])
                dtype = particle_id_dtype(ids)
                mask = np.isin(*(values.astype(dtype, copy=False) for values in ids))
                particles["SubhaloID"][mask] = subhaloID

    def load_subhalo(self, subhaloID: int):
        """
        Load particles and properties associated with a specific subhalo into the snapshot.
        """

        if not isinstance(subhaloID, (int, np.integer)):
            raise TypeError("subhaloID should be int")
        subhaloID = int(subhaloID)
        if subhaloID in self.__PT_loaded['Subhalo']:
            print(subhaloID, ' was already loaded into this Snapshot')
            print('So here update subhalos about this subhalo')
            if not len(self) or subhaloID in self['SubhaloID']:
                self.subhalos.update()
            else:
                print('No particle has this SubhaloID.')
                print('So here match this subhalo particle and modify their SubhaloID')
                self.match_subhalo(subhaloID)
            return
        if self._canloadPT:
            catalog = self._catalog_for_merge("Subhalo", subhaloID)
            host_id = int(catalog["SubhaloGrNr"])
            if not 0 <= host_id < self.properties["Halos_total"]:
                raise ValueError(f"Invalid FoF host {host_id} for Subhalo {subhaloID}")
            self.load_particle_para['particle_field'] = self.load_particle_para[
                'particle_field'
            ].lower()
            self.load_particle_para['particle_field'] = get_parttype(
                self.load_particle_para['particle_field']
            )
            f = self.load_particle(ID=subhaloID, groupType='Subhalo', decorate=False, lazy=False)
            if len(f):
                f["HaloID"][:] = host_id
            fmerge = simsnap_merge(self, f)
            self._commit_particle_merge(fmerge)
            self.subhalos._record_loaded(subhaloID, catalog)
            self.__PT_loaded['Subhalo'].add(subhaloID)
            self.__GC_loaded['Subhalo'].add(subhaloID)
        else:
            print('The pos and vel of the snapshot particles')
            print('are not in the coordinate system in the original box.')
            print('New particles can not be loaded')

    def loadable_keys(self, fam=None):
        return self._lazy_ctx.loadable_keys(fam)

    def _load_array(self, name, fam=None):
        self._lazy_ctx.load_array(name, fam)

    def read_particle_field(self, ID, family, field, groupType="Subhalo"):
        """Read an independent column in snapshot units, e.g. (12, "star", "mass").

        Uses the same file order and disk cache as a standalone load_particle
        object. A cutout requests the field itself on a cache miss.
        """
        from .TNGapi import CATALOG, PART_TYPE, ZOOM_PART_TYPE

        if groupType not in (*CATALOG, "Zoom") or not isinstance(ID, (int, np.integer)) or ID < 0:
            raise ValueError("Expected Halo, Subhalo or Zoom and a non-negative integer ID")
        if groupType == "Zoom" and self.properties["run"] != "TNG-Cluster":
            raise ValueError("Zoom selections require TNG-Cluster")
        families = ZOOM_PART_TYPE if groupType == "Zoom" else PART_TYPE
        if family not in families:
            raise ValueError(f"Expected one of {', '.join(families)}")
        names = self.loadable_parameters["snapshots"].get(family, {})
        dm_mass = family == "dm" and field in ("mass", "Masses")
        key = "Masses" if dm_mass else next((k for k in names if field in (k, snapshot_pa_name(k))), None)
        if family in ("dmlowres", "tracer"):
            key = {"mass": "Masses", "vel": "Velocities", "pos": "Coordinates",
                   "iord": "TracerID" if family == "tracer" else "ParticleIDs"}.get(field, field)
        if key is None:
            raise KeyError(f"Unknown snapshot field {family}.{field}")
        if self._source is None and groupType == "Zoom":
            base = self.properties["filedir"]
            if dm_mass:
                offsets = getSnapOffsets(base, self.snapshot, ID, "Group")
                original = loadSingle(base, self.snapshot, haloID=ID)["GroupOrigHaloID"]
                i = np.flatnonzero(offsets["HaloIDs"] == original).item()
                count = int(offsets["GroupsTotalLengthByType"][i, 1] + offsets["OuterFuzzTotalLengthByType"][i, 1])
            else:
                data = loadOriginalZoom(base, self.snapshot, ID, families[family], [key])
                count = int(data["count"])
                if count:
                    values = data[key]
        elif self._source is None:
            subset = getSnapOffsets(self.properties["filedir"], self.snapshot, ID, CATALOG[groupType])
            count = int(subset["lenType"][families[family]])
        else:
            counts = self._source.particle_counts(groupType, ID)
            count = int(counts[families[family]])
        if not count:
            raise ValueError(f"{groupType} {ID} has no {family} particles")
        unit = units.NoUnit() if family == "tracer" else snapshot_units(key)
        if dm_mass:
            values = np.full(count, self.properties["Mdm"].in_units(unit))
        elif self._source is None and groupType != "Zoom":
            values = loadSubset(self.properties["filedir"], self.snapshot, family, key, subset=subset, sq=True)
        elif self._source is not None:
            values = self._source.particles(groupType, ID, {family: [key]}, counts=counts)[family][key]
        if family == "tracer" and key == "FluidQuantities":
            return values  # The five columns have different physical units.
        result = SimArray(values, unit)
        result.sim = self
        return result

    def load_particle(
        self, ID: int, groupType: str = 'Subhalo', decorate=True, *, lazy=True, **kwargs
    ) -> SimSnap:
        '''
        ID: int, halo or subhalo id
        groupType: str, 'Halo' or 'Subhalo', 'Zoom' for TNG-Cluster
        Standalone API objects load particle fields on access. Use lazy=False
        to prefetch Basefields and the configured family fields together.

        '''
        
        if self._source is not None and groupType not in ("Halo", "Subhalo", "Zoom"):
            raise ValueError("Expected Halo, Subhalo or Zoom")
        if groupType == 'Zoom' and self._source is None:
            subset = getSnapOffsets(self.properties['filedir'], self.snapshot, ID, 'Group')

            # identify original halo ID and corresponding index
            halo = loadSingle(self.properties['filedir'], self.snapshot, haloID=ID,)
            assert 'GroupOrigHaloID' in halo, 'Error: loadOriginalZoom() only for the TNG-Cluster simulation.'
            orig_index = np.where(subset['HaloIDs'] == halo['GroupOrigHaloID'])[0][0]
            
            lenType = subset['GroupsTotalLengthByType'][orig_index, :] +subset['OuterFuzzTotalLengthByType'][orig_index, :]
            order = kwargs.get('order', self.load_particle_para['particle_field'])
            f = new(
                dm=int(lenType[1]),
                star=int(lenType[4]),
                gas=int(lenType[0]),
                bh=int(lenType[5]),
                order=order,
                class_=_TNGSimSnap,
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
                        loaddata = loadOriginalZoom(self.properties['filedir'], self.snapshot, ID, party, self.load_particle_para[party + '_fields'])
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
                        loaddata = loadOriginalZoom(self.properties['filedir'], self.snapshot, ID, party, self.load_particle_para[party + '_fields'])
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
                        loaddata =loadOriginalZoom(self.properties['filedir'], self.snapshot, ID, party, self.load_particle_para[party + '_fields'])
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
                        loaddata = loadOriginalZoom(self.properties['filedir'], self.snapshot, ID, party, self.load_particle_para[party + '_fields'])
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
                        
            f.properties = deepcopy(self.properties)
            for i in f.properties:
                if isinstance(f.properties[i], SimArray):
                    f.properties[i].sim = f
            f._filename = self.filename + '_' + groupType + '_' + str(ID)
            if decorate:
                return Halo(f)
            return f
            
            
            
        families = self.load_particle_para['particle_field'].split(',')
        order = kwargs.get('order', ','.join(families))
        deferred = self._source is not None and (lazy or groupType == "Zoom")
        if deferred:
            from .TNGapi import ApiObjectContext
            f = ApiObjectContext.new_snapshot(self._source, groupType, ID, families, order)
        else:
            f = self._read_particle_fields(ID, groupType, families, order)
        f._source = self._source
        f.properties = deepcopy(self.properties)
        for value in f.properties.values():
            if isinstance(value, SimArray):
                value.sim = f
        f._filename = self.filename + '_' + groupType + '_' + str(ID)
        if deferred:
            base = self.load_particle_para['Basefields']
            explicit = [] if lazy and tuple(base) == _BASE_FIELDS else list(base)
            prefetch = {fam.name: list(dict.fromkeys(explicit + self.load_particle_para[fam.name + '_fields']))
                        for fam in f.families()}
            f._lazy_ctx.read_fields({fam: keys for fam, keys in prefetch.items() if keys})
        if decorate and groupType in ('Halo', 'Subhalo', 'Zoom'):
            if groupType == 'Subhalo':
                return Subhalo(f, defer_catalog=deferred)
            return Halo(f, defer_catalog=groupType == 'Zoom')
        return f

    def _read_particle_fields(self, ID, groupType, families, order):
        from .TNGload import _LazySnap
        fields = {fam: list(dict.fromkeys(self.load_particle_para['Basefields'] +
                                          self.load_particle_para[fam + '_fields']))
                  for fam in families}
        dm_mass = 'Masses' in fields.get('dm', [])
        if dm_mass:
            fields['dm'].remove('Masses')
        pt = {'dm': 1, 'star': 4, 'gas': 0, 'bh': 5}
        if self._source is None:
            subset = getSnapOffsets(self.properties['filedir'], self.snapshot, ID,
                                    'Group' if groupType == 'Halo' else 'Subhalo')
            counts = {fam: int(subset['lenType'][pt[fam]]) for fam in families}
            data = {fam: loadSubset(self.properties['filedir'], self.snapshot, fam,
                                    names, subset=subset) for fam, names in fields.items()
                    if counts[fam] and names}
        else:
            # Eager/merge readers track membership by ID, including for a
            # custom Basefields list. The shared cache supplies existing IDs.
            data = self._source.particles(groupType, ID, {fam: names + ["ParticleIDs"]
                                                          for fam, names in fields.items()})
            counts = {fam: len(data[fam]['ParticleIDs']) if fam in data else 0 for fam in families}

        f = new(order=order, class_=_LazySnap,
                **{fam: counts.get(fam, 0) for fam in pt})
        other = 'Subhalo' if groupType == 'Halo' else 'Halo'
        for fam, count in counts.items():
            if not count:
                continue
            particles = f[get_family(fam)]
            for key, array in data.get(fam, {}).items():
                if key != 'count':
                    particles[snapshot_pa_name(key)] = SimArray(array, snapshot_units(key))
            if fam == 'dm' and dm_mass:
                unit = snapshot_units('Masses')
                particles['mass'] = SimArray(np.full(count, self.properties['Mdm'].in_units(unit)), unit)
            particles[groupType + 'ID'] = np.full(count, ID, dtype=np.int32)
            particles[other + 'ID'] = np.full(count, -1, dtype=np.int32)
            if self._source is None:
                offset = int(subset['offsetType'][pt[fam]])
                f._loaded_index[fam] = np.arange(offset, offset + count, dtype=np.int64)
            else:
                origin = int(ID) if groupType == 'Halo' else -1 - int(ID)
                f._loaded_index[fam] = np.full(count, origin, dtype=np.int64)
        f._lazy_ctx = self._lazy_ctx.bind_to(f)
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

        SnapshotHeader = self._source.header() if self._source else local_snapshot_header(BasePath, Snap)
        self.properties = SimDict()
        self.properties['filedir'] = BasePath
        if self._source is None:
            self.properties['filepath'] = BasePath
        else:
            self.properties['run'] = self._source.run
        self.properties['read_Snap_properties'] = SnapshotHeader
        self.properties['eps'], self.properties['Mdm'] = get_eps_Mdm(self, SnapshotHeader)
        for i in self.properties:
            if 'sim' in dir(self.properties[i]):
                self.properties[i].sim = self
        self.properties['Snapshot'] = Snap

    def __set_load_particle(self):
        pa = {}
        pa['particle_field'] = 'dm,star,gas,bh'
        pa['Basefields'] = list(_BASE_FIELDS)
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
