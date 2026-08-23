"""Lazy-load machinery for TNG snapshots (mirrors pynbody GadgetHDFSnap).

Both the merged Snapshot and the snapshot returned by load_particle are
SimSnap subclasses that override _load_array / loadable_keys. _LazyCtx is the
shared read logic keyed on per-family original file row indices (_loaded_index).
"""

import numpy as np
from pynbody import units
from pynbody.snapshot import SimSnap

from AnastrisTNG.illustris_python.snapshot import loadSubset, snapPath
from AnastrisTNG.illustris_python.util import partTypeNum
from AnastrisTNG.TNGunits import snapshot_pa_name, snapshot_units

_FAMILY_ORDER = ['dm', 'gas', 'star', 'bh']


def _split_runs(index_array):
    """Split an array of global row indices into contiguous runs (order-preserving)."""
    if len(index_array) == 0:
        return []
    spl = np.where(np.diff(index_array) != 1)[0] + 1
    return list(np.split(index_array, spl))


def parttype_for_family(family):
    """Return the illustris parttype string for a pynbody Family object (or a parttype string).

    pynbody Family objects have `.name` == 'dm'/'gas'/'star'/'bh'; a raw string
    argument is passed through (Family objects come from the framework, strings
    from user-facing calls like `loadable_keys('gas')`).
    """
    return getattr(family, 'name', family)


def pynbody_to_hdf(hdf_keys):
    """Build a pynbody-name -> HDF-name reverse mapping from HDF field names."""
    m = {}
    for k in hdf_keys:
        m[snapshot_pa_name(k)] = k
    return m


def first_hdf_group_for(basepath, snap, parttype):
    """Open (and return) the first h5py group 'PartTypeN' that exists across chunks."""
    import h5py
    n = partTypeNum(parttype)
    gname = "PartType" + str(n)
    i = 0
    while True:
        f = h5py.File(snapPath(basepath, snap, i), 'r')
        if gname in f:
            return f[gname]
        f.close()
        i += 1


class _LazyCtx:
    """Read logic for a lazily-loaded snapshot. `owner` is the SimSnap being read."""

    def __init__(self, owner):
        self._owner = owner
        self._basepath = None
        self._snap = None
        self._snap_offsets = None          # shape (6, nfiles)
        self._loadable_family_keys = {}    # Family -> [pynbody names]
        self._family_hdf_keys = {}         # Family -> set(HDF names)
        self._pynbody_to_hdf = {}

    def set_snapshot_meta(self, basepath, snap, snap_offsets,
                          loadable_family_keys, family_hdf_keys):
        self._basepath = basepath
        self._snap = snap
        self._snap_offsets = snap_offsets
        self._loadable_family_keys = loadable_family_keys
        self._family_hdf_keys = family_hdf_keys
        all_hdf = set()
        for keys in family_hdf_keys.values():
            all_hdf |= keys
        self._pynbody_to_hdf = pynbody_to_hdf(all_hdf)

    def bind_to(self, new_owner):
        c = _LazyCtx(new_owner)
        c.set_snapshot_meta(self._basepath, self._snap, self._snap_offsets,
                            self._loadable_family_keys, self._family_hdf_keys)
        return c

    @staticmethod
    def _families():
        return _FAMILY_ORDER

    def _fam_name(self, fam):
        """Normalize a Family object (from the framework) or a parttype string to a string."""
        return getattr(fam, 'name', fam)

    def _hdf_name(self, name):
        return self._pynbody_to_hdf.get(name, name)

    def _field_exists_for_family(self, fam, hdf_name):
        keys = self._family_hdf_keys.get(fam, set())
        return hdf_name in keys

    def _field_units(self, hdf_name):
        try:
            return snapshot_units(hdf_name)
        except KeyError:
            return units.no_unit

    def _field_spec(self, fam, hdf_name):
        grp = first_hdf_group_for(self._basepath, self._snap, parttype_for_family(fam))
        dset = grp[hdf_name]
        dtype = dset.dtype
        ndim = 1 if len(dset.shape) <= 1 else int(dset.shape[1])
        return dtype, ndim

    def _gather(self, fam, hdf_name, index_array):
        if len(index_array) == 0:
            return None
        pt = partTypeNum(parttype_for_family(fam))
        out = []
        for run in _split_runs(index_array):
            ot = np.zeros(6, dtype=np.int64)
            ot[pt] = int(run[0])
            lt = np.zeros(6, dtype=np.int64)
            lt[pt] = int(len(run))
            subset = {'offsetType': ot, 'snapOffsets': self._snap_offsets, 'lenType': lt}
            data = loadSubset(self._basepath, self._snap, parttype_for_family(fam),
                              [hdf_name], subset=subset)
            if hdf_name in data:
                out.append(data[hdf_name])
        if not out:
            raise OSError("No data for %r in family %r" % (hdf_name, fam))
        return np.concatenate(out)

    def loadable_keys(self, fam=None):
        if fam is None:
            res = []
            for _, v in self._loadable_family_keys.items():
                res += v
            return sorted(set(res))
        return self._loadable_family_keys.get(self._fam_name(fam), [])

    def load_array(self, name, fam=None):
        hdf_name = self._hdf_name(name)
        fam_str = self._fam_name(fam)          # None, or 'dm'/'gas'/'star'/'bh' (string key)
        if fam_str is None:
            target = self._owner
            fams = self._families()            # ['dm','gas','star','bh']
        else:
            target = self._owner[fam]          # fam is a Family (framework) — indexing works
            fams = [fam_str]

        dtype = ndim = None
        for fami in fams:
            if fami in self._family_hdf_keys and self._field_exists_for_family(fami, hdf_name):
                dtype, ndim = self._field_spec(fami, hdf_name)
                break
        if dtype is None:
            raise OSError("No such array on disk: %s" % name)

        target._create_array(name, ndim, dtype=dtype)
        u = self._field_units(hdf_name)
        if u is not units.no_unit:
            target[name].units = u

        for fami in fams:
            if fami not in self._owner._loaded_index:
                continue
            idx = self._owner._loaded_index[fami]
            if not self._field_exists_for_family(fami, hdf_name):
                continue
            vals = self._gather(fami, hdf_name, idx)
            if vals is None:
                continue
            if fam_str is None:
                self._owner[name][self._owner._family_slice[_family_obj(fami)]] = vals
            else:
                target[name][:] = vals


def _family_obj(name):
    from pynbody.family import get_family
    return get_family(name)


class _LazySnap(SimSnap):
    """A lightweight SimSnap that lazily reads particle fields from the TNG file."""

    def __init__(self):
        super().__init__()
        self._loaded_index = {f: np.array([], dtype=np.int64) for f in _FAMILY_ORDER}
        self._lazy_ctx = None

    def _load_array(self, name, fam=None):
        self._lazy_ctx.load_array(name, fam)

    def loadable_keys(self, fam=None):
        return self._lazy_ctx.loadable_keys(fam)
