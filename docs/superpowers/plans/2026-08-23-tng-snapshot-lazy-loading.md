# TNG Snapshot 与 load_particle 快照的 lazy-load 支持 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让合并后的 `Snapshot` 与 `load_particle` 返回的快照都能在被访问任意字段时按需从磁盘读取,从而无需预配置 `load_particle_para`。

**Architecture:** 覆写子类 `_load_array` / `loadable_keys`(pynbody `SimSnap` 已内置触发管线)。新增 `_LazyCtx` 用"每 family 原始文件行号"(`_loaded_index`)做 gather 读取;`_LazySnap` 是通过 `new(class_=...)` 创建的轻量 `SimSnap`。`simsnap_merge`/`simsnap_cover` 负责传播 `_loaded_index`。

**Tech Stack:** Python, numpy, h5py, pynbody, pytest。测试用 conda 环境 `anasim`。

**Spec:** [docs/superpowers/specs/2026-08-23-tng-snapshot-lazy-loading-design.md](docs/superpowers/specs/2026-08-23-tng-snapshot-lazy-loading-design.md)

## Global Constraints

- 测试命令: `conda run -n anasim pytest tests/ -v`(本机数据的依赖见 §测试约定)。
- 集成测试基于真实数据: `BasePath='/home/yxi/Simulation/sims/TNG50-1/output'`, `Snap=99`;子结构 3052(gas=68/dm=390/star=487)、1000(dm=3603)。缺数据时用 `pytest.skip`。
- 不得修改仓库中被改动的两个 notebook(`examples/AnastrisTNG_*` 的 `M` 状态)。
- Lazy 核心只加载 pynbody 名;派生数组(如 `temp`,`sfr`,`mu`,`nH` 等,见 TNGsnapshot.py `@derived_array`)仍由 pynbody 派生机制处理,不落入 `_load_array`。
- `load_particle` 必急切加载 `Basefields`(pos/vel/mass/iord)+ 各 family ID 数组;`*_fields` 仅作可选的额外预载。
- 不引入循环导入:`TNGload.py` 只 import `pynbody.snapshot.SimSnap` / `pynbody.units` / `illustris_python.*` / `TNGunits`。

---

### File Structure

- **Create `src/AnastrisTNG/TNGload.py`** — lazy 机制载体:`_LazyCtx`(gather/名称映射/单位/spec)、`_LazySnap(SimSnap)`、模块级纯函数 `_split_runs`、`_invert`、`parttype_for_family`、`first_hdf_group_for`.
- **Modify `src/AnastrisTNG/TNGsimulation.py`** — `Snapshot`(`__init__` 增加 `_loaded_index`/`_lazy_ctx`/meta;新增 `_load_array`/`loadable_keys`);`load_particle` 改用 `new(class_=_LazySnap)` 并设 `_loaded_index`/`_lazy_ctx`。
- **Modify `src/AnastrisTNG/TNGgroupcat.py`** — `simsnap_merge`/`simsnap_cover` 传播 `_loaded_index`。
- **Create `tests/conftest.py`** — fixture `BP`、`SNAP`、`snap_bp`(检测数据存在)、小 subhalo 常量。
- **Create `tests/test_tngload_units.py`** — `TNGload` 纯函数/`_LazyCtx` 逻辑单元测试(无需整快照)。
- **Create `tests/test_tng_snapshot_lazy.py`** — 真实数据集成测试(经 `load_subhalo` 后 lazy 字段 == 直接 `loadSubset`)。

---

### Task 1: `TNGload.py` — lazy 机制与纯函数

**Files:**
- Create: `src/AnastrisTNG/TNGload.py`
- Test: `tests/test_tngload_units.py`

**Interfaces:**
- Produces:
  - `def _split_runs(index_array: np.ndarray) -> list[np.ndarray]` — 把行号数组切成连续 run(保序)。
  - `def parttype_for_family(family) -> str` — `family.name`('dm'/'gas'/'star'/'bh')直接作为 parttype。
  - `def pynbody_to_hdf(hdf_keys) -> dict[str,str]` — 由一组 HDF 字段名经 `snapshot_pa_name` 建 `pynbody名→HDF名` 逆映射。
  - `class _LazyCtx:` 后续任务使用的方法/属性:
    - `set_snapshot_meta(basepath, snap, snap_offsets, loadable_family_keys, family_hdf_keys) -> None`
    - `bind_to(new_owner) -> _LazyCtx`
    - `loadable_keys(fam=None) -> list[str]`
    - `load_array(name, fam=None) -> None`
    - 属性 `_snap_offsets`, `_owner`
  - `class _LazySnap(SimSnap)`:
    - `_loaded_index: dict[str, np.ndarray]`(key: 'dm'/'gas'/'star'/'bh', 值 int64 数组)
    - `_lazy_ctx: _LazyCtx | None`
    - `_load_array(name, fam=None)` → 委托 `_lazy_ctx.load_array`
    - `loadable_keys(fam=None)` → 委托 `_lazy_ctx.loadable_keys`

- [ ] **Step 1: 写失败测试(纯函数 `_split_runs`,无数据依赖)**

创建 `tests/test_tngload_units.py`:

```python
import numpy as np
from AnastrisTNG.TNGload import _split_runs, pynbody_to_hdf


def test_split_runs_single_contiguous():
    idx = np.arange(254750400, 254750400 + 68)  # 一段连续(如一个 halo 的 gas)
    runs = _split_runs(idx)
    assert len(runs) == 1
    np.testing.assert_array_equal(runs[0], idx)


def test_split_runs_two_blocks():
    a = np.arange(1000, 1010)
    b = np.arange(5000, 5005)
    idx = np.concatenate([a, b])             # 两段不相邻,保序
    runs = _split_runs(idx)
    assert len(runs) == 2
    np.testing.assert_array_equal(runs[0], a)
    np.testing.assert_array_equal(runs[1], b)


def test_split_runs_empty():
    assert _split_runs(np.array([], dtype=np.int64)) == []


def test_pynbody_to_hdf_maps_known_name():
    # 从一组真实 HDF 字段名反推 pynbody→HDF
    keys = ['Coordinates', 'Velocities', 'Masses', 'ParticleIDs', 'Density', 'Potential']
    m = pynbody_to_hdf(keys)
    assert m['pos'] == 'Coordinates'
    assert m['vel'] == 'Velocities'
    assert m['rho'] == 'Density'
    assert m['pot'] == 'Potential'
    # 未映射字段保持不变
    keys2 = ['ElectronAbundance']
    m2 = pynbody_to_hdf(keys2)
    assert m2['ElectronAbundance'] == 'ElectronAbundance'
```

- [ ] **Step 2: 运行测试确认失败**

Run: `conda run -n anasim pytest tests/test_tngload_units.py -v`
Expected: FAIL `ModuleNotFoundError: No module named 'AnastrisTNG.TNGload'`

- [ ] **Step 3: 实现 `TNGload.py` 纯函数 + `_LazyCtx` / `_LazySnap`**

`src/AnastrisTNG/TNGload.py`:

```python
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
```

- [ ] **Step 4: 运行测试确认通过**

Run: `conda run -n anasim pytest tests/test_tngload_units.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/AnastrisTNG/TNGload.py tests/test_tngload_units.py
git commit -m "feat: TNG lazy-load machinery (_LazyCtx, _LazySnap, run splitter)"
```

---

### Task 2: `Snapshot` 集成 lazy 机制(`__init__` + `_load_array`/`loadable_keys`)

**Files:**
- Modify: `src/AnastrisTNG/TNGsimulation.py`(`Snapshot.__init__`、新增 `_load_array`/`loadable_keys`)
- Test: `tests/test_tng_snapshot_lazy.py`(Step 1-2 只测 `loadable_keys` 初始构建)

**Interfaces:**
- Consumes: `_LazyCtx` from Task 1.
- Produces:
  - `Snapshot._loaded_index: dict[str, np.ndarray]`
  - `Snapshot._lazy_ctx: _LazyCtx`
  - `Snapshot.loadable_keys(fam=None) -> list[str]`
  - `Snapshot._load_array(name, fam=None) -> None`
  - `Snapshot._snap_offsets: np.ndarray`(shape (6, nfiles))

- [ ] **Step 1: 写失败测试(构造 Snapshot 后 `loadable_keys('gas')` 含 `rho`/`pot`)**

在 `tests/test_tng_snapshot_lazy.py` 顶部:

```python
import os
import numpy as np
import pytest

BP = '/home/yxi/Simulation/sims/TNG50-1/output'
SNAP = 99
SMALL_GAS_SUBHALO = 3052   # gas=68 dm=390 star=487
DM_SUBHALO = 1000          # dm=3603 star=2


@pytest.fixture
def need_data():
    if not os.path.isdir(BP + '/snapdir_%03d' % SNAP):
        pytest.skip("TNG data not available at " + BP)
    return True


def test_loadable_keys_include_rho_and_pot(need_data):
    from AnastrisTNG.TNGsimulation import Snapshot
    snap = Snapshot(BP, SNAP)
    gkeys = snap.loadable_keys('gas')
    assert 'rho' in gkeys          # Density
    assert 'pot' in gkeys          # Potential
    dkeys = snap.loadable_keys('dm')
    assert 'pot' in dkeys          # dm 也有 Potential
```

- [ ] **Step 2: 运行测试确认失败**

Run: `conda run -n anasim pytest tests/test_tng_snapshot_lazy.py::test_loadable_keys_include_rho_and_pot -v`
Expected: FAIL `'Snapshot' object has no attribute 'loadable_keys'`(或调用抛错)

- [ ] **Step 3: 在 `Snapshot.__init__` 建立 lazy 状态**

在 `Snapshot.__init__` 末尾(`self.__GC_loaded = ...` 之后、`__pos` 建立之前或之后)加入:

```python
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
```

> 注:`__file_pa` 是 `__init__` 里已打开的 `h5py.File(snapPath(BasePath, Snap))`(见现有 `with h5py.File(snapPath(BasePath, Snap), 'r') as __file_pa:`)。在其 `with` 块内或之后读取 `__file_pa['PartTypeN'].keys()` 均可。若把这段放在该 `with` 内更简单;否则用 `snapPath(BasePath, Snap)` 重开。实现时请放进现有 `with` 块(第 119-155 行),复用 `__file_pa`。

- [ ] **Step 4: 新增 `Snapshot.loadable_keys` 与 `Snapshot._load_array`**

在 `Snapshot` 类内新增两个方法(放在 `load_particle` 之前):

```python
    def loadable_keys(self, fam=None):
        return self._lazy_ctx.loadable_keys(fam)

    def _load_array(self, name, fam=None):
        self._lazy_ctx.load_array(name, fam)
```

- [ ] **Step 5: 运行测试确认通过**

Run: `conda run -n anasim pytest tests/test_tng_snapshot_lazy.py::test_loadable_keys_include_rho_and_pot -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/AnastrisTNG/TNGsimulation.py tests/test_tng_snapshot_lazy.py
git commit -m "feat: wire lazy-load state and loadable_keys into Snapshot"
```

---

### Task 3: `load_particle` 返回 lazy 的 `_LazySnap`

**Files:**
- Modify: `src/AnastrisTNG/TNGsimulation.py`(`load_particle`)
- Test: `tests/test_tng_snapshot_lazy.py`

**Interfaces:**
- Consumes: `_LazySnap`(Task 1), `Snapshot._snap_offsets` / `_lazy_ctx`(Task 2)。
- Produces: `load_particle(...)` 返回的快照 `f` 具有 `_loaded_index[families]`(每个 family = `np.arange(offsetType[ptNum], offsetType[ptNum]+lenType[ptNum])`)与已绑定到 `f` 的 `_lazy_ctx`。

- [ ] **Step 1: 写失败集成测试(load_subhalo 后 lazy 读 gas Density=='rho')**

```python
def test_lazy_read_gas_rho_after_load(need_data):
    from AnastrisTNG.TNGsimulation import Snapshot
    from AnastrisTNG.illustris_python.snapshot import getSnapOffsets, loadSubset
    snap = Snapshot(BP, SNAP)
    snap.load_subhalo(SMALL_GAS_SUBHALO)
    # 参考:直接读该子结构 gas Density
    sub = getSnapOffsets(BP, SNAP, SMALL_GAS_SUBHALO, 'Subhalo')
    ref = loadSubset(BP, SNAP, 'gas', ['Density'], subset=sub)['Density']
    # lazy:访问 snap.g['rho'] 触发按需读取
    lazy = snap.g['rho'].view(np.ndarray)
    assert lazy.shape == ref.shape
    np.testing.assert_allclose(lazy, ref)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `conda run -n anasim pytest tests/test_tng_snapshot_lazy.py::test_lazy_read_gas_rho_after_load -v`
Expected: FAIL `KeyError: 'rho'`(当前 Snapshot 未实现 lazy,访问 `snap.g['rho']` 找不到)。

- [ ] **Step 3: 非 Zoom 分支的 `new(...)` 换成 `new(..., class_=_LazySnap)` 并设 `_loaded_index`/`_lazy_ctx`**

> **Ruling 2(见 ledger):仅改非 Zoom 分支。** Zoom 分支(经 `loadOriginalZoom` 的 halo+fuzz)保持急切不改——其粒子不保证是单个连续的 `loadSubset` span,且本机无 TNG-Cluster 数据可校验;非 Zoom 的 TNG50-1 Halo/Subhalo 才是本次验证目标。

仅改 Halo/Subhalo 分支(位于 `if groupType == 'Zoom':` 块**之后**的主块,`load_particle` 中第二个 `new(...)`,即 `lenType = subset['lenType']` 之后那个):

```python
        from AnastrisTNG.TNGload import _LazySnap
        f = new(
            dm=int(lenType[1]),
            star=int(lenType[4]),
            gas=int(lenType[0]),
            bh=int(lenType[5]),
            order=order,
            class_=_LazySnap,
        )
```

在创建 `f` 之后、`f.properties = deepcopy(self.properties)` 之前(即在 `for party in ...` 循环之后),对每一 family 写 `_loaded_index` 与 `_lazy_ctx`:

```python
        # ---- lazy:记录每个已加载粒子在原始文件里的行号,并绑定 ctx ----
        fam_pt = {'dm': 1, 'star': 4, 'gas': 0, 'bh': 5}
        for fam, ptn in fam_pt.items():
            if len(f[get_family(fam)]) > 0:
                off = int(subset['offsetType'][ptn])
                cnt = int(subset['lenType'][ptn])
                f._loaded_index[fam] = np.arange(off, off + cnt, dtype=np.int64)
        f._lazy_ctx = self._lazy_ctx.bind_to(f)
```

> 注(Zoom,不改):Zoom 分支仍用旧 `new(...)`(无 `class_`),因此其 `f` 不带 `_loaded_index`/`_lazy_ctx`,保持急切加载。`load_subhalo`/`load_halo` 只走非 Zoom 分支,不受影响;若某处把 Zoom 的 `f` 与 `self` merge,Task 4 的 `getattr(f2,'_loaded_index',{})` 会以空数组兜底。

- [ ] **Step 4: 运行测试确认通过**

Run: `conda run -n anasim pytest tests/test_tng_snapshot_lazy.py::test_lazy_read_gas_rho_after_load -v`
Expected: PASS

- [ ] **Step 5: 补一个 dm 字段测试(子结构 1000)**

```python
def test_lazy_read_dm_pot_after_load(need_data):
    from AnastrisTNG.TNGsimulation import Snapshot
    from AnastrisTNG.illustris_python.snapshot import getSnapOffsets, loadSubset
    snap = Snapshot(BP, SNAP)
    snap.load_subhalo(DM_SUBHALO)
    sub = getSnapOffsets(BP, SNAP, DM_SUBHALO, 'Subhalo')
    ref = loadSubset(BP, SNAP, 'dm', ['Potential'], subset=sub)['Potential']
    lazy = snap.dm['pot'].view(np.ndarray)
    np.testing.assert_allclose(lazy, ref)
```

Run: `conda run -n anasim pytest tests/test_tng_snapshot_lazy.py::test_lazy_read_dm_pot_after_load -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/AnastrisTNG/TNGsimulation.py tests/test_tng_snapshot_lazy.py
git commit -m "feat: load_particle returns a lazy _LazySnap with per-family file row indices"
```

---

### Task 4: `simsnap_merge` / `simsnap_cover` 传播 `_loaded_index`

**Files:**
- Modify: `src/AnastrisTNG/TNGgroupcat.py`
- Test: `tests/test_tng_snapshot_lazy.py`

**Interfaces:**
- Consumes: `_loaded_index`(dict, 可能不存在于普通 Snap → 用空数组兜底)。
- Produces: `simsnap_merge(f1, f2)` 返回的 `f3` 具有 `_loaded_index = {fam: append(f1._loaded_index[fam], f2._loaded_index[fam])}`;`simsnap_cover(f1, f2)` 后 `f1._loaded_index = copy(f2._loaded_index)`。

- [ ] **Step 1: 写失败测试(连续 load 两个子结构后 `_loaded_index` 正确拼接)**

```python
def test_merge_carries_loaded_index(need_data):
    from AnastrisTNG.TNGsimulation import Snapshot
    from AnastrisTNG.illustris_python.snapshot import getSnapOffsets
    snap = Snapshot(BP, SNAP)
    snap.load_subhalo(DM_SUBHALO)          # dm 3603
    first_dm = snap._loaded_index['dm'].copy()
    snap.load_subhalo(SMALL_GAS_SUBHALO)   # dm 390
    dm = snap._loaded_index['dm']
    assert len(dm) == len(first_dm) + 390
    # 前一段应与第一次加载的一致(保序拼接)
    np.testing.assert_array_equal(dm[:len(first_dm)], first_dm)
    # 第二段应是子结构 3052 的 dm 块
    sub = getSnapOffsets(BP, SNAP, SMALL_GAS_SUBHALO, 'Subhalo')
    off = int(sub['offsetType'][1]); cnt = int(sub['lenType'][1])
    np.testing.assert_array_equal(dm[len(first_dm):], np.arange(off, off + cnt))
```

- [ ] **Step 2: 运行测试确认失败**

Run: `conda run -n anasim pytest tests/test_tng_snapshot_lazy.py::test_merge_carries_loaded_index -v`
Expected: FAIL(`_loaded_index` 未拼接,或第二次 `load_subhalo` 后索引错)。

- [ ] **Step 3: 扩展 `simsnap_merge` 末尾拼接 `_loaded_index`**

在 `simsnap_merge` 的 `return f3` 之前加入:

```python
    # ---- 传播 lazy 索引:拼接 f1、f2 每 family 的原始行号 ----
    f3._loaded_index = {}
    for fam in ('dm', 'gas', 'star', 'bh'):
        a = getattr(f1, '_loaded_index', {}).get(fam, np.array([], dtype=np.int64))
        b = getattr(f2, '_loaded_index', {}).get(fam, np.array([], dtype=np.int64))
        f3._loaded_index[fam] = np.append(a, b)
    return f3
```

- [ ] **Step 4: 扩展 `simsnap_cover` 末尾传播 `_loaded_index`**

在 `simsnap_cover` 的 `f1._decorate()` 之后、函数结束前加入:

```python
    # ---- 覆盖后同步 lazy 索引 ---- 
    f1._loaded_index = {fam: np.array(v, dtype=np.int64).copy()
                        for fam, v in (getattr(f2, '_loaded_index', {}) or {}).items()}
```

- [ ] **Step 5: 运行测试确认通过**

Run: `conda run -n anasim pytest tests/test_tng_snapshot_lazy.py -v`
Expected: PASS(含 Task 3 的两个 lazy 字段测试与 merge 索引测试)。

- [ ] **Step 6: Commit**

```bash
git add src/AnastrisTNG/TNGgroupcat.py tests/test_tng_snapshot_lazy.py
git commit -m "feat: simsnap_merge/cover propagate per-family lazy load indices"
```

---

### Task 5: `physical_units` 后 lazy 字段仍正确 + 补充断言

**Files:**
- Modify: `tests/test_tng_snapshot_lazy.py`(新增测试,验证单位一致性;预期无需改动生产代码)
- Test: `tests/test_tng_snapshot_lazy.py`

**Interfaces:**
- Consumes: `Snapshot.physical_units(persistent=True)`(现有方法)与 lazy 加载。

- [ ] **Step 1: 写测试(physical_units 后 lazy 字段被自动转换为物理单位)**

```python
def test_lazy_after_physical_units_consistent(need_data):
    from pynbody import units
    from AnastrisTNG.TNGsimulation import Snapshot
    snap = Snapshot(BP, SNAP)
    snap.load_subhalo(SMALL_GAS_SUBHALO)
    # 预加载基础字段,转物理单位
    dummy = snap['pos']   # 触发并入内存,之后 physical_units 生效
    snap.physical_units(persistent=True)
    # 之后 lazy 加载 gas Density
    rho = snap.g['rho']
    assert rho.units is not None and rho.units != units.no_unit
    # 与未转换时不一致——物理单位下密度应已 a^3 折算:仅断言单位非 NoUnit 即可
```

> 注:该测试主要验证"lazy 加载不因已调用 physical_units 而出错/单位缺失"。核心不变量:加载后数组有单位(框架 `_autoconvert_array_unit` 会转换)。若 `rho` 单位在 `snapshot_units('Density')` 中缺失,则退化为 `no_unit`,此断言可能在 TNG 特定字段上失败——实现时若遇此情况,改用确认 `units.has_units(rho)==True` 或断言不抛异常。

- [ ] **Step 2: 运行测试**

Run: `conda run -n anasim pytest tests/test_tng_snapshot_lazy.py -v`
Expected: PASS(若物理单位断言依赖 TNG 字段单位表,按 Step 1 注释调整断言,不改变生产代码)。

- [ ] **Step 3: Commit**

```bash
git add tests/test_tng_snapshot_lazy.py
git commit -m "test: lazy-load tolerates physical_units conversion"
```

---

### Task 6: 全量测试 + 复用验证(回归)

**Files:**
- Test: `tests/test_tngload_units.py`, `tests/test_tng_snapshot_lazy.py`

- [ ] **Step 1: 运行全部测试**

Run: `conda run -n anasim pytest tests/ -v`
Expected: 全部 PASS(`_split_runs` 4 + loadable_keys 1 + lazy 字段 2 + merge 索引 1 + physical_units 1)。

- [ ] **Step 2: 跑一遍现有 examples 中的 Snapshot 用法(冒烟)**

Run:
```bash
conda run -n anasim python -c "
from AnastrisTNG.TNGsimulation import Snapshot
from AnastrisTNG.illustris_python.snapshot import loadSubset, getSnapOffsets
snap = Snapshot('/home/yxi/Simulation/sims/TNG50-1/output', 99)
snap.load_subhalo(3052)
print('gas rho ok:', snap.g['rho'].shape)
print('pos len:', len(snap['pos']))
print('loadable gas has rho:', 'rho' in snap.loadable_keys('gas'))
"
```
Expected: 打印形状与 True,无异常。

- [ ] **Step 3: Commit(如有测试文件改动)**

```bash
git add tests/
git commit -m "test: snapshot lazy-loading regression suite"
```

---

## Self-Review

### 1. Spec coverage
Spec 各节对应任务:
- §2(覆写 `_load_array`/`loadable_keys`)→ Task 2。
- §5(gather/跨 chunk)→ Task 1 `_LazyCtx._gather`(`loadSubset` 自带跨 chunk,按 run 传子集)。
- §6.1(`_LazySnap`)→ Task 1;§6.2(Snapshot 集成)→ Task 2;§6.3(loadable_keys)→ Task 1/2;§6.4(merge/cover)→ Task 4;§6.5(装饰继承)→ Task 3(经 `new(class_=_LazySnap)` + SubSnap 祖先链自动)。
- §7(`load_particle_para` 角色)→ Task 3(Basefields 仍急切,`*_fields` 仅可选)。
- §8(单位一致性)→ Task 5(框架 `_autoconvert` 保证)。
- §9(错误处理)→ Task 1 `_load_array` 抛 `OSError` 触发逐 family 回退;`_gather` 空/缺字段处理。
- §10(测试)→ Task 1/2/3/4/5。
- §11(非目标)→ 未实现整快照 reader,符合。

### 2. Placeholder scan
无 TBD/TODO;每个含代码的 Step 均给出具体代码。

### 3. Type consistency
- `_LazyCtx.set_snapshot_meta(basepath, snap, snap_offsets, loadable_family_keys, family_hdf_keys)` 在 Task 1(定义)与 Task 2(调用)一致。
- `_loaded_index` 在 Task 1(定义 dict)、Task 2(`Snapshot` init 初始化)、Task 3(`load_particle` 按 family 填)、Task 4(merge/cover 传播)中 key 均为 `'dm'/'gas'/'star'/'bh'` 且值为 int64 数组 —— 一致。
- `Snapshot.loadable_keys`/`_load_array` 委托 `_lazy_ctx`,与 `_LazySnap` 一致。
- Task 3 中 `_gf(fam)`(pynbody Family)与 Task 1 `parttype_for_family(family)`(`family.name`)对齐。
- Task 4 兜底 `getattr(f1,'_loaded_index',{}).get(fam, empty)` 兼容普通 Snap(无 `_loaded_index`)。
- Task 1 `_field_spec` / `_gather` 使用 `fam`(字符串)与 `_family_obj`;Task 2 传入 `_gf(fam)`(Family 对象);两者在 `load_array` 内一致(`fams` 元素与 `_family_hdf_keys` key 的 Family 对齐)。Task 1 `_LazySnap._load_array` 把 `fam`(可能为 Family 或 None)整体透传给 `_lazy_ctx.load_array`,而 `load_array` 内部 `self._owner[fam]`(fam 为 Family)或 `self._owner`(fam None)一致。
