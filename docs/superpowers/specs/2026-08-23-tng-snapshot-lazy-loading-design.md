# TNG Snapshot 与 load_particle 快照的 lazy-load 支持

- **日期**: 2026-08-23
- **状态**: 草案,待 review
- **作者**: gal3d / Claude Code
- **相关文件**:
  - [src/AnastrisTNG/TNGsimulation.py](../src/AnastrisTNG/TNGsimulation.py)
  - [src/AnastrisTNG/TNGgroupcat.py](../src/AnastrisTNG/TNGgroupcat.py)
  - `~/Simulation/pynbody/pynbody/snapshot/gadgethdf.py`(参考实现,pynbody 外部)

## 1. 动机与现状

当前 `Snapshot.load_particle` 在加载 halo/subhalo 时,**急切**地把 `load_particle_para` 里列出的所有字段读进内存并创建 `SimArray`([TNGsimulation.py:467-742](src/AnastrisTNG/TNGsimulation.py#L467-L742))。这带来两个问题:

1. **必须预先配置 `load_particle_para`**:用户必须通过 `{'dm_fields': [...], 'star_fields': [...], ...}` 指定要加载哪些字段,否则这些字段就取不到。
2. **无法便捷地补载新字段**:一旦一个 halo/subhalo 已经合并进 Snapshot,想再加一个之前没配置的字段(`temp`、`sfr`、`metals`…)没有干净的机制,只能重新 load 或 hack。

目标:参考 pynbody `GadgetHDFSnap` 的 lazy-load 机制,让**合并后的 `Snapshot(self)`** 与 **`load_particle` 返回的快照 `f`**(无论是否被 `Halo(f)`/`Subhalo(f)` 装饰)**都能按需从磁盘读取任意字段**。这样:

- `load_particle_para` 不再需要用户配置(`*_fields` 退化为可选的"额外预载"清单,默认空)。
- 对 `snap['temp']`、`snap.g['temp']` 的访问自动触发按需读取。

## 2. 关键洞察:lazy-load 管线已在基类里

pynbody `SimSnap` **已经内置** lazy-load 触发管线。访问 `snap['temp']` 时:

```
snap['temp']
  → SimSnap.__getitem__(str)
  → _get_array_with_lazy_actions('temp')              # 不在内存 → 尝试加载
  → __load_if_required('temp')
  → __load_array_and_perform_postprocessing('temp')    # 框架包装:单位转换等
  → _load_array('temp', fam=None)                      # ← 子类只需覆写这里
```

- `Snapshot` **已经继承 `SimSnap`**,所以该管线天然可用;问题只是 `_load_array` 目前走基类默认(`raise OSError`),[simsnap.py:884](src/AnastrisTNG/simsnap.py#L884)。
- 加载完成后,框架会**自动**对新建数组补单位并 `_autoconvert_array_unit`([simsnap.py:932-945](src/AnastrisTNG/simsnap.py#L932-L945))。因此只要在 `_load_array` 里先设好文件原始单位,`physical_units()` 后的单位一致性由框架保证。
- `loadable_keys(fam)` 返回"可被 lazy-load 的数组名",框架用它在 `_get_array_with_lazy_actions` 和 family 级解析中判断。

**结论**:只需要覆写 `_load_array` 与 `loadable_keys`,并让返回的快照 `f` 也是带 `_load_array` 的 `SimSnap` 子类。这比复刻整条 gather/chunk/unit 逻辑要轻得多。

## 3. 与 GadgetHDFSnap 的本质差别:需要跟踪粒子原始行号

GadgetHDFSnap 读的是**整个文件**——每个数组都是全量或由 `take` 索引的全局子集。而 TNG 的 Snapshot 是经 `load_halo`/`load_subhalo` **逐块合并的粒子子集**。要按需读字段,必须知道"**当前每个已加载粒子在原始文件 `PartTypeN` 数据集里的行号**"。

这个行号恰好可精确获得:`getSnapOffsets` 返回 `subset['offsetType'][ptNum]`(该 halo/subhalo 的这个粒子类型在快照里的全局起始行号)与 `subset['lenType'][ptNum]`(数量)。第 $i$ 个粒子的行号 = `offsetType[ptNum] + i` 。

> 注:我们**不**用 `iord`(particle ID)反推行号——ID→行号语义随 TNG 格式可能变化,脆弱。显式记录行号最稳健。

因此核心新状态:

```python
self._loaded_index = {'dm': np.array([], dtype=np.int64),
                      'star': np.array([], dtype=np.int64),
                      'gas': np.array([], dtype=np.int64),
                      'bh': np.array([], dtype=np.int64)}
```

每合并一个 halo/subhalo,就把该块的 `arange(offsetType[ptNum], offsetType[ptNum]+lenType[ptNum])` 按 family `append` 进来。**顺序必须与快照内该 family 的粒子顺序一致**(即逐块拼接顺序,不是全局排序)。

## 4. 架构

```
                 +---------------------------------------------+
                 | Simulation: Snapshot (SimSnap 子类)          |
                 |  _loaded_index: dict[family]->int64[]       |
                 |  _load_array(name, fam)     <-- lazy 核心    |
                 |  loadable_keys(fam)                          |
                 +-------------------^-------------------------+
                                     | 访问任意字段(snap['temp'])
                                     |
                 +---------------------------------------------+
                 | _LazySnap (SimSnap 子类,由 new(class_=..) 创建)|
                 |  即 load_particle 返回的 f                   |
                 |  _loaded_index, _load_array, loadable_keys   |
                 +-------------------^-------------------------+
                                     |
                 Halo(f) / Subhalo(f) 是 f 的 SubSnap 子视图
                 (祖先链指向 f) => lazy 沿祖先链自动继承
```

两块(合并后的 `Snapshot` 与 `_LazySnap`)共享同一套 `_load_array` / `loadable_keys` / gather 读取逻辑。

### 共享组件

将 lazy 逻辑抽成私有方法/基类,`Snapshot` 与 `_LazySnap` 都复用:

- **`_loadable_family_keys: dict[family]->list[str]`** — 每 family 可加载字段的 pynbody 名。快照级 = 各 family 并集。
- **`_hdf_name_for_pynbody(pynbody_name) -> str`** — `snapshot_pa_name` 的逆映射。未映射字段 name 保持不变(直接当成 HDF 名)。
- **`_get_field_spec(part_type, hdf_name) -> (dtype, ndim, unit)`** — 从 HDF 数据集推断 dtype / 维数 / 单位(单位用 `snapshot_units`)。
- **`_gather_field(basePath, part_type, hdf_name, index_array) -> np.ndarray`** — 核心 gather 读取(见 §5)。
- **`_pynbody_name_for_hdf / loadable_keys`** 等小工具。

### 数据流(以 `load_halo` 后访问 `snap.g['temp']` 为例)

1. `load_particle(ID, groupType='Halo')`:
   - `subset = getSnapOffsets(...)`;用 `new(dm=..., star=..., gas=..., bh=..., order=..., class_=_LazySnap)` 创建 `f`。
   - 给 `f` 设 `_loaded_index[family] = arange(offsetType[ptNum], offsetType[ptNum]+lenType[ptNum])`。
   - **只急切加载** `Basefields`(pos/vel/mass/iord)+ 各 family 的 `HaloID`/`SubhaloID`。这些是 `simsnap_merge`/`simsnap_cover` 真正要拷贝的数组。其余字段不读。
   - 返回 `f`(decorate=True 时若 `groupType` 对应族返回 `Halo(f)`/`Subhalo(f)`)。
2. `load_halo`:
   - `fmerge = simsnap_merge(self, f)`(或对非该 halo 的子集 merge);覆盖 `simsnap_merge` 以**同时拼接 `_loaded_index`**。
   - `simsnap_cover(self, fmerge)`;覆盖 `simsnap_cover` 以**传播 `_loaded_index`**。
   - 重建 `_family_index_cached` 等(维持现状)。
3. 用户访问 `snap.g['temp']`:
   - FamilySubSnap 沿祖先链到 `snap`;`_get_array_with_lazy_actions('temp')` 找不到 → `__load_array_and_perform_postprocessing('temp', fam='gas')` → `snap._load_array('temp', 'gas')`。
   - `_load_array` 逆映射得 HDF 名,取 `(dtype, ndim, unit)`,`_create_array('temp', ndim, dtype)` 并设单位。
   - `_gather_field(basePath, 'gas', hdf_name, snap._loaded_index['gas'])` 填充。
   - 框架随后自动做单位转换/变换。

## 5. gather 读取实现(`_gather_field`)

入参:全局行号数组 `index_array`(可能是多段连续 run 的拼接,**保序**,可能跨 chunk 文件)。做法:

1. 用记忆化的 `_snap_offsets[pt_num, :]`(每 chunk 文件该类型累计粒子数)确定每行落在哪个 chunk 文件。
2. 把 `index_array` 切成 **每个 chunk 文件内的连续子段**(保持原始顺序,不排序)。
3. 对每个子段,构造最小 `subset` 并调用 `loadSubset(basePath, snap, part_type, [hdf_name], subset=subset)`,取其返回的该字段,写入输出数组对应切片。
   - `subset = {'offsetType': <full 6 元数组,置 pt_num = 段起点>, 'snapOffsets': self._snap_offsets, 'lenType': <置 pt_num = 段长度>}`。
4. 聚合所有子段得到完整字段数组。

`_snap_offsets` 从 `getSnapOffsets(basePath, snap, 1, 'Group')['snapOffsets']` 一次性取并缓存(或读 HDF header 计算)。

> 并发/边界:`loadSubset` 对 `numToRead==0` 返回空 dict,需跳过;字段不存在时对特定 family 抛 `OSError` 以触发框架的逐 family 回退。

## 6. 组件细案

### 6.1 `_LazySnap(SimSnap)`

轻量子类,仅带 lazy 能力与 `_loaded_index`。由于 `new(class_=...)` 会调用 `class_()`,并已设置 `_num_particles`、`_family_slice`、preload pos/vel/mass,`_LazySnap` 只需:

```python
class _LazySnap(SimSnap):
    def __init__(self):
        super().__init__()
        self._loaded_index = {'dm': np.array([], np.int64), 'star': ..., 'gas': ..., 'bh': ...}
        self._lazy_ctx = None   # 指向共享 loader 上下文(basePath, snap, _snap_offsets, _loadable_family_keys)
        self._load_ctx_needs_setup = True
    def _load_array(self, name, fam=None): ...
    def loadable_keys(self, fam=None): ...
```

`Snapshot` 与 `_LazySnap` 通过共享同一 loader 上下文对象(`_LazyCtx`)复用读取逻辑。

### 6.2 `Snapshot` 集成

- `__init__`:保留现有的家族/数组结构初始化(空 `pos`/`vel`/`mass`、空 `family_slice`、`__set_load_particle`);**新增**初始化 `_loaded_index`;扫描 HDF `PartTypeN` 组构建 `_loadable_family_keys`(可与现有 `loadable_parameters['snapshots']` 的扫描合并);缓存 `_snap_offsets`;建立 `_LazyCtx`。
- 新增 `_load_array` / `loadable_keys`(与 `_LazySnap` 共享)。
- `load_particle`:改用 `new(..., class_=_LazySnap, order=...)`,急切加载范围**收缩为** Basefields + IDs(不再默认加载 `*_fields` 的扩展字段,除非用户显式设置),并给 `f` 设置 `_loaded_index` / `_LazyCtx`。合并后的 `self` 仍通过 `simsnap_cover` 重建 `pos`/`vel`/`mass` 等基础数组。

### 6.3 `loadable_keys(fam=None)`

返回该 family 可 lazy-load 的数组名列表(已用 `snapshot_pa_name` 映射为 pynbody 名)。`fam=None` 时返回所有 family 的并集。与 `keys()`(已加载)自然区分。

### 6.4 `simsnap_merge` / `simsnap_cover` 扩展

- `simsnap_merge(f1, f2)`:在创建 `f3` 后,对每个 family:`f3._loaded_index[family] = np.append(f1._loaded_index[family], f2._loaded_index[family], axis=0)`;复制 `_LazyCtx` 引用。若无 `_loaded_index`(普通 Snap 兼容),用空数组兜底。
- `simsnap_cover(f1, f2)`:在拷贝数组后,`f1._loaded_index = dict(f2._loaded_index)`(deepcopy 数组)。

这样"合并后的 Snapshot 仍 lazy、索引仍正确"。

### 6.5 装饰后的继承(`Halo(f)` / `Subhalo(f)`)

`Basehalo(SubSnap)` 在 `__init__` 里 `SubSnap.__init__(self, simarray, slice(len(simarray)))`([TNGsnapshot.py:55](src/AnastrisTNG/TNGsnapshot.py#L55)),是 `f` 的**子视图**,祖先链指向 `f`。对子视图访问数组时 lazy 委托给祖先 `f._load_array`,故 `halo.g['temp']` 自动 lazy。**无需改动 Halo/Subhalo 类**,但要确保 `f` 在装饰前已设置好 `_loaded_index` 与 `_LazyCtx`。

## 7. `load_particle_para` 的角色变化(向后兼容)

- `particle_field`:`new(order=...)` 用于 family 排序,保留。
- `Basefields`:仍是"急切加载的核心字段",默认 `['Coordinates','Velocities','Masses','ParticleIDs']`,保留。
- `*_fields`:退化为**可选的额外预载字段**;不为空时仍按旧逻辑急切加载,兼容旧调用([TNGtools.py:573](src/AnastrisTNG/TNGtools.py#L573) 等仍会设置它)。默认空 → 该字段仅在访问时 lazy 加载。
- 用户不再需要为了取 `temp` 而配置 `snap.load_particle_para['gas_fields']`。

## 8. 单位与 `physical_units` 一致性

- `_load_array` 用 `snapshot_units(hdf_name)` 设置文件原始单位。
- 框架在 `__load_array_and_perform_postprocessing` 内自动 `_autoconvert_array_unit`([simsnap.py:938](src/AnastrisTNG/simsnap.py#L938)),因此 `physical_units(persistent=True)` 之后 lazy 加载的新字段也会自动转到物理单位,与既有数组一致。
- `physical_units()` 仍会将 `_canloadPT=False`(阻止 *新增粒子*,与 lazy 字段读取无关)。

## 9. 错误处理与边界

- **字段在该 family 不存在**:`_load_array` 抛 `OSError` → 框架逐 family 回退([simsnap.py:920-924](src/AnastrisTNG/simsnap.py#L920-L924)),对不存在的 family 跳过。若所有 family 都无该字段,最终仍 `KeyError`,维持现状语义。
- **跨 chunk 文件的连续块**:§5 按文件切分处理,避免 `loadSubset` 单帧误读。
- **初始空 Snapshot**(未 load 任何粒子):`_loaded_index` 全空,`_load_array` 应安全返回空数组或抛 `KeyError`。
- **`_LazySnap` 未设置 `_LazyCtx`**:显式断言,避免静默错误。
- **`snapshot_units` 对未知字段抛 `KeyError`**:捕获并退化为 `units.no_unit`(与 GadgetHDFSnap 的 `NoUnit` 一致),避免因新字段单位缺失而崩。

## 10. 测试

新增测试目录 `tests/` 与测试文件(用少量真实/合成 TNG 快照)。核心用例:

1. **lazy 字段读取**:`load_subhalo(id)` 后 `snap.g['temp']` 与直接 `loadSubset(...,['Temperature'], subset)` 的结果逐元素一致。
2. **loadable_keys 覆盖**:`snap.loadable_keys('gas')` 包含 `temp` 等已配置字段的 pynbody 名。
3. **merge 后索引正确**:连续 `load_halo(A)` + `load_subhalo(B)` 后,`snap._loaded_index['star']` 是两块 `arange` 的保序拼接。
4. **装饰后 lazy 仍生效**:`Halo(load_particle(id, decorate=False))` 上访问 `.g['temp']` 能取到值。
5. **跨 chunk**:构造分布在两个 chunk 文件的粒子,验证 `_gather_field` 读全。
6. **物理单位**:`physical_units(persistent=True)` 后 lazy 加载字段与预加载字段单位一致。

## 11. 非目标(Non-goals)

- 不做"整快照全量 lazy 读取"(`take`/`loadSnapshot`)——那是另一个 GadgetHDFSnap 式的 reader,本次不实现。
- 不改变 `Halo`/`Subhalo` 的 GC(hierarchy)加载逻辑。
- 不做 lazy 写入/回写(`write_array`)。

## 12. 待决问题(Open Questions)

- 是否需要一个可直接"按快照全量"读取的 TNG reader(而非基于 load_halo 合并)?本次按用户范围聚焦在"合并后 Snapshot + load_particle 返回的 f"两者 lazy。
- 若后续发现某个 family 字段无法准确推断 dtype/维数,是否接受回退到全字段读一次(U_f64 兜底)?——按 §9 处理,留待实现计划细化。
