# AnastrisTNG

[English](README.md) · 简体中文

AnastrisTNG 可以在 pynbody 中读取 Illustris、IllustrisTNG 和 TNG-Cluster 的粒子、星表和 SubLink 并合树。无论从本地文件还是从 TNG 公共 API 读取，都可以使用同一套分析代码。

## 安装

使用 Python 3.8 或更高版本。

```shell
git clone https://github.com/wx-ys/AnastrisTNG.git
cd AnastrisTNG
python -m pip install -e .
```

## 读取第一个 subhalo

使用 API 前，设置环境变量 `TNG_API_KEY`，或在当前工作目录的 `.env` 文件中写入 `TNG_API_KEY=your_key_here`。[从 TNG 个人账户获取密钥](https://www.tng-project.org/users/profile/)。

下面读取 TNG50-1 在红移为零时的 12 号 subhalo。这个红移在 TNG 中对应快照 99，在原版 Illustris 中对应快照 135。

```python
from AnastrisTNG import Snapshot

snap = Snapshot.from_api("TNG50-1", 99)
sub = snap.load_particle(12)
sub.physical_units()

spin = sub.GC["SubhaloSpin"]
print(sub.gas["mHI"].sum())
```

`sub.gas["mHI"]` 给出每个气体单元的原子氢质量。对 12 号 subhalo 求和，结果约为 2.02 × 10⁸ 个太阳质量。`physical_units()` 对随后读取的粒子字段也生效。用 `sub.GC` 可以查看单个对象的星表属性，用 `snap.subhalos_GC("SubhaloGrNr")` 可以读取全部 subhalo 的一列数据。

完整示例还演示了如何将星系转到正面视角，并计算 30 kpc 内恒星转动动能占总动能的比例。运行 notebook 时，将 `Your_API_Key_Here` 换成自己的密钥，或移除 `api_key` 参数以使用已配置的密钥。如需关闭下载进度显示，可给 `from_api` 传入 `progress=False`。[打开 API 使用示例](examples/AnastrisTNG_api_usage.ipynb)。

## 其他读取方式

- **从本地读取**时，用 `snap = Snapshot("/data/TNG50-1/output", 99)` 打开快照。`output/` 中应有快照文件和星表。TNG 公开数据的 offsets 文件应放在相对于 `output/` 的 `../postprocessing/offsets/` 中。在本地追踪并合历史还需下载 SubLink 并合树。
- **已有部分数据**时，给 `Snapshot.from_api(...)` 传入 `local_dir="/data/TNG50-1/output"`，即可使用本地文件，并从 API 补齐缺失的数据。
- **离线分析**时，用 `Snapshot.from_api("TNG50-1", 99, offline=True)` 重新打开已读取的数据。默认缓存目录是 `~/.cache/AnastrisTNG`。如果之前指定了 `cache_dir`，离线时使用同一目录。

## 分析示例

| 想做什么 | Notebook |
| --- | --- |
| 加载 halo、subhalo 或 TNG-Cluster 的 zoom 区域 | [快速上手](examples/AnastrisTNG_quick_start-cn.ipynb) |
| 计算星系属性，调整星系视角 | [星系属性](examples/AnastrisTNG_galaxy_func-cn.ipynb) · [星系正面投影](examples/AnastrisTNG_galaxy_face_on-cn.ipynb) |
| 测量径向和垂向剖面 | [剖面分析](examples/AnastrisTNG_profile-cn.ipynb) |
| 追踪并合历史或跨快照追踪粒子 | [并合树](examples/AnastrisTNG_evolution_history-cn.ipynb) · [粒子追踪](examples/AnastrisTNG_trace_particles.ipynb) |
| 使用扩展分析工具 | [GalaxyPose 恒星形成历史](examples/AnastrisTNG_galpos_SFH-cn.ipynb) · [pynbody-extras](examples/AnastrisTNG_pynext_example-cn.ipynb) |

运行 GalaxyPose 示例前，可用 `python -m pip install galpos` 安装扩展包。[GalaxyPose 项目](https://github.com/GalaxySimAnalytics/GalaxyPose)

由 [Shuai Lu](https://github.com/wx-ys) 维护，采用 [MIT 许可证](LICENSE)。本项目基于 [pynbody](https://github.com/pynbody/pynbody)、[illustris_python](https://github.com/illustristng/illustris_python) 和 [pytreegrav](https://github.com/mikegrudic/pytreegrav)。
