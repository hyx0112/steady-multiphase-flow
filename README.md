# steady-multiphase-flow

**Steady-state multiphase pipe-flow simulator + a pluggable, paper-faithful closure library.**

[English](#english) · [中文](#中文)

> **Code only.** This repository contains no experimental datasets, no case queues, no
> per-case predictions, no error statistics and no analysis reports. You supply your own inputs.

---

## English

### What this is

Two independent pieces:

1. **`steady/`** — the simulator core: segmented forward marching, a PVT interface, flow-pattern
   detection, four two-phase models (Beggs–Brill, Mukherjee–Brill, drift-flux, Xiao), a thermal
   model, and a command-line entry point (`python -m steady.run`).
2. **`closures/`** — splits pressure-gradient and holdup calculation into **12 independently
   swappable closure slots**. Each slot implements several relations from the open literature;
   the default of every slot reproduces the original Xiao model. Use it for controlled
   "swap one closure, see how far the answer moves" experiments.

The design goal is not maximum accuracy but **attributability**: any change in a result traces
back to one equation in one slot, and that equation's implementation caliber (Fanning vs Darcy,
superficial vs actual phase velocity, hydraulic diameter vs pipe diameter, validity gates) is
documented in the code.

### Install

```bash
git clone https://github.com/hyx0112/steady-multiphase-flow.git
cd steady-multiphase-flow
pip install -e .              # or: pip install numpy pandas and use it in place
```

Requires Python ≥ 3.9, numpy and pandas. Optional extras: `pip install -e ".[plots,pvt]"`
(matplotlib for plots, thermo for PVT flash).

### Quickstart

```bash
python examples/quickstart.py
```

Same operating point, same flow pattern, only the wall-friction closure changes:

```
variant                     dP/dL      H_L  converged
------------------------------------------------------
xiao                        710.0   0.7033       True
kowalski1987                377.8   0.6439       True
ottens2001                  643.8   0.7132       True
petalas_wall               1111.0   0.7407       True
biberg1999_wall_full        710.0   0.7033       True
```

Units: `dP/dL` in Pa/m, `H_L` dimensionless.

Direct usage:

```python
import math
from closures.xiao_model_faithful import XiaoInput, xiao_pressure_gradient

inp = XiaoInput(
    total_mass_flow=40.0,        # kg/s
    gas_mass_fraction=0.05,      # -
    rho_gas=30.0, rho_liquid=830.0,
    mu_gas=1.6e-5, mu_liquid=3.0e-3,
    diameter=0.15, roughness=4.5e-5,
    angle_rad=math.radians(2.0),  # positive = uphill
    surface_tension=0.025,
    pressure_Pa=3.0e6,
    flow_pattern_override="stratified",             # pin the pattern, compare closures only
    stratified_wall_friction_closure="petalas_wall",
)
r = xiao_pressure_gradient(inp)
print(r.dp_dL_total, r.liquid_holdup_inclined, r.flow_pattern, r.converged)
```

**Every input and output is SI**: kg/s, kg/m³, Pa·s, m, rad, N/m, Pa, Pa/m.

### The 12 closure slots

| Slot | Default | What swapping it affects |
|---|---|---|
| `stratified_wall_friction_closure` | `xiao` | stratified pressure gradient (`fL` dominates) |
| `stratified_interfacial_closure` | `xiao` | stratified holdup and pressure gradient |
| `intermittent_slug_holdup_closure` | `gregory1978` | slug-body holdup |
| `intermittent_slug_length_closure` | `scott1987` | slug length (inert by default, see below) |
| `intermittent_translation_closure` | `xiao` | slug translation velocity |
| `intermittent_pressure_extra_closure` | `none` | additional pressure-loss term |
| `annular_entrainment_closure` | `oliemans` | annular entrainment and pressure gradient |
| `annular_interfacial_closure` | `xiao` | annular gas-core/film interfacial friction |
| `flow_pattern_closure` | `xiao` | flow-pattern map |
| `dispersed_bubble_closure` | `xiao` | dispersed-bubble solver |
| `intermittent_film_interfacial_closure` | `constant` | film-zone interfacial friction |
| `single_phase_limit_closure` | `legacy` | single-phase limit routing |

About 45 variants in total — each with its paper reference and caliber convention — are listed in
**[docs/closure_catalogue.md](docs/closure_catalogue.md)**.

### Three things to know before you use it

1. **The slug-length slot is inert by default.** With `pressure_extra_closure="none"`, `Ls` does
   not change the result inside the ratio-weighted structure of Xiao Eq. 30/31 — swapping the
   length relation gives bit-identical numbers. It is still evaluated, though, so out-of-domain
   parameters still raise `ValueError` and drop the case. **Inert is not the same as safe.**
2. **Slug holdup is clamped to `[0.48, 1]`.** In every branch except `gomez2000_orig` /
   `*_via_kim2020` — including the baseline `gregory1978` — the value is returned as
   `clamp(holdup, 0.48, 1.0)`. That is the existing behaviour of the original Xiao model; this
   library does not quietly remove it. Whether to apply the clamp is a separate experimental
   setting and must be recorded explicitly.
3. **`E` is a mass-flow fraction.** `annular_entrainment_closure` returns the **entrained liquid
   mass-flow fraction**, not a holdup, not a film fraction, not a volumetric ratio.

### Fidelity rules

This is a library for **comparative** studies, not a "calculates anything" library. The hard
constraints:

- No implicit empirical coefficients added inside a reference function, no ad-hoc clipping, no
  changing the exponents given in the paper.
- Relations the framework cannot host (missing dependency, single flow pattern only, undisclosed
  constant) are marked `MISSING_DEPENDENCY` / `PARTIAL_IMPLEMENTATION` / `SECONDARY_ONLY` rather
  than being forced in.
- Missing parameters or out-of-domain inputs raise explicitly. **Errors are not counted as
  successes, and there is no silent fallback to the baseline.**
- A variant is never retuned because it scores worse — that would be data fitting, not
  implementing the paper.

The docstring above every branch records the paper, journal volume/pages, equation numbers, the
unit-caliper chain, and where the implementation is approximate or cannot be completed.
**Reading the docstrings is more useful than reading this README.**

### Tests

```bash
python -m unittest discover -s steady -p "test_*.py" -t .
```

### Citing

If you use these closure implementations in a paper, please cite the **original literature**
(full references are in the code docstrings) alongside this repository. The original authors are
the authors of these relations.

Third-party implementation references (Mandhane flow-pattern boundaries and the Taitel–Dukler
boundary splines are adapted from [Caleb Bell's `fluids`](https://github.com/CalebBell/fluids),
MIT) are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

### License

MIT. See [LICENSE](LICENSE).

---

## 中文

稳态气液两相管流计算程序，以及一套**可插拔的论文闭合关系式库**。

> **本仓库只放程序。** 不含任何实验数据、算例队列、逐例预测输出、误差统计或分析报告。
> 想跑自己的算例，输入由你提供。

### 这是什么

两件事：

1. **`steady/`** —— 稳态管道仿真核心。分段正向求解、PVT 接口、流型判别、四个两相模型
   （Beggs–Brill、Mukherjee–Brill、漂移流、Xiao）、热模型、命令行入口 `python -m steady.run`。
2. **`closures/`** —— 把两相压降/持液率计算拆成 **12 个可独立替换的闭合槽位**，每个槽位
   实现了若干篇公开文献的关系式，默认值等于 Xiao 原模型。用来做"换一个闭合式，结果差多少"
   的对照实验。

设计目标不是"算得最准"，而是**可归因**：任何一次结果变化，都能追到某个槽位里的某一个式子，
并且那个式子的实现口径（Fanning/Darcy、用表观速还是实际相速、水力直径还是管径、
适用域闸门）在代码里有据可查。

### 安装

```bash
git clone https://github.com/hyx0112/steady-multiphase-flow.git
cd steady-multiphase-flow
pip install -e .            # 或： pip install numpy pandas 后直接用
```

依赖：Python ≥ 3.9、numpy、pandas。出图需要 matplotlib，PVT 闪蒸需要 thermo（均可选）。

### 快速开始

```bash
python examples/quickstart.py
```

同一工况、同一流型，只换壁面摩阻闭合式：

```
variant                     dP/dL      H_L  converged
------------------------------------------------------
xiao                        710.0   0.7033       True
kowalski1987                377.8   0.6439       True
ottens2001                  643.8   0.7132       True
petalas_wall               1111.0   0.7407       True
biberg1999_wall_full        710.0   0.7033       True

单位：dP/dL = Pa/m，H_L = 持液率（-）
```

直白用法：

```python
import math
from closures.xiao_model_faithful import XiaoInput, xiao_pressure_gradient

inp = XiaoInput(
    total_mass_flow=40.0,        # kg/s
    gas_mass_fraction=0.05,      # -
    rho_gas=30.0, rho_liquid=830.0,
    mu_gas=1.6e-5, mu_liquid=3.0e-3,
    diameter=0.15, roughness=4.5e-5,
    angle_rad=math.radians(2.0),  # 上坡为正
    surface_tension=0.025,
    pressure_Pa=3.0e6,
    flow_pattern_override="stratified",             # 钉住流型，只比较闭合式
    stratified_wall_friction_closure="petalas_wall",
)
r = xiao_pressure_gradient(inp)
print(r.dp_dL_total, r.liquid_holdup_inclined, r.flow_pattern, r.converged)
```

**全部输入输出都是 SI 单位**：kg/s、kg/m³、Pa·s、m、rad、N/m、Pa、Pa/m。

### 闭合槽位

| 槽位 | 默认 | 换它会影响 |
|---|---|---|
| `stratified_wall_friction_closure` | `xiao` | 分层流压降（`fL` 主导） |
| `stratified_interfacial_closure` | `xiao` | 分层流持液率与压降 |
| `intermittent_slug_holdup_closure` | `gregory1978` | 段塞流持液率 |
| `intermittent_slug_length_closure` | `scott1987` | 段塞长度（默认惰性，见下） |
| `intermittent_translation_closure` | `xiao` | 液塞速度 |
| `intermittent_pressure_extra_closure` | `none` | 压降附加项 |
| `annular_entrainment_closure` | `oliemans` | 环状流夹带率与压降 |
| `annular_interfacial_closure` | `xiao` | 环状流界面摩阻 |
| `flow_pattern_closure` | `xiao` | 判型 |
| `dispersed_bubble_closure` | `xiao` | 泡状流求解 |
| `intermittent_film_interfacial_closure` | `constant` | 液膜区界面摩阻 |
| `single_phase_limit_closure` | `legacy` | 单相极限路由 |

约 45 个变体的完整清单、每支的论文出处与口径约定见
**[docs/closure_catalogue.md](docs/closure_catalogue.md)**。

### 三条使用前必须知道的事

1. **段塞长度槽位默认是惰性的。** 在 `pressure_extra_closure="none"` 下，`Ls` 在 Xiao
   Eq.30/31 的比值加权结构里不改变结果——换长度关系式算出来的数会一模一样。但它仍然在求值，
   域外参数照样抛 `ValueError` 丢掉算例。**惰性 ≠ 安全。**
2. **液塞持液率有 `[0.48, 1]` 夹断。** 除 `gomez2000_orig` / `*_via_kim2020` 外，所有分支
   （含基线 `gregory1978`）都以 `clamp(holdup, 0.48, 1.0)` 收尾。这是 Xiao 原模型的既有行为，
   本库没有偷偷去掉它；要不要打开夹断，请作为独立实验设置并显式记录。
3. **`E` 是质量流量分数。** `annular_entrainment_closure` 输出的是夹带液体的**质量流量份额**，
   不是持液率、不是液膜分数、不是体积比。

### 忠实性规则

这是一个用于**对照研究**的库，不是"什么都能算"的库。约束：

- 不在论文式子里补隐式经验系数、不随意限幅、不改论文给的指数。
- 框架接不上的关系式（缺依赖量、只覆盖单一流型、常数未公开）标
  `MISSING_DEPENDENCY` / `PARTIAL_IMPLEMENTATION` / `SECONDARY_ONLY`，不硬凑。
- 缺参数、越出适用域 → 显式报错。**报错不计为成功，也不静默回退基线。**
- 不因为某支误差更大就回头改它的系数——那是数据拟合，不是实现论文。

代码里每个分支上方的 docstring 记录了论文、期刊卷期页码、式号、单位口径链，以及在哪些
地方做了近似、哪些地方做不到。**读 docstring 比读本 README 有用。**

### 测试

```bash
python -m unittest discover -s steady -p "test_*.py" -t .
```

### 引用

如果你在论文里用了这里的闭合实现，请同时引用**原始文献**（代码 docstring 里有完整出处）
和本仓库。原论文作者才是这些关系式的作者。

第三方实现参考（Mandhane 流型边界与 Taitel–Dukler 边界样条改编自
[Caleb Bell `fluids`](https://github.com/CalebBell/fluids)，MIT）见
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

### License

MIT。见 [LICENSE](LICENSE)。
