# 闭合关系式目录

`closures/xiao_model_faithful.py` 把两相管流拆成 **12 个可独立替换的闭合槽位**。每个槽位的
取值都是 `XiaoInput` 的一个字符串字段，默认值 = Xiao 博士论文原模型的实现。

所有分支的选择逻辑都写在**选择器函数**里（`_stratified_wall_f_selected`、
`_stratified_fi_selected`、`_slug_body_liquid_holdup`、`_slug_length`、
`_annular_entrainment_fraction`、`_annular_interfacial_fi`、`solve_intermittent`、
`solve_dispersed_bubble`），每个分支上方有一段 docstring，写明**论文、期刊卷期页码、式号、
Fanning/Darcy 口径换算、以及什么地方忠实实现了、什么地方没能实现**。要新增或核对某个关系式，
直接读那段 docstring —— 它是这个库的规格说明，比本文件更权威。

> 本文件只是索引。**式号/口径以代码 docstring 为准。**

## 槽位总表

| # | 槽位字段 | 默认 | 作用 |
|---|---|---|---|
| 1 | `single_phase_limit_closure` | `legacy` | 单相极限路由（`darcy` 用单相 Darcy 取代历史路由） |
| 2 | `flow_pattern_closure` | `xiao` | 流型判别图 |
| 3 | `dispersed_bubble_closure` | `xiao` | 分散泡状流求解器 |
| 4 | `stratified_interfacial_closure` | `xiao` | 分层流界面摩阻 `fi` |
| 5 | `stratified_wall_friction_closure` | `xiao` | 分层流壁面摩阻 `fL` / `fG` |
| 6 | `intermittent_translation_closure` | `xiao` | 液塞平移速度（Taylor 泡速度） |
| 7 | `intermittent_slug_holdup_closure` | `gregory1978` | 液塞体持液率 |
| 8 | `intermittent_film_interfacial_closure` | `constant` | 液膜区界面摩阻 |
| 9 | `intermittent_slug_length_closure` | `scott1987` | 段塞单元长度 `Ls` |
| 10 | `intermittent_pressure_extra_closure` | `none` | 压降附加项（混合损失等） |
| 11 | `annular_entrainment_closure` | `oliemans` | 环状流液滴夹带率 `E` |
| 12 | `annular_interfacial_closure` | `xiao` | 环状流气芯-液膜界面摩阻 |

## 1. 流型图 `flow_pattern_closure`

`xiao`（基线）/ `mukherjee_brill` / `xiao_mb_bubble` / `xiao_mb_bubble_gate` /
`xiao_mb_bubble_from_intermittent` / `xiao_td_bubble_from_intermittent`

后四者是**诊断性组合**：保留 Xiao 图的主体，只在某一个边界上换成 Mukherjee–Brill 或
Taitel–Dukler 的判据，用来隔离"判型改动"与"闭合式改动"的贡献。判型边界实现参考
Caleb Bell `fluids`（MIT），见 `THIRD_PARTY_NOTICES.md`。

## 2. 分层流界面摩阻 `stratified_interfacial_closure`

| 取值 | 说明 |
|---|---|
| `xiao` | 基线 |
| `constant` / `smooth_interface` | 常值 / 光滑界面（用于量化"界面摩阻有多重要"的负对照） |
| `xiao_baker_dg` | Xiao 式的 Baker 气体密度修正变体 |
| `andreussi1987` | Andreussi & Persen (1987)，含 `(h_L/D)^0.2` 修正项 |
| `andreussi1987_capped_3fg` | 上式 + **敏感性对照** `min(fi, 3·fG)`，原文未给此上界 |
| `kowalski1987` | Kowalski (1987) Eq (24)(25)，带 `Re_g` 适用闸门 |
| `kowalski1987_fi_smooth` | 同上 Eq (27) 光滑支（用表观气速 `Re_G⁺`） |
| `kowalski1987_capped_3fg` | `kowalski1987` + 敏感性上限 |
| `petalas` / `petalas_fi_paper` / `petalas_fi_faithful` | Petalas & Aziz 界面支的两个口径 |
| `tzotzi2013` / `tzotzi2013_rholiteral` | Tzotzi & Andritsos (2013) Eq (27) |
| `olga_slai` / `olga_slai_h1` / `olga_slai_capped_3fg` | OLGA SL A/I 口径及其实用变体 |
| `ouyang1996_fi` / `ouyang1996_fi_scan_diag` | Ouyang (1996) 界面式（含阻塞诊断变体） |
| `ullmann2004_fi_tau` | Ullmann et al. (2004) 界面剪应力闭合 |

## 3. 分层流壁面摩阻 `stratified_wall_friction_closure`

| 取值 | 说明 |
|---|---|
| `xiao` | 基线：水力直径 + 实际相速 + Colebrook |
| `kowalski1987` | 仅液壁，Eq 18，`f ∝ Re^-0.37`，表观液速 + 管径 |
| `ottens2001` | **双侧**替换，Taitel–Dukler model a，Eq 7a;b（`C·Re^-n`） |
| `ottens_model_b` / `ottens_model_a_taitel_dukler` | Ottens 的另一个模型口径 |
| `petalas_wall` / `petalas_wall_p52` | 仅液壁，Eq 10，Fanning→Darcy 经幂律 |
| `biberg1999_wall_full` / `biberg1999_wall_fs` / `biberg1999_wall_fs_dh` | Biberg (1999) Eq 45 / 50 口径链 |
| `ullmann2004_wall_mtf` | 层流两流体修正，Eq 22–24 |
| `ouyang1996_wall` / `ouyang1996_wall_re_actual` | Ouyang (1996) 壁面式 |

## 4. 液塞持液率 `intermittent_slug_holdup_closure`

`gregory1978`（基线）/ `gregory1978_fulldata` / `abdul_majeed2000` /
`abdul_majeed2000_via_kim2020` / `abdul_majeed2019` / `gomez2000` / `gomez2000_orig` /
`felizola_via_kim2020` / `kora2011` / `al_safran2015` / `al_ruhaimani` /
`soto_cortes2025` / `soto_cortes2025_darcy`

`gomez2000_orig` / `felizola_via_kim2020` / `abdul_majeed2000_via_kim2020` 三个分支
**早返回、不经宿主 `[0.48, 1]` 夹断**——这是刻意的，因为原式本身就该在更宽的区间取值。
其余所有分支（含基线 `gregory1978`）都以 `clamp(holdup, 0.48, 1.0)` 收尾，这是 Xiao 原模型
的既有行为，本库**没有**偷偷改掉它。夹断是否打开属于独立实验设置，需要显式记录。

## 5. 段塞长度 `intermittent_slug_length_closure`

`scott1987`（基线）/ `scott_1989_2024` / `brill_1981_2024` / `norris_1982_2024` /
`nydal1992` / `woods_hanratty1996` / `baba2018_gated` / `baba2018_rho_m`

> **注意**：在 `intermittent_pressure_extra_closure="none"` 下，`Ls` 在 Xiao Eq.30/31 的
> 比值加权结构里是**惰性量**（`Lu/Ls` 由连续性闭式决定），换长度关系式不改变结果。
> 只有配合 `pressure_extra`（如 Cook & Behnia 混合损失，显式含 `1/Lu`）才会影响压降。
>
> 但**惰性不等于安全**：长度式仍在求值，域外参数一样会抛 `ValueError` 把算例打成失败。
> 实测在 2,050 例间歇流队列上，`brill_1981_2024` / `norris_1982_2024` 各丢 111 例、
> `scott_1989_2024` 丢 305 例。换长度式在结果上免费，在可解性上不免费。

## 6. 液塞平移速度 `intermittent_translation_closure`

`xiao`（基线）/ `nicklin` / `theron1989` / `petalas` / `olga_horizontal` /
`olga_density_scaled_gated9` / `olga_density_scaled_smooth9`

## 7. 压降附加项 `intermittent_pressure_extra_closure`

`none`（基线）/ `cook_behnia2000`（液膜被卷入液塞鼻部的黏性耗散损失，显式含 `1/Lu`）

## 8. 环状流夹带率 `annular_entrainment_closure`

`oliemans`（基线）/ `none` / `olga_generalized`

⚠ `E` 是**夹带液体的质量流量分数**（entrained liquid mass-flow fraction），
不是持液率、不是液膜分数、也不是体积流量比。三者不可互换。

## 9. 环状流界面摩阻 `annular_interfacial_closure`

`xiao`（基线）/ `smooth_core` / `wallis` / `xiao_wallis_blend_50`

## 10. 其余槽位

| 槽位 | 取值 |
|---|---|
| `single_phase_limit_closure` | `legacy`（基线）/ `darcy` |
| `dispersed_bubble_closure` | `xiao`（基线）/ `petalas_dispersed` |
| `intermittent_film_interfacial_closure` | `constant`（基线，Xiao Eq.33）/ 其余分支见代码 |

## 命名约定

- `*_via_kim*`：论文原式经 Kim 等（2020）转引的版本，常数可能与该论文原始印刷值不同。
- `*_capped_3fg`：**敏感性对照，不是论文公式**，用于回答"给 `fi` 加上界会怎样"。
- `*_faithful` / `*_paper`：同一关系的两种口径，前者按可插拔的论文式实现，后者按论文叙述文本。
- `*_diag` / `*_scan_diag`：诊断变体，只为观测，不用于生产。

## 忠实性规则（本库的硬约束）

1. 不在参考函数里补隐式经验系数、不随意限幅、不改论文给的指数。
2. 式子在当前框架里接不上（缺依赖量、只覆盖一种流型、需要未公开常数）→ 标
   `MISSING_DEPENDENCY` / `PARTIAL_IMPLEMENTATION` / `SECONDARY_ONLY`，**不硬凑**。
3. 缺参数、越出适用域 → 抛 `ValueError` / 显式报错，**不算成功，也不静默回退基线**。
4. 不因为某个替换算出来误差更大，就回头改这个论文式的系数。
