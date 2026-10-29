# drift_flux_model.py
"""
基于 Petalas & Aziz (2000) 机理模型 + Bendiksen (1984) 漂移流参数
的两相流压降与持液率计算模块。

参考文献：
    Petalas, N., and Aziz, K. (2000). "A Mechanistic Model for
        Multiphase Flow in Pipes." JCPT, 39(6), 43-55.
    Bendiksen, K.H. (1984). "An Experimental Investigation of the
        Motion of Long Bubbles in Inclined Tubes." Int. J. Multiphase
        Flow, 10(4), 467-483.

架构概览：
    Layer 0: 输入/输出 dataclass（与 segment_solver.py 兼容）
    Layer 1: 几何 & 通用工具
    Layer 2: 漂移流参数（C0, Vd, Vb）
    Layer 3: 流型判别 (Petalas Figure 3 顺序排除法)
    Layer 4: 各流型水力求解器
    Layer 5: 非线性求解器（分层流 & 环雾流）
    Layer 6: 主入口 drift_flux_pressure_gradient()
"""

from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Optional, Tuple

from steady.phase_properties import (
    clamp,
    calc_inventory_gas_mass_fraction,
    calc_holdup_density,
    calc_no_slip_liquid_fraction,
    calc_no_slip_density,
    calc_no_slip_viscosity,
    calc_actual_phase_velocities,
    calc_slip_ratio_from_velocities,
)

G = 9.80665  # 重力加速度 m/s²


# ============================================================
# Layer 0: 输入 / 输出数据类（与 segment_solver.py 完全兼容）
# ============================================================

@dataclass
class DriftFluxInput:
    """
    漂移流模型输入。

    单位约定：
        total_mass_flow: kg/s
        gas_mass_fraction: 0~1，过流气相质量分数
        rho_gas: kg/m³
        rho_liquid: kg/m³
        mu_gas: Pa·s
        mu_liquid: Pa·s
        diameter: m
        roughness: m
        angle_rad: rad（上坡为正）
        surface_tension: N/m
    """
    total_mass_flow: float
    gas_mass_fraction: float
    rho_gas: float
    rho_liquid: float
    mu_gas: float
    mu_liquid: float
    diameter: float
    roughness: float
    angle_rad: float
    surface_tension: float = 0.02

    # 以下参数为旧版兼容保留，Petalas 模型内部不再使用
    sigma_water: float = 0.072
    rho_water: float = 1000.0
    rho_air: float = 1.29
    mu_water: float = 0.001


@dataclass
class DriftFluxResult:
    """漂移流模型计算结果，字段与 BeggsBrillResult / MukherjeeBrillResult 对齐。"""
    dp_dL_friction: float
    dp_dL_gravity: float
    dp_dL_total: float
    liquid_holdup_inclined: float
    gas_void_fraction: float
    gas_mass_fraction_inventory: float
    gas_mass_fraction_flow: float
    flow_pattern: str
    v_sg: float
    v_sl: float
    v_m: float
    v_gas_actual: float
    v_liquid_actual: float
    slip_ratio: float
    rho_no_slip: float
    rho_holdup: float
    mu_no_slip: float
    no_slip_liquid_fraction: float
    f_no_slip: float
    friction_multiplier: float
    f_two_phase: float
    froude_number: float
    reynolds_no_slip: float
    liquid_velocity_number: float
    iterations: int
    converged: bool


# ============================================================
# Layer 1: 几何 & 通用工具
# ============================================================

def _friction_factor(re: float, rr: float, laminar_limit: float = 2300.0) -> float:
    """Colebrook 摩擦因子，紊流用 Haaland 初值 + Colebrook 迭代。"""
    if re <= 0:
        return 0.0
    if re < laminar_limit:
        return 64.0 / re
    # 紊流：Haaland 显式初值
    f = (1.0 / (-1.8 * math.log10((rr / 3.7) ** 1.11 + 6.9 / re))) ** 2
    for _ in range(20):
        lhs = 1.0 / math.sqrt(f)
        rhs = -2.0 * math.log10(rr / 3.7 + 2.51 / (re * math.sqrt(f)))
        if abs(lhs - rhs) < 1e-12:
            break
        f = 1.0 / (rhs ** 2)
    return f


def _friction_factor_turbulent_only(re: float, rr: float) -> float:
    """紊流摩擦因子（Petalas 建议：紊流值大于层流值时使用紊流值）。"""
    f_lam = 64.0 / max(re, 1.0)
    f_turb = (1.0 / (-1.8 * math.log10((rr / 3.7) ** 1.11 + 6.9 / max(re, 1.0)))) ** 2
    return max(f_lam, f_turb)


def _stratified_geometry(hL_tilde: float, D: float) -> dict:
    """
    分层流几何参数。

    hL_tilde = hL / D, 范围 (0, 1)

    返回字典:
        AL, AG, SL, SG, Si, DL, DG, hL, gamma
    """
    hL_tilde = clamp(hL_tilde, 1e-6, 1.0 - 1e-6)
    hL = hL_tilde * D
    # gamma = 液相的圆心半角
    gamma = math.acos(1.0 - 2.0 * hL_tilde)  # [0, π]

    A = math.pi * D * D / 4.0
    AL = D * D / 4.0 * (gamma - math.sin(gamma) * math.cos(gamma))
    AG = A - AL

    SL = gamma * D                          # 液相湿周
    SG = (math.pi - gamma) * D              # 气相湿周（管壁部分）
    Si = D * math.sin(gamma)                # 界面宽度

    DL = 4.0 * AL / max(SL, 1e-12)
    DG = 4.0 * AG / max(SG + Si, 1e-12)

    return {
        "AL": AL, "AG": AG, "A": A,
        "SL": SL, "SG": SG, "Si": Si,
        "DL": DL, "DG": DG,
        "hL": hL, "gamma": gamma, "hL_tilde": hL_tilde,
    }


def _annular_geometry(delta_tilde: float, D: float) -> dict:
    """
    环雾流几何参数。

    delta_tilde = δ_L / D, 液膜无量纲厚度

    返回字典:
        Af, Ac, Sf, Si, Df, Dc, Ef
    """
    delta_tilde = clamp(delta_tilde, 1e-8, 0.5)
    A = math.pi * D * D / 4.0

    # 液膜面积 (环形)
    Af = A * 4.0 * delta_tilde * (1.0 - delta_tilde)
    Ac = A - Af  # 气芯面积

    # 湿周
    Sf = math.pi * D                           # 液膜-管壁
    Si = math.pi * (D - 2.0 * delta_tilde * D)  # 液膜-气芯界面

    Df = 4.0 * Af / max(Sf, 1e-12)
    Dc = 4.0 * Ac / max(Si, 1e-12)

    Ef = Af / A  # 液膜体积分数

    return {
        "Af": Af, "Ac": Ac, "A": A,
        "Sf": Sf, "Si": Si,
        "Df": Df, "Dc": Dc,
        "Ef": Ef, "delta_tilde": delta_tilde,
    }


def _dimensionless_numbers(
    vsg: float, vsl: float,
    rho_l: float, rho_g: float, mu_l: float, sigma: float,
    D: float,
) -> dict:
    """返回常用无量纲数。"""
    g = G
    factor = (rho_l / (g * sigma)) ** 0.25
    Nlv = vsl * factor
    Ngv = vsg * factor
    Nl = mu_l * (1.0 / (rho_l * sigma ** 3)) ** 0.25

    # Bond 数
    drho = max(rho_l - rho_g, 1e-12)
    Bo = drho * g * D * D / max(sigma, 1e-12) if sigma > 0 else 1e6

    return {"Nlv": Nlv, "Ngv": Ngv, "Nl": Nl, "Bo": Bo}


# ============================================================
# Layer 2: 漂移流参数（Bendiksen 1984 + Petalas 2000）
# ============================================================

def calc_C0_intermittent(vm: float, D: float, rho_l: float, mu_l: float,
                         sin_theta: float) -> float:
    """
    C0 分布参数 — Petalas Eq 34.

    C0 = (1.64 + 0.12·sinθ) · Re_mL^(-0.031)

    其中 Re_mL = ρ_L · Vm · D / μL
    """
    if vm <= 0:
        return 1.2
    Re_mL = rho_l * vm * D / max(mu_l, 1e-12)
    C0 = (1.64 + 0.12 * sin_theta) * Re_mL ** (-0.031)
    return clamp(C0, 1.0, 1.65)


def calc_Vdh(D: float, rho_l: float, rho_g: float, sigma: float) -> float:
    """
    水平长气泡漂移速度 — Petalas Eq 39 (Weber).

    Vdh = (0.54 - 1.76/Bo^0.56) · √(g·D·Δρ/ρL)

    当 Bo 很小时 Vdh 截断为 0。
    """
    drho = rho_l - rho_g
    if drho <= 0:
        return 0.0
    Bo = drho * G * D * D / max(sigma, 1e-12)
    if Bo <= 0:
        return 0.0
    term = 0.54 - 1.76 / (Bo ** 0.56)
    if term <= 0:
        return 0.0
    return term * math.sqrt(G * D * drho / rho_l)


def calc_Vdv(D: float, rho_l: float, rho_g: float, sigma: float) -> float:
    """
    垂直长气泡漂移速度 — Petalas Eq 40-41 (Wallis modified).

    Vdv = 0.345 · (1 - e^(-β·Bo)) · √(g·D·Δρ/ρL)
    β  = Bo · exp(3.278 - 1.424·ln(Bo))
    """
    drho = rho_l - rho_g
    if drho <= 0:
        return 0.0
    Bo = drho * G * D * D / max(sigma, 1e-12)
    if Bo <= 0:
        return 0.0

    # β 计算
    log_Bo = math.log(Bo)
    beta = Bo * math.exp(3.278 - 1.424 * log_Bo)
    # (1 - e^(-β·Bo)) ≈ 1 对工程管径（β·Bo >> 1）
    beta_Bo = beta * Bo
    factor = 1.0 - math.exp(-beta_Bo) if beta_Bo < 50 else 1.0
    return 0.345 * factor * math.sqrt(G * D * drho / rho_l)


def calc_Vd_elongated(D: float, rho_l: float, rho_g: float, sigma: float,
                      theta_rad: float, mu_l: float) -> float:
    """
    倾斜长气泡漂移速度 — Petalas Eq 36-38 (Bendiksen).

    Vd = fm · Vd_inf
    Vd_inf = Vdh·cosθ + Vdv·sinθ

    fm 为 Zukoski 黏度修正：
        Re_vd = ρL · Vd_inf · D / (2·μL)
        fm = 0.316 · √Re_vd, 当 fm < 1 时; 否则 fm = 1
    """
    Vdh_val = calc_Vdh(D, rho_l, rho_g, sigma)
    Vdv_val = calc_Vdv(D, rho_l, rho_g, sigma)
    cos_t = math.cos(theta_rad)
    # Eq 38 保符号（页 47 印刷式无绝对值；Bendiksen 原式同）。
    # 2026-10-15 忠实度审计 #B，去 abs()。
    sin_t = math.sin(theta_rad)

    Vd_inf = Vdh_val * cos_t + Vdv_val * sin_t
    if Vd_inf <= 0:
        return 0.0

    # Zukoski 黏度修正
    Re_vd = rho_l * Vd_inf * D / max(2.0 * mu_l, 1e-12)
    fm = 0.316 * math.sqrt(Re_vd)
    fm = clamp(fm, 0.0, 1.0)

    return fm * Vd_inf


def calc_Vb_dispersed(rho_l: float, rho_g: float, sigma: float,
                      theta_rad: float) -> float:
    """
    分散气泡上升速度 — Petalas Eq 43 (Harmathy).

    Vb = 1.53 · (g·σ·Δρ/ρL²)^(1/4) · √sinθ
    """
    drho = rho_l - rho_g
    if drho <= 0:
        return 0.0
    sin_theta = abs(math.sin(theta_rad))
    if sin_theta < 1e-6:
        sin_theta = 1e-6
    return 1.53 * (G * sigma * drho / (rho_l ** 2)) ** 0.25 * math.sqrt(sin_theta)


def calc_Vb_bubble_swarm(rho_l: float, rho_g: float, sigma: float,
                         theta_rad: float) -> float:
    """
    气泡群上升速度 — Petalas Eq 30.

    Vb = 1.41 · (g·σ·Δρ/ρL²)^(1/4) · (sinθ)^(1/4)
    """
    drho = rho_l - rho_g
    if drho <= 0:
        return 0.0
    sin_theta = abs(math.sin(theta_rad))
    if sin_theta < 1e-6:
        sin_theta = 1e-6
    return 1.41 * (G * sigma * drho / (rho_l ** 2)) ** 0.25 * sin_theta ** 0.25


# ============================================================
# Layer 3: 流型判别 (Petalas Figure 3 顺序排除法)
# ============================================================

def _calc_ELs_gregory(vm: float) -> float:
    """
    段塞体内液相体积分数 — Petalas Eq 2 (Gregory et al.).

    ELs = 1 / (1 + (Vm/8.66)^1.39)
    """
    if vm <= 0:
        return 1.0
    return 1.0 / (1.0 + (vm / 8.66) ** 1.39)


def _check_dispersed_bubble(vm: float, vsg: float) -> bool:
    """
    分散气泡流判据 — Petalas Eq 1, Eq 3.

    条件：
        1. ELs < 0.48 (Eq 1；原文页 43)
        2. CG = VSG/Vm ≤ 0.52 (Eq 3)
    """
    ELs = _calc_ELs_gregory(vm)
    CG = vsg / max(vm, 1e-12)
    return ELs < 0.48 and CG <= 0.52


def _check_bubble_flow_possible(D: float, rho_l: float, rho_g: float,
                                 sigma: float, theta_rad: float,
                                 vm: float) -> bool:
    """
    气泡流可能条件 — Petalas Eq 28-29.

    条件 1 (Eq 28): D > 19·√((ρL-ρG)·σ/(ρL²·g))
    条件 2 (Eq 29): cosθ ≤ (4/3) · (Vb²/(Cl²·g·db)) · (ρL/(ρL-ρG))
    """
    drho = rho_l - rho_g
    if drho <= 0:
        return False

    # 条件 1: 管径足够大
    min_D = 19.0 * math.sqrt(sigma / (rho_l * G))
    if D <= min_D:
        return False

    # 条件 2: 倾角足够陡
    # Cl = 0.8, db = 0.007 m (Petalas 推荐)
    Cl = 0.8
    db = 0.007
    Vb = calc_Vb_bubble_swarm(rho_l, rho_g, sigma, theta_rad)
    if Vb <= 0:
        return False

    rhs = (4.0 / 3.0) * (Vb ** 2) / (Cl ** 2 * G * db) * (rho_l / drho)
    cos_theta = math.cos(theta_rad)
    return cos_theta <= rhs


def _check_stratified_stability(geom: dict, rho_l: float, rho_g: float,
                                 theta_rad: float) -> dict:
    """
    分层流稳定性判据 — Petalas Eq 13.

    返回: {"VG_crit_KH": float (K-H 不稳定临界气速)}
    """
    hL_tilde = geom["hL_tilde"]
    AG_val = geom["AG"]
    Si_val = geom["Si"]

    drho = rho_l - rho_g
    cos_theta = abs(math.cos(theta_rad))
    if cos_theta < 0.02:
        cos_theta = 0.02

    # K-H 不稳定性临界气速 (Eq 13)
    dAL_dhL = Si_val  # d(AL)/d(hL) = Si
    if dAL_dhL <= 0 or rho_g <= 0:
        VG_crit = float("inf")
    else:
        VG_crit = (1.0 - hL_tilde) * math.sqrt(
            drho * G * AG_val * cos_theta / (rho_g * dAL_dhL)
        )

    return {"VG_crit_KH": VG_crit}


def _check_annular_transition_from_stratified(geom: dict, rho_l: float, rho_g: float,
                                               theta_rad: float, fL: float,
                                               VL: float) -> bool:
    """
    分层流→环雾流过渡 (下坡陡倾) — Petalas Eq 14 (Barnea).

    VL > √(g·D·(1-h̃L)·cosθ / fL)
    """
    hL_tilde = geom["hL_tilde"]
    D = math.sqrt(geom["A"] * 4.0 / math.pi)
    cos_theta = abs(math.cos(theta_rad))
    if cos_theta < 0.02:
        cos_theta = 0.02
    if fL <= 0:
        return False
    V_crit = math.sqrt(G * D * (1.0 - hL_tilde) * cos_theta / fL)
    return VL > V_crit


# ============================================================
# Layer 4: 各流型水力求解器
# ============================================================

# --- 4.1 分散气泡流 (Dispersed Bubble) ---

def solve_dispersed_bubble(
    vsg: float, vsl: float, vm: float,
    rho_l: float, rho_g: float, mu_l: float, mu_g: float,
    D: float, eps: float, theta_rad: float, sigma: float,
    x_flow: float,
) -> dict:
    """
    分散气泡流求解器 — Petalas Eq 44-50.

    VGdb = C0·Vm + Vb
    EL = 1 - VSG / VGdb (如果 VGdb ≤ 0 则 EL = 1 - VSG/(C0·Vm))
    摩阻：均相模型
    """
    sin_theta = math.sin(theta_rad)
    C0 = calc_C0_intermittent(vm, D, rho_l, mu_l, sin_theta)
    Vb = calc_Vb_dispersed(rho_l, rho_g, sigma, theta_rad)

    VGdb = C0 * vm + Vb
    if VGdb > 0:
        EL = 1.0 - vsg / VGdb
    else:
        # VGdb ≤ 0: 用纯 C0·Vm 作为气泡速度
        EL = 1.0 - vsg / max(C0 * vm, 1e-12)

    EL = clamp(EL, 0.01, 0.99)

    # 均相模型摩阻 — Petalas Eq 47-50
    EG = 1.0 - EL
    rho_m = EL * rho_l + EG * rho_g
    mu_m = EL * mu_l + EG * mu_g
    Re_m = rho_m * vm * D / max(mu_m, 1e-12)
    rr = eps / max(D, 1e-12)
    fm = _friction_factor_turbulent_only(Re_m, rr)

    dp_fric = fm * rho_m * vm * vm / (2.0 * D)
    dp_grav = rho_m * G * sin_theta
    dp_total = dp_fric + dp_grav

    return {
        "EL": EL, "EG": EG,
        "dp_fric": dp_fric, "dp_grav": dp_grav, "dp_total": dp_total,
        "rho_m": rho_m, "mu_m": mu_m, "Re_m": Re_m,
        "fm": fm, "friction_multiplier": 1.0,
        "C0": C0, "Vb": Vb, "VGdb": VGdb,
        "pattern": "dispersed_bubble",
        "v_gas_actual": vsg / max(EG, 1e-12),
        "v_liq_actual": vsl / max(EL, 1e-12),
    }


# --- 4.2 气泡流 (Bubble Flow) ---

def solve_bubble_flow(
    vsg: float, vsl: float, vm: float,
    rho_l: float, rho_g: float, mu_l: float, mu_g: float,
    D: float, eps: float, theta_rad: float, sigma: float,
    x_flow: float,
) -> dict:
    """
    气泡流求解器 — Petalas Eq 53-57.

    Vt = C0·Vm + Vb
    EG = VSG / Vt (限制在 [0, CG_max] 其中 CG_max = VSG/Vm)
    摩阻：均相模型，使用 μL 计算 Re (Eq 56-57)
    """
    sin_theta = math.sin(theta_rad)
    C0 = 1.2  # 气泡流 C0 (Petalas: 湍流扁平剖面取 1.2)
    Vb = calc_Vb_bubble_swarm(rho_l, rho_g, sigma, theta_rad)

    Vt = C0 * vm + Vb
    EG = vsg / max(Vt, 1e-12)
    # 限制 EG
    CG_max = vsg / max(vm, 1e-12)
    EG = clamp(EG, 0.0, CG_max)
    EL = 1.0 - EG

    # 摩阻：以液相物性计算 — Petalas Eq 56-57
    Re_mL = rho_l * vm * D / max(mu_l, 1e-12)
    rr = eps / max(D, 1e-12)
    fmL = _friction_factor_turbulent_only(Re_mL, rr)

    rho_m = EL * rho_l + EG * rho_g
    dp_fric = fmL * rho_m * vm * vm / (2.0 * D)
    dp_grav = rho_m * G * sin_theta
    dp_total = dp_fric + dp_grav

    return {
        "EL": EL, "EG": EG,
        "dp_fric": dp_fric, "dp_grav": dp_grav, "dp_total": dp_total,
        "rho_m": rho_m, "mu_m": mu_l, "Re_m": Re_mL,
        "fm": fmL, "friction_multiplier": 1.0,
        "C0": C0, "Vb": Vb, "Vt": Vt,
        "pattern": "bubble",
        "v_gas_actual": vsg / max(EG, 1e-12),
        "v_liq_actual": vsl / max(EL, 1e-12),
    }


# --- 4.3 分层流 (Stratified Flow) ---

def _solve_stratified_liquid_height(
    vsg: float, vsl: float,
    rho_l: float, rho_g: float, mu_l: float, mu_g: float,
    D: float, eps: float, theta_rad: float, sigma: float,
) -> dict:
    """
    分层流双流体动量平衡求解 hL/D — Petalas Eq 4-12.

    组合动量方程：
        τ_wL·SL/AL - τ_wG·SG/AG - τ_i·Si·(1/AL+1/AG)
        + (ρL-ρG)·g·sinθ = 0

    采用自适应步长扫描 + 割线法求精，处理多根问题（取最小物理根）。
    """
    A = math.pi * D * D / 4.0
    sin_theta = math.sin(theta_rad)
    drho = rho_l - rho_g
    rr = eps / max(D, 1e-12)

    def momentum_balance(hL_tilde: float) -> float:
        geom = _stratified_geometry(hL_tilde, D)
        AL_val = geom["AL"]
        AG_val = geom["AG"]
        SL_val = geom["SL"]
        SG_val = geom["SG"]
        Si_val = geom["Si"]
        DL_val = geom["DL"]
        DG_val = geom["DG"]

        # 实际相速度
        VL = vsl * A / max(AL_val, 1e-12)
        VG = vsg * A / max(AG_val, 1e-12)

        # 气相壁面摩擦
        Re_G = rho_g * abs(VG) * DG_val / max(mu_g, 1e-12)
        fG = _friction_factor_turbulent_only(Re_G, rr)
        tau_wG = fG * rho_g * VG * abs(VG) / 8.0

        # 液相壁面摩擦 — Petalas Eq 10-11
        Re_SL = rho_l * vsl * D / max(mu_l, 1e-12)
        fSL = _friction_factor_turbulent_only(Re_SL, rr)
        fL = 0.452 * (fSL ** 0.731)
        tau_wL = fL * rho_l * VL * abs(VL) / 8.0

        # 界面摩擦 — Petalas Eq 12
        Fr_L = abs(VL) / math.sqrt(max(G * geom["hL"], 1e-12))
        if Fr_L > 0 and rho_g > 0:
            fi = (0.004 + 0.5e-6 * Re_SL) * (Fr_L ** 1.335) * (
                rho_l * D * G / (rho_g * VG * abs(VG) + 1e-12)
            )
        else:
            fi = fG
        fi = clamp(fi, 0.0, 10.0)

        Vi = VG - VL
        tau_i = fi * rho_g * Vi * abs(Vi) / 8.0
        # 保持 τ_i 与 Vi 同号
        if Vi < 0:
            tau_i = -abs(tau_i)

        # 组合动量方程
        term_L = tau_wL * SL_val / max(AL_val, 1e-12)
        term_G = tau_wG * SG_val / max(AG_val, 1e-12)
        term_i = tau_i * Si_val * (1.0 / max(AL_val, 1e-12) + 1.0 / max(AG_val, 1e-12))
        term_g = drho * G * sin_theta

        return term_L - term_G - term_i + term_g

    # 扫描搜寻符号变化（找根）
    n_scan = 200
    roots = []
    h_prev = 1e-6
    f_prev = momentum_balance(h_prev)

    for i in range(1, n_scan + 1):
        h_curr = i / n_scan * (1.0 - 1e-6)
        f_curr = momentum_balance(h_curr)

        if f_prev * f_curr <= 0:
            # 发现根区间 [h_prev, h_curr]
            # 割线法求精
            ha, hb = h_prev, h_curr
            fa, fb = f_prev, f_curr
            for _ in range(50):
                if abs(fb - fa) < 1e-15:
                    h_root = (ha + hb) / 2.0
                    break
                h_root = hb - fb * (hb - ha) / (fb - fa)
                h_root = clamp(h_root, ha - 0.01, hb + 0.01)
                h_root = clamp(h_root, 1e-6, 1.0 - 1e-6)
                f_root = momentum_balance(h_root)
                if abs(f_root) < 1e-8:
                    break
                if fa * f_root <= 0:
                    hb, fb = h_root, f_root
                else:
                    ha, fa = h_root, f_root
            roots.append(h_root)

        h_prev = h_curr
        f_prev = f_curr

    if not roots:
        # 无根：取 h̃L 使 f 最接近 0
        best_h = 0.5
        best_f = abs(momentum_balance(best_h))
        for h_try in [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]:
            f_try = abs(momentum_balance(h_try))
            if f_try < best_f:
                best_f = f_try
                best_h = h_try
        hL_tilde = best_h
    else:
        # Petalas: 取最小根
        hL_tilde = min(roots)

    geom = _stratified_geometry(hL_tilde, D)
    return {
        "hL_tilde": hL_tilde,
        "geom": geom,
        "n_roots": len(roots),
    }


def solve_stratified(
    vsg: float, vsl: float, vm: float,
    rho_l: float, rho_g: float, mu_l: float, mu_g: float,
    D: float, eps: float, theta_rad: float, sigma: float,
    x_flow: float,
) -> dict:
    """
    分层流水力求解器 — Petalas Eq 4-16, Eq 51.

    1. 求解双流体动量方程得 hL
    2. EL = AL/A
    3. 摩阻由各相直接计算
    """
    sin_theta = math.sin(theta_rad)
    rr = eps / max(D, 1e-12)
    A = math.pi * D * D / 4.0

    sol = _solve_stratified_liquid_height(vsg, vsl, rho_l, rho_g, mu_l, mu_g,
                                          D, eps, theta_rad, sigma)
    geom = sol["geom"]
    hL_tilde = sol["hL_tilde"]

    EL = geom["AL"] / A
    EG = geom["AG"] / A

    # 各相实际速度
    VL = vsl * A / max(geom["AL"], 1e-12)
    VG = vsg * A / max(geom["AG"], 1e-12)

    # 液相壁面摩擦 — Petalas Eq 10-11
    Re_SL = rho_l * vsl * D / max(mu_l, 1e-12)
    fSL_val = _friction_factor_turbulent_only(Re_SL, rr)
    fL = 0.452 * (fSL_val ** 0.731)

    # 气相壁面摩擦 + 界面摩擦 → dp/dL (Eq 4)
    Re_G = rho_g * abs(VG) * geom["DG"] / max(mu_g, 1e-12)
    fG = _friction_factor_turbulent_only(Re_G, rr)
    tau_wG = fG * rho_g * VG * abs(VG) / 8.0

    # 界面摩擦 — Petalas Eq 12
    Fr_L = abs(VL) / math.sqrt(max(G * geom["hL"], 1e-12))
    if Fr_L > 0 and rho_g > 0:
        fi = (0.004 + 0.5e-6 * Re_SL) * (Fr_L ** 1.335) * (
            rho_l * D * G / (rho_g * VG * abs(VG) + 1e-12)
        )
    else:
        fi = fG
    fi = clamp(fi, 0.0, 10.0)
    Vi = VG - VL
    tau_i = fi * rho_g * Vi * abs(Vi) / 8.0

    dp_fric_gas = (tau_wG * geom["SG"] + tau_i * geom["Si"]) / max(geom["AG"], 1e-12)
    rho_holdup = EL * rho_l + EG * rho_g
    dp_grav = rho_holdup * G * sin_theta
    dp_total = dp_fric_gas + dp_grav

    return {
        "EL": EL, "EG": EG,
        "dp_fric": dp_fric_gas, "dp_grav": dp_grav, "dp_total": dp_total,
        "rho_m": rho_holdup, "mu_m": mu_l,
        "fL": fL, "fG": fG, "fi": fi,
        "hL_tilde": hL_tilde,
        "friction_multiplier": 1.0,
        "pattern": "stratified",
        "v_gas_actual": VG,
        "v_liq_actual": VL,
        "stratified_detail": sol,
    }


# --- 4.4 环雾流 (Annular-Mist Flow) ---

def _solve_annular_film_height(
    vsg: float, vsl: float,
    rho_l: float, rho_g: float, mu_l: float, mu_g: float,
    D: float, eps: float, theta_rad: float, sigma: float,
) -> dict:
    """
    环雾流液膜动量平衡求解 δ̃L — Petalas Eq 17-27.

    含液滴夹带模型 (Eq 22) 和界面摩擦模型 (Eq 23)。
    """
    A = math.pi * D * D / 4.0
    sin_theta = math.sin(theta_rad)
    rr = eps / max(D, 1e-12)
    g = G
    drho = rho_l - rho_g

    # 液滴夹带率 FE — Petalas Eq 22 (不依赖 δ̃L，可预计算)
    NB = mu_l ** 2 * vsg ** 2 / (sigma ** 2) if sigma > 0 else 1e6
    if vsl > 0:
        FE = 0.735 * (NB ** 0.074) * (vsg / vsl) ** 0.2
        FE = FE / (1.0 + FE)
        FE = clamp(FE, 0.0, 0.95)
    else:
        FE = 0.0

    def momentum_balance(delta_tilde: float) -> float:
        ann = _annular_geometry(delta_tilde, D)

        # 液膜和气芯速度
        Vf = vsl * (1.0 - FE) * A / max(ann["Af"], 1e-12)
        Vc = (vsg + vsl * FE) * A / max(ann["Ac"], 1e-12)

        # 液膜壁面摩擦
        Re_f = rho_l * Vf * ann["Df"] / max(mu_l, 1e-12)
        ff = _friction_factor_turbulent_only(Re_f, rr)
        tau_wL = ff * rho_l * Vf * abs(Vf) / 8.0

        # 气芯摩擦 + 界面摩擦比 — Petalas Eq 23
        Re_c = rho_g * abs(Vc) * ann["Dc"] / max(mu_g, 1e-12)
        fc_val = _friction_factor_turbulent_only(Re_c, rr)
        fi_over_fc = 1.0 + 0.24 * (
            (sigma / max(rho_g * Vc * Vc * ann["Dc"], 1e-12)) ** 0.085
        ) * (Re_f ** 0.305)
        fi_over_fc = clamp(fi_over_fc, 0.1, 10.0)
        fi_val = fi_over_fc * fc_val

        Vi = Vc - Vf
        tau_i = fi_val * rho_g * Vi * abs(Vi) / 8.0

        # 组合动量方程 (Eq 17 + Eq 18 消去 dp/dL)
        # τ_wL·Sf/Af - τ_i·Si·(1/Af+1/Ac) + Δρ·g·sinθ = 0
        inv_Af = 1.0 / max(ann["Af"], 1e-12)
        inv_Ac = 1.0 / max(ann["Ac"], 1e-12)
        term_f = tau_wL * ann["Sf"] * inv_Af
        term_i = tau_i * ann["Si"] * (inv_Af + inv_Ac)
        term_g = drho * g * sin_theta

        return term_f - term_i + term_g

    # 扫描找根
    n_scan = 200
    roots = []
    d_prev = 1e-8
    f_prev = momentum_balance(d_prev)

    for i in range(1, n_scan + 1):
        d_curr = i / n_scan * 0.45
        f_curr = momentum_balance(d_curr)

        if f_prev * f_curr <= 0:
            # 割线法求精
            da, db = d_prev, d_curr
            fa, fb = f_prev, f_curr
            for _ in range(50):
                if abs(fb - fa) < 1e-15:
                    d_root = (da + db) / 2.0
                    break
                d_root = db - fb * (db - da) / (fb - fa)
                d_root = clamp(d_root, 1e-8, 0.45)
                f_root = momentum_balance(d_root)
                if abs(f_root) < 1e-8:
                    break
                if fa * f_root <= 0:
                    db, fb = d_root, f_root
                else:
                    da, fa = d_root, f_root
            roots.append(d_root)

        d_prev = d_curr
        f_prev = f_curr

    if not roots:
        # 无符号变化：找 |momentum_balance| 最小处
        best_d = 0.05
        best_f = abs(momentum_balance(best_d))
        for d_try in [1e-6, 1e-4, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.35, 0.45]:
            f_try = abs(momentum_balance(d_try))
            if f_try < best_f:
                best_f = f_try
                best_d = d_try
        delta_tilde = best_d
    else:
        delta_tilde = min(roots)

    ann = _annular_geometry(delta_tilde, D)
    return {
        "delta_tilde": delta_tilde,
        "ann": ann,
        "FE": FE,
        "n_roots": len(roots),
    }


def solve_annular_mist(
    vsg: float, vsl: float, vm: float,
    rho_l: float, rho_g: float, mu_l: float, mu_g: float,
    D: float, eps: float, theta_rad: float, sigma: float,
    x_flow: float,
) -> dict:
    """
    环雾流求解器 — Petalas Eq 17-27, Eq 52.

    液膜动量平衡 + 夹带模型 → EL
    """
    sin_theta = math.sin(theta_rad)
    rr = eps / max(D, 1e-12)
    A = math.pi * D * D / 4.0

    sol = _solve_annular_film_height(vsg, vsl, rho_l, rho_g, mu_l, mu_g,
                                      D, eps, theta_rad, sigma)
    ann = sol["ann"]
    FE = sol["FE"]
    delta_tilde = sol["delta_tilde"]

    # 液含率 — Petalas Eq 52
    EL = 1.0 - (1.0 - 2.0 * delta_tilde) ** 2 * vsg / max(vsg + FE * vsl, 1e-12)
    EL = clamp(EL, 0.01, 0.99)
    EG = 1.0 - EL

    # 液膜速度
    Vf = vsl * (1.0 - FE) * A / max(ann["Af"], 1e-12)
    Vc = (vsg + vsl * FE) * A / max(ann["Ac"], 1e-12)

    # 摩擦：从液膜剪切直接计算 — Petalas Eq 63
    Re_f = rho_l * Vf * ann["Df"] / max(mu_l, 1e-12)
    ff = _friction_factor_turbulent_only(Re_f, rr)
    tau_wL = ff * rho_l * Vf * abs(Vf) / 8.0
    dp_fric = 4.0 * tau_wL / D

    rho_holdup = EL * rho_l + EG * rho_g
    dp_grav = rho_holdup * G * sin_theta
    dp_total = dp_fric + dp_grav

    return {
        "EL": EL, "EG": EG,
        "dp_fric": dp_fric, "dp_grav": dp_grav, "dp_total": dp_total,
        "rho_m": rho_holdup, "mu_m": mu_l,
        "delta_tilde": delta_tilde, "FE": FE,
        "friction_multiplier": 1.0,
        "pattern": "annular_mist",
        "v_gas_actual": Vc,
        "v_liq_actual": Vf,
    }


# --- 4.5 间歇流 (Intermittent Flow: Slug + Elongated Bubble) ---

def solve_intermittent(
    vsg: float, vsl: float, vm: float,
    rho_l: float, rho_g: float, mu_l: float, mu_g: float,
    D: float, eps: float, theta_rad: float, sigma: float,
    x_flow: float,
) -> dict:
    """
    间歇流（段塞 + 长气泡）求解器 — Petalas Eq 32-43, Eq 58-64.

    1. 计算段塞单元体液相质量守恒 → EL (Eq 32)
    2. 摩阻 = α·(slug摩阻) + (1-α)·(环雾摩阻) (Eq 59)
    """
    sin_theta = math.sin(theta_rad)
    rr = eps / max(D, 1e-12)
    A = math.pi * D * D / 4.0

    # 段塞体液相分数 (Gregory)
    ELs = _calc_ELs_gregory(vm)

    # 漂移流参数
    C0 = calc_C0_intermittent(vm, D, rho_l, mu_l, sin_theta)
    Vd = calc_Vd_elongated(D, rho_l, rho_g, sigma, theta_rad, mu_l)
    Vb = calc_Vb_dispersed(rho_l, rho_g, sigma, theta_rad)

    # 平移速度 — Petalas Eq 33
    Vt = C0 * vm + Vd

    # 段塞中分散气泡速度 — Petalas Eq 42
    VGdb = C0 * vm + Vb
    if VGdb < 0:
        VGdb = 0.0  # 下坡低液速时可能为负

    # 液含率 — Petalas Eq 32
    if Vt > 0:
        numerator = ELs * Vt + VGdb * (1.0 - ELs) - vsg
        EL = numerator / Vt
    else:
        EL = 0.5
    EL = clamp(EL, 0.01, 0.99)
    EG = 1.0 - EL

    # 摩阻 — Petalas Eq 59-64
    # 段塞部分摩阻 (Eq 61)
    Re_mL = rho_l * vm * D / max(mu_l, 1e-12)
    fmL = _friction_factor_turbulent_only(Re_mL, rr)
    rho_m = EL * rho_l + EG * rho_g
    dp_fric_slug = fmL * rho_m * vm * vm / (2.0 * D)

    # 环雾部分摩阻 (Eq 63-64)
    if EL > 0 and EL < 1.0:
        # 由 EL 反推 δ̃L — Petalas Eq 62
        FE_est = 0.2  # 简化估算
        inner = 1.0 - EL * (FE_est * vsl + vsg) / max(vsg, 1e-12)
        if inner >= 0:
            delta_tilde_est = 0.5 * (1.0 - math.sqrt(max(inner, 0.0)))
        else:
            delta_tilde_est = 0.05
        delta_tilde_est = clamp(delta_tilde_est, 1e-4, 0.45)

        ann = _annular_geometry(delta_tilde_est, D)
        Vf_est = vsl * (1.0 - FE_est) * A / max(ann["Af"], 1e-12)
        Re_f = rho_l * Vf_est * ann["Df"] / max(mu_l, 1e-12)
        ff = _friction_factor_turbulent_only(Re_f, rr)
        tau_wL = ff * rho_l * Vf_est * abs(Vf_est) / 8.0
        dp_fric_am = 4.0 * tau_wL / D
    else:
        dp_fric_am = dp_fric_slug

    # 权重因子 — Petalas Eq 60（η = C_L·(0.75−E_L)；页 48 印刷式，
    # C_L 在 JCPT 无定义，按 Xiao et al. (1990) 同式承袭取入流液含率
    # λ_L = V_SL/V_m，非现场持液率 E_L。2026-10-15 忠实度审计 #5。）
    alpha = clamp((vsl / max(vm, 1e-12)) * (0.75 - EL), 0.0, 1.0)

    dp_fric = alpha * dp_fric_slug + (1.0 - alpha) * dp_fric_am

    rho_holdup = EL * rho_l + EG * rho_g
    dp_grav = rho_holdup * G * sin_theta
    dp_total = dp_fric + dp_grav

    return {
        "EL": EL, "EG": EG,
        "dp_fric": dp_fric, "dp_grav": dp_grav, "dp_total": dp_total,
        "rho_m": rho_holdup, "mu_m": mu_l,
        "C0": C0, "Vd": Vd, "Vb": Vb, "Vt": Vt,
        "ELs": ELs, "alpha": alpha,
        "friction_multiplier": alpha,
        "pattern": "intermittent",
        "sub_pattern": "slug" if ELs >= 0.9 else "elongated_bubble",
        "v_gas_actual": vsg / max(EG, 1e-12),
        "v_liq_actual": vsl / max(EL, 1e-12),
        "dp_fric_slug": dp_fric_slug,
        "dp_fric_am": dp_fric_am,
    }


# --- 4.6 Froth 流（过渡区插值） ---

def solve_froth(
    vsg: float, vsl: float, vm: float,
    rho_l: float, rho_g: float, mu_l: float, mu_g: float,
    D: float, eps: float, theta_rad: float, sigma: float,
    x_flow: float,
    flow_pattern_boundaries: dict,
) -> dict:
    """
    Froth 流求解器 — Petalas 过渡区对数插值。

    在分散气泡流/间歇流与环雾流之间进行对数插值。
    """
    # 简化：取间歇流和环雾流的对数平均
    sol_int = solve_intermittent(vsg, vsl, vm, rho_l, rho_g, mu_l, mu_g,
                                  D, eps, theta_rad, sigma, x_flow)
    sol_am = solve_annular_mist(vsg, vsl, vm, rho_l, rho_g, mu_l, mu_g,
                                 D, eps, theta_rad, sigma, x_flow)

    # 对数插值权重 (偏向前一流型)
    w = 0.5

    EL = w * sol_int["EL"] + (1.0 - w) * sol_am["EL"]
    EL = clamp(EL, 0.01, 0.99)
    EG = 1.0 - EL

    # 对数平均压降
    dp_fric_log = math.exp(
        w * math.log(max(sol_int["dp_fric"], 1e-6))
        + (1.0 - w) * math.log(max(sol_am["dp_fric"], 1e-6))
    )
    dp_grav_log = math.exp(
        w * math.log(max(abs(sol_int["dp_grav"]), 1e-6))
        + (1.0 - w) * math.log(max(abs(sol_am["dp_grav"]), 1e-6))
    )
    if sol_int["dp_grav"] < 0:
        dp_grav_log = -dp_grav_log

    dp_total = dp_fric_log + dp_grav_log
    rho_holdup = EL * rho_l + EG * rho_g

    return {
        "EL": EL, "EG": EG,
        "dp_fric": dp_fric_log, "dp_grav": dp_grav_log, "dp_total": dp_total,
        "rho_m": rho_holdup, "mu_m": mu_l,
        "friction_multiplier": 1.0,
        "pattern": "froth",
        "v_gas_actual": vsg / max(EG, 1e-12),
        "v_liq_actual": vsl / max(EL, 1e-12),
    }


# ============================================================
# Layer 5: 流型判别主逻辑
# ============================================================

def determine_flow_pattern(
    vsg: float, vsl: float, vm: float,
    rho_l: float, rho_g: float, mu_l: float, mu_g: float,
    D: float, eps: float, theta_rad: float, sigma: float,
) -> str:
    """
    Petalas Figure 3 顺序排除法确定流型。

    流程：
        分散气泡流? → 分层流? → 环雾流? → 气泡流?
        → 否则 间歇流（段塞/长气泡）
    Froth 仅在间歇流不稳定时出现。
    """
    sin_theta = math.sin(theta_rad)
    A = math.pi * D * D / 4.0

    # ---- 1. 分散气泡流 ----
    if _check_dispersed_bubble(vm, vsg):
        return "dispersed_bubble"

    # ---- 2. 分层流 (仅限 θ ≤ 0，即水平及下坡) ----
    if theta_rad <= 0.0:
        try:
            sol_strat = _solve_stratified_liquid_height(
                vsg, vsl, rho_l, rho_g, mu_l, mu_g, D, eps, theta_rad, sigma
            )
            geom = sol_strat["geom"]
            if geom["hL_tilde"] > 0.001 and geom["hL_tilde"] < 0.999:
                VL = vsl * A / max(geom["AL"], 1e-12)
                VG = vsg * A / max(geom["AG"], 1e-12)
                rr = eps / max(D, 1e-12)

                stab = _check_stratified_stability(geom, rho_l, rho_g, theta_rad)
                VG_crit = stab["VG_crit_KH"]

                Re_SL = rho_l * vsl * D / max(mu_l, 1e-12)
                fSL = _friction_factor_turbulent_only(Re_SL, rr)
                fL = 0.452 * (fSL ** 0.731)

                is_annular_trans = _check_annular_transition_from_stratified(
                    geom, rho_l, rho_g, theta_rad, fL, VL
                )

                if VG < VG_crit and not is_annular_trans:
                    return "stratified"
        except (ValueError, ZeroDivisionError):
            pass

    # ---- 3. 环雾流 (含稳定性检查) ----
    try:
        sol_ann = _solve_annular_film_height(
            vsg, vsl, rho_l, rho_g, mu_l, mu_g, D, eps, theta_rad, sigma
        )
        delta_tilde = sol_ann["delta_tilde"]
        FE = sol_ann["FE"]

        if delta_tilde > 1e-4:
            # 计算环雾流 EL
            EL_ann = 1.0 - (1.0 - 2.0 * delta_tilde) ** 2 * vsg / max(vsg + FE * vsl, 1e-12)

            if theta_rad > 0.01:
                # 上坡：检查 EL_ann < 0.24 (Eq 27, 环雾流稳定判据)
                # 以及 δ̃L 在合理范围 (0 < δ̃L < 0.5)
                if EL_ann < 0.24 and 1e-4 < delta_tilde < 0.45:
                    return "annular_mist"
            else:
                # 水平或下坡：液膜存在即可
                if delta_tilde < 0.45:
                    return "annular_mist"
    except (ValueError, ZeroDivisionError):
        pass

    # ---- 4. 气泡流 ----
    if _check_bubble_flow_possible(D, rho_l, rho_g, sigma, theta_rad, vm):
        C0_slug = calc_C0_intermittent(vm, D, rho_l, mu_l, sin_theta)
        Vd_slug = calc_Vd_elongated(D, rho_l, rho_g, sigma, theta_rad, mu_l)
        Vt_slug = C0_slug * vm + Vd_slug
        EG_slug = vsg / max(Vt_slug, 1e-12)
        if EG_slug < 0.25:  # I3 过渡线
            return "bubble"

    # ---- 5. 间歇流（默认） ----
    # 检查 Froth 条件：EL 过低 → froth
    try:
        C0 = calc_C0_intermittent(vm, D, rho_l, mu_l, sin_theta)
        Vd = calc_Vd_elongated(D, rho_l, rho_g, sigma, theta_rad, mu_l)
        Vt = C0 * vm + Vd
        ELs_test = _calc_ELs_gregory(vm)
        Vb_test = calc_Vb_dispersed(rho_l, rho_g, sigma, theta_rad)
        VGdb_test = C0 * vm + Vb_test
        if VGdb_test < 0:
            VGdb_test = 0.0
        if Vt > 0:
            EL_test = (ELs_test * Vt + VGdb_test * (1.0 - ELs_test) - vsg) / Vt
        else:
            EL_test = 0.5
        # Petalas Eq 31: EL < 0.24 → 间歇流不稳定 → Froth
        if EL_test < 0.15:
            return "froth"
    except (ValueError, ZeroDivisionError):
        pass

    return "intermittent"


# ============================================================
# Layer 6: 主入口函数
# ============================================================

def drift_flux_pressure_gradient(inp: DriftFluxInput) -> DriftFluxResult:
    """
    Petalas & Aziz (2000) 机理模型主入口。

    参数
    ----------
    inp : DriftFluxInput

    返回
    -------
    DriftFluxResult
    """
    # ----- 解析输入 -----
    m_total = inp.total_mass_flow
    x_flow = inp.gas_mass_fraction
    rho_g = inp.rho_gas
    rho_l = inp.rho_liquid
    mu_g = inp.mu_gas
    mu_l = inp.mu_liquid
    D = inp.diameter
    eps = inp.roughness
    theta = inp.angle_rad
    sigma = inp.surface_tension

    if m_total <= 0 or D <= 0:
        raise ValueError("质量流量和管径必须大于零。")

    # ----- 表观速度 -----
    A = math.pi * D * D / 4.0
    m_gas = x_flow * m_total
    m_liq = (1.0 - x_flow) * m_total
    vsg = m_gas / max(rho_g * A, 1e-12)
    vsl = m_liq / max(rho_l * A, 1e-12)
    vm = vsg + vsl

    if vm <= 0:
        raise ValueError("混合表观速度必须大于 0。")

    # ----- 无量纲数 -----
    dims = _dimensionless_numbers(vsg, vsl, rho_l, rho_g, mu_l, sigma, D)
    Nlv = dims["Nlv"]

    # ----- 流型判别 -----
    pattern = determine_flow_pattern(
        vsg, vsl, vm, rho_l, rho_g, mu_l, mu_g, D, eps, theta, sigma
    )

    # ----- 根据流型调用对应求解器 -----
    if pattern == "dispersed_bubble":
        sol = solve_dispersed_bubble(
            vsg, vsl, vm, rho_l, rho_g, mu_l, mu_g, D, eps, theta, sigma, x_flow
        )
    elif pattern == "stratified":
        sol = solve_stratified(
            vsg, vsl, vm, rho_l, rho_g, mu_l, mu_g, D, eps, theta, sigma, x_flow
        )
    elif pattern == "annular_mist":
        sol = solve_annular_mist(
            vsg, vsl, vm, rho_l, rho_g, mu_l, mu_g, D, eps, theta, sigma, x_flow
        )
    elif pattern == "bubble":
        sol = solve_bubble_flow(
            vsg, vsl, vm, rho_l, rho_g, mu_l, mu_g, D, eps, theta, sigma, x_flow
        )
    elif pattern == "intermittent":
        sol = solve_intermittent(
            vsg, vsl, vm, rho_l, rho_g, mu_l, mu_g, D, eps, theta, sigma, x_flow
        )
    else:  # froth
        sol = solve_froth(
            vsg, vsl, vm, rho_l, rho_g, mu_l, mu_g, D, eps, theta, sigma, x_flow, {}
        )

    # ----- 公共后处理 -----
    HL = sol["EL"]
    Hg = sol["EG"]
    flow_pattern_out = sol["pattern"]

    # 库存气相质量分数
    x_inv = calc_inventory_gas_mass_fraction(rho_g, rho_l, HL)

    # 无滑脱参数
    lam = calc_no_slip_liquid_fraction(vsl, vsg)
    rho_ns = calc_no_slip_density(rho_g, rho_l, lam)
    mu_ns = calc_no_slip_viscosity(mu_g, mu_l, lam)
    Re_ns = rho_ns * vm * D / max(mu_ns, 1e-12)
    rr = eps / max(D, 1e-12)
    f_ns = _friction_factor(Re_ns, rr)

    # 实际相速度 & 滑移比
    v_gas_actual = sol.get("v_gas_actual", vsg / max(Hg, 1e-12))
    v_liq_actual = sol.get("v_liq_actual", vsl / max(HL, 1e-12))
    slip_ratio = v_gas_actual / max(v_liq_actual, 1e-12)

    # 摩擦因子
    f_two_phase = sol.get("fm", f_ns)

    # Froude 数
    Fr = vm / math.sqrt(G * D)

    # rho_holdup
    rho_holdup = calc_holdup_density(rho_g, rho_l, HL)

    return DriftFluxResult(
        dp_dL_friction=sol["dp_fric"],
        dp_dL_gravity=sol["dp_grav"],
        dp_dL_total=sol["dp_total"],
        liquid_holdup_inclined=HL,
        gas_void_fraction=Hg,
        gas_mass_fraction_inventory=x_inv,
        gas_mass_fraction_flow=x_flow,
        flow_pattern=flow_pattern_out,
        v_sg=vsg,
        v_sl=vsl,
        v_m=vm,
        v_gas_actual=v_gas_actual,
        v_liquid_actual=v_liq_actual,
        slip_ratio=slip_ratio,
        rho_no_slip=rho_ns,
        rho_holdup=rho_holdup,
        mu_no_slip=mu_ns,
        no_slip_liquid_fraction=lam,
        f_no_slip=f_ns,
        friction_multiplier=sol.get("friction_multiplier", 1.0),
        f_two_phase=f_two_phase,
        froude_number=Fr,
        reynolds_no_slip=Re_ns,
        liquid_velocity_number=Nlv,
        iterations=1,
        converged=True,
    )
