# xiao_model.py
"""
Xiao, Shoham & Brill (1990) 机理模型 — SPE 20631.

参考文献：
    Xiao, J.J., Shoham, O., and Brill, J.P. (1990). "A Comprehensive
        Mechanistic Model for Two-Phase Flow in Pipelines." SPE 20631.

适用范围：水平及近水平管线 (-15° ≤ α ≤ +15°)
流型：Stratified (Smooth/Wavy), Intermittent (Slug/Elongated Bubble),
       Annular, Dispersed Bubble

架构：
    Layer 0: 输入/输出 dataclass（与 segment_solver.py 兼容）
    Layer 1: 几何 & 通用工具（复用 drift_flux_model 私有函数）
    Layer 2: 流型判别 (Taitel-Dukler K-H 框架)
    Layer 3: 分层流求解器（双流体 + Andritsos/Baker 界面摩擦）
    Layer 4: 间歇流求解器（slug 单元模型）
    Layer 5: 环雾流求解器（Oliemans 夹带 + 界面摩擦）
    Layer 6: 分散气泡流求解器（均相模型）
    Layer 7: 主入口 xiao_pressure_gradient()
"""

from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Optional

from steady.phase_properties import (
    clamp, calc_inventory_gas_mass_fraction, calc_holdup_density,
    calc_no_slip_liquid_fraction, calc_no_slip_density, calc_no_slip_viscosity,
)
from steady.flow_pattern_candidates import (
    group_taitel_dukler_pattern,
    taitel_dukler_from_superficial_velocities,
)
from steady.mukherjee_brill import determine_flow_pattern_from_superficial_velocities

G = 9.80665

# Xiao (1990) reports friction factors in the Fanning convention.  This
# module keeps Darcy friction factors internally to match the other steady
# models, so fixed empirical friction constants are converted before use.
FANNING_TO_DARCY = 4.0
XIAO_FI_PATTERN_FANNING = 0.0142
XIAO_FI_PATTERN_DARCY = FANNING_TO_DARCY * XIAO_FI_PATTERN_FANNING
# 每个子区间仍以受限割线/二分精确求根；48 点粗扫描足以分离当前相关式的
# 连续物理解，同时避免在全线 P/T 固定点中重复做 200 点粗扫描。
XIAO_ROOT_SCAN_POINTS = 48


# ============================================================
# Layer 0: 输入 / 输出
# ============================================================

@dataclass
class XiaoInput:
    """与 DriftFluxInput / BeggsBrillInput 字段对齐"""
    total_mass_flow: float       # kg/s
    gas_mass_fraction: float     # 0~1
    rho_gas: float               # kg/m³
    rho_liquid: float            # kg/m³
    mu_gas: float                # Pa·s
    mu_liquid: float             # Pa·s
    diameter: float              # m
    roughness: float             # m
    angle_rad: float             # rad, 上坡为正
    surface_tension: float = 0.02  # N/m
    pressure_Pa: float = 101325.0   # 实际压力 Pa (Andritsos fi 需要)
    # 实验接口：将流型判别与各流型闭合拆开测试。
    flow_pattern_override: Optional[str] = None
    # 闪蒸精确单相极限：legacy 保持历史路由；darcy 使用单相 Darcy。
    single_phase_limit_closure: str = "legacy"
    # 流型图：xiao（基线）、mukherjee_brill（完整替换）、
    # xiao_mb_bubble（两种泡状判据的诊断性并集），或
    # xiao_mb_bubble_gate（仅以 MB 替换 Xiao 泡状边界）、
    # xiao_mb_bubble_from_intermittent（仅细分 Xiao 间歇流），或
    # xiao_td_bubble_from_intermittent（用独立 TD 图细分 Xiao 间歇流）。
    flow_pattern_closure: str = "xiao"
    # dispersed_bubble 下可选标准 Xiao，或 OLGA v3.8 漂移 + 不同壁面闭合。
    dispersed_bubble_closure: str = "xiao"
    # 分层流界面摩阻：Xiao、光滑界面、Petalas、OLGA 及混合闭合。
    stratified_interfacial_closure: str = "xiao"
    # 分层流壁面摩阻：Xiao（水力直径+实际相速+Colebrook）或论文闭合。
    stratified_wall_friction_closure: str = "xiao"
    # 间歇流平移速度：Xiao / OLGA / Petalas / Nicklin。
    intermittent_translation_closure: str = "xiao"
    # 液塞体持液率：Gregory 基线及公开文献中的替代关系。
    intermittent_slug_holdup_closure: str = "gregory1978"
    # 间歇流膜区界面摩阻；constant 保持 Xiao Eq.33。
    intermittent_film_interfacial_closure: str = "constant"
    # 段塞长度：Scott 1987 基线及文献替代关系。注意在 Xiao Eq.30/31 的
    # 比值加权结构中 Ls 本身是惰性量（Lu/Ls 由连续性闭式决定），只有
    # 配合 pressure_extra 中的单元频率项（如 Cook & Behnia 混合损失，
    # 显式含 1/Lu）才会影响压降；仅替换 Ls 关系是惰性对照。
    intermittent_slug_length_closure: str = "scott1987"
    # 间歇流压降附加项：none 保持 Xiao Eq.31 原状；cook_behnia2000 额外
    # 计入液膜被卷入液塞鼻部的混合（黏性耗散）损失 ΔP/Lu。
    intermittent_pressure_extra_closure: str = "none"
    # 环状流液滴夹带：oliemans / none / olga_generalized。
    annular_entrainment_closure: str = "oliemans"
    # 环状流膜-气芯界面摩阻：xiao / smooth_core / wallis。
    annular_interfacial_closure: str = "xiao"
    # 兼容旧接口的冗余字段
    sigma_water: float = 0.072
    rho_water: float = 1000.0
    rho_air: float = 1.29
    mu_water: float = 0.001


@dataclass
class XiaoResult:
    """与 DriftFluxResult / BeggsBrillResult 对齐"""
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
    debug_film_Ef: float = float("nan")
    debug_film_hL_tilde: float = float("nan")
    debug_film_root_count: int = 0
    debug_film_Vf: float = float("nan")
    debug_dp_slug: float = float("nan")
    debug_dp_film: float = float("nan")
    debug_Ls_over_Lu: float = float("nan")


# ============================================================
# Layer 1: 几何 & 通用工具（从 drift_flux_model 复用核心逻辑）
# ============================================================

def _stratified_geometry(hL_tilde: float, D: float) -> dict:
    """分层流几何参数。hL_tilde = hL/D ∈ (0,1)"""
    hL_tilde = clamp(hL_tilde, 1e-6, 1.0 - 1e-6)
    gamma = math.acos(1.0 - 2.0 * hL_tilde)
    A = math.pi * D * D / 4.0
    AL = D * D / 4.0 * (gamma - math.sin(gamma) * math.cos(gamma))
    AG = A - AL
    SL = gamma * D
    SG = (math.pi - gamma) * D
    Si = D * math.sin(gamma)
    DL = 4.0 * AL / max(SL, 1e-12)
    DG = 4.0 * AG / max(SG + Si, 1e-12)
    return {"AL": AL, "AG": AG, "A": A, "SL": SL, "SG": SG, "Si": Si,
            "DL": DL, "DG": DG, "hL": hL_tilde * D, "gamma": gamma, "hL_tilde": hL_tilde}


def _annular_geometry(delta_tilde: float, D: float) -> dict:
    """环雾流几何。delta_tilde = δ/D"""
    delta_tilde = clamp(delta_tilde, 1e-8, 0.5)
    A = math.pi * D * D / 4.0
    Af = A * 4.0 * delta_tilde * (1.0 - delta_tilde)
    Ac = A - Af
    Sf = math.pi * D
    Si = math.pi * (D - 2.0 * delta_tilde * D)
    Df = 4.0 * Af / max(Sf, 1e-12)
    Dc = 4.0 * Ac / max(Si, 1e-12)
    return {"Af": Af, "Ac": Ac, "A": A, "Sf": Sf, "Si": Si,
            "Df": Df, "Dc": Dc, "Ef": Af / A, "delta_tilde": delta_tilde}


def _bounded_unit_length_ratio(numerator: float, denominator: float) -> float:
    """Return the positive, bounded Xiao Eq.30 unit-length ratio ``Lu/Ls``."""
    if abs(denominator) <= 1.0e-12:
        return 20.0
    return clamp(abs(numerator / denominator), 1.0, 20.0)


def _bracketed_secant_bisection(
    function, low: float, high: float, f_low: float, f_high: float,
    *, tolerance: float = 1.0e-8, max_iter: int = 48,
) -> float:
    """在已知异号区间内求根，保留区间不变性。

    Xiao 的分层、液膜和环雾方程都可能有多根。论文建议取最小物理解，
    因而每个扫描得到的异号区间必须可靠收缩，不能让裸割线跳出区间。
    """
    if f_low == 0.0:
        return low
    if f_high == 0.0:
        return high
    if f_low * f_high > 0.0:
        raise ValueError("Xiao 内部求根未获得异号区间。")
    for _ in range(max_iter):
        width = high - low
        if width <= 1.0e-12:
            return 0.5 * (low + high)
        candidate = high - f_high * width / (f_high - f_low)
        # 受限割线；太靠近端点时二分，避免在陡峭膜厚区间停滞。
        if candidate <= low + 0.05 * width or candidate >= high - 0.05 * width:
            candidate = 0.5 * (low + high)
        f_candidate = function(candidate)
        if abs(f_candidate) <= tolerance:
            return candidate
        if f_low * f_candidate < 0.0:
            high, f_high = candidate, f_candidate
        else:
            low, f_low = candidate, f_candidate
    return 0.5 * (low + high)


# ============================================================
# 摩擦因子 — 全部使用 Darcy (Moody) 摩擦因子 f_D
#
#   f_D = 4 * f_F (Fanning)
#   层流: f_D = 64/Re
#   紊流: Xiao Eq. (14) 的 Fanning 形式，换算为 Darcy 口径
#   剪应力: τ = f_D * ρ * v² / 8  (等价于 f_F * ρ * v² / 2)
#   压降:   dp/dL = f_D * ρ * v² / (2D)
#
# Xiao 论文符号表使用 Fanning friction factor；本模块的固定经验摩擦
# 常数进入剪应力公式前要乘以 4，转为 Darcy 口径。
# ============================================================

def _friction_factor(re: float, rr: float) -> float:
    """Xiao Eq. (13)--(14) 对应的 Darcy 摩阻因子。

    原文使用 Fanning 因子 ``f_F``。令 ``f_D=4 f_F`` 后，Eq. (14) 变为
    ``1/sqrt(f_D)=1.74-2log10(2 eps/D+18.7/(Re sqrt(f_D)))``。
    """
    if re <= 0:
        return 0.0
    if re <= 2000:
        return 64.0 / re

    def residual(x: float) -> float:
        return x - 1.74 + 2.0 * math.log10(2.0 * rr + 18.7 * x / re)
    lo, hi = 1.0e-12, 100.0
    if residual(lo) >= 0.0 or residual(hi) <= 0.0:
        raise ValueError("Xiao Eq. (14) 摩阻因子未能建立正根区间。")
    # 此函数位于 200 点流型扫描的最内层。用有界 Newton 而非 80 次二分：
    # 保持正根区间的同时，典型工况 4--6 次便可达到机器精度。
    x = 7.0
    x = min(max(x, lo), hi)
    for _ in range(20):
        value = residual(x)
        if abs(value) <= 1.0e-13:
            break
        derivative = 1.0 + (2.0 / math.log(10.0)) * (18.7 / re) / (
            2.0 * rr + 18.7 * x / re
        )
        candidate = x - value / derivative
        if candidate <= lo or candidate >= hi or not math.isfinite(candidate):
            candidate = 0.5 * (lo + hi)
        value_candidate = residual(candidate)
        if value_candidate > 0.0:
            hi = candidate
        else:
            lo = candidate
        x = candidate
    return 1.0 / (x * x)


def _friction_factor_turbulent(re: float, rr: float) -> float:
    """按 Xiao Eq.13-14 选择层流或紊流 Darcy 摩擦因子。"""
    return _friction_factor(re, rr)


# ============================================================
# Layer 2: 流型判别 (Taitel-Dukler K-H 框架)
# ============================================================

def _solve_stratified_for_pattern(vsg: float, vsl: float,
                                   rho_l: float, rho_g: float,
                                   mu_l: float, mu_g: float,
                                   D: float, eps: float,
                                   theta_rad: float) -> dict:
    """
    流型判别用分层流求解 — 使用固定 fi=0.0142 Fanning
    (Shoham & Taitel 1984)，内部换算为 Darcy。
    返回 hL_tilde 和几何参数，若分层流无解则 hL_tilde=None。
    """
    A = math.pi * D * D / 4.0
    sin_theta = math.sin(theta_rad)
    drho = rho_l - rho_g
    rr = eps / max(D, 1e-12)
    Fi_const = XIAO_FI_PATTERN_DARCY

    def momentum_balance(hL_tilde: float) -> float:
        geom = _stratified_geometry(hL_tilde, D)
        AL_v, AG_v = geom["AL"], geom["AG"]
        VL = vsl * A / max(AL_v, 1e-12)
        VG = vsg * A / max(AG_v, 1e-12)

        # 气相摩擦
        Re_G = rho_g * abs(VG) * geom["DG"] / max(mu_g, 1e-12)
        fG = _friction_factor_turbulent(Re_G, rr)
        tau_wG = fG * rho_g * VG * abs(VG) / 8.0

        # 液相摩擦
        Re_L = rho_l * abs(VL) * geom["DL"] / max(mu_l, 1e-12)
        fL = _friction_factor_turbulent(Re_L, rr)
        tau_wL = fL * rho_l * VL * abs(VL) / 8.0

        # 界面摩擦（流型判别用常量）
        Vi = VG - VL
        tau_i = Fi_const * rho_g * Vi * abs(Vi) / 8.0

        term_L = tau_wL * geom["SL"] / max(AL_v, 1e-12)
        term_G = tau_wG * geom["SG"] / max(AG_v, 1e-12)
        term_i = tau_i * geom["Si"] * (1.0 / max(AL_v, 1e-12) + 1.0 / max(AG_v, 1e-12))
        term_g = drho * G * sin_theta
        return term_L - term_G - term_i + term_g

    # 扫描找根
    n_scan = XIAO_ROOT_SCAN_POINTS
    roots = []
    h_prev, f_prev = 1e-6, momentum_balance(1e-6)
    for i in range(1, n_scan + 1):
        h_curr = i / n_scan * (1.0 - 1e-6)
        f_curr = momentum_balance(h_curr)
        if f_prev * f_curr <= 0:
            roots.append(_bracketed_secant_bisection(
                momentum_balance, h_prev, h_curr, f_prev, f_curr
            ))
        h_prev, f_prev = h_curr, f_curr

    if not roots:
        return {"hL_tilde": None, "geom": None}
    hL_tilde = min(roots)
    return {"hL_tilde": hL_tilde, "geom": _stratified_geometry(hL_tilde, D)}


def _check_dispersed_bubble_xiao(vsl: float, vsg: float, D: float,
                                  rho_l: float, rho_g: float, mu_l: float,
                                  mu_g: float, eps: float, theta_rad: float) -> bool:
    """
    分散气泡流判据 — Xiao Eq 3 (Taitel-Dukler 湍流破坏气泡)。

    VL > 4*Ag/(Si*fL) * sqrt(g*D*cosα*(1 - ρg/ρL))
    """
    if vsl <= 0 or vsg <= 0:
        return False
    cos_theta = math.cos(theta_rad)
    if cos_theta < 0.02:
        cos_theta = 0.02

    # 需要 hL/D 来评估 Ag, Si, fL — 使用假设分层流解
    sol = _solve_stratified_for_pattern(
        vsg, vsl, rho_l, rho_g, mu_l, mu_g, D, eps, theta_rad
    )
    if sol["hL_tilde"] is None:
        # 回退：使用 no-slip 持液率估计 hL (分散气泡流滑差小，λ_L 接近 hL)
        lam_L = clamp(vsl / max(vsg + vsl, 1e-12), 0.01, 0.99)
        hL_est = lam_L
        geom = _stratified_geometry(hL_est, D)
    else:
        geom = sol["geom"]
    A = geom["A"]
    Ag = geom["AG"]
    Si = geom["Si"]
    rr = eps / max(D, 1e-12)
    VL = vsl * A / max(geom["AL"], 1e-12)
    Re_L = rho_l * abs(VL) * geom["DL"] / max(mu_l, 1e-12)
    fL_darcy = _friction_factor_turbulent(Re_L, rr)
    fL_fanning = fL_darcy / FANNING_TO_DARCY

    if Si <= 0 or fL_fanning <= 0:
        return False
    V_crit = math.sqrt(
        4.0 * Ag * G * cos_theta
        * (1.0 - rho_g / rho_l)
        / (Si * fL_fanning)
    )
    return VL > V_crit


def _determine_flow_pattern_xiao(
    vsg: float, vsl: float, vm: float,
    rho_l: float, rho_g: float, mu_l: float, mu_g: float,
    D: float, eps: float, theta_rad: float, sigma: float,
    check_dispersed_bubble: bool = True,
) -> str:
    """
    Xiao-Brill 流型判别 (Taitel-Dukler K-H 框架)。

    流程：
        分散气泡? → 求解假设分层流 hL/D (fi=0.0142) →
        K-H 不稳定? → hL/D<0.35? → annular : intermittent
        K-H 稳定?   → stratified
    """
    # 0. 倾角范围检查
    theta_deg = math.degrees(theta_rad)
    if abs(theta_deg) > 15.0:
        pass  # 超出 Xiao 适用范围，仍计算但结果可能不准确

    # 1. 分散气泡流
    if check_dispersed_bubble and _check_dispersed_bubble_xiao(
        vsl, vsg, D, rho_l, rho_g, mu_l, mu_g, eps, theta_rad
    ):
        return "dispersed_bubble"

    # 2. 分层流假设解 (fi=0.0142, 按 Xiao 论文要求)
    sol = _solve_stratified_for_pattern(vsg, vsl, rho_l, rho_g, mu_l, mu_g, D, eps, theta_rad)
    hL_tilde = sol["hL_tilde"]

    if hL_tilde is None or hL_tilde <= 0.001 or hL_tilde >= 0.999:
        # 分层流无解 — 使用 no-slip 持液率估计 hL 并检查重力波判据 (Eq 5)
        lam_L = vsl / max(vm, 1e-12)
        if theta_rad < -0.001:
            # 下坡：估计 hL (考虑滑移使液膜变薄)，检查重力波
            hL_est = clamp(lam_L * 0.8, 0.005, 0.5)
            geom_est = _stratified_geometry(hL_est, D)
            VL_est = vsl * geom_est["A"] / max(geom_est["AL"], 1e-12)
            Fr_L = abs(VL_est) / math.sqrt(max(G * geom_est["hL"], 1e-12))
            if Fr_L > 1.5:
                return "stratified_wavy"
            return "stratified"
        # 上坡/水平无解 → intermittent
        return "intermittent"

    geom = sol["geom"]
    A = geom["A"]
    VL = vsl * A / max(geom["AL"], 1e-12)
    VG = vsg * A / max(geom["AG"], 1e-12)
    drho = rho_l - rho_g
    cos_theta = abs(math.cos(theta_rad))
    if cos_theta < 0.02:
        cos_theta = 0.02

    # K-H 不稳定性临界气速 (Eq 1)
    dAL_dhL = geom["Si"]
    if dAL_dhL > 0 and rho_g > 0:
        VG_crit_KH = (1.0 - hL_tilde) * math.sqrt(
            drho * G * geom["AG"] * cos_theta / (rho_g * dAL_dhL)
        )
    else:
        VG_crit_KH = float("inf")

    if VG < VG_crit_KH:
        # 分层流稳定
        if theta_rad < -0.001:
            # 下坡：检查重力波 (Eq 5)
            Fr_L = abs(VL) / math.sqrt(max(G * geom["hL"], 1e-12))
            if Fr_L > 1.5:
                return "stratified_wavy"
        # 再分 smooth/wavy (3.4)
        s = 0.06
        if VL > 0 and rho_g > 0:
            VG_wavy = math.sqrt(
                4.0 * mu_l * drho * G * cos_theta / (s * rho_l * rho_g * VL)
            )
        else:
            VG_wavy = float("inf")
        if VG > VG_wavy:
            return "stratified_wavy"
        return "stratified_smooth"

    # 非分层流
    # 3. 间歇→环雾 (Eq 2, hL/D < 0.35)
    if hL_tilde < 0.35:
        return "annular"
    return "intermittent"


def determine_flow_pattern(
    vsg: float, vsl: float, vm: float,
    rho_l: float, rho_g: float, mu_l: float, mu_g: float,
    D: float, eps: float, theta_rad: float, sigma: float,
    closure: str = "xiao",
) -> str:
    """Select a flow-pattern map independently from the hydraulic closures.

    ``xiao_mb_bubble`` is deliberately a research hybrid: it only changes the
    Xiao result when the independent Mukherjee--Brill map identifies bubble
    flow.  It therefore tests whether Xiao's conservative dispersed-bubble
    boundary is the main classification bottleneck without replacing the
    stratified/intermittent/annular logic at the same time.

    ``xiao_mb_bubble_gate`` is the stricter closure replacement: the MB
    bubble boundary replaces (rather than augments) Xiao's bubble check.  If
    MB does not predict bubble flow, Xiao continues from its stratified/K-H
    tests with the original bubble check disabled.

    ``xiao_mb_bubble_from_intermittent`` is a hierarchical research routing
    rule.  It preserves Xiao separated-flow decisions and uses the MB bubble
    boundary only to split Xiao's intermittent region.  It is an engineering
    hybrid, not a correlation published by either paper.

    ``xiao_td_bubble_from_intermittent`` applies the same conservative routing
    rule with the independent Taitel--Dukler map.  It is retained as a research
    switch after classification testing; the default remains ``xiao``.
    """
    key = closure.strip().lower()
    if key not in {
        "xiao", "mukherjee_brill", "xiao_mb_bubble",
        "xiao_mb_bubble_gate", "xiao_mb_bubble_from_intermittent",
        "xiao_td_bubble_from_intermittent",
    }:
        raise ValueError(f"未知 Xiao 流型判别闭合: {closure}")

    xiao_pattern = _determine_flow_pattern_xiao(
        vsg, vsl, vm, rho_l, rho_g, mu_l, mu_g,
        D, eps, theta_rad, sigma,
    )
    if key == "xiao":
        return xiao_pattern

    # The alternative maps are defined only for genuine two-phase flow. During
    # pipeline flash iterations a phase can disappear locally; preserve Xiao's
    # existing limiting behavior instead of turning that state into a failure.
    if vsg <= 0.0 or vsl <= 0.0:
        return xiao_pattern

    if key == "xiao_td_bubble_from_intermittent":
        td_result = taitel_dukler_from_superficial_velocities(
            vsg=vsg, vsl=vsl, rho_g=rho_g, rho_l=rho_l,
            mu_g=mu_g, mu_l=mu_l, diameter=D, roughness=eps,
            theta_rad=theta_rad,
        )
        td_pattern = group_taitel_dukler_pattern(td_result.pattern)
        if xiao_pattern == "intermittent" and td_pattern == "dispersed_bubble":
            return "dispersed_bubble"
        return xiao_pattern

    mb_pattern = determine_flow_pattern_from_superficial_velocities(
        vsg=vsg, vsl=vsl, rho_l=rho_l, mu_l=mu_l,
        sigma=sigma, theta_rad=theta_rad,
    )
    normalized = {
        "bubble": "dispersed_bubble",
        "slug": "intermittent",
        "annular-mist": "annular",
        "stratified": "stratified",
    }[mb_pattern]
    if key == "mukherjee_brill":
        return normalized
    if key == "xiao_mb_bubble_from_intermittent":
        if xiao_pattern == "intermittent" and normalized == "dispersed_bubble":
            return "dispersed_bubble"
        return xiao_pattern
    if key == "xiao_mb_bubble" and normalized != "dispersed_bubble":
        return xiao_pattern
    if normalized == "dispersed_bubble":
        return "dispersed_bubble"
    return _determine_flow_pattern_xiao(
        vsg, vsl, vm, rho_l, rho_g, mu_l, mu_g,
        D, eps, theta_rad, sigma, check_dispersed_bubble=False,
    )


# ============================================================
# Layer 3: 分层流求解器
# ============================================================

def _stratified_fi_andritsos(vsg: float, hL_tilde: float,
                              fG: float, P_Pa: float) -> float:
    """
    Andritsos & Hanratty (1987) 界面摩擦 — Xiao Eq 16-18, 适用于 D ≤ 0.127m。

    Vsg,t = 5 * sqrt(101325 / P)
    若 Vsg ≤ Vsg,t: fi = fG
    若 Vsg > Vsg,t: fi/fG = 1 + 15*sqrt(hL/D)*(Vsg/Vsg,t - 1)
    """
    Vsg_t = 5.0 * math.sqrt(101325.0 / max(P_Pa, 1e-6))
    vsg_ref = abs(vsg)
    if vsg_ref <= Vsg_t:
        return fG
    ratio = 1.0 + 15.0 * math.sqrt(max(hL_tilde, 1e-6)) * (vsg_ref / Vsg_t - 1.0)
    return fG * ratio


def _interfacial_fi_updated_separated(
    VG: float,
    VL: float,
    geom: dict,
    rho_l: float,
    rho_g: float,
    mu_l: float,
    mu_g: float,
    D: float,
    eps: float,
) -> float:
    """
    更新后的相间摩阻闭合关系。

    这里采用仓库内已有的 Petalas 型分层界面摩擦闭合思路，统一替换
    Xiao 原先分层/膜区的 Andritsos/Baker 固定式，避免在低持液率和
    强滑移工况下出现过强或不稳定的界面剪切。
    """
    if rho_g <= 0:
        return 0.0

    rr = eps / max(D, 1e-12)
    Re_G = rho_g * abs(VG) * geom["DG"] / max(mu_g, 1e-12)
    fG = _friction_factor_turbulent(Re_G, rr)

    hL = max(geom["hL"], 1e-12)
    Fr_L = abs(VL) / math.sqrt(max(G * hL, 1e-12))
    # Petalas & Aziz (2000) Eq 11：Re_SL 用液相表观速度 V_SL（drift_flux_model
    # .calc_interfacial_friction_stratified 同口径）。此处由实际液速反推：
    # V_SL = V_L * A_L / A。旧版误用实际液速 |V_L|，Re 偏大 ~1/E_L 倍。
    vsl_equiv = abs(VL) * geom["AL"] / max(geom["A"], 1e-30)
    Re_SL = rho_l * vsl_equiv * D / max(mu_l, 1e-12)
    if Fr_L > 0.0 and abs(VG) > 1e-12:
        fi = (0.004 + 0.5e-6 * Re_SL) * (Fr_L ** 1.335) * (
            rho_l * D * G / max(rho_g * VG * abs(VG), 1e-12)
        )
    else:
        fi = fG
    return clamp(fi, 0.0, 10.0)


def _stratified_fi_baker(VG: float, VL: float, hL: float, D: float,
                         eps: float, rho_g: float, mu_g: float,
                         mu_l: float, sigma: float, rho_l: float,
                         *, DG=None) -> float:
    """Baker 大管径界面摩擦，按 Xiao Eq.19-22。

    DG 传入气层水力直径（geom["DG"]）时 Re_i 取 Eq.15 口径
    （Reg=ρg·vg·Dg/μg, Dg=4Ag/(Sg+Si)；轮二审计 MED CONFIRMED, 2026-10-22）。
    rr_i 恒为 Ei/D（管径）——与壁面基线同约定（Re 用水力直径、粗糙度比
    用管径），且 Ei≤0.25hL≤0.25D 保证 Colebrook 恒有正根。注意：生产筛查
    旧码 rr_i=Ei/Dg 在 h_t→1 时 Dg→0、rr→∞（无正根，上限 rr≈3.71），旧码
    靠不动点迭代接受负 1/sqrt(f) 伪收敛过关（见 closure_extraction_record.md
    §一.4 勘误），本实现不沿用该口径。"""
    VG_abs = abs(VG)
    if VG_abs < 1e-6 or rho_g <= 0:
        return XIAO_FI_PATTERN_DARCY

    VL_abs = max(abs(VL), 1e-8)
    nwe_nmu = (
        rho_g * VL_abs * VL_abs * mu_l * mu_l
        / max(rho_l * sigma * sigma, 1e-20)
    )
    scale = sigma / max(rho_g * VL_abs * VL_abs, 1e-20)
    if nwe_nmu <= 0.005:
        Ei = 34.0 * scale
    else:
        Ei = 170.0 * scale * nwe_nmu ** 0.3

    lower = max(eps, 1e-9)
    upper = max(0.25 * hL, lower)
    Ei = clamp(Ei, lower, upper)

    # Colebrook 式求 fi (Darcy 摩擦因子)。Re_i 特征长度：缺省管径 D（战役
    # 基线，已发布跑批数字的口径）；DG 传入时取气层水力直径（Eq.15 口径）。
    # rr_i 恒为 Ei/D（管径）——Eq.14 粗糙度项 ε/D vs ε/Dg 原文 OCR 不可辨
    # （UNVERIFIABLE），按壁面基线同约定取 /D（Ei≤0.25D 恒有正根；
    # 旧码 /Dg 口径伪收敛，见 docstring）。
    Re_len = D if DG is None else DG
    Re_i = rho_g * VG_abs * Re_len / max(mu_g, 1e-12)
    rr_i = Ei / max(D, 1e-12)
    fi = _friction_factor_turbulent(Re_i, rr_i)
    return fi


def _stratified_fi_xiao(
    *,
    vsg: float,
    VG: float,
    VL: float,
    hL_tilde: float,
    geom: dict,
    fG: float,
    rho_l: float,
    rho_g: float,
    mu_l: float,
    mu_g: float,
    D: float,
    eps: float,
    sigma: float,
    pressure_Pa: float,
) -> float:
    """Xiao Eq. (16)--(22) 的分层流界面摩阻（Darcy 口径）。

    小管径采用 Andritsos--Hanratty，大管径采用 Baker；两者均不同于
    流型判别阶段使用的常数 ``fi=0.0142``（Fanning）。
    """
    if D <= 0.127:
        return _stratified_fi_andritsos(vsg, hL_tilde, fG, pressure_Pa)
    return _stratified_fi_baker(
        VG, VL, geom["hL"], D, eps, rho_g, mu_g, mu_l, sigma, rho_l
    )


def _stratified_fi_olga_slai(
    *, VG: float, VL: float, geom: dict, rho_g: float,
    D: float, sigma: float,
) -> float:
    """OLGA 五流体蒸馏得到的 SLAI 关系，返回 Darcy 等效因子。"""
    delta_u = abs(VG - VL)
    if delta_u <= 1.0e-12 or rho_g <= 0.0:
        return 0.0
    velocity_scale = max(rho_g * delta_u / 8.0, 1.0e-30)
    depth_ratio = max(D / max(2.0 * geom["hL"], 1.0e-12), 1.0e-30)
    slai = math.exp(
        1.717 * math.log(velocity_scale)
        - 1.221 * math.log(depth_ratio)
        - 0.103 * math.log(max(sigma, 1.0e-30))
        - 2.043
    )
    return clamp(8.0 * slai / (rho_g * delta_u), 0.0, 10.0)

def _stratified_fi_olga_slai_h1(
    *, VG: float, VL: float, geom: dict, rho_g: float,
    D: float, sigma: float,
) -> float:
    """olga_slai 的 **H1 忠实语义对照变体**（轮二审计 HIGH CONFIRMED 后增设，
    2026-10-22）。SLAI 律四系数与速度尺度与 :676 完全同源，仅换算语义不同。

    基线式 clamp(8·slai/(ρg·ΔU)) 实现的是蒸馏战役 Phase-12 终判已**否决**的
    H0（τ_i = SLAI·ΔU）——slai/(ρg·ΔU) 量纲 1/m，漏乘几何因子 A/Si 与转换
    系数 K_TI。H1 正确语义（SLAI 为单位管体积界面动量传递系数 [kg/m³/s]，
    OLGA 手册量纲 + 洁净带 n=200 复算 med 1.044 [1.020,1.116] 双证）：

        SLAI·ΔU = τ_i·Si/A_pipe = fi·ρg·ΔU²·Si/(8A)
        ⇒  fi_H1 = 8·K_TI·SLAI·A/(Si·ρg·ΔU)，K_TI = 1.0437

    H0 相当于把 fi 抬高 Si/(K_TI·A)=4sinγ/(1.0437·πD) 倍（全库管径带
    1.5~24 倍，随管径单调）——已发布的 olga_slai 战绩（全库 H_mae 0.0614）
    系 H0 产物（closure_extraction_record.md §四/§五 该行需语义注记）。
    本变体重跑对照，**不静默改数**；clamp(0,10) 沿用仓库约定（工程护栏，
    非源式内容）。
    """
    delta_u = abs(VG - VL)
    if delta_u <= 1.0e-12 or rho_g <= 0.0:
        return 0.0
    velocity_scale = max(rho_g * delta_u / 8.0, 1.0e-30)
    depth_ratio = max(D / max(2.0 * geom["hL"], 1.0e-12), 1.0e-30)
    slai = math.exp(
        1.717 * math.log(velocity_scale)
        - 1.221 * math.log(depth_ratio)
        - 0.103 * math.log(max(sigma, 1.0e-30))
        - 2.043
    )
    return clamp(
        8.0 * 1.0437 * slai * geom["A"] / max(geom["Si"], 1.0e-30)
        / (rho_g * delta_u), 0.0, 10.0)


def _stratified_fi_andreussi(
    *, VG: float, geom: dict, rho_l: float, rho_g: float,
    theta_rad: float, fG: float,
) -> float:
    """Andreussi & Persen (1987) 界面摩阻 —— fi/fG 比值型，返回 Darcy 等效因子。

    原文（IJMF 13(4):565-575）Eq [16]/[17]：界面粗糙度以 hL/D 与 (F − F0) 描述，
    F = F0 时 fi ≈ fG；p9 最佳拟合式（OCR 逐字 `f--: = 1 + 29.7 (F - 0.36) °-67`，
    分母行 `fs`）即

        fi/fG = 1 + 29.7 · (F − F0)^0.67 ,  F > F0 ;  fi/fG = 1 ,  F ≤ F0

    其中 F0 = 0.36（p8 "This group, with F0 = 0.36, gives the best fit to the present data"）。
    无量纲数 F 的定义（p5 Eq [10]，本文自身对渠道判据 [8] 的管流推广 [9] 的无量纲化）：

        F = u_G · sqrt( ρ_G · (dA_L/dh_L) / ( (ρ_L − ρ_G) · A_G · g · cos θ ) )

    溯源：Eq [8] 为渠道稳定性判据（p5 "the stability criterion for flow in a channel"），
    p5 "For flow in a pipe, [8] can be modified into" → Eq [9]：将渠道气层高度 h_G 换成管中
    平均气层高度 A_G/(dA_L/dh_L)。故 A_G/(dA_L/dh_L) 即"平均气层厚度"，而非几何弦长。
    与 [9] 互为倒数。渠道情形 A_G/(dA_L/dh_L) ≡ h_G，F 退化回 Wallis–Dobson 渠道 Froude 数；
    论文自述其判据值 F = 0.44 "only slightly lower than ... Wallis & Dobson (1973), F = 0.5,
    for the transition to slug flow in a horizontal channel"，两常数互相印证。
    量纲自检：sqrt(m²·s⁻²/m²)·(m/s) = 1 ✓

    口径：比值型，Fanning 与 Darcy 同值，**不需要 ×4 换算**（模型 fi 槽为 Darcy，
    τi = fi·ρG·Vi|Vi|/8）。

    几何：`geom["Si"] = dA_L/dh_L` 由圆管分层几何恒等成立
    （A_L = ∫w dh，hL_tilde = hL/D ⇒ dA_L/dh_L = D·sin γ = geom["Si"]），
    故 dA_L/dh_L 直接取自模型自身的几何函数，与持液率口径天然一致。

    说明：R(F) 为 (F − F0)^0.67 的整式幂，F > F0 后单调无上界；比值 R 本体
    无截断，返回值在 :764 设 1e4 数值上限（激活例实测 fi/fG 最大 ~45×fG，
    clamp 实际不可达），另以 `_capped_{n}fg` 变体在算例筛选中检验截断敏感性。
    原文另有自述局限
    （p9）："Due to the limited set of data on which this equation is based, any
    generalization or the use of [17] for different fluid pairs or flow geometries
    may lead to appreciable errors."
    """
    f0 = 0.36
    cos_theta = math.cos(theta_rad)

    # 管流推广的稳定性判据 [9]/[10] 仅在 cosθ > 0（非近垂直）时有物理意义；
    # 竖直/倒流工况退回光滑界面，原文适用范围（下倾 0.65°/2.1°）亦不含此区。
    if cos_theta <= 1.0e-9:
        return fG

    a_g = geom["AG"]
    a_l = geom["AL"]
    if a_g <= 1.0e-30 or a_l <= 1.0e-30 or rho_g <= 0.0:
        return fG

    dadh = geom["Si"]          # = dA_L/dh_L（圆管分层几何恒等式）
    if dadh <= 1.0e-30:
        return fG
    h_bar_g = a_g / dadh       # 平均气层厚度，与 Eq [9]/[10] 中 A_G/(dA_L/dh_L) 同式

    drho = rho_l - rho_g
    if drho <= 0.0:
        return fG

    # 2026-10-21 审计勘误：此前分子多乘 dadh（F 被压低 2.3–12.5 倍、残留 √m 量纲，
    # Eq[17] 激活被压到 3.6–8.3%）。正确式（docstring :708 的论文 Eq[10]）：
    # dadh 与 h̄_g 只取其一。
    f_val = abs(VG) * math.sqrt(max(
        rho_g / max(drho * h_bar_g * G * cos_theta, 1.0e-30), 0.0))
    if f_val <= f0:
        return fG

    # Eq [17]：fi/fG = 1 + 29.7 · (F − F0)^0.67
    return clamp(fG * (1.0 + 29.7 * (f_val - f0) ** 0.67), 0.0, 1.0e4)


def _kowalski_liquid_area_fraction(hL_tilde: float) -> float:
    """圆管分层流液相面积份额 EL = (1 − alpha)，即 Kowalski 式 25 的 (1 − a)。

    (1 − a) = EL = (gamma − sin gamma · cos gamma) / pi ，gamma = acos(1 − 2·hL_tilde)。
    这正是分层几何的精确圆弓形面积份额（EL = AL/A），与模型自身几何函数一致；
    已用两路独立证据锁定该口径（详见 closure_extraction_record.md）：
      1) 原文 p6 Notation 逐字 `a = void fraction`，并给出
         `j_g = U_g · a ; j_L = U_L · (1 − a)`（`(1 − a)` 的减号与下标字母均经墨迹判读），
         自洽性要求 a 为含气率 ⇒ (1 − a) 为**液相**面积份额；
      2) Table 1 的 2×4 口径×惯用法数值表里（τ_i 经 Eq 24 相对速度换算为 f_i 的
         定案基准复算，2026-10-21），col8 + EL 为全域最优（spread 1.50，8 行
         C/7.5e-5 = 0.71–1.06；八组合中位 0.90–1.39。早期以 ug² 为基准的版本
         spread 1.636、中位 0.773–1.229，裁定结论一致，数值已被定案基准取代）。

    边界：hL_tilde→0 或 →1 时 gamma→0 或 pi，EL→0 或 1；对 0^(−0.25) 取最大值保护。
    """
    h = min(max(hL_tilde, 0.0), 1.0)
    gamma = math.acos(min(max(1.0 - 2.0 * h, -1.0), 1.0))
    return max((gamma - math.sin(gamma) * math.cos(gamma)) / math.pi, 1.0e-12)


def _stratified_fi_kowalski(
    *, VG: float, VL: float, hL_tilde: float, rho_g: float, rho_l: float,
    mu_g: float, mu_l: float, D: float,
) -> float:
    """Kowalski (1987) 界面摩阻式 25 —— 返回 **Darcy 等效因子**。

    原文（AIChE Journal, 33(2):274–281, 1987；2026-10-21 审计勘误——
    原注 "Exp. Thermal & Fluid Sci. 1(1):73-82" 系错误卷期，PDF 页眉核实）:
      Eq 24: f_i = 2·tau_i / [rho_g·(U_g − U_l)^2]   —— Fanning，且**以相对速度**为基准
      Eq 25: f_i = 7.5e-5 · (1 − a)^(−0.25) · Re_g^(−0.3) · Re_l^(0.83)
      适用范围（原文自述）: 22,600 < Re_g < 430,600 的**分层流**

    三条口径（均已逐字/墨迹核实，推导见 closure_extraction_record.md）：
      * 速度基准 = U_g − U_l（**相对速度**，非 U_g）。这是 Eq 24 的原始定义，
        与模型内部 `Vi = VG − VL` 完全一致。
      * Re 的**特征速度取实际相速度**（Re_g = U_g·D/nu_g），特征长度取**管径 D**，
        而非模型几何里的水力直径 geom["DG"]。依据是 p6 Notation 逐字区分
        `j = superficial velocity` 与 `U = average longitudinal velocity of phase`，
        而式 25 的 Re 用 U；原文明载 `Re_g = U_g·D/nu_g`。
      * (1 − a) = EL（液相面积份额），见 `_kowalski_liquid_area_fraction`。

    换算：tau_i = fi·rho_g·Vi|Vi|/8（模型的 Darcy 槽）。代入 Eq 24 的
      tau_i = f_i·rho_g·(U_g − U_l)^2/2 与 Vi = VG − VL 恒等，得

          fi_model = 4 · f_i

    ——**裸 ×4**，不引入 (VG/|Vi|)^2 之类的速度比因子（该因子曾是我的错误推导，
    已作废）；helper 自身无 clamp（第 857 行 `Vi*|Vi|` 已自带符号与 Vi→0 极限），
    选择器层（:892/:895）另设 1e4 数值上限与 `_capped_3fg` 的 min(3·fG) 档
    （登记痕迹，闸门 22600<Re_g<430600 下两 clamp 均数值惰性）。
    这与模块内 `FANNING_TO_DARCY = 4.0` 及 `_annular_interfacial_fi` 的
    `4.0 * 0.005 * ...` 先例一致。

    适用范围之外退回光滑界面 fG（由调用方给出，见选择器）：原文式 25 的
    回归样本仅覆盖 22,600 < Re_g < 430,600；超出区间时 (1−a)^(−0.25)·Re^(−0.3)
    的外推无依据，按"没数据不许伪造"原则不臆造延伸值。
    """
    if rho_g <= 0.0:
        return 0.0
    re_g = abs(VG) * D / max(mu_g / rho_g, 1.0e-30)   # = U_g·D/nu_g
    if not (22600.0 < re_g < 430600.0):
        return 0.0                                     # 超出适用范围 -> 由选择器退回 fG
    re_l = abs(VL) * D / max(mu_l / rho_l, 1.0e-30)    # = U_l·D/nu_l
    if re_l <= 0.0:
        return 0.0
    one_minus_alpha = _kowalski_liquid_area_fraction(hL_tilde)
    fi_k = (7.5e-5 * one_minus_alpha ** (-0.25)
            * re_g ** (-0.3) * re_l ** 0.83)
    return 4.0 * fi_k                                  # Fanning -> Darcy（裸 ×4）


# --- Tzotzi & Andritsos (2013)：水/空气参考态常数（闸门 Eq 18/19 用） --------
# 论文只声明参考态、未给数值表（期刊 p.48 逐字 "the subscripts W and A refer
# to water and air (at 1 atm and 20 °C), respectively."）→ 钉为 1 atm/20 °C
# 标准物性（显式假定项，随提取记录落档）。备选=模块 legacy 兼容字段值
# （见 XiaoInput.rho_air=1.29 等）：经 (ρG/ρA)^-0.5 使两闸门偏移 ≤±3.7%，
# 只移动分段边界、不改变分段结构——按"单位对齐以论文声明为准"取 20 °C 值。
TZOTZI_RHO_W = 998.2      # kg/m³，水 @ 1 atm/20 °C
TZOTZI_RHO_A = 1.204      # kg/m³，干空气 @ 1 atm/20 °C
TZOTZI_MU_W = 1.002e-3    # Pa·s，水 @ 20 °C
TZOTZI_SIGMA_W = 72.8e-3  # N/m，水—空气 @ 20 °C


def _stratified_fi_tzotzi2013(*, vsg: float, vsl: float, hL_tilde: float,
                              fG: float, rho_l: float, rho_g: float,
                              mu_l: float, sigma: float) -> float:
    """Tzotzi & Andritsos (2013) 分层流界面摩阻 —— 三段式 fi/fG，Darcy 输出。

    论文：IJMF 54 (2013) 43–54（水平管波状分层流界面剪切）。fi-only 替换：
    论文 Section 6 第 3 步自带的壁面闭合（Andritsos–Hanratty h+/σC, Eq 28+）
    与几何式 Eqs 22-27 不随本变体引入（判型与其余槽位不动）。

    口径链（单位对齐审计项，逐条落提取记录）：
      * τ 定义（p.45 Eq 6）τ_i = f_i·ρG·(UG−UL)|·|/2 为 **Fanning**；本
        模块 Darcy 槽 τ_i = fi·ρG·(VG−VL)|·|/8。论文给的是**比值 R** =
        fi/fG → 输出 fi_darcy = R·fG_darcy：×4 被同用消费端 fG 吸收
        （andreussi1987 同型比值先例；非 `_stratified_fi_kowalski` 裸 ×4）。
      * 速度口径 = **表观**流速 UGS/ULS、单位 m/s：闸门为经验拟合、量纲
        不齐次，0.8/1.39 按 m/s 标定（量纲自洽推断项，论文未明示单位）。
      * μ 口径 = 动力粘度（Fig 3 拟合轴 mPa·s，与库字段一致）；论文 Eq 16
        一处声明 "kinematic" 属内部不一致——运动粘度替代读法 = 动力比
        ×(ρW/ρL)（≈1.0–1.4），记为口径假定项，不采纳为主读。
      * 参考常数 TZOTZI_*：论文未给数值表 → 钉 1 atm/20 °C 标准值
        （见常量块注释的备选值敏感性）。

    三段判据（Section 6 设计流程 p.49 与正文 Eq 18-21 两处互证；
    速度均表观 m/s）：

      UGS,2−D (Eq 18) = (1/1.95)·(ρW/ρL)^−0.1·(ρG/ρA)^−0.5·(μL/μW)^0.35
                        ·ln[(0.8/ULS)·(μL/μW)^0.2]
      UGS,K−H (Eq 19) = (1/0.65)·(ρG/ρA)^−0.5·(ρW/ρL)^−0.5·(σW/σL)^−0.33
                        ·ln[(1.39/ULS)·(μL/μW)^−0.15]
      R (Eq 20, 2-D)  = 1 + 0.35·(h/D)^0.5·(UGS − UGS,2−D)
      R (Eq 21, K-H)  = 2·(h/D)^0.1·(μL/μW)^0.1 + 4·(h/D)^0.5·(UGS−UGS,K−H)

    ln 宗量分组为**兄弟乘积式**（(0.8/ULS) 整体乘 (μL/μW)^0.2）：Eq 18
    原式 p.48 与 Section 6 重述两处文本互证 + 负阈值物理不自洽四证链
    裁定；工作组 compound-denominator 读法已否决，差异随提取记录落档。

    边界与退化：
      * ULS→0⁺ 时 ln→+∞、闸门→+∞ → 光滑段（数学自洽极限）；ULS=0、
        UGS=0 或 μL=0 显式退回 fG（光滑界面）。
      * ln 宗量 <1 时闸门可为负 → 该段恒激活（论文无域约束，Section 6
        流程无条件套用；非异常）。
      * μL ≳ 20 mPa·s 时 UGS,2−D ≥ UGS,K−H → 2-D 段消失（论文原话
        "for liquid viscosities higher than about 20 mPa s this kind of
        waves does not appear at all"），由分段结构自然给出（无 2-D 窗口）。
      * R 各段 ≥1（两增量项非负）→ fi ≥ fG，无负值路径。
      * 无倾角硬闸门：公式 θ 无关；论文自验证 0°/−0.25°/−1°、Boussens
        验证至 −5°，库内 ±90° 属外推——由驱动侧逐工况包络标记呈现
        （Fig 6：D 25–500 mm、μL 1–90 mPa·s、ρG 0.5–5、ρL 1000–7000
        kg/m³、σ 20–100 mN/m；Fig 3 的 Eq-18 窄域 μL 1–19 mPa·s、
        ρL 800–3000），不在本式内静默回退。
    """
    if vsl <= 0.0 or vsg <= 0.0 or mu_l <= 0.0:
        return fG                        # 退化解：闸门→∞ → 光滑界面
    rho_l_s = max(rho_l, 1.0e-12)
    mu_ratio = mu_l / TZOTZI_MU_W
    ugs_2d = (
        (TZOTZI_RHO_W / rho_l_s) ** -0.1 * (rho_g / TZOTZI_RHO_A) ** -0.5
        * mu_ratio ** 0.35
        * math.log((0.8 / vsl) * mu_ratio ** 0.2) / 1.95
    )
    ugs_kh = (
        (rho_g / TZOTZI_RHO_A) ** -0.5 * (TZOTZI_RHO_W / rho_l_s) ** -0.5
        * (TZOTZI_SIGMA_W / max(sigma, 1.0e-12)) ** -0.33
        * math.log((1.39 / vsl) * mu_ratio ** -0.15) / 0.65
    )
    if ugs_2d < ugs_kh and ugs_2d <= vsg <= ugs_kh:
        ratio = 1.0 + 0.35 * math.sqrt(max(hL_tilde, 0.0)) * (vsg - ugs_2d)
    elif vsg > ugs_kh:
        ratio = (2.0 * max(hL_tilde, 0.0) ** 0.1 * mu_ratio ** 0.1
                 + 4.0 * math.sqrt(max(hL_tilde, 0.0)) * (vsg - ugs_kh))
    else:
        ratio = 1.0
    return ratio * fG


def _stratified_fi_selected(
    closure: str, *, vsg: float, VG: float, VL: float,
    hL_tilde: float, geom: dict, fG: float,
    rho_l: float, rho_g: float, mu_l: float, mu_g: float,
    D: float, eps: float, sigma: float, pressure_Pa: float,
    theta_rad: float = 0.0, vsl: float = 0.0,
) -> float:
    """分层流界面摩阻闭合选择器，返回 Darcy 等效因子。

    vsl（表观液速，m/s）为 Tzotzi 2013 闸门所需；关键字缺省 0.0 保住
    既有 15 个外部调用方的兼容（缺省即触发 tzotzi 的 ULS=0 光滑退化）。
    """
    if closure == "xiao":
        return _stratified_fi_xiao(
            vsg=vsg, VG=VG, VL=VL, hL_tilde=hL_tilde, geom=geom, fG=fG,
            rho_l=rho_l, rho_g=rho_g, mu_l=mu_l, mu_g=mu_g, D=D, eps=eps,
            sigma=sigma, pressure_Pa=pressure_Pa,
        )
    if closure == "xiao_baker_dg":
        # Xiao Eq.16-22 的水力直径口径对照（轮二审计 MED CONFIRMED，2026-10-22）：
        # Baker 分支（D>0.127）Re_i 取 geom["DG"]（=4Ag/(Sg+Si)，Eq.15 口径，
        # 与壁面基线的 Reg 同口径）；rr_i 恒为 Ei/D（恒有 Colebrook 正根；
        # 旧码 /Dg 伪收敛，见 _stratified_fi_baker docstring）；Andritsos 段不变。
        # 影响仅及 D>0.127 分段（本库 816/4759）。
        if D <= 0.127:
            return _stratified_fi_andritsos(vsg, hL_tilde, fG, pressure_Pa)
        return _stratified_fi_baker(
            VG, VL, geom["hL"], D, eps, rho_g, mu_g, mu_l, sigma, rho_l,
            DG=geom["DG"])
    if closure == "constant":
        return XIAO_FI_PATTERN_DARCY
    if closure == "smooth_interface":
        return fG
    if closure == "tzotzi2013":
        # Eq 18-21 三段式（see `_stratified_fi_tzotzi2013` 口径链 docstring）。
        return _stratified_fi_tzotzi2013(
            vsg=vsg, vsl=vsl, hL_tilde=hL_tilde, fG=fG,
            rho_l=rho_l, rho_g=rho_g, mu_l=mu_l, sigma=sigma,
        )
    if closure == "tzotzi2013_rholiteral":
        # 负对照（先例 petalas_fi_faithful / olga_slai_h1）：AH Eq 7 若按
        # 印刷下标 ρL 字面读，三段比值整体乘 ρL/ρG（≈10²–10³，油气工况
        # 界面剪切被放大 → 预期大幅劣化）。推荐口径为比值读（该分支仅
        # 用于论文演示"口径对齐错了会怎样"，见提取记录口径审计表）。
        fi_r = _stratified_fi_tzotzi2013(
            vsg=vsg, vsl=vsl, hL_tilde=hL_tilde, fG=fG,
            rho_l=rho_l, rho_g=rho_g, mu_l=mu_l, sigma=sigma,
        )
        return fi_r * (rho_l / max(rho_g, 1.0e-12))
    if closure == "petalas":
        return _interfacial_fi_updated_separated(
            VG, VL, geom, rho_l, rho_g, mu_l, mu_g, D, eps)
    if closure == "petalas_fi_faithful":
        # Petalas & Aziz 2000 Eq12 原文为 Fanning 域常数（0.004 + 0.5e-6·Re_SL）；
        # 仓库既有 "petalas" 分支（上游 drift_flux_model.py 同式）把 Fanning 值
        # 直接用于 Darcy 槽（τi 分母 8），界面剪切弱 4 倍。本变体整式 ×4
        # 补忠实口径对照（式对常数群线性，提取记录 §三.1 附带发现，2026-10-21）。
        return 4.0 * _interfacial_fi_updated_separated(
            VG, VL, geom, rho_l, rho_g, mu_l, mu_g, D, eps)
    if closure == "olga_slai_h1":
        # H1 忠实语义对照变体（轮二审计 HIGH CONFIRMED）。置于 startswith
        # ("olga_slai") 匹配之前，避免误算进 H0 族。
        return _stratified_fi_olga_slai_h1(
            VG=VG, VL=VL, geom=geom, rho_g=rho_g, D=D, sigma=sigma)
    olga_fi = None
    if closure.startswith("olga_slai") or closure.startswith("xiao_olga_blend_"):
        olga_fi = _stratified_fi_olga_slai(
            VG=VG, VL=VL, geom=geom, rho_g=rho_g, D=D, sigma=sigma)
    if closure == "olga_slai":
        return olga_fi
    if closure == "olga_slai_capped_3fg":
        return min(olga_fi, 3.0 * fG)
    andreussi_fi = None
    if closure.startswith("andreussi1987"):
        andreussi_fi = _stratified_fi_andreussi(
            VG=VG, geom=geom, rho_l=rho_l, rho_g=rho_g,
            theta_rad=theta_rad, fG=fG)
    if closure == "andreussi1987":
        return andreussi_fi
    if closure == "andreussi1987_capped_3fg":
        # 幂式 (F−F0)^0.67 无上界；本变体用于检验截断敏感性（原文未给上界）。
        return min(andreussi_fi, 3.0 * fG)
    if closure.startswith("kowalski1987"):
        # 式 25 的回归样本仅覆盖 22,600 < Re_g < 430,600：区间外返回 0 作哨兵，
        # 在此统一退回光滑界面 fG（不臆造外推值）。
        kow_fi = _stratified_fi_kowalski(
            VG=VG, VL=VL, hL_tilde=hL_tilde, rho_g=rho_g, rho_l=rho_l,
            mu_g=mu_g, mu_l=mu_l, D=D)
        if kow_fi <= 0.0:
            return fG
    if closure == "kowalski1987":
        return clamp(kow_fi, 0.0, 1.0e4)
    if closure == "kowalski1987_capped_3fg":
        # 幂式无上界（Re_l^0.83 在低液速时使 fi 抬头）；本变体检验截断敏感性。
        return clamp(min(kow_fi, 3.0 * fG), 0.0, 1.0e4)
    if closure.startswith("xiao_olga_blend_"):
        try:
            weight = float(closure.rsplit("_", 1)[1]) / 100.0
        except ValueError as exc:
            raise ValueError(f"未知分层流混合闭合: {closure}") from exc
        if not 0.0 <= weight <= 1.0:
            raise ValueError(f"分层流混合权重必须位于 0~100%: {closure}")
        xiao_fi = _stratified_fi_xiao(
            vsg=vsg, VG=VG, VL=VL, hL_tilde=hL_tilde, geom=geom, fG=fG,
            rho_l=rho_l, rho_g=rho_g, mu_l=mu_l, mu_g=mu_g, D=D, eps=eps,
            sigma=sigma, pressure_Pa=pressure_Pa,
        )
        # 在线性剪切系数空间混合，便于检验 SLAI 的增量而不是整式替换。
        return (1.0 - weight) * xiao_fi + weight * olga_fi
    raise ValueError(f"未知分层流界面摩阻闭合: {closure}")


# ullmann2004_wall_mtf 诊断计数器（跨调用累计，仅供冒烟/跑批口径审计；
# 不进任何生产路径）：calls=选择器调用数（不动点迭代逐步计入），
# active=实际启用论文式次数，gate=任一相湍流退基线次数，
# guard=退化输入守卫次数，neg_f=负 F 透传次数（论文原始设计，照实计数）
_ULLMANN2004_MTF_DIAG = {"calls": 0, "active": 0, "gate": 0, "guard": 0,
                         "neg_f": 0}


def _stratified_wall_f_selected(
    closure: str, *,
    vsg: float, vsl: float, VG: float, VL: float, geom: dict,
    rho_l: float, rho_g: float, mu_l: float, mu_g: float,
    D: float, eps: float,
) -> tuple[float, float]:
    """分层流壁面摩擦因子选择器 —— 返回 (fL, fG)，Darcy 口径。

    基线 "xiao"（Xiao Eq.13-14）：液/气各以**水力直径**（geom["DL"]/geom["DG"]）
    + **实际相速** VL/VG 求 Re，`_friction_factor_turbulent` 同一粗糙度 rr=eps/D
    （Re<=0 -> 0；Re<=2000 层流 64/Re；否则 Colebrook 有界 Newton）。

    论文闭合只换壁面 f（判型解与其余流型求解器不走本选择器，判型不变性照旧）；
    界面摩阻选择器拿到的 fG 仍是本选择器返回的 fG —— fi 经 fG 的耦合与生产
    一致，不人为隔离。口径链与忠实性脚注详见 wall_closure_extraction_record.md：

      * "kowalski1987"（仅液壁，Eq 18，AIChE J 33(2):274-281）：
        f_wL = 0.083·(Re_p*)^(-0.37)（Fanning），Re_p* = vsl·D·rho_l/mu_l
        （**表观液速 + 管径 D**，Notation 口径 j=superficial，期刊 p280=PDF p7）；
        光滑管、无粗糙度项。
        论文实验 Re_p* 全湍域、无层流分支 —— Re_p* <= 2000 时按槽位策略 (a)
        **退回基线 fL**（64/Re 粘性下界），闸门区与基线严格一致，不臆造外推。
        气壁论文无式（Eq 14/15 释义性提及；Eq 17 是 Agrawal 壁面-液相对比式，
        被本文 Eq 18 以精度取代——被论文否决的界面式是 Eq 26）—— 保持基线。Fanning -> Darcy 裸 ×4（先例 `_stratified_fi_kowalski`）。
      * "ottens2001"（**双侧**，唯一全壁面替换变体；model a = Taitel-Dukler
        Eq 7a;b, p583）：f = C·Re^(-n)，Re>2000: (C=0.046, n=0.2)、
        Re<=2000: (C=16, n=1)（Fanning，由 Eq 6a;b;c tau=0.5·f·rho·U^2 链确证）；
        Re = rho·U·D/mu（**实际相速 + 管径 D**，符号表 "U = actual phase
        velocity" 逐字）；光滑管。Darcy：湍域 0.184/Re^0.2、层流 64/Re ——
        层流与基线同型，策略 (b) 天然成立，无需 fallback。
        （勘误 2026-10-21：曾误记 "Eq 13a/13b, p586, 再 ×4=0.736"；
        0.184 已是 Darcy 值，再 ×4 会虚增壁面摩阻 4 倍。未进任何跑批。）
      * "petalas_wall"（仅液壁，Eq 10）：f_L = 0.452·f_SL^0.731（Fanning），
        Re_SL = vsl·D·rho_l/mu_l（**表观液速 + 管径 D** —— 与 hL 无关，fL 在
        hL 不动点迭代内为常数）；Fanning -> Darcy **经幂律**：
        fL = 4·0.452·(f_SL_D/4)^0.731，f_SL_D 沿用模型 Colebrook 同源式
        （f_SL 的取式选择经 ±5% 敏感性仅放大为 fL 的 ±3.6%）。
        忠实性脚注：论文 p.52 层流规则为 max(f_lam, f_turb) 连续过渡，本实现
        沿用硬切（与预检脚本 w1_petalas_wall_check2 同式，保证宣告的
        fL 中位 2.51× 采样口径可复现）；tau_wL 仍用实际 V_L（Eq 7）。
        气侧论文 Eq 6+9 与基线完全同构（仅层流规则差异、作用带 ~5%）——
        保持基线。
      * "biberg1999_wall_fs"（仅液壁，Eq 45 带内嵌口径链，τ_i=0 支）：
        层流液膜 fL = 64/Re_e，Re_e = ρ_l·D_e·|VL|/μ_l；D_e 由 Eq 50
        有理逼近（Λ = gamma/π，θ_l ≡ gamma 同构已证）。Re_l(D_h)>
        2000 退基线（策略 (a)）。"biberg1999_wall_fs_dh" = 64/Re_l
        水力直径版，**预期 no-op 对照**（与基线层流支同式，faithful 实现
        留待跑批复核）。τ_i 项（Eq 51 λ）不在壁面槽白名单 → 按零伪造
        规则舍去并显式记录（完整 Eq 45 不可插拔）。
      * "ullmann2004_wall_mtf"（双侧，层流两流体修正式；Ullmann, Goldstein,
        Zamir & Brauner, IJMF 30 (2004) 877-900）：单相 Poiseuille f=16/Re
        （Fanning，×4 吸收进 F）× 两相相互作用因子 F₁/F₂（Eq 22）——Darcy
        交付 fL=F₁·64/Re₁、fG=F₂·64/Re₂。口径链：X²=μ̃q̃=μ_l·vsl/(μ_g·vsg)
        （=两相各自满管层流压降梯度比，期刊 L302 逐字；q̃=Q₁/Q₂ 表观速比
        四重确证）；U=实际相速（Re=ρ|U|D/μ 的 U 即 in-situ，Eq 23）；水力
        直径 Eq 23 **界面计入两相湿润周长** D₁,₂=πDε₁,₂/(S̃₁,₂+S̃ᵢ)（≠
        框架 DL/DG——后者界面只归气侧，故本分支自算直径）；g 系数（Eq 24，
        γ≡geom["gamma"]≡φ₁ 同构已证）：g₁₁=γ/(γ+sinγ)、g₂₂=(π−γ)/
        (π−γ+sinγ)、g₁₂=[4/(π+2)]·(π−γ)/π、g₂₁=[4/(π+2)]·γ/π。
        双相层流闸门：Re₁,₂（ullmann 自身直径）>2000 → 退基线（论文
        closure 仅推导层流，策略 (a) 同 kowalski/biberg，不臆造湍域外推）。
        F<0 为论文 p.887 设计（倾角流近壁回流 → 反向壁面剪切），照实透传
        并按因子计数（诊断字典 neg_f；零伪造不截断——负 fL 如实进入
        不动点，发散则该工况记 error 项）。界面槽结构性不可插拔（σᵢ∝
        (U₂−U₁) 线性式 vs 消费端 ΔU|ΔU| 二次式，比值折叠 ΔU→0 奇异，
        且管道 F_i 为推理值非印刷值——ullmann2006 先例），本变体只动
        壁面，详见记录 §七。验证锚点（冒烟 ④）：μ̃=q̃=1 且 ε₁=½ →
        γ=π/2、F₁=F₂=π/(π+2)、D₁=D·π/(π+2) → τ_wL=fL·ρU|U|/8=
        8μU/D **精确 Poiseuille**。

    适用域：kowalski/ottens/petalas/biberg 原文均未印显式适用域（报告
    如实记录），硬闸门 Kowalski/Biberg 层流外退基线（策略 (a)）；
    ullmann2004 明确限层流两流体（双相 Re≤2000 闸门，同策略）。vsg/vsl
    在 ullmann2004 进 X²（表观速比对）已非占位（ouyang 的 R_v=vsg/vsl
    若日后解锁可直取，见记录 §三.2）。
    """
    rr = eps / max(D, 1.0e-12)
    # 基线（xiao）—— 与替换前的内联代码逐位一致（先气后液，同一 rr）
    re_g = rho_g * abs(VG) * geom["DG"] / max(mu_g, 1.0e-12)
    fG_base = _friction_factor_turbulent(re_g, rr)
    re_l = rho_l * abs(VL) * geom["DL"] / max(mu_l, 1.0e-12)
    fL_base = _friction_factor_turbulent(re_l, rr)

    if closure == "xiao":
        return fL_base, fG_base

    if closure == "kowalski1987":
        # Re_p* = 表观液速·管径（p280 Notation: j = superficial velocity）
        re_p = abs(vsl) * D * rho_l / max(mu_l, 1.0e-12)
        if re_p <= 2000.0:
            return fL_base, fG_base            # 策略 (a)：层流闸门退回基线
        f_wl = 0.083 * re_p ** (-0.37)
        return 4.0 * f_wl, fG_base             # Fanning -> Darcy（裸 ×4）

    if closure == "ottens2001":
        # 双侧 Blasius 式，实际相速 + 管径 D（光滑管，粗糙度不进式）
        re_g_pipe = rho_g * abs(VG) * D / max(mu_g, 1.0e-12)
        if re_g_pipe <= 0.0:
            fG_o = 0.0
        elif re_g_pipe <= 2000.0:
            fG_o = 64.0 / re_g_pipe            # 4×16/Re，与基线层流同型
        else:
            fG_o = 0.184 / re_g_pipe ** 0.2    # 4×0.046/Re^0.2（已是 Darcy）
        re_l_pipe = rho_l * abs(VL) * D / max(mu_l, 1.0e-12)
        if re_l_pipe <= 0.0:
            fL_o = 0.0
        elif re_l_pipe <= 2000.0:
            fL_o = 64.0 / re_l_pipe
        else:
            fL_o = 0.184 / re_l_pipe ** 0.2
        return fL_o, fG_o

    if closure == "petalas_wall":
        # Re_SL = 表观液速·管径 —— 与 hL 无关（不动点迭代内 fL 为常数）
        re_sl = rho_l * abs(vsl) * D / max(mu_l, 1.0e-12)
        f_sl_d = _friction_factor(re_sl, rr)   # 硬切（非 p.52 max 规则，见 docstring）
        fL_p = 4.0 * 0.452 * (f_sl_d / 4.0) ** 0.731
        return fL_p, fG_base                   # Fanning -> Darcy 经幂律

    if closure in ("biberg1999_wall_fs", "biberg1999_wall_fs_dh"):
        # Biberg et al. (1999) 液壁层流闭合 Eq 45（τ_i=0 简化，见下）。
        # 论文：J. Energy Resour. Technol. 121 (1999) 1071-1080，Eq 45/50
        # （期刊 p.1078-1079 = PDF p.5-6）。仅液壁；气壁保持基线
        # （论文气侧无独立式，τ_i 不在壁面槽白名单 → 舍 λ 项 Eq 51）。
        #
        # 完整 Eq 45: τ̄_wl = 8μ_l·U_l/D_e(θ_l) − λ·(S_i/S_l)·τ_i。τ_i 不在
        # 壁面槽输入白名单内、全库无逐工况值 —— 按零伪造规则只能实现
        # τ_i=0 简化支（显式记录：非忠实去项，是白名单约束下的最大可
        # 插拔子式；λ 项 Eq 51 因此同弃）。τ_i=0 时 fL = 8τ̄/(ρ_l VL|VL|)
        # = 64/Re_e，Re_e = ρ_l·D_e·|VL|/μ_l（Darcy；论文 p.1078 页底脚注
        # 图片确认 Darcy 口径 h_l = 64/Re 形式）。
        #
        # 口径对齐：
        #   * θ_l ≡ geom["gamma"]（论文 S_l = D·θ_l 与模块 SL = γ·D 逐位
        #     一致；Si = D·sinγ 同）→ Λ = gamma/π ∈ (0,1)。
        #   * U_l = VL 实际液速；D_e 由 Eq 50 有理逼近（Table 1 系数，
        #     图片逐位校验；Λ=1/2 与 Λ=1 两点解析自检 D_e/D→1.000000 ✓；
        #     a0=−1e-6 轻微为负 → Λ≈0 端 max(D_e/D, 0) 数值护栏）。
        #   * 闸门：Re_l = ρ_l·D_h·|VL|/μ_l（论文脚注口径，D_h=geom["DL"]
        #     = 4A_l/S_l）> 2000 → 退回基线 fL（层流闭合不外推至湍流，
        #     kowalski 策略 (a) 先例）。
        #   * "_fs_dh" 对照变体：fL = 64/Re_l 直接取水力直径 —— 与基线
        #     层流支数学恒等（基线 re_l 同式同径），登记为 **预期 no-op**
        #     对照（D_e 与 D_h 之差才是 Eq 50 的全部效果）；仍按公式忠实
        #     实现（非直通基线），让跑批以数据实证恒等而非仅靠码读论证。
        #   * 勘误注记：论文称嵌套求值 "12 multiplications"，Eq 50+51 系数
        #     实为 12 个（6+6）而 51 项为 4+7=11，二者取一为口误；不影响
        #     常数本身（Table 1 图片为准）。
        re_l_gate = rho_l * abs(VL) * geom["DL"] / max(mu_l, 1.0e-12)
        if re_l_gate > 2000.0:
            return fL_base, fG_base            # 策略 (a)：层流域外退基线
        if closure == "biberg1999_wall_fs_dh":
            if re_l_gate <= 0.0:
                return fL_base, fG_base        # VL=0 退化（基线同给 0）
            return 64.0 / re_l_gate, fG_base    # = 基线层流支（预期 no-op）
        lam = geom["gamma"] / math.pi
        num = (-0.000001 + 0.0003550 * lam + 6.7477988 * lam ** 2
               - 7.0926783 * lam ** 3 - 11.959489 * lam ** 4
               + 19.603771 * lam ** 5 - 7.0282088 * lam ** 6)
        den = (1.0 - 1.1126440 * lam + 1.5839297 * lam ** 2
               - 4.4506182 * lam ** 3 + 5.9410713 * lam ** 4
               - 3.6313889 * lam ** 5 + 0.9411974 * lam ** 6)
        de_over_d = num / max(den, 1.0e-12)
        if de_over_d <= 0.0:
            return fL_base, fG_base            # Λ≈0 数值护栏（a0 微负）
        re_e = rho_l * abs(VL) * (de_over_d * D) / max(mu_l, 1.0e-12)
        if re_e <= 0.0:
            return fL_base, fG_base            # VL=0 退化
        return 64.0 / re_e, fG_base            # 层流液膜 Darcy（fL = 64/Re_e）

    if closure == "ullmann2004_wall_mtf":
        # Ullmann et al. (2004) 层流两流体壁面闭合 Eq 22-24（IJMF 30:877-900，
        # 期刊 p.890 = PDF p.14）。单相 Poiseuille × 两相相互作用因子：
        #   f₁ = 16/Re₁·F₁（Fanning），F₁ = [1+(U₂/U₁)·{g₁₁X²((1−ε)/ε)²−g₁₂}]
        #                                    / [1+(U₂/U₁)·X²((1−ε)/ε)²]
        #   f₂ = 16/Re₂·F₂ 镜像（1/X²、ε/(1−ε)、g₂₂/g₂₁；下标 1=液 2=气——
        #   框架注入 U₁=VL、U₂=VG、ε=ε₁=holdup）
        # Fanning→Darcy ×4 吸收进 F 交付：fL=F₁·64/Re₁、fG=F₂·64/Re₂。
        # 分子括号归属、两条无系数分数线、g 系数全部字符坐标级确证
        # （u2004_eqchars.txt p14 带；记录 §七）。
        _ULLMANN2004_MTF_DIAG["calls"] += 1
        gam = geom["gamma"]
        sin_g = math.sin(gam)
        # 极端持液几何守卫：γ→0/π（ε₁→0/1）时 S̃₁+S̃ᵢ 或 S̃₂+S̃ᵢ→0，
        # 直径与 g 系数退化 → 基线（零伪造：宁可弃式不造数）
        if not (1.0e-6 < gam < math.pi - 1.0e-6):
            _ULLMANN2004_MTF_DIAG["guard"] += 1
            return fL_base, fG_base
        # 自算界面湿润直径（Eq 23：界面计入两相——≠框架 DL/DG）
        eps_l = geom["AL"] / geom["A"]
        eps_g = geom["AG"] / geom["A"]
        d1 = math.pi * D * eps_l / (gam + sin_g)
        d2 = math.pi * D * eps_g / (math.pi - gam + sin_g)
        # 退化输入守卫：层流闸门与速度比都依赖 |U|>0、X² 依赖 vsg/vsl>0
        if vsg <= 0.0 or vsl <= 0.0 or abs(VL) < 1.0e-12 or abs(VG) < 1.0e-12:
            _ULLMANN2004_MTF_DIAG["guard"] += 1
            return fL_base, fG_base
        # 双相层流闸门（ullmann 自身直径求 Re；论文 closure 仅层流域）
        re1 = rho_l * abs(VL) * d1 / max(mu_l, 1.0e-12)
        re2 = rho_g * abs(VG) * d2 / max(mu_g, 1.0e-12)
        if re1 > 2000.0 or re2 > 2000.0:
            _ULLMANN2004_MTF_DIAG["gate"] += 1
            return fL_base, fG_base
        # X² = μ̃q̃ = μ_l·vsl/(μ_g·vsg)（L302 逐字；=各自满管层流压降梯度比）
        x2 = (mu_l / mu_g) * (vsl / vsg)
        # g 系数（Eq 24.1/24.2；γ≡φ₁ 同构已证）
        g11 = gam / (gam + sin_g)
        g22 = (math.pi - gam) / (math.pi - gam + sin_g)
        g12 = (4.0 / (math.pi + 2.0)) * (math.pi - gam) / math.pi
        g21 = (4.0 / (math.pi + 2.0)) * gam / math.pi
        # F 因子（Eq 22.1/22.2 canonical；F<0 照实透传——期刊 p.887 印刷
        # 设计（fulltext L726-727）：倾角流近壁回流 → 反向壁面剪切，
        # neg_f 计数不截断）
        r21 = VG / VL          # U₂/U₁（in-situ 实际相速）
        r12 = VL / VG          # U₁/U₂
        w1 = x2 * ((1.0 - eps_l) / eps_l) ** 2
        w2 = (1.0 / x2) * (eps_l / (1.0 - eps_l)) ** 2
        f1_hat = (1.0 + r21 * (g11 * w1 - g12)) / (1.0 + r21 * w1)
        f2_hat = (1.0 + r12 * (g22 * w2 - g21)) / (1.0 + r12 * w2)
        if f1_hat < 0.0 or f2_hat < 0.0:
            _ULLMANN2004_MTF_DIAG["neg_f"] += 1
        _ULLMANN2004_MTF_DIAG["active"] += 1
        return (f1_hat * 64.0 / max(re1, 1.0e-12),
                f2_hat * 64.0 / max(re2, 1.0e-12))

    raise ValueError(f"未知分层流壁面摩阻闭合: {closure}")


def solve_stratified(
    vsg: float, vsl: float, vm: float,
    rho_l: float, rho_g: float, mu_l: float, mu_g: float,
    D: float, eps: float, theta_rad: float, sigma: float,
    P_Pa: float = 101325.0,
    interfacial_closure: str = "xiao",
    wall_closure: str = "xiao",
) -> dict:
    """
    分层流水力求解器 — Xiao Eq 8-11。

    使用实际的 Andritsos/Baker 界面摩擦（不同于流型判别的 fi=0.0142）。
    """
    A = math.pi * D * D / 4.0
    sin_theta = math.sin(theta_rad)
    drho = rho_l - rho_g

    def momentum_balance(hL_tilde: float) -> float:
        geom = _stratified_geometry(hL_tilde, D)
        AL_v, AG_v = geom["AL"], geom["AG"]
        VL = vsl * A / max(AL_v, 1e-12)
        VG = vsg * A / max(AG_v, 1e-12)

        fL, fG = _stratified_wall_f_selected(
            wall_closure,
            vsg=vsg, vsl=vsl, VG=VG, VL=VL, geom=geom,
            rho_l=rho_l, rho_g=rho_g, mu_l=mu_l, mu_g=mu_g,
            D=D, eps=eps,
        )
        tau_wG = fG * rho_g * VG * abs(VG) / 8.0
        tau_wL = fL * rho_l * VL * abs(VL) / 8.0

        fi = _stratified_fi_selected(
            interfacial_closure,
            vsg=vsg, VG=VG, VL=VL, hL_tilde=hL_tilde, geom=geom, fG=fG,
            rho_l=rho_l, rho_g=rho_g, mu_l=mu_l, mu_g=mu_g, D=D, eps=eps,
            sigma=sigma, pressure_Pa=P_Pa, theta_rad=theta_rad, vsl=vsl,
        )

        Vi = VG - VL
        tau_i = fi * rho_g * Vi * abs(Vi) / 8.0

        term_L = tau_wL * geom["SL"] / max(AL_v, 1e-12)
        term_G = tau_wG * geom["SG"] / max(AG_v, 1e-12)
        term_i = tau_i * geom["Si"] * (1.0 / max(AL_v, 1e-12) + 1.0 / max(AG_v, 1e-12))
        term_g = drho * G * sin_theta
        return term_L - term_G - term_i + term_g

    # 扫描+割线法
    n_scan = XIAO_ROOT_SCAN_POINTS
    roots = []
    h_prev, f_prev = 1e-6, momentum_balance(1e-6)
    for i in range(1, n_scan + 1):
        h_curr = i / n_scan * (1.0 - 1e-6)
        f_curr = momentum_balance(h_curr)
        if f_prev * f_curr <= 0:
            roots.append(_bracketed_secant_bisection(
                momentum_balance, h_prev, h_curr, f_prev, f_curr
            ))
        h_prev, f_prev = h_curr, f_curr

    if not roots:
        best_h = 0.5; best_f = abs(momentum_balance(best_h))
        for h_try in [0.01, 0.1, 0.3, 0.5, 0.7, 0.9]:
            f_try = abs(momentum_balance(h_try))
            if f_try < best_f: best_f, best_h = f_try, h_try
        hL_tilde = best_h
    else:
        hL_tilde = min(roots)

    geom = _stratified_geometry(hL_tilde, D)
    EL = geom["AL"] / A; EG = geom["AG"] / A
    VL = vsl * A / max(geom["AL"], 1e-12)
    VG = vsg * A / max(geom["AG"], 1e-12)

    # 压降 — Xiao Eq 11
    fL, fG = _stratified_wall_f_selected(
        wall_closure,
        vsg=vsg, vsl=vsl, VG=VG, VL=VL, geom=geom,
        rho_l=rho_l, rho_g=rho_g, mu_l=mu_l, mu_g=mu_g,
        D=D, eps=eps,
    )
    tau_wG = fG * rho_g * VG * abs(VG) / 8.0
    tau_wL = fL * rho_l * VL * abs(VL) / 8.0
    fi = _stratified_fi_selected(
        interfacial_closure,
        vsg=vsg, VG=VG, VL=VL, hL_tilde=hL_tilde, geom=geom, fG=fG,
        rho_l=rho_l, rho_g=rho_g, mu_l=mu_l, mu_g=mu_g, D=D, eps=eps,
        sigma=sigma, pressure_Pa=P_Pa, theta_rad=theta_rad, vsl=vsl,
    )
    tau_i = fi * rho_g * (VG - VL) * abs(VG - VL) / 8.0

    # Xiao Eq.11: eliminate interfacial shear and sum both wall stresses.
    dp_fric = (
        tau_wL * geom["SL"] + tau_wG * geom["SG"]
    ) / max(A, 1e-12)
    rho_m = EL * rho_l + EG * rho_g
    dp_grav = rho_m * G * sin_theta
    dp_total = dp_fric + dp_grav

    return {
        "EL": EL, "EG": EG,
        "dp_fric": dp_fric, "dp_grav": dp_grav, "dp_total": dp_total,
        "rho_m": rho_m, "mu_m": mu_l,
        "hL_tilde": hL_tilde, "fi": fi, "fG": fG, "fL": fL,
        "friction_multiplier": 1.0,
        "pattern": "stratified",
        "v_gas_actual": VG, "v_liq_actual": VL,
    }


# ============================================================
# Layer 4: 间歇流求解器 (slug 单元模型)
# ============================================================

def _soto_cortes2025_holdup(
    *, vm: float, rho_l: float, rho_g: float, mu_l: float, mu_g: float,
    D: float, theta_deg: float, sigma: float, fanning_factor: float,
) -> float:
    """Soto-Cortes, Pereyra & Sarica (2025) 统一液塞持液率闭合。

    Eq.(10)-(14)+(18 Option 1: A=23.78, omega=0.608)+(19) 的定点迭代
    形态（原文 Fig.6）：Gregory 初值，上限 100 轮。
    三项预登记忠实性决定（原文未印 fM 具体式）：
      (1) fM 取 Fanning 口径光滑管分支：ReM<2100 层流 16/ReM、
          湍流 Blasius 0.0791*ReM^-0.25；切换阈值 2100 为实现决定；
          fanning_factor=4.0 给出 Darcy 敏感性对照变体。
      (2) 式(19) 的 theta 以度代入（原文正文 b 区间裁定）。
      (3) b 在 Na=1 与 theta=0 处保持不连续（忠实版，不平滑）：
          Na>=1 任意倾角 b=1；Na<1 上升(0<=theta<=90)
          b=0.0099*theta+0.9672；下降(-30<=theta<0)
          b=0.02683*theta+0.8183；域外倾角按分支直线外推（须披露）。
    """
    if sigma <= 0.0:
        raise ValueError(
            "soto_cortes2025 要求 sigma>0（表面张力缺失，拒绝以默认值代填）")
    drho_local = max(rho_l - rho_g, 1.0e-12)
    # Bond 数与黏性数（Eq.13/14；Na 经原文自述恒等式 Na=Bo/Nf 裁定
    # 根号内为 drho/rhoL）
    bo = G * D * D * drho_local / sigma
    na = (mu_l / sigma) * math.sqrt(
        G * D * drho_local / max(rho_l, 1.0e-12))
    # C（Eq.18 Option 1）与 b（Eq.19，忠实不连续三分支）
    c_coeff = 23.78 * max(na, 1.0e-30) ** 0.608
    if na >= 1.0:
        b_coeff = 1.0
    elif theta_deg >= 0.0:
        b_coeff = 0.0099 * theta_deg + 0.9672
    else:
        b_coeff = 0.02683 * theta_deg + 0.8183
    # 内层定点迭代（Eq.20-22 + Fanning fM），Gregory 初值
    h_slug = 1.0 / (1.0 + (vm / 8.66) ** 1.39)
    for _ in range(100):
        rho_m = rho_l * h_slug + rho_g * (1.0 - h_slug)
        mu_m = mu_l * h_slug + mu_g * (1.0 - h_slug)
        re_m = rho_m * vm * D / max(mu_m, 1.0e-30)
        if re_m < 2100.0:
            f_m = 16.0 / max(re_m, 1.0e-9)
        else:
            f_m = 0.0791 * re_m ** -0.25
        f_m *= fanning_factor
        we_m = rho_m * vm * vm * D / sigma
        phi = f_m * we_m / math.sqrt(max(bo, 1.0e-30))
        h_new = 1.0 / (b_coeff + phi / c_coeff)
        if abs(h_new - h_slug) < 1.0e-10:
            return h_new
        h_slug = h_new
    return h_slug


def _slug_body_liquid_holdup(
    closure: str, *, vm: float, rho_l: float, rho_g: float,
    mu_l: float, mu_g: float, D: float, theta_rad: float,
    sigma: float = 0.0,
) -> float:
    """液塞体液相体积分数 ``E_s`` 的文献闭合。

    所有关系均保持原论文的 SI 量纲口径；最终的 ``[0.48, 1]`` 保护与
    Petalas--Aziz 的最大气泡堆积判据一致。除 Gregory 外，各式的原始
    标定域不同，批量报告必须另行按倾角、黏度和管径审查外推结果。
    """
    vm_abs = max(abs(vm), 1.0e-12)
    drho = max(rho_l - rho_g, 1.0e-12)
    theta_deg = math.degrees(theta_rad)

    if closure in {"xiao", "gregory1978"}:
        holdup = 1.0 / (1.0 + (vm_abs / 8.66) ** 1.39)
    elif closure == "gregory1978_fulldata":
        # Gregory et al. (1978) Table 3 全数据支（未剔除支, n=167,
        # a=10.2, beta=1.21），含段塞→环状过渡区 10 点。考古对照键。
        holdup = 1.0 / (1.0 + (vm_abs / 10.2) ** 1.21)
    elif closure == "soto_cortes2025":
        # Soto-Cortes et al. (2025) 统一闭合 Option 1 单式（Fanning）。
        holdup = _soto_cortes2025_holdup(
            vm=vm_abs, rho_l=rho_l, rho_g=rho_g, mu_l=mu_l, mu_g=mu_g,
            D=D, theta_deg=theta_deg, sigma=sigma, fanning_factor=1.0,
        )
    elif closure == "soto_cortes2025_darcy":
        # 敏感性对照：fM 口径从 Fanning 换成 Darcy（fD=4*fF）。
        # 原文未印 fM 具体式，口径风险 4 倍，此键量化该不确定性。
        holdup = _soto_cortes2025_holdup(
            vm=vm_abs, rho_l=rho_l, rho_g=rho_g, mu_l=mu_l, mu_g=mu_g,
            D=D, theta_deg=theta_deg, sigma=sigma, fanning_factor=4.0,
        )
    elif closure == "gomez2000":
        # Gomez, Shoham & Taitel (2000), as reproduced in
        # Abdul-Majeed & Al-Mashat (2019), Eq. (1)--(2).
        re_slug = D * rho_l * vm_abs / max(mu_l, 1.0e-12)
        holdup = math.exp(-(0.00784 * theta_deg + 2.48e-6 * re_slug))
    elif closure == "abdul_majeed2000":
        # Abdul-Majeed (2000), Eq. (3)--(5) in the 2019 comparison paper.
        coefficient = 0.06 + 1.3377 * mu_g / max(mu_l, 1.0e-12)
        exponent = 1.0 if theta_deg <= 0.0 else 1.0 - math.sin(theta_rad)
        holdup = max(1.0 - coefficient * vm_abs, 1.0e-12) ** exponent
    else:
        n_fr = (
            vm_abs / math.sqrt(max(G * D, 1.0e-30))
            * math.sqrt(rho_l / drho)
        )
        n_mu = vm_abs * mu_l / max(G * D * D * drho, 1.0e-30)

        if closure == "kora2011":
            x_value = n_fr * n_mu ** 0.2
            if x_value <= 0.15:
                holdup = 1.0
            elif x_value < 1.5:
                holdup = 1.012 * math.exp(-0.085 * x_value)
            else:
                holdup = 0.9473 * math.exp(-0.041 * x_value)
        elif closure == "al_safran2015":
            phi = n_fr * n_mu ** 0.2 - 0.89
            holdup = (
                0.85 - 0.075 * phi
                + 0.057 * math.sqrt(phi * phi + 2.27)
            )
        elif closure == "al_ruhaimani":
            n_f = math.sqrt(G * D ** 3 * rho_l * drho) / max(mu_l, 1.0e-12)
            holdup = 0.266 / max(n_fr * math.sqrt(n_f), 1.0e-30) + 0.912
        elif closure == "abdul_majeed2019":
            # Abdul-Majeed & Al-Mashat (2019), Eq. (20), (26).
            x_value = n_fr / max(n_mu ** 0.2, 1.0e-30)
            holdup = (
                1.016 - 0.000611 * theta_deg
                + (0.000124 * theta_deg - 0.0195) * x_value
            )
        else:
            raise ValueError(f"未知液塞体持液率闭合: {closure}")

    return clamp(holdup, 0.48, 1.0)


def _slug_length(
    closure: str, *, D: float, vm: float, vsl: float = 0.0,
    rho_l: float, rho_g: float, mu_l: float, mu_g: float,
) -> float:
    """段塞长度 ``L_s`` 的文献闭合。

    所有关系保持原论文 SI 量纲口径，返回液塞体长度（m）。
    Scott 1987 为宿主基线；其余各式原始标定域不同（管径/黏度/倾角），
    批量报告必须另行审查外推结果。注意在 Xiao Eq.30/31 的比值加权
    结构中，Ls 只通过需要绝对长度的项（单元频率型附加项）影响压降。
    """
    if closure in {"xiao", "scott1987"}:
        # Xiao (1990) 采用的 Scott et al. (1987) 式（原论文口径：
        # D 以英寸计的回归，此处代码沿用宿主历史实现）。
        if D < 0.0381:
            Ls = 30.0 * D
        else:
            ln_Ls = -26.6 + 28.5 * (math.log(D) + 3.67) ** 0.1
            Ls = math.exp(ln_Ls)
        return clamp(Ls, 10.0 * D, 100.0 * D)
    if closure in {"constant_30d", "dukler_hubbard1975"}:
        # Dukler & Hubbard (1975) 教科书默认：液塞长约 30D。
        return clamp(30.0 * D, 10.0 * D, 100.0 * D)
    if closure in {"constant_15d", "cook_behnia2000_len"}:
        # Cook & Behnia (2000) 全部计算取 Ls = 15D（其 Fig.5 敏感性
        # 表明结果对 Ls/D 不敏感，摩擦/黏性项近似互相补偿）。
        return clamp(15.0 * D, 10.0 * D, 100.0 * D)
    if closure == "nydal1992":
        # Nydal, Pintus & Andreussi (1992) Sec.4.2：水平空气-水,
        # D=52.9mm 时 Ls/D=15~20、D=90mm 时约 12~16, 对 vm 不敏感。
        # 常数取两管径中值(17.5D/14D), D 以 0.06 m 分界。
        if D < 0.06:
            return clamp(17.5 * D, 10.0 * D, 100.0 * D)
        return clamp(14.0 * D, 10.0 * D, 100.0 * D)
    if closure in {"baba2018", "baba2018_gated"}:
        # Baba et al. (2018) Eq.(11)：Ls/D = 3.35·Fr^0.03·Nμ^-0.2·Re^0.1
        # （Fr = vm/√(gD)，Nμ = μL/√(ρL(ρL−ρG)gD³)，Re = ρL·vm·D/μL，
        # 论文明示 μG 可忽略，全 SI）。标定域为水平空气-高黏油
        # （宣称有效域 0.18–6.0 Pa·s）。低黏外推严重过估：气水工况
        # 104–118D vs 论文自测 24–36D（不参与拟合）。本库间歇流队列
        # μL≤0.0744 Pa·s 全域落在标定域之外 —— 作为长度轴忠实对照收录，
        # 外推行为随战役报告披露。
        if closure == "baba2018_gated" and mu_l < 0.15:
            # 黏度门控版：μL 低于论文拟合下限（0.181 Pa·s 圆整为
            # 0.15）时旁路回 Scott 基线，不做域外外推。本生产队列
            # 门在全队列闭合（μL 最大 0.0744），因此激活组合应与
            # cook_mixing 逐位一致（域失配惰性对照）。
            return _slug_length(
                "scott1987", D=D, vm=vm, vsl=vsl,
                rho_l=rho_l, rho_g=rho_g, mu_l=mu_l, mu_g=mu_g)
        drho = max(rho_l - rho_g, 1.0e-12)
        fr = vm / math.sqrt(max(G * D, 1.0e-30))
        n_mu = mu_l / math.sqrt(max(drho * rho_l * G * D ** 3, 1.0e-30))
        re_l = rho_l * vm * D / max(mu_l, 1.0e-30)
        ls_d = (3.35 * max(fr, 1.0e-30) ** 0.03
                * max(n_mu, 1.0e-30) ** -0.2
                * max(re_l, 1.0e-30) ** 0.1)
        # 标准钳制保留：本队列低黏外推 104–118D 将触 100D 上界
        # （饱和比例随战役报告披露）；论文高黏域 4–16D 会触 10D
        # 下界，但该 μL 区间在队列内为 0 例，不构成可达路径。
        return clamp(ls_d * D, 10.0 * D, 100.0 * D)
    if closure == "woods_hanratty1996":
        # 派生组合式（经 kadri2009 Eq.(35)/(21) 链, 非原生长度式）：
        # 段塞频率 fs = 0.088*(USL+1.5)^2/(g*D)（Nydal 1991, 空气-水,
        # 水平充分发展域），代入 Woods & Hanratty (1996) 频率-长度关系
        # fS*D/USL = 1.2*(Ls/D - 1) 反解得 Ls。标定域为小管径空气-水,
        # 对大管径/高压/倾斜外推须在报告中显式披露。
        usl = max(vsl, 1.0e-9)
        fs = 0.088 * (usl + 1.5) ** 2 / max(G * D, 1.0e-30)
        Ls = D * (fs * D / (1.2 * usl) + 1.0)
        # 允许低于 10D：W&H 反解式的原始域约 1~50D，10D 下限会把
        # 该式几乎整段截断（测试域内 raw 值 ≈1.0D~1.4D），失去对照意义。
        # 100D 上界保留为与其他长度式一致的数值保护。
        return clamp(Ls, 1.0 * D, 100.0 * D)
    raise ValueError(f"未知段塞长度闭合: {closure}")


def _intermittent_pressure_extra(
    closure: str,
    *, vm: float, Vf: float, rho_l: float, Lu: float,
) -> float:
    """间歇流压降附加项（dp/dL，Pa/m，沿流动方向为正耗散）。

    ``none`` 保持 Xiao Eq.31 原状（0）。``cook_behnia2000`` 计入
    Cook & Behnia (2000) Eq.19 的液膜卷入液塞鼻部混合损失：每个
    单元的压力损失 ΔP = K·ρL/2·(um−uf)²，K=0.3（其正文由 20°
    扩散角查扩散器损失曲线取 0.3），按单元长度 Lu 平均折成梯度。
    um 对应宿主混合折算速度 vm；uf 对应膜区液速 Vf（Xiao Eq.24）。
    """
    if closure == "none":
        return 0.0
    if closure in {"cook_behnia2000", "cook_behnia2000_k0.3", "xiao_mixing"}:
        k_coeff = 0.3
        du = vm - Vf
        unit_loss = k_coeff * 0.5 * rho_l * du * du
        return unit_loss / max(Lu, 1.0e-9)
    raise ValueError(f"未知间歇流压降附加项: {closure}")


def solve_intermittent(
    vsg: float, vsl: float, vm: float,
    rho_l: float, rho_g: float, mu_l: float, mu_g: float,
    D: float, eps: float, theta_rad: float, sigma: float,
    P_Pa: float = 101325.0,
    translation_closure: str = "xiao",
    slug_holdup_closure: str = "gregory1978",
    film_interfacial_closure: str = "constant",
    slug_length_closure: str = "scott1987",
    pressure_extra_closure: str = "none",
) -> dict:
    """
    间歇流求解器 — Xiao Eq 23-40 (slug 单元模型 + 膜区动量平衡)。

    与初版的关键区别：求解膜区动量方程 (Eq 29) 得到 Ef，
    用于计算加权摩擦 (Eq 31)。
    """
    sin_theta = math.sin(theta_rad)
    cos_theta = abs(math.cos(theta_rad))
    if cos_theta < 0.02: cos_theta = 0.02
    rr = eps / max(D, 1e-12)
    A = math.pi * D * D / 4.0
    drho = rho_l - rho_g

    # (a) 液塞体含液率闭合。Gregory 保持原 Xiao 行为，其余为文献候选。
    Es = _slug_body_liquid_holdup(
        slug_holdup_closure, vm=vm, rho_l=rho_l, rho_g=rho_g,
        mu_l=mu_l, mu_g=mu_g, D=D, theta_rad=theta_rad, sigma=sigma,
    )

    # (b) 平移速度 — Bendiksen Eq 37
    Re_ns = rho_l * vm * D / max(mu_l, 1e-12)
    C0_xiao = 1.2 if Re_ns > 8000 else 2.0
    Vd_xiao = (
        0.35 * math.sqrt(G * D) * sin_theta
        + 0.54 * math.sqrt(G * D) * math.cos(theta_rad)
    )
    C0_olga_density = 1.0271
    Vd_olga_density = 0.5571 * math.sqrt(
        G * D * max(drho, 0.0) / max(rho_l, 1e-12))
    if translation_closure == "xiao":
        C0 = C0_xiao
        Vd = Vd_xiao
    elif translation_closure == "olga_horizontal":
        # OLGA 段塞 unit-cell 蒸馏：PV=1.0271*j+0.5571*sqrt(gD)。
        # 该式由水平数据得到，倾斜工况只作为迁移敏感性测试。
        C0 = 1.0271
        Vd = 0.5571 * math.sqrt(G * D)
    elif translation_closure in {
        "olga_density_scaled", "olga_density_scaled_gated9",
        "olga_density_scaled_smooth9",
    }:
        # 对 OLGA 水平式的浮力尺度作密度差修正，检验跨流体迁移性。
        if (translation_closure == "olga_density_scaled_gated9"
                and abs(math.degrees(theta_rad)) > 9.0):
            # 现有实验来源均衡搜索得到的透明适用域：高于 9° 时回退 Xiao。
            C0 = C0_xiao
            Vd = Vd_xiao
        elif translation_closure == "olga_density_scaled_smooth9":
            # 研究候选：在 8°--10° 用三次 smoothstep 连续回退，避免整管
            # 坡度穿越 9° 时 Vt、持液率和压降发生硬跳变。
            abs_theta_deg = abs(math.degrees(theta_rad))
            if abs_theta_deg <= 8.0:
                olga_weight = 1.0
            elif abs_theta_deg >= 10.0:
                olga_weight = 0.0
            else:
                x_gate = (abs_theta_deg - 8.0) / 2.0
                olga_weight = 1.0 - x_gate * x_gate * (3.0 - 2.0 * x_gate)
            C0 = olga_weight * C0_olga_density + (1.0 - olga_weight) * C0_xiao
            Vd = olga_weight * Vd_olga_density + (1.0 - olga_weight) * Vd_xiao
        else:
            C0 = C0_olga_density
            Vd = Vd_olga_density
    elif translation_closure == "petalas":
        from steady.drift_flux_model import calc_C0_intermittent, calc_Vd_elongated

        C0 = calc_C0_intermittent(vm, D, rho_l, mu_l, sin_theta)
        Vd = calc_Vd_elongated(D, rho_l, rho_g, sigma, theta_rad, mu_l)
    elif translation_closure == "nicklin":
        # Nicklin 型长泡速度：分布参数 1.2 + 浮力漂移项。
        C0 = 1.2
        Vd = 0.35 * math.sqrt(G * D * max(drho, 0.0) / max(rho_l, 1e-12))
    elif translation_closure == "theron1989":
        # Théron (1989) 连续倾角式；公式按 Rodrigues et al. (2007)
        # 的闭合关系综述表独立实现：Vt=C0*Vm+C1*sqrt(gD)。
        fr_l = vsl / max(math.sqrt(G * D), 1e-12)
        gamma = 1.0 + fr_l * math.cos(theta_rad) / 3.5
        C0 = 1.3 - 0.23 / gamma + 0.13 * sin_theta * sin_theta
        C1 = ((-0.5 + 0.8 / gamma) * math.cos(theta_rad)
              + 0.35 * sin_theta)
        Vd = C1 * math.sqrt(G * D)
    else:
        raise ValueError(f"未知间歇流平移速度闭合: {translation_closure}")
    Vt = C0 * vm + Vd
    if Vt <= 0: Vt = 0.1

    # (c) 分散气泡速度 — Xiao Eq.38
    V0 = 1.53 * (G * sigma * drho / max(rho_l ** 2, 1e-12)) ** 0.25
    V0 *= sin_theta * (Es ** 0.1)
    Vb = 1.2 * vm + V0

    # (d) 段塞长度 — Scott 1987 Eq 40 基线（可替换闭合）
    Ls = _slug_length(
        slug_length_closure, D=D, vm=vm, vsl=vsl,
        rho_l=rho_l, rho_g=rho_g, mu_l=mu_l, mu_g=mu_g,
    )

    # (e) 整体持液率 (Eq 28) — 初值
    if Vt > 0:
        EL_raw = (Vt * Es + Vb * (1.0 - Es) - vsg) / Vt
    else:
        EL_raw = float("nan")

    lambda_l = vsl / max(vm, 1e-12)
    if math.isfinite(EL_raw):
        EL = EL_raw
    else:
        EL = max(2.0 * lambda_l, 0.02)

    # (f) 膜区动量平衡 (Eq 29) — 求解 Ef
    # 膜区类似分层流，但 V_f 由连续性决定
    Ef_solution = _solve_film_region(
        vsg, vsl, vm, Vt, Vb, Es, EL, rho_l, rho_g,
        mu_l, mu_g, D, eps, theta_rad, sigma, P_Pa,
        interfacial_closure=film_interfacial_closure,
    )
    Ef = Ef_solution["Ef"]
    hL_film = Ef_solution["hL_tilde"]

    if not math.isfinite(EL):
        EL = lambda_l
    EL = clamp(EL, max(0.01, lambda_l), 0.99)

    # (g) Xiao Eq.30
    VL_slug = Ef_solution["VL_slug"]
    Vf = Ef_solution["Vf"]
    numerator = VL_slug * Es - Vf * Ef
    denominator = vsl - Vf * Ef
    # Lu/Ls 是正的几何长度比。Eq.30 的分母在膜区液体通量接近总液体
    # 通量时会穿过零点；若直接把负比值夹到 1，压降权重会从 1/20
    # 瞬间跳到 1，并令外层 P/T 定点迭代形成极限环。取比值幅值可连续
    # 延拓该长度尺度，随后仍由原有 [1, 20] 物理保护限制。
    Lu_over_Ls = _bounded_unit_length_ratio(numerator, denominator)
    Lu = Ls * Lu_over_Ls
    Ls_over_Lu = 1.0 / Lu_over_Ls
    Lf_over_Lu = 1.0 - Ls_over_Lu

    # (h) 压降 (Eq 31): 段塞区 + 膜区加权
    # 段塞区摩擦 — τ_s * πD / A (Xiao Eq 31 第一项)
    rho_s = Es * rho_l + (1.0 - Es) * rho_g
    mu_s = Es * mu_l + (1.0 - Es) * mu_g
    Re_s = rho_s * vm * D / max(mu_s, 1e-12)
    fs = _friction_factor_turbulent(Re_s, rr)
    tau_s = fs * rho_s * vm * abs(vm) / 8.0
    dp_slug = 4.0 * tau_s / D  # dp/dL = τ_s * πD / (πD²/4) = 4τ_s/D

    # Xiao Eq.31: film-zone liquid-wall and gas-wall shear.
    tau_wL_f = Ef_solution.get("tau_wL", tau_s)
    tau_wG_f = Ef_solution.get("tau_wG", tau_s)
    S_L_f = Ef_solution.get("S_L", math.pi * D)
    S_G_f = Ef_solution.get("S_G", math.pi * D)
    dp_film = (tau_wL_f * S_L_f + tau_wG_f * S_G_f) / max(A, 1e-12)

    # 加权总摩擦 (Eq 31) + 可选附加项（混合损失等）
    dp_fric = dp_slug * Ls_over_Lu + dp_film * Lf_over_Lu
    dp_extra = _intermittent_pressure_extra(
        pressure_extra_closure, vm=vm, Vf=Vf, rho_l=rho_l, Lu=Lu,
    )
    dp_fric = dp_fric + dp_extra

    rho_m = EL * rho_l + (1.0 - EL) * rho_g
    dp_grav = rho_m * G * sin_theta
    dp_total = dp_fric + dp_grav

    EG = 1.0 - EL
    return {
        "EL": EL, "EG": EG,
        "dp_fric": dp_fric, "dp_grav": dp_grav, "dp_total": dp_total,
        "rho_m": rho_m, "mu_m": mu_l,
        "Es": Es, "Ef": Ef, "Vt": Vt, "Vb": Vb,
        "Ls": Ls, "Lu": Lu, "Ls_over_Lu": Ls_over_Lu, "Lf_over_Lu": Lf_over_Lu,
        "dp_slug": dp_slug, "dp_film": dp_film,
        "film_Ef": Ef_solution["Ef"],
        "film_hL_tilde": Ef_solution["hL_tilde"],
        "film_root_count": Ef_solution["root_count"],
        "film_Vf": Ef_solution["Vf"],
        "friction_multiplier": 1.0,
        "pattern": "intermittent",
        "v_gas_actual": vsg / max(EG, 1e-12),
        "v_liq_actual": vsl / max(EL, 1e-12),
    }


def _solve_film_region(
    vsg: float, vsl: float, vm: float, Vt: float, Vb: float, Es: float,
    EL_target: float,
    rho_l: float, rho_g: float, mu_l: float, mu_g: float,
    D: float, eps: float, theta_rad: float, sigma: float,
    P_Pa: float = 101325.0,
    interfacial_closure: str = "constant",
) -> dict:
    """
    求解膜区动量方程 (Eq 29) — 类似分层流。

    Eq.25 gives slug-body liquid velocity. Eq.24 and Eq.26 then give
    film-zone liquid and gas velocities for each trial Ef.

    返回: Ef, hL_tilde, tau_f, S_f, A_f
    """
    A = math.pi * D * D / 4.0
    sin_theta = math.sin(theta_rad)
    rr = eps / max(D, 1e-12)
    drho = rho_l - rho_g
    VL_slug = (vm - Vb * (1.0 - Es)) / max(Es, 1e-12)

    def momentum_balance(hL_tilde: float) -> float:
        geom = _stratified_geometry(hL_tilde, D)
        AL_v, AG_v = geom["AL"], geom["AG"]
        if AL_v <= 0 or AG_v <= 0:
            return 1e6

        Ef_local = AL_v / A
        Vf = Vt - (Vt - VL_slug) * Es / max(Ef_local, 1e-12)
        VGf = (vm - Vf * Ef_local) / max(1.0 - Ef_local, 1e-12)

        # 液相壁面摩擦
        Re_f = rho_l * abs(Vf) * geom["DL"] / max(mu_l, 1e-12)
        fL = _friction_factor_turbulent(Re_f, rr)
        tau_wL = fL * rho_l * Vf * abs(Vf) / 8.0

        # 气相壁面摩擦
        Re_G = rho_g * abs(VGf) * geom["DG"] / max(mu_g, 1e-12)
        fG = _friction_factor_turbulent(Re_G, rr)
        tau_wG = fG * rho_g * VGf * abs(VGf) / 8.0

        # Xiao Eq.33 原式为常数；实验接口允许迁入分层流界面闭合。
        fi = _stratified_fi_selected(
            interfacial_closure,
            vsg=vsg, VG=VGf, VL=Vf, hL_tilde=hL_tilde, geom=geom, fG=fG,
            rho_l=rho_l, rho_g=rho_g, mu_l=mu_l, mu_g=mu_g,
            D=D, eps=eps, sigma=sigma, pressure_Pa=P_Pa, theta_rad=theta_rad,
            vsl=vsl,
        )

        Vi = VGf - Vf
        tau_i = fi * rho_g * Vi * abs(Vi) / 8.0

        term_L = tau_wL * geom["SL"] / max(AL_v, 1e-12)
        term_G = tau_wG * geom["SG"] / max(AG_v, 1e-12)
        term_i = tau_i * geom["Si"] * (1.0 / max(AL_v, 1e-12) + 1.0 / max(AG_v, 1e-12))
        term_g = drho * G * sin_theta
        return term_L - term_G - term_i + term_g

    # 扫描找根
    n_scan = XIAO_ROOT_SCAN_POINTS
    roots = []
    h_prev, f_prev = 1e-6, momentum_balance(1e-6)
    for i in range(1, n_scan + 1):
        h_curr = i / n_scan * (1.0 - 1e-6)
        f_curr = momentum_balance(h_curr)
        if f_prev * f_curr <= 0:
            roots.append(_bracketed_secant_bisection(
                momentum_balance, h_prev, h_curr, f_prev, f_curr
            ))
        h_prev, f_prev = h_curr, f_curr

    if not roots:
        # 回退: 使用整体 EL 作为 Ef 估计
        hL_tilde = 0.1
        best_f = abs(momentum_balance(hL_tilde))
        for h_try in [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]:
            f_try = abs(momentum_balance(h_try))
            if f_try < best_f: best_f, hL_tilde = f_try, h_try
    else:
        compatible = []
        for root in roots:
            Ef_root = _stratified_geometry(root, D)["AL"] / A
            if Ef_root <= EL_target <= Es:
                compatible.append(root)
        if compatible:
            # 膜区根还须与 Eq.27--30 的 slug-unit 平均持液率相容。
            # 取 Ef 最接近 EL_target 的物理解，避免在多个相容根之间因
            # 很小的 P/T 扰动跳到另一分支，导致压降不连续和定点极限环。
            hL_tilde = min(
                compatible,
                key=lambda root: abs(
                    _stratified_geometry(root, D)["AL"] / A - EL_target
                ),
            )
        else:
            hL_tilde = min(
                roots,
                key=lambda root: abs(
                    _stratified_geometry(root, D)["AL"] / A - EL_target
                ),
            )

    geom = _stratified_geometry(hL_tilde, D)
    Ef_val = geom["AL"] / A
    Vf_val = Vt - (Vt - VL_slug) * Es / max(Ef_val, 1e-12)
    VGf_val = (vm - Vf_val * Ef_val) / max(1.0 - Ef_val, 1e-12)

    # 液相壁面摩擦
    Re_f = rho_l * abs(Vf_val) * geom["DL"] / max(mu_l, 1e-12)
    fL = _friction_factor_turbulent(Re_f, rr)
    tau_f_val = fL * rho_l * Vf_val * abs(Vf_val) / 8.0

    # 气相壁面 + 界面摩擦 (用于膜区总压降)
    Re_G = rho_g * abs(VGf_val) * geom["DG"] / max(mu_g, 1e-12)
    fG_val = _friction_factor_turbulent(Re_G, rr)
    tau_wG_val = fG_val * rho_g * VGf_val * abs(VGf_val) / 8.0

    fi_val = _stratified_fi_selected(
        interfacial_closure,
        vsg=vsg, VG=VGf_val, VL=Vf_val, hL_tilde=hL_tilde,
        geom=geom, fG=fG_val, rho_l=rho_l, rho_g=rho_g,
        mu_l=mu_l, mu_g=mu_g, D=D, eps=eps, sigma=sigma,
        pressure_Pa=P_Pa, theta_rad=theta_rad, vsl=vsl,
    )
    tau_i_val = fi_val * rho_g * (VGf_val - Vf_val) * abs(VGf_val - Vf_val) / 8.0

    return {
        "Ef": clamp(Ef_val, 0.01, 0.99),
        "hL_tilde": hL_tilde,
        "VL_slug": VL_slug,
        "Vf": Vf_val,
        "VGf": VGf_val,
        "tau_f": tau_f_val,        # 液膜壁面剪应力
        "tau_wL": tau_f_val,
        "tau_wG": tau_wG_val,       # 气相壁面剪应力
        "tau_i": tau_i_val,         # 界面剪应力
        "S_f": geom["SL"],          # 液膜湿周
        "S_L": geom["SL"],
        "S_G": geom["SG"],          # 气相湿周
        "Si": geom["Si"],           # 界面宽度
        "A_f": geom["AL"],          # 液膜面积
        "A_G": geom["AG"],          # 气相面积
        "root_count": len(roots),
    }


# ============================================================
# Layer 5: 环雾流求解器 (Oliemans 夹带)
# ============================================================

def _oliemans_FE(rho_l: float, rho_g: float, mu_l: float, mu_g: float,
                  sigma: float, vsl: float, vsg: float, D: float) -> float:
    """
    Oliemans et al. (1986) 夹带分数 — Xiao Eq 51。

    FE/(1-FE) = 10^β0 * ρL^β1 * ρg^β2 * μL^β3 * μg^β4
                * σ^β5 * D^β6 * Vsl^β7 * Vsg^β8 * g^β9

    系数采用 Oliemans 全数据集回归值。
    """
    if vsl <= 0 or vsg <= 0:
        return 0.0
    # Oliemans et al. (1986) 全数据集回归系数（SI 单位）。
    beta = [-2.52, 1.08, 0.18, 0.27, 0.28, -1.80, 1.72, 0.70, 1.44, 0.46]
    arguments = [rho_l, rho_g, mu_l, mu_g, sigma, D, vsl, vsg, G]
    log10_ratio = beta[0]
    for exponent, value in zip(beta[1:], arguments):
        log10_ratio += exponent * math.log10(max(value, 1e-12))
    ratio = 10.0 ** clamp(log10_ratio, -20.0, 20.0)
    FE = ratio / (1.0 + ratio)
    return clamp(FE, 0.0, 0.95)


def _annular_entrainment_fraction(
    closure: str, rho_l: float, rho_g: float, mu_l: float, mu_g: float,
    sigma: float, vsl: float, vsg: float, D: float,
) -> float:
    """环状流夹带闭合选择器；OLGA 项是跨流型迁移实验。"""
    if closure == "oliemans":
        return _oliemans_FE(rho_l, rho_g, mu_l, mu_g, sigma, vsl, vsg, D)
    if closure == "none":
        return 0.0
    if closure.startswith("fixed_"):
        try:
            percent = float(closure.split("_", 1)[1])
        except ValueError as exc:
            raise ValueError(f"未知固定环状流夹带闭合: {closure}") from exc
        return clamp(percent / 100.0, 0.0, 0.95)
    if closure == "olga_generalized":
        # OLGA 五流体分层液滴场广义式，试验其迁移到环雾流 FE 的效果：
        # logit(E)=1.193ln(rho_g)-0.629ln(sigma)+2.308ln(USG)
        #          -1.811ln(D)-15.92。
        logit = (
            1.193 * math.log(max(rho_g, 1.0e-30))
            - 0.629 * math.log(max(sigma, 1.0e-30))
            + 2.308 * math.log(max(abs(vsg), 1.0e-30))
            - 1.811 * math.log(max(D, 1.0e-30))
            - 15.92
        )
        if logit >= 0.0:
            entrainment = 1.0 / (1.0 + math.exp(-min(logit, 700.0)))
        else:
            exponential = math.exp(max(logit, -700.0))
            entrainment = exponential / (1.0 + exponential)
        return clamp(entrainment, 0.0, 0.95)
    raise ValueError(f"未知环状流夹带闭合: {closure}")


def _annular_interfacial_fi(
    closure: str, *, fc: float, delta_tilde: float,
    rho_c: float, velocity_difference: float, sigma: float, D: float,
) -> float:
    """环状液膜-气芯界面摩阻，统一返回 Darcy 等效因子。"""
    interfacial_weber = (
        rho_c * velocity_difference * velocity_difference * delta_tilde * D
        / max(sigma, 1e-12)
    )
    xiao_fi = fc * (
        1.0 + 2250.0 * delta_tilde / max(interfacial_weber, 1e-12)
    )
    if closure == "xiao":
        return xiao_fi
    if closure == "smooth_core":
        return fc
    if closure == "wallis":
        # Wallis 液膜粗糙界面式：Fanning fi=0.005(1+300*delta/D)。
        return clamp(4.0 * 0.005 * (1.0 + 300.0 * delta_tilde), 0.0, 10.0)
    if closure == "xiao_wallis_blend_50":
        wallis_fi = 4.0 * 0.005 * (1.0 + 300.0 * delta_tilde)
        return clamp(0.5 * (xiao_fi + wallis_fi), 0.0, 10.0)
    raise ValueError(f"未知环状流界面摩阻闭合: {closure}")


def _solve_annular_film_height(
    vsg: float, vsl: float,
    rho_l: float, rho_g: float, mu_l: float, mu_g: float,
    D: float, eps: float, theta_rad: float, sigma: float,
    entrainment_closure: str = "oliemans",
    interfacial_closure: str = "xiao",
) -> dict:
    """环雾流膜厚求解 — Xiao Eq 45, 含 Oliemans 夹带 & 界面摩擦"""
    A = math.pi * D * D / 4.0
    sin_theta = math.sin(theta_rad)
    rr = eps / max(D, 1e-12)
    drho = rho_l - rho_g

    # 夹带分数 (不依赖 δ)
    FE = _annular_entrainment_fraction(
        entrainment_closure, rho_l, rho_g, mu_l, mu_g, sigma, vsl, vsg, D)

    def momentum_balance(delta_tilde: float) -> float:
        ann = _annular_geometry(delta_tilde, D)
        Vf = vsl * (1.0 - FE) * A / max(ann["Af"], 1e-12)
        Vc = (vsg + vsl * FE) * A / max(ann["Ac"], 1e-12)

        # 气芯密度 & 黏度 (含液滴)
        # Ec 为气芯中液滴的体积分数；上限 0.7 防止气芯密度接近液相
        # (物理上气芯不可能完全被液滴填满并保持环雾流型)
        Ec = clamp(vsl * FE / max(vsg + vsl * FE, 1e-12), 0.0, 0.7)
        rho_c = Ec * rho_l + (1.0 - Ec) * rho_g
        mu_c = Ec * mu_l + (1.0 - Ec) * mu_g

        # 液膜壁面摩擦
        Re_f = rho_l * abs(Vf) * ann["Df"] / max(mu_l, 1e-12)
        ff = _friction_factor_turbulent(Re_f, rr)
        tau_wL = ff * rho_l * Vf * abs(Vf) / 8.0

        # 气芯摩擦
        Re_c = rho_c * abs(Vc) * ann["Dc"] / max(mu_c, 1e-12)
        fc = _friction_factor_turbulent(Re_c, rr)

        Vi = Vc - Vf
        fi = _annular_interfacial_fi(
            interfacial_closure, fc=fc, delta_tilde=delta_tilde,
            rho_c=rho_c, velocity_difference=Vi, sigma=sigma, D=D,
        )
        tau_i = fi * rho_c * Vi * abs(Vi) / 8.0

        inv_Af = 1.0 / max(ann["Af"], 1e-12)
        inv_Ac = 1.0 / max(ann["Ac"], 1e-12)
        term_f = tau_wL * ann["Sf"] * inv_Af
        term_i = tau_i * ann["Si"] * (inv_Af + inv_Ac)
        term_g = (rho_l - rho_c) * G * sin_theta
        return term_f - term_i + term_g

    # 扫描找根
    n_scan = XIAO_ROOT_SCAN_POINTS
    roots = []
    d_prev, f_prev = 1e-8, momentum_balance(1e-8)
    for i in range(1, n_scan + 1):
        d_curr = i / n_scan * 0.45
        f_curr = momentum_balance(d_curr)
        if f_prev * f_curr <= 0:
            roots.append(_bracketed_secant_bisection(
                momentum_balance, d_prev, d_curr, f_prev, f_curr
            ))
        d_prev, f_prev = d_curr, f_curr

    if not roots:
        best_d, best_f = 0.05, abs(momentum_balance(0.05))
        for d_try in [1e-6, 1e-4, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.35, 0.45]:
            f_try = abs(momentum_balance(d_try))
            if f_try < best_f: best_f, best_d = f_try, d_try
        delta_tilde = best_d
        residual = best_f
    else:
        delta_tilde = min(roots)
        residual = abs(momentum_balance(delta_tilde))

    # 验证残差 — 若残差异常大，环雾流解可能不可靠
    if residual > 1e3:
        # 残差过大时回退到较保守的膜厚估计 (约 1% 管径)
        delta_tilde = max(delta_tilde, 0.01)
    elif residual > 1e-2:
        # 中等残差，保留解但标记为近似
        pass

    return {"delta_tilde": delta_tilde, "ann": _annular_geometry(delta_tilde, D), "FE": FE}


def solve_annular(
    vsg: float, vsl: float, vm: float,
    rho_l: float, rho_g: float, mu_l: float, mu_g: float,
    D: float, eps: float, theta_rad: float, sigma: float,
    entrainment_closure: str = "oliemans",
    interfacial_closure: str = "xiao",
) -> dict:
    """环雾流求解器 — Xiao Eq 41-47"""
    sin_theta = math.sin(theta_rad)
    A = math.pi * D * D / 4.0

    sol = _solve_annular_film_height(
        vsg, vsl, rho_l, rho_g, mu_l, mu_g, D, eps, theta_rad, sigma,
        entrainment_closure=entrainment_closure,
        interfacial_closure=interfacial_closure)
    delta_tilde = sol["delta_tilde"]
    ann = sol["ann"]
    FE = sol["FE"]

    # 持液率 — Eq 46
    EL = 1.0 - (1.0 - 2.0 * delta_tilde) ** 2 * vsg / max(vsg + FE * vsl, 1e-12)
    EL = clamp(EL, 0.01, 0.99)
    EG = 1.0 - EL

    Vf = vsl * (1.0 - FE) * A / max(ann["Af"], 1e-12)
    Vc = (vsg + vsl * FE) * A / max(ann["Ac"], 1e-12)

    # 压降 — Eq 47: dp/dL = 4*τ_wL/D + ρc*g*sinα ...简化
    rr = eps / max(D, 1e-12)
    Re_f = rho_l * abs(Vf) * ann["Df"] / max(mu_l, 1e-12)
    ff = _friction_factor_turbulent(Re_f, rr)
    tau_wL = ff * rho_l * Vf * abs(Vf) / 8.0
    dp_fric = 4.0 * tau_wL / D

    Ec = vsl * FE / max(vsg + vsl * FE, 1e-12)
    rho_c = Ec * rho_l + (1.0 - Ec) * rho_g
    dp_grav = (EL * rho_l + EG * rho_c) * G * sin_theta
    dp_total = dp_fric + dp_grav

    return {
        "EL": EL, "EG": EG,
        "dp_fric": dp_fric, "dp_grav": dp_grav, "dp_total": dp_total,
        "rho_m": EL * rho_l + EG * rho_c, "mu_m": mu_l,
        "delta_tilde": delta_tilde, "FE": FE,
        "friction_multiplier": 1.0,
        "pattern": "annular",
        "v_gas_actual": Vc, "v_liq_actual": Vf,
    }


# ============================================================
# Layer 6: 分散气泡流求解器
# ============================================================

def solve_dispersed_bubble(
    vsg: float, vsl: float, vm: float,
    rho_l: float, rho_g: float, mu_l: float, mu_g: float,
    D: float, eps: float, theta_rad: float, sigma: float,
    closure: str = "xiao",
) -> dict:
    """分散气泡流 — Xiao Eq 56-57: 均相无滑脱模型"""
    if closure in {"petalas_dispersed", "petalas_swarm"}:
        from steady.drift_flux_model import (
            calc_C0_intermittent, calc_Vb_bubble_swarm, calc_Vb_dispersed,
        )

        if closure == "petalas_dispersed":
            C0 = calc_C0_intermittent(vm, D, rho_l, mu_l, math.sin(theta_rad))
            Vb = calc_Vb_dispersed(rho_l, rho_g, sigma, theta_rad)
            liquid_wall_properties = False
        else:
            C0 = 1.2
            Vb = calc_Vb_bubble_swarm(rho_l, rho_g, sigma, theta_rad)
            liquid_wall_properties = True
        gas_velocity = max(C0 * vm + Vb, 1.0e-12)
        EG = clamp(vsg / gas_velocity, 0.0, vsg / max(vm, 1.0e-12))
        EL = 1.0 - EG
        rho_m = EL * rho_l + EG * rho_g
        mu_m = mu_l if liquid_wall_properties else EL * mu_l + EG * mu_g
        rho_re = rho_l if liquid_wall_properties else rho_m
        reynolds = rho_re * vm * D / max(mu_m, 1.0e-12)
        f = _friction_factor_turbulent(reynolds, eps / max(D, 1.0e-12))
        dp_fric = f * rho_m * vm * vm / (2.0 * D)
        dp_grav = rho_m * G * math.sin(theta_rad)
        return {
            "EL": EL, "EG": EG, "dp_fric": dp_fric, "dp_grav": dp_grav,
            "dp_total": dp_fric + dp_grav, "rho_m": rho_m, "mu_m": mu_m,
            "f": f, "C0": C0, "Vb": Vb, "friction_multiplier": 1.0,
            "pattern": "dispersed_bubble",
            "v_gas_actual": vsg / max(EG, 1.0e-12),
            "v_liq_actual": vsl / max(EL, 1.0e-12),
        }
    if closure != "xiao":
        from steady.olga_bubble_closure import solve_olga_bubble

        wall_closure = {
            "olga_v38_xiao_wall": "xiao_homogeneous",
            "olga_v38_v6": "olga_v6",
        }.get(closure)
        if wall_closure is None:
            raise ValueError(f"未知分散泡状流闭合: {closure}")
        return solve_olga_bubble(
            vsg, vsl, rho_l, rho_g, mu_l, mu_g, D, eps,
            theta_rad, sigma, wall_closure=wall_closure,
        )
    sin_theta = math.sin(theta_rad)
    rr = eps / max(D, 1e-12)

    EL = vsl / max(vm, 1e-12)  # 无滑脱
    EG = 1.0 - EL
    rho_m = EL * rho_l + EG * rho_g
    mu_m = EL * mu_l + EG * mu_g
    Re_m = rho_m * vm * D / max(mu_m, 1e-12)
    f = _friction_factor_turbulent(Re_m, rr)

    dp_fric = f * rho_m * vm * vm / (2.0 * D)  # Darcy-Weisbach
    dp_grav = rho_m * G * sin_theta
    dp_total = dp_fric + dp_grav

    return {
        "EL": EL, "EG": EG,
        "dp_fric": dp_fric, "dp_grav": dp_grav, "dp_total": dp_total,
        "rho_m": rho_m, "mu_m": mu_m, "f": f,
        "friction_multiplier": 1.0,
        "pattern": "dispersed_bubble",
        "v_gas_actual": vsg / max(EG, 1e-12),
        "v_liq_actual": vsl / max(EL, 1e-12),
    }


def solve_single_phase_limit(
    velocity: float,
    density: float,
    viscosity: float,
    D: float,
    eps: float,
    theta_rad: float,
    phase: str,
) -> dict:
    """Exact single-phase Darcy--Weisbach limit after flash phase loss."""
    if phase not in {"liquid", "gas"}:
        raise ValueError(f"未知单相极限: {phase}")
    reynolds = density * abs(velocity) * D / max(viscosity, 1.0e-12)
    friction = _friction_factor(reynolds, eps / max(D, 1.0e-12))
    dp_friction = friction * density * velocity * abs(velocity) / (2.0 * D)
    dp_gravity = density * G * math.sin(theta_rad)
    liquid = phase == "liquid"
    return {
        "EL": 1.0 if liquid else 0.0,
        "EG": 0.0 if liquid else 1.0,
        "dp_fric": dp_friction,
        "dp_grav": dp_gravity,
        "dp_total": dp_friction + dp_gravity,
        "rho_m": density,
        "mu_m": viscosity,
        "f": friction,
        "friction_multiplier": 1.0,
        "pattern": f"single_phase_{phase}",
        "v_gas_actual": 0.0 if liquid else velocity,
        "v_liq_actual": velocity if liquid else 0.0,
    }


# ============================================================
# Layer 7: 主入口
# ============================================================

def xiao_pressure_gradient(inp: XiaoInput) -> XiaoResult:
    """Xiao, Shoham & Brill (1990) 机理模型主入口。"""
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

    A = math.pi * D * D / 4.0
    vsg = x_flow * m_total / max(rho_g * A, 1e-12)
    vsl = (1.0 - x_flow) * m_total / max(rho_l * A, 1e-12)
    vm = vsg + vsl
    if vm <= 0:
        raise ValueError("混合表观速度必须大于 0。")

    phase_limit_key = inp.single_phase_limit_closure.strip().lower()
    if phase_limit_key not in {"legacy", "darcy"}:
        raise ValueError(
            f"未知 Xiao 单相极限闭合: {inp.single_phase_limit_closure}")

    sol = None
    pattern = None
    if phase_limit_key == "darcy" and x_flow <= 0.0:
        pattern = "single_phase_liquid"
        sol = solve_single_phase_limit(
            vsl, rho_l, mu_l, D, eps, theta, phase="liquid")
    elif phase_limit_key == "darcy" and x_flow >= 1.0:
        pattern = "single_phase_gas"
        sol = solve_single_phase_limit(
            vsg, rho_g, mu_g, D, eps, theta, phase="gas")

    # 两相区才调用流型图；精确单相点由上面的相态路由截获。
    if sol is None:
        if inp.flow_pattern_override is None:
            pattern = determine_flow_pattern(
                vsg, vsl, vm, rho_l, rho_g, mu_l, mu_g, D, eps, theta, sigma,
                closure=inp.flow_pattern_closure)
        else:
            pattern = inp.flow_pattern_override.strip().lower()
            allowed = {
                "stratified", "stratified_smooth", "stratified_wavy",
                "intermittent", "annular", "dispersed_bubble",
            }
            if pattern not in allowed:
                raise ValueError(f"未知 Xiao 强制流型: {inp.flow_pattern_override}")

    # 调用对应求解器 (将 smooth/wavy 归一化为 stratified)
    base_pattern = pattern.replace("stratified_smooth", "stratified").replace("stratified_wavy", "stratified")

    # 实际压力用于 Andritsos 界面摩擦
    P_actual = inp.pressure_Pa

    if sol is not None:
        pass
    elif base_pattern == "dispersed_bubble":
        sol = solve_dispersed_bubble(
            vsg, vsl, vm, rho_l, rho_g, mu_l, mu_g, D, eps, theta, sigma,
            closure=inp.dispersed_bubble_closure,
        )
    elif base_pattern == "stratified":
        sol = solve_stratified(
            vsg, vsl, vm, rho_l, rho_g, mu_l, mu_g, D, eps, theta, sigma,
            P_actual, interfacial_closure=inp.stratified_interfacial_closure,
            wall_closure=inp.stratified_wall_friction_closure)
    elif base_pattern == "annular":
        sol = solve_annular(
            vsg, vsl, vm, rho_l, rho_g, mu_l, mu_g, D, eps, theta, sigma,
            entrainment_closure=inp.annular_entrainment_closure,
            interfacial_closure=inp.annular_interfacial_closure)
    else:  # intermittent
        sol = solve_intermittent(
            vsg, vsl, vm, rho_l, rho_g, mu_l, mu_g, D, eps, theta, sigma,
            P_actual, translation_closure=inp.intermittent_translation_closure,
            slug_holdup_closure=inp.intermittent_slug_holdup_closure,
            film_interfacial_closure=inp.intermittent_film_interfacial_closure,
            slug_length_closure=inp.intermittent_slug_length_closure,
            pressure_extra_closure=inp.intermittent_pressure_extra_closure)

    HL = sol["EL"]
    Hg = sol["EG"]
    flow_pattern_out = base_pattern

    # 后处理
    x_inv = calc_inventory_gas_mass_fraction(rho_g, rho_l, HL)
    lam = calc_no_slip_liquid_fraction(vsl, vsg)
    rho_ns = calc_no_slip_density(rho_g, rho_l, lam)
    mu_ns = calc_no_slip_viscosity(mu_g, mu_l, lam)
    Re_ns = rho_ns * vm * D / max(mu_ns, 1e-12)
    rr = eps / max(D, 1e-12)
    f_ns = _friction_factor(Re_ns, rr)
    v_gas_actual = sol.get("v_gas_actual", vsg / max(Hg, 1e-12))
    v_liq_actual = sol.get("v_liq_actual", vsl / max(HL, 1e-12))
    slip_ratio = v_gas_actual / max(v_liq_actual, 1e-12)
    rho_holdup = calc_holdup_density(rho_g, rho_l, HL)
    Nlv = vsl * (rho_l / (G * sigma)) ** 0.25
    Fr = vm / math.sqrt(G * D)

    return XiaoResult(
        dp_dL_friction=sol["dp_fric"],
        dp_dL_gravity=sol["dp_grav"],
        dp_dL_total=sol["dp_total"],
        liquid_holdup_inclined=HL,
        gas_void_fraction=Hg,
        gas_mass_fraction_inventory=x_inv,
        gas_mass_fraction_flow=x_flow,
        flow_pattern=flow_pattern_out,
        v_sg=vsg, v_sl=vsl, v_m=vm,
        v_gas_actual=v_gas_actual, v_liquid_actual=v_liq_actual,
        slip_ratio=slip_ratio,
        rho_no_slip=rho_ns, rho_holdup=rho_holdup, mu_no_slip=mu_ns,
        no_slip_liquid_fraction=lam,
        f_no_slip=f_ns, friction_multiplier=sol.get("friction_multiplier", 1.0),
        f_two_phase=sol.get("f", sol.get("fi", f_ns)),
        froude_number=Fr, reynolds_no_slip=Re_ns, liquid_velocity_number=Nlv,
        iterations=1, converged=True,
        debug_film_Ef=sol.get("film_Ef", float("nan")),
        debug_film_hL_tilde=sol.get("film_hL_tilde", float("nan")),
        debug_film_root_count=sol.get("film_root_count", 0),
        debug_film_Vf=sol.get("film_Vf", float("nan")),
        debug_dp_slug=sol.get("dp_slug", float("nan")),
        debug_dp_film=sol.get("dp_film", float("nan")),
        debug_Ls_over_Lu=sol.get("Ls_over_Lu", float("nan")),
    )
