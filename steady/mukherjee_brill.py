# mukherjee_brill.py
"""
Mukherjee & Brill (1985) 两相流压降模型。

参考文献：
    Mukherjee, H., and Brill, J.P. (1985). "Pressure Drop Correlations for
    Inclined Two-Phase Flow." Journal of Energy Resources Technology, 107(4), 549-554.

本模块实现了：
    - 流型预测（气团/段塞/环雾/分层）
    - 液相持液率 HL（倾斜修正）
    - 摩擦压降梯度（不同流型）
    - 重力梯度
    - 加速梯度（除分层流外，按论文 Eq. (1) 的隐式形式求解）
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

# ============================================================
# 输入输出 dataclass
# ============================================================

@dataclass
class MukherjeeBrillInput:
    """Mukherjee & Brill 模型输入参数（含平均压力用于加速项）"""
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
    pressure_avg: float = 101325.0  # 平均压力 [Pa]，用于加速项计算


@dataclass
class MukherjeeBrillResult:
    """Mukherjee & Brill 模型输出结果"""
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
# 辅助函数
# ============================================================

def _colebrook(Re: float, rr: float) -> float:
    """MB 论文 Eq. (3)--(4) 的 Darcy/Moody 摩擦因子。

    这里不能以常见的 Colebrook--White ``3.7/2.51`` 形式替代。Mukherjee
    与 Brill 在 Eq. (4) 明确使用

    ``1/sqrt(f) = 1.74 - 2 log10(eps/(2d) + 18.7/(Re sqrt(f)))``。

    用 ``x=1/sqrt(f)`` 后左、右两侧之差在正区间单调，因此采用有界二分。
    相比固定点迭代，它在粗糙管和 Re 接近 2000 时不会产生负摩阻或振荡。
    """
    if Re <= 0:
        return 0.0
    if Re < 2000:
        return 64.0 / Re

    def residual(x: float) -> float:
        # x=1/sqrt(f); rr=eps/d，故 eps/(2d)=rr/2。
        return x - 1.74 + 2.0 * math.log10(rr / 2.0 + 18.7 * x / Re)

    lo, hi = 1.0e-12, 100.0
    # 对 Re>=2000 和 rr>=0，此根一定落在该区间；保留检查以免未来调用
    # 传入异常物性时静默返回非物理解。
    if residual(lo) >= 0.0 or residual(hi) <= 0.0:
        raise ValueError("MB Eq. (4) 摩阻因子求根未能建立正根区间。")
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        value = residual(mid)
        if abs(value) < 1.0e-13:
            lo = hi = mid
            break
        if value > 0.0:
            hi = mid
        else:
            lo = mid
    x = 0.5 * (lo + hi)
    return 1.0 / (x * x)


def _dimensionless_numbers(
    vsg: float, vsl: float, rho_l: float, mu_l: float, sigma: float
) -> tuple[float, float, float]:
    """计算液相速度数 Nlv，气相速度数 Ngv，液相黏度数 Nl"""
    g = 9.80665
    denom = (rho_l / (g * sigma)) ** 0.25
    Nlv = vsl * denom
    Ngv = vsg * denom
    Nl = mu_l * (g / (rho_l * sigma ** 3)) ** 0.25
    return Ngv, Nlv, Nl


def _predict_flow_pattern(
    Ngv: float, Nlv: float, Nl: float, theta_rad: float
) -> str:
    """
    流型预测 (Mukherjee & Brill 1985)

    论文第 3 页明确区分为：
    - 上坡: 使用 Eq.(6) 的 bubble-slug 边界
    - 水平与下坡: 使用 Eq.(8) 的 bubble-slug 边界，
      且允许进入 Eq.(9) 的 stratified 区域
    """
    sin_theta = math.sin(theta_rad)

    # --- 上坡 ---
    if theta_rad > 0.0:
        # 气团-段塞过渡 (Eq.6)
        Nlv_bubble_trans = 10 ** (
            math.log10(Ngv) + 0.940 + 0.074 * sin_theta - 0.855 * sin_theta ** 2 + 3.695 * Nl
        )
        if Nlv >= Nlv_bubble_trans:
            pattern = "bubble"
        else:
            pattern = "slug"

        # 段塞-环雾过渡 (Eq.7)，与倾角无关
        Ngv_trans_ann = 10 ** (1.401 - 2.694 * Nl + 0.521 * Nlv ** 0.329)
        if Ngv > Ngv_trans_ann:
            pattern = "annular-mist"
        return pattern

    # --- 水平与下坡 ---
    # Eq. (9) 的原文名称和正文均为 "downhill stratified flow boundary"。
    # 因此它只用于严格下坡；水平流仍应走 Eq. (8) 的 bubble--slug 边界。
    if theta_rad < 0.0:
        Nlv_strat = 10 ** (
            0.321 - 0.017 * Ngv - 4.267 * sin_theta - 2.972 * Nl
            - 0.033 * (math.log10(Ngv)) ** 2 - 3.925 * sin_theta ** 2
        )
        if Nlv <= Nlv_strat:
            return "stratified"

    # 非分层水平/下坡：气团-段塞过渡 (Eq.8)
    log10_Nlv = math.log10(Nlv)
    Ngv_slug = 10 ** (
        0.431 + 1.132 * sin_theta - 3.003 * Nl
        - 1.138 * log10_Nlv * sin_theta - 0.429 * (log10_Nlv ** 2) * sin_theta
    )
    if Ngv <= Ngv_slug:
        pattern = "bubble"
    else:
        pattern = "slug"

    # 段塞-环雾过渡 (Eq.7)
    Ngv_trans_ann = 10 ** (1.401 - 2.694 * Nl + 0.521 * Nlv ** 0.329)
    if Ngv > Ngv_trans_ann:
        pattern = "annular-mist"
    return pattern


def determine_flow_pattern_from_superficial_velocities(
    vsg: float,
    vsl: float,
    rho_l: float,
    mu_l: float,
    sigma: float,
    theta_rad: float,
) -> str:
    """Return the Mukherjee--Brill flow pattern from physical inputs.

    This small public adapter keeps the published dimensionless-number
    conversion and the flow-map equations in one place.  It is also used by
    the Xiao research switch, so the comparison does not duplicate or drift
    away from the standalone Mukherjee--Brill implementation.
    """
    if vsg <= 0.0 or vsl <= 0.0:
        raise ValueError("Mukherjee--Brill flow-pattern map requires vsg and vsl > 0.")
    if rho_l <= 0.0 or mu_l <= 0.0 or sigma <= 0.0:
        raise ValueError("Mukherjee--Brill flow-pattern properties must be positive.")
    Ngv, Nlv, Nl = _dimensionless_numbers(vsg, vsl, rho_l, mu_l, sigma)
    return _predict_flow_pattern(Ngv, Nlv, Nl, theta_rad)


def _calc_holdup(
    Ngv: float, Nlv: float, Nl: float, theta_rad: float, flow_pattern: str
) -> float:
    """
    计算液相持液率 HL (Table 1)
    Returns: Eq. (10) 的原始回归值。

    Notes
    -----
    论文未规定 ``HL >= lambda_L`` 的下限修正，也未规定将回归值静默截断
    到 [0, 1]。因此这里保留原始值；主函数会在值超出物理范围时显式报错，
    以便暴露相关式适用范围之外的工况。
    """
    sin_theta = math.sin(theta_rad)

    if theta_rad >= 0.0:
        # 上坡 & 水平: 通用参数
        C1, C2, C3, C4, C5, C6 = (
            -0.380113, 0.129875, -0.119788, 2.343227, 0.475686, 0.288657
        )
    else:
        # 下坡
        if flow_pattern == "stratified":
            C1, C2, C3, C4, C5, C6 = (
                -1.330282, 4.808139, 4.171584, 56.262268, 0.079951, 0.504887
            )
        else:
            C1, C2, C3, C4, C5, C6 = (
                -0.516644, 0.789805, 0.551627, 15.519214, 0.371771, 0.393952
            )

    exponent = (
        C1
        + C2 * sin_theta
        + C3 * sin_theta ** 2
        + C4 * Nl ** 2
    )
    HL = math.exp(exponent * (Ngv ** C5) / (Nlv ** C6))
    if not math.isfinite(HL):
        raise ValueError("Eq. (10) 持液率计算得到非有限值。")
    return HL


def _stratified_geometry(HL: float, d: float) -> tuple:
    """
    分层流几何参数 (Newton-Raphson 解中心角)

    Parameters
    ----------
    HL : 液相持液率 [0,1]
    d  : 管径 [m]

    Returns
    -------
    theta : 液相湿周中心角 [rad]
    A_l   : 液相面积 [m²]
    A_g   : 气相面积 [m²]
    S_l   : 液相壁面湿周 [m]
    S_g   : 气相壁面湿周 [m]
    S_i   : 气液界面宽度 [m]
    """
    A = math.pi * d * d / 4.0

    if HL <= 0.0:
        return 0.0, 0.0, A, 0.0, math.pi * d, 0.0
    if HL >= 1.0:
        return 2.0 * math.pi, A, 0.0, math.pi * d, 0.0, 0.0

    # HL = (theta - sin(theta)) / (2π)
    # 该方程在 HL 接近 0/1 时 Newton 法容易因导数趋近 0 而发散，
    # 这里改用单调区间 [0, 2π] 上的二分法求根。
    target = 2.0 * math.pi * HL
    lo = 0.0
    hi = 2.0 * math.pi
    theta = target
    for _ in range(80):
        theta = 0.5 * (lo + hi)
        f_val = theta - math.sin(theta) - target
        if abs(f_val) < 1e-14 or (hi - lo) < 1e-12:
            break
        if f_val > 0.0:
            hi = theta
        else:
            lo = theta

    A_l = A * HL
    A_g = A * (1.0 - HL)
    S_l = d * theta / 2.0
    S_g = math.pi * d - S_l
    S_i = d * math.sin(theta / 2.0)

    return theta, A_l, A_g, S_l, S_g, S_i


def _interp_fR(HR: float) -> float:
    """
    环雾流摩擦因子比 fR (Table 2)
    """
    HR_table = np.array([0.01, 0.20, 0.30, 0.40, 0.50, 0.70, 1.00, 10.00])
    fR_table = np.array([1.00, 0.98, 1.20, 1.25, 1.30, 1.25, 1.00, 1.00])
    # 外推时使用边界值
    if HR <= HR_table[0]:
        return fR_table[0]
    if HR >= HR_table[-1]:
        return fR_table[-1]
    return float(np.interp(HR, HR_table, fR_table))


# ============================================================
# 主函数
# ============================================================

def mukherjee_brill_pressure_gradient(inp: MukherjeeBrillInput) -> MukherjeeBrillResult:
    """
    使用 Mukherjee & Brill (1985) 模型计算两相流压降。

    参数
    ----------
    inp : MukherjeeBrillInput

    返回
    -------
    MukherjeeBrillResult
    """
    # --- 1. 解析输入 ---
    m_total = inp.total_mass_flow
    x = inp.gas_mass_fraction
    rho_g = inp.rho_gas
    rho_l = inp.rho_liquid
    mu_g = inp.mu_gas
    mu_l = inp.mu_liquid
    d = inp.diameter
    eps = inp.roughness
    theta = inp.angle_rad
    sigma = inp.surface_tension
    g = 9.80665

    if m_total <= 0 or d <= 0:
        raise ValueError("质量流量和管径必须大于零。")
    if not 0.0 <= x <= 1.0:
        raise ValueError("气相质量分数必须在 [0, 1] 内。")
    if rho_g <= 0 or rho_l <= 0 or mu_g <= 0 or mu_l <= 0:
        raise ValueError("气液密度和黏度必须大于零。")
    if sigma <= 0:
        raise ValueError("表面张力必须大于零。")
    if eps < 0:
        raise ValueError("绝对粗糙度不能为负。")

    # --- 1.5 单相边界保护 ---
    # MB 模型是气液两相相关式；x=0 或 x=1 时 Ngv/Nlv = 0，log10 报错
    if x <= 1e-12:
        # 纯液相
        A_pure = math.pi * d * d / 4.0
        vsl_pure = m_total / (rho_l * A_pure)
        vm_pure = vsl_pure
        Re_l = rho_l * vsl_pure * d / mu_l if mu_l > 0 else 1e6
        f_l = _colebrook(Re_l, eps / d)
        dp_dz_f = f_l * rho_l * vsl_pure ** 2 / (2.0 * d)
        dp_dz_g = rho_l * g * math.sin(theta)
        dp_dz_total = dp_dz_f + dp_dz_g
        return MukherjeeBrillResult(
            dp_dL_friction=dp_dz_f,
            dp_dL_gravity=dp_dz_g,
            dp_dL_total=dp_dz_total,
            liquid_holdup_inclined=1.0,
            gas_void_fraction=0.0,
            gas_mass_fraction_inventory=0.0,
            gas_mass_fraction_flow=x,
            flow_pattern="single-phase liquid",
            v_sg=0.0,
            v_sl=vsl_pure,
            v_m=vm_pure,
            v_gas_actual=0.0,
            v_liquid_actual=vsl_pure,
            slip_ratio=0.0,
            rho_no_slip=rho_l,
            rho_holdup=rho_l,
            mu_no_slip=mu_l,
            no_slip_liquid_fraction=1.0,
            f_no_slip=f_l,
            friction_multiplier=1.0,
            f_two_phase=f_l,
            froude_number=vsl_pure / math.sqrt(g * d),
            reynolds_no_slip=Re_l,
            liquid_velocity_number=0.0,
            iterations=1,
            converged=True,
        )
    if x >= 1.0 - 1e-12:
        # 纯气相
        A_pure = math.pi * d * d / 4.0
        vsg_pure = m_total / (rho_g * A_pure)
        vm_pure = vsg_pure
        Re_g = rho_g * vsg_pure * d / mu_g if mu_g > 0 else 1e6
        f_g = _colebrook(Re_g, eps / d)
        dp_dz_f = f_g * rho_g * vsg_pure ** 2 / (2.0 * d)
        dp_dz_g = rho_g * g * math.sin(theta)
        dp_dz_total = dp_dz_f + dp_dz_g
        return MukherjeeBrillResult(
            dp_dL_friction=dp_dz_f,
            dp_dL_gravity=dp_dz_g,
            dp_dL_total=dp_dz_total,
            liquid_holdup_inclined=0.0,
            gas_void_fraction=1.0,
            gas_mass_fraction_inventory=1.0,
            gas_mass_fraction_flow=x,
            flow_pattern="single-phase gas",
            v_sg=vsg_pure,
            v_sl=0.0,
            v_m=vm_pure,
            v_gas_actual=vsg_pure,
            v_liquid_actual=0.0,
            slip_ratio=1e6,
            rho_no_slip=rho_g,
            rho_holdup=rho_g,
            mu_no_slip=mu_g,
            no_slip_liquid_fraction=0.0,
            f_no_slip=f_g,
            friction_multiplier=1.0,
            f_two_phase=f_g,
            froude_number=vsg_pure / math.sqrt(g * d),
            reynolds_no_slip=Re_g,
            liquid_velocity_number=0.0,
            iterations=1,
            converged=True,
        )

    # --- 2. 计算表观速度 ---
    A = math.pi * d * d / 4.0
    m_gas = x * m_total
    m_liq = (1.0 - x) * m_total
    vsg = m_gas / (rho_g * A)
    vsl = m_liq / (rho_l * A)
    vm = vsg + vsl

    # --- 3. 无量纲数 ---
    Ngv, Nlv, Nl = _dimensionless_numbers(vsg, vsl, rho_l, mu_l, sigma)

    # --- 4. 流型预测 ---
    pattern = _predict_flow_pattern(Ngv, Nlv, Nl, theta)

    # --- 5. 无滑脱液相分率（用于 Eq. (14) 与环雾流 Eq. Table 2）---
    lam = vsl / (vsl + vsg) if (vsl + vsg) > 0 else 0.0

    # --- 6. 持液率 ---
    HL = _calc_holdup(Ngv, Nlv, Nl, theta, pattern)
    if not 0.0 < HL < 1.0:
        raise ValueError(
            "Mukherjee-Brill Eq. (10) 给出的持液率超出 (0, 1)；"
            "该工况超出相关式适用范围，未执行非论文规定的截断或无滑脱下限修正。"
        )
    Hg = 1.0 - HL

    # --- 7. 混合物密度 ---
    rho_holdup = HL * rho_l + Hg * rho_g
    rho_ns = lam * rho_l + (1.0 - lam) * rho_g

    # --- 8. 无滑脱黏度 ---
    mu_ns = lam * mu_l + (1.0 - lam) * mu_g

    # --- 9. 无滑脱摩擦因子 ---
    Re_ns = rho_ns * vm * d / mu_ns if mu_ns > 0 else 1e6
    rr = eps / d
    f_ns = _colebrook(Re_ns, rr)

    # --- 10. 摩擦压降梯度 ---
    friction_multiplier = 1.0
    f_two_phase = f_ns

    if pattern in ("bubble", "slug"):
        # 直接用无滑脱摩擦因子
        dp_dz_f = f_ns * rho_ns * vm ** 2 / (2.0 * d)

    elif pattern == "annular-mist":
        # 论文 Table 2: H_R = lambda_L / H_L（不是 H_L / lambda_L）
        HR = lam / HL
        fR = _interp_fR(HR)
        f_two_phase = fR * f_ns
        friction_multiplier = fR
        dp_dz_f = f_two_phase * rho_ns * vm ** 2 / (2.0 * d)

    elif pattern == "stratified":
        # 两流体动量模型。
        # 这里采用论文 Eq.(17) 的合并形式，界面剪切在两相方程相加后被消去。
        _, A_l, A_g, S_l, S_g, S_i = _stratified_geometry(HL, d)

        # 各相实际速度
        v_l_actual_local = vsl / HL if HL > 0 else 0.0
        v_g_actual_local = vsg / Hg if Hg > 0 else 0.0

        # 水力直径
        D_hyd_l = 4.0 * A_l / S_l if S_l > 0 else d
        D_hyd_g = 4.0 * A_g / (S_g + S_i) if (S_g + S_i) > 0 else d

        # 雷诺数
        Re_l_strat = rho_l * abs(v_l_actual_local) * D_hyd_l / mu_l if mu_l > 0 else 1e6
        Re_g_strat = rho_g * abs(v_g_actual_local) * D_hyd_g / mu_g if mu_g > 0 else 1e6

        # 摩擦因子
        f_l_strat = _colebrook(Re_l_strat, eps / D_hyd_l)
        f_g_strat = _colebrook(Re_g_strat, eps / D_hyd_g)

        # Eq. (26)--(27) 写成 T_w = f_F * gamma * v² / (2g)，其中 f_F
        # 是 Fanning 因子。本文其余 Eq. (2)/(14) 与 Colebrook 均采用
        # Darcy/Moody 因子 f_D=4*f_F，因此必须使用 T_w=f_D*rho*v²/8。
        # 这样在 HL -> 1 的单相极限中 Eq. (17) 才会退化为 Eq. (14)。
        tau_wl = f_l_strat * rho_l * v_l_actual_local ** 2 / 8.0
        tau_wg = f_g_strat * rho_g * v_g_actual_local ** 2 / 8.0

        # 摩擦压降: (τ_wl * S_l + τ_wg * S_g) / A
        dp_dz_f = (tau_wl * S_l + tau_wg * S_g) / A

        f_two_phase = f_ns  # 保留供输出参考
        friction_multiplier = dp_dz_f / (f_ns * rho_ns * vm ** 2 / (2.0 * d)) if f_ns > 0 else 1.0

    else:
        dp_dz_f = f_ns * rho_ns * vm ** 2 / (2.0 * d)

    # --- 11. 重力压降梯度 ---
    dp_dz_g = rho_holdup * g * math.sin(theta)

    # --- 12. 加速压降梯度 ---
    # 除分层流外均考虑加速项
    # dp/dz_total = (dp/dz_f + dp/dz_g) / (1 - Ek)
    # 其中 Ek = rho_holdup * vm * vsg / P_avg
    if pattern == "stratified":
        dp_dz_total = dp_dz_f + dp_dz_g
    else:
        P_avg = inp.pressure_avg
        if P_avg <= 0:
            raise ValueError("非分层流按 MB 加速项计算时，平均压力必须大于零。")
        Ek = rho_holdup * vm * vsg / P_avg
        if not 0.0 <= Ek < 1.0:
            raise ValueError(
                f"MB 加速项的 Ek={Ek:.6g} 不在 [0, 1) 内；"
                "稳态机械能方程在该压力条件下无有限解。"
            )
        dp_dz_total = (dp_dz_f + dp_dz_g) / (1.0 - Ek)

    # --- 13. 库存气相质量分数 ---
    mass_total_holdup = HL * rho_l + Hg * rho_g
    if mass_total_holdup > 0:
        gas_mass_inv = (Hg * rho_g) / mass_total_holdup
    else:
        gas_mass_inv = 0.0

    # --- 14. 实际速度 ---
    v_gas_actual = vsg / Hg if Hg > 0 else 0.0
    v_liq_actual = vsl / HL if HL > 0 else 0.0
    slip_ratio = v_gas_actual / v_liq_actual if v_liq_actual > 0 else 1e6

    # --- 15. Froude 数 (常规定义) ---
    Fr = vm / math.sqrt(g * d)

    # --- 16. 结果 ---
    return MukherjeeBrillResult(
        dp_dL_friction=dp_dz_f,
        dp_dL_gravity=dp_dz_g,
        dp_dL_total=dp_dz_total,
        liquid_holdup_inclined=HL,
        gas_void_fraction=Hg,
        gas_mass_fraction_inventory=gas_mass_inv,
        gas_mass_fraction_flow=x,
        flow_pattern=pattern,
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
        friction_multiplier=friction_multiplier,
        f_two_phase=f_two_phase,
        froude_number=Fr,
        reynolds_no_slip=Re_ns,
        liquid_velocity_number=Nlv,
        iterations=1,
        converged=True,
    )
