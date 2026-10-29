from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional
from steady.phase_properties import (
    calc_inventory_gas_mass_fraction,
    calc_holdup_density,
    calc_no_slip_density,
    calc_no_slip_viscosity,
    calc_no_slip_liquid_fraction,
    calc_actual_phase_velocities,
    calc_slip_ratio_from_velocities,
    clamp,
)


G = 9.80665


@dataclass
class BeggsBrillInput:

    """
    Beggs-Brill 压降模型输入。

    单位约定：
    total_mass_flow: kg/s，总质量流量
    gas_mass_fraction: 0~1，当前用于过流计算的气相质量分数
    rho_gas: kg/m3
    rho_liquid: kg/m3
    mu_gas: Pa.s
    mu_liquid: Pa.s
    diameter: m，管内径
    roughness: m，绝对粗糙度
    angle_rad: rad，管段倾角，沿流动方向上坡为正，下坡为负
    surface_tension: N/m，气液表面张力
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
    forced_liquid_holdup: Optional[float] = None   # 新增
    pressure_avg_Pa: Optional[float] = None


@dataclass
class BeggsBrillResult:
    """
    Beggs-Brill 计算结果。
    """

    flow_pattern: str

    # 分率
    gas_mass_fraction_flow: float
    gas_mass_fraction_inventory: float
    no_slip_liquid_fraction: float
    liquid_holdup_horizontal: float
    liquid_holdup_inclined: float
    gas_void_fraction: float

    # 速度
    v_sg: float
    v_sl: float
    v_m: float
    v_gas_actual: float
    v_liquid_actual: float
    slip_ratio: float

    # 密度和黏度
    rho_no_slip: float
    rho_holdup: float
    mu_no_slip: float

    # 无量纲数
    froude_number: float
    reynolds_no_slip: float
    liquid_velocity_number: float

    # 摩阻
    f_no_slip: float
    friction_multiplier: float
    f_two_phase: float

    # 压降梯度，Pa/m
    dp_dL_friction: float
    dp_dL_gravity: float
    dp_dL_total: float


def calc_superficial_velocities(
    total_mass_flow: float,
    gas_mass_fraction: float,
    rho_gas: float,
    rho_liquid: float,
    diameter: float,
) -> tuple[float, float, float]:
    """
    计算气液表观速度。

    m_g = x_g * m_total
    m_l = (1 - x_g) * m_total

    v_sg = q_g / A = m_g / (rho_g * A)
    v_sl = q_l / A = m_l / (rho_l * A)
    v_m = v_sg + v_sl
    """

    x_g = gas_mass_fraction

    if total_mass_flow <= 0:
        raise ValueError("total_mass_flow 必须大于 0。")

    if not 0.0 <= x_g <= 1.0:
        raise ValueError("gas_mass_fraction 必须在 [0, 1] 范围内。")

    if rho_gas <= 0 or rho_liquid <= 0:
        raise ValueError("rho_gas 和 rho_liquid 必须大于 0。")

    if diameter <= 0:
        raise ValueError("diameter 必须大于 0。")

    area = math.pi * diameter ** 2 / 4.0

    m_gas = x_g * total_mass_flow
    m_liquid = (1.0 - x_g) * total_mass_flow

    v_sg = m_gas / (rho_gas * area) if m_gas > 0 else 0.0
    v_sl = m_liquid / (rho_liquid * area) if m_liquid > 0 else 0.0
    v_m = v_sg + v_sl

    if v_m <= 0:
        raise ValueError("混合表观速度必须大于 0。")

    return v_sg, v_sl, v_m


def calc_froude_number(v_m: float, diameter: float) -> float:
    """
    Beggs-Brill Froude 数：

        N_Fr = v_m^2 / (g D)
    """

    if diameter <= 0:
        raise ValueError("diameter 必须大于 0。")

    return v_m ** 2 / (G * diameter)


def calc_liquid_velocity_number(
    v_sl: float,
    rho_liquid: float,
    surface_tension: float,
) -> float:
    """
    液体速度数 N_Lv。

    常用 SI 形式：
        N_Lv = v_sl * (rho_l / (g * sigma)) ** 0.25

    其中：
        v_sl: m/s
        rho_l: kg/m3
        sigma: N/m

    这个量主要用于 Beggs-Brill 倾角修正。
    """

    if v_sl < 0:
        raise ValueError("v_sl 不能小于 0。")

    if rho_liquid <= 0:
        raise ValueError("rho_liquid 必须大于 0。")

    if surface_tension <= 0:
        raise ValueError("surface_tension 必须大于 0。")

    return v_sl * (rho_liquid / (G * surface_tension)) ** 0.25


def calc_flow_pattern_boundaries(lambda_l: float) -> tuple[float, float, float, float]:
    """
    Beggs-Brill 水平流型边界。

    L1 = 316 * lambda_l^0.302
    L2 = 0.0009252 * lambda_l^-2.4684
    L3 = 0.1 * lambda_l^-1.4516
    L4 = 0.5 * lambda_l^-6.738

    lambda_l 是无滑脱液含率。
    """

    lam = clamp(lambda_l, 1.0e-8, 0.999999)

    L1 = 316.0 * lam ** 0.302
    L2 = 0.0009252 * lam ** (-2.4684)
    L3 = 0.1 * lam ** (-1.4516)
    L4 = 0.5 * lam ** (-6.738)

    return L1, L2, L3, L4


def determine_flow_pattern(lambda_l: float, froude_number: float) -> str:
    """
    Beggs-Brill 水平流型判断。

    返回：
        "segregated"
        "transition"
        "intermittent"
        "distributed"
    """

    lam = clamp(lambda_l, 1.0e-8, 0.999999)
    N_Fr = froude_number

    L1, L2, L3, L4 = calc_flow_pattern_boundaries(lam)

    # 分离流
    if lam < 0.01 and N_Fr < L1:
        return "segregated"

    if lam >= 0.01 and N_Fr < L2:
        return "segregated"

    # 过渡流
    if lam >= 0.01 and L2 <= N_Fr <= L3:
        return "transition"

    # 间歇流
    if 0.01 <= lam < 0.4 and L3 < N_Fr <= L1:
        return "intermittent"

    if lam >= 0.4 and L3 < N_Fr <= L4:
        return "intermittent"

    # 分散流
    return "distributed"


def calc_horizontal_liquid_holdup_for_pattern(
    lambda_l: float,
    froude_number: float,
    pattern: str,
) -> float:
    """
    计算水平管液相持液率 H_L(0)。

    Beggs-Brill 形式：
        H_L(0) = a * lambda_l^b / N_Fr^c

    不同流型采用不同 a, b, c。
    """

    lam = clamp(lambda_l, 1.0e-8, 0.999999)
    N_Fr = max(froude_number, 1.0e-12)

    if pattern == "segregated":
        a, b, c = 0.98, 0.4846, 0.0868
    elif pattern == "intermittent":
        a, b, c = 0.845, 0.5351, 0.0173
    elif pattern == "distributed":
        a, b, c = 1.065, 0.5824, 0.0609
    else:
        raise ValueError(f"该函数不直接处理 pattern={pattern!r}")

    H0 = a * lam ** b / N_Fr ** c

    # BB 中水平持液率不能小于无滑脱液含率
    H0 = max(H0, lam)

    return clamp(H0, 1.0e-8, 0.999999)


def calc_horizontal_liquid_holdup(
    lambda_l: float,
    froude_number: float,
    pattern: str,
) -> float:
    """
    计算水平液相持液率。

    对 transition 流型，采用 segregated 与 intermittent 之间的线性过渡。
    """

    lam = clamp(lambda_l, 1.0e-8, 0.999999)
    N_Fr = froude_number

    if pattern != "transition":
        return calc_horizontal_liquid_holdup_for_pattern(lam, N_Fr, pattern)

    L1, L2, L3, L4 = calc_flow_pattern_boundaries(lam)

    H_seg = calc_horizontal_liquid_holdup_for_pattern(lam, N_Fr, "segregated")
    H_int = calc_horizontal_liquid_holdup_for_pattern(lam, N_Fr, "intermittent")

    # 过渡权重：
    # N_Fr = L2 时接近 segregated
    # N_Fr = L3 时接近 intermittent
    if math.isclose(L2, L3):
        weight_seg = 0.5
    else:
        weight_seg = (L3 - N_Fr) / (L3 - L2)

    weight_seg = clamp(weight_seg, 0.0, 1.0)

    H0 = weight_seg * H_seg + (1.0 - weight_seg) * H_int

    return clamp(max(H0, lam), 1.0e-8, 0.999999)


def calc_inclination_C(
    lambda_l: float,
    froude_number: float,
    liquid_velocity_number: float,
    pattern: str,
    uphill: bool,
) -> float:
    """
    Beggs-Brill 倾角修正中的 C。

    C = (1 - lambda_l) * ln(
            d * lambda_l^e * N_Lv^f * N_Fr^g
        )

    不同流型、上坡/下坡有不同系数。

    注意：
    若 pattern 为 distributed，上坡 C 通常取 0。
    """

    lam = clamp(lambda_l, 1.0e-8, 0.999999)
    N_Fr = max(froude_number, 1.0e-12)
    N_Lv = max(liquid_velocity_number, 1.0e-12)

    if uphill:
        if pattern == "segregated":
            d, e, f, g = 0.011, -3.768, 3.539, -1.614
        elif pattern in {"intermittent", "transition"}:
            d, e, f, g = 2.96, 0.305, -0.4473, 0.0978
        elif pattern == "distributed":
            return 0.0
        else:
            raise ValueError(f"未知流型: {pattern}")
    else:
        # 下坡时，BB 对所有流型使用同一组修正系数
        d, e, f, g = 4.70, -0.3692, 0.1244, -0.5056

    argument = d * lam ** e * N_Lv ** f * N_Fr ** g

    if argument <= 0:
        return 0.0

    C = (1.0 - lam) * math.log(argument)

    # 工程计算中通常不允许 C 使修正异常发散
    return max(C, 0.0)


def calc_inclination_correction_factor(
    lambda_l: float,
    froude_number: float,
    liquid_velocity_number: float,
    angle_rad: float,
    pattern: str,
) -> float:
    """
    倾角修正因子 psi。

    H_L(theta) = H_L(0) * psi

    psi = 1 + C * [
        sin(1.8 theta) - 1/3 * sin^3(1.8 theta)
    ]

    theta 使用弧度。
    """

    if abs(angle_rad) < 1.0e-12:
        return 1.0

    uphill = angle_rad > 0.0

    C = calc_inclination_C(
        lambda_l=lambda_l,
        froude_number=froude_number,
        liquid_velocity_number=liquid_velocity_number,
        pattern=pattern,
        uphill=uphill,
    )

    s = math.sin(1.8 * angle_rad)

    psi = 1.0 + C * (s - (s ** 3) / 3.0)

    # 防止数值异常
    return max(psi, 0.0)


def calc_inclined_liquid_holdup(
    H0: float,
    lambda_l: float,
    froude_number: float,
    liquid_velocity_number: float,
    angle_rad: float,
    pattern: str,
) -> float:
    """
    计算倾斜管液相持液率 H_L(theta)。
    """

    psi = calc_inclination_correction_factor(
        lambda_l=lambda_l,
        froude_number=froude_number,
        liquid_velocity_number=liquid_velocity_number,
        angle_rad=angle_rad,
        pattern=pattern,
    )

    H = H0 * psi

    # 液相持液率不应小于无滑脱液含率，也不能超过 1
    H = max(H, lambda_l)

    return clamp(H, 1.0e-8, 0.999999)


def calc_friction_factor_haaland(
    reynolds: float,
    roughness: float,
    diameter: float,
) -> float:
    """
    用 Haaland 显式公式计算 Darcy 摩擦因子。

    层流：
        f = 64 / Re

    紊流：
        1/sqrt(f) = -1.8 log10[
            (epsilon / 3.7D)^1.11 + 6.9/Re
        ]

    返回 Darcy friction factor。
    """

    Re = reynolds

    if Re <= 0:
        raise ValueError("Reynolds 数必须大于 0。")

    if diameter <= 0:
        raise ValueError("diameter 必须大于 0。")

    if Re < 2300.0:
        return 64.0 / Re

    rel_roughness = max(roughness / diameter, 0.0)

    inside = (rel_roughness / 3.7) ** 1.11 + 6.9 / Re

    if inside <= 0:
        raise ValueError("Haaland 公式对数项异常。")

    inv_sqrt_f = -1.8 * math.log10(inside)

    return 1.0 / inv_sqrt_f ** 2


def calc_two_phase_friction_multiplier(
    lambda_l: float,
    liquid_holdup: float,
) -> float:
    """
    Beggs-Brill 两相摩擦因子修正。

    f_tp / f_ns = exp(S)

    y = lambda_l / H_L^2

    当 1 < y < 1.2:
        S = ln(2.2y - 1.2)

    否则：
        S = ln(y) / [
            -0.0523 + 3.182 ln(y)
            - 0.8725 ln(y)^2
            + 0.01853 ln(y)^4
        ]
    """

    lam = clamp(lambda_l, 1.0e-12, 0.999999)
    H = clamp(liquid_holdup, 1.0e-12, 0.999999)

    y = lam / (H ** 2)

    if y <= 0:
        raise ValueError("Beggs-Brill 摩擦修正中的 y 必须大于 0。")

    if 1.0 < y < 1.2:
        S = math.log(2.2 * y - 1.2)
    else:
        ln_y = math.log(y)
        denominator = (
            -0.0523
            + 3.182 * ln_y
            - 0.8725 * ln_y ** 2
            + 0.01853 * ln_y ** 4
        )

        if abs(denominator) < 1.0e-12:
            S = 0.0
        else:
            S = ln_y / denominator

    return math.exp(S)


def beggs_brill_pressure_gradient(inp: BeggsBrillInput) -> BeggsBrillResult:
    """
    Beggs-Brill 压降梯度计算主函数。

    返回的 dp_dL 单位为 Pa/m。
    """

    # 1. 表观速度
    v_sg, v_sl, v_m = calc_superficial_velocities(
        total_mass_flow=inp.total_mass_flow,
        gas_mass_fraction=inp.gas_mass_fraction,
        rho_gas=inp.rho_gas,
        rho_liquid=inp.rho_liquid,
        diameter=inp.diameter,
    )

    # 2. 无滑脱液含率
    lambda_l = calc_no_slip_liquid_fraction(v_sl=v_sl, v_sg=v_sg)

    # 3. Froude 数
    N_Fr = calc_froude_number(v_m=v_m, diameter=inp.diameter)

    # 4. 液体速度数
    N_Lv = calc_liquid_velocity_number(
        v_sl=v_sl,
        rho_liquid=inp.rho_liquid,
        surface_tension=inp.surface_tension,
    )

    # 5. 流型判断
    pattern = determine_flow_pattern(
        lambda_l=lambda_l,
        froude_number=N_Fr,
    )

    # 6. 水平持液率
    H0 = calc_horizontal_liquid_holdup(
        lambda_l=lambda_l,
        froude_number=N_Fr,
        pattern=pattern,
    )

    # 7. 倾角修正持液率
    # 在计算完 H_L（倾角修正持液率）之后：
    if inp.forced_liquid_holdup is not None:
        H_L = clamp(inp.forced_liquid_holdup, 1.0e-8, 0.999999)
    else:
        H_L = calc_inclined_liquid_holdup(
            H0=H0,
            lambda_l=lambda_l,
            froude_number=N_Fr,
            liquid_velocity_number=N_Lv,
            angle_rad=inp.angle_rad,
            pattern=pattern,
        )

    gas_void_fraction = 1.0 - H_L

    # 8. 管内存量气相质量分数
    x_inventory = calc_inventory_gas_mass_fraction(
        rho_gas=inp.rho_gas,
        rho_liquid=inp.rho_liquid,
        liquid_holdup=H_L,
    )

    # 9. 密度、黏度
    rho_holdup = calc_holdup_density(
        rho_gas=inp.rho_gas,
        rho_liquid=inp.rho_liquid,
        liquid_holdup=H_L,
    )

    rho_ns = calc_no_slip_density(
        rho_gas=inp.rho_gas,
        rho_liquid=inp.rho_liquid,
        no_slip_liquid_fraction=lambda_l,
    )

    mu_ns = calc_no_slip_viscosity(
        mu_gas=inp.mu_gas,
        mu_liquid=inp.mu_liquid,
        no_slip_liquid_fraction=lambda_l,
    )

    # 10. 实际相速度和滑移比
    v_gas_actual, v_liquid_actual = calc_actual_phase_velocities(
        v_sg=v_sg,
        v_sl=v_sl,
        liquid_holdup=H_L,
    )

    slip_ratio = calc_slip_ratio_from_velocities(
        v_gas_actual=v_gas_actual,
        v_liquid_actual=v_liquid_actual,
    )

    # 11. 无滑脱 Reynolds 数
    Re_ns = rho_ns * v_m * inp.diameter / mu_ns

    # 12. 无滑脱摩擦因子
    f_ns = calc_friction_factor_haaland(
        reynolds=Re_ns,
        roughness=inp.roughness,
        diameter=inp.diameter,
    )

    # 13. 两相摩擦修正
    multiplier = calc_two_phase_friction_multiplier(
        lambda_l=lambda_l,
        liquid_holdup=H_L,
    )

    f_tp = f_ns * multiplier

    # 14. 摩擦压降梯度
    # 使用 Darcy friction factor:
    # dp/dL = f * rho * v^2 / (2D)
    dp_dL_friction = f_tp * rho_ns * v_m ** 2 / (2.0 * inp.diameter)

    # 15. 重力压降梯度
    # 上坡 angle_rad > 0，重力项为正；
    # 下坡 angle_rad < 0，重力项可能为负。
    dp_dL_gravity = rho_holdup * G * math.sin(inp.angle_rad)

    # 16. 总压降梯度。Beggs--Brill 原式包含可压缩气相的加速度项：
    #   dP/dL = (friction + gravity) / (1 - E_k)
    #   E_k = rho_ns * v_m * v_sg / P
    # 使用微元平均绝压；未提供时保留近似不可压缩极限 E_k=0，便于旧接口调用。
    acceleration_factor = 0.0
    if inp.pressure_avg_Pa is not None:
        if inp.pressure_avg_Pa <= 0.0:
            raise ValueError("Beggs-Brill 加速度项的平均压力必须大于 0。")
        acceleration_factor = rho_ns * v_m * v_sg / inp.pressure_avg_Pa
        if acceleration_factor >= 1.0:
            raise ArithmeticError(
                "Beggs-Brill 加速度项 E_k >= 1，微元处于不可用的临界/阻塞状态。"
            )
    dp_dL_total = (dp_dL_friction + dp_dL_gravity) / (1.0 - acceleration_factor)

    return BeggsBrillResult(
        flow_pattern=pattern,
        gas_mass_fraction_flow=inp.gas_mass_fraction,
        gas_mass_fraction_inventory=x_inventory,
        no_slip_liquid_fraction=lambda_l,
        liquid_holdup_horizontal=H0,
        liquid_holdup_inclined=H_L,
        gas_void_fraction=gas_void_fraction,
        v_sg=v_sg,
        v_sl=v_sl,
        v_m=v_m,
        v_gas_actual=v_gas_actual,
        v_liquid_actual=v_liquid_actual,
        slip_ratio=slip_ratio,
        rho_no_slip=rho_ns,
        rho_holdup=rho_holdup,
        mu_no_slip=mu_ns,
        froude_number=N_Fr,
        reynolds_no_slip=Re_ns,
        liquid_velocity_number=N_Lv,
        f_no_slip=f_ns,
        friction_multiplier=multiplier,
        f_two_phase=f_tp,
        dp_dL_friction=dp_dL_friction,
        dp_dL_gravity=dp_dL_gravity,
        dp_dL_total=dp_dL_total,
    )
