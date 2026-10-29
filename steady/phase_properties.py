from __future__ import annotations

def clamp(value: float, lower: float, upper: float) -> float:
    """
    把 value 限制在 [lower, upper] 范围内。
    """
    return max(lower, min(upper, value))


def calc_liquid_density_from_total_density(
    rho_total: float,
    rho_gas: float,
    gas_mass_fraction: float,
) -> float:
    """
    由总体系密度、气相密度和气相质量分数反算液相密度。

    输入：
        rho_total: kg/m3，总体系密度
        rho_gas: kg/m3，气相密度
        gas_mass_fraction: 0~1，气相质量分数，通常来自 PVT 的 Weight% Vap

    返回：
        rho_liquid: kg/m3，液相密度
    """

    x = gas_mass_fraction

    if rho_total <= 0:
        raise ValueError("rho_total 必须大于 0。")

    if rho_gas <= 0:
        raise ValueError("rho_gas 必须大于 0。")

    if not 0.0 <= x <= 1.0:
        raise ValueError("gas_mass_fraction 必须在 [0, 1] 范围内。")

    if x >= 1.0:
        raise ValueError(
            "gas_mass_fraction 接近 1，说明几乎全为气相，无法由混合密度反算液相密度。"
        )

    denominator = 1.0 / rho_total - x / rho_gas

    if denominator <= 0:
        raise ValueError(
            "液相密度反算失败：分母小于等于 0。"
            "请检查 density_tot、density_vap、weightpct_vap 是否物理一致。"
        )

    rho_liquid = (1.0 - x) / denominator

    if rho_liquid <= 0:
        raise ValueError("反算得到的液相密度小于等于 0。")

    return rho_liquid


def calc_inventory_gas_mass_fraction(
    rho_gas: float,
    rho_liquid: float,
    liquid_holdup: float,
) -> float:
    """
    计算管内存量意义上的气相质量分数。

    输入：
        rho_gas: kg/m3，气相密度
        rho_liquid: kg/m3，液相密度
        liquid_holdup: H_L，液相持液率

    返回：
        x_inventory: 0~1，管内存量气相质量分数
    """

    H_L = liquid_holdup

    if rho_gas <= 0:
        raise ValueError("rho_gas 必须大于 0。")

    if rho_liquid <= 0:
        raise ValueError("rho_liquid 必须大于 0。")

    if not 0.0 <= H_L <= 1.0:
        raise ValueError("liquid_holdup 必须在 [0, 1] 范围内。")

    gas_void_fraction = 1.0 - H_L

    gas_mass_per_volume = rho_gas * gas_void_fraction
    liquid_mass_per_volume = rho_liquid * H_L

    total_mass_per_volume = gas_mass_per_volume + liquid_mass_per_volume

    if total_mass_per_volume <= 0:
        raise ValueError("单位管内容积质量小于等于 0。")

    return gas_mass_per_volume / total_mass_per_volume


def calc_holdup_density(
    rho_gas: float,
    rho_liquid: float,
    liquid_holdup: float,
) -> float:
    """
    用 BB 模型的液相持液率计算原位混合密度。

    rho_holdup = rho_l * H_L + rho_g * (1 - H_L)

    该密度主要用于重力压降项。
    """

    H_L = liquid_holdup

    if not 0.0 <= H_L <= 1.0:
        raise ValueError("liquid_holdup 必须在 [0, 1] 范围内。")

    return rho_liquid * H_L + rho_gas * (1.0 - H_L)


def calc_no_slip_liquid_fraction(
    v_sl: float,
    v_sg: float,
) -> float:
    """
    计算无滑脱液含率 lambda_l。

    lambda_l = v_sl / (v_sl + v_sg)

    v_sl: 液相表观速度，m/s
    v_sg: 气相表观速度，m/s
    """

    v_m = v_sl + v_sg

    if v_m <= 0:
        raise ValueError("混合表观速度必须大于 0。")

    return v_sl / v_m


def calc_no_slip_density(
    rho_gas: float,
    rho_liquid: float,
    no_slip_liquid_fraction: float,
) -> float:
    """
    计算无滑脱密度。

    rho_ns = rho_l * lambda_l + rho_g * (1 - lambda_l)

    该密度主要用于摩阻项中的无滑脱雷诺数和动压项。
    """

    lam_l = no_slip_liquid_fraction

    if not 0.0 <= lam_l <= 1.0:
        raise ValueError("no_slip_liquid_fraction 必须在 [0, 1] 范围内。")

    return rho_liquid * lam_l + rho_gas * (1.0 - lam_l)


def calc_no_slip_viscosity(
    mu_gas: float,
    mu_liquid: float,
    no_slip_liquid_fraction: float,
) -> float:
    """
    计算无滑脱混合黏度。

    这里第一版采用线性混合：
        mu_ns = mu_l * lambda_l + mu_g * (1 - lambda_l)

    后续如果要换成其他混合规则，只需要改这个函数。
    """

    lam_l = no_slip_liquid_fraction

    if mu_gas <= 0:
        raise ValueError("mu_gas 必须大于 0。")

    if mu_liquid <= 0:
        raise ValueError("mu_liquid 必须大于 0。")

    if not 0.0 <= lam_l <= 1.0:
        raise ValueError("no_slip_liquid_fraction 必须在 [0, 1] 范围内。")

    return mu_liquid * lam_l + mu_gas * (1.0 - lam_l)


def calc_actual_phase_velocities(
    v_sg: float,
    v_sl: float,
    liquid_holdup: float,
) -> tuple[float, float]:
    """
    根据表观速度和持液率计算气液实际速度。

    气相实际速度：
        v_g = v_sg / alpha_g

    液相实际速度：
        v_l = v_sl / H_L

    其中：
        alpha_g = 1 - H_L
    """

    H_L = liquid_holdup

    if not 0.0 < H_L < 1.0:
        raise ValueError("liquid_holdup 必须在 (0, 1) 范围内，才能计算实际相速度。")

    gas_void_fraction = 1.0 - H_L

    v_gas_actual = v_sg / gas_void_fraction
    v_liquid_actual = v_sl / H_L

    return v_gas_actual, v_liquid_actual


def calc_slip_ratio_from_velocities(
    v_gas_actual: float,
    v_liquid_actual: float,
) -> float:
    """
    计算相间滑移比
    """
    return v_gas_actual / v_liquid_actual