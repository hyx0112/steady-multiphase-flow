from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class ThermalInput:
    """
    单微元段温度计算输入。

    单位约定：
        length: m
        diameter: m
        overall_heat_transfer: W/(m2.K)
        mass_flow: kg/s

        env_temp_K: K
        outlet_temp_K: K，倒算时使用
        inlet_temp_K: K，正算时使用

        pressure_drop_Pa:
            沿流动方向的压力降，定义为 P_in - P_out。
            一般摩阻主导时为正。
            强下坡时可能为负。

        gas_mass_fraction_inventory:
            管内存量意义上的气相质量分数，0~1。

        mixture_cp: J/(kg.K)
        gas_cp: J/(kg.K)
        jt_coeff_K_per_Pa: K/Pa

        rho_liquid: kg/m3
        rho_gas: kg/m3
        liquid_holdup: 0~1，Beggs-Brill 持液率
    """

    length: float
    diameter: float
    overall_heat_transfer: float
    mass_flow: float

    env_temp_K: float

    # 倒算时用 outlet_temp_K，正算时用 inlet_temp_K
    outlet_temp_K: float | None
    inlet_temp_K: float | None

    pressure_drop_Pa: float

    gas_mass_fraction_inventory: float

    mixture_cp: float
    gas_cp: float
    jt_coeff_K_per_Pa: float

    rho_liquid: float
    rho_gas: float
    liquid_holdup: float

    @classmethod
    def from_pvt_raw_data_forward(
        cls,
        length: float,
        diameter: float,
        overall_heat_transfer: float,
        mass_flow: float,
        env_temp_K: float,
        inlet_temp_K: float,
        pressure_drop_Pa: float,
        gas_mass_fraction_inventory: float,
        mixture_cp_J: float,
        gas_cp_J: float,
        jt_coeff_K_per_Pa: float,
        rho_liquid: float,
        rho_gas: float,
        liquid_holdup: float,
    ) -> "ThermalInput":
        """
        正向温度计算输入。

        已知入口温度，计算出口温度。
        """

        return cls(
            length=length,
            diameter=diameter,
            overall_heat_transfer=overall_heat_transfer,
            mass_flow=mass_flow,
            env_temp_K=env_temp_K,
            outlet_temp_K=None,
            inlet_temp_K=inlet_temp_K,
            pressure_drop_Pa=pressure_drop_Pa,
            gas_mass_fraction_inventory=gas_mass_fraction_inventory,
            mixture_cp=mixture_cp_J,
            gas_cp=gas_cp_J,
            jt_coeff_K_per_Pa=jt_coeff_K_per_Pa,
            rho_liquid=rho_liquid,
            rho_gas=rho_gas,
            liquid_holdup=liquid_holdup,
        )


@dataclass
class ThermalResult:
    """
    单微元段温度计算结果。
    """

    a: float
    b: float

    hydraulic_gradient: float
    rho_mix_holdup: float

    term1: float
    term2: float
    term3: float

    inlet_temp_K: float
    outlet_temp_K: float

    temperature_change_K: float
    pressure_drop_Pa: float


def calc_mixture_cp_from_inventory_fraction(
    gas_mass_fraction_inventory: float,
    gas_cp: float,
    liquid_cp: float,
) -> float:
    """
    按管内存量质量分数计算混合物质量热容。

    Cp_mix = x * Cp_g + (1 - x) * Cp_l

    x 是管内存量意义上的气相质量分数。
    """

    x = gas_mass_fraction_inventory

    if not 0.0 <= x <= 1.0:
        raise ValueError("gas_mass_fraction_inventory 必须在 [0, 1] 范围内。")

    if gas_cp <= 0:
        raise ValueError("gas_cp 必须大于 0。")

    if liquid_cp <= 0:
        raise ValueError("liquid_cp 必须大于 0。")

    return x * gas_cp + (1.0 - x) * liquid_cp


def calc_holdup_mixture_density(
    rho_liquid: float,
    rho_gas: float,
    liquid_holdup: float,
) -> float:
    """
    根据 Beggs-Brill 持液率计算原位混合密度。

    rho_mix = rho_l * H_L + rho_g * (1 - H_L)
    """

    H_L = liquid_holdup

    if not 0.0 <= H_L <= 1.0:
        raise ValueError("liquid_holdup 必须在 [0, 1] 范围内。")

    if rho_liquid <= 0:
        raise ValueError("rho_liquid 必须大于 0。")

    if rho_gas <= 0:
        raise ValueError("rho_gas 必须大于 0。")

    return rho_liquid * H_L + rho_gas * (1.0 - H_L)


def _validate_common(inp: ThermalInput) -> None:
    """
    热力模型公共输入检查。
    """

    if inp.length <= 0:
        raise ValueError("length 必须大于 0。")

    if inp.diameter <= 0:
        raise ValueError("diameter 必须大于 0。")

    if inp.overall_heat_transfer < 0:
        raise ValueError("overall_heat_transfer 不能小于 0。")

    if inp.mass_flow <= 0:
        raise ValueError("mass_flow 必须大于 0。")

    if inp.env_temp_K <= 0:
        raise ValueError("env_temp_K 必须大于 0。")

    if inp.mixture_cp <= 0:
        raise ValueError("mixture_cp 必须大于 0。")

    if inp.gas_cp <= 0:
        raise ValueError("gas_cp 必须大于 0。")

    if not 0.0 <= inp.gas_mass_fraction_inventory <= 1.0:
        raise ValueError("gas_mass_fraction_inventory 必须在 [0, 1] 范围内。")


def _calculate_common_terms(inp: ThermalInput) -> tuple[float, float, float, float, float, float, float]:
    """
    计算热力模型公共中间项。

    返回：
        a
        b
        hydraulic_gradient
        rho_mix_holdup
        term1
        term2
        term3
    """

    _validate_common(inp)

    L = inp.length
    d = inp.diameter
    U = inp.overall_heat_transfer
    M = inp.mass_flow

    dp = inp.pressure_drop_Pa
    x = inp.gas_mass_fraction_inventory

    cp_mix = inp.mixture_cp
    cp_gas = inp.gas_cp
    mu_jt = inp.jt_coeff_K_per_Pa

    rho_mix_holdup = calc_holdup_mixture_density(
        rho_liquid=inp.rho_liquid,
        rho_gas=inp.rho_gas,
        liquid_holdup=inp.liquid_holdup,
    )

    g = 9.81

    hydraulic_gradient = dp / (g * rho_mix_holdup * L)

    if U == 0:
        # 保留旧模型形式，避免除零。
        # 如果后续需要严格无换热极限，可在这里单独推导公式。
        a = 1.0e-12
    else:
        # a = U*pi*d / (M*Cp_mix)
        a = (U * math.pi * d) / (M * cp_mix)

    # b = i*g*(1-x)/(a*Cp_mix)
    b = (hydraulic_gradient * g * (1.0 - x)) / (a * cp_mix)

    term1 = inp.env_temp_K + b

    exp_term = math.exp(-a * L)

    # 用 expm1 提高 aL 很小时的数值稳定性：
    # 1 - exp(-aL) = -expm1(-aL)
    one_minus_exp = -math.expm1(-a * L)

    # term2 = dp/(a*L)*(1-exp(-aL))
    term2 = (dp / (a * L)) * one_minus_exp

    # term3 = x*Cp_g*JT/Cp_mix * term2
    term3 = (x * cp_gas * mu_jt / cp_mix) * term2

    return a, b, hydraulic_gradient, rho_mix_holdup, term1, term2, term3


def calculate_temperature_forward(inp: ThermalInput) -> ThermalResult:
    """
    正算温度。

    已知：
        微元入口温度 T_in

    计算：
        微元出口温度 T_out

    与 calculate_temperature_backward 使用同一组解析式。
    """

    if inp.inlet_temp_K is None:
        raise ValueError("正算温度时 inlet_temp_K 不能为空。")

    if inp.inlet_temp_K <= 0:
        raise ValueError("inlet_temp_K 必须大于 0。")

    L = inp.length
    T_in = inp.inlet_temp_K

    a, b, hydraulic_gradient, rho_mix_holdup, term1, term2, term3 = _calculate_common_terms(inp)

    exp_term = math.exp(-a * L)

    # 倒算公式为：
    # T_in = ((T_out - term1 + term3) / exp_term) + term1
    #
    # 整理得到正算公式：
    # T_out = (T_in - term1) * exp_term + term1 - term3
    T_out = (T_in - term1) * exp_term + term1 - term3

    return ThermalResult(
        a=a,
        b=b,
        hydraulic_gradient=hydraulic_gradient,
        rho_mix_holdup=rho_mix_holdup,
        term1=term1,
        term2=term2,
        term3=term3,
        inlet_temp_K=T_in,
        outlet_temp_K=T_out,
        temperature_change_K=T_in - T_out,
        pressure_drop_Pa=inp.pressure_drop_Pa,
    )