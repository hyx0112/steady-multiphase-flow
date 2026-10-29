# steady/types.py — 共享数据类型
from __future__ import annotations

from dataclasses import dataclass, asdict
import math


@dataclass
class PipeSection:
    """大管段参数"""
    section_no: int
    outlet_node: int

    section_length_m: float
    elevation_change_m: float
    n_subsegments: int
    sub_length_m: float

    roughness_m: float
    overall_heat_transfer: float
    mass_flow_kg_s: float
    diameter_m: float
    env_temp_K: float


@dataclass
class PipelineCommonSettings:
    """全线通用计算设置"""
    surface_tension_N_m: float = 0.02

    micro_max_iter: int = 30
    pressure_tol_bara: float = 1.0e-5
    temperature_tol_K: float = 1.0e-4
    gas_fraction_tol: float = 1.0e-5
    relaxation: float = 0.5

    allow_extrapolation: bool = False

    two_phase_model: str = "beggs_brill"

    df_sigma_water: float = 0.072
    df_rho_water: float = 1000.0
    df_rho_air: float = 1.29
    df_mu_water: float = 0.001

    # Xiao 闭合关系配置；默认值完整保留原 Xiao 行为。
    xiao_single_phase_limit_closure: str = "legacy"
    xiao_flow_pattern_closure: str = "xiao"
    xiao_stratified_interfacial_closure: str = "xiao"
    xiao_intermittent_translation_closure: str = "xiao"
    xiao_intermittent_slug_holdup_closure: str = "gregory1978"
    xiao_intermittent_film_interfacial_closure: str = "constant"
    xiao_annular_entrainment_closure: str = "oliemans"
    xiao_annular_interfacial_closure: str = "xiao"
    xiao_dispersed_bubble_closure: str = "xiao"


@dataclass
class MicroSegmentResult:
    """单个小微元管段计算结果（正算/倒算通用输出）"""

    # 编号
    section_no: int
    micro_no: int

    # 水平距离坐标
    x_inlet_m: float
    x_outlet_m: float
    x_mid_m: float

    # 高程
    elevation_inlet_m: float
    elevation_outlet_m: float
    elevation_mid_m: float

    # 几何
    length_m: float
    horizontal_length_m: float
    elevation_change_m: float
    angle_deg: float

    # 压力
    pressure_inlet_bara: float
    pressure_outlet_bara: float
    pressure_avg_bara: float
    pressure_drop_Pa: float
    pressure_drop_bara: float

    # 温度
    temperature_inlet_K: float
    temperature_outlet_K: float
    temperature_avg_K: float
    temperature_inlet_C: float
    temperature_outlet_C: float
    temperature_avg_C: float
    temperature_change_K: float

    # 密度
    rho_total_kg_m3: float
    rho_gas_kg_m3: float
    rho_liquid_kg_m3: float
    rho_no_slip_kg_m3: float
    rho_holdup_kg_m3: float

    # 黏度
    mu_total_Pa_s: float
    mu_gas_Pa_s: float
    mu_liquid_Pa_s: float
    mu_no_slip_Pa_s: float

    # 热容
    cp_gas_J_kgK: float
    cp_total_J_kgK: float
    cp_liquid_J_kgK: float
    cp_mixture_J_kgK: float

    # JT 系数
    jt_coeff_C_bara: float
    jt_coeff_K_Pa: float

    # 分子量
    mol_weight_gas_g_mol: float
    mol_weight_total_g_mol: float

    # 相分率
    gas_mass_fraction_pvt: float
    gas_mass_fraction_flow: float
    gas_mass_fraction_inventory: float
    mole_fraction_gas_pvt: float
    volume_fraction_gas_pvt: float

    liquid_mass_fraction_inventory: float
    liquid_holdup: float
    gas_void_fraction: float
    no_slip_liquid_fraction: float

    # 速度
    v_sg_m_s: float
    v_sl_m_s: float
    v_m_m_s: float
    v_gas_actual_m_s: float
    v_liquid_actual_m_s: float
    slip_ratio: float

    # 无量纲数
    froude_number: float
    reynolds_no_slip: float
    liquid_velocity_number: float

    # 流型
    flow_pattern: str

    # 摩擦因子
    f_no_slip: float
    friction_multiplier: float
    f_two_phase: float

    # 压降梯度
    dp_dL_friction_Pa_m: float
    dp_dL_gravity_Pa_m: float
    dp_dL_total_Pa_m: float

    # 热力模型中间参数
    thermal_a: float
    thermal_b: float
    thermal_term1: float
    thermal_term2: float
    thermal_term3: float
    hydraulic_gradient: float

    # 迭代状态
    iterations: int
    converged: bool


def micro_result_to_dict(result: MicroSegmentResult) -> dict:
    """把 MicroSegmentResult 转成字典，方便 pandas 输出 CSV/Excel。"""
    return asdict(result)
