# steady/segment_solver.py — 正向微元求解器（优化版：纯显式单步计算）
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional
import math

from steady.pvt_provider import PVTPropertyProvider
from steady.beggs_brill import (
    BeggsBrillInput,
    beggs_brill_pressure_gradient,
)
from steady.mukherjee_brill import (
    MukherjeeBrillInput,
    mukherjee_brill_pressure_gradient,
)
from steady.drift_flux_model import (
    DriftFluxInput,
    drift_flux_pressure_gradient,
)
from steady.xiao_model import (
    XiaoInput,
    xiao_pressure_gradient,
)
from steady.thermal_model import (
    ThermalInput,
    calculate_temperature_forward,
    calc_mixture_cp_from_inventory_fraction,
)
from steady.steady_types import MicroSegmentResult


@dataclass
class MicroSegmentForwardInput:
    """单个小微元管段正向计算输入"""
    section_no: int
    micro_no: int

    x_inlet_m: float
    x_outlet_m: float
    elevation_inlet_m: float
    elevation_outlet_m: float

    pressure_inlet_bara: float
    temperature_inlet_K: float

    length_m: float
    horizontal_length_m: float
    elevation_change_m: float
    angle_rad: float

    diameter_m: float
    roughness_m: float

    mass_flow_kg_s: float
    overall_heat_transfer: float
    env_temp_K: float

    surface_tension_N_m: float = 0.02

    max_iter: int = 30
    pressure_tol_bara: float = 1.0e-5
    temperature_tol_K: float = 1.0e-4
    gas_fraction_tol: float = 1.0e-5
    relaxation: float = 0.5

    allow_extrapolation: bool = False
    debug: bool = False

    # 两相流模型选择
    two_phase_model: str = "beggs_brill"

    # 漂移流模型参数（从 PipelineCommonSettings 传入）
    df_sigma_water: float = 0.072
    df_rho_water: float = 1000.0
    df_rho_air: float = 1.29
    df_mu_water: float = 0.001

    # Xiao 闭合关系开关（由 PipelineCommonSettings 逐级传入）。
    xiao_single_phase_limit_closure: str = "legacy"
    xiao_flow_pattern_closure: str = "xiao"
    xiao_stratified_interfacial_closure: str = "xiao"
    xiao_intermittent_translation_closure: str = "xiao"
    xiao_intermittent_slug_holdup_closure: str = "gregory1978"
    xiao_intermittent_film_interfacial_closure: str = "constant"
    xiao_annular_entrainment_closure: str = "oliemans"
    xiao_annular_interfacial_closure: str = "xiao"
    xiao_dispersed_bubble_closure: str = "xiao"

    # 强制覆盖参数（用于清管器模拟等特殊场景）
    override_gas_mass_fraction: Optional[float] = None
    override_liquid_holdup: Optional[float] = None


def _validate_micro_forward_input(inp: MicroSegmentForwardInput) -> None:
    """输入参数校验"""
    if inp.length_m <= 0 or inp.horizontal_length_m <= 0:
        raise ValueError("length_m / horizontal_length_m 必须大于 0。")
    if inp.diameter_m <= 0 or inp.roughness_m < 0:
        raise ValueError("diameter_m / roughness_m 无效。")
    if inp.mass_flow_kg_s <= 0:
        raise ValueError("mass_flow_kg_s 必须大于 0。")
    if inp.overall_heat_transfer < 0:
        raise ValueError("overall_heat_transfer 不能为负。")
    if inp.env_temp_K <= 0 or inp.temperature_inlet_K <= 0 or inp.pressure_inlet_bara <= 0:
        raise ValueError("温度/压力必须大于 0 K/bara。")
    if inp.surface_tension_N_m <= 0 or inp.max_iter <= 0:
        raise ValueError("surface_tension/max_iter 无效。")
    if not 0.0 < inp.relaxation <= 1.0:
        raise ValueError("relaxation 必须在 (0,1] 范围内。")


def solve_micro_segment_forward(
    pvt: PVTPropertyProvider,
    inp: MicroSegmentForwardInput,
) -> MicroSegmentResult:
    """
    单个小微元管段正向计算（优化版）。

    以微元平均压力/温度查询预生成 PVT 表，并对出口 P/T 做松弛迭代。
    因此物性、可压缩加速度项与水力/热力结果处于同一状态点；迭代过程
    只做 PVT 表插值，不会触发逐微元闪蒸。
    """
    _validate_micro_forward_input(inp)

    P_in = inp.pressure_inlet_bara
    T_in = inp.temperature_inlet_K
    dx = inp.length_m

    def _hydraulic_at_average_state(P_avg_bara: float, T_avg_K: float):
        props_local = pvt.get_properties(
            P_avg_bara, T_avg_K - 273.15,
            allow_extrapolation=inp.allow_extrapolation,
        )
        gas_frac_local = (
            max(0.0, min(1.0, inp.override_gas_mass_fraction))
            if inp.override_gas_mass_fraction is not None
            else max(0.0, min(1.0, props_local.mass_frac_gas_eq))
        )
        sigma_local = props_local.surface_tension_N_m or inp.surface_tension_N_m
        pressure_avg_Pa = P_avg_bara * 1e5
        if inp.two_phase_model == "mukherjee_brill":
            model_result = mukherjee_brill_pressure_gradient(MukherjeeBrillInput(
                inp.mass_flow_kg_s, gas_frac_local, props_local.rho_gas,
                props_local.rho_liquid, props_local.mu_gas, props_local.mu_liquid,
                inp.diameter_m, inp.roughness_m, inp.angle_rad, sigma_local,
                pressure_avg=pressure_avg_Pa,
            ))
        elif inp.two_phase_model == "drift_flux":
            model_result = drift_flux_pressure_gradient(DriftFluxInput(
                inp.mass_flow_kg_s, gas_frac_local, props_local.rho_gas,
                props_local.rho_liquid, props_local.mu_gas, props_local.mu_liquid,
                inp.diameter_m, inp.roughness_m, inp.angle_rad, sigma_local,
                sigma_water=inp.df_sigma_water, rho_water=inp.df_rho_water,
                rho_air=inp.df_rho_air, mu_water=inp.df_mu_water,
            ))
        elif inp.two_phase_model == "xiao":
            model_result = xiao_pressure_gradient(XiaoInput(
                inp.mass_flow_kg_s, gas_frac_local, props_local.rho_gas,
                props_local.rho_liquid, props_local.mu_gas, props_local.mu_liquid,
                inp.diameter_m, inp.roughness_m, inp.angle_rad, sigma_local,
                pressure_Pa=pressure_avg_Pa,
                single_phase_limit_closure=(
                    inp.xiao_single_phase_limit_closure),
                flow_pattern_closure=inp.xiao_flow_pattern_closure,
                stratified_interfacial_closure=(
                    inp.xiao_stratified_interfacial_closure),
                intermittent_translation_closure=(
                    inp.xiao_intermittent_translation_closure),
                intermittent_slug_holdup_closure=(
                    inp.xiao_intermittent_slug_holdup_closure),
                intermittent_film_interfacial_closure=(
                    inp.xiao_intermittent_film_interfacial_closure),
                annular_entrainment_closure=(
                    inp.xiao_annular_entrainment_closure),
                annular_interfacial_closure=(
                    inp.xiao_annular_interfacial_closure),
                dispersed_bubble_closure=(
                    inp.xiao_dispersed_bubble_closure),
            ))
        else:
            model_result = beggs_brill_pressure_gradient(BeggsBrillInput(
                inp.mass_flow_kg_s, gas_frac_local, props_local.rho_gas,
                props_local.rho_liquid, props_local.mu_gas, props_local.mu_liquid,
                inp.diameter_m, inp.roughness_m, inp.angle_rad, sigma_local,
                forced_liquid_holdup=inp.override_liquid_holdup,
                pressure_avg_Pa=pressure_avg_Pa,
            ))
        return props_local, gas_frac_local, model_result

    P_out = P_in
    T_out = T_in
    converged_micro = False
    thermal = None
    relaxation = inp.relaxation
    previous_pressure_residual: float | None = None
    previous_temperature_residual: float | None = None
    debug_trace: list[tuple] = []
    for iteration in range(1, inp.max_iter + 1):
        P_avg = max(0.1, 0.5 * (P_in + P_out))
        T_avg = 0.5 * (T_in + T_out)
        props, gas_frac, bb = _hydraulic_at_average_state(P_avg, T_avg)
        dp_trial_Pa = bb.dp_dL_total * dx
        P_trial = max(0.1, P_in - dp_trial_Pa / 1e5)
        cp_mix = calc_mixture_cp_from_inventory_fraction(
            gas_mass_fraction_inventory=bb.gas_mass_fraction_inventory,
            gas_cp=props.cp_gas_mass,
            liquid_cp=props.cp_liquid_mass,
        )
        thermal = calculate_temperature_forward(ThermalInput.from_pvt_raw_data_forward(
            length=dx, diameter=inp.diameter_m,
            overall_heat_transfer=inp.overall_heat_transfer,
            mass_flow=inp.mass_flow_kg_s, env_temp_K=inp.env_temp_K,
            inlet_temp_K=T_in, pressure_drop_Pa=dp_trial_Pa,
            gas_mass_fraction_inventory=bb.gas_mass_fraction_inventory,
            mixture_cp_J=cp_mix, gas_cp_J=props.cp_gas_mass,
            jt_coeff_K_per_Pa=props.jt_coeff_gas, rho_liquid=props.rho_liquid,
            rho_gas=props.rho_gas, liquid_holdup=bb.liquid_holdup_inclined,
        ))
        T_trial = max(1.0, thermal.outlet_temp_K)
        pressure_residual = P_trial - P_out
        temperature_residual = T_trial - T_out
        if inp.debug:
            debug_trace.append((
                iteration, P_out, T_out, P_avg, T_avg,
                pressure_residual, temperature_residual, relaxation,
                bb.flow_pattern, bb.liquid_holdup_inclined,
                bb.dp_dL_total, getattr(bb, "debug_film_Ef", float("nan")),
                getattr(bb, "debug_film_hL_tilde", float("nan")),
                getattr(bb, "debug_film_root_count", 0),
                bb.reynolds_no_slip,
                getattr(bb, "debug_film_Vf", float("nan")),
                getattr(bb, "debug_dp_slug", float("nan")),
                getattr(bb, "debug_dp_film", float("nan")),
                getattr(bb, "debug_Ls_over_Lu", float("nan")),
            ))
        if (abs(pressure_residual) <= inp.pressure_tol_bara
                and abs(temperature_residual) <= inp.temperature_tol_K):
            P_out, T_out = P_trial, T_trial
            converged_micro = True
            break

        # 在强闪蒸/大压降微元中，P/T 固定点常呈交替过冲。只在检测到
        # 残差换号时降低松弛系数；正常单调收敛路径保持用户给定的系数，
        # 避免无谓地牺牲 BB、MB、Xiao 的计算速度。
        if previous_pressure_residual is not None:
            if (pressure_residual * previous_pressure_residual < 0.0
                    or temperature_residual * previous_temperature_residual < 0.0):
                relaxation = max(0.20, 0.5 * relaxation)
        P_out += relaxation * pressure_residual
        T_out += relaxation * temperature_residual
        previous_pressure_residual = pressure_residual
        previous_temperature_residual = temperature_residual

    if inp.debug and not converged_micro:
        print(
            f"[Micro debug] section={inp.section_no}, micro={inp.micro_no}, "
            f"Pin={P_in:.8g} bara, Tin={T_in:.8g} K, "
            f"angle={math.degrees(inp.angle_rad):.6g} deg"
        )
        print(
            "  iter | Pout(bara) | Tout(K) | Pres(bara) | Tres(K) | "
            "relax | pattern | holdup | dp/dL | film Ef/hL/n | Re_ns | "
            "Vf | dp_slug/dp_film/LsLu"
        )
        for trace_row in debug_trace[-20:]:
            (it, p_out_i, t_out_i, _p_avg_i, _t_avg_i, p_res_i,
             t_res_i, relax_i, pattern_i, holdup_i, dpdl_i,
             film_ef_i, film_h_i, film_roots_i, re_ns_i, vf_i,
             dp_slug_i, dp_film_i, ls_lu_i) = trace_row
            print(
                f"  {it:4d} | {p_out_i:10.6f} | {t_out_i:8.4f} | "
                f"{p_res_i:+11.4e} | {t_res_i:+10.3e} | {relax_i:.3f} | "
                f"{pattern_i} | {holdup_i:.6f} | {dpdl_i:.6g} | "
                f"{film_ef_i:.6g}/{film_h_i:.6g}/{film_roots_i} | {re_ns_i:.8g} | "
                f"{vf_i:.6g} | {dp_slug_i:.6g}/{dp_film_i:.6g}/{ls_lu_i:.6g}"
            )

    # 最后一轮评估所用的物性与模型结果对应于本微元的平均状态；压降则以
    # 收敛后的入口/出口压力守恒值写入结果表。
    dp_Pa = (P_in - P_out) * 1e5

    # === 7. 后处理平均量 ===
    P_avg_final = 0.5 * (P_in + P_out)
    T_avg_final = 0.5 * (T_in + T_out)
    flow_gas_frac = gas_frac if inp.override_gas_mass_fraction is None else gas_frac

    # === 8. 构造结果 ===
    return MicroSegmentResult(
        section_no=inp.section_no,
        micro_no=inp.micro_no,

        x_inlet_m=inp.x_inlet_m,
        x_outlet_m=inp.x_outlet_m,
        x_mid_m=0.5 * (inp.x_inlet_m + inp.x_outlet_m),
        elevation_inlet_m=inp.elevation_inlet_m,
        elevation_outlet_m=inp.elevation_outlet_m,
        elevation_mid_m=0.5 * (inp.elevation_inlet_m + inp.elevation_outlet_m),

        length_m=inp.length_m,
        horizontal_length_m=inp.horizontal_length_m,
        elevation_change_m=inp.elevation_change_m,
        angle_deg=math.degrees(inp.angle_rad),

        pressure_inlet_bara=P_in,
        pressure_outlet_bara=P_out,
        pressure_avg_bara=P_avg_final,
        pressure_drop_Pa=dp_Pa,
        pressure_drop_bara=dp_Pa / 1e5,

        temperature_inlet_K=T_in,
        temperature_outlet_K=T_out,
        temperature_avg_K=T_avg_final,
        temperature_inlet_C=T_in - 273.15,
        temperature_outlet_C=T_out - 273.15,
        temperature_avg_C=T_avg_final - 273.15,
        temperature_change_K=T_in - T_out,

        rho_total_kg_m3=props.rho_tot,
        rho_gas_kg_m3=props.rho_gas,
        rho_liquid_kg_m3=props.rho_liquid,
        rho_no_slip_kg_m3=bb.rho_no_slip,
        rho_holdup_kg_m3=bb.rho_holdup,

        mu_total_Pa_s=props.mu_tot,
        mu_gas_Pa_s=props.mu_gas,
        mu_liquid_Pa_s=props.mu_liquid,
        mu_no_slip_Pa_s=bb.mu_no_slip,

        cp_gas_J_kgK=props.cp_gas_mass,
        cp_total_J_kgK=props.cp_tot_mass,
        cp_liquid_J_kgK=props.cp_liquid_mass,
        cp_mixture_J_kgK=cp_mix,

        jt_coeff_C_bara=props.jt_coeff_gas * 1e5,
        jt_coeff_K_Pa=props.jt_coeff_gas,

        mol_weight_gas_g_mol=props.mol_weight_gas,
        mol_weight_total_g_mol=props.mol_weight_tot,

        gas_mass_fraction_pvt=props.mass_frac_gas_eq,
        gas_mass_fraction_flow=flow_gas_frac,
        gas_mass_fraction_inventory=bb.gas_mass_fraction_inventory,
        mole_fraction_gas_pvt=props.mole_frac_gas_eq,
        volume_fraction_gas_pvt=props.volume_frac_gas_eq,

        liquid_mass_fraction_inventory=1.0 - bb.gas_mass_fraction_inventory,
        liquid_holdup=bb.liquid_holdup_inclined,
        gas_void_fraction=bb.gas_void_fraction,
        no_slip_liquid_fraction=bb.no_slip_liquid_fraction,

        v_sg_m_s=bb.v_sg,
        v_sl_m_s=bb.v_sl,
        v_m_m_s=bb.v_m,
        v_gas_actual_m_s=bb.v_gas_actual,
        v_liquid_actual_m_s=bb.v_liquid_actual,
        slip_ratio=bb.slip_ratio,

        froude_number=bb.froude_number,
        reynolds_no_slip=bb.reynolds_no_slip,
        liquid_velocity_number=bb.liquid_velocity_number,

        flow_pattern=bb.flow_pattern,

        f_no_slip=bb.f_no_slip,
        friction_multiplier=bb.friction_multiplier,
        f_two_phase=bb.f_two_phase,

        dp_dL_friction_Pa_m=bb.dp_dL_friction,
        dp_dL_gravity_Pa_m=bb.dp_dL_gravity,
        dp_dL_total_Pa_m=bb.dp_dL_total,

        thermal_a=thermal.a,
        thermal_b=thermal.b,
        thermal_term1=thermal.term1,
        thermal_term2=thermal.term2,
        thermal_term3=thermal.term3,
        hydraulic_gradient=thermal.hydraulic_gradient,

        iterations=iteration,
        converged=converged_micro,
    )


def micro_result_to_dict(result: MicroSegmentResult) -> dict:
    """将 MicroSegmentResult 转为字典，用于输出 DataFrame"""
    return asdict(result)
