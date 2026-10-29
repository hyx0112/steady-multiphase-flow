# steady/pipeline_solver.py — 正向管线求解器
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import math

import pandas as pd

from steady.pvt_provider import PVTPropertyProvider
from steady.steady_types import (
    PipeSection,
    PipelineCommonSettings,
    MicroSegmentResult,
    micro_result_to_dict,
)
from steady.config import build_section_positions
from steady.segment_solver import (
    MicroSegmentForwardInput,
    solve_micro_segment_forward,
)


@dataclass
class PipelineForwardBoundary:
    """正向计算边界条件：固定入口温度 + 目标出口压力"""
    inlet_temperature_K: float
    target_outlet_pressure_bara: float


@dataclass
class PipelineForwardSolveResult:
    """全线正向求解结果"""
    inlet_pressure_bara: float
    inlet_temperature_K: float
    inlet_temperature_C: float

    outlet_pressure_calculated_bara: float
    outlet_pressure_target_bara: float
    outlet_pressure_error_bara: float

    outlet_temperature_calculated_K: float
    outlet_temperature_calculated_C: float

    total_pressure_drop_bara: float
    total_temperature_change_K: float

    micro_results: List[MicroSegmentResult]
    section_summary: pd.DataFrame
    micro_dataframe: pd.DataFrame

    converged: bool
    shooting_iterations: int


def solve_pipe_section_forward(
    pvt: PVTPropertyProvider,
    section: PipeSection,
    section_position: dict,
    pressure_inlet_bara: float,
    temperature_inlet_K: float,
    settings: PipelineCommonSettings,
    *,
    debug: bool = False,
) -> Tuple[float, float, List[MicroSegmentResult]]:
    """正向计算单个大管段：已知入口 P/T，逐微元计算至出口"""
    if section.section_length_m <= 0 or section.n_subsegments <= 0 or section.sub_length_m <= 0:
        raise ValueError(f"section {section.section_no} 参数无效。")
    if pressure_inlet_bara <= 0 or temperature_inlet_K <= 0:
        raise ValueError("入口压力/温度必须大于 0。")

    angle = math.atan2(section.elevation_change_m, section.section_length_m)
    x_start = section_position["x_start_m"]
    z_start = section_position["elevation_start_m"]
    sub_dx = section.section_length_m / section.n_subsegments
    sub_dz = section.elevation_change_m / section.n_subsegments
    sub_len = section.sub_length_m

    P = pressure_inlet_bara
    T = temperature_inlet_K
    results = []

    for micro_no in range(1, section.n_subsegments + 1):
        x_in = x_start + (micro_no - 1) * sub_dx
        x_out = x_start + micro_no * sub_dx
        z_in = z_start + (micro_no - 1) * sub_dz
        z_out = z_start + micro_no * sub_dz

        micro_in = MicroSegmentForwardInput(
            section_no=section.section_no,
            micro_no=micro_no,
            x_inlet_m=x_in,
            x_outlet_m=x_out,
            elevation_inlet_m=z_in,
            elevation_outlet_m=z_out,
            pressure_inlet_bara=P,
            temperature_inlet_K=T,
            length_m=sub_len,
            horizontal_length_m=sub_dx,
            elevation_change_m=sub_dz,
            angle_rad=angle,
            diameter_m=section.diameter_m,
            roughness_m=section.roughness_m,
            mass_flow_kg_s=section.mass_flow_kg_s,
            overall_heat_transfer=section.overall_heat_transfer,
            env_temp_K=section.env_temp_K,
            surface_tension_N_m=settings.surface_tension_N_m,
            max_iter=settings.micro_max_iter,
            pressure_tol_bara=settings.pressure_tol_bara,
            temperature_tol_K=settings.temperature_tol_K,
            gas_fraction_tol=settings.gas_fraction_tol,
            relaxation=settings.relaxation,
            allow_extrapolation=settings.allow_extrapolation,
            debug=debug,
            two_phase_model=settings.two_phase_model,
            df_sigma_water=settings.df_sigma_water,
            df_rho_water=settings.df_rho_water,
            df_rho_air=settings.df_rho_air,
            df_mu_water=settings.df_mu_water,
            xiao_single_phase_limit_closure=(
                settings.xiao_single_phase_limit_closure),
            xiao_flow_pattern_closure=settings.xiao_flow_pattern_closure,
            xiao_stratified_interfacial_closure=(
                settings.xiao_stratified_interfacial_closure),
            xiao_intermittent_translation_closure=(
                settings.xiao_intermittent_translation_closure),
            xiao_intermittent_slug_holdup_closure=(
                settings.xiao_intermittent_slug_holdup_closure),
            xiao_intermittent_film_interfacial_closure=(
                settings.xiao_intermittent_film_interfacial_closure),
            xiao_annular_entrainment_closure=(
                settings.xiao_annular_entrainment_closure),
            xiao_annular_interfacial_closure=(
                settings.xiao_annular_interfacial_closure),
            xiao_dispersed_bubble_closure=(
                settings.xiao_dispersed_bubble_closure),
        )
        res = solve_micro_segment_forward(pvt, micro_in)
        if not res.converged:
            # 未收敛的固定点不能作为打靶函数值；否则外层可能“收敛”到一个
            # 依赖迭代上限的伪出口压力。将它显式交给 shooting 的失效恢复。
            raise ArithmeticError(
                f"section {section.section_no}, micro {micro_no} 的 P/T 平均状态"
                f"迭代在 {settings.micro_max_iter} 次内未收敛。"
            )
        results.append(res)
        P, T = res.pressure_outlet_bara, res.temperature_outlet_K

    return P, T, results


def solve_pipeline_forward_given_inlet_pressure(
    pvt: PVTPropertyProvider,
    sections: List[PipeSection],
    inlet_pressure_bara: float,
    inlet_temperature_K: float,
    settings: PipelineCommonSettings,
    *,
    debug: bool = False,
) -> Tuple[float, float, List[MicroSegmentResult]]:
    """给定全线入口 P/T，正向计算全线出口 P/T（不做 shooting）"""
    if not sections or inlet_pressure_bara <= 0 or inlet_temperature_K <= 0:
        raise ValueError("参数无效。")

    positions = build_section_positions(sections)
    P, T = inlet_pressure_bara, inlet_temperature_K
    all_results = []

    for section in sections:
        P, T, sec_res = solve_pipe_section_forward(
            pvt, section, positions[section.section_no], P, T, settings, debug=debug
        )
        all_results.extend(sec_res)

    all_results.sort(key=lambda r: (r.x_mid_m, r.section_no, r.micro_no))
    return P, T, all_results


def solve_pipeline_with_fixed_inlet_temperature_and_target_outlet_pressure(
        pvt: PVTPropertyProvider,
        sections: List[PipeSection],
        boundary: PipelineForwardBoundary,
        settings: PipelineCommonSettings,
        inlet_pressure_guess_bara: float,
        inlet_pressure_low_bara: float,
        inlet_pressure_high_bara: float,
        shooting_tol_bara: float = 0.01,
        max_shooting_iter: int = 40,
        *,
        debug: bool = True,
) -> PipelineForwardSolveResult:
    """
    固定入口温度 + 目标出口压力的受限割线打靶。

    在尚未获得异号根区间时用割线搜索；一旦得到有效根区间，采用受限
    割线（试探点过于靠近端点时退回二分）。这保留割线的速度，同时避免
    相关式非单调或个别压力点超出适用域时跨越已经验证的物理边界。

    目标函数：f(P_in) = P_out(P_in) - target_outlet_pressure
    """
    if not sections or boundary.inlet_temperature_K <= 0 or boundary.target_outlet_pressure_bara <= 0:
        raise ValueError("输入参数无效。")

    target = boundary.target_outlet_pressure_bara
    T_in = boundary.inlet_temperature_K

    def run_once(P_in: float):
        P_out, T_out, micro = solve_pipeline_forward_given_inlet_pressure(
            pvt, sections, P_in, T_in, settings, debug=debug
        )
        return P_out - target, P_out, T_out, micro

    # ——— 硬边界（物理约束）———
    P_min = max(inlet_pressure_low_bara * 0.5, target * 0.9)
    P_max = min(inlet_pressure_high_bara * 1.5, target * 3.0)

    # ——— 初始点 ———
    P0 = max(P_min, min(P_max, inlet_pressure_guess_bara))
    err0, P_out0, T_out0, micro0 = run_once(P0)

    if debug:
        print(f"[Shooting] 纯牛顿割线法 | 目标 P_out={target:.4f} bara | "
              f"边界 [{P_min:.2f}, {P_max:.2f}]")
        print(f"[Shooting] Iter  0 | P_in={P0:8.4f} | Err={err0:+.4e} | P_out={P_out0:.4f}")

    # 跟踪最佳
    best_P, best_err = P0, err0
    best_P_out, best_T_out = P_out0, T_out0
    best_micro = micro0
    converged_flag = False
    final_iter = 0
    # 相关式在个别试探入口压力下可能超出适用范围（例如持液率回归值不在
    # (0, 1)）。这种“试探点失效”不是整个打靶问题的致命错误；记录其作为
    # 单侧不可行边界，并继续在仍有效的一侧寻找解或最优近似。
    invalid_lower: float | None = None
    invalid_upper: float | None = None
    # (P_low, err_low, P_high, err_high)，且 err_low * err_high < 0。
    # 只有两个端点均为有效的水力解时才建立，不把相关式域外点伪装成根。
    bracket: tuple[float, float, float, float] | None = None

    # 快速收敛检查
    if abs(err0) <= shooting_tol_bara:
        converged_flag = True
        if debug:
            print(f"[Shooting] 初始猜测已收敛！")

    if not converged_flag:
        # 构造第二个点：微小扰动
        delta = max(0.5, P0 * 0.005)
        P1 = P0 + delta
        if P1 > P_max:
            P1 = P0 - delta
        P1 = max(P_min, min(P_max, P1))
        err1, P_out1, T_out1, micro1 = run_once(P1)

        if debug:
            print(f"[Shooting] Iter  1 | P_in={P1:8.4f} | Err={err1:+.4e} | P_out={P_out1:.4f}")

        if abs(err1) < abs(best_err):
            best_P, best_err = P1, err1
            best_P_out, best_T_out = P_out1, T_out1
            best_micro = micro1

        if abs(err1) <= shooting_tol_bara:
            converged_flag = True
            final_iter = 1
            if debug:
                print(f"[Shooting] 第 1 次迭代收敛！")
        else:
            if err0 * err1 < 0.0:
                if P0 < P1:
                    bracket = (P0, err0, P1, err1)
                else:
                    bracket = (P1, err1, P0, err0)

            # ——— 受限割线/二分迭代 ———
            # 跟踪已探索的点，避免重复
            explored = {P0, P1}

            for it in range(2, max_shooting_iter + 1):
                denom = err1 - err0
                if bracket is not None:
                    P_low, err_low, P_high, err_high = bracket
                    width = P_high - P_low
                    # Illinois/Brent 风格的受限割线：不允许下一点贴近边界，
                    # 这样既保证区间收缩，也避免相关式在边界附近反复失效。
                    denom_bracket = err_high - err_low
                    if abs(denom_bracket) < 1e-12:
                        P_new = 0.5 * (P_low + P_high)
                    else:
                        P_secant = P_high - err_high * width / denom_bracket
                        if P_secant <= P_low + 0.1 * width or P_secant >= P_high - 0.1 * width:
                            P_new = 0.5 * (P_low + P_high)
                        else:
                            P_new = P_secant
                elif err1 < 0.0 and invalid_upper is not None:
                    # 当前出口压力偏低，应提高入口压力；但更高的一侧已有
                    # 无效试探点，因此在有效点与该边界之间折半。
                    P_new = 0.5 * (P1 + invalid_upper)
                elif err1 > 0.0 and invalid_lower is not None:
                    # 当前出口压力偏高，应降低入口压力；同理避开无效下界。
                    P_new = 0.5 * (P1 + invalid_lower)
                elif abs(denom) < 1e-12:
                    # 导数为零（函数平坦），需要朝减小 |error| 的方向探索
                    # err < 0 表示 P_out 偏低，应增大 P_in
                    # err > 0 表示 P_out 偏高，应减小 P_in
                    if err1 < 0:
                        # P_out 太低 → 需要更大 P_in，先尝试大步长
                        step = max(5.0, P1 * 0.05)
                        P_new = P1 + step
                    else:
                        step = max(5.0, P1 * 0.05)
                        P_new = P1 - step
                else:
                    # 纯割线法（牛顿法以差商近似导数）
                    P_new = P1 - err1 * (P1 - P0) / denom

                # 硬边界裁剪
                P_new = max(P_min, min(P_max, P_new))

                # 避免重复探索同一点
                if P_new in explored:
                    # 微调避开
                    P_new = P_new + (2.0 if err1 < 0 else -2.0)
                    P_new = max(P_min, min(P_max, P_new))

                explored.add(P_new)
                try:
                    err_new, P_out_new, T_out_new, micro_new = run_once(P_new)
                except (ArithmeticError, ValueError) as exc:
                    if P_new >= P1:
                        invalid_upper = (
                            P_new if invalid_upper is None else min(invalid_upper, P_new)
                        )
                    else:
                        invalid_lower = (
                            P_new if invalid_lower is None else max(invalid_lower, P_new)
                        )
                    if debug:
                        print(
                            f"[Shooting] Iter {it:2d} | P_in={P_new:8.4f} | "
                            f"试探失效，回退搜索：{exc}"
                        )
                    # 保留最近两个有效点。下一轮会根据无效侧边界做受限搜索。
                    continue

                if debug:
                    print(f"[Shooting] Iter {it:2d} | P_in={P_new:8.4f} | Err={err_new:+.4e} | "
                          f"P_out={P_out_new:.4f}")

                if abs(err_new) < abs(best_err):
                    best_P, best_err = P_new, err_new
                    best_P_out, best_T_out = P_out_new, T_out_new
                    best_micro = micro_new

                if abs(err_new) <= shooting_tol_bara:
                    converged_flag = True
                    final_iter = it
                    break

                # 一旦相邻两个有效点构成异号区间，后续试探始终留在该区间。
                if bracket is None:
                    if err1 * err_new < 0.0:
                        if P1 < P_new:
                            bracket = (P1, err1, P_new, err_new)
                        else:
                            bracket = (P_new, err_new, P1, err1)
                else:
                    P_low, err_low, P_high, err_high = bracket
                    if err_low * err_new < 0.0:
                        bracket = (P_low, err_low, P_new, err_new)
                    elif err_high * err_new < 0.0:
                        bracket = (P_new, err_new, P_high, err_high)

                # 轮换
                P0, err0 = P1, err1
                P1, err1 = P_new, err_new

                if it % 10 == 0 and debug:
                    print(f"  -> 当前最佳: P_in={best_P:.4f} | Err={best_err:.4e}")

            if not converged_flag:
                final_iter = max_shooting_iter

    return _build_pipeline_result(
        best_P, best_P_out, best_T_out, T_in, target, best_err,
        best_micro, converged=converged_flag, shooting_iterations=final_iter
    )
def _build_pipeline_result(P_in, P_out, T_out_K, T_in_K, target, err,
                           micro, converged, shooting_iterations):
    """构造 PipelineForwardSolveResult"""
    micro_df = make_micro_dataframe(micro)
    section_summary = make_section_summary(micro_df)
    return PipelineForwardSolveResult(
        inlet_pressure_bara=P_in,
        inlet_temperature_K=T_in_K,
        inlet_temperature_C=T_in_K - 273.15,
        outlet_pressure_calculated_bara=P_out,
        outlet_pressure_target_bara=target,
        outlet_pressure_error_bara=err,
        outlet_temperature_calculated_K=T_out_K,
        outlet_temperature_calculated_C=T_out_K - 273.15,
        total_pressure_drop_bara=P_in - P_out,
        total_temperature_change_K=T_in_K - T_out_K,
        micro_results=micro,
        section_summary=section_summary,
        micro_dataframe=micro_df,
        converged=converged,
        shooting_iterations=shooting_iterations,
    )


def make_micro_dataframe(micro_results: List[MicroSegmentResult]) -> pd.DataFrame:
    """生成小微元详细结果表"""
    rows = [micro_result_to_dict(r) for r in micro_results]
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["x_axis_horizontal_distance_m"] = df["x_mid_m"]
    df["x_axis_horizontal_distance_km"] = df["x_mid_m"] / 1000.0
    df["section_micro_label"] = df["section_no"].astype(str) + "-" + df["micro_no"].astype(str)

    inlet_p = float(df["pressure_inlet_bara"].iloc[0])
    df["cumulative_pressure_drop_bara_from_inlet"] = inlet_p - df["pressure_outlet_bara"]
    inlet_T = float(df["temperature_inlet_K"].iloc[0])
    df["cumulative_temperature_change_K_from_inlet"] = inlet_T - df["temperature_outlet_K"]
    df["micro_pressure_drop_bara_check"] = df["pressure_inlet_bara"] - df["pressure_outlet_bara"]
    df["micro_temperature_drop_K_check"] = df["temperature_inlet_K"] - df["temperature_outlet_K"]
    return df


def make_section_summary(micro_df: pd.DataFrame) -> pd.DataFrame:
    """生成大管段汇总表"""
    if micro_df.empty:
        return pd.DataFrame()

    rows = []
    for sec_no, grp in micro_df.groupby("section_no", sort=True):
        g = grp.sort_values("micro_no")
        first, last = g.iloc[0], g.iloc[-1]
        rows.append({
            "section_no": int(sec_no),
            "x_inlet_m": first["x_inlet_m"],
            "x_outlet_m": last["x_outlet_m"],
            "x_inlet_km": first["x_inlet_m"] / 1000.0,
            "x_outlet_km": last["x_outlet_m"] / 1000.0,
            "elevation_inlet_m": first["elevation_inlet_m"],
            "elevation_outlet_m": last["elevation_outlet_m"],
            "elevation_change_m": last["elevation_outlet_m"] - first["elevation_inlet_m"],
            "angle_deg_avg": g["angle_deg"].mean(),
            "pressure_inlet_bara": first["pressure_inlet_bara"],
            "pressure_outlet_bara": last["pressure_outlet_bara"],
            "pressure_drop_bara": first["pressure_inlet_bara"] - last["pressure_outlet_bara"],
            "temperature_inlet_C": first["temperature_inlet_C"],
            "temperature_outlet_C": last["temperature_outlet_C"],
            "temperature_change_K": first["temperature_inlet_K"] - last["temperature_outlet_K"],
            "avg_liquid_holdup": g["liquid_holdup"].mean(),
            "min_liquid_holdup": g["liquid_holdup"].min(),
            "max_liquid_holdup": g["liquid_holdup"].max(),
            "avg_gas_mass_fraction_pvt": g["gas_mass_fraction_pvt"].mean(),
            "avg_gas_mass_fraction_flow": g["gas_mass_fraction_flow"].mean(),
            "avg_gas_mass_fraction_inventory": g["gas_mass_fraction_inventory"].mean(),
            "avg_v_sg_m_s": g["v_sg_m_s"].mean(),
            "avg_v_sl_m_s": g["v_sl_m_s"].mean(),
            "avg_v_m_m_s": g["v_m_m_s"].mean(),
            "avg_v_gas_actual_m_s": g["v_gas_actual_m_s"].mean(),
            "avg_v_liquid_actual_m_s": g["v_liquid_actual_m_s"].mean(),
            "avg_rho_gas_kg_m3": g["rho_gas_kg_m3"].mean(),
            "avg_rho_liquid_kg_m3": g["rho_liquid_kg_m3"].mean(),
            "avg_rho_holdup_kg_m3": g["rho_holdup_kg_m3"].mean(),
            "avg_mu_gas_Pa_s": g["mu_gas_Pa_s"].mean(),
            "avg_mu_liquid_Pa_s": g["mu_liquid_Pa_s"].mean(),
            "avg_mu_no_slip_Pa_s": g["mu_no_slip_Pa_s"].mean(),
            "avg_cp_gas_J_kgK": g["cp_gas_J_kgK"].mean(),
            "avg_cp_liquid_J_kgK": g["cp_liquid_J_kgK"].mean(),
            "avg_cp_mixture_J_kgK": g["cp_mixture_J_kgK"].mean(),
            "avg_dp_dL_friction_Pa_m": g["dp_dL_friction_Pa_m"].mean(),
            "avg_dp_dL_gravity_Pa_m": g["dp_dL_gravity_Pa_m"].mean(),
            "avg_dp_dL_total_Pa_m": g["dp_dL_total_Pa_m"].mean(),
            "main_flow_pattern": g["flow_pattern"].mode().iloc[0],
            "micro_count": len(g),
            "all_micro_converged": bool(g["converged"].all()),
            "max_micro_iterations": int(g["iterations"].max()),
        })
    return pd.DataFrame(rows)
