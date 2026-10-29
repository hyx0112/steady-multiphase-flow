# steady/batch_run.py — 批量管道算例自动化计算
"""
批量读取 pipe_case_N.xlsx 算例，对每个算例依次使用 Beggs-Brill、
Mukherjee-Brill 和漂移流模型进行稳态仿真，输出 PIPESIM 格式结果 Excel。
"""

from __future__ import annotations

import os
import time
import glob
from pathlib import Path
from typing import List, Tuple, Optional

import pandas as pd

from steady.pvt_provider import PVTPropertyProvider
from steady.steady_types import PipeSection, PipelineCommonSettings
from steady.config import read_pipe_case_from_excel
from steady.pipeline_solver import (
    PipelineForwardBoundary,
    PipelineForwardSolveResult,
    solve_pipeline_with_fixed_inlet_temperature_and_target_outlet_pressure,
)

# ---------------------------------------------------------------------------
# 可配置参数
# ---------------------------------------------------------------------------

# PVT 表文件（默认使用项目根目录下的 fuqi1.xls，可在调用时覆盖）
PVT_FILE_NAME = "fuqi1.xls"

# pipe_case 文件所在目录
PIPE_CASE_DIR = "管道与组分 copy"

# 输出文件
OUTPUT_FILE = "steady_batch_results.xlsx"

# 全线收敛设置
SETTINGS_BASE = PipelineCommonSettings(
    surface_tension_N_m=0.02,
    micro_max_iter=40,
    pressure_tol_bara=1.0e-3,
    temperature_tol_K=1.0e-2,
    gas_fraction_tol=1.0e-3,
    relaxation=0.9,
    allow_extrapolation=False,
    two_phase_model="beggs_brill",  # 每轮覆盖
)

SHOOTING_TOL_BARA = 0.01
MAX_SHOOTING_ITER = 40

# 三种两相流模型
MODELS = {
    "BB": "beggs_brill",
    "MB": "mukherjee_brill",
    "DF": "drift_flux",
}

# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _make_settings(model_key: str) -> PipelineCommonSettings:
    """复制基础设置并指定两相流模型。"""
    return PipelineCommonSettings(
        surface_tension_N_m=SETTINGS_BASE.surface_tension_N_m,
        micro_max_iter=SETTINGS_BASE.micro_max_iter,
        pressure_tol_bara=SETTINGS_BASE.pressure_tol_bara,
        temperature_tol_K=SETTINGS_BASE.temperature_tol_K,
        gas_fraction_tol=SETTINGS_BASE.gas_fraction_tol,
        relaxation=SETTINGS_BASE.relaxation,
        allow_extrapolation=SETTINGS_BASE.allow_extrapolation,
        two_phase_model=MODELS[model_key],
    )


def _pressure_guess_range(outlet_P_bara: float) -> Tuple[float, float, float]:
    """根据目标出口压力估算入口压力打靶区间。"""
    guess = outlet_P_bara * 1.1
    low = outlet_P_bara * 0.85
    high = outlet_P_bara * 1.8
    return guess, low, high


def run_single_case(
    pvt: PVTPropertyProvider,
    sections: List[PipeSection],
    inlet_T_C: float,
    outlet_P_MPa: float,
    model_key: str,
    case_name: str,
) -> Optional[PipelineForwardSolveResult]:
    """
    运行单个算例的单个模型。

    返回 PipelineForwardSolveResult，失败返回 None。
    """
    settings = _make_settings(model_key)
    outlet_P_bara = outlet_P_MPa * 10.0

    boundary = PipelineForwardBoundary(
        inlet_temperature_K=inlet_T_C + 273.15,
        target_outlet_pressure_bara=outlet_P_bara,
    )

    pin_guess, pin_low, pin_high = _pressure_guess_range(outlet_P_bara)

    print(f"  [{model_key}] P_out={outlet_P_bara:.2f} bara, "
          f"T_in={inlet_T_C:.1f}°C, "
          f"P_in guess={pin_guess:.1f} bara")

    try:
        result = solve_pipeline_with_fixed_inlet_temperature_and_target_outlet_pressure(
            pvt=pvt,
            sections=sections,
            boundary=boundary,
            settings=settings,
            inlet_pressure_guess_bara=pin_guess,
            inlet_pressure_low_bara=pin_low,
            inlet_pressure_high_bara=pin_high,
            shooting_tol_bara=SHOOTING_TOL_BARA,
            max_shooting_iter=MAX_SHOOTING_ITER,
            debug=False,
        )
        if result.converged:
            print(f"  [{model_key}] Converged  P_in={result.inlet_pressure_bara:.2f} bara, "
                  f"P_out={result.outlet_pressure_calculated_bara:.2f} bara, "
                  f"iter={result.shooting_iterations}")
        else:
            print(f"  [{model_key}] NOT converged  Best P_in={result.inlet_pressure_bara:.2f} bara, "
                  f"err={result.outlet_pressure_error_bara:.4f} bara")
        return result
    except Exception as exc:
        print(f"  [{model_key}] 异常: {exc}")
        return None


def build_pipesim_profile(result: PipelineForwardSolveResult) -> pd.DataFrame:
    """
    从稳态结果提取 PIPESIM 格式的沿线分布表。
    """
    if result is None or result.micro_dataframe is None or result.micro_dataframe.empty:
        return pd.DataFrame()

    df = result.micro_dataframe.copy()
    profile = pd.DataFrame()
    profile["Total distance (m)"] = df["x_axis_horizontal_distance_m"]
    profile["Pressure (bara)"] = df["pressure_outlet_bara"]
    profile["Temperature (K)"] = df["temperature_outlet_K"]
    profile["Liquid holdup (%)"] = df["liquid_holdup"] * 100.0
    profile["Gas velocity (m/s)"] = df["v_gas_actual_m_s"]
    profile["Liq. velocity (m/s)"] = df["v_liquid_actual_m_s"]
    profile["G-L Pattern"] = df["flow_pattern"]
    return profile


def build_summary_row(
    case_name: str,
    model_key: str,
    result: Optional[PipelineForwardSolveResult],
) -> dict:
    """生成汇总表的一行。"""
    if result is None:
        return {
            "Case": case_name,
            "Model": model_key,
            "Converged": False,
            "Inlet P (bara)": None,
            "Outlet P (bara)": None,
            "P error (bara)": None,
            "Outlet T (K)": None,
            "Delta P (bara)": None,
            "Delta T (K)": None,
            "Inlet liquid holdup (%)": None,
            "Outlet liquid holdup (%)": None,
            "Inlet gas velocity (m/s)": None,
            "Outlet gas velocity (m/s)": None,
            "Shooting iters": None,
        }

    df = result.micro_dataframe
    first = df.iloc[0]
    last = df.iloc[-1]
    return {
        "Case": case_name,
        "Model": model_key,
        "Converged": result.converged,
        "Inlet P (bara)": round(result.inlet_pressure_bara, 4),
        "Outlet P (bara)": round(result.outlet_pressure_calculated_bara, 4),
        "P error (bara)": round(result.outlet_pressure_error_bara, 6),
        "Outlet T (K)": round(result.outlet_temperature_calculated_K, 2),
        "Delta P (bara)": round(result.total_pressure_drop_bara, 4),
        "Delta T (K)": round(result.total_temperature_change_K, 2),
        "Inlet liquid holdup (%)": round(first["liquid_holdup"] * 100, 2),
        "Outlet liquid holdup (%)": round(last["liquid_holdup"] * 100, 2),
        "Inlet gas velocity (m/s)": round(first["v_gas_actual_m_s"], 4),
        "Outlet gas velocity (m/s)": round(last["v_gas_actual_m_s"], 4),
        "Shooting iters": result.shooting_iterations,
    }


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def main() -> None:
    script_dir = Path(__file__).resolve().parent.parent
    case_dir = script_dir / PIPE_CASE_DIR
    if not case_dir.is_dir():
        print(f"错误：算例目录不存在: {case_dir}")
        return

    case_files = sorted(glob.glob(str(case_dir / "pipe_case_*.xlsx")))
    if not case_files:
        print(f"未找到 pipe_case_*.xlsx 文件: {case_dir}")
        return

    print(f"找到 {len(case_files)} 个算例文件")

    # PVT 文件路径：默认在项目根目录下
    pvt_file = script_dir / PVT_FILE_NAME
    if not pvt_file.exists():
        raise FileNotFoundError(f"PVT 文件不存在: {pvt_file}")
    print(f"PVT 文件: {pvt_file}")
    print(f"模型: {list(MODELS.keys())}")
    print("=" * 60)

    # 加载 PVT（所有算例共用）
    print("\n加载 PVT 表 ...")
    pvt = PVTPropertyProvider(str(pvt_file))
    print(pvt.summary())

    # 批量计算
    all_profiles: dict[str, pd.DataFrame] = {}
    summary_rows: list[dict] = []
    total_start = time.time()

    for case_file in case_files:
        case_name = os.path.splitext(os.path.basename(case_file))[0]
        print(f"\n{'=' * 60}")
        print(f"算例: {case_name}")
        print(f"{'=' * 60}")

        # 读取算例
        sections, case_info = read_pipe_case_from_excel(case_file)
        inlet_T_C = case_info["inlet_temperature_C"]
        outlet_P_MPa = case_info["outlet_pressure_MPa"]
        print(f"  管段数: {len(sections)}, "
              f"总微元数: {sum(s.n_subsegments for s in sections)}, "
              f"入口温度: {inlet_T_C}°C, "
              f"出口压力: {outlet_P_MPa} MPa")

        # 三种模型
        for model_key in MODELS:
            t0 = time.time()
            result = run_single_case(
                pvt=pvt,
                sections=sections,
                inlet_T_C=inlet_T_C,
                outlet_P_MPa=outlet_P_MPa,
                model_key=model_key,
                case_name=case_name,
            )
            elapsed = time.time() - t0
            print(f"  [{model_key}] 耗时: {elapsed:.1f} s")

            # 输出剖面
            profile = build_pipesim_profile(result)
            sheet_name = f"{case_name}_{model_key}"
            all_profiles[sheet_name] = profile

            # 汇总行
            summary_rows.append(build_summary_row(case_name, model_key, result))

    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"全部完成，总耗时: {total_elapsed:.1f} s ({total_elapsed / 60:.1f} min)")

    # 输出 Excel
    output_path = script_dir / OUTPUT_FILE
    print(f"\n写入结果: {output_path}")
    summary_df = pd.DataFrame(summary_rows)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        for sheet_name, profile_df in all_profiles.items():
            # sheet 名称 Excel 限制 31 字符
            safe_name = sheet_name[:31]
            profile_df.to_excel(writer, sheet_name=safe_name, index=False)

    # 汇总到控制台
    print("\n========== 批量计算汇总 ==========")
    print(summary_df.to_string(index=False))
    print(f"\n结果已保存至: {output_path.resolve()}")


if __name__ == "__main__":
    main()
