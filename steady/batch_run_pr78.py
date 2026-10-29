"""按算例生成详细 PR78 PVTtab 后批跑 BB、MB、Xiao。

示例：
    python -m steady.batch_run_pr78
    python -m steady.batch_run_pr78 --limit 3 --force-pvt

每个算例只闪蒸一次生成 PVT 表；三种水力模型及其入口压力打靶过程均只
对该表插值。输出目录默认为 ``03_测试模型结果_PR78``，从而不覆盖旧结果。
"""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import pandas as pd

from steady.config import read_pipe_case_from_excel
from steady.pipeline_solver import (
    PipelineForwardBoundary,
    solve_pipeline_with_fixed_inlet_temperature_and_target_outlet_pressure,
)
from steady.pvt_provider import PVTPropertyProvider
from steady.steady_types import PipelineCommonSettings


MODEL_NAMES = {
    "bb": "beggs_brill",
    "mb": "mukherjee_brill",
    "xiao": "xiao",
}

XIAO_PROFILES = {
    "baseline": {},
    "single_phase_guard": {
        "xiao_single_phase_limit_closure": "darcy",
    },
    "recommended": {
        "xiao_intermittent_translation_closure": "olga_density_scaled_gated9",
    },
    "smooth9_candidate": {
        "xiao_intermittent_translation_closure": "olga_density_scaled_smooth9",
    },
    "smooth9_abdul2000": {
        "xiao_intermittent_translation_closure": "olga_density_scaled_smooth9",
        "xiao_intermittent_slug_holdup_closure": "abdul_majeed2000",
    },
    "smooth9_abdul2000_phase_guard": {
        "xiao_single_phase_limit_closure": "darcy",
        "xiao_intermittent_translation_closure": "olga_density_scaled_smooth9",
        "xiao_intermittent_slug_holdup_closure": "abdul_majeed2000",
    },
    "flow_mb": {
        "xiao_flow_pattern_closure": "mukherjee_brill",
    },
    "flow_mb_bubble": {
        "xiao_flow_pattern_closure": "xiao_mb_bubble",
    },
    "flow_mb_bubble_gate": {
        "xiao_flow_pattern_closure": "xiao_mb_bubble_gate",
    },
    "flow_mb_bubble_from_intermittent": {
        "xiao_flow_pattern_closure": "xiao_mb_bubble_from_intermittent",
    },
    "flow_td_bubble_from_intermittent": {
        "xiao_flow_pattern_closure": "xiao_td_bubble_from_intermittent",
    },
    "flow_td_hydraulic_combined": {
        "xiao_flow_pattern_closure": "xiao_td_bubble_from_intermittent",
        "xiao_intermittent_translation_closure": "olga_density_scaled_smooth9",
        "xiao_intermittent_slug_holdup_closure": "abdul_majeed2000",
        "xiao_dispersed_bubble_closure": "petalas_dispersed",
    },
    "flow_hierarchical_combined": {
        "xiao_flow_pattern_closure": "xiao_mb_bubble_from_intermittent",
        "xiao_intermittent_translation_closure": "olga_density_scaled_smooth9",
        "xiao_intermittent_slug_holdup_closure": "abdul_majeed2000",
        "xiao_dispersed_bubble_closure": "petalas_dispersed",
    },
    "combo": {
        "xiao_intermittent_translation_closure": "olga_density_scaled",
        "xiao_intermittent_film_interfacial_closure": "petalas",
    },
    "bubble_candidate": {
        "xiao_dispersed_bubble_closure": "petalas_dispersed",
    },
}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="详细 PR78 PVTtab 稳态批跑")
    parser.add_argument("--case-dir", type=Path, default=Path("01_算例源文件"))
    parser.add_argument("--output-root", type=Path, default=Path("03_测试模型结果_PR78"))
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument("--limit", type=int, default=None, help="仅运行排序后的前 N 个算例")
    parser.add_argument("--force-pvt", action="store_true", help="强制重建每个算例的 PR78 表")
    parser.add_argument("--micro-max-iter", type=int, default=8)
    parser.add_argument("--max-shooting-iter", type=int, default=40)
    parser.add_argument(
        "--xiao-profiles", nargs="+", choices=XIAO_PROFILES,
        default=["baseline"],
        help=("Xiao 闭合配置；可同时运行 baseline/single_phase_guard/"
              "recommended/smooth9_candidate/smooth9_abdul2000/"
              "smooth9_abdul2000_phase_guard/flow_mb/flow_mb_bubble/"
              "flow_mb_bubble_gate/flow_mb_bubble_from_intermittent/"
              "flow_td_bubble_from_intermittent/flow_td_hydraulic_combined/"
              "flow_hierarchical_combined/combo/bubble_candidate"),
    )
    return parser.parse_args()


def _make_settings(
    model: str, micro_max_iter: int, xiao_profile: str = "baseline",
) -> PipelineCommonSettings:
    if xiao_profile not in XIAO_PROFILES:
        raise ValueError(f"未知 Xiao 闭合配置: {xiao_profile}")
    return PipelineCommonSettings(
        surface_tension_N_m=0.02,
        micro_max_iter=micro_max_iter,
        pressure_tol_bara=1.0e-3,
        temperature_tol_K=1.0e-2,
        gas_fraction_tol=1.0e-3,
        relaxation=0.9,
        allow_extrapolation=False,
        two_phase_model=model,
        **XIAO_PROFILES[xiao_profile],
    )


def _pvt_grid(sections, inlet_temperature_C: float, outlet_pressure_bara: float) -> tuple[float, float, int, float, float, int]:
    ambient_C = [section.env_temp_K - 273.15 for section in sections]
    return (
        0.1, outlet_pressure_bara * 4.0, 28,
        min([inlet_temperature_C, *ambient_C]) - 25.0,
        max([inlet_temperature_C, *ambient_C]) + 25.0, 12,
    )


def _summary_row(case_name: str, model_key: str, result, elapsed_s: float) -> dict:
    if result is None:
        return {"case": case_name, "model": model_key, "converged": False, "elapsed_s": elapsed_s}
    dataframe = result.micro_dataframe
    return {
        "case": case_name,
        "model": model_key,
        "converged": result.converged,
        "elapsed_s": elapsed_s,
        "inlet_pressure_bara": result.inlet_pressure_bara,
        "outlet_pressure_bara": result.outlet_pressure_calculated_bara,
        "outlet_error_bara": result.outlet_pressure_error_bara,
        "pressure_drop_bara": result.total_pressure_drop_bara,
        "shooting_iterations": result.shooting_iterations,
        "all_micro_converged": bool(dataframe["converged"].all()) if not dataframe.empty else False,
        "average_micro_iterations": float(dataframe["iterations"].mean()) if not dataframe.empty else None,
    }


def main() -> None:
    # PR78 预处理器属于可选的重型依赖；延迟到实际批跑时加载，避免仅导入
    # 闭合配置或运行水力单测时被外部预处理包阻塞。
    from steady.transient_pr_pvt import generate_transient_pr_pvttab_csv

    args = _arguments()
    root = Path(__file__).resolve().parent.parent
    case_dir = (root / args.case_dir).resolve()
    output_root = (root / args.output_root).resolve()
    files = sorted(case_dir.glob("pipeline_two_phase_case_*.xlsx"))
    if args.limit is not None:
        files = files[:args.limit]
    if not files:
        raise FileNotFoundError(f"未找到算例：{case_dir}")

    summaries: list[dict] = []
    for index, case_path in enumerate(files, start=1):
        case_name = case_path.stem
        case_output = output_root / case_name
        case_output.mkdir(parents=True, exist_ok=True)
        sections, case_info = read_pipe_case_from_excel(str(case_path))
        target_bara = case_info["outlet_pressure_MPa"] * 10.0
        pmin, pmax, pn, tmin, tmax, tn = _pvt_grid(sections, case_info["inlet_temperature_C"], target_bara)
        pvt_file = generate_transient_pr_pvttab_csv(
            workbook_path=case_path,
            output_path=case_output / "PVTtab_PR78.csv",
            pmin_bara=pmin, pmax_bara=pmax, pn=pn,
            tmin_C=tmin, tmax_C=tmax, tn=tn,
            force=args.force_pvt,
        )
        pvt = PVTPropertyProvider(str(pvt_file))
        boundary = PipelineForwardBoundary(case_info["inlet_temperature_C"] + 273.15, target_bara)
        guess, low, high = target_bara * 1.1, target_bara * 0.85, target_bara * 1.8
        print(f"[{index}/{len(files)}] {case_name}: PR78 PVTtab={pvt_file.name}")

        for key in args.models:
            model = MODEL_NAMES[key]
            profiles = args.xiao_profiles if key == "xiao" else ["baseline"]
            for profile in profiles:
                run_key = key if profile == "baseline" else f"{key}_{profile}"
                started = time.perf_counter()
                result = None
                try:
                    result = solve_pipeline_with_fixed_inlet_temperature_and_target_outlet_pressure(
                        pvt=pvt, sections=sections, boundary=boundary,
                        settings=_make_settings(model, args.micro_max_iter, profile),
                        inlet_pressure_guess_bara=guess, inlet_pressure_low_bara=low,
                        inlet_pressure_high_bara=high, shooting_tol_bara=0.01,
                        max_shooting_iter=args.max_shooting_iter, debug=False,
                    )
                except (ArithmeticError, ValueError) as exc:
                    print(f"  {run_key}: failed: {exc}")
                elapsed = time.perf_counter() - started
                model_output = case_output / f"{run_key}_{model}"
                model_output.mkdir(exist_ok=True)
                if result is not None and not result.micro_dataframe.empty:
                    filename = "pipeline_forward_plot_data.csv" if result.converged else "pipeline_forward_best_effort_data.csv"
                    result.micro_dataframe.to_csv(model_output / filename, index=False)
                summaries.append(_summary_row(case_name, run_key, result, elapsed))
                state = "ok" if result is not None and result.converged else "not-converged"
                print(f"  {run_key}: {state}, {elapsed:.1f}s")

    output_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(output_root / "summary_PR78.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
