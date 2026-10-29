#!/usr/bin/env python
"""steady.run — CLI entry point for steady-state pipeline simulation.

Usage:
    python -m steady.run --pvt-source flash --pipe case.xlsx --model beggs_brill
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="稳态管道仿真计算")
    parser.add_argument("--pvt-source", choices=["flash", "file", "transient-pr"], default="flash",
                        help="PVT 来源: flash (thermo)、file (预生成 CSV) 或 transient-pr (Standalone 本地 PR)")
    parser.add_argument("--pvt-file", type=str, default=None,
                        help="预生成的 PVT CSV 文件路径 (当 --pvt-source=file 时使用)")
    parser.add_argument("--pipe", type=str, required=True, help="管道算例 Excel 文件")
    parser.add_argument("--model", type=str, default="beggs_brill",
                        choices=["beggs_brill", "mukherjee_brill", "drift_flux", "xiao"],
                        help="两相流模型")
    parser.add_argument("--no-debug", action="store_true", help="关闭调试输出")
    parser.add_argument("--no-plots", action="store_true", help="不生成图表")
    parser.add_argument("--output", type=str, help="输出目录（相对于项目根目录）")

    # Flash PVT 参数
    parser.add_argument("--flash-eos", type=str, default="srk",
                        choices=["srk", "srk-classic", "pr"], help="状态方程类型")
    parser.add_argument("--flash-pmin", type=float, default=None, help="PVT 表压力下限 bara")
    parser.add_argument("--flash-pmax", type=float, default=None, help="PVT 表压力上限 bara")
    parser.add_argument("--flash-pn", type=int, default=10, help="PVT 压力网格数")
    parser.add_argument("--flash-tmin", type=float, default=None, help="PVT 表温度下限 C")
    parser.add_argument("--flash-tmax", type=float, default=None, help="PVT 表温度上限 C")
    parser.add_argument("--flash-tn", type=int, default=10, help="PVT 温度网格数")
    parser.add_argument("--flash-force", action="store_true", help="强制重新生成 PVT 表")

    # Shooting 参数
    parser.add_argument("--pin-guess", type=float, default=None, help="入口压力初猜 bara")
    parser.add_argument("--pin-low", type=float, default=None, help="入口压力下限 bara")
    parser.add_argument("--pin-high", type=float, default=None, help="入口压力上限 bara")
    parser.add_argument("--max-shooting-iter", type=int, default=40,
                        help="入口压力打靶最大迭代次数（默认 40）")
    parser.add_argument("--micro-max-iter", type=int, default=8,
                        help="每个微元平均 P/T 迭代上限（默认 8）")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipe_path = Path(args.pipe).resolve()
    if not pipe_path.exists():
        print(f"错误：算例文件不存在: {pipe_path}")
        sys.exit(1)

    from steady.config import read_pipe_case_from_excel, read_pipe_sections_from_excel
    from steady.pvt_provider import PVTPropertyProvider
    from steady.steady_types import PipelineCommonSettings
    from steady.pipeline_solver import (
        PipelineForwardBoundary,
        solve_pipeline_with_fixed_inlet_temperature_and_target_outlet_pressure,
    )
    from steady.flash_pvt import (
        export_flash_fluid_input_from_workbook,
        read_boundary_conditions_from_workbook,
        estimate_pvt_grid_from_conditions,
        FlashPVTPropertyProvider,
    )

    # Read pipe case
    sections, case_info = read_pipe_case_from_excel(str(pipe_path))
    inlet_T_C = case_info["inlet_temperature_C"]
    outlet_P_MPa = case_info["outlet_pressure_MPa"]
    outlet_P_bara = outlet_P_MPa * 10.0

    # PVT setup
    output_dir = Path(args.output) if args.output else PROJECT_ROOT / "steady" / "flash_generated_pvt"
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.pvt_source == "flash":
        # Read boundary conditions from workbook
        bc = read_boundary_conditions_from_workbook(str(pipe_path))

        # Determine P/T grid
        pin_guess = args.pin_guess or outlet_P_bara * 1.1
        pin_low = args.pin_low or outlet_P_bara * 0.85
        pin_high = args.pin_high or outlet_P_bara * 1.8

        grid = estimate_pvt_grid_from_conditions(
            inlet_temperature_C=inlet_T_C,
            target_outlet_pressure_bara=outlet_P_bara,
            inlet_pressure_guess_bara=pin_guess,
            inlet_pressure_low_bara=pin_low,
            inlet_pressure_high_bara=pin_high,
            env_temperature_C=bc.get("env_temperature_C"),
        )

        pmin = args.flash_pmin or grid["pmin"]
        pmax = args.flash_pmax or grid["pmax"]
        pn = args.flash_pn
        tmin = args.flash_tmin or grid["tmin"]
        tmax = args.flash_tmax or grid["tmax"]
        tn = args.flash_tn

        # Prepare composition
        comp_path = export_flash_fluid_input_from_workbook(
            workbook_path=str(pipe_path),
            output_dir=output_dir / "compositions",
        )

        # Read composition from the exported file
        import json
        with open(str(comp_path), 'r', encoding='utf-8') as f:
            fluid_data = json.load(f)

        # Build composition as list[dict] (supports pseudo-components)
        composition: list[dict] = []
        comps = fluid_data.get("components", {})
        if isinstance(comps, dict):
            for name, z in comps.items():
                composition.append({"name": name, "moleFraction": z})
        elif isinstance(comps, list):
            composition.extend(comps)

        # Add TBP fractions as pseudo-components
        for tbp in fluid_data.get("tbpFractions", []):
            composition.append({
                "name": tbp["name"],
                "moleFraction": tbp["moleFraction"],
                "molarMassKgPerMol": tbp["molarMassKgPerMol"],
                "densityKgPerM3": tbp.get("densityKgPerM3", 800.0),
            })
        # Add plus fractions as pseudo-components
        for pf in fluid_data.get("plusFractions", []):
            composition.append({
                "name": pf["name"],
                "moleFraction": pf["moleFraction"],
                "molarMassKgPerMol": pf["molarMassKgPerMol"],
                "densityKgPerM3": pf.get("densityKgPerM3", 800.0),
            })

        print(f"闪蒸 PVT 生成: P [{pmin}, {pmax}] × {pn}, T [{tmin}, {tmax}] × {tn}")
        comp_names = [c['name'] for c in composition[:8]]
        print(f"组分: {comp_names}...")

        provider = FlashPVTPropertyProvider(
            composition=composition,
            output_dir=output_dir,
            eos=args.flash_eos,
            pmin=pmin, pmax=pmax, pn=pn,
            tmin=tmin, tmax=tmax, tn=tn,
            force=args.flash_force,
        )
        pvt = provider._provider
    elif args.pvt_source == "file":
        pvt_file = args.pvt_file or str(PROJECT_ROOT / "steady" / "fuqi1.xls")
        pvt = PVTPropertyProvider(pvt_file)
    else:
        from steady.transient_pr_pvt import generate_transient_pr_pvttab_csv

        # 先由 Standalone PR 写出 PVTtab；后续全线计算仅由 PVTPropertyProvider
        # 对该表插值，绝不在沿线/打靶循环中执行闪蒸。
        p_min = 0.1
        p_max = outlet_P_bara * 4.00
        ambient_temperatures_C = [section.env_temp_K - 273.15 for section in sections]
        t_min = min([inlet_T_C, *ambient_temperatures_C]) - 25.0
        t_max = max([inlet_T_C, *ambient_temperatures_C]) + 25.0
        print(
            "生成/复用 Standalone PR PVTtab: "
            f"P [{p_min:.2f}, {p_max:.2f}] bara × 28, "
            f"T [{t_min:.2f}, {t_max:.2f}] °C × 12"
        )
        steady_output = Path(args.output) if args.output else PROJECT_ROOT / "steady" / "output"
        pvttab_path = generate_transient_pr_pvttab_csv(
            workbook_path=pipe_path,
            output_path=steady_output.parent / "PVTtab_StandalonePR.csv",
            pmin_bara=p_min,
            pmax_bara=p_max,
            pn=28,
            tmin_C=t_min,
            tmax_C=t_max,
            tn=12,
            force=args.flash_force,
        )
        pvt = PVTPropertyProvider(str(pvttab_path))
        print(f"稳态 PVT 插值表: {pvttab_path}")

    # Pipeline simulation
    settings = PipelineCommonSettings(
        surface_tension_N_m=0.02,
        micro_max_iter=args.micro_max_iter,
        pressure_tol_bara=1.0e-3,
        temperature_tol_K=1.0e-2,
        gas_fraction_tol=1.0e-3,
        relaxation=0.9,
        allow_extrapolation=False,
        two_phase_model=args.model,
    )

    boundary = PipelineForwardBoundary(
        inlet_temperature_K=inlet_T_C + 273.15,
        target_outlet_pressure_bara=outlet_P_bara,
    )

    pin_guess = args.pin_guess or outlet_P_bara * 1.1
    pin_low = args.pin_low or outlet_P_bara * 0.85
    pin_high = args.pin_high or outlet_P_bara * 1.8

    print(f"\n管道仿真: 模型={args.model}, T_in={inlet_T_C:.1f}°C, P_out={outlet_P_bara:.2f} bara")
    print(f"入口压力打靶: guess={pin_guess:.1f}, low={pin_low:.1f}, high={pin_high:.1f} bara")

    try:
        result = solve_pipeline_with_fixed_inlet_temperature_and_target_outlet_pressure(
            pvt=pvt,
            sections=sections,
            boundary=boundary,
            settings=settings,
            inlet_pressure_guess_bara=pin_guess,
            inlet_pressure_low_bara=pin_low,
            inlet_pressure_high_bara=pin_high,
            shooting_tol_bara=0.01,
            max_shooting_iter=args.max_shooting_iter,
            debug=not args.no_debug,
        )

        if result.converged:
            print(f"\n收敛! 入口压力={result.inlet_pressure_bara:.2f} bara, "
                  f"出口压力={result.outlet_pressure_calculated_bara:.2f} bara, "
                  f"打靶迭代={result.shooting_iterations}")

            # Save results
            if result.micro_dataframe is not None and not result.micro_dataframe.empty:
                out_dir = Path(args.output) if args.output else PROJECT_ROOT / "steady" / "output"
                out_dir.mkdir(parents=True, exist_ok=True)
                result.micro_dataframe.to_csv(out_dir / "pipeline_forward_plot_data.csv", index=False)
                print(f"沿线数据已保存至: {out_dir}")
        else:
            print(f"\n未收敛! 误差={result.outlet_pressure_error_bara:.4f} bara")
            # 严格相关式可能在部分试探点越出适用域。保留最佳有效试探的
            # 沿线结果，便于区分“打靶未找到根”和“全线计算本身失败”。
            if result.micro_dataframe is not None and not result.micro_dataframe.empty:
                out_dir = Path(args.output) if args.output else PROJECT_ROOT / "steady" / "output"
                out_dir.mkdir(parents=True, exist_ok=True)
                diagnostic_path = out_dir / "pipeline_forward_best_effort_data.csv"
                result.micro_dataframe.to_csv(diagnostic_path, index=False)
                print(f"最佳有效试探沿线数据已保存至: {diagnostic_path}")

    except Exception as exc:
        print(f"仿真失败: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
