from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CASE_DIR = PROJECT_ROOT / "steady" / "算例" / "Pipeline_Cases_Output"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "steady_batch_outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="筛查四模型结果中持液率明显异常的算例，并可删除源算例和输出。"
    )
    parser.add_argument("--case-dir", type=Path, default=DEFAULT_CASE_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--low-threshold", type=float, default=0.01)
    parser.add_argument("--high-threshold", type=float, default=0.99)
    parser.add_argument(
        "--delete",
        action="store_true",
        help="执行删除；不提供时只生成审核报告。",
    )
    return parser.parse_args()


def inspect_case(case_output_dir: Path, low: float, high: float) -> dict[str, object] | None:
    model_files = sorted(case_output_dir.glob("*_*/pipeline_forward_plot_data.csv"))
    if len(model_files) != 4:
        return None

    model_stats: list[tuple[str, float, float, float]] = []
    invalid = False
    for csv_path in model_files:
        try:
            frame = pd.read_csv(csv_path, encoding="utf-8-sig")
            holdup = pd.to_numeric(frame["liquid_holdup"], errors="coerce")
            if holdup.empty or holdup.isna().any() or ((holdup < 0.0) | (holdup > 1.0)).any():
                invalid = True
            model_stats.append((
                csv_path.parent.name,
                float(holdup.min()),
                float(holdup.mean()),
                float(holdup.max()),
            ))
        except Exception:
            invalid = True

    averages = [stats[2] for stats in model_stats]
    if invalid:
        reason = "invalid_or_out_of_range_holdup"
    elif max(averages) <= low:
        reason = "all_models_extremely_low_holdup"
    elif min(averages) >= high:
        reason = "all_models_extremely_high_holdup"
    else:
        return None

    row: dict[str, object] = {
        "case": case_output_dir.name,
        "reason": reason,
        "low_threshold": low,
        "high_threshold": high,
    }
    for model_name, minimum, average, maximum in model_stats:
        row[f"{model_name}_min"] = minimum
        row[f"{model_name}_avg"] = average
        row[f"{model_name}_max"] = maximum
    return row


def remove_status_rows(output_root: Path, deleted_cases: set[str]) -> None:
    status_path = output_root / "batch_run_status.csv"
    if status_path.exists():
        status = pd.read_csv(status_path, encoding="utf-8-sig", dtype={"case_id": str})
        deleted_ids = {name.rsplit("_", 1)[-1] for name in deleted_cases}
        status = status.loc[~status["case_id"].str.zfill(4).isin(deleted_ids)]
        status.to_csv(status_path, index=False, encoding="utf-8-sig")

    master_path = output_root / "all_cases_model_comparison_summary.csv"
    if master_path.exists():
        master = pd.read_csv(master_path, encoding="utf-8-sig")
        master = master.loc[~master["case"].isin(deleted_cases)]
        master.to_csv(master_path, index=False, encoding="utf-8-sig")


def main() -> None:
    args = parse_args()
    case_dir = args.case_dir.resolve()
    output_root = args.output_root.resolve()
    if not 0.0 <= args.low_threshold < args.high_threshold <= 1.0:
        raise ValueError("阈值必须满足 0 <= low < high <= 1")

    rows = []
    for case_output_dir in sorted(output_root.glob("pipeline_two_phase_case_*")):
        result = inspect_case(case_output_dir, args.low_threshold, args.high_threshold)
        if result is not None:
            rows.append(result)

    report_path = output_root / "unphysical_cases_report.csv"
    pd.DataFrame(rows).to_csv(report_path, index=False, encoding="utf-8-sig")
    print(f"异常算例数量: {len(rows)}")
    print(f"审核报告: {report_path}")

    if not args.delete or not rows:
        return

    deleted_cases: set[str] = set()
    for row in rows:
        case_name = str(row["case"])
        source_path = case_dir / f"{case_name}.xlsx"
        output_path = output_root / case_name
        if source_path.exists():
            source_path.unlink()
        if output_path.exists():
            shutil.rmtree(output_path)
        deleted_cases.add(case_name)
        print(f"已删除: {case_name} ({row['reason']})")

    remove_status_rows(output_root, deleted_cases)
    print(f"已删除算例总数: {len(deleted_cases)}")


if __name__ == "__main__":
    main()
