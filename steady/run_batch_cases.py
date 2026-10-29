from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CASE_DIR = PROJECT_ROOT / "steady" / "算例" / "Pipeline_Cases_Output"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "steady_batch_outputs"

MODELS = {
    "bb": "beggs_brill",
    "mb": "mukherjee_brill",
    "df": "drift_flux",
    "xiao": "xiao",
}

COMPARE_MODEL_NAMES = {
    "bb": "Beggs-Brill",
    "mb": "Mukherjee-Brill",
    "df": "Drift Flux",
    "xiao": "Xiao",
}


@dataclass
class RunStatus:
    case_id: str
    case_file: Path
    model_key: str
    output_dir: Path
    return_code: int
    elapsed_s: float
    skipped: bool
    message: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="批量运行 steady 算例目录下的四模型计算，并生成每个算例的对比图。"
    )
    parser.add_argument("--case-dir", type=Path, default=DEFAULT_CASE_DIR, help="算例 xlsx 目录")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="统一输出根目录")
    parser.add_argument("--start", type=int, default=None, help="起始算例编号，例如 9")
    parser.add_argument("--end", type=int, default=None, help="结束算例编号，例如 20")
    parser.add_argument("--limit", type=int, default=None, help="最多运行多少个算例")
    parser.add_argument("--resume", action="store_true", help="依据 batch_run_status.csv 从最后记录算例的下一个编号继续")
    parser.add_argument("--models", nargs="+", choices=sorted(MODELS), default=list(MODELS), help="要运行的模型")
    parser.add_argument("--rerun", action="store_true", help="重新运行已存在结果的模型")
    parser.add_argument("--flash-force", action="store_true", help="强制重新生成 PVT 表")
    parser.add_argument("--pmin", type=float, default=None, help="PVT 表压力下限 bara")
    parser.add_argument("--pmax", type=float, default=None, help="PVT 表压力上限 bara")
    parser.add_argument("--pn", type=int, default=8, help="PVT 压力网格数")
    parser.add_argument("--tmin", type=float, default=None, help="PVT 表温度下限 C")
    parser.add_argument("--tmax", type=float, default=None, help="PVT 表温度上限 C")
    parser.add_argument("--tn", type=int, default=8, help="PVT 温度网格数")
    parser.add_argument("--no-compare", action="store_true", help="只运行模型，不生成对比图")
    parser.add_argument("--keep-partial-compare", action="store_true", help="模型缺失时仍尝试生成已有模型对比汇总")
    return parser.parse_args()


def case_number(path: Path) -> int:
    return int(path.stem.rsplit("_", 1)[-1])


def discover_cases(case_dir: Path, start: int | None, end: int | None, limit: int | None) -> list[Path]:
    if not case_dir.is_dir():
        raise FileNotFoundError(f"算例目录不存在: {case_dir}")

    cases = sorted(case_dir.glob("pipeline_two_phase_case_*.xlsx"), key=case_number)
    selected = []
    for case_file in cases:
        number = case_number(case_file)
        if start is not None and number < start:
            continue
        if end is not None and number > end:
            continue
        selected.append(case_file)
        if limit is not None and len(selected) >= limit:
            break
    return selected


def output_dir_for(output_root: Path, case_file: Path, model_key: str) -> Path:
    return output_root / case_file.stem / f"{model_key}_{MODELS[model_key]}"


def has_complete_model_output(output_dir: Path) -> bool:
    return (output_dir / "pipeline_forward_plot_data.csv").exists()


def build_run_command(args: argparse.Namespace, case_file: Path, model_key: str, output_dir: Path) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "steady.run",
        "--pvt-source",
        "flash",
        "--pipe",
        str(case_file),
        "--model",
        MODELS[model_key],
        "--no-debug",
        "--no-plots",
        "--output",
        str(output_dir.relative_to(PROJECT_ROOT)),
        "--flash-pn",
        str(args.pn),
        "--flash-tn",
        str(args.tn),
    ]

    optional_pairs = [
        ("--flash-pmin", args.pmin),
        ("--flash-pmax", args.pmax),
        ("--flash-tmin", args.tmin),
        ("--flash-tmax", args.tmax),
    ]
    for flag, value in optional_pairs:
        if value is not None:
            command.extend([flag, str(value)])
    if args.flash_force:
        command.append("--flash-force")
    return command


def run_model(args: argparse.Namespace, case_file: Path, model_key: str) -> RunStatus:
    output_dir = output_dir_for(args.output_root, case_file, model_key)
    case_id = case_file.stem.rsplit("_", 1)[-1]
    if has_complete_model_output(output_dir) and not args.rerun:
        return RunStatus(case_id, case_file, model_key, output_dir, 0, 0.0, True, "skipped_existing")

    if args.rerun and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    command = build_run_command(args, case_file, model_key, output_dir)
    started = time.time()
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    elapsed = time.time() - started
    log_path = output_dir / "run.log"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path.write_text(completed.stdout, encoding="utf-8")

    message = "ok" if completed.returncode == 0 else f"failed_return_code_{completed.returncode}"
    return RunStatus(case_id, case_file, model_key, output_dir, completed.returncode, elapsed, False, message)


def write_run_log(output_root: Path, statuses: list[RunStatus]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    log_path = output_root / "batch_run_status.csv"
    with log_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "case_id",
                "case_file",
                "model",
                "output_dir",
                "return_code",
                "elapsed_s",
                "skipped",
                "message",
            ],
        )
        writer.writeheader()
        for status in statuses:
            writer.writerow({
                "case_id": status.case_id,
                "case_file": str(status.case_file),
                "model": status.model_key,
                "output_dir": str(status.output_dir),
                "return_code": status.return_code,
                "elapsed_s": round(status.elapsed_s, 3),
                "skipped": status.skipped,
                "message": status.message,
            })


def load_existing_statuses(output_root: Path) -> list[dict[str, str]]:
    log_path = output_root / "batch_run_status.csv"
    if not log_path.exists():
        return []
    with log_path.open("r", newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def write_merged_run_log(
    output_root: Path,
    existing_rows: list[dict[str, str]],
    statuses: list[RunStatus],
) -> None:
    new_rows = []
    for status in statuses:
        new_rows.append({
            "case_id": status.case_id,
            "case_file": str(status.case_file),
            "model": status.model_key,
            "output_dir": str(status.output_dir),
            "return_code": str(status.return_code),
            "elapsed_s": str(round(status.elapsed_s, 3)),
            "skipped": str(status.skipped),
            "message": status.message,
        })

    merged: dict[tuple[str, str], dict[str, str]] = {}
    for row in [*existing_rows, *new_rows]:
        merged[(row["case_id"], row["model"])] = row

    output_root.mkdir(parents=True, exist_ok=True)
    log_path = output_root / "batch_run_status.csv"
    fieldnames = [
        "case_id",
        "case_file",
        "model",
        "output_dir",
        "return_code",
        "elapsed_s",
        "skipped",
        "message",
    ]
    with log_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for key in sorted(merged, key=lambda item: (int(item[0]), item[1])):
            writer.writerow(merged[key])


def run_compare(case_file: Path, output_root: Path, models: list[str], keep_partial: bool) -> Path | None:
    available = {
        model_key: output_dir_for(output_root, case_file, model_key) / "pipeline_forward_plot_data.csv"
        for model_key in models
        if has_complete_model_output(output_dir_for(output_root, case_file, model_key))
    }
    required = ["bb", "mb", "df", "xiao"]
    if not keep_partial and any(model_key not in available for model_key in required):
        return None
    if any(model_key not in available for model_key in required):
        return None

    compare_dir = output_root / case_file.stem / "compare"
    command = [
        sys.executable,
        str(PROJECT_ROOT / "steady" / "compare_case_models.py"),
        "--case-name",
        case_file.stem,
        "--output-dir",
        str(compare_dir),
        "--bb",
        str(available["bb"]),
        "--mb",
        str(available["mb"]),
        "--df",
        str(available["df"]),
        "--xiao",
        str(available["xiao"]),
    ]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    return compare_dir


def collect_master_summary(output_root: Path) -> None:
    frames = []
    for summary_path in sorted(output_root.glob("pipeline_two_phase_case_*/compare/model_comparison_summary.csv")):
        if not summary_path.exists():
            continue
        df = pd.read_csv(summary_path, encoding="utf-8-sig")
        df.insert(0, "case", summary_path.parents[1].name)
        frames.append(df)
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(
            output_root / "all_cases_model_comparison_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )


def main() -> None:
    args = parse_args()
    args.case_dir = args.case_dir.resolve()
    args.output_root = args.output_root.resolve()
    existing_status_rows = load_existing_statuses(args.output_root)

    if args.resume and args.start is None and existing_status_rows:
        args.start = max(int(row["case_id"]) for row in existing_status_rows) + 1
        print(f"依据状态日志续跑，起始算例编号: {args.start:04d}")

    case_files = discover_cases(args.case_dir, args.start, args.end, args.limit)
    if not case_files:
        print("未找到匹配的算例文件。")
        return

    print(f"算例目录: {args.case_dir}")
    print(f"输出目录: {args.output_root}")
    print(f"算例数量: {len(case_files)}")
    print(f"模型: {', '.join(args.models)}")

    statuses: list[RunStatus] = []
    total_started = time.time()
    for index, case_file in enumerate(case_files, start=1):
        print(f"\n[{index}/{len(case_files)}] {case_file.name}")
        case_failed = False
        for model_key in args.models:
            if case_failed:
                skipped_dir = output_dir_for(args.output_root, case_file, model_key)
                status = RunStatus(
                    case_id=case_file.stem.rsplit("_", 1)[-1],
                    case_file=case_file,
                    model_key=model_key,
                    output_dir=skipped_dir,
                    return_code=1,
                    elapsed_s=0.0,
                    skipped=True,
                    message="skipped_after_case_failure",
                )
                statuses.append(status)
                print(f"  {model_key:4s} SKIP      0.0s  skipped_after_case_failure")
                continue

            status = run_model(args, case_file, model_key)
            statuses.append(status)
            state = "SKIP" if status.skipped else ("OK" if status.return_code == 0 else "FAIL")
            print(f"  {model_key:4s} {state:4s} {status.elapsed_s:8.1f}s  {status.output_dir}")
            if status.return_code != 0:
                case_failed = True
                print("  case SKIP  当前算例模型计算失败，跳过该算例剩余模型并继续下一个算例")

        if not args.no_compare:
            compare_dir = run_compare(case_file, args.output_root, args.models, args.keep_partial_compare)
            if compare_dir is not None:
                print(f"  compare OK    {compare_dir}")
            else:
                print("  compare SKIP  四个模型结果不完整")

        write_merged_run_log(args.output_root, existing_status_rows, statuses)
        collect_master_summary(args.output_root)

    elapsed = time.time() - total_started
    print(f"\n批量完成，用时 {elapsed:.1f} s")
    print(f"运行状态: {args.output_root / 'batch_run_status.csv'}")
    print(f"汇总结果: {args.output_root / 'all_cases_model_comparison_summary.csv'}")


if __name__ == "__main__":
    main()
