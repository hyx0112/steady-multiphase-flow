"""NeqSim-to-OLGA PVT table adapter for the detailed HGC compositions.

This module is deliberately an *offline table generator*.  It does not put a
JVM or a foreign flash call inside the five-equation transient time loop.
Instead, it converts the project's detailed-component CSV into a NeqSim PR
fluid, produces an OLGA keyword-format ``.tab`` file, and records an SI audit
at representative P-T points.

Unit contract
-------------
Project / audit:
    pressure [Pa], temperature [K], composition [mol/mol], density [kg/m3].
NeqSim API boundary:
    pressure [bara], temperature [K].
OLGA PT-table text:
    pressure [Pa], temperature [degC] (written by NeqSim).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PA_PER_BARA = 1.0e5
KELVIN_OFFSET_C = 273.15
TWO_PHASE_COLUMNS = (
    "PT", "TM", "ROG", "ROHL", "DROGDP", "DROHLDP", "DROGDT", "DROHLDT",
    "RS", "VISG", "VISHL", "CPG", "CPHL", "HG", "HHL", "TCG", "TCHL",
    "SIGGHL", "SEG", "SEHL",
)


class NeqSimSetupError(RuntimeError):
    """Raised when the optional NeqSim/JVM backend is unavailable."""


@dataclass(frozen=True)
class Composition:
    """Detailed composition in the project-standard mol/mol representation."""

    name: str
    components: tuple[str, ...]
    neqsim_components: tuple[str, ...]
    mole_fractions: tuple[float, ...]


@dataclass(frozen=True)
class FlashAuditPoint:
    """A NeqSim TP-flash result expressed only in project SI units."""

    pressure_pa: float
    temperature_k: float
    neqsim_pressure_bara: float
    temperature_c: float
    phase_count: int
    gas_molar_fraction: float
    oil_molar_fraction: float
    gas_density_kg_m3: float | None
    oil_density_kg_m3: float | None
    mixture_density_kg_m3: float
    local_pr_vapor_fraction: float | None
    local_pr_gas_density_kg_m3: float | None
    local_pr_liquid_density_kg_m3: float | None


@dataclass(frozen=True)
class TableAudit:
    """Structural audit of an OLGA keyword-format PVT table."""

    path: str
    table_label_present: bool
    phase_two_present: bool
    columns: tuple[str, ...]
    point_count: int
    pressure_count: int
    temperature_count: int
    pressure_min_pa: float
    pressure_max_pa: float
    temperature_min_c: float
    temperature_max_c: float
    expected_point_count: int
    finite_points: bool


_EXPLICIT_COMPONENTS = {
    "Nitrogen": "nitrogen",
    "Carbon dioxide": "CO2",
    "Methane": "methane",
    "Ethane": "ethane",
    "Propane": "propane",
    "i-Butane": "i-butane",
    "n-Butane": "n-butane",
    "i-Pentane": "i-pentane",
    "n-Pentane": "n-pentane",
    "n-Hexane": "n-hexane",
    "n-Heptane": "n-heptane",
    "n-Octane": "n-octane",
    "n-Nonane": "n-nonane",
}

_LOCAL_PR_COMPONENTS = {
    "Nitrogen": "nitrogen",
    "Carbon dioxide": "carbon-dioxide",
    "Methane": "methane",
    "Ethane": "ethane",
    "Propane": "propane",
    "i-Butane": "i-butane",
    "n-Butane": "n-butane",
    "i-Pentane": "i-pentane",
    "n-Pentane": "n-pentane",
    "n-Hexane": "n-hexane",
    "n-Heptane": "n-heptane",
    "n-Octane": "n-octane",
    "n-Nonane": "n-nonane",
    "n-Decane": "n-decane",
    "n-Undecane": "n-undecane",
    "n-Dodecane": "n-dodecane",
    "n-Tridecane": "n-tridecane",
    "n-Tetradecane": "n-tetradecane",
    "n-Pentadecane": "n-pentadecane",
    "n-Hexadecane": "n-hexadecane",
    "n-Heptadecane": "n-heptadecane",
    "n-Octadecane": "n-octadecane",
    "n-Nonadecane": "n-nonadecane",
    "n-Eicosane": "n-eicosane",
    "n-Heneicosane": "n-heneicosane",
    "n-Docosane": "n-docosane",
    "n-Tricosane": "n-tricosane",
    "n-Tetracosane": "n-tetracosane",
    "n-Pentacosane": "n-pentacosane",
    "n-Hexacosane": "n-hexacosane",
    "n-Heptacosane": "n-heptacosane",
    "n-Octacosane": "n-octacosane",
    "n-Nonacosane": "n-nonacosane",
    "n-Triacontane": "n-triacontane",
}


def _candidate_jdk_roots() -> Iterable[Path]:
    java = shutil.which("java")
    if java:
        java_path = Path(java).resolve()
        yield java_path.parent.parent
    for root in (
        Path("C:/Program Files/Java/jdk-21"),
        Path("C:/Program Files/Java/jdk-17"),
        Path("C:/Program Files/Eclipse Adoptium/jdk-21"),
    ):
        yield root


def prepare_java_home() -> Path:
    """Make NeqSim's JPype bridge find an installed JDK without global edits."""
    configured = os.environ.get("JAVA_HOME")
    if configured:
        root = Path(configured)
        if (root / "bin" / "server" / "jvm.dll").is_file():
            return root

    for root in _candidate_jdk_roots():
        if (root / "bin" / "server" / "jvm.dll").is_file():
            os.environ["JAVA_HOME"] = str(root)
            return root

    raise NeqSimSetupError(
        "未找到可用 JDK。请安装 64 位 JDK 17+，或把 JAVA_HOME 指向含 "
        "bin\\server\\jvm.dll 的 JDK 根目录。"
    )


def _import_neqsim() -> Any:
    prepare_java_home()
    try:
        from neqsim import jneqsim
    except ImportError as exc:
        raise NeqSimSetupError(
            "未安装 NeqSim。请使用 .venv_neqsim\\Scripts\\python.exe -m pip "
            "install -r requirements-neqsim.txt。"
        ) from exc
    return jneqsim


def _neqsim_name(component: str, formula: str) -> str:
    if component in _EXPLICIT_COMPONENTS:
        return _EXPLICIT_COMPONENTS[component]
    match = re.fullmatch(r"C(1[0-9]|2[0-9]|30)", formula.strip(), flags=re.IGNORECASE)
    if match:
        return f"nC{int(match.group(1))}"
    raise ValueError(f"没有定义 NeqSim 详细组分映射：{component!r} ({formula!r})")


def load_composition(csv_path: Path, composition_column: str) -> Composition:
    """Read a project HGC CSV and return normalized detailed molar fractions."""
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"组分文件为空：{csv_path}")
    required = {"component", "formula", composition_column}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"组分文件缺少列：{sorted(missing)}")

    components = tuple(row["component"].strip() for row in rows)
    neqsim_components = tuple(_neqsim_name(row["component"].strip(), row["formula"]) for row in rows)
    mol_pct = np.asarray([float(row[composition_column]) for row in rows], dtype=float)
    total = float(mol_pct.sum())
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError(f"{composition_column} 的摩尔百分数总和非法：{total}")
    if not np.isclose(total, 100.0, rtol=0.0, atol=1.0e-6):
        raise ValueError(f"{composition_column} 的摩尔百分数必须为 100，当前为 {total:.9g}")
    return Composition(
        name=composition_column.removesuffix("_mol_pct"),
        components=components,
        neqsim_components=neqsim_components,
        mole_fractions=tuple((mol_pct / 100.0).tolist()),
    )


def build_pr_fluid(
    composition: Composition,
    pressure_pa: float,
    temperature_k: float,
) -> Any:
    """Build a NeqSim Peng-Robinson system from the normalized composition."""
    if pressure_pa <= 0.0 or temperature_k <= 0.0:
        raise ValueError("压力和温度必须为正值（Pa、K）。")
    jneqsim = _import_neqsim()
    fluid = jneqsim.thermo.system.SystemPrEos(temperature_k, pressure_pa / PA_PER_BARA)
    for name, amount in zip(composition.neqsim_components, composition.mole_fractions):
        fluid.addComponent(name, amount)
    fluid.createDatabase(True)
    fluid.setMixingRule("classic")
    fluid.setMultiPhaseCheck(True)
    return fluid


def flash_audit_point(composition: Composition, pressure_pa: float, temperature_k: float) -> FlashAuditPoint:
    """Run a NeqSim TP flash and return phase quantities in project SI units."""
    fluid = build_pr_fluid(composition, pressure_pa, temperature_k)
    jneqsim = _import_neqsim()
    operations = jneqsim.thermodynamicoperations.ThermodynamicOperations(fluid)
    operations.TPflash()
    fluid.initProperties()

    phase_count = int(fluid.getNumberOfPhases())
    gas_beta = oil_beta = 0.0
    gas_density = oil_density = None
    for index in range(phase_count):
        phase = fluid.getPhase(index)
        phase_name = str(phase.getPhaseTypeName()).lower()
        beta = float(phase.getBeta())
        density = float(phase.getDensity("kg/m3"))
        if phase_name == "gas":
            gas_beta, gas_density = beta, density
        elif phase_name in {"oil", "aqueous"}:
            oil_beta, oil_density = beta, density

    from .peng_robinson import PengRobinsonFlashProvider

    local_composition = {
        _LOCAL_PR_COMPONENTS[name]: fraction
        for name, fraction in zip(composition.components, composition.mole_fractions)
    }
    local = PengRobinsonFlashProvider(local_composition).flash_point(pressure_pa, temperature_k)
    local_vapor = float(local.vapor_fraction_molar)
    local_gas_density = float(local.rho_g)
    local_liquid_density = float(local.rho_l)

    return FlashAuditPoint(
        pressure_pa=float(pressure_pa),
        temperature_k=float(temperature_k),
        neqsim_pressure_bara=float(pressure_pa / PA_PER_BARA),
        temperature_c=float(temperature_k - KELVIN_OFFSET_C),
        phase_count=phase_count,
        gas_molar_fraction=gas_beta,
        oil_molar_fraction=oil_beta,
        gas_density_kg_m3=gas_density,
        oil_density_kg_m3=oil_density,
        mixture_density_kg_m3=float(fluid.getDensity("kg/m3")),
        local_pr_vapor_fraction=local_vapor,
        local_pr_gas_density_kg_m3=local_gas_density,
        local_pr_liquid_density_kg_m3=local_liquid_density,
    )


def _validate_two_phase_table_point(
    values: tuple[float, ...],
    *,
    pressure_bara: float,
    temperature_k: float,
    phase_count: int,
) -> None:
    """Reject a P-T point that cannot safely represent a gas--oil OLGA row."""
    if phase_count < 2:
        raise ValueError(
            "两相 OLGA 表网格出现单相状态："
            f"P={pressure_bara:.6g} bara, T={temperature_k - KELVIN_OFFSET_C:.6g} °C。"
            "请缩小到气--烃液两相包络内，或使用支持单相区的独立物性策略。"
        )
    if not all(math.isfinite(value) for value in values):
        raise ValueError(
            "NeqSim 在两相物性表网格产生 NaN/Infinity："
            f"P={pressure_bara:.6g} bara, T={temperature_k - KELVIN_OFFSET_C:.6g} °C。"
        )
    gas_density, oil_density = values[2], values[3]
    gas_cp, oil_cp = values[11], values[12]
    gas_enthalpy, oil_enthalpy = values[13], values[14]
    surface_tension = values[17]
    if not (0.01 < gas_density < 1000.0 and 100.0 < oil_density < 2000.0):
        raise ValueError("NeqSim 生成了非物理的气/烃液密度，已拒绝写入 OLGA 表。")
    if not (100.0 < gas_cp < 2.0e4 and 100.0 < oil_cp < 2.0e4):
        raise ValueError(
            "NeqSim 生成了非物理的定压比热，已拒绝写入 OLGA 表："
            f"P={pressure_bara:.6g} bara, T={temperature_k - KELVIN_OFFSET_C:.6g} °C。"
            "当前 NeqSim 3.16 已知 nC21+ 重端数据库会触发此问题；"
            "请改用已验证的 C20 及以下详细组分，或继续使用 Multiflash/PVTSim 表。"
        )
    if not (abs(gas_enthalpy) < 1.0e7 and abs(oil_enthalpy) < 1.0e7 and 0.0 < surface_tension < 1.0):
        raise ValueError("NeqSim 生成了非物理的焓或气液界面张力，已拒绝写入 OLGA 表。")


def generate_olga_tab(
    composition: Composition,
    output_tab: Path,
    *,
    pressure_min_bara: float,
    pressure_max_bara: float,
    pressure_count: int,
    temperature_min_c: float,
    temperature_max_c: float,
    temperature_count: int,
    fluid_label: str,
) -> None:
    """Create a two-phase OLGA keyword table using NeqSim flash properties.

    NeqSim 3.16 contains a two-phase writer, but its public writer targets a
    hard-coded developer path.  This compact writer retains the same NeqSim
    property definitions while using the supplied project output path.
    """
    if pressure_count < 2 or temperature_count < 2:
        raise ValueError("OLGA PT 表至少需要 2 个压力点和 2 个温度点。")
    if not (pressure_min_bara > 0.0 and pressure_max_bara > pressure_min_bara):
        raise ValueError("压力范围必须满足 0 < Pmin < Pmax（bara）。")
    if temperature_max_c <= temperature_min_c:
        raise ValueError("温度范围必须满足 Tmin < Tmax（°C）。")
    if not fluid_label.strip():
        raise ValueError("OLGA 流体标签不能为空。")
    output_tab.parent.mkdir(parents=True, exist_ok=True)
    fluid = build_pr_fluid(composition, pressure_min_bara * PA_PER_BARA, temperature_min_c + KELVIN_OFFSET_C)
    jneqsim = _import_neqsim()
    operations = jneqsim.thermodynamicoperations.ThermodynamicOperations(fluid)
    pressures_bara = np.linspace(pressure_min_bara, pressure_max_bara, pressure_count)
    temperatures_k = np.linspace(temperature_min_c + KELVIN_OFFSET_C, temperature_max_c + KELVIN_OFFSET_C, temperature_count)
    points: list[tuple[float, ...]] = []
    for pressure_bara in pressures_bara:
        for temperature_k in temperatures_k:
            fluid.setPressure(float(pressure_bara))
            fluid.setTemperature(float(temperature_k))
            operations.TPflash()
            fluid.init(3)
            fluid.initPhysicalProperties()
            gas, liquid = fluid.getPhase(0), fluid.getPhase(1)
            point = (
                pressure_bara * PA_PER_BARA,
                temperature_k - KELVIN_OFFSET_C,
                gas.getPhysicalProperties().getDensity(),
                liquid.getPhysicalProperties().getDensity(),
                gas.getdrhodP() / PA_PER_BARA,
                liquid.getdrhodP() / PA_PER_BARA,
                gas.getdrhodT(),
                liquid.getdrhodT(),
                gas.getBeta() * gas.getMolarMass() / fluid.getMolarMass(),
                gas.getPhysicalProperties().getViscosity(),
                liquid.getPhysicalProperties().getViscosity(),
                gas.getCp() / gas.getNumberOfMolesInPhase() / gas.getMolarMass(),
                liquid.getCp() / liquid.getNumberOfMolesInPhase() / liquid.getMolarMass(),
                gas.getEnthalpy() / gas.getNumberOfMolesInPhase() / gas.getMolarMass(),
                liquid.getEnthalpy() / liquid.getNumberOfMolesInPhase() / liquid.getMolarMass(),
                gas.getPhysicalProperties().getConductivity(),
                liquid.getPhysicalProperties().getConductivity(),
                fluid.getInterphaseProperties().getSurfaceTension(0, 1),
                gas.getEntropy() / gas.getNumberOfMolesInPhase() / gas.getMolarMass(),
                liquid.getEntropy() / liquid.getNumberOfMolesInPhase() / liquid.getMolarMass(),
            )
            _validate_two_phase_table_point(
                point,
                pressure_bara=float(pressure_bara),
                temperature_k=float(temperature_k),
                phase_count=int(fluid.getNumberOfPhases()),
            )
            points.append(point)

    fluid.setPressure(101325.0 / PA_PER_BARA)
    fluid.setTemperature(288.70556)
    operations.TPflash()
    fluid.initPhysicalProperties()
    gas, liquid = fluid.getPhase(0), fluid.getPhase(1)
    components = [fluid.getPhase(0).getComponent(i) for i in range(fluid.getPhase(0).getNumberOfComponents())]

    def values(items: Iterable[float]) -> str:
        return ",".join(f"{float(item):.12g}" for item in items)

    component_names = ",".join(f'"{item.getComponentName()}"' for item in components)
    molar_fractions = values(item.getz() for item in components)
    molecular_weights = values(item.getMolarMass() * 1000.0 for item in components)
    normal_densities = values(item.getNormalLiquidDensity() for item in components)
    gor = gas.getTotalVolume() / liquid.getTotalVolume()
    lines = [
        f'PVTTABLE LABEL = "{fluid_label.strip()}", PHASE = TWO,\\',
        'EOS = "NeqSim SystemPrEos; classic mixing rule",\\',
        f"COMPONENTS = ({component_names}),\\",
        f"MOLES = ({molar_fractions}),\\",
        f"MOLWEIGHT = ({molecular_weights}) g/mol,\\",
        f"DENSITY = ({normal_densities}) g/cm3,\\",
        "STDPRESSURE = 101325 Pa,\\",
        "STDTEMPERATURE = 15.55556 C,\\",
        f"GOR = {float(gor):.12g} Sm3/Sm3,\\",
        f"GLR = {float(gor):.12g} Sm3/Sm3,\\",
        "WC = -999,\\",
        f"STDGASDENSITY = {float(gas.getPhysicalProperties().getDensity()):.12g} kg/m3,\\",
        f"STDOILDENSITY = {float(liquid.getPhysicalProperties().getDensity()):.12g} kg/m3,\\",
        "STDWATDENSITY = -999 kg/m3,\\",
        "MESHTYPE = STANDARD,\\",
        f"PRESSURE = ({values(pressures_bara * PA_PER_BARA)}) Pa,\\",
        f"TEMPERATURE = ({values(temperatures_k - KELVIN_OFFSET_C)}) C,\\",
        f"COLUMNS = ({','.join(TWO_PHASE_COLUMNS)})",
    ]
    lines.extend(f"PVTTABLE POINT = ({values(point)})" for point in points)
    output_tab.write_text("\n".join(lines) + "\n", encoding="utf-8")


def inspect_olga_tab(output_tab: Path, expected_point_count: int) -> TableAudit:
    """Verify only format and units; it is not a substitute for OLGA execution."""
    text = output_tab.read_text(encoding="utf-8")
    header = "PVTTABLE LABEL" in text
    phase_two = "PHASE = TWO" in text
    columns_match = re.search(r"COLUMNS\s*=\s*\(([^)]*)\)", text)
    columns = tuple(item.strip() for item in columns_match.group(1).split(",")) if columns_match else ()
    point_blocks = re.findall(r"PVTTABLE POINT\s*=\s*\(([^)]*)\)", text)
    rows: list[list[float]] = []
    finite = True
    for block in point_blocks:
        try:
            values = [float(item.strip()) for item in block.split(",")]
        except ValueError:
            finite = False
            continue
        finite = finite and all(math.isfinite(value) for value in values)
        rows.append(values)
    if not rows:
        raise ValueError(f"OLGA 表未找到 PVTTABLE POINT：{output_tab}")
    pressure = np.asarray([row[0] for row in rows], dtype=float)
    temperature = np.asarray([row[1] for row in rows], dtype=float)
    return TableAudit(
        path=str(output_tab),
        table_label_present=header,
        phase_two_present=phase_two,
        columns=columns,
        point_count=len(rows),
        pressure_count=len(np.unique(pressure)),
        temperature_count=len(np.unique(temperature)),
        pressure_min_pa=float(np.min(pressure)),
        pressure_max_pa=float(np.max(pressure)),
        temperature_min_c=float(np.min(temperature)),
        temperature_max_c=float(np.max(temperature)),
        expected_point_count=int(expected_point_count),
        finite_points=finite,
    )


def _write_audit_csv(path: Path, audits: Iterable[FlashAuditPoint]) -> None:
    records = [asdict(point) for point in audits]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FlashAuditPoint.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(records)


def run_workflow(args: argparse.Namespace) -> dict[str, Any]:
    composition = load_composition(Path(args.composition_csv), args.composition_column)
    output_tab = Path(args.output_tab)
    flash_points = (
        (args.pressure_min_bara * PA_PER_BARA, args.temperature_min_c + KELVIN_OFFSET_C),
        (50.0 * PA_PER_BARA, 25.0 + KELVIN_OFFSET_C),
        (args.pressure_max_bara * PA_PER_BARA, args.temperature_max_c + KELVIN_OFFSET_C),
    )
    flashes = [flash_audit_point(composition, pressure, temperature) for pressure, temperature in flash_points]
    generate_olga_tab(
        composition,
        output_tab,
        pressure_min_bara=args.pressure_min_bara,
        pressure_max_bara=args.pressure_max_bara,
        pressure_count=args.pressure_count,
        temperature_min_c=args.temperature_min_c,
        temperature_max_c=args.temperature_max_c,
        temperature_count=args.temperature_count,
        fluid_label=args.fluid_label,
    )
    table = inspect_olga_tab(output_tab, args.pressure_count * args.temperature_count)
    if not table.finite_points:
        raise ValueError("OLGA 表包含 NaN 或 Infinity，已拒绝作为有效输出。请检查 P-T 网格或物性计算。")
    if not table.table_label_present or not table.phase_two_present:
        raise ValueError("OLGA 表缺少 PVTTABLE LABEL 或 PHASE = TWO 声明。")
    if table.columns != TWO_PHASE_COLUMNS:
        raise ValueError(f"OLGA 两相表列不符合项目约定：{table.columns}")
    if table.point_count != table.expected_point_count:
        raise ValueError(
            f"OLGA 表网格点数不完整：{table.point_count}，预期 {table.expected_point_count}。"
        )
    audit_csv = output_tab.with_name(f"{output_tab.stem}_flash_audit.csv")
    _write_audit_csv(audit_csv, flashes)
    manifest = {
        "backend": "NeqSim 3.16.0 / SystemPrEos / classic mixing rule",
        "olga_fluid_label": args.fluid_label,
        "java_home": str(prepare_java_home()),
        "composition": asdict(composition),
        "unit_contract": {
            "project_pressure": "Pa",
            "project_temperature": "K",
            "composition": "mol/mol",
            "neqsim_pressure": "bara",
            "olga_table_pressure": "Pa",
            "olga_table_temperature": "degC",
        },
        "flash_audit": [asdict(point) for point in flashes],
        "table_audit": asdict(table),
        "flash_audit_csv": str(audit_csv),
    }
    manifest_path = output_tab.with_name(f"{output_tab.stem}_manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="详细组分 NeqSim → OLGA .tab 批量物性接口")
    parser.add_argument("--composition-csv", type=Path, required=True, help="HGC 详细组分 CSV")
    parser.add_argument("--composition-column", required=True, help="如 HGC_A_mol_pct")
    parser.add_argument("--output-tab", type=Path, required=True, help="输出 OLGA .tab 文件")
    parser.add_argument("--pressure-min-bara", type=float, default=1.0)
    parser.add_argument("--pressure-max-bara", type=float, default=120.0)
    parser.add_argument("--pressure-count", type=int, default=25)
    parser.add_argument("--temperature-min-c", type=float, default=0.0)
    parser.add_argument("--temperature-max-c", type=float, default=90.0)
    parser.add_argument("--temperature-count", type=int, default=25)
    parser.add_argument("--fluid-label", default="FLUID-A", help="必须与 GENKEY 中 BRANCH FLUID 一致。")
    return parser


def main() -> None:
    manifest = run_workflow(build_parser().parse_args())
    table = manifest["table_audit"]
    print(f"NeqSim OLGA 表已生成：{table['path']}")
    print(
        "表格检查："
        f"{table['point_count']} 点，P={table['pressure_min_pa']:.6g}–{table['pressure_max_pa']:.6g} Pa，"
        f"T={table['temperature_min_c']:.6g}–{table['temperature_max_c']:.6g} °C"
    )


if __name__ == "__main__":
    main()
