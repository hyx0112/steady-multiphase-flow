# steady/config.py — PVT 与管道几何参数准备模块
from __future__ import annotations

from typing import List, Tuple
import math

import pandas as pd

from steady.steady_types import PipeSection


def read_pipe_sections_from_excel(file_path: str, sheet_name=0) -> List[PipeSection]:
    """从 Excel 读取管线参数表"""
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    aliases = {
        "出口节点": ["出口节点"],
        "管段长度": ["管段长度", "管段长度_m"],
        "管段高程": ["管段高程", "管段高程_m"],
        "划分段数_m": ["划分段数_m", "划分段数"],
        "具体各段长度_m": ["具体各段长度_m"],
        "粗糙度_mm": ["粗糙度_mm"],
        "总传热系数W/m2/℃": ["总传热系数W/m2/℃"],
        "质量流量kg/s": ["质量流量kg/s"],
        "管道直径_mm": ["管道直径_mm"],
        "环境温度℃": ["环境温度℃"],
    }
    col = {}
    for canonical, names in aliases.items():
        col[canonical] = next((name for name in names if name in df.columns), None)
    missing = [name for name, actual in col.items() if actual is None]
    if missing:
        raise ValueError(f"Excel 缺少列: {missing}")

    sections = []
    for idx, row in df.iterrows():
        sections.append(PipeSection(
            section_no=idx + 1,
            outlet_node=int(row[col["出口节点"]]),
            section_length_m=float(row[col["管段长度"]]),
            elevation_change_m=float(row[col["管段高程"]]),
            n_subsegments=int(row[col["划分段数_m"]]),
            sub_length_m=float(row[col["具体各段长度_m"]]),
            roughness_m=float(row[col["粗糙度_mm"]]) / 1000.0,
            overall_heat_transfer=float(row[col["总传热系数W/m2/℃"]]),
            mass_flow_kg_s=float(row[col["质量流量kg/s"]]),
            diameter_m=float(row[col["管道直径_mm"]]) / 1000.0,
            env_temp_K=float(row[col["环境温度℃"]]) + 273.15,
        ))
    return sections


def make_pipe_sections_from_table() -> List[PipeSection]:
    """根据内置的管线参数生成大管段列表"""
    raw = [
        (1, 40000.0, 20.0, 400, 100.0, 0.5348, 6.5, 2.192, 535.2, 3.0),
        (2, 80000.0, -120.0, 800, 100.0, 0.5348, 6.5, 2.192, 535.2, 3.0),
        (3, 40000.0, 220.0, 400, 100.0, 0.5348, 6.5, 2.192, 535.2, 3.0),
        (4, 40000.0, -90.0, 400, 100.0, 0.5348, 6.5, 2.192, 535.2, 3.0),
    ]
    sections = []
    for idx, (outlet_node, length, elev, n_sub, sub_len, rough_mm, u, mfr, diam_mm, env_c) in enumerate(raw, 1):
        sections.append(PipeSection(
            section_no=idx,
            outlet_node=int(outlet_node),
            section_length_m=float(length),
            elevation_change_m=float(elev),
            n_subsegments=int(n_sub),
            sub_length_m=float(sub_len),
            roughness_m=float(rough_mm) / 1000.0,
            overall_heat_transfer=float(u),
            mass_flow_kg_s=float(mfr),
            diameter_m=float(diam_mm) / 1000.0,
            env_temp_K=float(env_c) + 273.15,
        ))
    return sections


def build_section_positions(sections: List[PipeSection]) -> dict:
    """为每个大管段建立水平距离和高程位置"""
    positions = {}
    x = 0.0
    z = 0.0
    for sec in sections:
        x_end = x + sec.section_length_m
        z_end = z + sec.elevation_change_m
        positions[sec.section_no] = {
            "x_start_m": x, "x_end_m": x_end,
            "elevation_start_m": z, "elevation_end_m": z_end,
        }
        x, z = x_end, z_end
    return positions


def load_pvt_provider(file_path: str):
    """加载 PVT 属性提供器"""
    from steady.pvt_provider import PVTPropertyProvider
    return PVTPropertyProvider(file_path)


def read_pipe_case_from_excel(file_path: str) -> tuple[list[PipeSection], dict]:
    """
    读取 pipe_case_N.xlsx 格式的管线算例文件。

    与 read_pipe_sections_from_excel 的区别：
        - 列名为中文（入口节点 / 出口节点 / 管段长度 等）
        - 包含"组分"和"边界条件"两个额外 sheet

    返回
    ----
    sections:
        PipeSection 列表。

    case_info:
        {"name": str, "composition": DataFrame, "inlet_T_C": float, "outlet_P_MPa": float}
    """
    import pandas as pd

    xl = pd.ExcelFile(file_path)
    sheet_names = xl.sheet_names

    # 管道 sheet
    pipe_sheet = sheet_names[0]
    df = pd.read_excel(file_path, sheet_name=pipe_sheet)

    # 柔性列名匹配：用 aliases 逐个匹配
    aliases = {
        "inlet_node": ["入口节点", "入口节点", "inlet_node"],
        "outlet_node": ["出口节点", "出口节点", "outlet_node"],
        "length": ["管段长度_m", "管段长度", "length", "section_length"],
        "elevation": ["管段高程_m", "管段高程", "elevation"],
        "n_sub": ["划分段数_m", "划分段数", "n_sub"],
        "sub_len": ["具体各段长度_m", "具体各段长度", "sub_len"],
        "roughness_mm": ["粗糙度_mm", "粗糙度", "roughness_mm"],
        "u": ["总传热系数W/m2/℃", "总传热系数W/m2/�", "u"],
        "mfr": ["质量流量kg/s", "质量流量", "mass_flow"],
        "diam_mm": ["管道直径_mm", "管道直径", "diam_mm"],
        "env_c": ["环境温度℃", "环境温度", "env_c"],
    }

    col_rename = {}
    for new_name, candidates in aliases.items():
        for c in candidates:
            if c in df.columns:
                col_rename[c] = new_name
                break

    missing = [new for new, cands in aliases.items() if new not in col_rename.values()]
    if missing:
        raise ValueError(f"Excel 缺少列，无法映射: {missing}。实际列名: {list(df.columns)}")

    df = df.rename(columns=col_rename)

    sections = []
    for idx, row in df.iterrows():
        sections.append(PipeSection(
            section_no=idx + 1,
            outlet_node=int(row["outlet_node"]),
            section_length_m=float(row["length"]),
            elevation_change_m=float(row["elevation"]),
            n_subsegments=int(row["n_sub"]),
            sub_length_m=float(row["sub_len"]),
            roughness_m=float(row["roughness_mm"]) / 1000.0,
            overall_heat_transfer=float(row["u"]),
            mass_flow_kg_s=float(row["mfr"]),
            diameter_m=float(row["diam_mm"]) / 1000.0,
            env_temp_K=float(row["env_c"]) + 273.15,
        ))

    # 组分 sheet
    comp_df = pd.read_excel(file_path, sheet_name=sheet_names[1]) if len(sheet_names) > 1 else None

    # 边界条件 sheet
    bc_df = pd.read_excel(file_path, sheet_name=sheet_names[2]) if len(sheet_names) > 2 else None
    if bc_df is not None:
        # 柔性匹配列名
        bc_aliases = {
            "inlet_T": ["入口温度 (℃)", "入口温度℃", "入口温度", "inlet_temperature"],
            "outlet_P": ["出口压力 (MPa)", "出口压力MPa", "出口压力", "outlet_pressure"],
        }
        bc_cols = {}
        for key, candidates in bc_aliases.items():
            for c in candidates:
                if c in bc_df.columns:
                    bc_cols[key] = c
                    break

        if "inlet_T" in bc_cols:
            inlet_T_C = float(bc_df.iloc[0][bc_cols["inlet_T"]])
        else:
            # Fallback: try positional (old format)
            inlet_T_C = float(bc_df.iloc[0, 0])

        if "outlet_P" in bc_cols:
            outlet_P_MPa = float(bc_df.iloc[0][bc_cols["outlet_P"]])
        else:
            outlet_P_MPa = float(bc_df.iloc[0, 1])
    else:
        raise ValueError(f"算例文件 {file_path} 缺少边界条件 sheet")

    case_info = {
        "composition": comp_df,
        "inlet_temperature_C": inlet_T_C,
        "outlet_pressure_MPa": outlet_P_MPa,
    }
    return sections, case_info


def prepare_pipeline_data(
    pvt_path: str,
    pipe_excel_path: str,
    sheet_name: str = "管道",
):
    """一站式准备：加载 PVT 提供器 + 读取管道参数"""
    pvt = load_pvt_provider(pvt_path)
    sections = read_pipe_sections_from_excel(pipe_excel_path, sheet_name=sheet_name)
    return pvt, sections
