
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


_NUM_RE = re.compile(r'^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?$')


def _is_number_token(token: str) -> bool:
    token = token.strip()
    if not token:
        return False
    token = token.replace(",", "")
    if token in {"-", "—", "–"}:
        return False
    return bool(_NUM_RE.match(token))


def _to_float(token: str) -> float:
    token = token.strip().replace(",", "")
    if token in {"", "-", "—", "–"}:
        return np.nan
    if not _is_number_token(token):
        raise ValueError(f"Cannot parse float from token: {token!r}")
    return float(token)


def _normalize_text(text: str) -> str:
    repl = {
        "\ufeff": "",
        "\u3000": " ",
        "¡ãC": "°C",
        "℃": "°C",
        "\r\n": "\n",
        "\r": "\n",
    }
    for a, b in repl.items():
        text = text.replace(a, b)
    return text


def _slugify(name: str) -> str:
    s = name.strip().lower()
    s = s.replace("%", "pct")
    s = re.sub(r'[^a-zA-Z0-9]+', '_', s)
    s = re.sub(r'_+', '_', s).strip('_')
    return s or "property"


def _is_probable_unit(text: str) -> bool:
    s = text.strip()
    if not s:
        return False
    if any(ch.isdigit() for ch in s):
        return False
    unit_tokens = [
        "bara", "bar", "pa", "kpa", "mpa", "psi",
        "°c", "degc", "k",
        "kg", "g", "mol", "kmol", "m3", "sm3", "rm3",
        "cp", "mpa.s", "kg/m3", "g/cm3", "j", "kj", "mj",
        "fraction", "wt%", "mol%", "-", "api", "sg"
    ]
    low = s.lower()
    return any(tok in low for tok in unit_tokens) or len(s) <= 12


def _line_has_any_number(tokens: List[str]) -> bool:
    return any(_is_number_token(t) for t in tokens)


def _tokenize_line(line: str) -> List[str]:
    line = line.strip("\n")
    if "\t" in line:
        tokens = [t.strip() for t in line.split("\t")]
    else:
        tokens = [t.strip() for t in re.split(r'\s{2,}|\t|,', line) if t.strip() != ""]
        if len(tokens) <= 1:
            tokens = [t.strip() for t in line.split() if t.strip()]
    return tokens


@dataclass
class PropertySeries:
    display_name: str
    code: str
    unit: str
    values: np.ndarray


class PVTParseError(Exception):
    pass


class PVTTable:
    def __init__(
        self,
        meta: Dict[str, str],
        properties: Dict[str, PropertySeries],
        dataframe: pd.DataFrame,
    ) -> None:
        self.meta = meta
        self.properties = properties
        self.dataframe = dataframe.copy()

        if "pressure" not in self.dataframe.columns:
            raise PVTParseError("解析结果中缺少 pressure 列。")
        if "temperature" not in self.dataframe.columns:
            raise PVTParseError("解析结果中缺少 temperature 列。")

        self._grid_cache: Dict[str, Dict[str, np.ndarray]] = {}
        # 规则网格与否只由表本身决定；稳态求解会对每个微元请求十余个
        # 物性，若每次都执行 pandas duplicated/unique 扫描，PVT 插值会
        # 反过来主导总耗时。初始化后缓存该判定。
        self._regular_grid_cache: bool | None = None

    @classmethod
    def from_file(cls, file_path: str | Path) -> "PVTTable":
        file_path = Path(file_path)
        if file_path.suffix.lower() == ".csv":
            try:
                return cls._from_wide_csv(file_path)
            except PVTParseError:
                raise
            except Exception:
                # Fall through to the legacy PVTsim text parser. Some PVTsim
                # exports use .csv as a plain text extension.
                pass

        text = cls._read_any_as_text(file_path)
        meta, properties = cls._parse_text(text)
        df = cls._build_dataframe(properties)
        return cls(meta=meta, properties=properties, dataframe=df)

    @classmethod
    def _from_wide_csv(cls, file_path: Path) -> "PVTTable":
        df = pd.read_csv(file_path, comment="#")
        df.columns = [str(c).strip() for c in df.columns]

        lower_to_original = {c.lower(): c for c in df.columns}
        if "pressure" not in lower_to_original or "temperature" not in lower_to_original:
            raise ValueError("Not a steady/flash wide PVT CSV.")

        df = df.rename(
            columns={
                lower_to_original["pressure"]: "pressure",
                lower_to_original["temperature"]: "temperature",
            }
        )

        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df[~(df["pressure"].isna() | df["temperature"].isna())].reset_index(drop=True)
        if df.empty:
            raise PVTParseError(f"CSV 文件 {file_path} 中没有有效 pressure/temperature 行。")
        df = cls._fill_wide_csv_grid_nans(df)

        unit_map = {
            "pressure": "bara",
            "temperature": "°C",
            "molepct_tot": "-",
            "molepct_vap": "-",
            "weightpct_tot": "-",
            "weightpct_vap": "-",
            "volumepct_tot": "-",
            "volumepct_vap": "-",
            "density_tot": "g/cm3",
            "density_vap": "g/cm3",
            "density_liq": "g/cm3",
            "z_factor_tot": "-",
            "z_factor_vap": "-",
            "mol_wgt_tot": "g/mol",
            "mol_wgt_vap": "g/mol",
            "enthalpy_tot": "J/mol",
            "enthalpy_vap": "J/mol",
            "heat_cap_cp_tot": "J/mol C",
            "heat_cap_cp_vap": "J/mol C",
            "jt_coeff_vap": "C/bara",
            "viscosity_vap": "cP",
            "viscosity_liq": "cP",
            "viscosity_tot": "cP",
            "therm_cond_vap": "mW/m C",
        }

        properties = {}
        for col in df.columns:
            code = _slugify(col)
            if col in {"pressure", "temperature"}:
                code = col
            properties[code] = PropertySeries(
                display_name=col,
                code=code,
                unit=unit_map.get(code, ""),
                values=df[col].to_numpy(dtype=float),
            )

        if "pressure" not in properties or "temperature" not in properties:
            raise PVTParseError("CSV 缺少 pressure 或 temperature 列。")

        return cls(
            meta={
                "Source": "flash wide CSV (thermo)",
                "Output Format": "PT Grid",
            },
            properties=properties,
            dataframe=df.rename(columns={c: _slugify(c) for c in df.columns if c not in {"pressure", "temperature"}}),
        )

    @staticmethod
    def _fill_wide_csv_grid_nans(df: pd.DataFrame) -> pd.DataFrame:
        """Fill sparse phase-property NaNs on a regular P/T grid."""
        pressures = np.sort(df["pressure"].dropna().unique())
        temperatures = np.sort(df["temperature"].dropna().unique())
        if len(pressures) < 2 or len(temperatures) < 2:
            return df

        out = df.copy()
        index_cols = ["pressure", "temperature"]
        for col in out.columns:
            if col in index_cols or not out[col].isna().any():
                continue
            grid = out.pivot_table(index="pressure", columns="temperature", values=col, aggfunc="mean")
            grid = grid.reindex(index=pressures, columns=temperatures)
            if grid.isna().all().all():
                continue
            grid = grid.interpolate(axis=0, limit_direction="both")
            grid = grid.interpolate(axis=1, limit_direction="both")
            grid = grid.ffill(axis=0).bfill(axis=0).ffill(axis=1).bfill(axis=1)
            filled = grid.stack().rename(col).reset_index()
            out = out.drop(columns=[col]).merge(filled, on=index_cols, how="left")
        return out

    @staticmethod
    def _read_any_as_text(file_path: Path) -> str:
        suffix = file_path.suffix.lower()

        if suffix in {".tab", ".txt", ".dat", ".csv"}:
            return _normalize_text(file_path.read_text(encoding="utf-8", errors="ignore"))

        if suffix in {".xls", ".xlsx"}:
            try:
                excel = pd.ExcelFile(file_path)
            except Exception as exc:
                raise PVTParseError(
                    f"无法读取 Excel 文件 {file_path.name}。"
                    f"如果是 .xls，请确认环境已安装可用读取引擎。原始错误: {exc}"
                ) from exc

            lines: List[str] = []
            for sheet_name in excel.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
                lines.append(f"# SHEET: {sheet_name}")
                for _, row in df.iterrows():
                    vals = []
                    for x in row.tolist():
                        if pd.isna(x):
                            vals.append("")
                        else:
                            vals.append(str(x))
                    lines.append("\t".join(vals))
            return _normalize_text("\n".join(lines))

        raise PVTParseError(f"不支持的文件类型: {file_path.suffix}")

    @classmethod
    def _parse_text(cls, text: str) -> Tuple[Dict[str, str], Dict[str, PropertySeries]]:
        lines = text.splitlines()
        meta: Dict[str, str] = {}

        begin_idx = None
        for i, raw in enumerate(lines):
            if "Begin Properties" in raw:
                begin_idx = i
                break

        if begin_idx is None:
            raise PVTParseError("未找到 'Begin Properties'，无法识别为 PVTsim 属性表。")

        header_lines = lines[:begin_idx]
        for i in range(len(header_lines) - 1):
            k = header_lines[i].strip()
            v = header_lines[i + 1].strip()
            if k and v and not _line_has_any_number(_tokenize_line(k)):
                if len(_tokenize_line(k)) <= 6 and len(_tokenize_line(v)) <= 20:
                    if k.lower() in {
                        "table format",
                        "number of phases",
                        "output format",
                        "code for missing property",
                        "rows per block",
                        "columns per block",
                        "blocks per fluid",
                        "number of fluids",
                    }:
                        meta[k] = v

        body = lines[begin_idx + 1:]

        cleaned_body = []
        for raw in body:
            s = raw.strip()
            if not s:
                cleaned_body.append(raw)
                continue
            if "EOS" in s or s.startswith("# SHEET:"):
                cleaned_body.append(raw)
                continue
            cleaned_body.append(raw)

        properties: Dict[str, PropertySeries] = {}

        i = 0
        n = len(cleaned_body)
        while i < n:
            line = cleaned_body[i].strip()

            if not line:
                i += 1
                continue

            if "EOS" in line or line.startswith("# SHEET:"):
                i += 1
                continue

            tokens = _tokenize_line(line)

            if not _line_has_any_number(tokens):
                prop_name = " ".join(tokens).strip()
                j = i + 1

                while j < n and not cleaned_body[j].strip():
                    j += 1
                if j >= n:
                    break

                unit_line = cleaned_body[j].strip()
                unit_tokens = _tokenize_line(unit_line)
                if _line_has_any_number(unit_tokens):
                    i += 1
                    continue

                unit = " ".join(unit_tokens).strip()
                if not _is_probable_unit(unit):
                    i += 1
                    continue

                value_rows: List[List[float]] = []
                k = j + 1
                while k < n:
                    cur = cleaned_body[k].strip()
                    if not cur:
                        look = k + 1
                        while look < n and not cleaned_body[look].strip():
                            look += 1
                        if look < n:
                            look_tokens = _tokenize_line(cleaned_body[look].strip())
                            if not _line_has_any_number(look_tokens):
                                break
                        k += 1
                        continue

                    cur_tokens = _tokenize_line(cur)

                    if not _line_has_any_number(cur_tokens):
                        break

                    row_vals = []
                    for tok in cur_tokens:
                        if _is_number_token(tok):
                            row_vals.append(_to_float(tok))

                    if row_vals:
                        value_rows.append(row_vals)

                    k += 1

                if value_rows:
                    flat = np.array([x for row in value_rows for x in row], dtype=float)
                    code = _slugify(prop_name)

                    base = code
                    idx = 2
                    while code in properties:
                        code = f"{base}_{idx}"
                        idx += 1

                    properties[code] = PropertySeries(
                        display_name=prop_name,
                        code=code,
                        unit=unit,
                        values=flat,
                    )
                    i = k
                    continue

            i += 1

        if not properties:
            raise PVTParseError("未解析出任何物性段。请检查文件格式。")

        return meta, cls._apply_aliases(properties)

    @staticmethod
    def _apply_aliases(properties: Dict[str, PropertySeries]) -> Dict[str, PropertySeries]:
        alias_map: Dict[str, str] = {}
        for code, prop in properties.items():
            name = prop.display_name.strip().lower()
            if name == "pressure":
                alias_map[code] = "pressure"
            elif name == "temperature":
                alias_map[code] = "temperature"

        new_props: Dict[str, PropertySeries] = {}
        for code, prop in properties.items():
            new_code = alias_map.get(code, code)
            if new_code in new_props and new_code not in {"pressure", "temperature"}:
                new_code = f"{new_code}_dup"
            new_props[new_code] = PropertySeries(
                display_name=prop.display_name,
                code=new_code,
                unit=prop.unit,
                values=prop.values,
            )
        return new_props

    @staticmethod
    def _build_dataframe(properties: Dict[str, PropertySeries]) -> pd.DataFrame:
        lengths = {code: len(prop.values) for code, prop in properties.items()}
        nonzero_lengths = {k: v for k, v in lengths.items() if v > 0}
        if not nonzero_lengths:
            raise PVTParseError("所有属性都没有可用数值。")

        freq: Dict[int, int] = {}
        for v in nonzero_lengths.values():
            freq[v] = freq.get(v, 0) + 1
        target_len = sorted(freq.items(), key=lambda kv: (-kv[1], -kv[0]))[0][0]

        data: Dict[str, np.ndarray] = {}
        dropped: List[Tuple[str, int]] = []

        for code, prop in properties.items():
            if len(prop.values) == target_len:
                data[code] = prop.values
            else:
                dropped.append((code, len(prop.values)))

        if "pressure" not in data:
            raise PVTParseError("未找到可用于对齐的 Pressure 数据。")
        if "temperature" not in data:
            raise PVTParseError("未找到可用于对齐的 Temperature 数据。")

        df = pd.DataFrame(data)
        df = df[~(df["pressure"].isna() | df["temperature"].isna())].reset_index(drop=True)

        if df.empty:
            raise PVTParseError("Pressure/Temperature 对齐后无有效状态点。")

        if dropped:
            msg = ", ".join([f"{k}({v})" for k, v in dropped[:8]])
            print(f"[WARN] 以下属性长度与主长度不一致，已跳过: {msg}")

        return df

    def list_properties(self) -> List[Tuple[str, str, str]]:
        rows = []
        for code, prop in self.properties.items():
            if code in self.dataframe.columns:
                rows.append((code, prop.display_name, prop.unit))
        return rows

    def property_info(self, prop: str) -> Dict[str, str]:
        code = self._resolve_prop(prop)
        p = self.properties[code]
        return {
            "code": p.code,
            "display_name": p.display_name,
            "unit": p.unit,
        }

    def _resolve_prop(self, prop: str) -> str:
        if prop in self.dataframe.columns:
            return prop

        low = prop.strip().lower()
        for code, ps in self.properties.items():
            if code in self.dataframe.columns:
                if ps.display_name.strip().lower() == low:
                    return code

        raise KeyError(f"找不到物性: {prop!r}")

    def is_regular_grid(self) -> bool:
        if self._regular_grid_cache is not None:
            return self._regular_grid_cache
        p = self.dataframe["pressure"].to_numpy()
        t = self.dataframe["temperature"].to_numpy()
        pairs = pd.DataFrame({"p": p, "t": t})

        if pairs.duplicated().any():
            self._regular_grid_cache = False
            return False

        p_unique = np.unique(p)
        t_unique = np.unique(t)

        self._regular_grid_cache = len(p_unique) * len(t_unique) == len(self.dataframe)
        return self._regular_grid_cache

    def _build_grid_for(self, prop: str) -> Dict[str, np.ndarray]:
        code = self._resolve_prop(prop)
        if code in self._grid_cache:
            return self._grid_cache[code]

        if not self.is_regular_grid():
            raise ValueError("当前数据不是完整规则网格，无法构建双线性插值网格。")

        sub = self.dataframe[["pressure", "temperature", code]].copy()
        pivot = sub.pivot(index="pressure", columns="temperature", values=code)
        pivot = pivot.sort_index().sort_index(axis=1)

        P = pivot.index.to_numpy(dtype=float)
        T = pivot.columns.to_numpy(dtype=float)
        Z = pivot.to_numpy(dtype=float)

        cache = {"P": P, "T": T, "Z": Z}
        self._grid_cache[code] = cache
        return cache

    def interpolate(
        self,
        pressure: float,
        temperature: float,
        prop: str,
        *,
        method: str = "auto",
        allow_extrapolation: bool = False,
        idw_k: int = 8,
        idw_power: float = 2.0,
    ) -> Dict[str, float | str]:
        code = self._resolve_prop(prop)

        if method == "auto":
            method = "bilinear" if self.is_regular_grid() else "idw"

        if method == "bilinear":
            value = self._interp_bilinear(
                pressure=pressure,
                temperature=temperature,
                prop=code,
                allow_extrapolation=allow_extrapolation,
            )
        elif method == "idw":
            value = self._interp_idw(
                pressure=pressure,
                temperature=temperature,
                prop=code,
                allow_extrapolation=allow_extrapolation,
                k=idw_k,
                power=idw_power,
            )
        else:
            raise ValueError(f"不支持的 method: {method}")

        info = self.property_info(code)
        return {
            "property": info["display_name"],
            "code": info["code"],
            "unit": info["unit"],
            "value": float(value),
            "pressure": float(pressure),
            "temperature": float(temperature),
            "method": method,
        }

    def interpolate_many(
        self,
        pressures: Iterable[float],
        temperatures: Iterable[float],
        prop: str,
        *,
        method: str = "auto",
        allow_extrapolation: bool = False,
        idw_k: int = 8,
        idw_power: float = 2.0,
    ) -> pd.DataFrame:
        rows = []
        for p, t in zip(pressures, temperatures):
            rows.append(
                self.interpolate(
                    pressure=float(p),
                    temperature=float(t),
                    prop=prop,
                    method=method,
                    allow_extrapolation=allow_extrapolation,
                    idw_k=idw_k,
                    idw_power=idw_power,
                )
            )
        return pd.DataFrame(rows)

    def _interp_bilinear(
        self,
        pressure: float,
        temperature: float,
        prop: str,
        *,
        allow_extrapolation: bool = False,
    ) -> float:
        grid = self._build_grid_for(prop)
        P = grid["P"]
        T = grid["T"]
        Z = grid["Z"]

        if (not allow_extrapolation) and (
            pressure < P.min() or pressure > P.max() or
            temperature < T.min() or temperature > T.max()
        ):
            raise ValueError(
                f"目标点超出数据范围: P in [{P.min()}, {P.max()}], "
                f"T in [{T.min()}, {T.max()}]"
            )

        p = float(np.clip(pressure, P.min(), P.max()))
        t = float(np.clip(temperature, T.min(), T.max()))

        i = np.searchsorted(P, p, side="right") - 1
        j = np.searchsorted(T, t, side="right") - 1

        i = max(0, min(i, len(P) - 2))
        j = max(0, min(j, len(T) - 2))

        p1, p2 = P[i], P[i + 1]
        t1, t2 = T[j], T[j + 1]

        z11 = Z[i, j]
        z12 = Z[i, j + 1]
        z21 = Z[i + 1, j]
        z22 = Z[i + 1, j + 1]

        if np.isclose(p1, p2) and np.isclose(t1, t2):
            return float(z11)
        if np.isclose(p1, p2):
            wt = 0.0 if np.isclose(t1, t2) else (t - t1) / (t2 - t1)
            return float(z11 * (1 - wt) + z12 * wt)
        if np.isclose(t1, t2):
            wp = (p - p1) / (p2 - p1)
            return float(z11 * (1 - wp) + z21 * wp)

        wp = (p - p1) / (p2 - p1)
        wt = (t - t1) / (t2 - t1)

        z_top = z11 * (1 - wt) + z12 * wt
        z_bot = z21 * (1 - wt) + z22 * wt
        z = z_top * (1 - wp) + z_bot * wp
        return float(z)

    def _interp_idw(
        self,
        pressure: float,
        temperature: float,
        prop: str,
        *,
        allow_extrapolation: bool = False,
        k: int = 8,
        power: float = 2.0,
    ) -> float:
        code = self._resolve_prop(prop)
        sub = self.dataframe[["pressure", "temperature", code]].dropna().copy()

        p = sub["pressure"].to_numpy(dtype=float)
        t = sub["temperature"].to_numpy(dtype=float)
        z = sub[code].to_numpy(dtype=float)

        if len(sub) == 0:
            raise ValueError(f"物性 {prop!r} 没有可用数据点。")

        pmin, pmax = np.min(p), np.max(p)
        tmin, tmax = np.min(t), np.max(t)
        if (not allow_extrapolation) and (
            pressure < pmin or pressure > pmax or temperature < tmin or temperature > tmax
        ):
            raise ValueError(
                f"目标点超出数据范围: P in [{pmin}, {pmax}], T in [{tmin}, {tmax}]"
            )

        dp = p - pressure
        dt = t - temperature
        d2 = dp * dp + dt * dt

        zero_idx = np.where(np.isclose(d2, 0.0))[0]
        if len(zero_idx) > 0:
            return float(z[zero_idx[0]])

        order = np.argsort(d2)
        k = max(1, min(k, len(order)))
        take = order[:k]

        dist = np.sqrt(d2[take])
        weights = 1.0 / np.power(dist, power)
        value = np.sum(weights * z[take]) / np.sum(weights)
        return float(value)

    def nearest_points(self, pressure: float, temperature: float, prop: str, k: int = 4) -> pd.DataFrame:
        code = self._resolve_prop(prop)
        sub = self.dataframe[["pressure", "temperature", code]].dropna().copy()
        sub["distance"] = np.sqrt((sub["pressure"] - pressure) ** 2 + (sub["temperature"] - temperature) ** 2)
        return sub.sort_values("distance").head(k).reset_index(drop=True)

    def summary(self) -> Dict[str, object]:
        return {
            "meta": self.meta,
            "n_points": int(len(self.dataframe)),
            "n_properties": int(len([c for c in self.dataframe.columns if c not in {"pressure", "temperature"}])),
            "regular_grid": self.is_regular_grid(),
            "pressure_range": (
                float(self.dataframe["pressure"].min()),
                float(self.dataframe["pressure"].max()),
            ),
            "temperature_range": (
                float(self.dataframe["temperature"].min()),
                float(self.dataframe["temperature"].max()),
            ),
        }


def _main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Parse and interpolate PVTsim PT table.")
    parser.add_argument("file", help="Path to .tab/.txt/.csv/.xls/.xlsx")
    parser.add_argument("--prop", help="Property code or display name")
    parser.add_argument("--pressure", type=float, help="Pressure value")
    parser.add_argument("--temperature", type=float, help="Temperature value")
    parser.add_argument("--method", default="auto", choices=["auto", "bilinear", "idw"])
    parser.add_argument("--allow-extrapolation", action="store_true")
    parser.add_argument("--list", action="store_true", help="List all parsed properties")
    args = parser.parse_args()

    table = PVTTable.from_file(args.file)

    if args.list:
        print("Parsed properties:")
        for code, name, unit in table.list_properties():
            print(f"  - {code:24s} | {name:20s} | {unit}")
        print("\nSummary:")
        print(json.dumps(table.summary(), ensure_ascii=False, indent=2))
        return

    if args.prop is None or args.pressure is None or args.temperature is None:
        print(json.dumps(table.summary(), ensure_ascii=False, indent=2))
        print("\n使用示例:")
        print(f"  python {Path(__file__).name} your_file.tab --list")
        print(f"  python {Path(__file__).name} your_file.tab --prop 'Mole% Vap' --pressure 35 --temperature 12")
        return

    result = table.interpolate(
        pressure=args.pressure,
        temperature=args.temperature,
        prop=args.prop,
        method=args.method,
        allow_extrapolation=args.allow_extrapolation,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _main()
