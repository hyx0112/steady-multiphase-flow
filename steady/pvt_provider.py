from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import math

from steady.pvtsim_tab_interpolator import PVTTable


@dataclass
class PVTProperties:
    """
    某一压力、温度下的 PVT 物性。

    pressure_bara: bara
    temperature_C: ℃
    """

    pressure_bara: float
    temperature_C: float

    # 密度
    rho_tot: float  # kg/m3，总体系密度
    rho_gas: float  # kg/m3，气相密度
    rho_liquid: float  # kg/m3，液相密度

    # 黏度
    mu_tot: float  # Pa.s，总体系黏度；如果表里没有，则为 nan
    mu_gas: float  # Pa.s，气相黏度
    mu_liquid: float  # Pa.s，液相黏度
    mu_liquid_source: str  # 液相黏度来源说明

    # PVT 静态相平衡分数
    mole_frac_gas_eq: float  # 0~1
    mass_frac_gas_eq: float  # 0~1
    volume_frac_gas_eq: float  # 0~1

    # 状态方程参数
    z_factor_gas: float
    mol_weight_gas: float  # g/mol
    mol_weight_tot: float  # g/mol

    # 热力学性质
    cp_gas_molar: float  # J/mol/K
    cp_tot_molar: float  # J/mol/K

    cp_gas_mass: float  # J/kg/K
    cp_tot_mass: float  # J/kg/K
    cp_liquid_mass: float  # J/kg/K

    jt_coeff_gas: float  # K/Pa (已换算：原表为 ℃/bara)

    # 可选的 PVT 计算表面张力；为 None 时由管线全局设定提供。
    surface_tension_N_m: Optional[float] = None

    enthalpy_gas_molar: Optional[float] = None
    enthalpy_tot_molar: Optional[float] = None

    # 【新增】质量基焓值 (J/kg)，供管段热力学模型直接调用
    enthalpy_gas_mass: Optional[float] = None
    enthalpy_tot_mass: Optional[float] = None
    enthalpy_liquid_mass: Optional[float] = None


class PVTPropertyProvider:
    """
    PVT 物性适配层。

    当前这份 PVT 表已确认有：
        density_tot, density_vap, weightpct_vap, molepct_vap, volumepct_vap,
        viscosity_vap, heat_cap_cp_tot, heat_cap_cp_vap, mol_wgt_tot, mol_wgt_vap,
        z_factor_vap, jt_coeff_vap, enthalpy_tot, enthalpy_vap
    """

    def __init__(
            self,
            file_path: str,
            *,
            fallback_mu_liquid_Pa_s: float = 0.5e-3,
            fallback_rho_liquid_kg_m3: float = 650.0,
            fallback_cp_liquid_J_kgK: float = 2200.0,
            gas_mass_fraction_upper_limit: float = 0.999999,
            fraction_tolerance: float = 1.0e-6,
    ):
        self.file_path = file_path
        self.table = PVTTable.from_file(file_path)

        if fallback_mu_liquid_Pa_s <= 0:
            raise ValueError("fallback_mu_liquid_Pa_s 必须大于 0。")

        if fallback_rho_liquid_kg_m3 <= 0:
            raise ValueError("fallback_rho_liquid_kg_m3 必须大于 0。")

        if fallback_cp_liquid_J_kgK <= 0:
            raise ValueError("fallback_cp_liquid_J_kgK 必须大于 0。")

        if not 0.0 < gas_mass_fraction_upper_limit < 1.0:
            raise ValueError("gas_mass_fraction_upper_limit 必须在 0 和 1 之间。")

        if fraction_tolerance < 0:
            raise ValueError("fraction_tolerance 不能小于 0。")

        self.fallback_mu_liquid_Pa_s = fallback_mu_liquid_Pa_s
        self.fallback_rho_liquid_kg_m3 = fallback_rho_liquid_kg_m3
        self.fallback_cp_liquid_J_kgK = fallback_cp_liquid_J_kgK
        self.gas_mass_fraction_upper_limit = gas_mass_fraction_upper_limit
        self.fraction_tolerance = fraction_tolerance

    def summary(self):
        return self.table.summary()

    def list_properties(self):
        return self.table.list_properties()

    def _get(
            self,
            pressure_bara: float,
            temperature_C: float,
            prop: str,
            *,
            allow_extrapolation: bool = False,
    ) -> float:
        result = self.table.interpolate(
            pressure=pressure_bara,
            temperature=temperature_C,
            prop=prop,
            method="auto",
            allow_extrapolation=allow_extrapolation,
        )
        return float(result["value"])

    def _try_get(
            self,
            pressure_bara: float,
            temperature_C: float,
            prop: str,
            *,
            allow_extrapolation: bool = False,
    ) -> Optional[float]:
        try:
            value = self._get(
                pressure_bara,
                temperature_C,
                prop,
                allow_extrapolation=allow_extrapolation,
            )
            if value is None or math.isnan(value) or math.isinf(value):
                return None
            return value
        except Exception:
            return None

    def _clip_fraction(self, value: float, name: str) -> float:
        if math.isnan(value) or math.isinf(value):
            raise ValueError(f"{name} 为非法数值: {value}")

        if value < -self.fraction_tolerance or value > 1.0 + self.fraction_tolerance:
            raise ValueError(f"{name} 转换后的分率超出 [0, 1] 过多: {value}")

        if value < 0.0:
            return 0.0
        if value > 1.0:
            return 1.0
        return value

    def get_properties(
            self,
            pressure_bara: float,
            temperature_C: float,
            *,
            allow_extrapolation: bool = False,
    ) -> PVTProperties:

        # ============================================================
        # 1. 密度 (换算 kg/m3)
        # ============================================================
        rho_tot = self._get(
            pressure_bara,
            temperature_C,
            "density_tot",
            allow_extrapolation=allow_extrapolation,
        ) * 1000.0

        rho_gas = self._get(
            pressure_bara,
            temperature_C,
            "density_vap",
            allow_extrapolation=allow_extrapolation,
        ) * 1000.0

        if rho_tot <= 0:
            raise ValueError(f"density_tot 换算后 <= 0: {rho_tot}")
        if rho_gas <= 0:
            raise ValueError(f"density_vap 换算后 <= 0: {rho_gas}")

        # ============================================================
        # 2. 黏度 (换算 Pa.s)
        # ============================================================
        mu_gas = self._get(
            pressure_bara,
            temperature_C,
            "viscosity_vap",
            allow_extrapolation=allow_extrapolation,
        ) * 1.0e-3

        if mu_gas <= 0:
            raise ValueError(f"viscosity_vap 换算后 <= 0: {mu_gas}")

        mu_tot = math.nan

        # ============================================================
        # 3. 静态相分率
        # ============================================================
        mole_frac_gas_eq_raw = self._get(pressure_bara, temperature_C, "molepct_vap",
                                         allow_extrapolation=allow_extrapolation) / 100.0
        mass_frac_gas_eq_raw = self._get(pressure_bara, temperature_C, "weightpct_vap",
                                         allow_extrapolation=allow_extrapolation) / 100.0
        volume_frac_gas_eq_raw = self._get(pressure_bara, temperature_C, "volumepct_vap",
                                           allow_extrapolation=allow_extrapolation) / 100.0

        mole_frac_gas_eq = self._clip_fraction(mole_frac_gas_eq_raw, "molepct_vap")
        mass_frac_gas_eq_unclipped = self._clip_fraction(mass_frac_gas_eq_raw, "weightpct_vap")
        volume_frac_gas_eq = self._clip_fraction(volume_frac_gas_eq_raw, "volumepct_vap")

        x = min(mass_frac_gas_eq_unclipped, self.gas_mass_fraction_upper_limit)
        gas_only_or_nearly_gas_only = (mass_frac_gas_eq_unclipped >= self.gas_mass_fraction_upper_limit)

        # ============================================================
        # 4. Z 因子和分子量
        # ============================================================
        z_factor_gas = self._get(pressure_bara, temperature_C, "z_factor_vap", allow_extrapolation=allow_extrapolation)
        mol_weight_gas = self._get(pressure_bara, temperature_C, "mol_wgt_vap", allow_extrapolation=allow_extrapolation)
        mol_weight_tot = self._get(pressure_bara, temperature_C, "mol_wgt_tot", allow_extrapolation=allow_extrapolation)

        if mol_weight_gas <= 0:
            raise ValueError("mol_wgt_vap 小于等于 0，无法换算气相质量基热容。")
        if mol_weight_tot <= 0:
            raise ValueError("mol_wgt_tot 小于等于 0，无法换算总体系质量基热容。")

        # ============================================================
        # 5. 热容 (换算 J/kg/K)
        # ============================================================
        cp_gas_molar = self._get(pressure_bara, temperature_C, "heat_cap_cp_vap",
                                 allow_extrapolation=allow_extrapolation)
        cp_tot_molar = self._get(pressure_bara, temperature_C, "heat_cap_cp_tot",
                                 allow_extrapolation=allow_extrapolation)

        cp_gas_mass = cp_gas_molar / (mol_weight_gas / 1000.0)
        cp_tot_mass = cp_tot_molar / (mol_weight_tot / 1000.0)

        if cp_gas_mass <= 0:
            raise ValueError(f"cp_gas_mass <= 0: {cp_gas_mass}")
        if cp_tot_mass <= 0:
            raise ValueError(f"cp_tot_mass <= 0: {cp_tot_mass}")

        # ============================================================
        # 6. JT 系数 (换算 K/Pa)  【此处已修改】
        # ============================================================
        jt_coeff_gas_raw = self._get(
            pressure_bara,
            temperature_C,
            "jt_coeff_vap",
            allow_extrapolation=allow_extrapolation,
        )
        jt_coeff_gas = jt_coeff_gas_raw / 1.0e5  # ℃/bara 转换为 K/Pa

        surface_tension = self._try_get(
            pressure_bara,
            temperature_C,
            "surface_tension",
            allow_extrapolation=allow_extrapolation,
        )
        if surface_tension is not None and surface_tension <= 0.0:
            surface_tension = None

        # ============================================================
        # 7. 液相密度
        # ============================================================
        if gas_only_or_nearly_gas_only:
            rho_liquid = self.fallback_rho_liquid_kg_m3
        else:
            rho_liq_raw = self._try_get(
                pressure_bara,
                temperature_C,
                "density_liq",
                allow_extrapolation=allow_extrapolation,
            )
            if rho_liq_raw is not None and rho_liq_raw > 0:
                rho_liquid = rho_liq_raw * 1000.0
            else:
                density_denominator = 1.0 / rho_tot - x / rho_gas
                if density_denominator <= 0:
                    # PVT 表插值不一致时，分母可能 <= 0（如 rho_gas > rho_tot 或
                    # 近临界区数值误差），此时回退到 fallback 值，不中断计算。
                    import warnings
                    warnings.warn(
                        f"液相密度反算分母 <= 0 (P={pressure_bara:.2f} bara, T={temperature_C:.2f} °C, "
                        f"rho_tot={rho_tot:.2f}, rho_gas={rho_gas:.2f}, x={x:.6f}, "
                        f"denom={density_denominator:.6e})，使用 fallback 值 {self.fallback_rho_liquid_kg_m3} kg/m³"
                    )
                    rho_liquid = self.fallback_rho_liquid_kg_m3
                else:
                    rho_liquid = (1.0 - x) / density_denominator
                    if rho_liquid <= 0:
                        import warnings
                        warnings.warn(
                            f"反算 rho_liquid <= 0 (P={pressure_bara:.2f} bara, T={temperature_C:.2f} °C, "
                            f"rho_liquid={rho_liquid:.2f})，使用 fallback 值 {self.fallback_rho_liquid_kg_m3} kg/m³"
                        )
                        rho_liquid = self.fallback_rho_liquid_kg_m3

        # ============================================================
        # 8. 液相黏度
        # ============================================================
        mu_liquid = self.fallback_mu_liquid_Pa_s
        mu_liquid_source = "fallback: 默认液相黏度"

        mu_liq_raw = self._try_get(pressure_bara, temperature_C, "viscosity_liq",
                                   allow_extrapolation=allow_extrapolation)

        if mu_liq_raw is not None:
            mu_liquid = mu_liq_raw * 1.0e-3
            mu_liquid_source = "PVT: viscosity_liq"
        else:
            mu_tot_raw = self._try_get(pressure_bara, temperature_C, "viscosity_tot",
                                       allow_extrapolation=allow_extrapolation)
            if mu_tot_raw is not None:
                mu_tot = mu_tot_raw * 1.0e-3
                mu_liquid = (mu_tot - x * mu_gas) / (1.0 - x)
                mu_liquid_source = "calculated from viscosity_tot and viscosity_vap"

        if mu_liquid <= 0:
            raise ValueError("液相黏度小于等于 0。")

        # ============================================================
        # 9. 液相热容
        # ============================================================
        if gas_only_or_nearly_gas_only:
            cp_liquid_mass = self.fallback_cp_liquid_J_kgK
        else:
            cp_liquid_mass = (cp_tot_mass - x * cp_gas_mass) / (1.0 - x)
            if cp_liquid_mass <= 0:
                raise ValueError("液相热容反算失败：cp_liquid_mass 小于等于 0。")

        # ============================================================
        # 10. 焓 (新增换算 J/kg) 【此处已修改并新增】
        # ============================================================
        enthalpy_gas_molar = self._try_get(
            pressure_bara,
            temperature_C,
            "enthalpy_vap",
            allow_extrapolation=allow_extrapolation,
        )

        enthalpy_tot_molar = self._try_get(
            pressure_bara,
            temperature_C,
            "enthalpy_tot",
            allow_extrapolation=allow_extrapolation,
        )

        # 换算气相质量基焓值
        enthalpy_gas_mass = None
        if enthalpy_gas_molar is not None:
            enthalpy_gas_mass = enthalpy_gas_molar / (mol_weight_gas / 1000.0)

        # 换算整体质量基焓值
        enthalpy_tot_mass = None
        if enthalpy_tot_molar is not None:
            enthalpy_tot_mass = enthalpy_tot_molar / (mol_weight_tot / 1000.0)

        # 反算液相质量基焓值 (如果下游求解器需要用到两相做功)
        enthalpy_liquid_mass = None
        if enthalpy_gas_mass is not None and enthalpy_tot_mass is not None:
            if gas_only_or_nearly_gas_only:
                enthalpy_liquid_mass = 0.0  # 全气相时设为 0 或 fallback
            else:
                enthalpy_liquid_mass = (enthalpy_tot_mass - x * enthalpy_gas_mass) / (1.0 - x)

        return PVTProperties(
            pressure_bara=pressure_bara,
            temperature_C=temperature_C,

            rho_tot=rho_tot,
            rho_gas=rho_gas,
            rho_liquid=rho_liquid,

            mu_tot=mu_tot,
            mu_gas=mu_gas,
            mu_liquid=mu_liquid,
            mu_liquid_source=mu_liquid_source,

            mole_frac_gas_eq=mole_frac_gas_eq,
            mass_frac_gas_eq=x,
            volume_frac_gas_eq=volume_frac_gas_eq,

            z_factor_gas=z_factor_gas,
            mol_weight_gas=mol_weight_gas,
            mol_weight_tot=mol_weight_tot,

            cp_gas_molar=cp_gas_molar,
            cp_tot_molar=cp_tot_molar,

            cp_gas_mass=cp_gas_mass,
            cp_tot_mass=cp_tot_mass,
            cp_liquid_mass=cp_liquid_mass,

            jt_coeff_gas=jt_coeff_gas,
            surface_tension_N_m=surface_tension,

            enthalpy_gas_molar=enthalpy_gas_molar,
            enthalpy_tot_molar=enthalpy_tot_molar,

            # 返回质量基焓值
            enthalpy_gas_mass=enthalpy_gas_mass,
            enthalpy_tot_mass=enthalpy_tot_mass,
            enthalpy_liquid_mass=enthalpy_liquid_mass,
        )
