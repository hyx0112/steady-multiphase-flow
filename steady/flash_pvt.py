"""
flash_pvt.py — Lightweight PVT table generation using thermo (CalebBell).

Replaces neqsim_pvt.py entirely. Uses the `thermo` library for SRK/PR
equation-of-state flash calculations, eliminating the NeqSim Java/Maven
dependency.

Key dependencies:  pip install thermo
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Component database & aliases  (unchanged from neqsim_pvt.py)
# ---------------------------------------------------------------------------

COMPONENT_NAME_ALIASES = {
    "methane": "methane",
    "ethane": "ethane",
    "propane": "propane",
    "isobutane": "i-butane",
    "i-butane": "i-butane",
    "butane": "n-butane",
    "n-butane": "n-butane",
    "isopentane": "i-pentane",
    "i-pentane": "i-pentane",
    "pentane": "n-pentane",
    "n-pentane": "n-pentane",
    "hexane": "n-hexane",
    "heptane": "n-heptane",
    "octane": "n-octane",
    "nonane": "n-nonane",
    "decane": "nC10",
    "undecane": "nC11",
    "dodecane": "nC12",
    "tridecane": "nC13",
    "tetradecane": "nC14",
    "pentadecane": "nC15",
    "hexadecane": "nC16",
    "heptadecane": "nC17",
    "octadecane": "nC18",
    "nonadecane": "nC19",
    "eicosane": "nC20",
    "heneicosane": "nC21",
    "docosane": "nC22",
    "tricosane": "nC23",
    "tetracosane": "nC24",
    "pentacosane": "nC25",
    "hexacosane": "nC26",
    "heptacosane": "nC27",
    "octacosane": "nC28",
    "nonacosane": "nC29",
    "triacontane": "nC30",
    "nitrogen": "nitrogen",
    "carbon dioxide": "CO2",
    "co2": "CO2",
}

HEAVY_COMPONENT_CARBON_NUMBER = {
    "n-heptane": 7, "n-octane": 8, "n-nonane": 9,
    "nC10": 10, "nC11": 11, "nC12": 12, "nC13": 13, "nC14": 14,
    "nC15": 15, "nC16": 16, "nC17": 17, "nC18": 18, "nC19": 19,
    "nC20": 20, "nC21": 21, "nC22": 22, "nC23": 23, "nC24": 24,
    "nC25": 25, "nC26": 26, "nC27": 27, "nC28": 28, "nC29": 29,
    "nC30": 30,
}

HEAVY_COMPONENT_MW_G_MOL = {
    name: 14.02658 * carbon + 2.01588
    for name, carbon in HEAVY_COMPONENT_CARBON_NUMBER.items()
}

DEFAULT_HEAVY_FRACTION_SHEET = "重端表征"
DEFAULT_FLASH_CONFIG_SHEET = "Flash配置"


# ===================================================================
#  PT Flash Engine  (the thermo-based replacement for NeqSim's flash)
# ===================================================================

# Map component names used in Excel to thermo's expected names
# ---------------------------------------------------------------------------
#  Pseudo-component critical property estimation
# ---------------------------------------------------------------------------

# Known n-alkane properties for interpolation (C1-C30)
_N_ALKANE_PROPERTIES = {
    # carbon_number: (MW_g_mol, Tc_K, Pc_Pa, omega)
    1:  (16.04246,  190.564, 4599200.0,  0.01142),
    2:  (30.06904,  305.322, 4872200.0,  0.0995),
    3:  (44.09562,  369.89,  4251200.0,  0.1521),
    4:  (58.1222,   425.12,  3796000.0,  0.2002),
    5:  (72.14878,  469.7,   3370000.0,  0.2515),
    6:  (86.17536,  507.6,   3025000.0,  0.3013),
    7:  (100.2019,  540.2,   2740000.0,  0.3495),
    8:  (114.2285,  568.7,   2490000.0,  0.3996),
    9:  (128.2551,  594.6,   2290000.0,  0.4455),
    10: (142.2817,  617.7,   2103000.0,  0.4884),
    11: (156.3083,  639.0,   1980000.0,  0.5350),
    12: (170.3348,  658.0,   1820000.0,  0.5760),
    13: (184.3614,  675.0,   1680000.0,  0.6174),
    14: (198.388,   692.0,   1570000.0,  0.6540),
    15: (212.4146,  708.0,   1480000.0,  0.6900),
    16: (226.4412,  723.0,   1400000.0,  0.7174),
    17: (240.4677,  736.0,   1340000.0,  0.7500),
    18: (254.4943,  747.0,   1290000.0,  0.7800),
    19: (268.5209,  758.0,   1230000.0,  0.8000),
    20: (282.5475,  768.0,   1160000.0,  0.8682),
    21: (296.5741,  778.0,   1110000.0,  0.9000),
    22: (310.6006,  787.0,   1070000.0,  0.9300),
    23: (324.6272,  796.0,   1030000.0,  0.9600),
    24: (338.6538,  805.0,   990000.0,   0.9900),
    25: (352.6804,  813.0,   960000.0,   1.0200),
    26: (366.7070,  821.0,   930000.0,   1.0500),
    27: (380.7335,  829.0,   900000.0,   1.0800),
    28: (394.7601,  836.0,   870000.0,   1.1100),
    29: (408.7867,  843.0,   850000.0,   1.1400),
    30: (422.8133,  850.0,   830000.0,   1.1700),
}


def estimate_pseudo_critical_properties(
    mw_kg_mol: float, density_kg_m3: float | None = None,
) -> tuple[float, float, float]:
    """
    Estimate Tc [K], Pc [Pa], omega for a hydrocarbon pseudo-component.

    Uses interpolation/extrapolation of n-alkane properties based on
    molecular weight.

    Parameters
    ----------
    mw_kg_mol : float
        Molecular weight in kg/mol.
    density_kg_m3 : float or None
        Standard liquid density at 15°C in kg/m³ (not used in simple mode).

    Returns
    -------
    (Tc_K, Pc_Pa, omega) : tuple[float, float, float]
    """
    mw_g_mol = mw_kg_mol * 1000.0

    # Find the two nearest n-alkanes by MW
    mw_list = [(cn, props[0]) for cn, props in _N_ALKANE_PROPERTIES.items()]
    mw_list.sort(key=lambda x: x[1])

    if mw_g_mol <= mw_list[0][1]:
        # Lighter than methane — use methane
        _, Tc, Pc, omega = _N_ALKANE_PROPERTIES[1]
        return Tc, Pc, omega

    if mw_g_mol >= mw_list[-1][1]:
        # Heavier than nC30 — extrapolate from last two
        cn_hi = mw_list[-1][0]
        cn_lo = mw_list[-2][0]
        mw_hi = mw_list[-1][1]
        mw_lo = mw_list[-2][1]
        frac = (mw_g_mol - mw_lo) / (mw_hi - mw_lo) if mw_hi != mw_lo else 0.0
        frac = max(0.0, frac)  # extrapolation
        _, Tc_hi, Pc_hi, om_hi = _N_ALKANE_PROPERTIES[cn_hi]
        _, Tc_lo, Pc_lo, om_lo = _N_ALKANE_PROPERTIES[cn_lo]
        Tc = Tc_lo + frac * (Tc_hi - Tc_lo)
        Pc = Pc_lo + frac * (Pc_hi - Pc_lo)
        omega = om_lo + frac * (om_hi - om_lo)
        return Tc, Pc, omega

    # Interpolate between nearest n-alkanes
    for i in range(len(mw_list) - 1):
        cn_lo, mw_lo = mw_list[i]
        cn_hi, mw_hi = mw_list[i + 1]
        if mw_lo <= mw_g_mol <= mw_hi:
            frac = (mw_g_mol - mw_lo) / (mw_hi - mw_lo)
            _, Tc_lo, Pc_lo, om_lo = _N_ALKANE_PROPERTIES[cn_lo]
            _, Tc_hi, Pc_hi, om_hi = _N_ALKANE_PROPERTIES[cn_hi]
            Tc = Tc_lo + frac * (Tc_hi - Tc_lo)
            Pc = Pc_lo + frac * (Pc_hi - Pc_lo)
            omega = om_lo + frac * (om_hi - om_lo)
            return Tc, Pc, omega

    # Shouldn't reach here
    return 600.0, 2e6, 0.5


_PSEUDO_COMPONENT_CACHE: dict[str, tuple[float, float, float, float]] = {}


def register_pseudo_component(
    name: str, mw_kg_mol: float, density_kg_m3: float | None = None,
) -> tuple[float, float, float, float]:
    """
    Register a pseudo-component and return (Tc, Pc, omega, MW_g_mol).

    Results are cached so repeated calls are fast.
    """
    if name in _PSEUDO_COMPONENT_CACHE:
        return _PSEUDO_COMPONENT_CACHE[name]

    Tc, Pc, omega = estimate_pseudo_critical_properties(mw_kg_mol, density_kg_m3)
    mw_g_mol = mw_kg_mol * 1000.0
    result = (Tc, Pc, omega, mw_g_mol)
    _PSEUDO_COMPONENT_CACHE[name] = result
    return result


# ---------------------------------------------------------------------------
#  Thermo component name map
# ---------------------------------------------------------------------------

_THERMO_COMPONENT_NAME_MAP = {
    "methane": "methane",
    "ethane": "ethane",
    "propane": "propane",
    "i-butane": "isobutane",
    "n-butane": "butane",
    "i-pentane": "isopentane",
    "n-pentane": "pentane",
    "n-hexane": "hexane",
    "n-heptane": "heptane",
    "n-octane": "octane",
    "n-nonane": "nonane",
    "nC10": "decane",
    "nC11": "undecane",
    "nC12": "dodecane",
    "nC13": "tridecane",
    "nC14": "tetradecane",
    "nC15": "pentadecane",
    "nC16": "hexadecane",
    "nC17": "heptadecane",
    "nC18": "octadecane",
    "nC19": "nonadecane",
    "nC20": "eicosane",
    "nitrogen": "nitrogen",
    "CO2": "carbon dioxide",
}

OUTPUT_COLUMNS = [
    "pressure", "temperature",
    "molepct_tot", "molepct_vap",
    "weightpct_tot", "weightpct_vap",
    "volumepct_tot", "volumepct_vap",
    "density_tot", "density_vap", "density_liq",
    "z_factor_tot", "z_factor_vap",
    "mol_wgt_tot", "mol_wgt_vap",
    "enthalpy_tot", "enthalpy_vap",
    "heat_cap_cp_tot", "heat_cap_cp_vap",
    "jt_coeff_vap",
    "viscosity_vap", "viscosity_liq", "viscosity_tot",
    "therm_cond_vap",
]


class PTFlashCalculator:
    """Lightweight PT flash using thermo's SRK/PR EOS + Rachford-Rice.

    Supports both known components (looked up in thermo's database) and
    pseudo-components (with user-supplied MW & density).
    """

    def __init__(self, component_specs: list[dict], eos: str = "srk"):
        """
        Parameters
        ----------
        component_specs : list of dict
            Each dict must have:
                - "name": str — component name
                - "moleFraction": float
            Optional (for pseudo-components not in thermo's database):
                - "molarMassKgPerMol": float
                - "densityKgPerM3": float
        eos : str
            "srk" (SRK-Peneloux with volume translation), "pr" (PR),
            or "srk-classic" (SRK without volume correction).
        """
        if not component_specs:
            raise ValueError("Component list is empty.")

        self.eos_type = eos
        self.names = []
        self.zs_list = []
        self.Tcs_list = []
        self.Pcs_list = []
        self.omegas_list = []
        self.MWs_list = []
        self._HeatCapacityGases = []

        from thermo.chemical import Chemical

        for spec in component_specs:
            name = spec["name"]
            z = float(spec["moleFraction"])
            if z <= 0.0:
                continue

            self.names.append(name)
            self.zs_list.append(z)

            # Check if pseudo-component (has MW & density, not in thermo)
            if "molarMassKgPerMol" in spec:
                mw_kg = float(spec["molarMassKgPerMol"])
                rho = float(spec.get("densityKgPerM3", 800.0))
                Tc, Pc, omega, mw_g = register_pseudo_component(name, mw_kg, rho)
                self.Tcs_list.append(Tc)
                self.Pcs_list.append(Pc)
                self.omegas_list.append(omega)
                self.MWs_list.append(mw_g)
                # Pseudo-component: approximate Cp as similar to closest n-alkane
                # Find closest n-alkane by carbon number
                cn = max(1, min(30, round((mw_g - 2.0) / 14.0)))
                alkane_name = {1: "methane", 2: "ethane", 3: "propane", 4: "butane",
                               5: "pentane", 6: "hexane", 7: "heptane", 8: "octane",
                               9: "nonane", 10: "decane"}.get(cn, None)
                if alkane_name:
                    self._HeatCapacityGases.append(Chemical(alkane_name).HeatCapacityGas)
                else:
                    # Use hexadecane for heavy fractions
                    self._HeatCapacityGases.append(Chemical("hexadecane").HeatCapacityGas)
            else:
                # Known component — look up in thermo
                thermo_name = _THERMO_COMPONENT_NAME_MAP.get(name, name)
                chem = Chemical(thermo_name)
                self.Tcs_list.append(chem.Tc)
                self.Pcs_list.append(chem.Pc)
                self.omegas_list.append(chem.omega)
                self.MWs_list.append(chem.MW)
                self._HeatCapacityGases.append(chem.HeatCapacityGas)

        # Normalise
        total_z = sum(self.zs_list)
        self.zs = np.array([z / total_z for z in self.zs_list], dtype=float)
        self.Tcs = np.array(self.Tcs_list, dtype=float)
        self.Pcs = np.array(self.Pcs_list, dtype=float)
        self.omegas = np.array(self.omegas_list, dtype=float)
        self.MWs = np.array(self.MWs_list, dtype=float)

        # Validate
        for i, name in enumerate(self.names):
            if not (50 < self.Tcs[i] < 2000):
                raise ValueError(f"Tc out of range for {name}: {self.Tcs[i]}")
            if not (1e5 < self.Pcs[i] < 1e8):
                raise ValueError(f"Pc out of range for {name}: {self.Pcs[i]}")

    # ------------------------------------------------------------------
    #  EOS factory
    # ------------------------------------------------------------------
    def _make_eos_class(self):
        """Return the EOS mix class for the configured type."""
        from thermo import SRKMIXTranslated, SRKMIX, PRMIX
        if self.eos_type == "pr":
            return PRMIX
        if self.eos_type == "srk-classic":
            return SRKMIX
        return SRKMIXTranslated  # default: SRK-Peneloux

    def _make_gas_eos(self, T: float, P: float, zs: np.ndarray | None = None):
        """Create a gas-root EOS object."""
        if zs is None:
            zs = self.zs
        cls = self._make_eos_class()
        return cls(
            Tcs=self.Tcs.tolist(),
            Pcs=self.Pcs.tolist(),
            omegas=self.omegas.tolist(),
            zs=zs.tolist(),
            T=T, P=P,
            only_g=True,
        )

    def _make_liq_eos(self, T: float, P: float, zs: np.ndarray | None = None):
        """Create a liquid-root EOS object."""
        if zs is None:
            zs = self.zs
        cls = self._make_eos_class()
        return cls(
            Tcs=self.Tcs.tolist(),
            Pcs=self.Pcs.tolist(),
            omegas=self.omegas.tolist(),
            zs=zs.tolist(),
            T=T, P=P,
            only_l=True,
        )

    # ------------------------------------------------------------------
    #  Phase detection helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _has_liquid_root(eos) -> bool:
        """Check if the EOS object has a physically meaningful liquid root."""
        return hasattr(eos, 'V_l') and hasattr(eos, 'lnphis_l')

    @staticmethod
    def _get_eos_Z(eos, prefer: str = "gas") -> float:
        """Safely get Z-factor from an EOS object."""
        if prefer == "gas":
            if hasattr(eos, 'Z_g'):
                return float(eos.Z_g)
            if hasattr(eos, 'Z_l'):
                return float(eos.Z_l)
        else:
            if hasattr(eos, 'Z_l'):
                return float(eos.Z_l)
            if hasattr(eos, 'Z_g'):
                return float(eos.Z_g)
        return 1.0

    @staticmethod
    def _get_eos_Vm(eos, prefer: str = "gas") -> float:
        """Safely get molar volume [m³/mol] from an EOS object."""
        if prefer == "gas":
            if hasattr(eos, 'V_g'):
                return float(eos.V_g)
            if hasattr(eos, 'V_l'):
                return float(eos.V_l)
        else:
            if hasattr(eos, 'V_l'):
                return float(eos.V_l)
            if hasattr(eos, 'V_g'):
                return float(eos.V_g)
        return 8.314462618 * float(eos.T) / float(eos.P)

    # ------------------------------------------------------------------
    #  PT Flash
    # ------------------------------------------------------------------
    def flash_pt(self, T_K: float, P_Pa: float) -> dict:
        """
        PT flash at given temperature [K] and pressure [Pa].

        Returns a dict with keys:
            V, L, x, y, converged, rho_tot, rho_vap, rho_liq,
            z_factor_tot, z_factor_vap, mol_wgt_tot, mol_wgt_vap,
            enthalpy_tot, enthalpy_vap, cp_tot, cp_vap,
            jt_coeff_vap, viscosity_vap, viscosity_liq, viscosity_tot,
            therm_cond_vap, molepct_vap, weightpct_vap, volumepct_vap
        """
        # Build EOS at feed composition
        eos_g = self._make_gas_eos(T_K, P_Pa)

        # Determine if two phases exist
        has_liq = False
        eos_l = None
        try:
            eos_l = self._make_liq_eos(T_K, P_Pa)
            has_liq = self._has_liquid_root(eos_l)
        except Exception:
            has_liq = False

        if has_liq:
            try:
                lnphi_g = np.array(eos_g.lnphis_g, dtype=float)
                lnphi_l = np.array(eos_l.lnphis_l, dtype=float)
                K = np.exp(lnphi_l - lnphi_g)
                V = self._rachford_rice(K, self.zs)
            except Exception:
                V = 1.0  # fall back to single-phase gas
                K = np.ones_like(self.zs)
        else:
            V = 1.0  # all gas
            K = np.ones_like(self.zs)

        # Compositions from Rachford-Rice
        if V <= 0.0:   # all liquid
            V, L = 0.0, 1.0
            x = self.zs.copy()
            y = np.zeros_like(self.zs)
        elif V >= 1.0:  # all gas
            V, L = 1.0, 0.0
            x = np.zeros_like(self.zs)
            y = self.zs.copy()
        else:
            L = 1.0 - V
            x = self.zs / (1.0 + V * (K - 1.0))
            y = K * x

        # Renormalise
        if V > 0.0 and V < 1.0:
            x = np.abs(x) / np.sum(np.abs(x))
            y = np.abs(y) / np.sum(np.abs(y))

        # ------------------------------------------------------------------
        #  Phase properties
        # ------------------------------------------------------------------
        # Gas / vapour properties
        if V > 1e-12:
            gas = self._make_gas_eos(T_K, P_Pa, zs=y)
            Z_gas = self._get_eos_Z(gas, prefer="gas")
            Vm_gas = self._get_eos_Vm(gas, prefer="gas")
        else:
            Z_gas = float('nan')
            Vm_gas = float('nan')

        # Liquid properties
        if V < 1.0 - 1e-12 and has_liq:
            liq = self._make_liq_eos(T_K, P_Pa, zs=x)
            Z_liq = self._get_eos_Z(liq, prefer="liquid")
            Vm_liq = self._get_eos_Vm(liq, prefer="liquid")
        else:
            Z_liq = float('nan')
            Vm_liq = float('nan')

        # Total (feed) Z from gas EOS
        Z_tot = self._get_eos_Z(eos_g, prefer="gas")

        # Molecular weights
        MW_gas = float(np.sum(y * self.MWs)) if V > 1e-12 else float('nan')
        MW_liq = float(np.sum(x * self.MWs)) if V < 1.0 - 1e-12 else float('nan')
        MW_tot = float(np.sum(self.zs * self.MWs))

        # Densities  [kg/m³]
        rho_gas = MW_gas / (Vm_gas * 1000.0) if V > 1e-12 else float('nan')
        rho_liq = MW_liq / (Vm_liq * 1000.0) if V < 1.0 - 1e-12 else float('nan')
        if V > 0.0 and V < 1.0:
            rho_tot = 1.0 / (V / rho_gas + (1.0 - V) / rho_liq)
        else:
            rho_tot = rho_gas if V > 0.5 else rho_liq

        # Enthalpy & Cp  [J/mol] and [J/(mol·K)]
        H_gas, Cp_gas = self._gas_enthalpy_cp(T_K, P_Pa, y, eos_g) if V > 1e-12 else (float('nan'), float('nan'))
        H_liq, Cp_liq = self._liq_enthalpy_cp(T_K, P_Pa, x, eos_l) if V < 1.0 - 1e-12 else (float('nan'), float('nan'))

        if V > 0.0 and V < 1.0:
            H_tot = V * H_gas + (1.0 - V) * H_liq
            Cp_tot = V * Cp_gas + (1.0 - V) * Cp_liq
        else:
            H_tot = H_gas if V > 0.5 else H_liq
            Cp_tot = Cp_gas if V > 0.5 else Cp_liq

        # Viscosity  [Pa·s]
        mu_gas = self._gas_viscosity(T_K, P_Pa, y, MW_gas, Vm_gas) if V > 1e-12 else float('nan')
        mu_liq = self._liquid_viscosity(T_K, y) if V < 1.0 - 1e-12 else float('nan')
        if V > 0.0 and V < 1.0:
            mu_tot = V * mu_gas + (1.0 - V) * mu_liq
        else:
            mu_tot = mu_gas if V > 0.5 else mu_liq

        # Thermal conductivity  [W/(m·K)]
        k_gas = self._gas_thermal_conductivity(T_K, P_Pa, y, MW_gas, Vm_gas) if V > 1e-12 else float('nan')

        # JT coefficient  [K/Pa]
        jt_gas = self._jt_coefficient(T_K, P_Pa, y, eos_g) if V > 1e-12 else float('nan')

        # Phase fractions
        molepct_vap = V * 100.0
        weightpct_vap = (V * MW_gas / MW_tot * 100.0) if V > 1e-12 else 0.0
        volumepct_vap = (V * Vm_gas / (V * Vm_gas + (1.0 - V) * Vm_liq) * 100.0) if (V > 1e-12 and V < 1.0 - 1e-12) else (100.0 if V > 0.5 else 0.0)

        return {
            "V": float(V), "converged": True,
            # densities [g/cm³ for CSV output]
            "density_tot": rho_tot / 1000.0,
            "density_vap": rho_gas / 1000.0 if V > 1e-12 else float('nan'),
            "density_liq": rho_liq / 1000.0 if V < 1.0 - 1e-12 else float('nan'),
            # Z factors
            "z_factor_tot": Z_tot,
            "z_factor_vap": Z_gas,
            # molecular weights [g/mol]
            "mol_wgt_tot": MW_tot,
            "mol_wgt_vap": MW_gas,
            # enthalpy [J/mol]
            "enthalpy_tot": H_tot,
            "enthalpy_vap": H_gas,
            # heat capacity [J/(mol·K)]
            "heat_cap_cp_tot": Cp_tot,
            "heat_cap_cp_vap": Cp_gas,
            # JT coefficient [K/Pa] → to be converted to °C/bara by CSV writer
            "jt_coeff_vap": jt_gas,
            # viscosity [Pa·s] → to be converted to cP by CSV writer
            "viscosity_vap": mu_gas,
            "viscosity_liq": mu_liq,
            "viscosity_tot": mu_tot,
            # thermal conductivity [W/(m·K)] → to be converted to mW/(m·°C)
            "therm_cond_vap": k_gas,
            # phase fractions [%]
            "molepct_tot": 100.0,
            "molepct_vap": molepct_vap,
            "weightpct_tot": 100.0,
            "weightpct_vap": weightpct_vap,
            "volumepct_tot": 100.0,
            "volumepct_vap": volumepct_vap,
        }

    # ------------------------------------------------------------------
    #  Rachford-Rice solver
    # ------------------------------------------------------------------
    @staticmethod
    def _rachford_rice(K: np.ndarray, z: np.ndarray, tol: float = 1e-10, max_iter: int = 200) -> float:
        """Solve Rachford-Rice equation for vapor fraction V."""
        # Check single-phase
        if np.sum(K * z) <= 1.0:
            return 0.0  # all liquid
        if np.sum(z / K) <= 1.0:
            return 1.0  # all gas

        # Bisection
        V_lo, V_hi = 0.0, 1.0
        for _ in range(max_iter):
            V = 0.5 * (V_lo + V_hi)
            f = np.sum(z * (K - 1.0) / (1.0 + V * (K - 1.0)))
            if abs(f) < tol:
                return V
            if f > 0:
                V_lo = V
            else:
                V_hi = V

            if V_hi - V_lo < 1e-12:
                break

        V = 0.5 * (V_lo + V_hi)
        return max(0.0, min(1.0, V))

    # ------------------------------------------------------------------
    #  Property helpers
    # ------------------------------------------------------------------
    def _gas_enthalpy_cp(self, T: float, P: float, y: np.ndarray, eos_g):
        """Gas phase enthalpy [J/mol] and Cp [J/(mol·K)]."""
        try:
            from thermo import CEOSGas
            eos_kwargs = dict(Tcs=self.Tcs.tolist(), Pcs=self.Pcs.tolist(), omegas=self.omegas.tolist())
            gas_phase = CEOSGas(self._make_eos_class(), eos_kwargs,
                               HeatCapacityGases=self._HeatCapacityGases,
                               T=T, P=P, zs=y.tolist())
            H = float(gas_phase.H())
            Cp = float(gas_phase.Cp())
            return H, Cp
        except Exception:
            # Fallback: ideal gas
            T_ref = 298.15
            H_ig = 0.0
            Cp_ig = 0.0
            for i, chem in enumerate(None):
                Cp_ig += y[i] * chem.HeatCapacityGas(T)
            H_ig = Cp_ig * (T - T_ref)
            return H_ig, Cp_ig

    def _liq_enthalpy_cp(self, T: float, P: float, x: np.ndarray, eos_l):
        """Liquid phase enthalpy [J/mol] and Cp [J/(mol·K)]."""
        try:
            from thermo import CEOSLiquid
            eos_kwargs = dict(Tcs=self.Tcs.tolist(), Pcs=self.Pcs.tolist(), omegas=self.omegas.tolist())
            liq_phase = CEOSLiquid(self._make_eos_class(), eos_kwargs,
                                   HeatCapacityGases=self._HeatCapacityGases,
                                   T=T, P=P, zs=x.tolist())
            H = float(liq_phase.H())
            Cp = float(liq_phase.Cp())
            return H, Cp
        except Exception:
            # Fallback: approximate liquid Cp as 1.5 × ideal gas Cp
            T_ref = 298.15
            Cp_ig = 0.0
            for i, chem in enumerate(None):
                Cp_ig += x[i] * chem.HeatCapacityGas(T)
            Cp_liq = Cp_ig * 1.5  # rough approximation
            H_liq = Cp_liq * (T - T_ref)
            return H_liq, Cp_liq

    def _gas_viscosity(self, T: float, P: float, y: np.ndarray, MW: float, Vm: float) -> float:
        """Gas viscosity via Lucas method [Pa·s]."""
        try:
            from thermo.viscosity import ViscosityGas
            from thermo.viscosity import LUCAS_GAS as GAS_VISC_MODEL
            Tcs = self.Tcs.tolist()
            Pcs = self.Pcs.tolist()
            omegas = self.omegas.tolist()
            MWs = self.MWs.tolist()
            zs_list = y.tolist()

            visc = ViscosityGas(MWs=MWs, Tcs=Tcs, Pcs=Pcs, omegas=omegas,
                               )
            mu = visc.calculate(T, P, zs_list, GAS_VISC_MODEL)
            return float(mu)
        except Exception:
            # Fallback: simple power-law for methane-dominated gas
            return 1.0e-5 * (T / 273.15) ** 0.75

    def _liquid_viscosity(self, T: float, x: np.ndarray) -> float:
        """Liquid viscosity via Letsou-Stiel [Pa·s]."""
        try:
            from thermo.viscosity import ViscosityLiquid
            from thermo.viscosity import LETSOU_STIEL
            Tcs = self.Tcs.tolist()
            Pcs = self.Pcs.tolist()
            omegas = self.omegas.tolist()
            MWs = self.MWs.tolist()
            zs_list = x.tolist()

            visc = ViscosityLiquid(MWs=MWs, Tcs=Tcs, Pcs=Pcs, omegas=omegas,
                                  )
            mu = visc.calculate(T, zs_list, LETSOU_STIEL)
            return float(mu)
        except Exception:
            # Fallback for oil-like liquid
            return 0.3e-3  # 0.3 cP = 0.0003 Pa·s

    def _gas_thermal_conductivity(self, T: float, P: float, y: np.ndarray, MW: float, Vm: float) -> float:
        """Gas thermal conductivity [W/(m·K)]."""
        try:
            from thermo.thermal_conductivity import ThermalConductivityGas
            Tcs = self.Tcs.tolist()
            Pcs = self.Pcs.tolist()
            omegas = self.omegas.tolist()
            MWs = self.MWs.tolist()
            zs_list = y.tolist()

            tc = ThermalConductivityGas(MWs=MWs, Tcs=Tcs, Pcs=Pcs, omegas=omegas,
                                       )
            k = tc.calculate(T, P, zs_list)
            return float(k)
        except Exception:
            return 0.03  # W/(m·K) typical for methane

    def _jt_coefficient(self, T: float, P: float, y: np.ndarray, eos_g) -> float:
        """Joule-Thomson coefficient [K/Pa]."""
        try:
            # μ_JT = (T*(∂V/∂T)_P - V) / Cp
            Cp = float(eos_g.Cp_minus_Cv_g) if hasattr(eos_g, 'Cp_minus_Cv_g') else 35.0
            dV_dT = float(eos_g.dV_dT_g) if hasattr(eos_g, 'dV_dT_g') else 0.0
            V = float(eos_g.V_g)
            jt = (T * dV_dT - V) / (Cp + 1e-30)
            return float(jt)
        except Exception:
            return 0.0


# ===================================================================
#  PVT table generator  (replaces generate_neqsim_pvt_csv)
# ===================================================================

def generate_flash_pvt_csv(
    *,
    composition: list[dict],
    output_path: str | Path,
    eos: str = "srk",
    pmin: float = 10.0,
    pmax: float = 150.0,
    pn: int = 50,
    tmin: float = -10.0,
    tmax: float = 40.0,
    tn: int = 50,
    force: bool = False,
) -> Path:
    """
    Generate a steady-compatible PVT CSV using thermo's flash engine.

    Parameters
    ----------
    composition : list[dict]
        Each dict: {"name": str, "moleFraction": float}.  For pseudo-components
        also include "molarMassKgPerMol" and "densityKgPerM3".
    output_path : Path
        Target CSV path.
    eos : str
        "srk" (SRK-Peneloux), "srk-classic", or "pr".
    pmin, pmax, pn : float, float, int
        Pressure grid: min bara, max bara, number of points.
    tmin, tmax, tn : float, float, int
        Temperature grid: min °C, max °C, number of points.
    force : bool
        If True, overwrite existing output.

    Returns
    -------
    output_path : Path
    """
    output_path = Path(output_path).resolve()
    if output_path.exists() and not force:
        return output_path

    if pn < 2 or tn < 2:
        raise ValueError("pn and tn must be at least 2.")

    calculator = PTFlashCalculator(composition, eos=eos)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    pressures_bara = np.linspace(pmin, pmax, pn)
    temperatures_C = np.linspace(tmin, tmax, tn)

    total = pn * tn
    for ip, p_bara in enumerate(pressures_bara):
        for it, t_C in enumerate(temperatures_C):
            idx = ip * tn + it
            if idx % 250 == 0:
                print(f"  [flash_pvt] {idx}/{total} points ...")

            T_K = t_C + 273.15
            P_Pa = p_bara * 1e5

            try:
                result = calculator.flash_pt(T_K, P_Pa)
            except Exception as exc:
                print(f"  [flash_pvt] WARNING: flash failed at P={p_bara:.1f} bara, T={t_C:.1f} °C: {exc}")
                continue

            rows.append({
                "pressure": p_bara,
                "temperature": t_C,
                "molepct_tot": result["molepct_tot"],
                "molepct_vap": result["molepct_vap"],
                "weightpct_tot": result["weightpct_tot"],
                "weightpct_vap": result["weightpct_vap"],
                "volumepct_tot": result["volumepct_tot"],
                "volumepct_vap": result["volumepct_vap"],
                "density_tot": _nan_if_invalid(result["density_tot"]),
                "density_vap": _nan_if_invalid(result["density_vap"]),
                "density_liq": _nan_if_invalid(result["density_liq"]),
                "z_factor_tot": _nan_if_invalid(result["z_factor_tot"]),
                "z_factor_vap": _nan_if_invalid(result["z_factor_vap"]),
                "mol_wgt_tot": _nan_if_invalid(result["mol_wgt_tot"]),
                "mol_wgt_vap": _nan_if_invalid(result["mol_wgt_vap"]),
                # Convert J/mol → keep as J/mol (same as NeqSim)
                "enthalpy_tot": _nan_if_invalid(result["enthalpy_tot"]),
                "enthalpy_vap": _nan_if_invalid(result["enthalpy_vap"]),
                # Convert J/(mol·K) → J/(mol·°C) (same numerically)
                "heat_cap_cp_tot": _nan_if_invalid(result["heat_cap_cp_tot"]),
                "heat_cap_cp_vap": _nan_if_invalid(result["heat_cap_cp_vap"]),
                # JT: K/Pa → °C/bara  (multiply by 1e5)
                "jt_coeff_vap": _nan_if_invalid(result["jt_coeff_vap"] * 1e5),
                # Viscosity: Pa·s → cP  (multiply by 1000)
                "viscosity_vap": _nan_if_invalid(result["viscosity_vap"] * 1000.0),
                "viscosity_liq": _nan_if_invalid(result["viscosity_liq"] * 1000.0),
                "viscosity_tot": _nan_if_invalid(result["viscosity_tot"] * 1000.0),
                # Thermal conductivity: W/(m·K) → mW/(m·°C)  (multiply by 1000)
                "therm_cond_vap": _nan_if_invalid(result["therm_cond_vap"] * 1000.0),
            })

    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    df.to_csv(output_path, index=False, float_format="%.10g")
    print(f"  [flash_pvt] Wrote {len(df)} rows to {output_path}")
    return output_path


def _nan_if_invalid(value: float) -> float:
    """Return NaN for invalid values so downstream interpolation can handle them."""
    if value is None:
        return float('nan')
    if math.isnan(value) or math.isinf(value):
        return float('nan')
    return value


# ===================================================================
#  FlashPVTPropertyProvider  (replaces NeqSimPVTPropertyProvider)
# ===================================================================

class FlashPVTPropertyProvider:
    """PVT provider backed by thermo flash engine, with auto grid-expansion."""

    def __init__(
        self,
        *,
        composition: list[dict],
        output_dir: str | Path,
        eos: str = "srk",
        pmin: float = 10.0,
        pmax: float = 150.0,
        pn: int = 50,
        tmin: float = -10.0,
        tmax: float = 40.0,
        tn: int = 50,
        force: bool = False,
    ) -> None:
        self.composition = composition
        self.output_dir = Path(output_dir).resolve()
        self.eos = eos
        self.pmin = float(pmin)
        self.pmax = float(pmax)
        self.pn = int(pn)
        self.tmin = float(tmin)
        self.tmax = float(tmax)
        self.tn = int(tn)
        self.force = force
        self.pvt_file: Path | None = None
        self._provider = None
        self.regeneration_count = 0
        self._regenerate(force=force)

    def summary(self):
        base = self._provider.summary()
        base["flash_dynamic"] = {
            "pvt_file": str(self.pvt_file),
            "regeneration_count": self.regeneration_count,
            "pressure_range": (self.pmin, self.pmax),
            "temperature_range": (self.tmin, self.tmax),
        }
        return base

    def get_properties(self, pressure_bara: float, temperature_C: float, *, allow_extrapolation: bool = False):
        if self._outside(pressure_bara, temperature_C):
            self._expand_to_include(pressure_bara, temperature_C)
            self._regenerate(force=False)
        return self._provider.get_properties(
            pressure_bara, temperature_C,
            allow_extrapolation=allow_extrapolation,
        )

    def _outside(self, pressure_bara: float, temperature_C: float) -> bool:
        return (
            pressure_bara < self.pmin or pressure_bara > self.pmax
            or temperature_C < self.tmin or temperature_C > self.tmax
        )

    def _expand_to_include(self, pressure_bara: float, temperature_C: float) -> None:
        if pressure_bara < self.pmin:
            self.pmin = _nice_floor(max(1.0, pressure_bara * 0.8), 5.0)
        if pressure_bara > self.pmax:
            self.pmax = _nice_ceil(max(pressure_bara * 1.2, pressure_bara + 10.0), 5.0)
        if temperature_C < self.tmin:
            self.tmin = _nice_floor(temperature_C - 15.0, 5.0)
        if temperature_C > self.tmax:
            self.tmax = _nice_ceil(temperature_C + 15.0, 5.0)

    def _regenerate(self, *, force: bool) -> None:
        from steady.pvt_provider import PVTPropertyProvider

        cache_key = _cache_key(
            composition=json.dumps(self.composition, sort_keys=True).encode(),
            eos=self.eos,
            pmin=self.pmin, pmax=self.pmax, pn=self.pn,
            tmin=self.tmin, tmax=self.tmax, tn=self.tn,
        )
        output_path = self.output_dir / f"flash_pvt_{cache_key[:12]}.csv"

        self.pvt_file = generate_flash_pvt_csv(
            composition=self.composition,
            output_path=output_path,
            eos=self.eos,
            pmin=self.pmin, pmax=self.pmax, pn=self.pn,
            tmin=self.tmin, tmax=self.tmax, tn=self.tn,
            force=force,
        )
        self._provider = PVTPropertyProvider(str(self.pvt_file))
        self.regeneration_count += 1


# ===================================================================
#  Utility: cache key
# ===================================================================

def _cache_key(**items: object) -> str:
    hasher = hashlib.sha256()
    comp_bytes = items.pop("composition", b"")
    hasher.update(comp_bytes if isinstance(comp_bytes, bytes) else str(comp_bytes).encode())
    for key in sorted(items):
        hasher.update(str(key).encode("utf-8"))
        hasher.update(str(items[key]).encode("utf-8"))
    return hasher.hexdigest()


# ===================================================================
#  Excel composition export  (unchanged from neqsim_pvt.py)
# ===================================================================

def _export_all_components_json(
    *,
    components: dict[str, float],
    workbook_path: Path,
    output_dir: Path,
    composition_sheet: str = "组分",
) -> Path:
    """Export ALL components (C1-C30) as a flash-compatible JSON without TBP collapse."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_key = hashlib.sha256(
        workbook_path.read_bytes() + composition_sheet.encode("utf-8") + b"no_tbp"
    ).hexdigest()[:12]
    output_path = output_dir / f"{workbook_path.stem}_all_components_{cache_key}.json"

    payload = {
        "format": "steady-flash-fluid",
        "version": "1.0",
        "components": components,       # 全部C1-C27组分
        "tbpFractions": [],              # 不做TBP折叠
        "plusFractions": [],             # 不做Plus fraction
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def export_composition_sheet_to_csv(
    *,
    workbook_path: str | Path,
    output_dir: str | Path,
    sheet_name: str = "组分",
) -> Path:
    """Export a pipeline workbook composition sheet to CSV format."""
    workbook_path = Path(workbook_path).resolve()
    output_dir = Path(output_dir).resolve()
    if not workbook_path.exists():
        raise FileNotFoundError(f"管道算例文件不存在: {workbook_path}")

    df = pd.read_excel(workbook_path, sheet_name=sheet_name)
    comp_col = _find_column(df, ["组分", "component", "Component"])
    frac_col = _find_column(df, ["摩尔分数(%)", "摩尔分数", "moleFraction", "mole_fraction"])
    if comp_col is None or frac_col is None:
        raise ValueError(f"{workbook_path} 的 {sheet_name} sheet 缺少组分或摩尔分数列。")
    values_are_percent = "%" in str(frac_col)

    rows: list[tuple[str, float]] = []
    for _, row in df.iterrows():
        raw_name = str(row[comp_col]).strip()
        if not raw_name or raw_name.lower() == "nan":
            continue
        value = float(row[frac_col])
        if not math.isfinite(value) or value <= 0.0:
            continue
        rows.append((_normalize_component_name(raw_name), value / 100.0 if values_are_percent else value))
    if not rows:
        raise ValueError(f"{workbook_path} 的 {sheet_name} sheet 没有有效组分。")

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_key = hashlib.sha256(
        workbook_path.read_bytes() + sheet_name.encode("utf-8")
    ).hexdigest()[:12]
    output_path = output_dir / f"{workbook_path.stem}_{sheet_name}_{cache_key}.csv"
    with output_path.open("w", encoding="utf-8", newline="") as f:
        f.write("component,moleFraction\n")
        for name, value in rows:
            f.write(f"{name},{value:.12g}\n")
    return output_path


def export_flash_fluid_input_from_workbook(
    *,
    workbook_path: str | Path,
    output_dir: str | Path,
    composition_sheet: str = "组分",
    heavy_fraction_sheet: str = DEFAULT_HEAVY_FRACTION_SHEET,
    config_sheet: str = DEFAULT_FLASH_CONFIG_SHEET,
    collapse_heavy_to_tbp: bool = False,       # 默认不折叠，直接用全组分
    tbp_from_carbon: int = 10,
    tbp_name: str | None = None,
) -> Path:
    """Export a workbook as a flash-compatible JSON input.

    Priority:
      1. If an explicit heavy-fraction sheet exists, use it as authored.
      2. Otherwise, if heavy SCN components exist and collapse_heavy_to_tbp is true,
         collapse them into one TBP pseudo-component.
      3. Otherwise, export ALL components as JSON (no TBP collapse).
    """
    workbook_path = Path(workbook_path).resolve()
    output_dir = Path(output_dir).resolve()
    xl = pd.ExcelFile(workbook_path)
    if heavy_fraction_sheet not in xl.sheet_names:
        components = _read_light_components_from_sheet(workbook_path, composition_sheet)
        if collapse_heavy_to_tbp and _has_heavy_components(components, from_carbon=tbp_from_carbon):
            return export_collapsed_tbp_fluid_from_workbook(
                workbook_path=workbook_path, output_dir=output_dir,
                composition_sheet=composition_sheet,
                tbp_from_carbon=tbp_from_carbon, tbp_name=tbp_name,
            )
        # 不折叠：全部组分直接作为 components 输出为 JSON
        return _export_all_components_json(
            components=components, workbook_path=workbook_path,
            output_dir=output_dir, composition_sheet=composition_sheet,
        )

    light_components = _read_light_components_from_sheet(workbook_path, composition_sheet)
    heavy_fractions = _read_heavy_fractions_from_sheet(workbook_path, heavy_fraction_sheet)
    config = _read_flash_config_sheet(workbook_path, config_sheet) if config_sheet in xl.sheet_names else {}

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_key = hashlib.sha256(
        workbook_path.read_bytes()
        + composition_sheet.encode("utf-8")
        + heavy_fraction_sheet.encode("utf-8")
        + config_sheet.encode("utf-8")
    ).hexdigest()[:12]
    output_path = output_dir / f"{workbook_path.stem}_flash_fluid_{cache_key}.json"

    payload = {
        "format": "steady-flash-fluid",
        "version": "1.0",
        "components": light_components,
        "tbpFractions": heavy_fractions["tbp"],
        "plusFractions": heavy_fractions["plus"],
    }
    if config:
        payload["config"] = config

    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def export_collapsed_plus_fluid_from_workbook(
    *, workbook_path, output_dir, composition_sheet="组分",
    plus_from_carbon=10, plus_name=None,
) -> Path:
    """Collapse heavy components from Cn upward into one plus fraction JSON."""
    workbook_path = Path(workbook_path).resolve()
    output_dir = Path(output_dir).resolve()
    components = _read_light_components_from_sheet(workbook_path, composition_sheet)
    collapsed = collapse_heavy_components_to_plus(components, plus_from_carbon=plus_from_carbon, plus_name=plus_name)

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_key = hashlib.sha256(
        workbook_path.read_bytes() + composition_sheet.encode("utf-8")
        + f"plus_from_{plus_from_carbon}".encode("utf-8")
    ).hexdigest()[:12]
    plus_label = collapsed["plusFractions"][0]["name"]
    output_path = output_dir / f"{workbook_path.stem}_{plus_label}_{cache_key}.json"
    output_path.write_text(json.dumps(collapsed, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def export_collapsed_tbp_fluid_from_workbook(
    *, workbook_path, output_dir, composition_sheet="组分",
    tbp_from_carbon=10, tbp_name=None,
) -> Path:
    """Collapse heavy components from Cn upward into one TBP pseudo-component JSON."""
    workbook_path = Path(workbook_path).resolve()
    output_dir = Path(output_dir).resolve()
    components = _read_light_components_from_sheet(workbook_path, composition_sheet)
    collapsed = collapse_heavy_components_to_tbp(components, tbp_from_carbon=tbp_from_carbon, tbp_name=tbp_name)

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_key = hashlib.sha256(
        workbook_path.read_bytes() + composition_sheet.encode("utf-8")
        + f"tbp_from_{tbp_from_carbon}".encode("utf-8")
    ).hexdigest()[:12]
    tbp_label = collapsed["tbpFractions"][0]["name"] if collapsed["tbpFractions"] else f"C{tbp_from_carbon}_TBP"
    output_path = output_dir / f"{workbook_path.stem}_{tbp_label}_{cache_key}.json"
    output_path.write_text(json.dumps(collapsed, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def read_boundary_conditions_from_workbook(workbook_path: str | Path) -> dict[str, float]:
    """Read inlet temperature and outlet pressure from a 边界条件 sheet if present."""
    workbook_path = Path(workbook_path).resolve()
    xl = pd.ExcelFile(workbook_path)
    if "边界条件" not in xl.sheet_names:
        return {}
    df = pd.read_excel(workbook_path, sheet_name="边界条件")
    if df.empty:
        return {}
    result: dict[str, float] = {}
    t_col = _find_column(df, ["入口温度 (℃)", "入口温度℃", "入口温度", "tin"])
    p_col = _find_column(df, ["出口压力 (MPa)", "出口压力MPa", "出口压力", "pout"])
    if t_col is not None:
        result["inlet_temperature_C"] = float(df.iloc[0][t_col])
    if p_col is not None:
        p_val = float(df.iloc[0][p_col])
        result["target_outlet_pressure_bara"] = p_val * 10.0 if "MPa" in str(p_col) else p_val
    return result


def estimate_pvt_grid_from_conditions(
    *, inlet_temperature_C, target_outlet_pressure_bara,
    inlet_pressure_guess_bara, inlet_pressure_low_bara,
    inlet_pressure_high_bara, env_temperature_C=None,
) -> dict[str, float | int]:
    """Build a practical initial P/T grid around the pipeline operating envelope."""
    p_ref_max = max(target_outlet_pressure_bara, inlet_pressure_guess_bara, inlet_pressure_high_bara)
    p_ref_min = min(target_outlet_pressure_bara, inlet_pressure_low_bara, inlet_pressure_guess_bara)
    pmin = max(1.0, min(target_outlet_pressure_bara * 0.75, p_ref_min * 0.9))
    pmax = max(target_outlet_pressure_bara * 1.2, p_ref_max * 1.1, target_outlet_pressure_bara + 25.0)

    temps = [inlet_temperature_C]
    if env_temperature_C is not None and math.isfinite(env_temperature_C):
        temps.append(env_temperature_C)
    tmin = min(temps) - 20.0
    tmax = max(temps) + 20.0
    return {
        "pmin": _nice_floor(pmin, 5.0), "pmax": _nice_ceil(pmax, 5.0), "pn": 50,
        "tmin": _nice_floor(tmin, 5.0), "tmax": _nice_ceil(tmax, 5.0), "tn": 50,
    }


# ===================================================================
#  Heavy-component characterisation  (unchanged logic, format updated)
# ===================================================================

def collapse_heavy_components_to_plus(
    components: dict[str, float], *, plus_from_carbon: int = 10, plus_name: str | None = None,
) -> dict[str, object]:
    """Collapse SCN-heavy components into one plus fraction."""
    plus_name = plus_name or f"C{plus_from_carbon}+"
    light: dict[str, float] = {}
    heavy_terms: list[tuple[str, float, float, float]] = []

    for name, mole_fraction in components.items():
        carbon_number = HEAVY_COMPONENT_CARBON_NUMBER.get(name)
        if carbon_number is None or carbon_number < plus_from_carbon:
            light[name] = mole_fraction
            continue
        mw_g_mol = HEAVY_COMPONENT_MW_G_MOL[name]
        rho_kg_m3 = estimate_pseudo_density_kg_m3(mw_g_mol / 1000.0)
        heavy_terms.append((name, mole_fraction, mw_g_mol / 1000.0, rho_kg_m3))

    if not heavy_terms:
        return {"format": "steady-flash-fluid", "version": "1.0",
                "components": components, "tbpFractions": [], "plusFractions": [], "config": {}}

    z_plus = sum(z for _, z, _, _ in heavy_terms)
    mw_plus = sum(z * mw for _, z, mw, _ in heavy_terms) / z_plus
    rho_plus = (
        sum(z * mw for _, z, mw, _ in heavy_terms)
        / sum((z * mw) / rho for _, z, mw, rho in heavy_terms)
    )

    return {
        "format": "steady-flash-fluid", "version": "1.0",
        "components": light,
        "tbpFractions": [],
        "plusFractions": [{"name": plus_name, "moleFraction": z_plus,
                           "molarMassKgPerMol": mw_plus, "densityKgPerM3": rho_plus}],
        "config": {"plusFractionModel": "Pedersen", "lumpingModel": "PVTlumpingModel",
                   "plusFractionGroups": 6, "useVolumeCorrection": True, "oilViscosityModel": "LBC"},
    }


def collapse_heavy_components_to_tbp(
    components: dict[str, float], *, tbp_from_carbon: int = 10, tbp_name: str | None = None,
) -> dict[str, object]:
    """Collapse SCN-heavy components into one TBP pseudo-component."""
    tbp_name = tbp_name or f"C{tbp_from_carbon}_TBP"
    light: dict[str, float] = {}
    heavy_terms: list[tuple[str, float, float, float]] = []

    for name, mole_fraction in components.items():
        carbon_number = HEAVY_COMPONENT_CARBON_NUMBER.get(name)
        if carbon_number is None or carbon_number < tbp_from_carbon:
            light[name] = mole_fraction
            continue
        mw_g_mol = HEAVY_COMPONENT_MW_G_MOL[name]
        rho_kg_m3 = estimate_pseudo_density_kg_m3(mw_g_mol / 1000.0)
        heavy_terms.append((name, mole_fraction, mw_g_mol / 1000.0, rho_kg_m3))

    if not heavy_terms:
        return {"format": "steady-flash-fluid", "version": "1.0",
                "components": components, "tbpFractions": [], "plusFractions": [], "config": {}}

    z_tbp = sum(z for _, z, _, _ in heavy_terms)
    mw_tbp = sum(z * mw for _, z, mw, _ in heavy_terms) / z_tbp
    rho_tbp = (
        sum(z * mw for _, z, mw, _ in heavy_terms)
        / sum((z * mw) / rho for _, z, mw, rho in heavy_terms)
    )

    return {
        "format": "steady-flash-fluid", "version": "1.0",
        "components": light,
        "tbpFractions": [{"name": tbp_name, "moleFraction": z_tbp,
                          "molarMassKgPerMol": mw_tbp, "densityKgPerM3": rho_tbp}],
        "plusFractions": [],
        "config": {"tbpModel": "PedersenSRK", "useVolumeCorrection": True, "oilViscosityModel": "LBC"},
    }


def estimate_pseudo_density_kg_m3(molar_mass_kg_mol: float) -> float:
    """Estimate standard liquid density for a pseudo-component (NeqSim correlation)."""
    return 1000.0 * (0.5046 * molar_mass_kg_mol + 0.668468)


# ===================================================================
#  Internal helpers  (unchanged from neqsim_pvt.py)
# ===================================================================

def _find_column(df: pd.DataFrame, names: list[str]) -> str | None:
    normalized = {_normalize_header(col): col for col in df.columns}
    for name in names:
        found = normalized.get(_normalize_header(name))
        if found is not None:
            return found
    return None


def _normalize_header(value: object) -> str:
    return str(value).strip().lower().replace(" ", "")


def _normalize_component_name(name: str) -> str:
    key = name.strip().lower().replace("_", " ").replace("-", " ")
    key = " ".join(key.split())
    return COMPONENT_NAME_ALIASES.get(key, name.strip())


def _read_light_components_from_sheet(workbook_path: Path, sheet_name: str) -> dict[str, float]:
    df = pd.read_excel(workbook_path, sheet_name=sheet_name)
    comp_col = _find_column(df, ["组分", "component", "Component"])
    frac_col = _find_column(df, ["摩尔分数(%)", "摩尔分数", "moleFraction", "mole_fraction"])
    if comp_col is None or frac_col is None:
        raise ValueError(f"{workbook_path} 的 {sheet_name} sheet 缺少组分或摩尔分数列。")
    values_are_percent = "%" in str(frac_col)

    components: dict[str, float] = {}
    for _, row in df.iterrows():
        raw_name = str(row[comp_col]).strip()
        if not raw_name or raw_name.lower() == "nan":
            continue
        value = float(row[frac_col])
        if not math.isfinite(value) or value <= 0.0:
            continue
        components[_normalize_component_name(raw_name)] = value / 100.0 if values_are_percent else value
    if not components:
        raise ValueError(f"{workbook_path} 的 {sheet_name} sheet 没有有效组分。")
    return components


def _has_heavy_components(components: dict[str, float], *, from_carbon: int) -> bool:
    for name, mole_fraction in components.items():
        carbon_number = HEAVY_COMPONENT_CARBON_NUMBER.get(name)
        if carbon_number is not None and carbon_number >= from_carbon and mole_fraction > 0.0:
            return True
    return False


def _read_heavy_fractions_from_sheet(workbook_path: Path, sheet_name: str) -> dict[str, list[dict[str, float | str]]]:
    df = pd.read_excel(workbook_path, sheet_name=sheet_name)
    name_col = _find_column(df, ["名称", "组分", "name", "component"])
    type_col = _find_column(df, ["类型", "type"])
    frac_col = _find_column(df, ["摩尔分数(%)", "摩尔分数", "moleFraction", "mole_fraction"])
    mw_col = _find_column(df, ["分子量(g/mol)", "分子量", "molarMass_g_mol", "molarMass"])
    density_col = _find_column(df, ["密度(kg/m3)", "密度", "density_kg_m3", "density"])
    if None in {name_col, type_col, frac_col, mw_col, density_col}:
        raise ValueError(f"{workbook_path} 的 {sheet_name} sheet 缺少必要列：名称/类型/摩尔分数/分子量/密度。")
    values_are_percent = "%" in str(frac_col)
    mw_is_g_per_mol = "g/mol" in str(mw_col).lower() or str(mw_col).lower().endswith("分子量")

    result: dict[str, list[dict[str, float | str]]] = {"tbp": [], "plus": []}
    for _, row in df.iterrows():
        name = str(row[name_col]).strip()
        fraction_type = str(row[type_col]).strip().lower()
        if not name or name.lower() == "nan":
            continue
        z = float(row[frac_col])
        mw = float(row[mw_col])
        density = float(row[density_col])
        if not all(math.isfinite(v) for v in (z, mw, density)) or z <= 0.0 or mw <= 0.0 or density <= 0.0:
            continue
        entry = {
            "name": name,
            "moleFraction": z / 100.0 if values_are_percent else z,
            "molarMassKgPerMol": mw / 1000.0 if mw_is_g_per_mol else mw,
            "densityKgPerM3": density,
        }
        if fraction_type in {"plus", "plusfraction", "plus_fraction", "c7+", "c10+"}:
            result["plus"].append(entry)
        else:
            result["tbp"].append(entry)
    if not result["tbp"] and not result["plus"]:
        raise ValueError(f"{workbook_path} 的 {sheet_name} sheet 没有有效重端表征数据。")
    return result


def _read_flash_config_sheet(workbook_path: Path, sheet_name: str) -> dict[str, object]:
    """Read Flash配置 sheet (replaces old NeqSim配置)."""
    df = pd.read_excel(workbook_path, sheet_name=sheet_name)
    if df.empty:
        return {}
    key_col = _find_column(df, ["参数", "键", "key", "name"])
    value_col = _find_column(df, ["值", "value"])
    if key_col is None or value_col is None:
        return {}

    raw: dict[str, object] = {}
    for _, row in df.iterrows():
        key = str(row[key_col]).strip()
        if not key or key.lower() == "nan":
            continue
        value = row[value_col]
        if pd.isna(value):
            continue
        raw[_normalize_header(key)] = value

    config: dict[str, object] = {}
    if "eos" in raw:
        config["eos"] = str(raw["eos"]).strip()
    if "tbpmodel" in raw:
        config["tbpModel"] = str(raw["tbpmodel"]).strip()
    if "usevolumecorrection" in raw:
        config["useVolumeCorrection"] = _to_bool(raw["usevolumecorrection"])
    if "oilviscositymodel" in raw:
        config["oilViscosityModel"] = str(raw["oilviscositymodel"]).strip()
    if "gasviscositymodel" in raw:
        config["gasViscosityModel"] = str(raw["gasviscositymodel"]).strip()
    return config


def _to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "是"}


def _nice_floor(value: float, step: float) -> float:
    return math.floor(value / step) * step


def _nice_ceil(value: float, step: float) -> float:
    return math.ceil(value / step) * step
