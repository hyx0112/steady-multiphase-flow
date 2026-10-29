"""本地 Peng-Robinson EOS 与等温等压气液闪蒸。

实现不依赖外部热力学包，包含 Wilson K 初值、Rachford-Rice 相分率、
PR 逸度系数迭代及相态物性。适用于烃类初步工程计算；正式项目仍需
用实验 PVT/PVTsim 数据校正重组分、二元交互系数和输运性质。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import PropertyBatch, PropertyProvider

R = 8.31446261815324
SQRT2 = np.sqrt(2.0)


@dataclass(frozen=True)
class Component:
    name: str
    critical_temperature: float
    critical_pressure: float
    acentric_factor: float
    molecular_weight: float
    ideal_cp: float
    liquid_viscosity_ref: float


COMPONENTS: dict[str, Component] = {
    "methane": Component("methane", 190.564, 4.5992e6, 0.01142, 0.016043, 2220.0, 1.0e-4),
    "ethane": Component("ethane", 305.322, 4.8722e6, 0.0995, 0.030070, 1750.0, 1.2e-4),
    "propane": Component("propane", 369.830, 4.2480e6, 0.1523, 0.044097, 1670.0, 1.5e-4),
    "i-butane": Component("i-butane", 407.810, 3.6290e6, 0.1840, 0.058124, 1650.0, 1.8e-4),
    "n-butane": Component("n-butane", 425.125, 3.7960e6, 0.2002, 0.058124, 1680.0, 1.9e-4),
    "i-pentane": Component("i-pentane", 460.350, 3.3780e6, 0.2274, 0.072151, 1700.0, 2.2e-4),
    "n-pentane": Component("n-pentane", 469.700, 3.3700e6, 0.2510, 0.072151, 1710.0, 2.4e-4),
    "n-hexane": Component("n-hexane", 507.820, 3.0340e6, 0.2990, 0.086178, 1760.0, 3.0e-4),
    "n-heptane": Component("n-heptane", 540.130, 2.7360e6, 0.3490, 0.100205, 1820.0, 3.9e-4),
    "n-octane": Component("n-octane", 568.740, 2.4860e6, 0.3990, 0.114232, 1880.0, 5.1e-4),
    "n-nonane": Component("n-nonane", 594.550, 2.2810e6, 0.4440, 0.128259, 1940.0, 6.8e-4),
    "n-decane": Component("n-decane", 617.700, 2.1030e6, 0.4880, 0.142286, 2010.0, 9.2e-4),
    # 详细重端组分：参数来自项目 HGC_component_reference.csv，作为
    # 本地 PR 闪蒸的工程基线；不把这些组分合并为 plus fraction。
    "n-undecane": Component("n-undecane", 638.800, 1.9700e6, 0.5350, 0.156313, 2080.0, 1.1e-3),
    "n-dodecane": Component("n-dodecane", 658.200, 1.8170e6, 0.5740, 0.170340, 2150.0, 1.3e-3),
    "n-tridecane": Component("n-tridecane", 675.000, 1.7100e6, 0.6040, 0.184367, 2220.0, 1.5e-3),
    "n-tetradecane": Component("n-tetradecane", 691.000, 1.6200e6, 0.6320, 0.198394, 2290.0, 1.8e-3),
    "n-pentadecane": Component("n-pentadecane", 706.500, 1.5200e6, 0.6600, 0.212420, 2360.0, 2.1e-3),
    "n-hexadecane": Component("n-hexadecane", 720.000, 1.4300e6, 0.6900, 0.226447, 2430.0, 2.5e-3),
    "n-heptadecane": Component("n-heptadecane", 735.000, 1.3500e6, 0.7200, 0.240474, 0.2500e4, 2.9e-3),
    "n-octadecane": Component("n-octadecane", 747.000, 1.2900e6, 0.7500, 0.254500, 2570.0, 3.4e-3),
    "n-nonadecane": Component("n-nonadecane", 760.000, 1.2300e6, 0.7800, 0.268527, 2640.0, 3.9e-3),
    "n-eicosane": Component("n-eicosane", 768.000, 1.1700e6, 0.8060, 0.282550, 2710.0, 4.5e-3),
    "n-heneicosane": Component("n-heneicosane", 782.000, 1.1200e6, 0.8300, 0.296577, 2780.0, 5.1e-3),
    "n-docosane": Component("n-docosane", 795.000, 1.0700e6, 0.8500, 0.310604, 2850.0, 5.8e-3),
    "n-tricosane": Component("n-tricosane", 807.000, 1.0300e6, 0.8700, 0.324631, 2920.0, 6.5e-3),
    "n-tetracosane": Component("n-tetracosane", 818.000, 0.9900e6, 0.8900, 0.338658, 2990.0, 7.2e-3),
    "n-pentacosane": Component("n-pentacosane", 829.000, 0.9500e6, 0.9100, 0.352685, 3060.0, 8.0e-3),
    "n-hexacosane": Component("n-hexacosane", 839.000, 0.9200e6, 0.9300, 0.366712, 3130.0, 8.8e-3),
    "n-heptacosane": Component("n-heptacosane", 849.000, 0.8900e6, 0.9500, 0.380739, 3200.0, 9.6e-3),
    "n-octacosane": Component("n-octacosane", 858.000, 0.8600e6, 0.9700, 0.394766, 3270.0, 1.04e-2),
    "n-nonacosane": Component("n-nonacosane", 867.000, 0.8300e6, 0.9900, 0.408793, 3340.0, 1.12e-2),
    "n-triacontane": Component("n-triacontane", 876.000, 0.8000e6, 1.0100, 0.422820, 3410.0, 1.20e-2),
    "nitrogen": Component("nitrogen", 126.192, 3.3958e6, 0.0372, 0.028013, 1040.0, 8.0e-5),
    "carbon-dioxide": Component("carbon-dioxide", 304.128, 7.3773e6, 0.2239, 0.044010, 850.0, 1.0e-4),
}


@dataclass(frozen=True)
class FlashPoint:
    vapor_fraction_molar: float
    vapor_fraction_mass: float
    liquid_composition: np.ndarray
    vapor_composition: np.ndarray
    rho_l: float
    rho_g: float
    mu_l: float
    mu_g: float
    h_l: float
    h_g: float
    cp_l: float
    cp_g: float
    sigma: float
    z_l: float
    z_g: float
    iterations: int
    converged: bool


class PengRobinsonFlashProvider(PropertyProvider):
    def __init__(
        self,
        composition: dict[str, float],
        binary_interactions: np.ndarray | None = None,
        max_flash_iterations: int = 80,
        flash_tolerance: float = 1.0e-9,
    ):
        unknown = set(composition) - set(COMPONENTS)
        if unknown:
            raise ValueError(f"本地组分库缺少: {sorted(unknown)}")
        self.components = [COMPONENTS[name] for name in composition]
        z = np.asarray(list(composition.values()), dtype=float)
        if np.any(z < 0.0) or z.sum() <= 0.0:
            raise ValueError("组分摩尔分数必须非负且总和大于 0")
        self.z = z / z.sum()
        n = len(z)
        self.kij = (
            np.zeros((n, n), dtype=float)
            if binary_interactions is None
            else np.asarray(binary_interactions, dtype=float)
        )
        if self.kij.shape != (n, n):
            raise ValueError(f"binary_interactions 必须为 {(n, n)}")
        self.max_flash_iterations = max_flash_iterations
        self.flash_tolerance = flash_tolerance
        self.tc = np.array([item.critical_temperature for item in self.components])
        self.pc = np.array([item.critical_pressure for item in self.components])
        self.omega = np.array([item.acentric_factor for item in self.components])
        self.mw = np.array([item.molecular_weight for item in self.components])
        self.cp = np.array([item.ideal_cp for item in self.components])
        self.mu_l_ref = np.array([item.liquid_viscosity_ref for item in self.components])

    @property
    def name(self) -> str:
        return "local-peng-robinson"

    def _pure_parameters(self, temperature: float) -> tuple[np.ndarray, ...]:
        m = 0.37464 + 1.54226 * self.omega - 0.26992 * self.omega**2
        sqrt_tr = np.sqrt(temperature / self.tc)
        sqrt_alpha = 1.0 + m * (1.0 - sqrt_tr)
        alpha = sqrt_alpha**2
        a0 = 0.45724 * R**2 * self.tc**2 / self.pc
        ai = a0 * alpha
        dai_dt = a0 * (-m * sqrt_alpha / (self.tc * sqrt_tr))
        bi = 0.07780 * R * self.tc / self.pc
        return ai, dai_dt, bi

    def _mixture_parameters(
        self, composition: np.ndarray, temperature: float
    ) -> tuple[float, float, float, np.ndarray, np.ndarray]:
        ai, dai_dt, bi = self._pure_parameters(temperature)
        aij = np.sqrt(np.outer(ai, ai)) * (1.0 - self.kij)
        daij_dt = 0.5 * aij * (
            dai_dt[:, None] / ai[:, None] + dai_dt[None, :] / ai[None, :]
        )
        a_mix = float(composition @ aij @ composition)
        da_mix_dt = float(composition @ daij_dt @ composition)
        b_mix = float(composition @ bi)
        return a_mix, da_mix_dt, b_mix, bi, aij

    @staticmethod
    def _z_roots(a_reduced: float, b_reduced: float) -> np.ndarray:
        coefficients = [
            1.0,
            -(1.0 - b_reduced),
            a_reduced - 3.0 * b_reduced**2 - 2.0 * b_reduced,
            -(a_reduced * b_reduced - b_reduced**2 - b_reduced**3),
        ]
        roots = np.roots(coefficients)
        real = np.sort(roots.real[np.abs(roots.imag) < 1.0e-9])
        valid = real[real > b_reduced + 1.0e-12]
        return valid if len(valid) else np.array([max(real[-1], b_reduced + 1.0e-9)])

    def _phase(
        self, composition: np.ndarray, pressure: float, temperature: float, vapor: bool
    ) -> tuple[float, np.ndarray, float, float, float]:
        a, da_dt, b, bi, aij = self._mixture_parameters(composition, temperature)
        a_reduced = a * pressure / (R**2 * temperature**2)
        b_reduced = b * pressure / (R * temperature)
        roots = self._z_roots(a_reduced, b_reduced)
        z_factor = float(roots[-1] if vapor else roots[0])
        log_ratio = np.log(
            (z_factor + (1.0 + SQRT2) * b_reduced)
            / (z_factor + (1.0 - SQRT2) * b_reduced)
        )
        sum_aij = aij @ composition
        if abs(b_reduced) < 1.0e-14 or a <= 0.0:
            ln_phi = np.zeros_like(composition)
        else:
            ln_phi = (
                bi / b * (z_factor - 1.0)
                - np.log(z_factor - b_reduced)
                - a_reduced
                / (2.0 * SQRT2 * b_reduced)
                * (2.0 * sum_aij / a - bi / b)
                * log_ratio
            )
        molecular_weight = float(composition @ self.mw)
        density = pressure * molecular_weight / (z_factor * R * temperature)
        cp_mass = float(composition @ (self.cp * self.mw) / molecular_weight)
        ideal_h = cp_mass * (temperature - 298.15)
        departure_molar = (
            R * temperature * (z_factor - 1.0)
            + (temperature * da_dt - a) / (2.0 * SQRT2 * b) * log_ratio
        )
        enthalpy = ideal_h + departure_molar / molecular_weight
        return z_factor, ln_phi, density, enthalpy, cp_mass

    @staticmethod
    def _rachford_rice(z: np.ndarray, k_values: np.ndarray) -> float:
        delta = k_values - 1.0
        f0 = float(np.sum(z * delta))
        f1 = float(np.sum(z * delta / np.maximum(1.0 + delta, 1.0e-15)))
        if f0 <= 0.0:
            return 0.0
        if f1 >= 0.0:
            return 1.0
        low, high = 0.0, 1.0
        for _ in range(100):
            vapor = 0.5 * (low + high)
            value = float(np.sum(z * delta / (1.0 + vapor * delta)))
            if value > 0.0:
                low = vapor
            else:
                high = vapor
            if high - low < 1.0e-13:
                break
        return 0.5 * (low + high)

    def flash_point(self, pressure: float, temperature: float) -> FlashPoint:
        wilson = self.pc / pressure * np.exp(
            5.373 * (1.0 + self.omega) * (1.0 - self.tc / temperature)
        )
        k_values = np.clip(wilson, 1.0e-8, 1.0e8)
        converged = False
        liquid = self.z.copy()
        vapor = self.z.copy()
        vapor_fraction = self._rachford_rice(self.z, k_values)

        for iteration in range(1, self.max_flash_iterations + 1):
            denominator = 1.0 + vapor_fraction * (k_values - 1.0)
            liquid = self.z / np.maximum(denominator, 1.0e-20)
            vapor = k_values * liquid
            liquid /= liquid.sum()
            vapor /= vapor.sum()
            _, ln_phi_l, _, _, _ = self._phase(liquid, pressure, temperature, False)
            _, ln_phi_g, _, _, _ = self._phase(vapor, pressure, temperature, True)
            log_k_new = np.clip(ln_phi_l - ln_phi_g, -30.0, 30.0)
            error = float(np.max(np.abs(log_k_new - np.log(k_values))))
            k_values = np.exp(0.5 * np.log(k_values) + 0.5 * log_k_new)
            vapor_fraction = self._rachford_rice(self.z, k_values)
            if error < self.flash_tolerance:
                converged = True
                break

        z_l, _, rho_l, h_l, cp_l = self._phase(liquid, pressure, temperature, False)
        z_g, _, rho_g, h_g, cp_g = self._phase(vapor, pressure, temperature, True)
        mw_l = float(liquid @ self.mw)
        mw_g = float(vapor @ self.mw)
        mass_denominator = vapor_fraction * mw_g + (1.0 - vapor_fraction) * mw_l
        vapor_mass = vapor_fraction * mw_g / max(mass_denominator, 1.0e-20)

        reduced_t = np.clip(temperature / float(liquid @ self.tc), 0.35, 1.5)
        mu_l = float(liquid @ self.mu_l_ref) * np.exp(2.5 * (1.0 / reduced_t - 1.0))
        mu_l = float(np.clip(mu_l, 8.0e-5, 0.2))
        mw_g_gmol = mw_g * 1000.0
        lee_k = (9.4 + 0.02 * mw_g_gmol) * temperature**1.5 / (
            209.0 + 19.0 * mw_g_gmol + temperature
        )
        lee_x = 3.5 + 986.0 / temperature + 0.01 * mw_g_gmol
        lee_y = 2.4 - 0.2 * lee_x
        mu_g = lee_k * 1.0e-7 * np.exp(lee_x * (rho_g / 1000.0) ** lee_y)
        mu_g = float(np.clip(mu_g, 5.0e-6, 1.0e-4))
        sigma = float(np.clip(0.035 * (1.0 - min(reduced_t, 0.999)) ** 1.2, 0.001, 0.04))
        return FlashPoint(
            vapor_fraction,
            vapor_mass,
            liquid,
            vapor,
            rho_l,
            rho_g,
            mu_l,
            mu_g,
            h_l,
            h_g,
            cp_l,
            cp_g,
            sigma,
            z_l,
            z_g,
            iteration,
            converged,
        )

    def evaluate(self, pressure: np.ndarray, temperature: np.ndarray) -> PropertyBatch:
        p, t = np.broadcast_arrays(
            np.asarray(pressure, dtype=float), np.asarray(temperature, dtype=float)
        )
        flat_results = [
            self.flash_point(float(pi), float(ti)) for pi, ti in zip(p.flat, t.flat)
        ]

        def collect(attribute: str) -> np.ndarray:
            return np.array([getattr(item, attribute) for item in flat_results]).reshape(p.shape)

        return PropertyBatch(
            rho_l=collect("rho_l"),
            rho_g=collect("rho_g"),
            mu_l=collect("mu_l"),
            mu_g=collect("mu_g"),
            h_l=collect("h_l"),
            h_g=collect("h_g"),
            cp_l=collect("cp_l"),
            cp_g=collect("cp_g"),
            sigma=collect("sigma"),
            gas_mass_fraction_eq=collect("vapor_fraction_mass"),
        )
