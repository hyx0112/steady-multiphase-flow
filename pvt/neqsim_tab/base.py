from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PropertyBatch:
    rho_l: np.ndarray
    rho_g: np.ndarray
    mu_l: np.ndarray
    mu_g: np.ndarray
    h_l: np.ndarray
    h_g: np.ndarray
    cp_l: np.ndarray
    cp_g: np.ndarray
    sigma: np.ndarray
    gas_mass_fraction_eq: np.ndarray


@dataclass(frozen=True)
class PropertyDerivatives:
    """First P/T derivatives generated from the same property representation."""

    d_dp: PropertyBatch
    d_dT: PropertyBatch


class PropertyProvider(ABC):
    @abstractmethod
    def evaluate(self, pressure: np.ndarray, temperature: np.ndarray) -> PropertyBatch:
        """以 SI 单位批量返回气液两相物性。"""

    def evaluate_with_derivatives(
        self, pressure: np.ndarray, temperature: np.ndarray
    ) -> tuple[PropertyBatch, PropertyDerivatives]:
        """Return values and consistent first derivatives.

        Generic providers use centred finite differences. Table/EOS backends
        should override this method with analytic or automatic derivatives.
        """
        p, t = np.broadcast_arrays(
            np.asarray(pressure, dtype=float), np.asarray(temperature, dtype=float)
        )
        dp = np.maximum(np.abs(p) * 1.0e-5, 10.0)
        dT = np.maximum(np.abs(t) * 1.0e-5, 1.0e-3)
        values = self.evaluate(p, t)
        plus_p, minus_p = self.evaluate(p + dp, t), self.evaluate(p - dp, t)
        plus_t, minus_t = self.evaluate(p, t + dT), self.evaluate(p, t - dT)

        def difference(plus: PropertyBatch, minus: PropertyBatch, step: np.ndarray):
            return PropertyBatch(**{
                name: (getattr(plus, name) - getattr(minus, name)) / (2.0 * step)
                for name in PropertyBatch.__dataclass_fields__
            })

        return values, PropertyDerivatives(
            d_dp=difference(plus_p, minus_p, dp),
            d_dT=difference(plus_t, minus_t, dT),
        )

    @property
    @abstractmethod
    def name(self) -> str:
        ...
