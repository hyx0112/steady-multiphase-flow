"""Mukherjee--Brill (1985) equation-level regression tests."""

from __future__ import annotations

import math
import unittest
from unittest.mock import patch

from steady import mukherjee_brill as mb


def _input(**overrides: float) -> mb.MukherjeeBrillInput:
    values = dict(
        total_mass_flow=1.0,
        gas_mass_fraction=0.10,
        rho_gas=10.0,
        rho_liquid=800.0,
        mu_gas=1.5e-5,
        mu_liquid=1.0e-3,
        diameter=0.10,
        roughness=1.0e-5,
        angle_rad=0.0,
        surface_tension=0.03,
        pressure_avg=1.0e6,
    )
    values.update(overrides)
    return mb.MukherjeeBrillInput(**values)


class MukherjeeBrillPaperEquationTests(unittest.TestCase):
    def test_paper_moody_equation_is_solved_directly(self) -> None:
        reynolds = 1.0e5
        relative_roughness = 1.0e-3
        f = mb._colebrook(reynolds, relative_roughness)
        residual = (
            1.0 / math.sqrt(f) - 1.74
            + 2.0 * math.log10(
                relative_roughness / 2.0 + 18.7 / (reynolds * math.sqrt(f))
            )
        )
        self.assertAlmostEqual(residual, 0.0, places=12)

    def test_horizontal_flow_does_not_use_downhill_stratified_boundary(self) -> None:
        # 此点满足 Eq. (9)，但 theta=0 时论文要求走 Eq. (8)。
        self.assertEqual(
            mb._predict_flow_pattern(0.1, 0.001, 0.1, 0.0),
            "bubble",
        )

    def test_liquid_viscosity_number_uses_gravity(self) -> None:
        _, _, n_l = mb._dimensionless_numbers(2.0, 1.0, 800.0, 1.0e-3, 0.03)
        expected = 1.0e-3 * (9.80665 / (800.0 * 0.03 ** 3)) ** 0.25
        self.assertAlmostEqual(n_l, expected, places=15)

    @patch.object(mb, "_calc_holdup", return_value=0.01)
    @patch.object(mb, "_predict_flow_pattern", return_value="slug")
    def test_holdup_is_not_forced_to_no_slip_value(self, *_: object) -> None:
        result = mb.mukherjee_brill_pressure_gradient(
            _input(gas_mass_fraction=0.50)
        )
        self.assertGreater(result.no_slip_liquid_fraction, 0.01)
        self.assertEqual(result.liquid_holdup_inclined, 0.01)

    @patch.object(mb, "_calc_holdup", return_value=0.40)
    @patch.object(mb, "_predict_flow_pattern", return_value="annular-mist")
    def test_annular_holdup_ratio_is_lambda_over_holdup(self, *_: object) -> None:
        result = mb.mukherjee_brill_pressure_gradient(_input())
        expected_hr = result.no_slip_liquid_fraction / 0.40
        self.assertAlmostEqual(
            result.friction_multiplier, mb._interp_fR(expected_hr), places=14
        )

    @patch.object(mb, "_colebrook", return_value=0.02)
    @patch.object(mb, "_calc_holdup", return_value=0.50)
    @patch.object(mb, "_predict_flow_pattern", return_value="stratified")
    def test_stratified_wall_shear_uses_darcy_consistent_form(self, *_: object) -> None:
        result = mb.mukherjee_brill_pressure_gradient(_input())
        d = 0.10
        _, _, _, s_l, s_g, _ = mb._stratified_geometry(0.50, d)
        area = math.pi * d ** 2 / 4.0
        tau_l = 0.02 * 800.0 * (result.v_sl / 0.50) ** 2 / 8.0
        tau_g = 0.02 * 10.0 * (result.v_sg / 0.50) ** 2 / 8.0
        expected = (tau_l * s_l + tau_g * s_g) / area
        self.assertAlmostEqual(result.dp_dL_friction, expected, places=12)

    @patch.object(mb, "_calc_holdup", return_value=1.01)
    @patch.object(mb, "_predict_flow_pattern", return_value="slug")
    def test_out_of_range_holdup_is_reported_not_silently_clipped(self, *_: object) -> None:
        with self.assertRaisesRegex(ValueError, "超出"):
            mb.mukherjee_brill_pressure_gradient(_input())


if __name__ == "__main__":
    unittest.main()
