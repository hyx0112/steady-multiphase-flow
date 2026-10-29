import math
import unittest

from steady.flow_pattern_candidates import (
    _darcy_friction_factor,
    group_mandhane_pattern,
    group_taitel_dukler_pattern,
    mandhane_gregory_aziz_horizontal,
    taitel_dukler_from_superficial_velocities,
)


class MandhaneFlowMapTests(unittest.TestCase):
    def test_reference_case_matches_fluids_upstream(self):
        # Upstream example is specified by mass flow/quality.  Its converted
        # superficial velocities are vSL=0.296522 m/s and vSG=12.818 m/s.
        raw = mandhane_gregory_aziz_horizontal(
            vsg=12.818231821153995,
            vsl=0.29652156195248475,
            rho_g=2.67,
            rho_l=915.12,
            mu_g=14.0e-6,
            mu_l=180.0e-6,
            sigma=0.065,
        )
        self.assertEqual(raw, "slug")
        self.assertEqual(group_mandhane_pattern(raw), "intermittent")

    def test_invalid_single_phase_input_is_explicit(self):
        with self.assertRaisesRegex(ValueError, "vsg and vsl"):
            mandhane_gregory_aziz_horizontal(
                vsg=0.0, vsl=1.0, rho_g=1.2, rho_l=1000.0,
                mu_g=1.8e-5, mu_l=1.0e-3, sigma=0.072,
            )


class TaitelDuklerFlowMapTests(unittest.TestCase):
    def test_blasius_option_uses_darcy_convention(self):
        reynolds = 1.0e5
        self.assertAlmostEqual(
            _darcy_friction_factor(reynolds, 0.0, model="blasius"),
            0.3164 * reynolds ** -0.25,
            places=14,
        )
        with self.assertRaisesRegex(ValueError, "friction model"):
            _darcy_friction_factor(reynolds, 0.0, model="invalid")

    def test_reference_case_matches_fluids_upstream(self):
        # fluids upstream example: m=0.6 kg/s, x=0.112, D=0.05 m.
        result = taitel_dukler_from_superficial_velocities(
            vsg=12.818231821153995,
            vsl=0.29652156195248475,
            rho_g=2.67,
            rho_l=915.12,
            mu_g=14.0e-6,
            mu_l=180.0e-6,
            diameter=0.05,
            roughness=0.0,
            theta_rad=0.0,
        )
        self.assertEqual(result.pattern, "annular")
        self.assertEqual(group_taitel_dukler_pattern(result.pattern), "annular")
        self.assertAlmostEqual(result.f, 0.9902249725092789, places=12)
        self.assertAlmostEqual(result.k, 271.86280111125365, places=10)
        self.assertAlmostEqual(result.t, 0.04144054776101148, places=12)
        self.assertAlmostEqual(result.x, 0.4505119305984412, places=12)

    def test_upstream_reference_regimes(self):
        cases = (
            (1.0, 0.05, 600.12, 80.67, 0.02, "bubbly"),
            (1.0, 0.05, 600.12, 80.67, 0.021, "intermittent"),
            (0.06, 0.5, 900.12, 90.67, 0.05, "stratified_smooth"),
            (0.07, 0.5, 900.12, 90.67, 0.05, "stratified_wavy"),
        )
        for mass_flow, quality, rho_l, rho_g, diameter, expected in cases:
            area = math.pi * diameter * diameter / 4.0
            result = taitel_dukler_from_superficial_velocities(
                vsg=mass_flow * quality / (rho_g * area),
                vsl=mass_flow * (1.0 - quality) / (rho_l * area),
                rho_g=rho_g,
                rho_l=rho_l,
                mu_g=14.0e-6,
                mu_l=180.0e-6,
                diameter=diameter,
                roughness=0.0,
                theta_rad=0.0,
            )
            self.assertEqual(result.pattern, expected)

    def test_invalid_single_phase_input_is_explicit(self):
        with self.assertRaisesRegex(ValueError, "vsg and vsl"):
            taitel_dukler_from_superficial_velocities(
                vsg=0.0, vsl=1.0, rho_g=1.2, rho_l=1000.0,
                mu_g=1.8e-5, mu_l=1.0e-3, diameter=0.05,
                roughness=0.0, theta_rad=0.0,
            )


if __name__ == "__main__":
    unittest.main()
