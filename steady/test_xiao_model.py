import unittest
import math

from steady.xiao_model import (
    XiaoInput,
    _bounded_unit_length_ratio,
    _friction_factor,
    _oliemans_FE,
    _slug_body_liquid_holdup,
    _stratified_fi_andritsos,
    _stratified_fi_xiao,
    _stratified_geometry,
    determine_flow_pattern,
    solve_intermittent,
    xiao_pressure_gradient,
)
from steady.olga_bubble_closure import solve_olga_bubble
from steady.batch_run_pr78 import _make_settings


class XiaoPaperClosureTests(unittest.TestCase):
    @staticmethod
    def _single_phase_input(gas_mass_fraction, **overrides):
        values = dict(
            total_mass_flow=2.0, gas_mass_fraction=gas_mass_fraction,
            rho_gas=12.0, rho_liquid=850.0,
            mu_gas=1.5e-5, mu_liquid=2.0e-3,
            diameter=0.1, roughness=1.0e-5,
            angle_rad=math.radians(6.0), surface_tension=0.025,
            pressure_Pa=2.0e6,
        )
        values.update(overrides)
        return XiaoInput(**values)

    def test_single_phase_guard_matches_liquid_darcy_limit(self):
        inp = self._single_phase_input(
            0.0, single_phase_limit_closure="darcy",
            flow_pattern_override="intermittent")
        result = xiao_pressure_gradient(inp)
        area = math.pi * inp.diameter ** 2 / 4.0
        velocity = inp.total_mass_flow / (inp.rho_liquid * area)
        reynolds = inp.rho_liquid * velocity * inp.diameter / inp.mu_liquid
        friction = _friction_factor(reynolds, inp.roughness / inp.diameter)
        expected_friction = (
            friction * inp.rho_liquid * velocity ** 2 / (2.0 * inp.diameter))
        expected_gravity = inp.rho_liquid * 9.80665 * math.sin(inp.angle_rad)
        self.assertEqual(result.flow_pattern, "single_phase_liquid")
        self.assertEqual(result.liquid_holdup_inclined, 1.0)
        self.assertEqual(result.gas_void_fraction, 0.0)
        self.assertAlmostEqual(result.dp_dL_friction, expected_friction, places=12)
        self.assertAlmostEqual(result.dp_dL_gravity, expected_gravity, places=12)
        self.assertAlmostEqual(
            result.dp_dL_total, expected_friction + expected_gravity, places=12)

    def test_single_phase_guard_matches_gas_darcy_limit(self):
        inp = self._single_phase_input(
            1.0, single_phase_limit_closure="darcy",
            flow_pattern_override="annular")
        result = xiao_pressure_gradient(inp)
        area = math.pi * inp.diameter ** 2 / 4.0
        velocity = inp.total_mass_flow / (inp.rho_gas * area)
        reynolds = inp.rho_gas * velocity * inp.diameter / inp.mu_gas
        friction = _friction_factor(reynolds, inp.roughness / inp.diameter)
        expected_friction = (
            friction * inp.rho_gas * velocity ** 2 / (2.0 * inp.diameter))
        expected_gravity = inp.rho_gas * 9.80665 * math.sin(inp.angle_rad)
        self.assertEqual(result.flow_pattern, "single_phase_gas")
        self.assertEqual(result.liquid_holdup_inclined, 0.0)
        self.assertEqual(result.gas_void_fraction, 1.0)
        self.assertAlmostEqual(result.dp_dL_friction, expected_friction, places=12)
        self.assertAlmostEqual(result.dp_dL_gravity, expected_gravity, places=12)

    def test_single_phase_guard_is_opt_in_and_validated(self):
        legacy = xiao_pressure_gradient(self._single_phase_input(0.0))
        guarded = xiao_pressure_gradient(self._single_phase_input(
            0.0, single_phase_limit_closure="darcy"))
        self.assertNotEqual(legacy.flow_pattern, "single_phase_liquid")
        self.assertEqual(guarded.flow_pattern, "single_phase_liquid")
        with self.assertRaisesRegex(ValueError, "单相极限闭合"):
            xiao_pressure_gradient(self._single_phase_input(
                0.0, single_phase_limit_closure="not_a_model"))

    def test_single_phase_guard_is_identical_inside_two_phase_region(self):
        legacy = xiao_pressure_gradient(self._single_phase_input(0.2))
        guarded = xiao_pressure_gradient(self._single_phase_input(
            0.2, single_phase_limit_closure="darcy"))
        self.assertEqual(guarded, legacy)

    def test_slug_body_holdup_closures_match_published_equations(self):
        common = dict(
            vm=1.5, rho_l=900.0, rho_g=1.3,
            mu_l=0.5, mu_g=2.0e-5, D=0.08,
            theta_rad=math.radians(30.0),
        )
        gomez = _slug_body_liquid_holdup("gomez2000", **common)
        re_slug = common["D"] * common["rho_l"] * common["vm"] / common["mu_l"]
        expected_gomez = math.exp(-(0.00784 * 30.0 + 2.48e-6 * re_slug))
        self.assertAlmostEqual(gomez, expected_gomez, places=12)

        unified = _slug_body_liquid_holdup("abdul_majeed2019", **common)
        drho = common["rho_l"] - common["rho_g"]
        n_fr = common["vm"] / math.sqrt(9.80665 * common["D"]) * math.sqrt(common["rho_l"] / drho)
        n_mu = common["vm"] * common["mu_l"] / (9.80665 * common["D"] ** 2 * drho)
        x_value = n_fr / n_mu ** 0.2
        expected_unified = 1.016 - 0.000611 * 30.0 + (0.000124 * 30.0 - 0.0195) * x_value
        self.assertAlmostEqual(unified, expected_unified, places=12)

    def test_unit_length_ratio_stays_on_long_film_branch_through_zero(self):
        self.assertEqual(_bounded_unit_length_ratio(0.2, 0.0), 20.0)
        self.assertEqual(_bounded_unit_length_ratio(0.2, 1.0e-10), 20.0)
        self.assertEqual(_bounded_unit_length_ratio(0.2, -1.0e-10), 20.0)
        self.assertAlmostEqual(_bounded_unit_length_ratio(0.2, 0.02), 10.0)
        self.assertAlmostEqual(_bounded_unit_length_ratio(0.2, -0.02), 10.0)

    def test_friction_factor_satisfies_xiao_equation_14(self):
        reynolds, rr = 1.0e5, 1.0e-3
        f_darcy = _friction_factor(reynolds, rr)
        residual = (
            1.0 / math.sqrt(f_darcy) - 1.74
            + 2.0 * math.log10(2.0 * rr + 18.7 / (reynolds * math.sqrt(f_darcy)))
        )
        self.assertAlmostEqual(residual, 0.0, places=12)

    def test_oliemans_full_dataset_entrainment_coefficients(self):
        # Xiao Eq. (51) / Oliemans et al. full-data regression, SI units.
        fe = _oliemans_FE(800.0, 20.0, 3.0e-3, 1.2e-5, 0.02, 1.0e-3, 5.0, 0.1)
        self.assertAlmostEqual(fe, 0.2366065766, places=9)

    def test_small_diameter_stratified_closure_uses_andritsos(self):
        geom = _stratified_geometry(0.2, 0.1)
        actual = _stratified_fi_xiao(
            vsg=4.0, VG=5.0, VL=0.2, hL_tilde=0.2, geom=geom, fG=0.02,
            rho_l=800.0, rho_g=20.0, mu_l=3.0e-3, mu_g=1.2e-5,
            D=0.1, eps=1.0e-5, sigma=0.02, pressure_Pa=2.0e6,
        )
        expected = _stratified_fi_andritsos(4.0, 0.2, 0.02, 2.0e6)
        self.assertAlmostEqual(actual, expected, places=12)

    def test_flow_pattern_override_separates_classification_from_closure(self):
        inp = XiaoInput(
            total_mass_flow=10.0, gas_mass_fraction=0.05,
            rho_gas=20.0, rho_liquid=800.0,
            mu_gas=1.0e-5, mu_liquid=3.0e-3,
            diameter=0.15, roughness=1.0e-5,
            angle_rad=0.1, surface_tension=0.025,
            pressure_Pa=5.0e6, flow_pattern_override="dispersed_bubble",
        )
        result = xiao_pressure_gradient(inp)
        self.assertEqual(result.flow_pattern, "dispersed_bubble")

    def test_flow_pattern_candidates_are_independently_selectable(self):
        common = dict(
            vsg=0.2, vsl=2.0, vm=2.2,
            rho_l=850.0, rho_g=10.0,
            mu_l=2.0e-3, mu_g=1.5e-5,
            D=0.1, eps=1.0e-5, theta_rad=0.0, sigma=0.025,
        )
        baseline = determine_flow_pattern(**common)
        full_mb = determine_flow_pattern(**common, closure="mukherjee_brill")
        hybrid = determine_flow_pattern(**common, closure="xiao_mb_bubble")
        replacement = determine_flow_pattern(
            **common, closure="xiao_mb_bubble_gate")
        hierarchical = determine_flow_pattern(
            **common, closure="xiao_mb_bubble_from_intermittent")
        td_hierarchical = determine_flow_pattern(
            **common, closure="xiao_td_bubble_from_intermittent")
        self.assertIn(baseline, {
            "stratified", "stratified_smooth", "stratified_wavy",
            "intermittent", "annular", "dispersed_bubble",
        })
        self.assertIn(full_mb, {
            "stratified", "intermittent", "annular", "dispersed_bubble",
        })
        if full_mb == "dispersed_bubble":
            self.assertEqual(hybrid, "dispersed_bubble")
            self.assertEqual(replacement, "dispersed_bubble")
        else:
            self.assertEqual(hybrid, baseline)
            self.assertNotEqual(replacement, "dispersed_bubble")
        if baseline == "intermittent" and full_mb == "dispersed_bubble":
            self.assertEqual(hierarchical, "dispersed_bubble")
        else:
            self.assertEqual(hierarchical, baseline)
        self.assertIn(td_hierarchical, {
            "stratified", "stratified_smooth", "stratified_wavy",
            "intermittent", "annular", "dispersed_bubble",
        })
        with self.assertRaisesRegex(ValueError, "流型判别闭合"):
            determine_flow_pattern(**common, closure="not_a_model")

    def test_td_hierarchical_candidate_promotes_reference_bubble_case(self):
        common = dict(
            vsg=0.974, vsl=3.048, vm=4.022,
            rho_l=857.52581, rho_g=4.036935,
            mu_l=0.0066, mu_g=1.8e-5,
            D=0.0258, eps=0.0, theta_rad=0.0, sigma=0.0309,
        )
        self.assertEqual(determine_flow_pattern(**common), "intermittent")
        self.assertEqual(determine_flow_pattern(
            **common, closure="xiao_td_bubble_from_intermittent"),
            "dispersed_bubble",
        )

    def test_mb_flow_pattern_candidates_fall_back_at_single_phase_limit(self):
        common = dict(
            vsg=0.0, vsl=1.0, vm=1.0,
            rho_l=850.0, rho_g=10.0,
            mu_l=2.0e-3, mu_g=1.5e-5,
            D=0.1, eps=1.0e-5, theta_rad=0.0, sigma=0.025,
        )
        baseline = determine_flow_pattern(**common)
        for closure in (
            "mukherjee_brill", "xiao_mb_bubble",
            "xiao_mb_bubble_gate", "xiao_mb_bubble_from_intermittent",
            "xiao_td_bubble_from_intermittent",
        ):
            self.assertEqual(
                determine_flow_pattern(**common, closure=closure), baseline)

    def test_olga_bubble_void_fraction_closes_superficial_gas_flux(self):
        result = solve_olga_bubble(
            vsg=0.5, vsl=1.5, rho_l=800.0, rho_g=20.0,
            mu_l=3.0e-3, mu_g=1.0e-5, diameter=0.15,
            roughness=1.0e-5, theta_rad=math.radians(10.0), sigma=0.025,
        )
        self.assertTrue(result["converged"])
        self.assertGreater(result["EG"], 0.0)
        self.assertLess(result["EG"], 1.0)
        self.assertAlmostEqual(result["EG"] * result["v_gas_actual"], 0.5, places=12)

    def test_olga_closure_is_opt_in(self):
        common = dict(
            total_mass_flow=10.0, gas_mass_fraction=0.05,
            rho_gas=20.0, rho_liquid=800.0,
            mu_gas=1.0e-5, mu_liquid=3.0e-3,
            diameter=0.15, roughness=1.0e-5,
            angle_rad=math.radians(10.0), surface_tension=0.025,
            pressure_Pa=5.0e6, flow_pattern_override="dispersed_bubble",
        )
        baseline = xiao_pressure_gradient(XiaoInput(**common))
        olga = xiao_pressure_gradient(XiaoInput(
            **common, dispersed_bubble_closure="olga_v38_v6"))
        self.assertNotAlmostEqual(
            baseline.liquid_holdup_inclined,
            olga.liquid_holdup_inclined,
            places=8,
        )

    def test_stage_closure_options_are_independently_selectable(self):
        common = dict(
            total_mass_flow=2.0, gas_mass_fraction=0.2,
            rho_gas=10.0, rho_liquid=850.0,
            mu_gas=1.5e-5, mu_liquid=2.0e-3,
            diameter=0.1, roughness=1.0e-5,
            angle_rad=0.0, surface_tension=0.025,
            pressure_Pa=2.0e6,
        )
        stratified = xiao_pressure_gradient(XiaoInput(
            **common, flow_pattern_override="stratified",
            stratified_interfacial_closure="olga_slai"))
        intermittent = xiao_pressure_gradient(XiaoInput(
            **common, flow_pattern_override="intermittent",
            intermittent_translation_closure="olga_horizontal"))
        annular = xiao_pressure_gradient(XiaoInput(
            **common, flow_pattern_override="annular",
            annular_entrainment_closure="olga_generalized"))
        intermittent_film = xiao_pressure_gradient(XiaoInput(
            **common, flow_pattern_override="intermittent",
            intermittent_translation_closure="petalas",
            intermittent_film_interfacial_closure="olga_slai"))
        annular_wallis = xiao_pressure_gradient(XiaoInput(
            **common, flow_pattern_override="annular",
            annular_entrainment_closure="fixed_25",
            annular_interfacial_closure="wallis"))
        bubble_petalas = xiao_pressure_gradient(XiaoInput(
            **common, flow_pattern_override="dispersed_bubble",
            dispersed_bubble_closure="petalas_dispersed"))
        for result in (
            stratified, intermittent, annular, intermittent_film,
            annular_wallis, bubble_petalas,
        ):
            self.assertTrue(math.isfinite(result.dp_dL_total))
            self.assertGreater(result.liquid_holdup_inclined, 0.0)
            self.assertLess(result.liquid_holdup_inclined, 1.0)

    def test_pipeline_xiao_profiles_preserve_baseline_and_expose_candidates(self):
        baseline = _make_settings("xiao", 8, "baseline")
        recommended = _make_settings("xiao", 8, "recommended")
        smooth9 = _make_settings("xiao", 8, "smooth9_candidate")
        smooth9_abdul = _make_settings("xiao", 8, "smooth9_abdul2000")
        single_phase_guard = _make_settings("xiao", 8, "single_phase_guard")
        smooth9_abdul_guard = _make_settings(
            "xiao", 8, "smooth9_abdul2000_phase_guard")
        combo = _make_settings("xiao", 8, "combo")
        bubble = _make_settings("xiao", 8, "bubble_candidate")
        flow_mb = _make_settings("xiao", 8, "flow_mb")
        flow_mb_bubble = _make_settings("xiao", 8, "flow_mb_bubble")
        flow_mb_bubble_gate = _make_settings("xiao", 8, "flow_mb_bubble_gate")
        flow_mb_hierarchical = _make_settings(
            "xiao", 8, "flow_mb_bubble_from_intermittent")
        flow_td_hierarchical = _make_settings(
            "xiao", 8, "flow_td_bubble_from_intermittent")
        flow_td_combined = _make_settings(
            "xiao", 8, "flow_td_hydraulic_combined")
        flow_hierarchical_combined = _make_settings(
            "xiao", 8, "flow_hierarchical_combined")
        self.assertEqual(baseline.xiao_intermittent_translation_closure, "xiao")
        self.assertEqual(baseline.xiao_dispersed_bubble_closure, "xiao")
        self.assertEqual(
            recommended.xiao_intermittent_translation_closure,
            "olga_density_scaled_gated9",
        )
        self.assertEqual(recommended.xiao_dispersed_bubble_closure, "xiao")
        self.assertEqual(
            smooth9.xiao_intermittent_translation_closure,
            "olga_density_scaled_smooth9",
        )
        self.assertEqual(
            smooth9_abdul.xiao_intermittent_slug_holdup_closure,
            "abdul_majeed2000",
        )
        self.assertEqual(baseline.xiao_single_phase_limit_closure, "legacy")
        self.assertEqual(
            single_phase_guard.xiao_single_phase_limit_closure, "darcy")
        self.assertEqual(
            smooth9_abdul_guard.xiao_single_phase_limit_closure, "darcy")
        self.assertEqual(
            smooth9_abdul_guard.xiao_intermittent_translation_closure,
            "olga_density_scaled_smooth9",
        )
        self.assertEqual(
            smooth9_abdul_guard.xiao_intermittent_slug_holdup_closure,
            "abdul_majeed2000",
        )
        self.assertEqual(combo.xiao_intermittent_film_interfacial_closure, "petalas")
        self.assertEqual(bubble.xiao_dispersed_bubble_closure, "petalas_dispersed")
        self.assertEqual(flow_mb.xiao_flow_pattern_closure, "mukherjee_brill")
        self.assertEqual(flow_mb_bubble.xiao_flow_pattern_closure, "xiao_mb_bubble")
        self.assertEqual(
            flow_mb_bubble_gate.xiao_flow_pattern_closure,
            "xiao_mb_bubble_gate",
        )
        self.assertEqual(
            flow_mb_hierarchical.xiao_flow_pattern_closure,
            "xiao_mb_bubble_from_intermittent",
        )
        self.assertEqual(
            flow_td_hierarchical.xiao_flow_pattern_closure,
            "xiao_td_bubble_from_intermittent",
        )
        self.assertEqual(
            flow_td_combined.xiao_flow_pattern_closure,
            "xiao_td_bubble_from_intermittent",
        )
        self.assertEqual(
            flow_td_combined.xiao_intermittent_translation_closure,
            "olga_density_scaled_smooth9",
        )
        self.assertEqual(
            flow_td_combined.xiao_dispersed_bubble_closure,
            "petalas_dispersed",
        )
        self.assertEqual(
            flow_hierarchical_combined.xiao_flow_pattern_closure,
            "xiao_mb_bubble_from_intermittent",
        )
        self.assertEqual(
            flow_hierarchical_combined.xiao_intermittent_translation_closure,
            "olga_density_scaled_smooth9",
        )
        self.assertEqual(
            flow_hierarchical_combined.xiao_dispersed_bubble_closure,
            "petalas_dispersed",
        )

    def test_gated_olga_translation_falls_back_above_nine_degrees(self):
        common = dict(
            total_mass_flow=2.0, gas_mass_fraction=0.2,
            rho_gas=10.0, rho_liquid=850.0,
            mu_gas=1.5e-5, mu_liquid=2.0e-3,
            diameter=0.1, roughness=1.0e-5,
            surface_tension=0.025, pressure_Pa=2.0e6,
            flow_pattern_override="intermittent",
        )
        baseline_high = xiao_pressure_gradient(XiaoInput(
            **common, angle_rad=math.radians(10.0)))
        gated_high = xiao_pressure_gradient(XiaoInput(
            **common, angle_rad=math.radians(10.0),
            intermittent_translation_closure="olga_density_scaled_gated9"))
        baseline_low = xiao_pressure_gradient(XiaoInput(
            **common, angle_rad=math.radians(5.0)))
        gated_low = xiao_pressure_gradient(XiaoInput(
            **common, angle_rad=math.radians(5.0),
            intermittent_translation_closure="olga_density_scaled_gated9"))
        self.assertAlmostEqual(
            baseline_high.liquid_holdup_inclined,
            gated_high.liquid_holdup_inclined,
            places=12,
        )
        self.assertNotAlmostEqual(
            baseline_low.liquid_holdup_inclined,
            gated_low.liquid_holdup_inclined,
            places=8,
        )

    def test_smooth_olga_translation_blends_across_nine_degrees(self):
        common = dict(
            vsg=1.0, vsl=0.5, vm=1.5,
            rho_l=850.0, rho_g=10.0,
            mu_l=2.0e-3, mu_g=1.5e-5,
            D=0.1, eps=1.0e-5, sigma=0.025,
            P_Pa=2.0e6,
        )

        def vt(angle_deg, closure):
            return solve_intermittent(
                **common, theta_rad=math.radians(angle_deg),
                translation_closure=closure,
            )["Vt"]

        self.assertAlmostEqual(
            vt(5.0, "olga_density_scaled_smooth9"),
            vt(5.0, "olga_density_scaled"), places=12)
        self.assertAlmostEqual(
            vt(12.0, "olga_density_scaled_smooth9"),
            vt(12.0, "xiao"), places=12)
        self.assertAlmostEqual(
            vt(9.0, "olga_density_scaled_smooth9"),
            0.5 * (vt(9.0, "olga_density_scaled") + vt(9.0, "xiao")),
            places=12,
        )

    def test_theron_translation_matches_review_equation(self):
        angle_rad = math.radians(30.0)
        result = solve_intermittent(
            vsg=1.0, vsl=0.5, vm=1.5,
            rho_l=850.0, rho_g=10.0,
            mu_l=2.0e-3, mu_g=1.5e-5,
            D=0.1, eps=1.0e-5, theta_rad=angle_rad,
            sigma=0.025, P_Pa=2.0e6,
            translation_closure="theron1989",
        )
        gamma = 1.0 + (0.5 / math.sqrt(9.80665 * 0.1)) * math.cos(angle_rad) / 3.5
        c0 = 1.3 - 0.23 / gamma + 0.13 * math.sin(angle_rad) ** 2
        c1 = (-0.5 + 0.8 / gamma) * math.cos(angle_rad) + 0.35 * math.sin(angle_rad)
        expected = c0 * 1.5 + c1 * math.sqrt(9.80665 * 0.1)
        self.assertAlmostEqual(result["Vt"], expected, places=12)
