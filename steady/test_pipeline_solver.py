import unittest
from types import SimpleNamespace
from unittest.mock import patch

from steady.pipeline_solver import (
    PipelineForwardBoundary,
    solve_pipe_section_forward,
    solve_pipeline_with_fixed_inlet_temperature_and_target_outlet_pressure,
)
from steady.steady_types import PipeSection, PipelineCommonSettings


class SafeguardedShootingTests(unittest.TestCase):
    def test_unconverged_micro_segment_is_not_accepted_as_forward_solution(self):
        section = PipeSection(
            section_no=1, outlet_node=1, section_length_m=10.0,
            elevation_change_m=0.0, n_subsegments=1, sub_length_m=10.0,
            roughness_m=1.0e-5, overall_heat_transfer=0.0,
            mass_flow_kg_s=1.0, diameter_m=0.1, env_temp_K=293.15,
        )
        settings = PipelineCommonSettings(micro_max_iter=1)
        failed = SimpleNamespace(converged=False)
        with patch("steady.pipeline_solver.solve_micro_segment_forward", return_value=failed):
            with self.assertRaisesRegex(ArithmeticError, "未收敛"):
                solve_pipe_section_forward(
                    pvt=object(), section=section,
                    section_position={"x_start_m": 0.0, "elevation_start_m": 0.0},
                    pressure_inlet_bara=100.0, temperature_inlet_K=300.0,
                    settings=settings,
                )

    def test_bracketed_shooting_keeps_trials_inside_valid_sign_interval(self):
        trials = []

        def fake_forward(_pvt, _sections, inlet_pressure, inlet_temperature, _settings, **_kwargs):
            trials.append(inlet_pressure)
            # Root at 100.25 bara: initial P0=100 and P1=100.5 bracket it.
            return inlet_pressure - 0.25, inlet_temperature, []

        with patch("steady.pipeline_solver.solve_pipeline_forward_given_inlet_pressure", fake_forward):
            result = solve_pipeline_with_fixed_inlet_temperature_and_target_outlet_pressure(
                pvt=object(),
                sections=[object()],
                boundary=PipelineForwardBoundary(300.0, 100.0),
                settings=object(),
                inlet_pressure_guess_bara=100.0,
                inlet_pressure_low_bara=90.0,
                inlet_pressure_high_bara=150.0,
                shooting_tol_bara=1.0e-8,
                max_shooting_iter=6,
                debug=False,
            )

        self.assertTrue(result.converged)
        self.assertAlmostEqual(result.inlet_pressure_bara, 100.25, places=8)
        self.assertTrue(all(100.0 <= point <= 100.5 for point in trials))
