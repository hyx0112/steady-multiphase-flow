import unittest

from steady.beggs_brill import BeggsBrillInput, beggs_brill_pressure_gradient


class BeggsBrillEquationTests(unittest.TestCase):
    def test_acceleration_term_uses_average_absolute_pressure(self):
        result = beggs_brill_pressure_gradient(BeggsBrillInput(
            total_mass_flow=8.0,
            gas_mass_fraction=0.8,
            rho_gas=25.0,
            rho_liquid=700.0,
            mu_gas=1.2e-5,
            mu_liquid=1.0e-3,
            diameter=0.15,
            roughness=4.5e-5,
            angle_rad=0.03,
            surface_tension=0.025,
            pressure_avg_Pa=2.0e6,
        ))
        ek = result.rho_no_slip * result.v_m * result.v_sg / 2.0e6
        expected = (result.dp_dL_friction + result.dp_dL_gravity) / (1.0 - ek)
        self.assertGreater(ek, 0.0)
        self.assertAlmostEqual(result.dp_dL_total, expected, places=10)

