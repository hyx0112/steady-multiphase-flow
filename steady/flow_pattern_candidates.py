"""Independent literature flow-pattern candidates used by Xiao research.

These functions are kept outside :mod:`steady.xiao_model` until a candidate
passes classification, hydraulic-coupling, and pipeline-stability gates.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import math


G = 9.80665


@dataclass(frozen=True)
class TaitelDuklerResult:
    """Taitel--Dukler classification and auditable boundary diagnostics."""

    pattern: str
    x: float
    t: float
    f: float
    k: float
    boundary_a: float
    boundary_c: float
    boundary_d: float


# Cubic log-log splines digitized from the Taitel--Dukler map by ``fluids``.
# They are kept here as an independent reference implementation; provenance
# and the upstream MIT license are recorded in THIRD_PARTY_NOTICES.md.
_TD_A_T = (
    -2.4791105294648372, -2.4791105294648372, -2.4791105294648372,
    -2.4791105294648372, 0.14360803483759585, 1.7199938263676038,
    1.7199938263676038, 1.7199938263676038, 1.7199938263676038,
)
_TD_A_C = (
    0.21299880246561081, 0.16299733301915248,
    -0.042340970712679615, -1.9967836909384598,
    -2.9917366639619414, 0.0, 0.0, 0.0, 0.0,
)
_TD_C_T = (
    -1.8323873272724698, -1.8323873272724698, -1.8323873272724698,
    -1.8323873272724698, -0.15428198203334137, 1.7031193462360779,
    1.7031193462360779, 1.7031193462360779, 1.7031193462360779,
)
_TD_C_C = (
    0.2827776229531682, 0.6207113329042158, 1.0609541626742232,
    0.44917638072891825, 0.014664597632360495,
    0.0, 0.0, 0.0, 0.0,
)
_TD_D_T = (
    0.2532652936901574, 0.2532652936901574, 0.2532652936901574,
    0.2532652936901574, 3.5567847823070253, 3.5567847823070253,
    3.5567847823070253, 3.5567847823070253,
)
_TD_D_C = (
    0.09054274779541564, -0.05102629221303253,
    -0.23907463153703945, -0.7757156285450911,
    0.0, 0.0, 0.0, 0.0,
)


def _cubic_bspline(value: float, knots: tuple[float, ...],
                   coefficients: tuple[float, ...]) -> float:
    """Evaluate a cubic B-spline, including end-polynomial extrapolation."""
    degree = 3
    n_coefficients = len(knots) - degree - 1
    if value < knots[degree]:
        span = degree
    elif value >= knots[n_coefficients]:
        span = n_coefficients - 1
    else:
        span = min(
            max(bisect_right(knots, value) - 1, degree),
            n_coefficients - 1,
        )
    work = [
        coefficients[span - degree + offset]
        for offset in range(degree + 1)
    ]
    for recursion in range(1, degree + 1):
        for offset in range(degree, recursion - 1, -1):
            left_index = span - degree + offset
            denominator = (
                knots[offset + 1 + span - recursion] - knots[left_index]
            )
            alpha = ((value - knots[left_index]) / denominator
                     if denominator else 0.0)
            work[offset] = (
                (1.0 - alpha) * work[offset - 1]
                + alpha * work[offset]
            )
    return work[degree]


def _td_boundary(x_value: float, knots: tuple[float, ...],
                 coefficients: tuple[float, ...]) -> float:
    return 10.0 ** _cubic_bspline(
        math.log10(x_value), knots, coefficients)


def _darcy_friction_factor(reynolds: float, relative_roughness: float,
                           model: str = "colebrook") -> float:
    """Darcy factor for the selectable TD reference convention."""
    if reynolds <= 0.0:
        raise ValueError("Reynolds number must be positive.")
    if model not in {"colebrook", "blasius"}:
        raise ValueError(f"Unknown Taitel-Dukler friction model: {model}")
    if reynolds < 2040.0:
        return 64.0 / reynolds
    if model == "blasius":
        return 0.3164 * reynolds ** -0.25

    # Solve Colebrook in y=1/sqrt(f).  This matches the exact default used by
    # the upstream reference while avoiding a runtime dependency on fluids.
    y = 7.0
    for _ in range(30):
        argument = relative_roughness / 3.7 + 2.51 * y / reynolds
        residual = y + 2.0 * math.log10(argument)
        derivative = 1.0 + (2.0 / math.log(10.0)) * (
            2.51 / reynolds) / argument
        next_y = y - residual / derivative
        if abs(next_y - y) <= 1.0e-13 * max(1.0, abs(next_y)):
            y = next_y
            break
        y = next_y
    return 1.0 / (y * y)


def taitel_dukler_from_superficial_velocities(
    vsg: float,
    vsl: float,
    rho_g: float,
    rho_l: float,
    mu_g: float,
    mu_l: float,
    diameter: float,
    roughness: float,
    theta_rad: float,
    friction_model: str = "colebrook",
) -> TaitelDuklerResult:
    """Classify horizontal/near-horizontal flow using Taitel--Dukler (1976).

    This is an independent map-level benchmark.  It returns the four
    dimensionless coordinates and all three digitized boundaries so unit,
    friction-factor, and boundary differences can be diagnosed separately.
    It is intentionally not wired into the Xiao hydraulic branches.
    """
    if vsg <= 0.0 or vsl <= 0.0:
        raise ValueError("Taitel-Dukler requires vsg and vsl > 0.")
    if min(rho_g, rho_l, mu_g, mu_l, diameter) <= 0.0:
        raise ValueError("Taitel-Dukler properties and diameter must be positive.")
    if rho_l <= rho_g:
        raise ValueError("Taitel-Dukler requires rho_l > rho_g.")
    if roughness < 0.0:
        raise ValueError("Pipe roughness cannot be negative.")
    cos_theta = math.cos(theta_rad)
    if cos_theta <= 0.0:
        raise ValueError("Taitel-Dukler requires |theta| < 90 degrees.")

    re_l = rho_l * vsl * diameter / mu_l
    re_g = rho_g * vsg * diameter / mu_g
    relative_roughness = roughness / diameter
    f_l = _darcy_friction_factor(
        re_l, relative_roughness, model=friction_model)
    f_g = _darcy_friction_factor(
        re_g, relative_roughness, model=friction_model)
    dp_l = f_l / diameter * 0.5 * rho_l * vsl * vsl
    dp_g = f_g / diameter * 0.5 * rho_g * vsg * vsg
    density_difference = rho_l - rho_g

    x_value = math.sqrt(dp_l / dp_g)
    f_value = math.sqrt(rho_g / density_difference) * vsg / math.sqrt(
        diameter * G * cos_theta)
    t_value = math.sqrt(
        dp_l / (density_difference * G * cos_theta))
    k_value = math.sqrt(
        rho_g * vsg * vsg * vsl
        / (density_difference * G * (mu_l / rho_l) * cos_theta)
    )

    boundary_a = _td_boundary(x_value, _TD_A_T, _TD_A_C)
    boundary_c = _td_boundary(x_value, _TD_C_T, _TD_C_C)
    boundary_d = _td_boundary(x_value, _TD_D_T, _TD_D_C)
    if f_value >= boundary_a and x_value <= 1.7917:
        pattern = "annular"
    elif f_value >= boundary_a:
        pattern = "bubbly" if t_value >= boundary_d else "intermittent"
    else:
        pattern = "stratified_wavy" if k_value >= boundary_c else "stratified_smooth"

    return TaitelDuklerResult(
        pattern=pattern,
        x=x_value,
        t=t_value,
        f=f_value,
        k=k_value,
        boundary_a=boundary_a,
        boundary_c=boundary_c,
        boundary_d=boundary_d,
    )


def group_taitel_dukler_pattern(pattern: str) -> str:
    """Map the five TD classes to the project's four Xiao classes."""
    return {
        "annular": "annular",
        "bubbly": "dispersed_bubble",
        "intermittent": "intermittent",
        "stratified_wavy": "stratified",
        "stratified_smooth": "stratified",
    }[pattern]


def mandhane_gregory_aziz_horizontal(
    vsg: float,
    vsl: float,
    rho_g: float,
    rho_l: float,
    mu_g: float,
    mu_l: float,
    sigma: float,
) -> str:
    """Return the six-class Mandhane--Gregory--Aziz (1974) regime.

    Parameters are SI, while the published map boundaries use superficial
    velocities in ft/s.  The property corrections follow the paper's computer
    implementation.  This implementation was cross-checked against the MIT
    licensed ``fluids`` implementation recorded in ``THIRD_PARTY_NOTICES.md``.

    The map is for horizontal pipes; inclination gating belongs to the caller.
    """
    if vsg <= 0.0 or vsl <= 0.0:
        raise ValueError("Mandhane horizontal map requires vsg and vsl > 0.")
    if min(rho_g, rho_l, mu_g, mu_l, sigma) <= 0.0:
        raise ValueError("Mandhane horizontal map properties must be positive.")

    # The original boundary program uses ft/s for the two map coordinates.
    vsl_ft_s = vsl / 0.3048
    vsg_ft_s = vsg / 0.3048

    gas_correction = (
        (rho_g / 1.294292) ** (1.0 / 3.0)
        * (rho_l * 0.0724 / (999.552 * sigma)) ** 0.25
        * (mu_g * 1.8e5) ** 0.2
    )
    liquid_correction = (
        (rho_l * 0.0724 / (999.552 * sigma)) ** 0.25
        * (mu_l * 1.0e3) ** 0.2
    )

    if vsl_ft_s < 14.0 * liquid_correction:
        if vsl_ft_s <= 0.1:
            separated_boundary = 14.0 * (vsl_ft_s / 0.1) ** -0.368
        elif vsl_ft_s <= 0.2:
            separated_boundary = 14.0 * (vsl_ft_s / 0.1) ** -0.415
        elif vsl_ft_s <= 1.15:
            separated_boundary = 10.5 * (vsl_ft_s / 0.2) ** -0.816
        elif vsl_ft_s <= 4.8:
            separated_boundary = 2.5
        else:
            separated_boundary = 2.5 * (vsl_ft_s / 4.8) ** 0.248

        if vsl_ft_s <= 0.1:
            annular_boundary = 70.0 * (vsl_ft_s / 0.01) ** -0.0675
        elif vsl_ft_s <= 0.3:
            annular_boundary = 60.0 * (vsl_ft_s / 0.1) ** -0.415
        elif vsl_ft_s <= 0.56:
            annular_boundary = 38.0 * (vsl_ft_s / 0.3) ** 0.0813
        elif vsl_ft_s <= 1.0:
            annular_boundary = 40.0 * (vsl_ft_s / 0.56) ** 0.385
        elif vsl_ft_s <= 2.5:
            annular_boundary = 50.0 * vsl_ft_s ** 0.756
        else:
            annular_boundary = 100.0 * (vsl_ft_s / 2.5) ** 0.463

        elongated_to_stratified = 0.5 / liquid_correction
        slug_to_wave = 0.3 * liquid_correction
        separated_boundary *= gas_correction
        annular_boundary *= gas_correction

        if vsg_ft_s <= separated_boundary and vsl_ft_s >= elongated_to_stratified:
            return "elongated_bubble"
        if vsg_ft_s <= separated_boundary and vsl_ft_s <= elongated_to_stratified:
            return "stratified"
        if vsg_ft_s <= annular_boundary and vsl_ft_s > slug_to_wave:
            return "slug"
        if vsg_ft_s <= annular_boundary and vsl_ft_s <= slug_to_wave:
            return "wave"
        return "annular_mist"

    dispersed_to_annular = (
        230.0 * (vsl_ft_s / 14.0) ** 0.206 * gas_correction)
    if vsg_ft_s <= dispersed_to_annular:
        return "dispersed_bubble"
    return "annular_mist"


def group_mandhane_pattern(pattern: str) -> str:
    """Map the six-class horizontal map to the project's four Xiao classes."""
    return {
        "elongated_bubble": "intermittent",
        "stratified": "stratified",
        "slug": "intermittent",
        "wave": "stratified",
        "annular_mist": "annular",
        "dispersed_bubble": "dispersed_bubble",
    }[pattern]
