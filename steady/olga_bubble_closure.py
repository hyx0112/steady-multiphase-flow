"""OLGA 蒸馏得到的泡状流闭合（v3.8 漂移 + v6 壁面）。

公式和参数来自相邻项目 ``D:/2026/olga_batch`` 的：

* ``distillation/real_campaign/bubble_fit_v38.json``
* ``distillation/real_campaign/bubble_wall_v6.json``
* ``distillation/real_campaign/validate_bubble_forward_v9.py``

该模块只实现局部水力闭合，不实现闪蒸。相分配应由调用方的 PVT/EOS 给出，
避免把 OLGA 黑油 ``GF + RS - RS_in`` 关系叠加到已经完成总体组成闪蒸的
PR78 气相质量分数上。
"""

from __future__ import annotations

import math
from typing import Dict, Tuple


G = 9.81
_STEP = 0.866

_V0 = 0.003278940887147455
_C = 1.0615774760335164
_K = 1.9152257609662942
_M = 0.6581616279797853
_TRANSITION_S = 1.4717957038868117

_C0_TABLE = {
    -0.0174: 0.991265, 0.0000: 0.98955, 0.0174: 0.991096,
    0.0524: 0.991758, 0.0872: 0.991725, 0.1736: 0.992629,
    0.2588: 0.991714, 0.3420: 0.991976, 0.4226: 0.991969,
    0.5000: 0.991898, 0.5736: 0.99178, 0.6428: 0.991726,
    0.7071: 0.991699, 0.8660: 0.991608, 0.9063: 1.008506,
    0.9397: 1.009743, 0.9659: 1.010754, 0.9848: 1.011541,
    1.0000: 1.012162,
}

_A_TABLE = {
    0.0000: 70.165, 0.0174: 48.978, 0.2588: 66.641,
    0.3420: 65.263, 0.4226: 65.506, 0.5000: 65.84,
    0.5736: 66.354, 0.6428: 66.371, 0.7071: 66.212,
    0.8660: 69.804, 0.9063: 9.075, 0.9397: 5.792,
    0.9659: 2.67, 0.9848: 0.325, 1.0000: -1.563,
}

_WALL = {
    "in_domain": (1.0503023441117398, 0.7263418827363408),
    "transition": (1.1739099787288323, 0.9007833517176407),
}


def _piecewise_linear_clamped(x: float, table: Dict[float, float]) -> float:
    """按 v3.8 的 60° 台阶分支做支内线性插值、端外平推。"""
    keys = sorted(table)
    branch = [k for k in keys if (k <= _STEP) == (x <= _STEP)]
    if x <= branch[0]:
        return table[branch[0]]
    if x >= branch[-1]:
        return table[branch[-1]]
    for low, high in zip(branch, branch[1:]):
        if low <= x <= high:
            weight = (x - low) / (high - low)
            return table[low] + weight * (table[high] - table[low])
    raise RuntimeError("OLGA 泡状流锚表插值失败。")


def drift_velocity(sin_theta: float, gas_void_fraction: float,
                   mixture_superficial_velocity: float,
                   terminal_velocity_scale: float) -> float:
    """返回 v3.8 关系给出的气相实际速度。"""
    alpha = gas_void_fraction
    c0 = _piecewise_linear_clamped(sin_theta, _C0_TABLE)
    inclination_shape = sin_theta ** _M if sin_theta > 1.0e-12 else 0.0
    velocity = (
        c0 * mixture_superficial_velocity
        + _V0
        + _C * (1.0 - alpha) ** _K * inclination_shape * terminal_velocity_scale
    )
    if alpha >= 0.45:
        amplitude = _piecewise_linear_clamped(sin_theta, _A_TABLE)
        velocity += amplitude * (alpha - 0.45) ** _TRANSITION_S * terminal_velocity_scale
    return velocity


def _solve_void_fraction(vsg: float, vsl: float, rho_l: float, rho_g: float,
                         sigma: float, sin_theta: float) -> Tuple[float, float]:
    """由 ``vsg/alpha = U_g(alpha)`` 求气相体积分数及残差。"""
    if vsg <= 0.0:
        return 1.0e-8, 0.0
    mixture_velocity = vsg + vsl
    terminal_scale = (
        G * max(sigma, 1.0e-12) * max(rho_l - rho_g, 1.0e-12)
        / max(rho_l * rho_l, 1.0e-12)
    ) ** 0.25

    def residual(alpha: float) -> float:
        return vsg / alpha - drift_velocity(
            sin_theta, alpha, mixture_velocity, terminal_scale)

    low, high = 1.0e-8, 1.0 - 1.0e-8
    f_low, f_high = residual(low), residual(high)
    if not (math.isfinite(f_low) and math.isfinite(f_high) and f_low >= 0.0 >= f_high):
        raise ValueError("OLGA 泡状流漂移闭合在 (0,1) 内没有物理解。")
    for _ in range(100):
        middle = 0.5 * (low + high)
        f_middle = residual(middle)
        if f_middle > 0.0:
            low = middle
        else:
            high = middle
    alpha = 0.5 * (low + high)
    return alpha, residual(alpha)


def _colebrook_darcy(reynolds: float, relative_roughness: float) -> float:
    """复现 OLGA 蒸馏验证脚本使用的 Darcy 摩阻因子。"""
    reynolds = max(reynolds, 1.0e-12)
    if reynolds < 10.0:
        return 64.0 / reynolds
    inverse_sqrt = -1.8 * math.log10(
        6.9 / reynolds + (max(relative_roughness, 0.0) / 3.7) ** 1.11
    )
    for _ in range(10):
        inverse_sqrt = -2.0 * math.log10(
            2.51 * inverse_sqrt / reynolds + max(relative_roughness, 0.0) / 3.7
        )
    turbulent = 1.0 / (inverse_sqrt * inverse_sqrt)
    return max(turbulent, 64.0 / reynolds)


def solve_olga_bubble(
    vsg: float, vsl: float, rho_l: float, rho_g: float,
    mu_l: float, mu_g: float, diameter: float, roughness: float,
    theta_rad: float, sigma: float,
    *, wall_closure: str = "olga_v6",
) -> dict:
    """计算 OLGA 泡状流闭合；可只替换滑移而保留 Xiao 均相摩阻。

    ``wall_closure`` 可取 ``"olga_v6"`` 或 ``"xiao_homogeneous"``。
    """
    if diameter <= 0.0 or rho_l <= 0.0 or rho_g <= 0.0:
        raise ValueError("管径和相密度必须为正。")
    if wall_closure not in {"olga_v6", "xiao_homogeneous"}:
        raise ValueError(f"未知泡状流壁面闭合: {wall_closure}")

    sin_theta = math.sin(theta_rad)
    alpha, residual = _solve_void_fraction(
        vsg, vsl, rho_l, rho_g, sigma, sin_theta)
    liquid_holdup = 1.0 - alpha
    mixture_velocity = vsg + vsl
    gas_velocity = vsg / max(alpha, 1.0e-12)
    liquid_velocity = vsl / max(liquid_holdup, 1.0e-12)
    rho_mix = alpha * rho_g + liquid_holdup * rho_l

    if wall_closure == "olga_v6":
        reynolds_l = rho_l * abs(liquid_velocity) * diameter / max(mu_l, 1.0e-12)
        friction = _colebrook_darcy(reynolds_l, roughness / diameter)
        band = "in_domain" if alpha < 0.45 else "transition"
        amplitude, exponent = _WALL[band]
        tau_wall = (
            amplitude * liquid_holdup ** exponent * friction / 8.0
            * rho_l * liquid_velocity * abs(liquid_velocity)
        )
        dp_fric = 4.0 * tau_wall / diameter
    else:
        mu_mix = liquid_holdup * mu_l + alpha * mu_g
        reynolds_mix = rho_mix * abs(mixture_velocity) * diameter / max(mu_mix, 1.0e-12)
        friction = _colebrook_darcy(reynolds_mix, roughness / diameter)
        tau_wall = friction * rho_mix * mixture_velocity * abs(mixture_velocity) / 8.0
        dp_fric = 4.0 * tau_wall / diameter

    dp_grav = rho_mix * G * sin_theta
    return {
        "EL": liquid_holdup,
        "EG": alpha,
        "dp_fric": dp_fric,
        "dp_grav": dp_grav,
        "dp_total": dp_fric + dp_grav,
        "rho_m": rho_mix,
        "mu_m": liquid_holdup * mu_l + alpha * mu_g,
        "f": friction,
        "tau_wall": tau_wall,
        "friction_multiplier": 1.0,
        "pattern": "dispersed_bubble",
        "v_gas_actual": gas_velocity,
        "v_liq_actual": liquid_velocity,
        "iterations": 100,
        "converged": abs(residual) <= 1.0e-8,
        "closure": f"olga_v38+{wall_closure}",
        "inside_primary_void_domain": alpha < 0.45,
        "inside_calibrated_diameter_range": 0.1 <= diameter <= 0.3,
        "inside_calibrated_angle_range": -0.0174 <= sin_theta <= 1.0,
    }
