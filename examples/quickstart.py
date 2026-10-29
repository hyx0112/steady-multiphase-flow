"""最小可跑示例：同一工况、同一流型下换闭合关系式，看结果怎么变。

运行：  python examples/quickstart.py
"""
from __future__ import annotations
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from closures.xiao_model_faithful import XiaoInput, xiao_pressure_gradient

# 一个固定的分层流工况（SI 单位）。flow_pattern_override 把流型钉住，
# 这样比较的就纯粹是"同一个流型求解器 + 不同闭合式"的差异。
CASE = dict(
    total_mass_flow=40.0,        # kg/s
    gas_mass_fraction=0.05,      # -
    rho_gas=30.0,                # kg/m3
    rho_liquid=830.0,            # kg/m3
    mu_gas=1.6e-5,               # Pa*s
    mu_liquid=3.0e-3,            # Pa*s
    diameter=0.15,               # m
    roughness=4.5e-5,            # m
    angle_rad=math.radians(2.0),  # rad，上坡为正
    surface_tension=0.025,       # N/m
    pressure_Pa=3.0e6,           # Pa
    flow_pattern_override="stratified",
)

WALL_VARIANTS = [
    "xiao",            # 基线：水力直径 + 实际相速 + Colebrook
    "kowalski1987",    # 液壁，AIChE J 33(2):274-281, Eq 18
    "ottens2001",      # 双侧，Taitel-Dukler model a, Eq 7a;b
    "petalas_wall",    # 液壁，Eq 10
    "biberg1999_wall_full",
]


def main() -> None:
    print(f"{'variant':22s} {'dP/dL':>10s} {'H_L':>8s} {'converged':>10s}")
    print("-" * 54)
    for name in WALL_VARIANTS:
        r = xiao_pressure_gradient(
            XiaoInput(**CASE, stratified_wall_friction_closure=name))
        print(f"{name:22s} {r.dp_dL_total:10.1f} {r.liquid_holdup_inclined:8.4f} "
              f"{str(r.converged):>10s}")
    print("\n单位：dP/dL = Pa/m，H_L = 持液率（-）")

    # 持液率槽位：段塞体持液率。注意宿主 [0.48, 1] 夹断是模型内置行为，
    # 不在本示例中隐性改掉。
    print(f"\n{'slug holdup variant':22s} {'dP/dL':>10s} {'H_L':>8s}")
    print("-" * 42)
    slug_case = dict(CASE, angle_rad=0.0, flow_pattern_override="intermittent")
    for name in ("gregory1978", "abdul_majeed2019", "gomez2000_orig"):
        r = xiao_pressure_gradient(
            XiaoInput(**slug_case, intermittent_slug_holdup_closure=name))
        print(f"{name:22s} {r.dp_dL_total:10.1f} {r.liquid_holdup_inclined:8.4f}")


if __name__ == "__main__":
    main()
