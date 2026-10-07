"""Inventory policy package.

Phase 3 implements the JIT recommendation engine. Traditional policy and
Simulated Micro-Expiry remain later research-only stages.
"""

from policies.jit import (
    JITRecommendation,
    apply_jit_policy,
    expected_demand_by_group,
    load_phase1_forecast,
    recommend_jit,
)

__all__ = [
    "JITRecommendation",
    "apply_jit_policy",
    "expected_demand_by_group",
    "load_phase1_forecast",
    "recommend_jit",
]
