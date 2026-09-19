"""Transparent annualisation of reported heating and electricity expenditure."""
from __future__ import annotations
import math
from typing import Any

SENTINELS = {98, 99, 999, 9999, 99999}

def _valid(value: Any, *, allow_zero: bool = True) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number in SENTINELS or number < 0 or (not allow_zero and number == 0):
        return None
    return number

def annual_heating_cost(season_total: Any, monthly: Any, months: Any, *, no_heating: bool = False) -> tuple[float | None, str]:
    if no_heating:
        return 0.0, "explicit_no_heating_system"
    total = _valid(season_total)
    if total is not None:
        return total, "reported_heating_season_total"
    month_cost, month_count = _valid(monthly), _valid(months, allow_zero=False)
    if month_cost is not None and month_count is not None and month_count <= 12:
        return month_cost * month_count, "monthly_times_paid_months"
    return None, "missing"

def annual_electricity_cost(annual: Any, monthly: Any) -> tuple[float | None, str]:
    total = _valid(annual)
    if total is not None:
        return total, "reported_annual"
    month_cost = _valid(monthly)
    if month_cost is not None:
        return month_cost * 12, "monthly_times_12"
    return None, "missing"

def combine_energy_cost(heating: float | None, electricity: float | None) -> tuple[float | None, str]:
    if heating is None or electricity is None:
        present = "heating_only" if heating is not None else "electricity_only" if electricity is not None else "missing_both"
        return None, present
    return heating + electricity, "complete"
