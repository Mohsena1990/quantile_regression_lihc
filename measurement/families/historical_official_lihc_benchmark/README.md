# historical_official_lihc_benchmark

This folder will contain an implementation of the historical official English
LIHC methodology adapted as a harmonised cross-country benchmark.

Important:
- Do NOT silently approximate missing official inputs.
- If exact engineering inputs are unavailable (insulation U-values, glazing
  areas, exact heating system efficiencies), implementations must keep an
  "exact" and an "adapted" branch and document differences.

Planned files:
- implementation_stub.py — explains required inputs and provides API skeleton
- compute_required_cost.py — engine to compute household `y_i` (BREDEM-style)
- tests/ — unit tests with synthetic fixtures
