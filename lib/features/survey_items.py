"""Every raw ENABLE survey item usable as a B1 candidate ("all variables").

Rule: answered by >= 80% of development households in EACH of Bulgaria, Hungary, Italy and Poland, with more than one value.
Removed: the B0 items (H1 dwelling type, H2 age, H3 floor area, H9 heating control, S4 birth year), the items that build the
label (H7*, H8* energy costs, S9* income), free-text and Spain-only fields, and items not asked everywhere (H13B gas smart
meter, H15 environmental attitudes, SettlementSize, the P and C modules). S7 and S8 enter in their cleaned form from the
loader. Missing codes (98/99, and 9 for S2, 6 for S6) become missing values. S3 (employment status) is nominal.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

ITEMS = ["H4A", "H4B", "H4C", "H4D", "H4E", "H4F", "H4G", "H4H", "H4I", "H4J",          # vehicles by type
         "H5A1", "H5A2", "H5A3", "H5A4", "H5A99",                                          # insulation
         "H6A1", "H6A2", "H6A3", "H6A4", "H6A5", "H6A6", "H6A7", "H6A8", "H6A9", "H6A10", "H6A11", "H6A99",   # heating sources
         "H10A1", "H10A2", "H10A3", "H10A4", "H10A5",                                      # renewables
         "H11A", "H11B", "H11C", "H11D", "H11E", "H11F", "H11G", "H11H",                   # appliance ages
         "H12A", "H12B", "H13A", "H13C", "H14A8",                                          # lighting, smart meters
         "S1Ac1", "S1Ac2", "S1Ac3", "S1Bc1", "S1Bc2", "S1Bc3",                             # household composition
         "S2", "S3", "S5", "S6"]                                                           # education, employment, gender, area
NOMINAL_ITEMS = {"S3"}
EXTRA_MISSING = {"S2": {9}, "S6": {6}}
ITEM_NAMES = {"H4A": "petrol cars", "H4B": "diesel cars", "H4C": "LPG cars", "H4D": "electric cars", "H4E": "hybrid cars",
              "H4F": "motorcycles", "H4G": "e-motorcycles", "H4H": "vans", "H4I": "bicycles", "H4J": "e-bicycles",
              "H5A1": "roof insulation", "H5A2": "cavity wall insulation", "H5A3": "external wall insulation",
              "H5A4": "no insulation", "H5A99": "insulation unknown", "H6A1": "heating % electricity", "H6A2": "heating % district",
              "H6A3": "heating % gas", "H6A4": "heating % wood", "H6A5": "heating % coal", "H6A6": "heating % pellets",
              "H6A7": "heating % fuel oil", "H6A8": "heating % waste", "H6A9": "heating % biomass", "H6A10": "heating % heat pump",
              "H6A11": "heating % other", "H6A99": "heating source unknown", "H10A1": "solar PV", "H10A2": "biomass generation",
              "H10A3": "solar water heating", "H10A4": "heat pump", "H10A5": "no renewables", "H11A": "cooker age",
              "H11B": "dishwasher age", "H11C": "washing machine age", "H11D": "fridge age", "H11E": "air conditioner age",
              "H11F": "portable heater age", "H11G": "electric water heater age", "H11H": "TV age",
              "H12A": "incandescent bulbs", "H12B": "efficient bulbs", "H13A": "electricity smart meter",
              "H13C": "heating smart meter", "H14A8": "smart meter: other reason", "S1Ac1": "girls under 18",
              "S1Ac2": "women 18-65", "S1Ac3": "women over 65", "S1Bc1": "boys under 18", "S1Bc2": "men 18-65",
              "S1Bc3": "men over 65", "S2": "education", "S3": "employment status", "S5": "gender", "S6": "area type"}


def clean_items(frame: pd.DataFrame) -> pd.DataFrame:
    """Numeric survey items with missing codes set to NaN (S3 kept as a string category)."""
    out = pd.DataFrame(index=frame.index)
    for c in ITEMS:
        v = pd.to_numeric(frame[c], errors="coerce")
        v = v.where(~v.isin({98, 99, 999, 9999} | EXTRA_MISSING.get(c, set())))
        out[c] = v.map(lambda z: f"code_{int(z)}" if pd.notna(z) else np.nan) if c in NOMINAL_ITEMS else v
    return out
