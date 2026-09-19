"""Historical local-currency to EUR conversion with explicit reference rows."""
from __future__ import annotations
import pandas as pd

def convert_local_to_eur(frame:pd.DataFrame,rates:pd.DataFrame,*,country_col='country',value_col='observed_energy_cost_local')->pd.Series:
    required={'country','local_currency_per_eur'}
    if required-set(rates): raise ValueError('currency reference is missing required columns')
    if rates.country.duplicated().any(): raise ValueError('currency reference has duplicate countries')
    rate=frame[country_col].map(rates.set_index('country').local_currency_per_eur)
    if rate.isna().any(): raise ValueError(f"Currency conversion unavailable for: {sorted(frame.loc[rate.isna(),country_col].unique())}")
    if (rate<=0).any(): raise ValueError('currency rates must be positive')
    return pd.to_numeric(frame[value_col],errors='coerce')/rate
