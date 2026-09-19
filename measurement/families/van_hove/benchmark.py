"""M0 country-specific absolute annual-expenditure Q80 rules."""
from __future__ import annotations
import pandas as pd
from lib.labels.four_class import allocate_risk_class
from lib.labels.income_harmonization import low_income_flag

def country_thresholds(frame:pd.DataFrame,*,cost_col:str='observed_energy_cost_eur_reference',q:float=.8)->dict[str,float]:
    if not 0<q<1: raise ValueError('q must be between zero and one')
    clean=frame.dropna(subset=['country',cost_col]); return clean.groupby('country')[cost_col].quantile(q).to_dict()
def label_with_thresholds(frame:pd.DataFrame,thresholds:dict[str,float],*,cost_col:str='observed_energy_cost_eur_reference',income_col:str='income_bracket')->pd.DataFrame:
    out=frame.copy(); missing_countries=set(out.country.dropna())-set(thresholds)
    if missing_countries: raise ValueError(f'Missing country thresholds: {sorted(missing_countries)}')
    out['conditional_threshold']=out.country.map(thresholds); out['high_cost']=out[cost_col]>out.conditional_threshold
    out['low_income']=low_income_flag(out[income_col],out.country)
    classes=out.apply(lambda r:allocate_risk_class(low_income=r.low_income,high_cost=r.high_cost),axis=1);out['risk_class_code']=[x[0] for x in classes];out['risk_class_name']=[x[1] for x in classes]
    return out
