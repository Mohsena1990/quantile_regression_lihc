"""Leakage-safe model measurement labels for M1/M2/M3."""
from __future__ import annotations
import pandas as pd
from lib.labels.four_class import allocate_risk_class
from lib.labels.income_harmonization import low_income_flag
from .estimators import fit_family

def fit_measurement_strategy(family,train,apply,*,features,target='observed_energy_cost_eur_reference',
 quantile=.6,country_col='country',income_col='income_bracket',tune=True):
    # Imported lazily to avoid a circular import: tuning.regression_search imports
    # measurement.families.model_based.estimators, whose package __init__ also imports
    # this module -- so importing tune_family at module scope here would try to import
    # tuning.regression_search while it is still mid-initialisation the very first time
    # either module is loaded first.
    from lib.tuning.regression_search import tune_family
    params=tune_family(family,train,features=features,target=target,quantile=quantile,country_col=country_col) if tune else {}
    model,pred,native=fit_family(family,train,apply,features=features,target=target,quantile=quantile,country_col=country_col,params=params)
    out=apply.copy();out['predicted_cost']=pred
    if native: out['conditional_threshold']=pred
    else:
        _,train_pred,_=fit_family(family,train,train,features=features,target=target,quantile=quantile,country_col=country_col,params=params)
        residual=pd.to_numeric(train[target],errors='coerce')-train_pred
        offsets=train.assign(_r=residual).groupby(country_col)._r.quantile(quantile).to_dict()
        global_offset=float(residual.quantile(quantile));out['conditional_threshold']=pred+out[country_col].map(offsets).fillna(global_offset)
    cost=pd.to_numeric(out[target],errors='coerce');income=pd.to_numeric(out[income_col],errors='coerce')
    eligible=cost.notna()&income.notna()&out.conditional_threshold.notna();out['high_cost']=pd.Series(pd.NA,index=out.index,dtype='boolean');out.loc[eligible,'high_cost']=cost[eligible]>out.loc[eligible,'conditional_threshold']
    out['low_income']=pd.Series(pd.NA,index=out.index,dtype='boolean');out.loc[eligible,'low_income']=low_income_flag(income[eligible],out.loc[eligible,country_col])
    labels=[allocate_risk_class(low_income=bool(li),high_cost=bool(hc)) if ok else (pd.NA,pd.NA) for li,hc,ok in zip(out.low_income,out.high_cost,eligible)]
    out['risk_class_code']=pd.array([x[0] for x in labels],dtype='Int64');out['risk_class_name']=pd.array([x[1] for x in labels],dtype='string');out['measurement_eligible']=eligible
    return model,out
