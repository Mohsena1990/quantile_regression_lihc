"""Executable adapters for the three cost-threshold estimators M1/M2/M3 using project-declared dependencies.

Naming (2026-10-13): numbered in sequence M1, M2, M3 -- formerly M2, M5 and M7 of a larger family list.
Kept to exactly one family per genuinely distinct functional form (linear /
smooth-nonlinear / flexible-nonlinear), after an empirical audit found the other candidates' predictions
correlated 0.86-0.98 (Spearman) with the rest of the pack on real validation data -- high redundancy given the
shared ~0.15-0.23 R^2 ceiling, not independent evidence. They were removed rather than kept-but-unused:
- CatBoost quantile regression correlated 0.959 with M3 (both native quantile tree-boosting, same loss,
  different library) -- redundant with M3.
- A lognormal mixed-effects model correlated 0.976 with M1, and its country random-intercept variance converged
  to ~0 once NUTS1 fixed effects were in the design -- empirically a plain log-linear model, i.e. M1 with an
  extra transform, not a genuinely distinct method.
- Random forest and a GBM mean model with an offset were the next most redundant with the M1/M2/M3 trio and
  added a second and third tree ensemble without a new objective or functional form.

The retained trio:
- M1: linear quantile regression -- interpretable coefficients, native quantile loss.
- M2: spline-basis GAM -- smooth nonlinear, still supports partial-dependence-style
  interpretation of a factor's effect.
- M3: gradient-boosted quantile regression -- flexible nonlinear, and shares M1's
  exact pinball-loss objective, isolating "does nonlinearity help" from "does the
  loss function differ" when comparing the two.
"""
from __future__ import annotations
import numpy as np,pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import GammaRegressor,QuantileRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder,SplineTransformer,StandardScaler

def _design(frame,features):
    cat=[x for x in features if not pd.api.types.is_numeric_dtype(frame[x])];num=[x for x in features if x not in cat]
    return ColumnTransformer([('num',make_pipeline(SimpleImputer(strategy='median'),StandardScaler()),num),
      ('cat',make_pipeline(SimpleImputer(strategy='most_frequent'),OneHotEncoder(handle_unknown='ignore',sparse_output=False,drop='first')),cat)])

# Default hyperparameters, used when the caller does not supply a tuned `params` dict.
DEFAULT_PARAMS={'M1':{'alpha':.001},'M2':{'alpha':.1},'M3':{'n_estimators':250,'max_depth':3}}

def fit_family(family:str,train:pd.DataFrame,predict:pd.DataFrame,*,features:list[str],target:str,
               quantile:float=.6,country_col:str='country',params:dict|None=None):
    """Fit one family and return positive predictions plus native-quantile flag.

    `params` overrides that family's DEFAULT_PARAMS entry (see tuning/regression_search.py,
    which selects it via internal k-fold pinball-loss search on the caller's train frame only).
    """
    if family not in {'M1','M2','M3'}: raise ValueError(f'unsupported model family {family}')
    p={**DEFAULT_PARAMS.get(family,{}),**(params or {})}
    X=train[features];Xp=predict[features];y=pd.to_numeric(train[target],errors='coerce')
    valid=y.notna()&(y>0);X=X.loc[valid];y=y.loc[valid]
    if len(y)<10: raise ValueError('INSUFFICIENT_TRAINING_ROWS')
    prep=_design(train,features);native=family in {'M1','M3'}
    if family=='M1': est=QuantileRegressor(quantile=quantile,alpha=p['alpha'],solver='highs')
    elif family=='M2':
        # Penalised spline-basis GAM; categorical terms remain one-hot linear effects.
        num=[x for x in features if pd.api.types.is_numeric_dtype(train[x])]
        cat=[x for x in features if x not in num]
        prep=ColumnTransformer([('smooth',make_pipeline(SimpleImputer(strategy='median'),SplineTransformer(n_knots=5,degree=3),StandardScaler()),num),
          ('cat',make_pipeline(SimpleImputer(strategy='most_frequent'),OneHotEncoder(handle_unknown='ignore',sparse_output=False,drop='first')),cat)])
        est=GammaRegressor(alpha=p['alpha'],max_iter=2000)
    elif family=='M3': est=GradientBoostingRegressor(loss='quantile',alpha=quantile,n_estimators=p['n_estimators'],max_depth=p['max_depth'],random_state=20260811)
    else: raise AssertionError('unreachable: family already validated above')
    model=make_pipeline(prep,est);model.fit(X,y);pred=np.asarray(model.predict(Xp),float)
    return model,np.maximum(pred,0),native
