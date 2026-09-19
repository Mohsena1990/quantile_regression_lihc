"""Executable adapters for M2/M5/M7 using project-declared dependencies.

Kept to exactly one family per genuinely distinct functional form (linear /
smooth-nonlinear / flexible-nonlinear), after an empirical audit found M3/M4/M6/M8's
predictions correlated 0.86-0.98 (Spearman) with the rest of the pack on real
validation data -- high redundancy given the shared ~0.15-0.23 R^2 ceiling, not
independent evidence. M3/M4/M6/M8 were removed rather than kept-but-unused:
- M3 (CatBoost quantile) correlated 0.959 with M7 (both native quantile
  tree-boosting, same loss, different library) -- redundant with M7.
- M8 (Lognormal mixed-effects) correlated 0.976 with M2, and its country
  random-intercept variance converged to ~0 once NUTS1 fixed effects were in the
  design -- empirically it had degenerated into a plain log-linear model, i.e. into
  M2 with an extra transform, not a genuinely distinct method.
- M4 (Random Forest) and M6 (GBM mean + offset) were the next most redundant with
  the M2/M5/M7 trio and added a second and third boosting/bagging tree ensemble
  without a new objective or functional form.

The retained trio:
- M2: linear quantile regression -- interpretable coefficients, native quantile loss.
- M5: spline-basis GAM -- smooth nonlinear, still supports partial-dependence-style
  interpretation of a factor's effect.
- M7: gradient-boosted quantile regression -- flexible nonlinear, and shares M2's
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
DEFAULT_PARAMS={'M2':{'alpha':.001},'M5':{'alpha':.1},'M7':{'n_estimators':250,'max_depth':3}}

def fit_family(family:str,train:pd.DataFrame,predict:pd.DataFrame,*,features:list[str],target:str,
               quantile:float=.6,country_col:str='country',params:dict|None=None):
    """Fit one family and return positive predictions plus native-quantile flag.

    `params` overrides that family's DEFAULT_PARAMS entry (see tuning/regression_search.py,
    which selects it via internal k-fold pinball-loss search on the caller's train frame only).
    """
    if family not in {'M2','M5','M7'}: raise ValueError(f'unsupported model family {family}')
    p={**DEFAULT_PARAMS.get(family,{}),**(params or {})}
    X=train[features];Xp=predict[features];y=pd.to_numeric(train[target],errors='coerce')
    valid=y.notna()&(y>0);X=X.loc[valid];y=y.loc[valid]
    if len(y)<10: raise ValueError('INSUFFICIENT_TRAINING_ROWS')
    prep=_design(train,features);native=family in {'M2','M7'}
    if family=='M2': est=QuantileRegressor(quantile=quantile,alpha=p['alpha'],solver='highs')
    elif family=='M5':
        # Penalised spline-basis GAM; categorical terms remain one-hot linear effects.
        num=[x for x in features if pd.api.types.is_numeric_dtype(train[x])]
        cat=[x for x in features if x not in num]
        prep=ColumnTransformer([('smooth',make_pipeline(SimpleImputer(strategy='median'),SplineTransformer(n_knots=5,degree=3),StandardScaler()),num),
          ('cat',make_pipeline(SimpleImputer(strategy='most_frequent'),OneHotEncoder(handle_unknown='ignore',sparse_output=False,drop='first')),cat)])
        est=GammaRegressor(alpha=p['alpha'],max_iter=2000)
    elif family=='M7': est=GradientBoostingRegressor(loss='quantile',alpha=quantile,n_estimators=p['n_estimators'],max_depth=p['max_depth'],random_state=20260811)
    else: raise AssertionError('unreachable: family already validated above')
    model=make_pipeline(prep,est);model.fit(X,y);pred=np.asarray(model.predict(Xp),float)
    return model,np.maximum(pred,0),native
