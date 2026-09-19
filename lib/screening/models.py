"""Level-3 screening models with fixed four-class ontology, cost-sensitive class weighting."""
from __future__ import annotations
import numpy as np,pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import balanced_accuracy_score,f1_score,log_loss,recall_score
from lib.imbalance.strategies import class_weights
from lib.tuning.search import holdout_split,select_by_score

DEGENERATE_MAJORITY_SHARE=.97  # refuse to silently persist a fit collapsed to (near-)constant predictions

def _weights(y): return class_weights(y)
def _prep_frame(tr,te,used):
 cat=[i for i,x in enumerate(used) if not pd.api.types.is_numeric_dtype(tr[x])]
 xtr=tr[used].copy();xte=te[used].copy()
 for x in used:
  if xtr[x].dtype=='object' or x in [used[i] for i in cat]: xtr[x]=xtr[x].fillna('__MISSING__').astype(str);xte[x]=xte[x].fillna('__MISSING__').astype(str)
  else: xtr[x]=pd.to_numeric(xtr[x],errors='coerce').fillna(xtr[x].median());xte[x]=pd.to_numeric(xte[x],errors='coerce').fillna(xtr[x].median())
 return xtr,xte,cat
def _fit_catboost(xtr,ytr,cat,weights,seed=20260811,iterations=100,depth=6):
 model=CatBoostClassifier(iterations=iterations,depth=depth,loss_function='MultiClass',verbose=False,random_seed=seed,class_weights=weights)
 model.fit(xtr,ytr,cat_features=cat);return model
def _aligned_proba(model,x,n_classes=4):
 raw=model.predict_proba(x);out=np.zeros((len(x),n_classes));out[:,np.asarray(model.classes_,int)]=raw;return out
MIN_COUNTRY_SUPPORT=20
CATBOOST_CANDIDATES=((75,4),(100,6),(150,6),(150,8))
HIERARCHICAL_LAMBDA_CANDIDATES=(10,25,50,100,200,400)

def _country_experts(xtr,ytr,country,cat,seed=20260811,use_weights=False):
 """Fit one CatBoost per country with enough class-diverse support; others fall back to the global model (alpha_c -> 0)."""
 models={}
 for c in pd.unique(country):
  mask=(country==c).to_numpy();n_c=int(mask.sum())
  if n_c<MIN_COUNTRY_SUPPORT or ytr[mask].nunique()<2: continue
  w=_weights(ytr[mask]) if use_weights else None
  models[c]=(_fit_catboost(xtr[mask],ytr[mask],cat,w,seed),n_c)
 return models
def _blend_hierarchical(xtr,ytr,country_tr,xte,country_te,cat,weights,*,lam,seed=20260811,iterations=100,depth=6):
 """P_final = (1-alpha_c)*P_global + alpha_c*P_country, alpha_c = n_c/(n_c+lambda)."""
 global_model=_fit_catboost(xtr,ytr,cat,weights,seed,iterations=iterations,depth=depth);probs=_aligned_proba(global_model,xte)
 for c,(country_model,n_c) in _country_experts(xtr,ytr,country_tr,cat,seed,use_weights=weights is not None).items():
  mask_te=(country_te==c).to_numpy()
  if not mask_te.any(): continue
  alpha=n_c/(n_c+lam);probs[mask_te]=(1-alpha)*probs[mask_te]+alpha*_aligned_proba(country_model,xte[mask_te])
 return global_model,probs

def _train_holdout(n,ytr,*,seed,val_fraction=.2):
 """Shared internal train-only holdout for every hyperparameter search below -- never the
 caller's test/validation frame (see tuning/search.py), so tuning is safe to call identically
 in development and final_test stage."""
 fit_idx,hold_idx=holdout_split(n,val_fraction=val_fraction,seed=seed)
 if len(hold_idx)==0 or ytr.iloc[fit_idx].nunique()<2: return None,None
 return fit_idx,hold_idx

def select_hierarchical_lambda(xtr,ytr,country_tr,cat,weights,*,seed=20260811,candidates=HIERARCHICAL_LAMBDA_CANDIDATES,val_fraction=.2):
 """Tune the shrinkage lambda on an internal train-only holdout."""
 fit_idx,hold_idx=_train_holdout(len(xtr),ytr,seed=seed,val_fraction=val_fraction)
 if fit_idx is None: return candidates[0]
 fit_weights=_weights(ytr.iloc[fit_idx]) if weights is not None else None
 def score(lam):
  _,probs=_blend_hierarchical(xtr.iloc[fit_idx],ytr.iloc[fit_idx],country_tr.iloc[fit_idx],xtr.iloc[hold_idx],country_tr.iloc[hold_idx],cat,fit_weights,lam=lam,seed=seed)
  return f1_score(ytr.iloc[hold_idx],probs.argmax(1),average='macro',zero_division=0)
 best,_,_=select_by_score(candidates,score,maximize=True);return best

def select_catboost_hyperparams(xtr,ytr,cat,weights,*,seed=20260811,candidates=CATBOOST_CANDIDATES,val_fraction=.2):
 """Tune (iterations, depth) on an internal train-only holdout."""
 fit_idx,hold_idx=_train_holdout(len(xtr),ytr,seed=seed,val_fraction=val_fraction)
 if fit_idx is None: return candidates[0]
 fit_weights=_weights(ytr.iloc[fit_idx]) if weights is not None else None
 def score(candidate):
  iterations,depth=candidate
  model=_fit_catboost(xtr.iloc[fit_idx],ytr.iloc[fit_idx],cat,fit_weights,seed,iterations=iterations,depth=depth)
  return f1_score(ytr.iloc[hold_idx],model.predict(xtr.iloc[hold_idx]).astype(int).ravel(),average='macro',zero_division=0)
 best,_,_=select_by_score(candidates,score,maximize=True);return best

def _assert_not_degenerate(truth,pred,*,context):
 """Final sanity check before results are written: refuse to silently persist a fit that
 collapsed to (near-)constant predictions despite a class-diverse test set."""
 truth=np.asarray(truth);pred=np.asarray(pred)
 if pd.Series(truth).nunique()<2: return
 majority_share=pd.Series(pred).value_counts(normalize=True).iloc[0]
 if majority_share>DEGENERATE_MAJORITY_SHARE:
  raise RuntimeError(f'SCREENING_PREDICTIONS_DEGENERATE ({context}): predicted-class majority share '
                      f'{majority_share:.2f} exceeds {DEGENERATE_MAJORITY_SHARE} despite a class-diverse test set')

def fit_screening_model(kind,train,test,*,features,target='risk_class_code',country_col='country'):
    """Fit CatBoost or hierarchical CatBoost, cost-sensitive class weighting throughout."""
    if kind not in {'catboost','hierarchical_catboost'}: raise ValueError('unknown screening model')
    tr=train.dropna(subset=[target]).copy();te=test.dropna(subset=[target]).copy();tr[target]=tr[target].astype(int)
    weights=_weights(tr[target])
    used=list(features)
    if kind=='hierarchical_catboost':
        xtr,xte,cat=_prep_frame(tr,te,used)
        country_tr=tr[country_col].reset_index(drop=True);country_te=te[country_col].reset_index(drop=True)
        xtr=xtr.reset_index(drop=True);xte=xte.reset_index(drop=True);ytr=tr[target].reset_index(drop=True)
        lam=select_hierarchical_lambda(xtr,ytr,country_tr,cat,weights)
        model,probs=_blend_hierarchical(xtr,ytr,country_tr,xte,country_te,cat,weights,lam=lam)
        pred=probs.argmax(1)
    else:
        xtr,xte,cat=_prep_frame(tr,te,used)
        ytr=tr[target].reset_index(drop=True)
        iterations,depth=select_catboost_hyperparams(xtr.reset_index(drop=True),ytr,cat,weights)
        model=_fit_catboost(xtr,tr[target],cat,weights,iterations=iterations,depth=depth);pred=model.predict(xte).astype(int).ravel();probs=_aligned_proba(model,xte)
    _assert_not_degenerate(te[target],pred,context=kind)
    metrics={'macro_f1':float(f1_score(te[target],pred,average='macro',zero_division=0)),'double_risk_f1':float(f1_score(te[target],pred,labels=[3],average='macro',zero_division=0)),'double_risk_recall':float(recall_score(te[target],pred,labels=[3],average='macro',zero_division=0)),'balanced_accuracy':float(balanced_accuracy_score(te[target],pred)),'log_loss':float(log_loss(te[target],probs,labels=[0,1,2,3])),'n_test':len(te)}
    output=pd.DataFrame({'UID':te.UID,'truth':te[target].astype(int),'prediction':pred})
    for k in range(4):output[f'prob_{k}']=probs[:,k]
    return model,output,metrics
