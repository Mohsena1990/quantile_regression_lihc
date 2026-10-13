import numpy as np,pandas as pd
from measurement.families.model_based import fit_family,fit_measurement_strategy

def fixture():
 rng=np.random.default_rng(7);n=80
 return pd.DataFrame({'UID':range(n),'country':np.where(np.arange(n)%2,'A','B'),'x':rng.normal(size=n),'cat':np.where(np.arange(n)%3,'u','v'),'income_bracket':1+np.arange(n)%10,'observed_energy_cost_eur_reference':np.exp(5+.2*rng.normal(size=n))})

def test_all_model_family_adapters_predict_positive():
 d=fixture();tr=d.iloc[:60];te=d.iloc[60:]
 for family in ('M1','M2','M3'):
  _,pred,_=fit_family(family,tr,te,features=['x','cat'],target='observed_energy_cost_eur_reference',quantile=.6)
  assert len(pred)==20 and (pred>=0).all()

def test_unsupported_family_rejected():
 # Only M1/M2/M3 exist (the redundant families were removed, see estimators.py's module
 # docstring) -- confirm they're actually gone, not silently still fittable.
 d=fixture()
 for family in ('M0','M4','M5','M7','M8'):
  try:
   fit_family(family,d.iloc[:60],d.iloc[60:],features=['x','cat'],target='observed_energy_cost_eur_reference')
   assert False, f'{family} should have been rejected'
  except ValueError:
   pass

def test_strategy_emits_fixed_four_classes():
 d=fixture();_,out=fit_measurement_strategy('M3',d.iloc[:60],d.iloc[60:],features=['x','cat'])
 assert out.measurement_eligible.all();assert set(out.risk_class_code)<=set(range(4))
