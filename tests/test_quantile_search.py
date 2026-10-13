import numpy as np,pandas as pd
from lib.tuning.quantile_search import DEFAULT_BOUNDS,_calibration_error,_oof_predict,find_optimal_quantile,oof_fold


def fixture():
 rng=np.random.default_rng(11);n=300
 d=pd.DataFrame({'UID':[str(i) for i in range(n)],'country':np.where(np.arange(n)%3==0,'A',np.where(np.arange(n)%3==1,'B','C')),
                  'x':rng.normal(size=n),'income_bracket':1+np.arange(n)%10,
                  'observed_energy_cost_eur_reference':np.exp(5+.3*rng.normal(size=n))})
 d['oof_fold']=d.UID.map(oof_fold)
 return d


def test_optimizer_stays_within_bounds_and_is_deterministic():
 d=fixture()
 r1=find_optimal_quantile('M2',d,features=['x'],target='observed_energy_cost_eur_reference')
 r2=find_optimal_quantile('M2',d,features=['x'],target='observed_energy_cost_eur_reference')
 assert DEFAULT_BOUNDS[0]<=r1['quantile']<=DEFAULT_BOUNDS[1]
 assert r1['quantile']==r2['quantile']
 assert r1['absolute_calibration_error']==r2['absolute_calibration_error']


def test_search_trace_no_worse_than_coarse_reference_scan():
 """The optimizer's own logged search-phase trace (un-tuned, see tune_during_search)
 must reach an error at least as good as a manual reference scan over the same bounds
 -- a concrete safety net against the local refine step converging to a bad plateau on
 the non-smooth calibration objective (see the module docstring)."""
 d=fixture()
 result=find_optimal_quantile('M2',d,features=['x'],target='observed_energy_cost_eur_reference')
 best_trace_error=min(err for _,err in result['evaluations'])
 reference_errors=[_calibration_error(_oof_predict('M2',d,float(q),features=['x'],
                                                     target='observed_energy_cost_eur_reference',
                                                     country_col='country',tune=False),float(q))
                    for q in np.linspace(*DEFAULT_BOUNDS,7)]
 assert best_trace_error<=min(reference_errors)+1e-9
