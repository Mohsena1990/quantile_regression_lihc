import pandas as pd
from lib.labels.four_class import allocate_risk_class
from measurement.families.van_hove import country_thresholds,label_with_thresholds

def test_van_q80_is_country_specific_and_strict():
 d=pd.DataFrame({'country':['A']*5+['B']*5,'cost':[1,2,3,4,5,10,20,30,40,50],'income':[4,3,4,3,4,4,3,4,3,4]})
 t=country_thresholds(d,cost_col='cost');assert t=={'A':4.2,'B':42.0}
 out=label_with_thresholds(d,t,cost_col='cost',income_col='income');assert out.groupby('country').high_cost.sum().to_dict()=={'A':1,'B':1}

def test_four_class_ontology_is_fixed():
 assert allocate_risk_class(low_income=False,high_cost=False)==(0,'No Risk')
 assert allocate_risk_class(low_income=True,high_cost=False)==(1,'Income Risk')
 assert allocate_risk_class(low_income=False,high_cost=True)==(2,'Expenditure Risk')
 assert allocate_risk_class(low_income=True,high_cost=True)==(3,'Double Risk')
