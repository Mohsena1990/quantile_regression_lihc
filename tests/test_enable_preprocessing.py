import pandas as pd
import pytest
from data.enable import retained_cohort
from lib.expenditure.reconstruct import annual_electricity_cost, annual_heating_cost, combine_energy_cost

def test_country_cohort_and_nuts_validation():
    frame=pd.DataFrame({"UID":[1,2,3],"Country":[1,4,2],"NUTS1":["BG3","HU1",None]})
    assert retained_cohort(frame).UID.tolist()==[1,2]

def test_retained_missing_nuts_fails():
    with pytest.raises(ValueError,match="missing NUTS1"):
        retained_cohort(pd.DataFrame({"UID":[1],"Country":[1],"NUTS1":[None]}))

def test_heating_reconstruction_precedence_and_sentinels():
    assert annual_heating_cost(600,100,7)==(600.0,"reported_heating_season_total")
    assert annual_heating_cost(None,100,7)==(700.0,"monthly_times_paid_months")
    assert annual_heating_cost(None,100,99)==(None,"missing")
    assert annual_heating_cost(None,None,None,no_heating=True)==(0.0,"explicit_no_heating_system")

def test_electricity_and_complete_cost():
    assert annual_electricity_cost(1200,50)==(1200.0,"reported_annual")
    assert annual_electricity_cost(None,50)==(600.0,"monthly_times_12")
    assert combine_energy_cost(700,600)==(1300,"complete")
    assert combine_energy_cost(None,600)==(None,"electricity_only")
