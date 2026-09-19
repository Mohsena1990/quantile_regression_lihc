"""Canonical risk-class allocation shared by four-class strategies."""
RISK_NAMES={0:'No Risk',1:'Income Risk',2:'Expenditure Risk',3:'Double Risk'}
def allocate_risk_class(*,low_income:bool,high_cost:bool)->tuple[int,str]:
    code=int(bool(low_income))+2*int(bool(high_cost));return code,RISK_NAMES[code]
