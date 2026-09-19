"""Resumable CDS ERA5-Land daily-statistics acquisition."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import cdsapi,pandas as pd
ROOT=Path(__file__).resolve().parents[1]
PERIODS={'annual_2017':('2017-01-01','2017-12-31','PRIMARY'),'heating_season_2017_18':('2017-11-01','2018-02-28','SENSITIVITY'),'normal_1991_2020':('1991-01-01','2020-12-31','SENSITIVITY'),'annual_2017_canarias':('2017-01-01','2017-12-31','SUPPLEMENTARY')}
# Default area is the mainland-Europe bounding box [North, West, South, East]. ES7 (Canarias,
# lat ~27.6-29.4N, lon ~-18.2 to -13.4E) falls entirely south of that box's 34N boundary and was
# never covered by the primary annual_2017 request (see AUDIT_AND_REPAIR_REPORT.md ISS "ERA5
# Canary Islands gap"). annual_2017_canarias is a supplementary, separately-named request (same
# dataset/variable/period) covering just that region, padded ~0.5-0.6 degrees beyond ES7's bounds.
AREAS={'annual_2017_canarias':[30,-19,27,-13]}
DEFAULT_AREA=[72,-25,34,45]
def main():
 p=argparse.ArgumentParser();p.add_argument('--configuration',choices=PERIODS,default='annual_2017');p.add_argument('--force',action='store_true');a=p.parse_args();start,end,role=PERIODS[a.configuration]
 dates=pd.date_range(start,end,freq='D');base={'variable':['2m_temperature'],'year':sorted(set(dates.strftime('%Y'))),'month':sorted(set(dates.strftime('%m'))),'day':[f'{x:02d}' for x in range(1,32)],'time_zone':'utc+00:00','frequency':'1_hourly','area':AREAS.get(a.configuration,DEFAULT_AREA)}
 folder=ROOT/'artifacts/climate';folder.mkdir(parents=True,exist_ok=True);outputs={};client=cdsapi.Client()
 for statistic in ('daily_mean','daily_minimum'):
  out=folder/f'era5_land_{a.configuration}_{statistic}.nc';req={**base,'daily_statistic':statistic}
  if not out.exists() or a.force:
   try: client.retrieve('derived-era5-land-daily-statistics',req,str(out))
   except Exception as e: raise SystemExit(f'BLOCKED_EXTERNAL_CDS: accept the dataset licence and configure CDS credentials ({e})')
  outputs[statistic]={'path':str(out.relative_to(ROOT)),'sha256':hashlib.sha256(out.read_bytes()).hexdigest()}
 manifest={'configuration':a.configuration,'start':start,'end':end,'role':role,'dataset':'derived-era5-land-daily-statistics','doi':'10.24381/cds.e9c9c792','request_base':base,'outputs':outputs}
 (folder/f'era5_land_{a.configuration}.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
