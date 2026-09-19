"""Fuse ERA5-Land daily temperatures to NUTS1 with a strict 24/24 QC gate."""
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
import geopandas as gpd,numpy as np,pandas as pd,xarray as xr
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from lib.climate.degree_days import climate_summary
from data.enable import HOUSEHOLD_SHEET,retained_cohort
from lib.reporting.figures import save_figure
from scripts.build_policy_maps import country_label_points,_label_countries

# Supplementary sources: additional, separately-downloaded ERA5-Land requests (different
# bounding box, same dataset/variable/period) whose NUTS1 regions fall outside a primary
# configuration's box. Each entry's files must already exist (acquire_era5_land.py --configuration
# <name>) -- this script never submits a download itself. A region matched by more than one
# source raises an error rather than silently picking one (see the `matched_regions` guard below).
SUPPLEMENTARY_SOURCES={'annual_2017':['annual_2017_canarias']}

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def load_temperature(path):
 # ERA5-Land is a land-only dataset: the raw pan-European bounding-box grid legitimately
 # contains ~50% NaN over sea/ocean cells, so a blanket isfinite() check over the raw grid
 # would always fail. Finiteness is instead enforced later, only over the NUTS1 land points
 # actually used for the regional daily aggregates (see `daily_finite_ok` below).
 ds=xr.open_dataset(path);name='t2m' if 't2m' in ds else '2m_temperature';a=ds[name];time='valid_time' if 'valid_time' in a.dims else 'time'
 if float(a.min())<180 or float(a.max())>340:raise SystemExit(f'BLOCKED_ERA5_QC: unreasonable Kelvin range in {path}')
 return a-273.15,time

def main():
 p=argparse.ArgumentParser();p.add_argument('--configuration',default='annual_2017');a=p.parse_args();folder=ROOT/'artifacts/climate';mean_path=folder/f'era5_land_{a.configuration}_daily_mean.nc';min_path=folder/f'era5_land_{a.configuration}_daily_minimum.nc'
 if not mean_path.exists() or not min_path.exists():raise SystemExit('MISSING ERA5 daily mean/minimum files; do not duplicate an active CDS request')
 manifest_path=folder/f'era5_land_{a.configuration}.manifest.json';manifest=json.loads(manifest_path.read_text());active_path=folder/f'era5_land_{a.configuration}.active_request.json';active=json.loads(active_path.read_text()) if active_path.exists() else {};manifest['request_id']=manifest.get('request_id') or active.get('request_id')
 start,end=pd.Timestamp(manifest['start']),pd.Timestamp(manifest['end']);expected=pd.date_range(start,end,freq='D')
 nuts_path=ROOT/'artifacts/reference/NUTS_RG_01M_2016_4326_LEVL_1.geojson';nuts=gpd.read_file(nuts_path)[['NUTS_ID','geometry']]
 cohort=retained_cohort(pd.read_excel(ROOT/'data/raw/ENABLE.EU_dataset_survey of households.xlsx',sheet_name=HOUSEHOLD_SHEET,usecols=['UID','Country','NUTS1']));expected_regions=set(cohort.NUTS1.unique());country_by_nuts=cohort[['NUTS1','country']].drop_duplicates()
 source_names=[a.configuration]+[c for c in SUPPLEMENTARY_SOURCES.get(a.configuration,[]) if (folder/f'era5_land_{c}_daily_mean.nc').exists() and (folder/f'era5_land_{c}_daily_minimum.nc').exists()]
 rows=[];daily=[];matched_regions=set();source_hashes=[];request_ids={}
 for src in source_names:
  src_mean_path=folder/f'era5_land_{src}_daily_mean.nc';src_min_path=folder/f'era5_land_{src}_daily_minimum.nc'
  src_manifest_path=folder/f'era5_land_{src}.manifest.json'
  request_ids[src]=json.loads(src_manifest_path.read_text()).get('request_id') if src_manifest_path.exists() else None
  mean,time=load_temperature(src_mean_path);minimum,min_time=load_temperature(src_min_path)
  if time!=min_time or not mean.latitude.identical(minimum.latitude) or not mean.longitude.identical(minimum.longitude):raise SystemExit(f'BLOCKED_ERA5_QC: mean/minimum grids differ for {src}')
  mean=mean.sel({time:slice(start,end)});minimum=minimum.sel({time:slice(start,end)});dates=pd.DatetimeIndex(mean[time].values);min_dates=pd.DatetimeIndex(minimum[time].values)
  if not dates.equals(expected) or not min_dates.equals(expected) or dates.duplicated().any():raise SystemExit(f'BLOCKED_ERA5_QC: exact date coverage failed for {src}')
  grid=mean.mean(time).to_dataframe(name='x').reset_index();pts=gpd.GeoDataFrame(grid,geometry=gpd.points_from_xy(grid.longitude,grid.latitude),crs=4326);joined=gpd.sjoin(pts,nuts,predicate='within',how='inner')
  remaining=expected_regions-matched_regions
  for code,g in joined[joined.NUTS_ID.isin(remaining)].groupby('NUTS_ID'):
   if code in matched_regions:raise SystemExit(f'BLOCKED_ERA5_QC: NUTS1 region {code} matched by more than one source grid')
   selector={'latitude':xr.DataArray(g.latitude.to_numpy(),dims='points'),'longitude':xr.DataArray(g.longitude.to_numpy(),dims='points')};means=mean.sel(selector).mean('points').values;mins=minimum.sel(selector).mean('points').values
   for date,mn,mi in zip(dates,means,mins):daily.append({'NUTS1':code,'date':date,'temperature_mean':float(mn),'temperature_minimum':float(mi)})
   rows.append({'NUTS1':code,**climate_summary(means,mins),'configuration':a.configuration,'grid_points':len(g),'source_configuration':src})
   matched_regions.add(code)
  source_hashes.append({'source_configuration':src,'mean_sha256':sha(src_mean_path),'minimum_sha256':sha(src_min_path)})
 daily=pd.DataFrame(daily).merge(country_by_nuts,on='NUTS1',validate='many_to_one');result=pd.DataFrame(rows).merge(country_by_nuts,on='NUTS1',validate='one_to_one');found=set(result.NUTS1);missing=sorted(expected_regions-found)
 daily_ok=not daily.duplicated(['NUTS1','date']).any() and len(daily)==len(expected)*24;finite_ok=bool(daily[['temperature_mean','temperature_minimum']].notna().all().all());range_ok=bool(daily.temperature_mean.between(-80,60).all() and daily.temperature_minimum.between(-90,55).all());order_ok=bool(daily.temperature_mean.ge(daily.temperature_minimum-1e-8).all());degree_ok=bool(result.hdd.ge(0).all() and result.cdd.ge(0).all() and result.frost_days.between(0,len(expected)).all());prefix_ok=all(code[:2] in {'BG','HU','IT','PL','ES'} for code in found)
 qc={'configuration':a.configuration,'dataset':'derived-era5-land-daily-statistics','request_id':manifest.get('request_id'),'source_configurations':source_names,'source_request_ids':request_ids,'start':str(start.date()),'end':str(end.date()),'temperature_conversion':'Celsius = Kelvin - 273.15','aggregation_method':'unweighted arithmetic mean of identical ERA5 grid-point membership for daily mean and daily minimum (NaN sea/lake pixels skipped)','nuts_version':'NUTS 2016 GISCO 1M EPSG:4326','nuts_geometry_sha256':sha(nuts_path),'raw_sources_sha256':source_hashes,'expected_regions':24,'fused_regions':len(found),'coverage':len(found)/24,'missing_regions':missing,'expected_days':len(expected),'daily_rows':len(daily),'no_duplicate_date_region':daily_ok,'all_day_counts_valid':daily_ok,'no_missing_regional_temperature':finite_ok,'mean_ge_minimum':order_ok,'reasonable_celsius_ranges':range_ok,'degree_day_checks':degree_ok,'country_prefix_check':prefix_ok,'all_grid_points_positive':bool(result.grid_points.gt(0).all())}
 failures=[k for k,v in {'coverage':len(found)==24 and not missing,'daily':daily_ok,'finite':finite_ok,'order':order_ok,'range':range_ok,'degree':degree_ok,'country':prefix_ok,'grid':qc['all_grid_points_positive']}.items() if not v]
 qc['qc_all_clear']=not failures
 (folder/f'nuts1_{a.configuration}.qc.json').write_text(json.dumps(qc,indent=2)+'\n');daily.to_parquet(folder/f'nuts1_daily_{a.configuration}.parquet',index=False)
 if failures:raise SystemExit(f'BLOCKED_ERA5_QC: {failures}')
 out=folder/f'nuts1_{a.configuration}.parquet';result.to_parquet(out,index=False);qc['derived_summary_sha256']=sha(out);qc['derived_daily_sha256']=sha(folder/f'nuts1_daily_{a.configuration}.parquet');qc['freeze_hash']=hashlib.sha256(json.dumps(qc,sort_keys=True).encode()).hexdigest();(folder/f'nuts1_{a.configuration}.qc.json').write_text(json.dumps(qc,indent=2)+'\n')
 result.to_csv(ROOT/f'outputs/tables/table_climate_summary_{a.configuration}.csv',index=False)
 geo=nuts[nuts.NUTS_ID.isin(expected_regions)].merge(result,left_on='NUTS_ID',right_on='NUTS1',validate='one_to_one')
 geo['CNTR_CODE']=geo.NUTS_ID.str[:2];label_points=country_label_points(geo)
 for col,title in [('hdd','Heating degree days'),('temperature_mean','Mean temperature (°C)'),('frost_days','Frost days')]:
  ax=geo.plot(column=col,legend=True,cmap='viridis',edgecolor='white',linewidth=.3,figsize=(9,7));_label_countries(ax,label_points);ax.set_axis_off();ax.set_title(title);plt.tight_layout();save_figure(ax,ROOT/f'outputs/figures/climate_{col}_{a.configuration}');plt.close()
 print(json.dumps(qc,indent=2));print(out)
if __name__=='__main__':main()
