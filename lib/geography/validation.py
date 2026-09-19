"""Validate survey geography without manufacturing NUTS identifiers."""
from __future__ import annotations
import json
from pathlib import Path
from collections.abc import Iterable

def official_nuts_ids(geojson_path:str|Path,level:int=1)->frozenset[str]:
    payload=json.loads(Path(geojson_path).read_text(encoding='utf-8'))
    return frozenset(f['properties']['NUTS_ID'] for f in payload['features'] if int(f['properties']['LEVL_CODE'])==level)

def validate_nuts_coverage(survey_ids:Iterable[str],official_ids:Iterable[str])->dict[str,object]:
    survey=frozenset(str(x) for x in survey_ids); official=frozenset(official_ids); missing=survey-official
    return {'survey_regions':len(survey),'matched_regions':len(survey&official),'missing_regions':sorted(missing),'complete':not missing}
