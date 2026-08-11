"""Convert the LIHC data requirements CSV to an XLSX file.

Run this from the repository root inside a Python environment with pandas and openpyxl installed:

pip install pandas openpyxl
python scripts/make_lihc_requirements_xlsx.py
"""
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
csv_path = ROOT / "outputs" / "tables" / "lihc_data_requirements.csv"
xlsx_path = ROOT / "outputs" / "tables" / "lihc_data_requirements.xlsx"

df = pd.read_csv(csv_path)
df.to_excel(xlsx_path, index=False)
print(f"Wrote {xlsx_path}")
