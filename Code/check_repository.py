"""Quick repository sanity check: file presence, data sizes, and key metric values."""
from __future__ import annotations

from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
required = [
    ROOT / "Data/raw/Cu-Cr-X dataset.xlsx",
    ROOT / "Data/processed/cucrx_hardness_processed.csv",
    ROOT / "Data/processed/cucrx_conductivity_processed.csv",
    ROOT / "Tables_and_Results/metrics_summary_normalized.csv",
    ROOT / "Tables_and_Results/predictions.csv",
    ROOT / "Tables_and_Results/modeling_state.pkl",
]
missing = [str(p.relative_to(ROOT)) for p in required if not p.exists()]
if missing:
    raise FileNotFoundError("Missing required files: " + ", ".join(missing))

h = pd.read_csv(ROOT / "Data/processed/cucrx_hardness_processed.csv")
c = pd.read_csv(ROOT / "Data/processed/cucrx_conductivity_processed.csv")
metrics = pd.read_csv(ROOT / "Tables_and_Results/metrics_summary_normalized.csv")
with open(ROOT / "Tables_and_Results/split_info.json", encoding="utf-8") as f:
    split = json.load(f)

print(f"Hardness records: {len(h)}")
print(f"Electrical-conductivity records: {len(c)}")
print("Split info:", split)
print(metrics.loc[(metrics.target == "hardness") & (metrics.model == "LGBM"),
                  ["val_r2_percent", "val_nrmse_percent", "val_nmae_percent"]].to_string(index=False))
print(metrics.loc[(metrics.target == "conductivity") & (metrics.model == "LGBM"),
                  ["val_r2_percent", "val_nrmse_percent", "val_nmae_percent"]].to_string(index=False))
print("Repository check completed.")
