import pandas as pd, numpy as np, json
cfg = json.load(open("config.json"))
s = pd.read_parquet(f"{cfg['weekly_results_dir']}/surface_predictions.parquet")
w = s[s.iso_week == 30]

# cells per latitude row, around the suspect band
c = w.groupby("lat").size()
band = c[(c.index > 25.0) & (c.index < 25.8)]
print(band.sort_index())

# is the row missing, or present-but-NaN?
print("NaN probs in band:",
      w[(w.lat > 25.0) & (w.lat < 25.8)]["prob_xgboost"].isna().sum())
col = [c for c in w.columns if c.startswith("prob_")][0]
print("column:", col)
b = w[(w.lat > 25.2) & (w.lat < 25.6)]
print("NaN:", b[col].isna().sum(), "of", len(b))
print("NaN opacity:", b["opacity"].isna().sum())

# longitude extent of the thinnest row vs its neighbours
thin = band.idxmin()
for L in sorted(band.index)[max(0, sorted(band.index).index(thin)-2):][:5]:
    r = w[w.lat == L]
    print(f"{L:.3f}  n={len(r):3d}  lon {r.lon.min():.2f}..{r.lon.max():.2f}")