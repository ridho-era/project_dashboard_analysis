import pandas as pd
H = pd.read_parquet("data/processed/aqms_hourly_clean.parquet", columns=["stasiun","ts_jam","pm25_ug_m3"])
p = H.pivot(index="ts_jam", columns="stasiun", values="pm25_ug_m3")[["KTG-01","KTG-02"]].dropna()
print(p.corr().iloc[0, 1]); print((p["KTG-01"] - p["KTG-02"]).abs().describe())