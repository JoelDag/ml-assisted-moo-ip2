import json
from pathlib import Path
import pandas as pd

target_tp = 5
target_tf = 5
target_jut = 1.1

agg_dir = Path("aggregated_per_param")
rows = []

for file in agg_dir.glob("*.json"):
    if f"_tp{target_tp}_" not in file.name: continue
    if f"_tf{target_tf}_" not in file.name: continue
    if f"_jut{str(target_jut)}" not in file.name: continue

    with file.open() as f:
        data = json.load(f)

    rows.append({
        "problem": data["problem"],
        "hv_ip2_final": data["hv_ip2_final"],
        "hv_nsga2_final": data["hv_nsga2_final"],
        "igd_ip2_final": data["igd_ip2_final"],
        "igd_nsga2_final": data["igd_nsga2_final"]
    })

df = pd.DataFrame(rows).sort_values("problem")
df.reset_index(drop=True, inplace=True)
print(df.to_string(index=False))
df.to_csv(f"results_tp{target_tp}_tf{target_tf}_jut{target_jut}.csv", index=False)
