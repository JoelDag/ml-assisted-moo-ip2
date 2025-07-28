import os
import json
import pandas as pd
from glob import glob
from collections import defaultdict

ROOT_DIR = "/local/topic2/joel/final_random_search_runs/run_20250716_165631"
EXT = "_agg.json"
PROBLEMS = [
    "makeMMF1Function", "makeMMF1eFunction", "makeMMF1zFunction", "makeMMF2Function",
    "makeMMF3Function", "makeMMF4Function", "makeMMF5Function", "makeMMF6Function",
    "makeMMF7Function", "makeMMF8Function", "makeMMF9Function", "makeMMF10Function",
    "makeMMF11Function", "makeMMF12Function", "makeMMF13Function", "makeMMF14Function",
    "makeMMF14aFunction", "makeMMF15Function", "makeMMF15aFunction",
    "makeOmniTestFunction", "makeSYMPARTrotatedFunction", "makeSYMPARTsimpleFunction"
]

print("Collecting .json files...")
all_files = glob(os.path.join(ROOT_DIR, f"*{EXT}"))
files_by_problem = defaultdict(list)

for file in all_files:
    for problem in PROBLEMS:
        if os.path.basename(file).startswith(problem):
            files_by_problem[problem].append(file)
            break

# ----------- STEP 2: PARSE & FIND BESTS -----------
records = []

for problem, file_list in files_by_problem.items():
    best_igd_ip2 = None
    best_hv_ip2 = None

    for fpath in file_list:
        try:
            with open(fpath, 'r') as f:
                data = json.load(f)
                igd_ip2 = data.get('igd_ip2_mean', float('inf'))
                hv_ip2 = data.get('hv_ip2_mean', float('-inf'))

                # Track best IGD (lower is better)
                if best_igd_ip2 is None or igd_ip2 < best_igd_ip2['igd_ip2_mean']:
                    best_igd_ip2 = {
                        'file': fpath,
                        'igd_ip2_mean': igd_ip2,
                        'igd_nsga2_mean': data.get('igd_nsga2_mean', None),
                        'hv_ip2_mean': data.get('hv_ip2_mean', None),
                        'hv_nsga2_mean': data.get('hv_nsga2_mean', None)
                    }

                # Track best HV (higher is better)
                if best_hv_ip2 is None or hv_ip2 > best_hv_ip2['hv_ip2_mean']:
                    best_hv_ip2 = {
                        'file': fpath,
                        'igd_ip2_mean': data.get('igd_ip2_mean', None),
                        'igd_nsga2_mean': data.get('igd_nsga2_mean', None),
                        'hv_ip2_mean': hv_ip2,
                        'hv_nsga2_mean': data.get('hv_nsga2_mean', None)
                    }
        except Exception as e:
            print(f"Error processing {fpath}: {e}")

    if best_igd_ip2:
        records.append({
            'problem': problem,
            'type': 'best_igd_ip2',
            **best_igd_ip2
        })
    if best_hv_ip2:
        records.append({
            'problem': problem,
            'type': 'best_hv_ip2',
            **best_hv_ip2
        })

import re

def extract_params(filename):
    match = re.search(r'_tp(?P<tp>[^_]+)_tf(?P<tf>[^_]+)_jut(?P<jut>[^_]+)_nest(?P<nest>[^_]+)_mdepth(?P<mdepth>[^_]+)', filename)
    if match:
        return match.groupdict()
    return {'tp': None, 'tf': None, 'jut': None, 'nest': None, 'mdepth': None}

df = pd.DataFrame(records)
df['filename'] = df['file'].apply(os.path.basename)
params_df = df['filename'].apply(extract_params).apply(pd.Series)
df = pd.concat([df, params_df], axis=1)

df = df[['problem', 'type', 'tp', 'tf', 'jut', 'nest', 'mdepth', 'igd_nsga2_mean', 'igd_ip2_mean', 'hv_nsga2_mean', 'hv_ip2_mean']]


def sort_key(problem_name):
    match = re.search(r'make.*?(\d+)', problem_name)
    return int(match.group(1)) if match else float('inf')

df = df.sort_values(by='problem', key=lambda col: col.map(sort_key))

pivot_df = df.pivot(index='problem', columns='type')

for t, group in df.groupby('type'):
    group.sort_values('problem').drop(columns='type').to_csv(f"{t}.csv", index=False)


df.to_csv("best_runs_comparison.csv", index=False)
pivot_df.to_excel("best_runs_pivot_table.xlsx")

