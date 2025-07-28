import json, re
from pathlib import Path
import numpy as np

"""
This averages the results form the json over all seeds
"""

run_dir = Path('/local/joeldag/ml4moo_topic2/src/runs/run_20250706_062120')
out_dir = Path('aggregated_per_param')
out_dir.mkdir(exist_ok=True)

pattern = re.compile(r'(?P<problem>.+)_tp(?P<tp>\d+)_tf(?P<tf>\d+)_jut(?P<jut>[\d.]+)_seed(?P<seed>\d+)\.json')

groups = {}
for f in run_dir.glob('*.json'):
    m = pattern.match(f.name)
    if not m: continue
    key = (m['problem'], int(m['tp']), int(m['tf']), float(m['jut']))
    with f.open() as fp:
        groups.setdefault(key, []).append(json.load(fp))

for (problem, tp, tf, jut), runs in groups.items():
    result = {
        'problem': problem, 't_past': tp, 't_freq': tf, 'jutting': jut,
        'n_seeds': len(runs)
    }
    for k in ['hv_ip2_final', 'hv_nsga2_final', 'igd_ip2_final', 'igd_nsga2_final']:
        result[k] = float(np.mean([r[k] for r in runs]))
    for k in ['hv_ip2', 'hv_nsga2', 'igd_ip2', 'igd_nsga2']:
        arrs = [np.array(r[k]) for r in runs]
        result[k] = np.mean(np.stack(arrs), axis=0).tolist()
    fname = f'{problem}_tp{tp}_tf{tf}_jut{jut}.json'
    with open(out_dir / fname, 'w') as f:
        json.dump(result, f, indent=2)
