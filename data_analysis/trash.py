import os, json

files = os.listdir('/local/topic2/joel/final_random_search_runs/run_20250716_165631')
with open('files_list.json', 'w') as f:
    for file in files:
        json.dump(file, f)
        f.write('\n')