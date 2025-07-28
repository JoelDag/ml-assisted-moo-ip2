import pandas as pd
import wandb

api = wandb.Api()
runs = api.runs("joeldag-paderborn-university/ml4moo_topic2")

rows = []

for run in runs:
    config = {k: v for k, v in run.config.items() if not k.startswith('_')}
    summary = run.summary._json_dict

    # Combine config and summary into one dictionary
    row = {**config, **summary}
    row['name'] = run.name
    rows.append(row)

# Create a DataFrame from all the flattened runs
df = pd.DataFrame(rows)

# OPTIONAL: Display available columns to decide what to average
print("Available columns:", df.columns.tolist())

# Group by parameters that define a unique setup
group_cols = ["problem", 't_freq', 't_past', 'jutting', 'hv_nsga2_final']  # Add more if needed

# Average all numeric columns per group
agg_df = df.groupby(group_cols).mean(numeric_only=True).reset_index()

# Save results
agg_df.to_csv("averaged_results_over_3_seeds.csv", index=False)

print(agg_df.head())
