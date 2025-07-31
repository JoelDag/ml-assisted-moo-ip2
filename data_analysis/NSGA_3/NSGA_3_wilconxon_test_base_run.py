from scipy.stats import wilcoxon
import pandas as pd

# Performs Wilcoxon tests to compare NSGA-III and NSGA-III+IP2 metrics per problem and outputs LaTeX table.
df = pd.read_csv('../all_metrics_NSGA_3.csv')
df = df.sort_values('problem')
print(df.columns)


problems = df['problem'].unique()
ergebnisse = []

for prob in problems:
    subset = df[df['problem'] == prob]
    
    hv_ip2 = subset['hv_ip2_final']
    hv_nsga2 = subset['hv_nsga2_final']
    igd_ip2 = subset['igd_ip2_final']
    igd_nsga2 = subset['igd_nsga2_final']
    
    hv_stat, hv_p = wilcoxon(hv_ip2, hv_nsga2)
    igd_stat, igd_p = wilcoxon(igd_ip2, igd_nsga2)
    
    ergebnisse.append({
        'Problem': prob,
        'hv_ip2_mean': hv_ip2.mean(),
        'hv_ip2_std': hv_ip2.std(),
        'hv_nsga2_mean': hv_nsga2.mean(),
        'hv_nsga2_std': hv_nsga2.std(),
        'hv_signifikant': hv_p < 0.05,
        'igd_ip2_mean': igd_ip2.mean(),
        'igd_ip2_std': igd_ip2.std(),
        'igd_nsga2_mean': igd_nsga2.mean(),
        'igd_nsga2_std': igd_nsga2.std(),
        'igd_signifikant': igd_p < 0.05
    })

# In Tabelle packen
result_df = pd.DataFrame(ergebnisse)
print(result_df)

def fmt(mean, std, h, bold=False):
    num = f"{mean:.6f} $\\pm$ {std:.6f} ({int(h)})"
    if bold:
        return f"\\textbf{{{num}}}"
    else:
        return num

rows = []
for idx, row in result_df.iterrows():
    # for HV, higher better for IGD lower 
    hv_best_is_ip2 = row['hv_ip2_mean'] > row['hv_nsga2_mean']
    igd_best_is_ip2 = row['igd_ip2_mean'] < row['igd_nsga2_mean']
    
    # h = 0 für besten Wert (nicht signifikant schlechter als der andere), sonst 1
    hv_ip2_h = 0 if (hv_best_is_ip2 or not row['hv_signifikant']) else 1
    hv_nsga2_h = 0 if (not hv_best_is_ip2 or not row['hv_signifikant']) else 1
    igd_ip2_h = 0 if (igd_best_is_ip2 or not row['igd_signifikant']) else 1
    igd_nsga2_h = 0 if (not igd_best_is_ip2 or not row['igd_signifikant']) else 1
    
    rows.append(
        f"{row['Problem']} & "
        f"{fmt(row['hv_ip2_mean'], row['hv_ip2_std'], hv_ip2_h, bold=hv_best_is_ip2)} & "
        f"{fmt(row['hv_nsga2_mean'], row['hv_nsga2_std'], hv_nsga2_h, bold=not hv_best_is_ip2)} & "
        f"{fmt(row['igd_ip2_mean'], row['igd_ip2_std'], igd_ip2_h, bold=igd_best_is_ip2)} & "
        f"{fmt(row['igd_nsga2_mean'], row['igd_nsga2_std'], igd_nsga2_h, bold=not igd_best_is_ip2)} \\\\"
    )
    
hv_ip2_better = (result_df['hv_ip2_mean'] > result_df['hv_nsga2_mean']).sum()
hv_nsga2_better = (result_df['hv_ip2_mean'] < result_df['hv_nsga2_mean']).sum()
hv_equal = (result_df['hv_ip2_mean'] == result_df['hv_nsga2_mean']).sum()

igd_ip2_better = (result_df['igd_ip2_mean'] < result_df['igd_nsga2_mean']).sum()
igd_nsga2_better = (result_df['igd_ip2_mean'] > result_df['igd_nsga2_mean']).sum()
igd_equal = (result_df['igd_ip2_mean'] == result_df['igd_nsga2_mean']).sum()

summary_row = (
    "Total better & "
    f"{hv_ip2_better} & {hv_nsga2_better} & {igd_ip2_better} & {igd_nsga2_better} \\\\"
)


table_latex = """
\\begin{table}[ht]
\\centering
\\caption{Comparison of NSGA-III and NSGA-III+IP$^2$ (mean $\\pm$ std (h)).}
\\begin{tabular}{l|cc|cc}
\\hline
Problem & HV IP$^2$ & HV NSGA-III & IGD IP$^2$ & IGD NSGA-III \\\\
\\hline
"""
table_latex += "\n".join(rows)
table_latex += "\n" + summary_row + "\n"
table_latex += """
\\hline
\\end{tabular}
\\end{table}
"""

print(table_latex)