import pandas as pd
import matplotlib.pyplot as plt
import os

# Generates and saves HV and IGD boxplots for each problem from CSV metrics.
csv_file = '../all_metrics_main_table_NSGA_2.csv'
output_dir = 'submission_NSGA2_boxplots'

df = pd.read_csv(csv_file)
problems = df['problem'].unique()

os.makedirs(output_dir, exist_ok=True)

for problem in problems:
    df_p = df[df['problem'] == problem]
    
    hv_ip2 = df_p['hv_ip2_final'].values
    hv_nsga2 = df_p['hv_nsga2_final'].values
    igd_ip2 = df_p['igd_ip2_final'].values
    igd_nsga2 = df_p['igd_nsga2_final'].values
    
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    
    # HV Boxplot
    axs[0].boxplot([hv_ip2, hv_nsga2], labels=['NSGA-II+IP$^2$', 'NSGA-II'])
    axs[0].set_title('HV')
    axs[0].set_ylabel('HV')
    axs[0].grid(True, linestyle='--', alpha=0.5)
    
    # IGD Boxplot
    axs[1].boxplot([igd_ip2, igd_nsga2], labels=['NSGA-II+IP$^2$', 'NSGA-II'])
    axs[1].set_title('IGD')
    axs[1].set_ylabel('IGD')
    axs[1].grid(True, linestyle='--', alpha=0.5)
    
    fig.suptitle(f'Performance on {problem}')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save
    filename = os.path.join(output_dir, f'{problem}_boxplots.png')
    plt.savefig(filename, dpi=300)
    plt.close()

print(f'All boxplots saved in: {output_dir}/')
