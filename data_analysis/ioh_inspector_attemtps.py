import iohinspector
import matplotlib.pyplot as plt
import polars as pl

manager = iohinspector.DataManager()
manager.add_folder("../src/runs/NSGA_2_ioh_inspector_20250730_024827/IP2_makeMMF2Function_tp5_tf5_jut1.1_nest403_mdepthNA/")
manager.add_folder("../src/runs/NSGA_2_ioh_inspector_20250730_024827/NSGA2_makeMMF2Function_tp5_tf5_jut1.1_nest403_mdepthNA/")

ids = manager.overview.filter(
    (pl.col('algorithm_name').is_in(["IP2", "NSGA2"])) &
    (pl.col('n_estimators') == 403)
)['data_id']

manager_sub = manager.select(ids)
meta = manager_sub.overview
df = manager_sub.load()
df = df.join(meta, on='data_id', how='left')

df = iohinspector.metrics.add_normalized_objectives(df, obj_cols=['raw_y', 'F2'])

ref_set = iohinspector.indicators.get_reference_set(df, ['obj1', 'obj2'], 1000)
igdp_indicator = iohinspector.indicators.anytime.IGDPlus(reference_set=ref_set)
plot_df = iohinspector.plot.plot_indicator_over_time(
    df, ['obj1', 'obj2'], igdp_indicator,
    evals_min=1, evals_max=100, nr_eval_steps=20,
    free_variable='algorithm_name'
)
plt.title("Anytime IGD+ (lower is better)")
plt.savefig("trash_ioh_igdplus.png")
plt.show()
