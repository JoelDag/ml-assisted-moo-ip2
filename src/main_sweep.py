import os
import argparse
import multiprocessing
import json
import wandb
import numpy as np

from src.IP2.evolutionaryComputation import evolutionaryRunner
from src.IP2.utils import get_three_objectives_problems, setup_logger

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def parse_or_default(value, default):
    return None if value == "None" or value is None else value

def clean_val(val):
    return str(val) if val is not None else "NA"

def run_problem(args_tuple):
    problem, t_past, t_freq, jutting_param, seed, use_wandb, rf_params = args_tuple
    print(f"Running {problem} with t_past={t_past}, t_freq={t_freq}, jutting={jutting_param}, seed={seed}")

    if problem in get_three_objectives_problems():
        n_var, m_obj, pop_size = 3, 3, 105
    elif problem == 'makeMMF13Function':
        n_var, m_obj, pop_size = 3, 2, 100
    else:
        n_var, m_obj, pop_size = 2, 2, 100

    runner = evolutionaryRunner(pop_size=pop_size,
                                n_gen=100,
                                n_var=n_var,
                                m_obj=m_obj,
                                t_past=t_past,
                                t_freq=t_freq,
                                test_problem=problem,
                                jutting_param=jutting_param,
                                h_interval=3,
                                seed=seed,
                                rf_params=rf_params)
    res = runner.run()

    if use_wandb:
        wandb.log({k: v for k, v in res.items() if "_final" in k})
        wandb.run.summary.update({k: v for k, v in res.items() if "_final" in k})

    job_id = (
        f"{problem}_tp{t_past}_tf{t_freq}_jut{jutting_param}_seed{seed}"
        f"_nest{clean_val(rf_params.get('n_estimators'))}"
        f"_mdepth{clean_val(rf_params.get('max_depth'))}"
    )
    output_dir = os.environ.get("RUN_OUTPUT_DIR", "runs")
    os.makedirs(output_dir, exist_ok=True)
    result_path = os.path.join(output_dir, job_id + ".json")

    with open(result_path, "w") as f:
        json.dump(res, f, indent=2)

    print(f"[run_problem] Saved result to {result_path}")

    return (problem, res)


if __name__ == "__main__":
    setup_logger()

    parser = argparse.ArgumentParser(description="Starting Evolutionary Computation parallel or sequentially for multiple test problems.")
    parser.add_argument("--jobs", "-j", type=int, default=max(1, multiprocessing.cpu_count() - 1), help="num of worker processes (when parallel is active)")
    parser.add_argument("--logdir", type=str, default=None, help="dir to save logs and results. Defaults to './runs/'")
    parser.add_argument("--wandb", action="store_true", help = "If set, log every run to Weights & Biases")
    args = parser.parse_args()

    if args.wandb:
        run = wandb.init(
            project=os.getenv("WANDB_PROJECT"),
            entity=os.getenv("WANDB_ENTITY"),
            job_type="sweep_run",
        )
        config = wandb.config
        
        rf_params = {
            "n_estimators": parse_or_default(config.n_estimators, None),
            "max_depth": parse_or_default(config.max_depth, None),
        }

        #seeds = [0, 1, 2, 3, 4]
        seeds = list(range(14))
        all_results = []

        for seed in seeds:
            _, res = run_problem((
                config.problem,
                config.t_past,
                config.t_freq,
                config.jutting,
                seed,
                False,
                rf_params
            ))
            all_results.append(res)

        agg = {}
        for key in all_results[0].keys():
            if key.endswith("_final"):
                vals = [r[key] for r in all_results]
                agg[key.replace("_final", "_mean")] = float(np.mean(vals))
                agg[key.replace("_final", "_std")] = float(np.std(vals))

        wandb.log(agg)

        output_dir = os.environ.get("RUN_OUTPUT_DIR", "runs")
        os.makedirs(output_dir, exist_ok=True)
        agg_path = os.path.join(
            output_dir,
            f"{config.problem}_tp{config.t_past}_tf{config.t_freq}_jut{config.jutting}"
            f"_nest{clean_val(config.n_estimators)}_mdepth{clean_val(config.max_depth)}_agg.json"
        )
        with open(agg_path, "w") as f:
            json.dump(agg, f, indent=2)
        wandb.finish()
    else:
        # Optional: test one job manually
        job = ("makeMMF1Function", 5, 8, 1.1, 0, False, {})
        run_problem(job)
