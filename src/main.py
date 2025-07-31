import os
import random
import argparse
import multiprocessing
import json
import traceback
import wandb

from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
from src.IP2.evolutionaryComputation import evolutionaryRunner
from src.IP2.utils import get_three_objectives_problems, setup_logger

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def grid_search_space():
    t_past_options = [2, 5, 8, 10]
    t_freq_options = [1, 5, 8, 10]      #TODO: add time analysis for setups, #TOOD: distinguish btw generations with /without using RandomForest
    jutting_options = [1.0, 1.1, 1.3]
    return list(product(t_past_options, t_freq_options, jutting_options))

def random_search_space(num_samples=20):
    return [
        (
            random.choice([2, 5, 10]),      #tpast
            random.choice([1, 5, 10]),      #tfrq
            random.choice([1.0, 1.1, 1.3])  #jutting
        )
        for _ in range(num_samples)
    ]

def run_problem(args_tuple, model_performance):
    use_wandb = args_tuple[-1]
    if use_wandb:
        wandb.init()
        config = wandb.config
        problem = config.problem
        t_past = config.t_past
        t_freq = config.t_freq
        jutting_param = config.jutting
        seed = config.seed
    else:
        problem, t_past, t_freq, jutting_param, seed = args_tuple[:-1]
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
                                model_performance=model_performance)
    project = os.getenv("WANDB_PROJECT")
    entity = os.getenv("WANDB_ENTITY")
    if use_wandb:
        run = wandb.init(project=project, entity=entity, job_type=problem, config=dict(problem=problem, t_past=t_past, t_freq=t_freq, jutting=jutting_param, seed=seed, pop_size=pop_size, n_gen=100))
    res = runner.run()
    if use_wandb:
        wandb.log({k: v for k, v in res.items() if "_final" in k})
        run.summary.update({k: v for k, v in res.items() if "_final" in k})
        run.finish()

    job_id = f"{problem}_tp{t_past}_tf{t_freq}_jut{jutting_param}_seed{seed}"
    output_dir = os.environ.get("RUN_OUTPUT_DIR", "runs")
    os.makedirs(output_dir, exist_ok=True)
    result_path = os.path.join(output_dir, job_id + ".json")

    with open(result_path, "w") as f:
        json.dump(res, f, indent=2)

    print(f"[run_problem] Saved result to {result_path}")

    return (problem, res)

def parallelization(parallel, job_list, jobs, model_performance):
    results: dict = {}
    if parallel:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = {pool.submit(run_problem, job, model_performance): job for job in job_list}
            for fut in as_completed(futures):
                key = futures[fut]
                try:
                    results[key] = fut.result()
                except Exception as e:
                    print(f"[main] Failed job {key}: {e}")
                    traceback.print_exc()
    else:
        for job in job_list:
            try:
                results[job] = run_problem(job, model_performance)
            except Exception as e:
                print(f"[main] Failed job {job}: {e}")
                traceback.print_exc()
    print("[main] All computations finished.")
    return results

if __name__ == "__main__":
    setup_logger()

    parser = argparse.ArgumentParser(description="Starting Evolutionary Computation parallel or sequentially for multiple test problems.")
    parser.add_argument("--parallel", action="store_true", help="If set, run problems parallel")
    parser.add_argument("--jobs", "-j", type=int, default=max(1, multiprocessing.cpu_count() - 1), help="num of worker processes (when parallel is active)")
    parser.add_argument("--logdir", type=str, default=None, help="dir to save logs and results. Defaults to './runs/'")
    parser.add_argument("--grid-search", action="store_true", help="Use grid search for hyperparameters")
    parser.add_argument("--random-search", action="store_true", help="Use random search for hyperparameters")
    parser.add_argument("--wandb", action="store_true", help = "If set, log every run to Weights & Biases")
    parser.add_argument("--model-performance", action="store_true", help="If set, evaluate the model performance after training")

    args = parser.parse_args()

    if args.grid_search:
        search_space = grid_search_space()
    elif args.random_search:
        search_space = random_search_space()
    else:
        search_space = [(5, 5, 1.1)] #default values from paper (tpast, t_freq, jutting)

    if args.grid_search or args.random_search:
        SEEDS = list(range(3))
    else:
        SEEDS = [0]

    test_problems = [
        "makeMMF1Function",
        "makeMMF1eFunction",
        "makeMMF1zFunction",
        "makeMMF2Function",
        "makeMMF3Function",
        "makeMMF4Function",
        "makeMMF5Function",
        "makeMMF6Function",
        "makeMMF7Function",
        "makeMMF8Function",
        "makeMMF9Function",
        "makeMMF10Function",
        "makeMMF11Function",
        "makeMMF12Function",
        "makeMMF13Function",
        "makeMMF14Function",
        "makeMMF14aFunction",
        "makeMMF15Function",
        "makeMMF15aFunction",
        "makeOmniTestFunction",
        "makeSYMPARTrotatedFunction",
        "makeSYMPARTsimpleFunction"]
    # test_problems = [
    #     "makeMMF1Function",
    #     "makeMMF3Function",
    #     "makeMMF8Function",
    #     "makeMMF11Function",
    #     "makeMMF14Function",
    #     "makeMMF14aFunction",
    #     "makeMMF15Function",
    #     "makeMMF15aFunction"]
    # test_problems = ["makeMMF8Function"]

    job_list = [
        (problem, t_past, t_freq, jutting, seed, args.wandb)
        for (t_past, t_freq, jutting) in search_space
        for problem in test_problems
        for seed in SEEDS
    ]

    results = {}
    results = parallelization(args.parallel, job_list, args.jobs, args.model_performance)

    for prob, res in results.items():
        print(f"[main] Results for {prob}: {res}")
