import os
import subprocess
import random
import argparse
import multiprocessing
import json

from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
from src.IP2.evolutionaryComputation import evolutionaryRunner
from src.IP2.utils import get_three_objectives_problems

os.chdir(os.path.dirname(os.path.abspath(__file__)))


def install_requirements():
    subprocess.run(["pip", "install", "-r", "requirements.txt"])

def grid_search_space():
    t_past_options = [2, 5, 10]
    t_freq_options = [1, 5, 10]
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

def run_problem(args_tuple):
    problem, t_past, t_freq, jutting_param, seed = args_tuple
    print(f"Running {problem} with t_past={t_past}, t_freq={t_freq}, jutting={jutting_param}, seed={seed}")

    if problem in get_three_objectives_problems():
        n_var, m_obj = 3, 3
    elif problem == 'makeMMF13Function':
        n_var, m_obj = 3, 2
    else:
        n_var, m_obj = 2, 2
    runner = evolutionaryRunner(pop_size=100,
                                n_gen=100,
                                n_var=n_var,
                                m_obj=m_obj,
                                t_past=t_past,
                                t_freq=t_freq,
                                test_problem=problem,
                                jutting_param=jutting_param,
                                h_interval=3,
                                seed=seed)

    res = runner.run()

    job_id = f"{problem}_tp{t_past}_tf{t_freq}_jut{jutting_param}_seed{seed}"
    output_dir = os.environ.get("RUN_OUTPUT_DIR", "runs")
    os.makedirs(output_dir, exist_ok=True)
    result_path = os.path.join(output_dir, job_id + ".json")

    with open(result_path, "w") as f:
        json.dump(res, f, indent=2)

    print(f"[run_problem] Saved result to {result_path}")

    return (problem, res)

def parallelization(parallel, job_list, jobs):
    results: dict = {}
    if parallel:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = {pool.submit(run_problem, job): job for job in job_list}
            for fut in as_completed(futures):
                key = futures[fut]
                try:
                    results[key] = fut.result()
                except Exception as e:
                    print(f"[main] Failed job {key}: {e}")
    else:
        for job in job_list:
            try:
                results[job] = run_problem(job)
            except Exception as e:
                print(f"[main] Failed job {job}: {e}")
    print("[main] All computations finished.")
    return results

if __name__ == "__main__":
    install_requirements()

    parser = argparse.ArgumentParser(description="Starting Evolutionary Computation parallel or sequentially for multiple test problems.")
    parser.add_argument("--no-parallel", action="store_true", help="Run problems sequentially")
    parser.add_argument("--jobs", "-j", type=int, default=max(1, multiprocessing.cpu_count() - 1), help="num of worker processes (when parallel is active)")
    parser.add_argument("--logdir", type=str, default=None, help="dir to save logs and results. Defaults to './runs/'")
    parser.add_argument("--grid-search", action="store_true", help="Use grid search for hyperparameters")
    parser.add_argument("--random-search", action="store_true", help="Use random search for hyperparameters")
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
    test_problems = [
        "makeMMF1Function",
        "makeMMF3Function",
        "makeMMF8Function",
        "makeMMF11Function",
        "makeMMF14Function",
        "makeMMF14aFunction",
        "makeMMF15Function",
        "makeMMF15aFunction"]
    #test_problems = ["makeMMF8Function"]

    job_list = [
        (problem, t_past, t_freq, jutting, seed)
        for (t_past, t_freq, jutting) in search_space
        for problem in test_problems
        for seed in SEEDS
    ]

    results = {}
    results = parallelization(args.no_parallel, job_list, args.jobs)

    for prob, res in results.items():
        print(f"[main] Results for {prob}: {res}")
