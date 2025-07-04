import os
import subprocess
import random
import argparse
import multiprocessing

from concurrent.futures import ProcessPoolExecutor, as_completed
from src.IP2.evolutionaryComputation import evolutionaryRunner
from src.IP2.utils import get_three_objectives_problems

os.chdir(os.path.dirname(os.path.abspath(__file__)))


def install_requirements():
    subprocess.run(["pip", "install", "-r", "requirements.txt"])

def random_search_space(num_samples=20):
    return [
        (
            random.choice([2, 5, 10]),      #tpast
            random.choice([1, 5, 10]),      #tfrq
            random.choice([1.0, 1.1, 1.3])  #jutting
        )
        for _ in range(num_samples)
    ]
search_space = random_search_space()

def run_problem(args_tuple):
    print(f"DEBUG: args_tuple = {args_tuple}")
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
        "makeMMF14Function",
        "makeMMF14aFunction",
        "makeMMF15Function",
        "makeMMF15aFunction"]
    test_problems = ["makeMMF8Function"]

    SEEDS = list(range(15))
    job_list = [
        (problem, t_past, t_freq, jutting, seed)
        for (t_past, t_freq, jutting) in search_space
        for problem in test_problems
        for seed in SEEDS
    ]
    
    parser = argparse.ArgumentParser(
        description="Starting Evolutionary Computation parallel or sequentially for multiple test problems.")
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Run problems sequentially")
    parser.add_argument(
        "--jobs", "-j",
        type=int,
        default=max(1, multiprocessing.cpu_count() - 1),
        help="Number of worker processes (only relevant if parallel mode is active)")
    args = parser.parse_args()

    results = {}
    results = parallelization(args.no_parallel, job_list, args.jobs)

    for prob, res in results.items():
        print(f"[main] Results for {prob}: {res}")
