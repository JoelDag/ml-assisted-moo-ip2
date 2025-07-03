import os
import subprocess
import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from src.IP2.evolutionaryComputation import evolutionaryRunner
from src.IP2.utils import get_three_objectives_problems

os.chdir(os.path.dirname(os.path.abspath(__file__)))


def install_requirements():
    subprocess.run(["pip", "install", "-r", "requirements.txt"])

def run_problem(problem):
    print(f"Running evolutionary computation for {problem}...")
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
                                t_past=5,
                                t_freq=5,
                                test_problem=problem,
                                jutting_param=1.1,
                                h_interval=3)

    res = runner.run()
    return problem, res

def parallelization(parallel, test_problems, jobs):
    if parallel or len(test_problems) == 1:
        for prob in test_problems:
            print(f"[main] Starting Computation for {prob} ...")
            _, res = run_problem(prob)
            results[prob] = res
            print(f"[main] Finished with {prob}")
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = {pool.submit(run_problem, prob): prob for prob in test_problems}
            for fut in as_completed(futures):
                prob, res = fut.result()
                results[prob] = res
                print(f"[main] Finished with {prob}")
    print("[main] All computations finished.")

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
    results = parallelization(args.no_parallel, test_problems, args.jobs)

    for prob, res in results.items():
        print(f"[main] Results for {prob}: {res}")
