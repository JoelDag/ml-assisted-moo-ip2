# Machine Learning Assisted Evolutionary Multi- and Many-Objective Optimization

This repository contains the implementation of the Enhanced Innovized Progress Operator for evolutionary multi- and many-objective optimization, developed for the seminar Machine Learning Assisted Evolutionary Multi- and Many-Objective Optimization in SS 2025 at the University of Paderborn.
The code implements the Innovized Progress operator (IP²) mechanism on top of NSGA-II/NSGA-III for solving multimodal multi-objective benchmark problems.
The implementation relies on Python and pymo for the evolutionary algorithm and uses rpy2 to call test functions from the R package smoof. Reference data for the benchmark suite is included under Reference_PSPF_data.

## Repository Structure

- `src/main.py` – entry point that starts the experiments and installs dependencies on first run.
- `src/IP2/` – modules implementing the IP2 method:
  - `evolutionaryComputation.py` – manages the optimization loop and evaluation metrics.
  - `integration.py` – wraps NSGA-II/III with archive handling and machine learning based progress operations.
  - `ml_training_module.py` – trains Random Forest models to guide offspring repair.
  - `input_archive.py` – utilities for maintaining the target and training archives.
  - `utils.py` – helper functions for hypervolume, IGD and plotting.
- `src/MMFProblem/` – wrapper around R **smoof** functions defining the MMF test problems.
- `Reference_PSPF_data/` – MATLAB files providing reference Pareto sets/fronts for performance metrics.
- `Papers/` – related literature used during the seminar.

## Requirements

- Python 3.x
- R installation with the `smoof` package available to `rpy2`

Install Python dependencies with:

```bash
pip install -r src/requirements.txt
```

## Running the Example

Execute the main script from the repository root:

```bash
python -m src.main
```

You can modify the `test_problems` list in `src/main.py` to evaluate specific problems only. 
Use `--no-parallel` for sequential execution or `-j <jobs>` to specify the number of worker processes.
During execution, the script records the hypervolume and IGD values and saves plots in `plots_for_<algorithm>/`.
