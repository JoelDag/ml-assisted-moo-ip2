import rpy2.robjects as robjects
import numpy as np
from pymoo.core.problem import Problem
from src.IP2.utils import get_three_objectives_problems


class MMFfunction(Problem):

    def __init__(self, ea, smoofname: str = None):
        if smoofname in get_three_objectives_problems():
            self.func = robjects.r[smoofname](ea.n, ea.m)
        else:
            self.func = robjects.r[smoofname]()
        xl = robjects.r['getLowerBoxConstraints'](self.func)
        xu = robjects.r['getUpperBoxConstraints'](self.func)
        super().__init__(n_var=ea.n, n_obj=ea.m, xl=xl, xu=xu)

    def _evaluate(self, x: np.ndarray, out, *args, **kwargs):
        evals = np.zeros((x.shape[0], self.n_obj))
        for i, xin in enumerate(x):
            evals[i,:] = list(self.func(xin.tolist()))
        out["F"] = evals