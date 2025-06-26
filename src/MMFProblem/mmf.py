import rpy2.robjects as robjects
import numpy as np
from pymoo.core.problem import Problem

class MMFfunction(Problem):

    def __init__(self, smoofname: str = None):
        self.func = robjects.r[smoofname]()
        xl = robjects.r['getLowerBoxConstraints'](self.func)
        xu = robjects.r['getUpperBoxConstraints'](self.func)
        super().__init__(n_var=2, n_obj=2, xl=xl, xu=xu)

    def _evaluate(self, x: np.ndarray, out, *args, **kwargs):
        evals = np.zeros((x.shape[0], self.n_obj))
        for i, xin in enumerate(x):
            evals[i,:] = list(self.func(xin.tolist()))
        out["F"] = evals