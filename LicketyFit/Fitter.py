from iminuit import Minuit
import numpy as np

class Fitter:
    """
    A class to perform fitting using the Minuit library.

    Attributes:
        function (callable): The function to be minimized.
        initial_params (dict): A dictionary of initial parameter values.
        param_errors (dict): A dictionary of parameter errors.
        limits (dict): A dictionary of parameter limits.
        fixed (dict): A dictionary indicating which parameters are fixed.
        minimizer (Minuit): The Minuit minimizer object.
    """

    def __init__(self, function, initial_params, param_errors=None, limits=None, fixed=None):
        if not callable(function) and not isinstance(function, str):
            raise TypeError("function must be callable or a string representing a particular likelihood calculation")
        if not isinstance(initial_params, dict) or not all(isinstance(k, str) and isinstance(v, (int, float)) for k, v in initial_params.items()):
            raise TypeError("initial_params must be a dictionary with string keys and numeric values")
        if param_errors is not None and (not isinstance(param_errors, dict) or not all(isinstance(k, str) and isinstance(v, (int, float)) for k, v in param_errors.items())):
            raise TypeError("param_errors must be a dictionary with string keys and numeric values")
        if limits is not None and (not isinstance(limits, dict) or not all(isinstance(k, str) and isinstance(v, tuple) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v) for k, v in limits.items())):
            raise TypeError("limits must be a dictionary with string keys and tuple of two numeric values")
        if fixed is not None and (not isinstance(fixed, dict) or not all(isinstance(k, str) and isinstance(v, bool) for k, v in fixed.items())):
            raise TypeError("fixed must be a dictionary with string keys and boolean values")

        self.function = function
        self.initial_params = initial_params
        self.param_errors = param_errors if param_errors is not None else {k: 1.0 for k in initial_params}
        self.limits = limits if limits is not None else {}
        self.fixed = fixed if fixed is not None else {k: False for k in initial_params}

        self.minimizer = Minuit(self.function, **self.initial_params)
        for param in self.initial_params:
            if param in self.param_errors:
                self.minimizer.errors[param] = self.param_errors[param]
            if param in self.limits:
                self.minimizer.limits[param] = self.limits[param]
            if param in self.fixed:
                self.minimizer.fixed[param] = self.fixed[param]
        self.minimizer.errordef = Minuit.LIKELIHOOD

    def minimize(self):
        """Perform the minimization using Minuit."""
        self.minimizer.migrad()
        if not self.minimizer.valid:
            raise RuntimeError("Minimization did not converge")
        return self.minimizer.values, self.minimizer.errors

