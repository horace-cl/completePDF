from typing import List, Optional
from collections import OrderedDict

from zfit.minimizers.baseminimizer import BaseMinimizer
from zfit.minimizers.fitresult import FitResult
from zfit.core.interfaces import ZfitLoss
from zfit.core.parameter import Parameter

from scipy.optimize import minimize

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
import zfit

class SLSQP(BaseMinimizer):
    
    def __init__(self, tolerance=None, verbosity=5, name='SLSQP', 
                 constraints = (),
                 **minimizer_options):
        
        self.constraints = constraints
        super().__init__(tolerance=tolerance, 
                         name=name, verbosity=verbosity, 
                         minimizer_options=minimizer_options)
        
        
        
    def _minimize(self, loss: ZfitLoss, params: List[Parameter]):
        
        start_values = [p.numpy() for p in params]
        limits = tuple(tuple((p.lower, p.upper)) for p in params)
        
        def func(values):
            params = loss.get_params()
            with zfit.param.set_values(params, values):
                 val = loss.value()
            return val
        
        start_values = zfit.run(params)
        minimizer = minimize(
            fun=func,  x0=start_values,
            args=(), method='SLSQP', bounds=limits, 
            constraints=self.constraints, tol=self.tolerance,
            callback=None, 
            options = self.minimizer_options)
        
        self._update_params(params=loss.get_params(), values=minimizer.x)

        params = OrderedDict((p, res) for p, res in zip(loss.get_params(), minimizer.x))
        fitresult = FitResult(
                              loss = loss, minimizer=minimize,
                              params =params,
                              edm = -1.0, fmin = minimizer.fun,
                              status =minimizer.status,
                              converged = minimizer.success,
                              info = dict(minimizer) )
        
        return fitresult
    
    
    
    
    
    

