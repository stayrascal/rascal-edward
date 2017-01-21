import numpy as np
import pymc3 as pm
import theano
from edward.models import PyMC3Model

x_obs = theano.shared(np.zeros(1))
with pm.Model() as pm_model:
  p = pm.Beta('p', 1, 1, transform=None)
  x = pm.Bernoulli('x', p, observed=x_obs)

model = PyMC3Model(pm_model)