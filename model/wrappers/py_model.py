import numpy as np
import edward as ed
import tensorflow as tf
from edward.models import PythonModel, Beta
from scipy.stats import bernoulli, beta

class BetaBernoulli(PythonModel):
    """p(x, p) = Bernoulli(x | p) * Beta(p | 1, 1)"""
    def _py_log_prob(self, xs, zs):
        log_prior = beta.logpdf(zs['p'], a=1.0, b=1.0)
        log_lik = np.sum(bernoulli.logpmf(xs['x'], p=zs['p']))
        return log_lik + log_prior


ed.set_seed(42)
data = {'x': np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 1])}

model = BetaBernoulli()

qp_a = tf.nn.softplus(tf.Variable(tf.random_normal([])))
qp_b = tf.nn.softplus(tf.Variable(tf.random_normal([])))
qp = Beta(a=qp_a, b=qp_b)

inference = ed.KLqp({'p': qp}, data, model)
inference.run(n_iter=500)