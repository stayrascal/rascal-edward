import tensorflow as tf
from edward.stats import bernoulli, beta
from edward.models import Beta
import numpy as np
import edward as ed


class BetaBernoulli:
    """p(x, p) = Bernoullix(x | p) * Beta(p | 1, 1)"""
    def log_prob(self, xs, zs):
        log_prior = beta.logpdf(zs['p'], a=1.0, b=1.0)
        log_lik = tf.reduce_sum(bernoulli.logpmf(xs['x'], p=zs['p']))
        return log_lik + log_prior

model = BetaBernoulli()
qp = Beta(a=tf.nn.softplus(tf.Variable(0.0)),
        b=tf.nn.softplus(tf.Variable(0.0)))
data = {'x': np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 1])}
inference = ed.KLqp({'p': qp}, data, model)
inference.run(n_iter=500)