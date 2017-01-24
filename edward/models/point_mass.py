"""The Point Mass distribution class."""

from __future__ import absolute_import, division, print_function

import tensorflow as tf

from edward.util import tile
from tensorflow.contrib.distributions.python.ops import distribution
from tensorflow.python.framework import dtypes, ops, tensor_shape
from tensorflow.python.ops import array_ops, math_ops

class PointMass(distribution.Distribution):
    """PointMass distribution.abs

    It is analogous to an Empirical random variable with one sample, but
    its parameter argument does not have an outer dimension.
    """
    def __init__(self, params, validate_args=False, allow_nan_stats=True, name="PointMass"):
        with ops.name_scope(name, values=[params]) as ns:
            with ops.control_dependencies([]):
                self._params = array_ops.identity(params, name="params")
                super(PointMass, self).__init__(
                    dtype=self._params.dtype,
                    parameters={"params": self._params},
                    is_continuous=False,
                    is_reparameterized=True,
                    validate_args=validate_args,
                    allow_nan_stats=allow_nan_stats,
                    name=ns
                )
    
    @staticmethod
    def _param_shapes(sample_shape):
        """distribution parameter."""
        return self._params

    def _batch_shape(self):
        return array_ops.constant([], dtype=dtypes.int32)

    def _get_batch_shape(self):
        return tensor_shape.scalar()

    def _event_shape(self):
        return array_ops.shape(self._params)