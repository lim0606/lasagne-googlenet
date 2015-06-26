"""
Functions to generate Theano update dictionaries for training.

The update functions implement different methods to control the learning
rate for use with stochastic gradient descent.

Update functions take a loss expression or a list of gradient expressions and
a list of parameters as input and return an ordered dictionary of updates:

.. autosummary::
    :nosignatures:

    adagrad_w_prior
"""

from collections import OrderedDict

import numpy as np

import theano
import theano.tensor as T
import lasagne.utils

__all__ = [
    "adagrad_w_prior",
]


def get_or_compute_grads(loss_or_grads, params):
    """Helper function returning a list of gradients

    Parameters
    ----------
    loss_or_grads : symbolic expression or list of expressions
        A scalar loss expression, or a list of gradient expressions
    params : list of shared variables
        The variables to return the gradients for

    Returns
    -------
    list of expressions
        If `loss_or_grads` is a list, it is assumed to be a list of
        gradients and returned as is, unless it does not match the length
        of `params`, in which case a `ValueError` is raised.
        Otherwise, `loss_or_grads` is assumed to be a cost expression and
        the function returns `theano.grad(loss_or_grads, params)`.
    """
    if isinstance(loss_or_grads, list):
        if not len(loss_or_grads) == len(params):
            raise ValueError("Got %d gradient expressions for %d parameters" %
                             (len(loss_or_grads), len(params)))
        return loss_or_grads
    else:
        return theano.grad(loss_or_grads, params)


def adagrad_w_prior(loss_or_grads, params, learning_rate=1.0, epsilon=1e-6, batchsize=1, num_data=1):
    """Adagrad updates

    Scale learning rates by dividing with the square root of accumulated
    squared gradients. See [1]_ for further description.

    Parameters
    ----------
    loss_or_grads : symbolic expression or list of expressions
        A scalar loss expression, or a list of gradient expressions
    params : list of shared variables
        The variables to generate update expressions for
    learning_rate : float or symbolic scalar
        The learning rate controlling the size of update steps
    epsilon : float or symbolic scalar
        Small value added for numerical stability

    Returns
    -------
    OrderedDict
        A dictionary mapping each parameter to its update expression

    Notes
    -----
    Using step size eta Adagrad calculates the learning rate for feature i at
    time step t as:

    .. math:: \\eta_{t,i} = \\frac{\\eta}
       {\\sqrt{\\sum^t_{t^\\prime} g^2_{t^\\prime,i}+\\epsilon}} g_{t,i}

    as such the learning rate is monotonically decreasing.

    Epsilon is not included in the typical formula, see [2]_.

    References
    ----------
    .. [1] Duchi, J., Hazan, E., & Singer, Y. (2011):
           Adaptive subgradient methods for online learning and stochastic
           optimization. JMLR, 12:2121-2159.

    .. [2] Chris Dyer:
           Notes on AdaGrad. http://www.ark.cs.cmu.edu/cdyer/adagrad.pdf
    """

    grads = get_or_compute_grads(loss_or_grads, params)
    updates = OrderedDict()

    for param, grad in zip(params, grads):
        value = param.get_value(borrow=True)
        accu = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                             broadcastable=param.broadcastable)
        accu_new = accu + grad ** 2
        updates[accu] = accu_new
        if param.name == 'W':
            p = 0.5
        else: # param.name == 'b'
            p = 0
        updates[param] = param \
                         - (learning_rate * grad / T.sqrt(accu_new + epsilon)) \
                         - learning_rate / T.sqrt(accu_new + epsilon) * p * np.array(batchsize, dtype=value.dtype) / np.array(num_data, dtype=value.dtype) * param

    return updates

'''
# manual update rule
prior = np.array([0.5, 0, 0.5, 0, 0.5, 0, 0.5, 0, 0.5, 0]) * all_params
#raise NameError("hi")
learning_rate = 0.01
epsilon = 0. #1e-6

from collections import OrderedDict

updates = OrderedDict()

for param, grad, p in zip(all_params, all_grads, prior):
    value = param.get_value(borrow=True)
    accu = theano.shared(0.01 * np.ones(value.shape, dtype=value.dtype),
                             broadcastable=param.broadcastable)
    accu_new = accu + grad ** 2
    updates[accu] = accu_new
    updates[param] = param + (learning_rate * (grad - p * param * np.array(batchsize, dtype=value.dtype) / np.array(num_data, dtype=value.dtype)) /
                              theano.tensor.sqrt(accu_new))
#                     param + learning_rate / theano.tensor.sqrt(accu_new) * grad - learning_rate / theano.tensor.sqrt(accu_new) * p * np.array(batchsize, dtype=value.dtype) / np.array(num_data, dtype=value.dtype) * param
'''
