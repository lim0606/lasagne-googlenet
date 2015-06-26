import numpy as np
import theano.tensor as T

from lasagne import init # from .. import init
from lasagne import nonlinearities # from .. import nonlinearities

from lasagne.layers.base import Layer # from .base import Layer


__all__ = [
    "BNLayer",
]


class BNLayer(Layer):
    """
    lasagne.layers.BNLayer(incoming, nonlinearity=lasagne.nonlinearities.rectify, **kwargs)

    A batch normalization layer.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape

    W : Theano shared variable, numpy array or callable
        An initializer for the weights of the layer. If a shared variable or a
        numpy array is provided the shape should  be (num_inputs, num_units).
        See :meth:`Layer.create_param` for more information.

    b : Theano shared variable, numpy array, callable or None
        An initializer for the biases of the layer. If a shared variable or a
        numpy array is provided the shape should be (num_units,).
        If None is provided the layer will have no biases.
        See :meth:`Layer.create_param` for more information.

    nonlinearity : callable or None
        The nonlinearity that is applied to the layer activations. If None
        is provided, the layer will be linear.

    Examples
    --------
    >>> from lasagne.layers import InputLayer, DenseLayer
    >>> l_in = InputLayer((100, 20))
    >>> l1 = BNLayer(l_in)

    Notes
    -----
    If the input to this layer has more than two axes, it will flatten the
    trailing axes. This is useful for when a dense layer follows a
    convolutional layer, for example. It is not necessary to insert a
    :class:`FlattenLayer` in this case.
    """
    def __init__(self, incoming, gamma=1.0, beta=0., nonlinearity=None, epsilon=1e-6,
                 **kwargs):
        super(BNLayer, self).__init__(incoming, **kwargs)
        self.nonlinearity = (nonlinearities.identity if nonlinearity is None
                             else nonlinearity)

        # get output shape of incoming
        #self.n_channels = self.input_shape[1]
        #print self.input_shape
        #raise NameError("Hi")
 
        self.epsilon = epsilon

        if len(self.input_shape) is 4:
            self.gamma = self.add_param(init.Constant(gamma), (self.input_shape[1],), name='gamma', regularizable=False).dimshuffle(('x',0,'x','x'))
            self.beta = self.add_param(init.Constant(beta), (self.input_shape[1],), name='beta', regularizable=False).dimshuffle(('x',0,'x','x'))
 
        elif len(self.input_shape) is 2:
            self.gamma = self.add_param(init.Constant(gamma), (self.input_shape[1],), name='gamma', regularizable=False).dimshuffle(('x',0))
            self.beta = self.add_param(init.Constant(beta), (self.input_shape[1],), name='beta', regularizable=False).dimshuffle(('x',0)) 

        else: # input should be 4d tensor or 2d matrix
            raise ValueError('input of BNLayer should be 4d tensor or 2d matrix')
 
        # done init 


    def get_output_shape_for(self, input_shape):
        #return (input_shape[0], self.num_units)
        return input_shape

 
    def get_output_for(self, input, **kwargs):
        if input.ndim is 4: # 4d tensor
            self.mean = T.mean(input, axis=[0, 2, 3]).dimshuffle(('x', 0, 'x', 'x'))
            self.var = T.sum(T.sqr(input - self.mean), axis=[0, 2, 3]).dimshuffle(('x', 0, 'x', 'x')) / self.input_shape[0]

        else: # elif input.ndim is 2: # 2d matrix
            self.mean = T.mean(input, axis=0).dimshuffle(('x',0))
            self.var = T.sum(T.sqr(input - self.mean), axis=0).dimshuffle(('x', 0)) / self.input_shape[0]

        activation = (input - self.mean) / T.sqrt(self.var + self.epsilon)
        activation = self.gamma * activation + self.beta
        return self.nonlinearity(activation)



