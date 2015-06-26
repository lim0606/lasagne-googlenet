import theano

def tensor_repeat(tensor_in, size=1, axis=0):
    if axis > 1 or axis < 0: 
        raise ValueError('this function only support for axis 0 or 1')
    if type(size) is not type(1): 
        raise ValueError('this function only support repeating given tensor in single axis. Thus, size should be int. given %s' % type(size))
    
    tensor_out_list = []
    for i in xrange(size):
        tensor_out_list.append(theano.tensor.ones_like(tensor_in))
        tensor_out_list[i] = tensor_out_list[i] * tensor_in

    if axis is 0:
        tensor_out = theano.tensor.concatenate(tensor_out_list, axis=0)

    else: #if axis is 1:
        tensor_out = theano.tensor.concatenate(tensor_out_list, axis=1)
        
    return tensor_out
