l_inception_5b_1x1 = dnn.Conv2DDNNLayer(
    l_inception_5a_output,
    num_filters=384,
    filter_size=(1, 1),
    nonlinearity=lasagne.nonlinearities.identity,
    W=lasagne.init.GlorotUniform(),
    b=None,
    name='inception_5b_1x1'
)
l_inception_5b_1x1_bn = vae.layers.BNLayer(
    l_inception_5b_1x1,
    name='inception_5b_1x1_bn'
)
l_inception_5b_1x1_relu = lasagne.layers.NonlinearityLayer(
    l_inception_5b_1x1_bn,
    #nonlinearity=lasagne.nonlinearities.TruncatedRectify(trunc=0.1),
    nonlinearity=lasagne.nonlinearities.rectify,
    name='inception_5b_1x1_relu'
)

l_inception_5b_3x3_reduce = dnn.Conv2DDNNLayer(
    l_inception_5a_output,
    num_filters=192,
    filter_size=(1, 1),
    nonlinearity=lasagne.nonlinearities.identity,
    W=lasagne.init.GlorotUniform(),
    b=None,
    name='inception_5b_3x3_reduce'
)
l_inception_5b_3x3_reduce_bn = vae.layers.BNLayer(
    l_inception_5b_3x3_reduce,
    name='inception_5b_3x3_reduce_bn'
)
l_inception_5b_3x3_reduce_relu = lasagne.layers.NonlinearityLayer(
    l_inception_5b_3x3_reduce_bn,
    #nonlinearity=lasagne.nonlinearities.TruncatedRectify(trunc=0.1),
    nonlinearity=lasagne.nonlinearities.rectify,
    name='inception_5b_3x3_reduce_relu'
)
l_inception_5b_3x3 = dnn.Conv2DDNNLayer(
    l_inception_5b_3x3_reduce_relu,
    num_filters=384,
    filter_size=(3, 3),
    pad=1,
    nonlinearity=lasagne.nonlinearities.identity,
    W=lasagne.init.GlorotUniform(),
    b=None,
    name='inception_5b_3x3'
)
l_inception_5b_3x3_bn = vae.layers.BNLayer(
    l_inception_5b_3x3,
    name='inception_5b_3x3_bn'
)
l_inception_5b_3x3_relu = lasagne.layers.NonlinearityLayer(
    l_inception_5b_3x3_bn,
    #nonlinearity=lasagne.nonlinearities.TruncatedRectify(trunc=0.1),
    nonlinearity=lasagne.nonlinearities.rectify,
    name='inception_5b_3x3_relu'
)

l_inception_5b_5x5_reduce = dnn.Conv2DDNNLayer(
    l_inception_5a_output,
    num_filters=48,
    filter_size=(1, 1),
    nonlinearity=lasagne.nonlinearities.identity,
    W=lasagne.init.GlorotUniform(),
    b=None,
    name='inception_5b_5x5_reduce'
)
l_inception_5b_5x5_reduce_bn = vae.layers.BNLayer(
    l_inception_5b_5x5_reduce,
    name='inception_5b_5x5_reduce_bn'
)
l_inception_5b_5x5_reduce_relu = lasagne.layers.NonlinearityLayer(
    l_inception_5b_5x5_reduce_bn,
    #nonlinearity=lasagne.nonlinearities.TruncatedRectify(trunc=0.1),
    nonlinearity=lasagne.nonlinearities.rectify,
    name='inception_5b_5x5_reduce_relu'
)
l_inception_5b_5x5 = dnn.Conv2DDNNLayer(
    l_inception_5b_5x5_reduce_relu,
    num_filters=128,
    filter_size=(5, 5),
    pad=2,
    nonlinearity=lasagne.nonlinearities.identity,
    W=lasagne.init.GlorotUniform(),
    b=None,
    name='inception_5b_5x5'
)
l_inception_5b_5x5_bn = vae.layers.BNLayer(
    l_inception_5b_5x5,
    name='inception_5b_5x5_bn'
)
l_inception_5b_5x5_relu = lasagne.layers.NonlinearityLayer(
    l_inception_5b_5x5_bn,
    #nonlinearity=lasagne.nonlinearities.TruncatedRectify(trunc=0.1),
    nonlinearity=lasagne.nonlinearities.rectify,
    name='inception_5b_5x5_relu'
)

l_inception_5b_pool = dnn.MaxPool2DDNNLayer(
    l_inception_5a_output, 
    pool_size=(3, 3), 
    pad=1, 
    stride=(1, 1),
    name='inception_5b_pool'
)
l_inception_5b_pool_proj = dnn.Conv2DDNNLayer(
    l_inception_5b_pool,
    num_filters=128,
    filter_size=(1, 1),
    nonlinearity=lasagne.nonlinearities.identity,
    W=lasagne.init.GlorotUniform(),
    b=None,
    name='inception_5b_pool_proj'
)
l_inception_5b_pool_proj_bn = vae.layers.BNLayer(
    l_inception_5b_pool_proj,
    name='inception_5b_pool_proj_bn'
)
l_inception_5b_pool_proj_relu = lasagne.layers.NonlinearityLayer(
    l_inception_5b_pool_proj_bn,
    #nonlinearity=lasagne.nonlinearities.TruncatedRectify(trunc=0.1),
    nonlinearity=lasagne.nonlinearities.rectify,
    name='inception_5b_pool_proj_relu'
)

l_inception_5b_output = lasagne.layers.ConcatLayer(incomings=[
    l_inception_5b_1x1_relu,
    l_inception_5b_3x3_relu,
    l_inception_5b_5x5_relu,
    l_inception_5b_pool_proj_relu],
    axis=1,
    name='inception_5b_output'
) # batchsize x n_channels x height x width
