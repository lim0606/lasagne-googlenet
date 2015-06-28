from __future__ import division, absolute_import
from __future__ import unicode_literals

import theano
import theano.tensor as T

import numpy as np
import lasagne
from lasagne.layers import dnn
from lasagne.objectives import categorical_crossentropy, aggregate

# custom layers and functions
import googlenet
#from utils.tensor_repeat import tensor_repeat
from utils.updates import adagrad_w_prior

# pickle
try:
    import cPickle as pickle
except:
    #import pickle
    print("hi")

# Step 0: load data ####################################
#import gzip, cPickle

#f = gzip.open('mnist.pkl.gz', 'rb')
#(X_train, y_train), (X_valid, y_valid), (X_test, y_test) = cPickle.load(f)
#f.close()
#print "X_train.shape: ", X_train.shape
#print "X_train[0,:]", X_train[0,:]
#print "y_train.shape: ", y_train.shape
X_train = np.random.rand(128, 3, 224, 224).astype(theano.config.floatX)
y_train = np.random.randint(0, 1000, 128).reshape((128,))
X_valid = np.random.rand(128, 3, 224, 224).astype(theano.config.floatX)
y_valid = np.random.randint(0, 1000, 128).reshape((128,))

 
# Step 1: initialization ##############################
num_data = X_train.shape[0]
batchsize = 32
update_rules = 'momentum' # you can choose either 1) momentum, 2) adagrad, and 3) adagrad_w_prior. 
num_epochs = 70

img_height = 224
img_width = 224
n_channels = 3

# Step 2: build model -> equals to build model #########
# architecture as follows; 
# 
# - image preprocessing (mean subtraction)
#  
# - conv1  kernel size: 7, pad: 3, stride: 2,
#          num_output: 64
#          xavier 
#          no bias
# - bn for conv1
# - relu for conv1
# - pooling for conv1  
#          type: max, kernel_size: 3, stride: 2,
#                
# - conv2_reduce  
#          kernel size: 1
#          num_output: 64
# - bn for conv2_reduce
# - relu for conv2_reduce
# - conv2  kernel size: 3, pad: 1
#          num_output: 192
#          xavier
#          no bias
# - bn for conv2
# - relu for conv2
# - pooling for conv2  
#          type: max, kernel_size: 3, stride: 2,
#
# - inception_3a/1x1 (conv)
#          kernel_size: 1
#          num_output: 96
# - bn for inception_3a/1x1
# - relu for inception_3a/1x1
#
# - inception_3a/3x3_reduce (conv)
#          kernel_size: 1
#          num_output: 96
# - bn for inception_3a/3x3_reduce
# - relu for inception_3a/3x3_reduce
# - inception_3a/3x3 (conv)
#          kernel_size: 3, pad: 1
#          num_output: 128
#          xavier
#          no bias
# - bn for inception_3a/3x3
# - relu for inception_3a/3x3
#  
# - inception_3a/5x5_reduce (conv)
#          kernel_size: 1
#          num_output: 16
#          xavier
#          no bias
# - bn for inception_3a/5x5_reduce
# - relu for inception_3a/5x5_reduce
# - inception_3a/5x5 (conv)
#          kernel_size: 5, pad 2
#          num_output: 32
#          xavier
#          no bias
# - bn for inception_3a/5x5
# - relu for inception_3a/5x5
# 
# - inception_3a/pool (pool)
#          type: max, kernel_size: 3, pad: 1, stride: 1 
# - inception_3a/pool_proj (conv)
#          kernel_size: 1
#          num_output: 32
# - bn for inception/pool_proj
# - relu for inception/pool_proj
# 
# - concat inception_3a/1x1/bn
#          inception_3a/3x3/bn
#          inception_3a/5x5/bn
#          inception_3a/pool_proj/bn
#      
#

l_in = lasagne.layers.InputLayer(
    shape=(batchsize, n_channels, img_height, img_width),
    name='input',
)

labels = T.ivector('label')

l_conv1 = dnn.Conv2DDNNLayer(
    l_in,
    num_filters=64,
    filter_size=(7, 7),
    pad=3,
    stride=(2, 2),
    nonlinearity=lasagne.nonlinearities.identity,
    W=lasagne.init.GlorotUniform(),
    b=None,
    name='conv1',
)
l_conv1_bn = googlenet.layers.BNLayer(
    l_conv1,
    name='conv1_bn'
)
l_conv1_relu = lasagne.layers.NonlinearityLayer(
    l_conv1_bn, 
    #nonlinearity=lasagne.nonlinearities.TruncatedRectify(trunc=0.1),
    nonlinearity=lasagne.nonlinearities.rectify,
    name='conv1_relu'
)
l_pool1 = dnn.MaxPool2DDNNLayer(
    l_conv1_relu, 
    pool_size=(3, 3), 
    pad=1,
    stride=(2, 2),
    name='pool1'
)

l_conv2_reduce = dnn.Conv2DDNNLayer(
    l_pool1,
    num_filters=64,
    filter_size=(1, 1),
    nonlinearity=lasagne.nonlinearities.identity,
    W=lasagne.init.GlorotUniform(),
    b=None,
    name='conv2_reduce'
)
l_conv2_reduce_bn = googlenet.layers.BNLayer(
    l_conv2_reduce,
    name='conv2_reduce_bn'
)
l_conv2_reduce_relu = lasagne.layers.NonlinearityLayer(
    l_conv2_reduce_bn,
    #nonlinearity=lasagne.nonlinearities.TruncatedRectify(trunc=0.1),
    nonlinearity=lasagne.nonlinearities.rectify,
    name='conv2_reduce_relu'
)
l_conv2 = dnn.Conv2DDNNLayer(
    l_conv2_reduce_relu,
    num_filters=192,
    filter_size=(3, 3),
    pad=1,
    nonlinearity=lasagne.nonlinearities.identity,
    W=lasagne.init.GlorotUniform(),
    b=None,
    name='conv2'
)
l_conv2_bn = googlenet.layers.BNLayer(
    l_conv2,
    name='conv2_bn'
)
l_conv2_relu = lasagne.layers.NonlinearityLayer(
    l_conv2_bn,
    #nonlinearity=lasagne.nonlinearities.TruncatedRectify(trunc=0.1),
    nonlinearity=lasagne.nonlinearities.rectify,
    name='conv2_relu'
)
l_pool2 = dnn.MaxPool2DDNNLayer(
    l_conv2_relu, 
    pool_size=(3, 3), 
    pad=1,
    stride=(2, 2),
    name='pool2'
)

l_inception_3a_1x1 = dnn.Conv2DDNNLayer(
    l_pool2,
    num_filters=64,
    filter_size=(1, 1),
    nonlinearity=lasagne.nonlinearities.identity,
    W=lasagne.init.GlorotUniform(),
    b=None,
    name='inception_3a_1x1'
)
l_inception_3a_1x1_bn = googlenet.layers.BNLayer(
    l_inception_3a_1x1,
    name='inception_3a_1x1_bn'
)
l_inception_3a_1x1_relu = lasagne.layers.NonlinearityLayer(
    l_inception_3a_1x1_bn,
    #nonlinearity=lasagne.nonlinearities.TruncatedRectify(trunc=0.1),
    nonlinearity=lasagne.nonlinearities.rectify,
    name='inception_3a_1x1_relu'
)

l_inception_3a_3x3_reduce = dnn.Conv2DDNNLayer(
    l_pool2,
    num_filters=96,
    filter_size=(1, 1),
    nonlinearity=lasagne.nonlinearities.identity,
    W=lasagne.init.GlorotUniform(),
    b=None,
    name='inception_3a_3x3_reduce'
)
l_inception_3a_3x3_reduce_bn = googlenet.layers.BNLayer(
    l_inception_3a_3x3_reduce,
    name='inception_3a_3x3_reduce_bn'
)
l_inception_3a_3x3_reduce_relu = lasagne.layers.NonlinearityLayer(
    l_inception_3a_3x3_reduce_bn,
    #nonlinearity=lasagne.nonlinearities.TruncatedRectify(trunc=0.1),
    nonlinearity=lasagne.nonlinearities.rectify,
    name='inception_3a_3x3_reduce_relu'
)
l_inception_3a_3x3 = dnn.Conv2DDNNLayer(
    l_inception_3a_3x3_reduce_relu,
    num_filters=128,
    filter_size=(3, 3),
    pad=1,
    nonlinearity=lasagne.nonlinearities.identity,
    W=lasagne.init.GlorotUniform(),
    b=None,
    name='inception_3a_3x3'
)
l_inception_3a_3x3_bn = googlenet.layers.BNLayer(
    l_inception_3a_3x3,
    name='inception_3a_3x3_bn'
)
l_inception_3a_3x3_relu = lasagne.layers.NonlinearityLayer(
    l_inception_3a_3x3_bn,
    #nonlinearity=lasagne.nonlinearities.TruncatedRectify(trunc=0.1),
    nonlinearity=lasagne.nonlinearities.rectify,
    name='inception_3a_3x3_relu'
)

l_inception_3a_5x5_reduce = dnn.Conv2DDNNLayer(
    l_pool2,
    num_filters=16,
    filter_size=(1, 1),
    nonlinearity=lasagne.nonlinearities.identity,
    W=lasagne.init.GlorotUniform(),
    b=None,
    name='inception_3a_5x5_reduce'
)
l_inception_3a_5x5_reduce_bn = googlenet.layers.BNLayer(
    l_inception_3a_5x5_reduce,
    name='inception_3a_5x5_reduce_bn'
)
l_inception_3a_5x5_reduce_relu = lasagne.layers.NonlinearityLayer(
    l_inception_3a_5x5_reduce_bn,
    #nonlinearity=lasagne.nonlinearities.TruncatedRectify(trunc=0.1),
    nonlinearity=lasagne.nonlinearities.rectify,
    name='inception_3a_5x5_reduce_relu'
)
l_inception_3a_5x5 = dnn.Conv2DDNNLayer(
    l_inception_3a_5x5_reduce_relu,
    num_filters=32,
    filter_size=(5, 5),
    pad=2,
    nonlinearity=lasagne.nonlinearities.identity,
    W=lasagne.init.GlorotUniform(),
    b=None,
    name='inception_3a_5x5'
)
l_inception_3a_5x5_bn = googlenet.layers.BNLayer(
    l_inception_3a_5x5,
    name='inception_3a_5x5_bn'
)
l_inception_3a_5x5_relu = lasagne.layers.NonlinearityLayer(
    l_inception_3a_5x5_bn,
    #nonlinearity=lasagne.nonlinearities.TruncatedRectify(trunc=0.1),
    nonlinearity=lasagne.nonlinearities.rectify,
    name='inception_3a_5x5_relu'
)

l_inception_3a_pool = dnn.MaxPool2DDNNLayer(
    l_pool2, 
    pool_size=(3, 3), 
    pad=1, 
    stride=(1, 1),
    name='inception_3a_pool'
)
l_inception_3a_pool_proj = dnn.Conv2DDNNLayer(
    l_inception_3a_pool,
    num_filters=32,
    filter_size=(1, 1),
    nonlinearity=lasagne.nonlinearities.identity,
    W=lasagne.init.GlorotUniform(),
    b=None,
    name='inception_3a_pool_proj'
)
l_inception_3a_pool_proj_bn = googlenet.layers.BNLayer(
    l_inception_3a_pool_proj,
    name='inception_3a_pool_proj_bn'
)
l_inception_3a_pool_proj_relu = lasagne.layers.NonlinearityLayer(
    l_inception_3a_pool_proj_bn,
    #nonlinearity=lasagne.nonlinearities.TruncatedRectify(trunc=0.1),
    nonlinearity=lasagne.nonlinearities.rectify,
    name='inception_3a_pool_proj_relu'
)

l_inception_3a_output = lasagne.layers.ConcatLayer(incomings=[
    l_inception_3a_1x1_relu,
    l_inception_3a_3x3_relu,
    l_inception_3a_5x5_relu,
    l_inception_3a_pool_proj_relu],
    axis=1,
    name='inception_3a_output'
) # batchsize x n_channels x height x width


l_inception_3b_1x1 = dnn.Conv2DDNNLayer(
    l_inception_3a_output,
    num_filters=128,
    filter_size=(1, 1),
    nonlinearity=lasagne.nonlinearities.identity,
    W=lasagne.init.GlorotUniform(),
    b=None,
    name='inception_3b_1x1'
)
l_inception_3b_1x1_bn = googlenet.layers.BNLayer(
    l_inception_3b_1x1,
    name='inception_3b_1x1_bn'
)
l_inception_3b_1x1_relu = lasagne.layers.NonlinearityLayer(
    l_inception_3b_1x1_bn,
    #nonlinearity=lasagne.nonlinearities.TruncatedRectify(trunc=0.1),
    nonlinearity=lasagne.nonlinearities.rectify,
    name='inception_3b_1x1_relu'
)

l_inception_3b_3x3_reduce = dnn.Conv2DDNNLayer(
    l_inception_3a_output,
    num_filters=128,
    filter_size=(1, 1),
    nonlinearity=lasagne.nonlinearities.identity,
    W=lasagne.init.GlorotUniform(),
    b=None,
    name='inception_3b_3x3_reduce'
)
l_inception_3b_3x3_reduce_bn = googlenet.layers.BNLayer(
    l_inception_3b_3x3_reduce,
    name='inception_3b_3x3_reduce_bn'
)
l_inception_3b_3x3_reduce_relu = lasagne.layers.NonlinearityLayer(
    l_inception_3b_3x3_reduce_bn,
    #nonlinearity=lasagne.nonlinearities.TruncatedRectify(trunc=0.1),
    nonlinearity=lasagne.nonlinearities.rectify,
    name='inception_3b_3x3_reduce_relu'
)
l_inception_3b_3x3 = dnn.Conv2DDNNLayer(
    l_inception_3b_3x3_reduce_relu,
    num_filters=192,
    filter_size=(3, 3),
    pad=1,
    nonlinearity=lasagne.nonlinearities.identity,
    W=lasagne.init.GlorotUniform(),
    b=None,
    name='inception_3b_3x3'
)
l_inception_3b_3x3_bn = googlenet.layers.BNLayer(
    l_inception_3b_3x3,
    name='inception_3b_3x3_bn'
)
l_inception_3b_3x3_relu = lasagne.layers.NonlinearityLayer(
    l_inception_3b_3x3_bn,
    #nonlinearity=lasagne.nonlinearities.TruncatedRectify(trunc=0.1),
    nonlinearity=lasagne.nonlinearities.rectify,
    name='inception_3b_3x3_relu'
)

l_inception_3b_5x5_reduce = dnn.Conv2DDNNLayer(
    l_inception_3a_output,
    num_filters=32,
    filter_size=(1, 1),
    nonlinearity=lasagne.nonlinearities.identity,
    W=lasagne.init.GlorotUniform(),
    b=None,
    name='inception_3b_5x5_reduce'
)
l_inception_3b_5x5_reduce_bn = googlenet.layers.BNLayer(
    l_inception_3b_5x5_reduce,
    name='inception_3b_5x5_reduce_bn'
)
l_inception_3b_5x5_reduce_relu = lasagne.layers.NonlinearityLayer(
    l_inception_3b_5x5_reduce_bn,
    #nonlinearity=lasagne.nonlinearities.TruncatedRectify(trunc=0.1),
    nonlinearity=lasagne.nonlinearities.rectify,
    name='inception_3b_5x5_reduce_relu'
)
l_inception_3b_5x5 = dnn.Conv2DDNNLayer(
    l_inception_3b_5x5_reduce_relu,
    num_filters=96,
    filter_size=(5, 5),
    pad=2,
    nonlinearity=lasagne.nonlinearities.identity,
    W=lasagne.init.GlorotUniform(),
    b=None,
    name='inception_3b_5x5'
)
l_inception_3b_5x5_bn = googlenet.layers.BNLayer(
    l_inception_3b_5x5,
    name='inception_3b_5x5_bn'
)
l_inception_3b_5x5_relu = lasagne.layers.NonlinearityLayer(
    l_inception_3b_5x5_bn,
    #nonlinearity=lasagne.nonlinearities.TruncatedRectify(trunc=0.1),
    nonlinearity=lasagne.nonlinearities.rectify,
    name='inception_3b_5x5_relu'
)

l_inception_3b_pool = dnn.MaxPool2DDNNLayer(
    l_inception_3a_output,
    pool_size=(3, 3),
    pad=1,
    stride=(1, 1),
    name='inception_3b_pool'
)
l_inception_3b_pool_proj = dnn.Conv2DDNNLayer(
    l_inception_3b_pool,
    num_filters=64,
    filter_size=(1, 1),
    nonlinearity=lasagne.nonlinearities.identity,
    W=lasagne.init.GlorotUniform(),
    b=None,
    name='inception_3b_pool_proj'
)
l_inception_3b_pool_proj_bn = googlenet.layers.BNLayer(
    l_inception_3b_pool_proj,
    name='inception_3b_pool_proj_bn'
)
l_inception_3b_pool_proj_relu = lasagne.layers.NonlinearityLayer(
    l_inception_3b_pool_proj_bn,
    #nonlinearity=lasagne.nonlinearities.TruncatedRectify(trunc=0.1),
    nonlinearity=lasagne.nonlinearities.rectify,
    name='inception_3b_pool_proj_relu'
)

l_inception_3b_output = lasagne.layers.ConcatLayer(incomings=[
    l_inception_3b_1x1_relu,
    l_inception_3b_3x3_relu,
    l_inception_3b_5x5_relu,
    l_inception_3b_pool_proj_relu],
    axis=1,
    name='inception_3b_output'
) # batchsize x n_channels x height x width

l_pool3 = dnn.MaxPool2DDNNLayer(
    l_inception_3b_output,
    pool_size=(3, 3),
    pad=1,
    stride=(2, 2),
    name='pool3'
)

l_inception_4a_1x1 = dnn.Conv2DDNNLayer(
    l_pool3,
    num_filters=192,
    filter_size=(1, 1),
    nonlinearity=lasagne.nonlinearities.identity,
    W=lasagne.init.GlorotUniform(),
    b=None,
    name='inception_4a_1x1'
)
l_inception_4a_1x1_bn = googlenet.layers.BNLayer(
    l_inception_4a_1x1,
    name='inception_4a_1x1_bn'
)
l_inception_4a_1x1_relu = lasagne.layers.NonlinearityLayer(
    l_inception_4a_1x1_bn,
    #nonlinearity=lasagne.nonlinearities.TruncatedRectify(trunc=0.1),
    nonlinearity=lasagne.nonlinearities.rectify,
    name='inception_4a_1x1_relu'
)

l_inception_4a_3x3_reduce = dnn.Conv2DDNNLayer(
    l_pool3,
    num_filters=96,
    filter_size=(1, 1),
    nonlinearity=lasagne.nonlinearities.identity,
    W=lasagne.init.GlorotUniform(),
    b=None,
    name='inception_4a_3x3_reduce'
)
l_inception_4a_3x3_reduce_bn = googlenet.layers.BNLayer(
    l_inception_4a_3x3_reduce,
    name='inception_4a_3x3_reduce_bn'
)
l_inception_4a_3x3_reduce_relu = lasagne.layers.NonlinearityLayer(
    l_inception_4a_3x3_reduce_bn,
    #nonlinearity=lasagne.nonlinearities.TruncatedRectify(trunc=0.1),
    nonlinearity=lasagne.nonlinearities.rectify,
    name='inception_4a_3x3_reduce_relu'
)
l_inception_4a_3x3 = dnn.Conv2DDNNLayer(
    l_inception_4a_3x3_reduce_relu,
    num_filters=208,
    filter_size=(3, 3),
    pad=1,
    nonlinearity=lasagne.nonlinearities.identity,
    W=lasagne.init.GlorotUniform(),
    b=None,
    name='inception_4a_3x3'
)
l_inception_4a_3x3_bn = googlenet.layers.BNLayer(
    l_inception_4a_3x3,
    name='inception_4a_3x3_bn'
)
l_inception_4a_3x3_relu = lasagne.layers.NonlinearityLayer(
    l_inception_4a_3x3_bn,
    #nonlinearity=lasagne.nonlinearities.TruncatedRectify(trunc=0.1),
    nonlinearity=lasagne.nonlinearities.rectify,
    name='inception_4a_3x3_relu'
)

l_inception_4a_5x5_reduce = dnn.Conv2DDNNLayer(
    l_pool3,
    num_filters=16,
    filter_size=(1, 1),
    nonlinearity=lasagne.nonlinearities.identity,
    W=lasagne.init.GlorotUniform(),
    b=None,
    name='inception_4a_5x5_reduce'
)
l_inception_4a_5x5_reduce_bn = googlenet.layers.BNLayer(
    l_inception_4a_5x5_reduce,
    name='inception_4a_5x5_reduce_bn'
)
l_inception_4a_5x5_reduce_relu = lasagne.layers.NonlinearityLayer(
    l_inception_4a_5x5_reduce_bn,
    #nonlinearity=lasagne.nonlinearities.TruncatedRectify(trunc=0.1),
    nonlinearity=lasagne.nonlinearities.rectify,
    name='inception_4a_5x5_reduce_relu'
)
l_inception_4a_5x5 = dnn.Conv2DDNNLayer(
    l_inception_4a_5x5_reduce_relu,
    num_filters=48,
    filter_size=(5, 5),
    pad=2,
    nonlinearity=lasagne.nonlinearities.identity,
    W=lasagne.init.GlorotUniform(),
    b=None,
    name='inception_4a_5x5'
)
l_inception_4a_5x5_bn = googlenet.layers.BNLayer(
    l_inception_4a_5x5,
    name='inception_4a_5x5_bn'
)
l_inception_4a_5x5_relu = lasagne.layers.NonlinearityLayer(
    l_inception_4a_5x5_bn,
    #nonlinearity=lasagne.nonlinearities.TruncatedRectify(trunc=0.1),
    nonlinearity=lasagne.nonlinearities.rectify,
    name='inception_4a_5x5_relu'
)

l_inception_4a_pool = dnn.MaxPool2DDNNLayer(
    l_pool3,
    pool_size=(3, 3),
    pad=1,
    stride=(1, 1),
    name='inception_4a_pool'
)
l_inception_4a_pool_proj = dnn.Conv2DDNNLayer(
    l_inception_4a_pool,
    num_filters=64,
    filter_size=(1, 1),
    nonlinearity=lasagne.nonlinearities.identity,
    W=lasagne.init.GlorotUniform(),
    b=None,
    name='inception_4a_pool_proj'
)
l_inception_4a_pool_proj_bn = googlenet.layers.BNLayer(
    l_inception_4a_pool_proj,
    name='inception_4a_pool_proj_bn'
)
l_inception_4a_pool_proj_relu = lasagne.layers.NonlinearityLayer(
    l_inception_4a_pool_proj_bn,
    #nonlinearity=lasagne.nonlinearities.TruncatedRectify(trunc=0.1),
    nonlinearity=lasagne.nonlinearities.rectify,
    name='inception_4a_pool_proj_relu'
)

l_inception_4a_output = lasagne.layers.ConcatLayer(incomings=[
    l_inception_4a_1x1_relu,
    l_inception_4a_3x3_relu,
    l_inception_4a_5x5_relu,
    l_inception_4a_pool_proj_relu],
    axis=1,
    name='inception_4a_output'
) # batchsize x n_channels x height x width

l_loss1_ave_pool = dnn.Pool2DDNNLayer(
    incoming = l_inception_4a_output,
    pool_size=(5, 5),
    stride=(3, 3),
    mode='average_inc_pad',
    name='loss1_ave_pool',
)
l_loss1_conv = dnn.Conv2DDNNLayer(
    l_loss1_ave_pool,
    num_filters=128,
    filter_size=(1, 1),
    nonlinearity=lasagne.nonlinearities.identity,
    W=lasagne.init.GlorotUniform(),
    b=None,
    name='loss1_conv'
)
l_loss1_conv_bn = googlenet.layers.BNLayer(
    l_loss1_conv,
    name='loss1_conv_bn'
)
l_loss1_conv_relu = lasagne.layers.NonlinearityLayer(
    l_loss1_conv_bn,
    #nonlinearity=lasagne.nonlinearities.TruncatedRectify(trunc=0.1),
    nonlinearity=lasagne.nonlinearities.rectify,
    name='loss1_conv_relu'
)
l_loss1_fc = lasagne.layers.DenseLayer(
    l_loss1_conv_relu,
    num_units=1024,
    nonlinearity=lasagne.nonlinearities.identity,
    W=lasagne.init.GlorotUniform(),
    b=None,
    name='loss1_fc'
)
l_loss1_fc_bn = googlenet.layers.BNLayer(
    l_loss1_fc,
    name='loss1_fc_bn'
)
l_loss1_fc_relu = lasagne.layers.NonlinearityLayer(
    l_loss1_fc_bn,
    #nonlinearity=lasagne.nonlinearities.TruncatedRectify(trunc=0.1),
    nonlinearity=lasagne.nonlinearities.rectify,
    name='loss1_fc_relu'
)
l_loss1_classifier = lasagne.layers.DenseLayer(
    l_loss1_fc_relu,
    num_units=1000,
    nonlinearity=lasagne.nonlinearities.softmax,
    W=lasagne.init.GlorotUniform(),
    #b=None,
    name='loss1_classifier'
) 

loss1_probs = lasagne.layers.get_output(l_loss1_classifier, lasagne.layers.get_output(l_in))
loss1 = categorical_crossentropy(loss1_probs, labels)
loss1 = aggregate(loss1, mode='mean')

loss1_preds_top1 = theano.tensor.argmax(loss1_probs, axis=1)
loss1_acc_top1 = theano.tensor.mean(
    theano.tensor.eq(loss1_preds_top1, labels), 
    dtype=theano.config.floatX)

loss1_preds_top5 = theano.tensor.argsort(loss1_probs, axis=1)[:,:5]
'''loss1_acc_top5 = theano.tensor.mean(
    theano.tensor.eq(loss1_preds_top5[:,0], labels) + 
    theano.tensor.eq(loss1_preds_top5[:,1], labels) +
    theano.tensor.eq(loss1_preds_top5[:,2], labels) +
    theano.tensor.eq(loss1_preds_top5[:,3], labels) +
    theano.tensor.eq(loss1_preds_top5[:,4], labels),
    dtype=theano.config.floatX)
'''
l_inception_4b_1x1 = dnn.Conv2DDNNLayer(
    l_inception_4a_output,
    num_filters=160,
    filter_size=(1, 1),
    nonlinearity=lasagne.nonlinearities.identity,
    W=lasagne.init.GlorotUniform(),
    b=None,
    name='inception_4b_1x1'
)
l_inception_4b_1x1_bn = googlenet.layers.BNLayer(
    l_inception_4b_1x1,
    name='inception_4b_1x1_bn'
)
l_inception_4b_1x1_relu = lasagne.layers.NonlinearityLayer(
    l_inception_4b_1x1_bn,
    #nonlinearity=lasagne.nonlinearities.TruncatedRectify(trunc=0.1),
    nonlinearity=lasagne.nonlinearities.rectify,
    name='inception_4b_1x1_relu'
)

l_inception_4b_3x3_reduce = dnn.Conv2DDNNLayer(
    l_inception_4a_output,
    num_filters=112,
    filter_size=(1, 1),
    nonlinearity=lasagne.nonlinearities.identity,
    W=lasagne.init.GlorotUniform(),
    b=None,
    name='inception_4b_3x3_reduce'
)
l_inception_4b_3x3_reduce_bn = googlenet.layers.BNLayer(
    l_inception_4b_3x3_reduce,
    name='inception_4b_3x3_reduce_bn'
)
l_inception_4b_3x3_reduce_relu = lasagne.layers.NonlinearityLayer(
    l_inception_4b_3x3_reduce_bn,
    #nonlinearity=lasagne.nonlinearities.TruncatedRectify(trunc=0.1),
    nonlinearity=lasagne.nonlinearities.rectify,
    name='inception_4b_3x3_reduce_relu'
)
l_inception_4b_3x3 = dnn.Conv2DDNNLayer(
    l_inception_4b_3x3_reduce_relu,
    num_filters=224,
    filter_size=(3, 3),
    pad=1,
    nonlinearity=lasagne.nonlinearities.identity,
    W=lasagne.init.GlorotUniform(),
    b=None,
    name='inception_4b_3x3'
)
l_inception_4b_3x3_bn = googlenet.layers.BNLayer(
    l_inception_4b_3x3,
    name='inception_4b_3x3_bn'
)
l_inception_4b_3x3_relu = lasagne.layers.NonlinearityLayer(
    l_inception_4b_3x3_bn,
    #nonlinearity=lasagne.nonlinearities.TruncatedRectify(trunc=0.1),
    nonlinearity=lasagne.nonlinearities.rectify,
    name='inception_4b_3x3_relu'
)

l_inception_4b_5x5_reduce = dnn.Conv2DDNNLayer(
    l_inception_4a_output,
    num_filters=24,
    filter_size=(1, 1),
    nonlinearity=lasagne.nonlinearities.identity,
    W=lasagne.init.GlorotUniform(),
    b=None,
    name='inception_4b_5x5_reduce'
)
l_inception_4b_5x5_reduce_bn = googlenet.layers.BNLayer(
    l_inception_4b_5x5_reduce,
    name='inception_4b_5x5_reduce_bn'
)
l_inception_4b_5x5_reduce_relu = lasagne.layers.NonlinearityLayer(
    l_inception_4b_5x5_reduce_bn,
    #nonlinearity=lasagne.nonlinearities.TruncatedRectify(trunc=0.1),
    nonlinearity=lasagne.nonlinearities.rectify,
    name='inception_4b_5x5_reduce_relu'
)
l_inception_4b_5x5 = dnn.Conv2DDNNLayer(
    l_inception_4b_5x5_reduce_relu,
    num_filters=64,
    filter_size=(5, 5),
    pad=2,
    nonlinearity=lasagne.nonlinearities.identity,
    W=lasagne.init.GlorotUniform(),
    b=None,
    name='inception_4b_5x5'
)
l_inception_4b_5x5_bn = googlenet.layers.BNLayer(
    l_inception_4b_5x5,
    name='inception_4b_5x5_bn'
)
l_inception_4b_5x5_relu = lasagne.layers.NonlinearityLayer(
    l_inception_4b_5x5_bn,
    #nonlinearity=lasagne.nonlinearities.TruncatedRectify(trunc=0.1),
    nonlinearity=lasagne.nonlinearities.rectify,
    name='inception_4b_5x5_relu'
)

l_inception_4b_pool = dnn.MaxPool2DDNNLayer(
    l_inception_4a_output,
    pool_size=(3, 3),
    pad=1,
    stride=(1, 1),
    name='inception_4b_pool'
)
l_inception_4b_pool_proj = dnn.Conv2DDNNLayer(
    l_inception_4b_pool,
    num_filters=64,
    filter_size=(1, 1),
    nonlinearity=lasagne.nonlinearities.identity,
    W=lasagne.init.GlorotUniform(),
    b=None,
    name='inception_4b_pool_proj'
)
l_inception_4b_pool_proj_bn = googlenet.layers.BNLayer(
    l_inception_4b_pool_proj,
    name='inception_4b_pool_proj_bn'
)
l_inception_4b_pool_proj_relu = lasagne.layers.NonlinearityLayer(
    l_inception_4b_pool_proj_bn,
    #nonlinearity=lasagne.nonlinearities.TruncatedRectify(trunc=0.1),
    nonlinearity=lasagne.nonlinearities.rectify,
    name='inception_4b_pool_proj_relu'
)

l_inception_4b_output = lasagne.layers.ConcatLayer(incomings=[
    l_inception_4b_1x1_relu,
    l_inception_4b_3x3_relu,
    l_inception_4b_5x5_relu,
    l_inception_4b_pool_proj_relu],
    axis=1,
    name='inception_4b_output'
) # batchsize x n_channels x height x width

l_inception_4c_1x1 = dnn.Conv2DDNNLayer(
    l_inception_4b_output,
    num_filters=128,
    filter_size=(1, 1),
    nonlinearity=lasagne.nonlinearities.identity,
    W=lasagne.init.GlorotUniform(),
    b=None,
    name='inception_4c_1x1'
)
l_inception_4c_1x1_bn = googlenet.layers.BNLayer(
    l_inception_4c_1x1,
    name='inception_4c_1x1_bn'
)
l_inception_4c_1x1_relu = lasagne.layers.NonlinearityLayer(
    l_inception_4c_1x1_bn,
    #nonlinearity=lasagne.nonlinearities.TruncatedRectify(trunc=0.1),
    nonlinearity=lasagne.nonlinearities.rectify,
    name='inception_4c_1x1_relu'
)

l_inception_4c_3x3_reduce = dnn.Conv2DDNNLayer(
    l_inception_4b_output,
    num_filters=128,
    filter_size=(1, 1),
    nonlinearity=lasagne.nonlinearities.identity,
    W=lasagne.init.GlorotUniform(),
    b=None,
    name='inception_4c_3x3_reduce'
)
l_inception_4c_3x3_reduce_bn = googlenet.layers.BNLayer(
    l_inception_4c_3x3_reduce,
    name='inception_4c_3x3_reduce_bn'
)
l_inception_4c_3x3_reduce_relu = lasagne.layers.NonlinearityLayer(
    l_inception_4c_3x3_reduce_bn,
    #nonlinearity=lasagne.nonlinearities.TruncatedRectify(trunc=0.1),
    nonlinearity=lasagne.nonlinearities.rectify,
    name='inception_4c_3x3_reduce_relu'
)
l_inception_4c_3x3 = dnn.Conv2DDNNLayer(
    l_inception_4c_3x3_reduce_relu,
    num_filters=256,
    filter_size=(3, 3),
    pad=1,
    nonlinearity=lasagne.nonlinearities.identity,
    W=lasagne.init.GlorotUniform(),
    b=None,
    name='inception_4c_3x3'
)
l_inception_4c_3x3_bn = googlenet.layers.BNLayer(
    l_inception_4c_3x3,
    name='inception_4c_3x3_bn'
)
l_inception_4c_3x3_relu = lasagne.layers.NonlinearityLayer(
    l_inception_4c_3x3_bn,
    #nonlinearity=lasagne.nonlinearities.TruncatedRectify(trunc=0.1),
    nonlinearity=lasagne.nonlinearities.rectify,
    name='inception_4c_3x3_relu'
)

l_inception_4c_5x5_reduce = dnn.Conv2DDNNLayer(
    l_inception_4b_output,
    num_filters=24,
    filter_size=(1, 1),
    nonlinearity=lasagne.nonlinearities.identity,
    W=lasagne.init.GlorotUniform(),
    b=None,
    name='inception_4c_5x5_reduce'
)
l_inception_4c_5x5_reduce_bn = googlenet.layers.BNLayer(
    l_inception_4c_5x5_reduce,
    name='inception_4c_5x5_reduce_bn'
)
l_inception_4c_5x5_reduce_relu = lasagne.layers.NonlinearityLayer(
    l_inception_4c_5x5_reduce_bn,
    #nonlinearity=lasagne.nonlinearities.TruncatedRectify(trunc=0.1),
    nonlinearity=lasagne.nonlinearities.rectify,
    name='inception_4c_5x5_reduce_relu'
)
l_inception_4c_5x5 = dnn.Conv2DDNNLayer(
    l_inception_4c_5x5_reduce_relu,
    num_filters=64,
    filter_size=(5, 5),
    pad=2,
    nonlinearity=lasagne.nonlinearities.identity,
    W=lasagne.init.GlorotUniform(),
    b=None,
    name='inception_4c_5x5'
)
l_inception_4c_5x5_bn = googlenet.layers.BNLayer(
    l_inception_4c_5x5,
    name='inception_4c_5x5_bn'
)
l_inception_4c_5x5_relu = lasagne.layers.NonlinearityLayer(
    l_inception_4c_5x5_bn,
    #nonlinearity=lasagne.nonlinearities.TruncatedRectify(trunc=0.1),
    nonlinearity=lasagne.nonlinearities.rectify,
    name='inception_4c_5x5_relu'
)

l_inception_4c_pool = dnn.MaxPool2DDNNLayer(
    l_inception_4b_output,
    pool_size=(3, 3),
    pad=1,
    stride=(1, 1),
    name='inception_4c_pool'
)
l_inception_4c_pool_proj = dnn.Conv2DDNNLayer(
    l_inception_4c_pool,
    num_filters=64,
    filter_size=(1, 1),
    nonlinearity=lasagne.nonlinearities.identity,
    W=lasagne.init.GlorotUniform(),
    b=None,
    name='inception_4c_pool_proj'
)
l_inception_4c_pool_proj_bn = googlenet.layers.BNLayer(
    l_inception_4c_pool_proj,
    name='inception_4c_pool_proj_bn'
)
l_inception_4c_pool_proj_relu = lasagne.layers.NonlinearityLayer(
    l_inception_4c_pool_proj_bn,
    #nonlinearity=lasagne.nonlinearities.TruncatedRectify(trunc=0.1),
    nonlinearity=lasagne.nonlinearities.rectify,
    name='inception_4c_pool_proj_relu'
)

l_inception_4c_output = lasagne.layers.ConcatLayer(incomings=[
    l_inception_4c_1x1_relu,
    l_inception_4c_3x3_relu,
    l_inception_4c_5x5_relu,
    l_inception_4c_pool_proj_relu],
    axis=1,
    name='inception_4c_output'
) # batchsize x n_channels x height x width

l_inception_4d_1x1 = dnn.Conv2DDNNLayer(
    l_inception_4c_output,
    num_filters=128,
    filter_size=(1, 1),
    nonlinearity=lasagne.nonlinearities.identity,
    W=lasagne.init.GlorotUniform(),
    b=None,
    name='inception_4d_1x1'
)
l_inception_4d_1x1_bn = googlenet.layers.BNLayer(
    l_inception_4d_1x1,
    name='inception_4d_1x1_bn'
)
l_inception_4d_1x1_relu = lasagne.layers.NonlinearityLayer(
    l_inception_4d_1x1_bn,
    #nonlinearity=lasagne.nonlinearities.TruncatedRectify(trunc=0.1),
    nonlinearity=lasagne.nonlinearities.rectify,
    name='inception_4d_1x1_relu'
)

l_inception_4d_3x3_reduce = dnn.Conv2DDNNLayer(
    l_inception_4c_output,
    num_filters=128,
    filter_size=(1, 1),
    nonlinearity=lasagne.nonlinearities.identity,
    W=lasagne.init.GlorotUniform(),
    b=None,
    name='inception_4d_3x3_reduce'
)
l_inception_4d_3x3_reduce_bn = googlenet.layers.BNLayer(
    l_inception_4d_3x3_reduce,
    name='inception_4d_3x3_reduce_bn'
)
l_inception_4d_3x3_reduce_relu = lasagne.layers.NonlinearityLayer(
    l_inception_4d_3x3_reduce_bn,
    #nonlinearity=lasagne.nonlinearities.TruncatedRectify(trunc=0.1),
    nonlinearity=lasagne.nonlinearities.rectify,
    name='inception_4d_3x3_reduce_relu'
)
l_inception_4d_3x3 = dnn.Conv2DDNNLayer(
    l_inception_4d_3x3_reduce_relu,
    num_filters=256,
    filter_size=(3, 3),
    pad=1,
    nonlinearity=lasagne.nonlinearities.identity,
    W=lasagne.init.GlorotUniform(),
    b=None,
    name='inception_4d_3x3'
)
l_inception_4d_3x3_bn = googlenet.layers.BNLayer(
    l_inception_4d_3x3,
    name='inception_4d_3x3_bn'
)
l_inception_4d_3x3_relu = lasagne.layers.NonlinearityLayer(
    l_inception_4d_3x3_bn,
    #nonlinearity=lasagne.nonlinearities.TruncatedRectify(trunc=0.1),
    nonlinearity=lasagne.nonlinearities.rectify,
    name='inception_4d_3x3_relu'
)

l_inception_4d_5x5_reduce = dnn.Conv2DDNNLayer(
    l_inception_4c_output,
    num_filters=24,
    filter_size=(1, 1),
    nonlinearity=lasagne.nonlinearities.identity,
    W=lasagne.init.GlorotUniform(),
    b=None,
    name='inception_4d_5x5_reduce'
)
l_inception_4d_5x5_reduce_bn = googlenet.layers.BNLayer(
    l_inception_4d_5x5_reduce,
    name='inception_4d_5x5_reduce_bn'
)
l_inception_4d_5x5_reduce_relu = lasagne.layers.NonlinearityLayer(
    l_inception_4d_5x5_reduce_bn,
    #nonlinearity=lasagne.nonlinearities.TruncatedRectify(trunc=0.1),
    nonlinearity=lasagne.nonlinearities.rectify,
    name='inception_4d_5x5_reduce_relu'
)
l_inception_4d_5x5 = dnn.Conv2DDNNLayer(
    l_inception_4d_5x5_reduce_relu,
    num_filters=64,
    filter_size=(5, 5),
    pad=2,
    nonlinearity=lasagne.nonlinearities.identity,
    W=lasagne.init.GlorotUniform(),
    b=None,
    name='inception_4d_5x5'
)
l_inception_4d_5x5_bn = googlenet.layers.BNLayer(
    l_inception_4d_5x5,
    name='inception_4d_5x5_bn'
)
l_inception_4d_5x5_relu = lasagne.layers.NonlinearityLayer(
    l_inception_4d_5x5_bn,
    #nonlinearity=lasagne.nonlinearities.TruncatedRectify(trunc=0.1),
    nonlinearity=lasagne.nonlinearities.rectify,
    name='inception_4d_5x5_relu'
)

l_inception_4d_pool = dnn.MaxPool2DDNNLayer(
    l_inception_4c_output,
    pool_size=(3, 3),
    pad=1,
    stride=(1, 1),
    name='inception_4d_pool'
)
l_inception_4d_pool_proj = dnn.Conv2DDNNLayer(
    l_inception_4d_pool,
    num_filters=64,
    filter_size=(1, 1),
    nonlinearity=lasagne.nonlinearities.identity,
    W=lasagne.init.GlorotUniform(),
    b=None,
    name='inception_4d_pool_proj'
)
l_inception_4d_pool_proj_bn = googlenet.layers.BNLayer(
    l_inception_4d_pool_proj,
    name='inception_4d_pool_proj_bn'
)
l_inception_4d_pool_proj_relu = lasagne.layers.NonlinearityLayer(
    l_inception_4d_pool_proj_bn,
    #nonlinearity=lasagne.nonlinearities.TruncatedRectify(trunc=0.1),
    nonlinearity=lasagne.nonlinearities.rectify,
    name='inception_4d_pool_proj_relu'
)

l_inception_4d_output = lasagne.layers.ConcatLayer(incomings=[
    l_inception_4d_1x1_relu,
    l_inception_4d_3x3_relu,
    l_inception_4d_5x5_relu,
    l_inception_4d_pool_proj_relu],
    axis=1,
    name='inception_4d_output'
) # batchsize x n_channels x height x width
 
l_loss2_ave_pool = dnn.Pool2DDNNLayer(
    incoming = l_inception_4d_output,
    pool_size=(5, 5),
    stride=(3, 3),
    mode='average_inc_pad',
    name='loss2_ave_pool',
)
l_loss2_conv = dnn.Conv2DDNNLayer(
    l_loss2_ave_pool,
    num_filters=128,
    filter_size=(1, 1),
    nonlinearity=lasagne.nonlinearities.identity,
    W=lasagne.init.GlorotUniform(),
    b=None,
    name='loss2_conv'
)
l_loss2_conv_bn = googlenet.layers.BNLayer(
    l_loss2_conv,
    name='loss2_conv_bn'
)
l_loss2_conv_relu = lasagne.layers.NonlinearityLayer(
    l_loss2_conv_bn,
    #nonlinearity=lasagne.nonlinearities.TruncatedRectify(trunc=0.1),
    nonlinearity=lasagne.nonlinearities.rectify,
    name='loss2_conv_relu'
)
l_loss2_fc = lasagne.layers.DenseLayer(
    l_loss2_conv_relu,
    num_units=1024,
    nonlinearity=lasagne.nonlinearities.identity,
    W=lasagne.init.GlorotUniform(),
    b=None,
    name='loss2_fc'
)
l_loss2_fc_bn = googlenet.layers.BNLayer(
    l_loss2_fc,
    name='loss2_fc_bn'
)
l_loss2_fc_relu = lasagne.layers.NonlinearityLayer(
    l_loss2_fc_bn,
    #nonlinearity=lasagne.nonlinearities.TruncatedRectify(trunc=0.1),
    nonlinearity=lasagne.nonlinearities.rectify,
    name='loss2_fc_relu'
)
l_loss2_classifier = lasagne.layers.DenseLayer(
    l_loss2_fc_relu,
    num_units=1000,
    nonlinearity=lasagne.nonlinearities.softmax,
    W=lasagne.init.GlorotUniform(),
    #b=None,
    name='loss2_classifier'
)

loss2_probs = lasagne.layers.get_output(l_loss2_classifier, lasagne.layers.get_output(l_in))
loss2 = categorical_crossentropy(loss2_probs, labels)
loss2 = aggregate(loss2, mode='mean')

loss2_preds_top1 = theano.tensor.argmax(loss2_probs, axis=1)
loss2_acc_top1 = theano.tensor.mean(
    theano.tensor.eq(loss2_preds_top1, labels),
    dtype=theano.config.floatX)

loss2_preds_top5 = theano.tensor.argsort(loss2_probs, axis=1)[:,:5]
'''loss2_acc_top5 = theano.tensor.mean(
    theano.tensor.eq(loss2_preds_top5[:,0], labels) +
    theano.tensor.eq(loss2_preds_top5[:,1], labels) +
    theano.tensor.eq(loss2_preds_top5[:,2], labels) +
    theano.tensor.eq(loss2_preds_top5[:,3], labels) +
    theano.tensor.eq(loss2_preds_top5[:,4], labels),
    dtype=theano.config.floatX)
'''
l_inception_4e_1x1 = dnn.Conv2DDNNLayer(
    l_inception_4d_output,
    num_filters=256,
    filter_size=(1, 1),
    nonlinearity=lasagne.nonlinearities.identity,
    W=lasagne.init.GlorotUniform(),
    b=None,
    name='inception_4e_1x1'
)
l_inception_4e_1x1_bn = googlenet.layers.BNLayer(
    l_inception_4e_1x1,
    name='inception_4e_1x1_bn'
)
l_inception_4e_1x1_relu = lasagne.layers.NonlinearityLayer(
    l_inception_4e_1x1_bn,
    #nonlinearity=lasagne.nonlinearities.TruncatedRectify(trunc=0.1),
    nonlinearity=lasagne.nonlinearities.rectify,
    name='inception_4e_1x1_relu'
)

l_inception_4e_3x3_reduce = dnn.Conv2DDNNLayer(
    l_inception_4d_output,
    num_filters=160,
    filter_size=(1, 1),
    nonlinearity=lasagne.nonlinearities.identity,
    W=lasagne.init.GlorotUniform(),
    b=None,
    name='inception_4e_3x3_reduce'
)
l_inception_4e_3x3_reduce_bn = googlenet.layers.BNLayer(
    l_inception_4e_3x3_reduce,
    name='inception_4e_3x3_reduce_bn'
)
l_inception_4e_3x3_reduce_relu = lasagne.layers.NonlinearityLayer(
    l_inception_4e_3x3_reduce_bn,
    #nonlinearity=lasagne.nonlinearities.TruncatedRectify(trunc=0.1),
    nonlinearity=lasagne.nonlinearities.rectify,
    name='inception_4e_3x3_reduce_relu'
)
l_inception_4e_3x3 = dnn.Conv2DDNNLayer(
    l_inception_4e_3x3_reduce_relu,
    num_filters=320,
    filter_size=(3, 3),
    pad=1,
    nonlinearity=lasagne.nonlinearities.identity,
    W=lasagne.init.GlorotUniform(),
    b=None,
    name='inception_4e_3x3'
)
l_inception_4e_3x3_bn = googlenet.layers.BNLayer(
    l_inception_4e_3x3,
    name='inception_4e_3x3_bn'
)
l_inception_4e_3x3_relu = lasagne.layers.NonlinearityLayer(
    l_inception_4e_3x3_bn,
    #nonlinearity=lasagne.nonlinearities.TruncatedRectify(trunc=0.1),
    nonlinearity=lasagne.nonlinearities.rectify,
    name='inception_4e_3x3_relu'
)

l_inception_4e_5x5_reduce = dnn.Conv2DDNNLayer(
    l_inception_4d_output,
    num_filters=32,
    filter_size=(1, 1),
    nonlinearity=lasagne.nonlinearities.identity,
    W=lasagne.init.GlorotUniform(),
    b=None,
    name='inception_4e_5x5_reduce'
)
l_inception_4e_5x5_reduce_bn = googlenet.layers.BNLayer(
    l_inception_4e_5x5_reduce,
    name='inception_4e_5x5_reduce_bn'
)
l_inception_4e_5x5_reduce_relu = lasagne.layers.NonlinearityLayer(
    l_inception_4e_5x5_reduce_bn,
    #nonlinearity=lasagne.nonlinearities.TruncatedRectify(trunc=0.1),
    nonlinearity=lasagne.nonlinearities.rectify,
    name='inception_4e_5x5_reduce_relu'
)
l_inception_4e_5x5 = dnn.Conv2DDNNLayer(
    l_inception_4e_5x5_reduce_relu,
    num_filters=128,
    filter_size=(5, 5),
    pad=2,
    nonlinearity=lasagne.nonlinearities.identity,
    W=lasagne.init.GlorotUniform(),
    b=None,
    name='inception_4e_5x5'
)
l_inception_4e_5x5_bn = googlenet.layers.BNLayer(
    l_inception_4e_5x5,
    name='inception_4e_5x5_bn'
)
l_inception_4e_5x5_relu = lasagne.layers.NonlinearityLayer(
    l_inception_4e_5x5_bn,
    #nonlinearity=lasagne.nonlinearities.TruncatedRectify(trunc=0.1),
    nonlinearity=lasagne.nonlinearities.rectify,
    name='inception_4e_5x5_relu'
)

l_inception_4e_pool = dnn.MaxPool2DDNNLayer(
    l_inception_4d_output,
    pool_size=(3, 3),
    pad=1,
    stride=(1, 1),
    name='inception_4e_pool'
)
l_inception_4e_pool_proj = dnn.Conv2DDNNLayer(
    l_inception_4e_pool,
    num_filters=128,
    filter_size=(1, 1),
    nonlinearity=lasagne.nonlinearities.identity,
    W=lasagne.init.GlorotUniform(),
    b=None,
    name='inception_4e_pool_proj'
)
l_inception_4e_pool_proj_bn = googlenet.layers.BNLayer(
    l_inception_4e_pool_proj,
    name='inception_4e_pool_proj_bn'
)
l_inception_4e_pool_proj_relu = lasagne.layers.NonlinearityLayer(
    l_inception_4e_pool_proj_bn,
    #nonlinearity=lasagne.nonlinearities.TruncatedRectify(trunc=0.1),
    nonlinearity=lasagne.nonlinearities.rectify,
    name='inception_4e_pool_proj_relu'
)

l_inception_4e_output = lasagne.layers.ConcatLayer(incomings=[
    l_inception_4e_1x1_relu,
    l_inception_4e_3x3_relu,
    l_inception_4e_5x5_relu,
    l_inception_4e_pool_proj_relu],
    axis=1,
    name='inception_4e_output'
) # batchsize x n_channels x height x width
 
l_pool4 = dnn.MaxPool2DDNNLayer(
    l_inception_4e_output,
    pool_size=(3, 3),
    pad=1,
    stride=(2, 2),
    name='pool4'
)

l_inception_5a_1x1 = dnn.Conv2DDNNLayer(
    l_pool4,
    num_filters=256,
    filter_size=(1, 1),
    nonlinearity=lasagne.nonlinearities.identity,
    W=lasagne.init.GlorotUniform(),
    b=None,
    name='inception_5a_1x1'
)
l_inception_5a_1x1_bn = googlenet.layers.BNLayer(
    l_inception_5a_1x1,
    name='inception_5a_1x1_bn'
)
l_inception_5a_1x1_relu = lasagne.layers.NonlinearityLayer(
    l_inception_5a_1x1_bn,
    #nonlinearity=lasagne.nonlinearities.TruncatedRectify(trunc=0.1),
    nonlinearity=lasagne.nonlinearities.rectify,
    name='inception_5a_1x1_relu'
)

l_inception_5a_3x3_reduce = dnn.Conv2DDNNLayer(
    l_pool4,
    num_filters=160,
    filter_size=(1, 1),
    nonlinearity=lasagne.nonlinearities.identity,
    W=lasagne.init.GlorotUniform(),
    b=None,
    name='inception_5a_3x3_reduce'
)
l_inception_5a_3x3_reduce_bn = googlenet.layers.BNLayer(
    l_inception_5a_3x3_reduce,
    name='inception_5a_3x3_reduce_bn'
)
l_inception_5a_3x3_reduce_relu = lasagne.layers.NonlinearityLayer(
    l_inception_5a_3x3_reduce_bn,
    #nonlinearity=lasagne.nonlinearities.TruncatedRectify(trunc=0.1),
    nonlinearity=lasagne.nonlinearities.rectify,
    name='inception_5a_3x3_reduce_relu'
)
l_inception_5a_3x3 = dnn.Conv2DDNNLayer(
    l_inception_5a_3x3_reduce_relu,
    num_filters=320,
    filter_size=(3, 3),
    pad=1,
    nonlinearity=lasagne.nonlinearities.identity,
    W=lasagne.init.GlorotUniform(),
    b=None,
    name='inception_5a_3x3'
)
l_inception_5a_3x3_bn = googlenet.layers.BNLayer(
    l_inception_5a_3x3,
    name='inception_5a_3x3_bn'
)
l_inception_5a_3x3_relu = lasagne.layers.NonlinearityLayer(
    l_inception_5a_3x3_bn,
    #nonlinearity=lasagne.nonlinearities.TruncatedRectify(trunc=0.1),
    nonlinearity=lasagne.nonlinearities.rectify,
    name='inception_5a_3x3_relu'
)

l_inception_5a_5x5_reduce = dnn.Conv2DDNNLayer(
    l_pool4,
    num_filters=32,
    filter_size=(1, 1),
    nonlinearity=lasagne.nonlinearities.identity,
    W=lasagne.init.GlorotUniform(),
    b=None,
    name='inception_5a_5x5_reduce'
)
l_inception_5a_5x5_reduce_bn = googlenet.layers.BNLayer(
    l_inception_5a_5x5_reduce,
    name='inception_5a_5x5_reduce_bn'
)
l_inception_5a_5x5_reduce_relu = lasagne.layers.NonlinearityLayer(
    l_inception_5a_5x5_reduce_bn,
    #nonlinearity=lasagne.nonlinearities.TruncatedRectify(trunc=0.1),
    nonlinearity=lasagne.nonlinearities.rectify,
    name='inception_5a_5x5_reduce_relu'
)
l_inception_5a_5x5 = dnn.Conv2DDNNLayer(
    l_inception_5a_5x5_reduce_relu,
    num_filters=128,
    filter_size=(5, 5),
    pad=2,
    nonlinearity=lasagne.nonlinearities.identity,
    W=lasagne.init.GlorotUniform(),
    b=None,
    name='inception_5a_5x5'
)
l_inception_5a_5x5_bn = googlenet.layers.BNLayer(
    l_inception_5a_5x5,
    name='inception_5a_5x5_bn'
)
l_inception_5a_5x5_relu = lasagne.layers.NonlinearityLayer(
    l_inception_5a_5x5_bn,
    #nonlinearity=lasagne.nonlinearities.TruncatedRectify(trunc=0.1),
    nonlinearity=lasagne.nonlinearities.rectify,
    name='inception_5a_5x5_relu'
)

l_inception_5a_pool = dnn.MaxPool2DDNNLayer(
    l_pool4,
    pool_size=(3, 3),
    pad=1,
    stride=(1, 1),
    name='inception_5a_pool'
)
l_inception_5a_pool_proj = dnn.Conv2DDNNLayer(
    l_inception_5a_pool,
    num_filters=128,
    filter_size=(1, 1),
    nonlinearity=lasagne.nonlinearities.identity,
    W=lasagne.init.GlorotUniform(),
    b=None,
    name='inception_5a_pool_proj'
)
l_inception_5a_pool_proj_bn = googlenet.layers.BNLayer(
    l_inception_5a_pool_proj,
    name='inception_5a_pool_proj_bn'
)
l_inception_5a_pool_proj_relu = lasagne.layers.NonlinearityLayer(
    l_inception_5a_pool_proj_bn,
    #nonlinearity=lasagne.nonlinearities.TruncatedRectify(trunc=0.1),
    nonlinearity=lasagne.nonlinearities.rectify,
    name='inception_5a_pool_proj_relu'
)

l_inception_5a_output = lasagne.layers.ConcatLayer(incomings=[
    l_inception_5a_1x1_relu,
    l_inception_5a_3x3_relu,
    l_inception_5a_5x5_relu,
    l_inception_5a_pool_proj_relu],
    axis=1,
    name='inception_5a_output'
) # batchsize x n_channels x height x width
 
l_inception_5b_1x1 = dnn.Conv2DDNNLayer(
    l_inception_5a_output,
    num_filters=384,
    filter_size=(1, 1),
    nonlinearity=lasagne.nonlinearities.identity,
    W=lasagne.init.GlorotUniform(),
    b=None,
    name='inception_5b_1x1'
)
l_inception_5b_1x1_bn = googlenet.layers.BNLayer(
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
l_inception_5b_3x3_reduce_bn = googlenet.layers.BNLayer(
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
l_inception_5b_3x3_bn = googlenet.layers.BNLayer(
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
l_inception_5b_5x5_reduce_bn = googlenet.layers.BNLayer(
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
l_inception_5b_5x5_bn = googlenet.layers.BNLayer(
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
l_inception_5b_pool_proj_bn = googlenet.layers.BNLayer(
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
 
l_pool5 = dnn.Pool2DDNNLayer(
    incoming = l_inception_5b_output,
    pool_size=(7, 7),
    stride=(1, 1),
    mode='average_inc_pad',
    name='pool5',
)

l_loss3_classifier = lasagne.layers.DenseLayer(
    l_pool5,
    num_units=1000,
    nonlinearity=lasagne.nonlinearities.softmax,
    W=lasagne.init.GlorotUniform(),
    #b=None,
    name='loss3_classifier'
)

loss3_probs = lasagne.layers.get_output(l_loss3_classifier, lasagne.layers.get_output(l_in))
loss3 = categorical_crossentropy(loss3_probs, labels)
loss3 = aggregate(loss3, mode='mean')

loss3_preds_top1 = theano.tensor.argmax(loss3_probs, axis=1)
loss3_acc_top1 = theano.tensor.mean(
    theano.tensor.eq(loss3_preds_top1, labels),
    dtype=theano.config.floatX)

loss3_preds_top5 = theano.tensor.argsort(loss3_probs, axis=1)[:,:5]
'''loss3_acc_top5 = theano.tensor.mean(
    theano.tensor.eq(loss3_preds_top5[:,0], labels) +
    theano.tensor.eq(loss3_preds_top5[:,1], labels) +
    theano.tensor.eq(loss3_preds_top5[:,2], labels) +
    theano.tensor.eq(loss3_preds_top5[:,3], labels) +
    theano.tensor.eq(loss3_preds_top5[:,4], labels),
    dtype=theano.config.floatX)
'''
loss = loss1 * 0.3 + loss2 * 0.3 + loss3 * 1 

print "Network Architecture ---------------"
all_layers = lasagne.layers.get_all_layers([l_loss1_classifier, l_loss2_classifier, l_loss3_classifier])
for layer in all_layers:
    print layer.name, ": ", lasagne.layers.get_output_shape(layer)

# - calculate gradient
all_params = lasagne.layers.get_all_params([l_loss1_classifier, l_loss2_classifier, l_loss3_classifier])
all_grads = theano.grad(loss, all_params)
print "Parameters -------------------------"
for param in all_params:
    print param, param.eval().shape

# - update rules 
#   you can choose update rules 
#   1) momentum, 2) adagrad, and 3) adagrad_w_prior
# - momentum is not really work for googlenet (exploding easily)
if update_rules == 'momentum':
    updates = lasagne.updates.momentum(
        loss_or_grads=all_grads,
        params=all_params,
        learning_rate=0.01,
        momentum=0.9)
elif update_rules == 'adagrad':
    updates = lasagne.updates.adagrad(
        loss_or_grads=all_grads,
        params=all_params,
        learning_rate=0.01,
    )
elif update_rules == 'adagrad_w_prior':
    updates = adagrad_w_prior(
        loss_or_grads=all_grads,
        params=all_params,
        learning_rate=0.01,
        batchsize=batchsize,
        num_data=num_data,
    )
else:
    raise ValueError('Please specify learning rule')

print "Compiling funtions..."
import time
start_time = time.time()
# - create a function that also updates the weights
# - this function takes in 2 arguments: the input batch of images and a
#   target vector (the y's) and returns a list with a single scalar
#   element (the loss)
train_fn = theano.function(inputs=[l_in.input_var, labels],
                           outputs=[loss],
                           updates=updates,
                           allow_input_downcast=True)

# - create a function that does not update the weights, and doesn't
#   use dropout
# - same interface as previous the previous function, but now the
#   output is a list where the first element is the loss, and the
#   second element is the actual predicted probabilities for the
#   input data
valid_fn = theano.function(inputs=[l_in.input_var, labels],
                           outputs=[loss,
                                    loss1_acc_top1,
                                    #loss1_acc_top5,
                                    loss1_probs,
                                    loss2_acc_top1,
                                    #loss2_acc_top5,
                                    loss2_probs,
                                    loss3_acc_top1,
                                    #loss3_acc_top5, 
                                    loss3_probs],
                           allow_input_downcast=True)
end_time = time.time()
print "%.3f sec" % (end_time-start_time)

# ################################# training #################################

print("Starting training...")

from utils import batchiterator
batchitertrain = batchiterator.BatchIterator(range(num_data), batchsize, data=(X_train, y_train))
batchitertrain = batchiterator.threaded_generator(batchitertrain,3)
                                                                                
batchiterval = batchiterator.BatchIterator(range(X_valid.shape[0]), batchsize, data=(X_valid, y_valid)) 
batchiterval = batchiterator.threaded_generator(batchiterval,3)             

import datetime
now = datetime.datetime.now()
output_filename = "output_%04d%02d%02d_%02d%02d%02d_%03d.log" % (now.year, now.month, now.day, now.hour, now.minute, now.second, now.microsecond)
with open(output_filename, "w") as f:
    f.write("Experiment Log: GoogLeNet with BN\n")
                                                                        
#num_epochs = 100
for epoch_num in range(num_epochs):
    start_time = time.time()

    # iterate over training minibatches and update the weights
    num_batches_train = int(np.ceil(len(X_train) / batchsize))
    train_losses = []
    for batch_num in range(num_batches_train):
	start_time = time.time()
        '''batch_slice = slice(batchsize * batch_num,
                            batchsize * (batch_num + 1))
        X_batch = X_train[batch_slice]
        y_batch = y_train[batch_slice]'''
        [X_batch, y_batch] = batchitertrain.next()

        #loss, = train_fn(X_batch, y_batch)
        loss, = train_fn(X_batch, y_batch)
         
        train_losses.append(loss)

        #if (epoch_num * batchsize + batch_num + 1) is 1:
        #    start_time = time.time()

        '''if (epoch_num * batchsize + batch_num + 1) % 10 is 0:
            end_time = time.time()
            
            out_str = "Iter: %d, train_loss=%f    (%.3f sec)" % (epoch_num * num_batches_train + batch_num + 1, loss, end_time-start_time)
            print(out_str)
            start_time = time.time()
        '''
    end_time = time.time()
    out_str = "Iter: %d, train_loss=%f    (%.3f sec)" % (epoch_num * num_batches_train + batch_num + 1, loss, end_time-start_time)
    print(out_str)
    # aggregate training losses for each minibatch into scalar
    train_loss = np.mean(train_losses)

    # calculate validation loss
    num_batches_valid = int(np.ceil(len(X_valid) / batchsize))
    valid_losses = []
    list_of_probabilities_batch = []
    for batch_num in range(num_batches_valid):
        '''batch_slice = slice(batchsize * batch_num,
                            batchsize * (batch_num + 1))
        X_batch = X_valid[batch_slice]
        y_batch = y_valid[batch_slice]'''
        [X_batch, y_batch] = batchiterval.next()

        #loss, probabilities_batch = valid_fn(X_batch, y_batch)
        '''loss, \
        loss1_acc_top1, loss1_acc_top5, loss1_probs, \
        loss2_acc_top1, loss2_acc_top5, loss2_probs, \
        loss3_acc_top1, loss3_acc_top5, loss3_probs = valid_fn(X_batch, y_batch)'''
        loss, \
        loss1_acc_top1, loss1_probs, \
        loss2_acc_top1, loss2_probs, \
        loss3_acc_top1, loss3_probs = valid_fn(X_batch, y_batch)
        #print(loss3_probs.shape)

        valid_losses.append(loss)
        list_of_probabilities_batch.append(loss3_probs)
    valid_loss = np.mean(valid_losses)
    # concatenate probabilities for each batch into a matrix
    probabilities = np.concatenate(list_of_probabilities_batch)
    # calculate classes from the probabilities
    predicted_classes = np.argmax(probabilities, axis=1)
    # calculate accuracy for this epoch
    #accuracy = sklearn.metrics.accuracy_score(y_valid, predicted_classes)

    out_str = "Epoch: %d, train_loss=%f, valid_loss=%f" % (epoch_num + 1, train_loss, valid_loss)
    print(out_str)
    #print("Epoch: %d, train_loss=%f, valid_loss=%f, valid_accuracy=%f"
    #      % (epoch_num + 1, train_loss, valid_loss, accuracy))

    with open(output_filename, "a") as f:
            f.write(out_str + "\n")

    if (epoch_num + 1) % 100 == 0:
        # save
        weights_save = lasagne.layers.get_all_param_values([l_loss1_classifier, l_loss2_classifier, l_loss3_classifier])
        pickle.dump( weights_save, open( "googlenet_bn_h_%d_z_%d_epoch_%d.weight.pkl" % (hidden_size, z_size, epoch_num), "wb" ) )
        # load
        #weights_load = pickle.load( open( "weights.pkl", "rb" ) )
        #lasagne.layers.set_all_param_values(output_layer, weights_load) 
