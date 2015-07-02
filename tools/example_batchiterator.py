import time
from utils import batchiterator
import numpy as np
import theano

batchsize = 32
datalist = '/media/data1/image/ilsvrc12/train.txt'
data_root = '/media/data1/image/ilsvrc12/train/'
batchitertrain = batchiterator.ImagenetBatchIterator(batchsize=batchsize, database='/media/data0/image/ilsvrc12/ilsvrc12_train_lmdb')
batchitertrain = batchiterator.threaded_generator(batchitertrain,3)

time.sleep(1.)    # pause 1. seconds

num_epoch = 2000
for i in xrange(num_epoch):
    start_time = time.time()
    [X_batch, y_batch] = batchitertrain.next()
    end_time = time.time()
    
    time.sleep(0.4) # approximated processing time 
    print "time (", i, "): ", end_time-start_time, " sec"

