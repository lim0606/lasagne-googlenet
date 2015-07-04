import time
from utils import batchiterator
import numpy as np
import theano

batchsize = 32

print "imagenet batchiterator"
batchitertrain = batchiterator.ImagenetBatchIterator(batchsize=batchsize, database='/media/data0/image/ilsvrc12/ilsvrc12_train_lmdb')
batchitertrain = batchiterator.threaded_generator(batchitertrain,3)

time.sleep(1.)    # pause 1. seconds

num_epoch = 30
for i in xrange(num_epoch):
    start_time = time.time()
    [X_batch, y_batch] = batchitertrain.next()
    end_time = time.time()
    
    time.sleep(0.4) # approximated processing time 
    print "time (", i, "): ", end_time-start_time, " sec"


print "imagenet batchiterator (crop)" 

batchitertrain = batchiterator.ImagenetBatchCropIterator(batchsize=batchsize, crop='random', crop_height=224, crop_width=224, flip=True, database='/media/data0/image/ilsvrc12/ilsvrc12_train_lmdb')
batchitertrain = batchiterator.threaded_generator(batchitertrain,3)

time.sleep(1.)    # pause 1. seconds

num_epoch = 30
for i in xrange(num_epoch):
    start_time = time.time()
    [X_batch, y_batch] = batchitertrain.next()
    end_time = time.time()

    time.sleep(0.4) # approximated processing time
    print "time (", i, "): ", end_time-start_time, " sec"

