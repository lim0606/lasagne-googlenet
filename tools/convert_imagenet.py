"""
   This code is a modified version of Yoann Baveye's code in https://groups.google.com/forum/#!searchin/caffe-users/python$20script/caffe-users/-vWuaM3bnro/RB4xzpI8xeUJ 
"""
import sys

import lmdb
import re, fileinput, math
import numpy as np

import theano

# pickle
try:
    import cPickle as pickle
except:
    #import pickle
    print("hi")

import cv2
from random import shuffle

# Command line to check created files:
# python -mlmdb stat --env=./Downloads/caffe-master/data/liris-accede/train_score_lmdb/

data = 'train.txt'
lmdb_data_name = 'ilsvrc12_train_python_lmdb'

Inputs = []

for line in fileinput.input(data):
	entries = re.split(' ', line.strip())
	Inputs.append(('train/' + entries[0], int(entries[1])))

# shuffle 
shuffle(Inputs)

print('Writing data')
for idx in range(int(math.ceil(len(Inputs)/1000.0))):
    in_db_data = lmdb.open(lmdb_data_name, map_size=int(1e12)) # map_size is about 931 GB
    with in_db_data.begin(write=True) as in_txn:
        for in_idx, (in_, label_) in enumerate(Inputs[(1000*idx):(1000*(idx+1))]):
            # read image
            #im = caffe.io.load_image(in_)
            im = cv2.imread(in_)
            im = cv2.resize(im, (256, 256), interpolation=cv2.INTER_CUBIC)
            im = np.transpose(im, [2, 1, 0]) # channel, height, width
            im = im.astype(theano.config.floatX) 

            # serialize image data
            #im_dat = caffe.io.array_to_datum(im.astype(float).transpose((2, 0, 1)))
            #in_txn.put('{:0>10d}'.format(1000*idx + in_idx), im_dat.SerializeToString())
            im_dat = pickle.dumps((im, label_))
            in_txn.put('{:0>10d}'.format(1000*idx + in_idx), im_dat)

            string_ = str(1000*idx+in_idx+1) + ' / ' + str(len(Inputs))
            sys.stdout.write("\r%s" % string_)
            sys.stdout.flush()
    in_db_data.close()
print('')
