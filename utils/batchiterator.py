"""
This is code a modefied one from https://github.com/skaae/lasagne-draw/blob/master/deepmodels/batchiterator.py
"""

import numpy as np
from random import shuffle
import lmdb
import cv2
import time
import sys

# pickle
try:
    import cPickle as pickle
except:
    import pickle

class BatchIterator(object):
    """
     Cyclic Iterators over batch indexes. Permutes and restarts at end
    """

    def __init__(self, batch_indices, batchsize, data, testing=False, process_func=None):
        if isinstance(batch_indices, int):
            self.n = batch_indices
            self.batchidx = np.arange(batch_indices)
        else:
            self.n = len(batch_indices)
            self.batchidx = np.array(batch_indices)

        self.batchsize = batchsize
        self.testing = testing

        if process_func is None:
            process_func = lambda x:x
        self.process_func = process_func

        if not isinstance(data, (list, tuple)):
            data = [data]

        self.data = data
        if not self.testing:
            self.createindices = lambda: np.random.permutation(self.n)
        else: # testing == true
            assert self.n % self.batchsize == 0, "for testing n must be multiple of batch size"
            self.createindices = lambda: range(self.n)

        self.perm = self.createindices()
        assert self.n > self.batchsize

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def _get_permuted_batches(self,n_batches):
        # return a list of permuted batch indeces
        batches = []
        for i in range(n_batches):

            # extend random permuation if shorter than batchsize
            if len(self.perm) <= self.batchsize:
                new_perm = self.createindices()
                self.perm = np.hstack([self.perm, new_perm])

            batches.append(self.perm[:self.batchsize])
            self.perm = self.perm[self.batchsize:]
        return batches

    def next(self):
        batch = self._get_permuted_batches(1)[0]   # extract a single batch
        data_batches = [self.process_func(data_n[batch]) for data_n in self.data]
        return data_batches


class ImagenetBatchIterator(object):
    """
     Cyclic Iterators over batch indexes. Permutes and restarts at end
    """

    def __init__(self, batchsize, database, testing=False, process_func=None):
        # open data 
         
        # initialize data
        self.in_db_data = lmdb.open(database, readonly=True)
        self.n = int(self.in_db_data.stat()['entries']) # number of data

        with self.in_db_data.begin() as in_txn:
            cursor = in_txn.cursor()
            cursor.first()
            (key, value) = cursor.item()
            img_dat = bytes(value)
            (img, label) = pickle.loads(img_dat)     
            self.img_height = img.shape[1] # img = np array with (channels, height, width)
            self.img_width = img.shape[2]
            self.img_channels = img.shape[0]

        self.in_txn = self.in_db_data.begin(buffers=True)
        self.cursor = self.in_txn.cursor()
        self.cursor.first()
 
        self.batchsize = batchsize
        self.testing = testing

        if process_func is None:
            process_func = lambda x:x
        self.process_func = process_func

        if self.testing:
            assert self.n % self.batchsize == 0, "for testing n must be multiple of batch size"

        assert self.n > self.batchsize

        self.data_batches = np.zeros((self.batchsize, self.img_channels, self.img_width, self.img_height))
        self.labels_batches = np.zeros((self.batchsize,)).astype(int)

        print "n: ", self.n
        print "img_height: ", self.img_height
        print "img_width: ", self.img_width
        print "img_channels: ", self.img_channels

    def __del__(self):
        self.cursor.close()
        self.in_db_data.close()

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        start_time = time.time()
        try: 
            for i in xrange(self.batchsize):
                for j in xrange(np.random.randint(self.batchsize)):
                    if self.cursor.next() is False:
                        self.cursor.first()
                
                '''if self.cursor.next() is False:
                    self.cursor.first()'''

                (key, value) = self.cursor.item()
                img_dat = bytes(value)
                (img, label) = pickle.loads(img_dat)
  
                self.data_batches[i,:,:,:] = self.process_func(img)
                self.labels_batches[i] = label
        except:
            self.cursor.close() 
            self.in_db_data.close()
            print "Unexpected error:", sys.exc_info()[0]
            raise 
        end_time = time.time()
        print "imagenet batchiterator: ", end_time - start_time, " sec"

        return [self.data_batches, self.labels_batches]


def threaded_generator(generator, num_cached=50):
    # this code is writte by jan Schluter
    # copied from https://github.com/benanne/Lasagne/issues/12
    import Queue
    queue = Queue.Queue(maxsize=num_cached)
    sentinel = object()  # guaranteed unique reference

    # define producer (putting items into queue)
    def producer():
        for item in generator:
            queue.put(item)
        queue.put(sentinel)

    # start producer (in a background thread)
    import threading
    thread = threading.Thread(target=producer)
    thread.daemon = True
    thread.start()

    # run as consumer (read items from queue, in current thread)
    item = queue.get()
    while item is not sentinel:
        yield item
        queue.task_done()
        item = queue.get()
