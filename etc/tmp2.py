
l_loss3_classifier = lasagne.layers.DenseLayer(
    l_pool5,
    num_units=1000,
    nonlinearity=lasagne.nonlinearities.softmax,
    W=lasagne.init.GlorotUniform(),
    #b=None,
    name='loss3_classifier'
)

loss3_probs = lasagne.layers.get_output(l_loss2_classifier, data)
loss3 = categorical_crossentropy(loss2_probs, labels)
loss3 = aggregate(loss2, mode='mean')

loss3_preds_top1 = theano.tensor.argmax(loss_probs, axis=1)
loss3_acc_top1 = theano.tensor.mean(
    theano.tensor.eq(loss3_preds_top1, labels),
    dtype=theano.config.floatX)

loss3_preds_top5 = theano.tensor.argsort(loss_probs, axis=1)[:,:5]
loss3_acc_top5 = theano.tensor.mean(
    theano.tensor.eq(loss3_preds_top5[:,0], labels) +
    theano.tensor.eq(loss3_preds_top5[:,1], labels) +
    theano.tensor.eq(loss3_preds_top5[:,2], labels) +
    theano.tensor.eq(loss3_preds_top5[:,3], labels) +
    theano.tensor.eq(loss3_preds_top5[:,4], labels),
    dtype=theano.config.floatX)
