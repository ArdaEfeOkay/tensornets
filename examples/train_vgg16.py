import sys
sys.path.append("..") 

import time
import numpy as np
import tensorflow as tf
import tensornets as nets

from tensornets.datasets import voc
#TODO: Change the data_dir
data_dir = "/home/taehoonlee/Data/VOCdevkit/VOC%d"
#TODO: Write a load_train function for TcP dataset
trains = voc.load_train([data_dir % 2007, data_dir % 2012],
                        'trainval', batch_size=48)

# Define a model
inputs = tf.placeholder(tf.float32, [None, 224, 224, 3])
is_training = tf.placeholder(tf.bool)
model = nets.VGG16(inputs, classes = 2, is_training=is_training)

#def yolov2(x, stem_fn, stem_out=None, is_training=False, classes=20, scope=None, reuse=None):
#def vgg16(x, is_training=False, classes=1000, stem=False, scope=None, reuse=None):
#def vgg(x, blocks, is_training, classes, stem, scope=None, reuse=None):

# Define an optimizer
'''
step = tf.Variable(0, trainable=False)
lr = tf.train.piecewise_constant(
    step, [100, 180, 320, 570, 1000, 40000, 60000],
    [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-4, 1e-5])
train = tf.train.MomentumOptimizer(lr, 0.9).minimize(model.loss,
                                                     global_step=step)
'''

new_learning_rate = 0.001
loss = tf.losses.sparse_softmax_cross_entropy(labels=next_output, logits=logits)   
optimizer = tf.train.GradientDescentOptimizer(new_learning_rate)
train = optimizer.minimize(model.loss)

with tf.Session() as sess:

    # Load Darknet19
    sess.run(tf.global_variables_initializer())
    sess.run(model.stem.pretrained())

    # We run "sess.run(model.pretrained())" before the testing VGG16
    # sess.run(model.pretrained())

    losses = []
    for i in range(233):

        # Iterate on VOC07+12 trainval once
        _t = time.time()
        for (imgs, metas) in trains:
            # `trains` returns None when it covers the full batch once
            if imgs is None:
                break
            metas.insert(0, model.preprocess(imgs))  # for `inputs`
            metas.append(True)  # for `is_training`
            outs = sess.run([train, model.loss],
                            dict(zip(model.inputs, metas)))
            losses.append(outs[1])

        # Report step, learning rate, loss, weight decay, runtime
        print('***** %d %.5f %.5f %.5f %.5f *****' %
              (sess.run(step), sess.run(lr),
               losses[-1], sess.run(tf.losses.get_regularization_loss()),
               time.time() - _t))

        # Report with VOC07 test
        results = []
        tests = voc.load(data_dir % 2007, 'test', total_num=100)
        for (img, scale) in tests:
            outs = sess.run(model, {inputs: model.preprocess(img),
                                    is_training: False})
            results.append(model.get_boxes(outs, img.shape[1:3]))
        print(voc.evaluate(results, data_dir % 2007, 'test'))


