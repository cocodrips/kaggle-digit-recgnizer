import numpy as np
import pandas as pd
import time

import tensorflow as tf
from sklearn.model_selection import train_test_split
from multi_layer_mnist import *

FILENAME = "model/multi-layer-mnist2.ckpt"

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
label = train['label']

# dense_to_one_hot
Y = pd.get_dummies(label)
X = train.drop('label', axis=1) / 255  # normalize
X = X.astype(np.float32)

test = test.astype(np.float32)

train_X, test_X, train_Y, test_Y = train_test_split(X, Y)

saver = tf.train.Saver()
predict = tf.argmax(y_conv, 1)

N = len(test)
with tf.Session() as sess:
    load_start = time.time()
    saver.restore(sess, FILENAME)
    load_end = time.time()

    print('FINISH! - load model {}s'.format(load_end - load_start))
    print('test accuracy %g' % accuracy.eval(
        feed_dict={x: test_X, y_: test_Y, keep_prob: 1.0}))

    predicted_labels = sess.run(predict,
                                feed_dict={x: test[:N], keep_prob: 1.0})

    np.savetxt('data/submission_softmax.csv',
               np.c_[range(1, N + 1), predicted_labels],
               delimiter=',',
               header='ImageId,Label', comments='', fmt='%d')

    print('finish predict')
