import numpy as np
import pandas as pd

import tensorflow as tf
from sklearn.model_selection import train_test_split
from multi_layer_mnist import *

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("BATCH_SIZE", 50, "バッチのサイズ")
tf.app.flags.DEFINE_integer("max_steps", 100, "訓練試行回数")
tf.app.flags.DEFINE_string("log_dir", "./log", "summary保管場所")

FILENAME = "model/multi-layer-mnist-vis.ckpt"


def monitor(__x, __y, __loss):
    y_info = tf.summary.scalar("y", )


def train():
    sess = tf.InteractiveSession()
    X, Y = get_data(False)
    start_idx, end_idx = 0, 0

    # 訓練データと検証データに分割
    train_X, test_X, train_Y, test_Y = train_test_split(X, Y)
    indexes = train_X.index

    def get_feed_dict(train):
        if train:
            _x = train_X.loc[indexes[start_idx:end_idx], :]
            _y = train_Y.loc[indexes[start_idx:end_idx], :]
            _keep_prob = 0.5
        else:
            _x = test_X
            _y = test_Y
            _keep_prob = 1
        return {
            x: _x,
            y_: _y,
            keep_prob: _keep_prob
        }

    train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train',
                                         sess.graph)
    test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')
    merged = tf.summary.merge_all()

    tf.global_variables_initializer().run()
    for i in range(FLAGS.max_steps):
        end_idx = start_idx + FLAGS.BATCH_SIZE

        # loss関数の確認
        if i % 30 == 0:
            summary, acc = sess.run([merged, accuracy],
                                    feed_dict=get_feed_dict(False))
            test_writer.add_summary(summary, i)
            print('step %d, training accuracy %g' % (i, acc))
        else:
            summary, _ = sess.run([merged, train_step],
                                  feed_dict=get_feed_dict(True))
            train_writer.add_summary(summary, i)

        # 次のループへ
        if end_idx > train_X.shape[0]:
            indexes = np.random.permutation(train_X.index)
            end_idx = 0
        start_idx = end_idx

    train_writer.close()
    test_writer.close()

    saver = tf.train.Saver()
    saver.save(sess, FILENAME)


def main(argv):
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    train()


# print('test accuracy %g' % accuracy.eval(
#     feed_dict={x: test_X, y_: test_Y, keep_prob: 1.0}))

if __name__ == '__main__':
    tf.app.run()
