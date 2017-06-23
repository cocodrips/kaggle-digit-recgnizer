import tensorflow as tf
import numpy as np
import pandas as pd

img_size = 28 * 28


def get_data(test=False):
    if not test:
        train = pd.read_csv('data/train.csv')
        label = train['label']
        Y = pd.get_dummies(label)

        X = train.astype(np.float32)
        X = X.drop('label', axis=1) / 255  # normalize

        return X, Y

    X = pd.read_csv('data/test.csv')
    X = X.astype(np.float32)
    return X


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


# 重みパラメータを作る
def weight_variable(shape):
    # std=0.1 regular expression で初期化
    init = tf.truncated_normal(shape,
                               stddev=0.1)
    return tf.Variable(init)


# バイアス
def bias_variable(shape):
    init = tf.constant(0.1, shape=shape)  # 0.1固定
    return tf.Variable(init)


# CNN

# Convolutin layer
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# pool 2x2 
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding='SAME')


def nn(x):
    """
    :type x: np.array
    """
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    # Convolution layer 1

    # Filter size 5x5, input_size, output_size
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])  # ↑の4つ目と同じサイズ
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # Convolution layer 2
    W_conv2 = weight_variable([5, 5, 32, 64])  # 32 -> 64
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # 接続層
    # 7x7の画像 -> 1024のニューロンの完全連結レイヤへ
    W_fc = weight_variable([7 * 7 * 64, 1024])
    b_fc = bias_variable([1024])

    h_pool2flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc = tf.nn.relu(tf.matmul(h_pool2flat, W_fc) + b_fc)

    # Dropout層 検証中はdropoutしないようにノードをつくって切り替えられるようにする
    keep_prob = tf.placeholder(tf.float32)  # node
    h_fc_drop = tf.nn.dropout(h_fc, keep_prob)  # dropout済みのh_fc

    # 読み出し層
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.matmul(h_fc_drop, W_fc2) + b_fc2
    return y_conv, keep_prob


x = tf.placeholder(tf.float32, shape=(None, img_size))
y_ = tf.placeholder(tf.float32, shape=(None, 10))

y_conv, keep_prob = nn(x)
# loss
loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
tf.summary.scalar('loss', loss)

# Adam Optimizer
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy', accuracy)
