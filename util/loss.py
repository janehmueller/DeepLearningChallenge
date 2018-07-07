from tensorflow.python.ops.nn_ops import softmax_cross_entropy_with_logits

import tensorflow as tf


def categorical_crossentropy_from_logits(y_true, y_pred):
    y_pred1 = tf.Print(y_pred[0, :], [y_pred[0, :]], '\n1PRED: ')
    y_pred2 = tf.Print(y_pred[1, :], [y_pred[1, :]], '\n2PRED: ')
    y_own_pred = tf.cast(y_pred, tf.float32)
    sum = tf.reduce_sum(y_own_pred, axis=0) + (y_pred1[0] * 0) + (y_pred2[0] * 0)
    factor = (tf.reduce_max(sum, axis=1))
    factor = tf.Print(factor, [factor], message='Factor: ')

    return softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred) * tf.cast(factor, dtype=tf.float32)
