from tensorflow.python.ops.nn_ops import softmax_cross_entropy_with_logits

import tensorflow as tf


def categorical_crossentropy_from_logits(y_true, y_pred):
    #y_pred1 = tf.Print(y_pred[0, :], [y_pred[0, :]], '\n1PRED: ')
    #y_pred2 = tf.Print(y_pred[1, :], [y_pred[1, :]], '\n2PRED: ')
    y_own_pred = tf.cast(y_pred, tf.float32)
    y_pred_max = tf.argmax(y_own_pred, axis=2)
    one_hot = tf.one_hot(y_pred_max, tf.shape(y_own_pred)[2])
    sum = tf.reduce_sum(one_hot, axis=0)# + (y_pred1[0] * 0) + (y_pred2[0] * 0)
    #max = tf.reduce_max(y_own_pred, axis=0)
    sum = sum - 1
    sum = tf.maximum(sum, 0)
    sum = tf.Print(sum, [sum], 'SUM: ')
    factor = (tf.reduce_sum(sum, axis=1))
    factor = tf.Print(factor, [factor], message='Factor: ')
    factor = factor * 0.6

    result = softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred) + factor
    result = tf.Print(result, [result], 'RESULT:')

    return result
