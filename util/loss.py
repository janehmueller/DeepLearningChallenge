from keras.losses import categorical_crossentropy
from tensorflow.python.ops.nn_ops import softmax_cross_entropy_with_logits

import tensorflow as tf


# taken from https://github.com/tensorflow/tensorflow/issues/8246
def tf_repeat(tensor, repeats):
    """
    Args:

    input: A Tensor. 1-D or higher.
    repeats: A list. Number of repeat for each dimension, length must be the same as the number of dimensions in input

    Returns:

    A Tensor. Has the same type as input. Has the shape of tensor.shape * repeats
    """
    with tf.variable_scope("repeat"):
        expanded_tensor = tf.expand_dims(tensor, -1)
        multiples = [1] + repeats
        tiled_tensor = tf.tile(expanded_tensor, multiples=multiples)
        repeated_tesnor = tf.reshape(tiled_tensor, tf.shape(tensor) * repeats)
    return repeated_tesnor


def categorical_crossentropy_from_logits(y_true, y_pred):
    #with tf.Session() as sess:
    #    print("SHAPE: " + sess.run(tf.shape(y_pred)))

    print(y_true.get_shape())
    print(y_pred.get_shape())

    y_pred_float = tf.cast(y_pred, tf.float32)
    y_pred_float = tf.Print(y_pred_float, [y_pred_float], 'Y_PRED: ', summarize=1000)
    y_true_float = tf.cast(y_true, tf.float32)
    y_true_float = tf.Print(y_true_float, [y_true_float], 'Y_TRUE: ', summarize=1000)
    sum = tf.reduce_mean(y_pred_float, axis=1, keepdims=True)
    print(sum.get_shape())

    max_sum = tf.reduce_max(sum, axis=1, keepdims=True)
    max_sum = max_sum - .1
    max_sum = tf.Print(max_sum, [max_sum], 'MAX_SUM: ')

    pred_sum = y_pred_float - 1000
    pred_sum = tf.maximum(pred_sum, 0)
    #pred_sum = tf.Print(pred_sum, [pred_sum], 'pred_sum5: ')
    square_content = (tf.maximum(sum - (max_sum * 0.9), 0) * 10) + 1
    #square_content = tf.Print(square_content, [square_content], '\nSQUARE_CONT: ', summarize=1000)
    squared = tf.square(square_content) - 1
    #squared = tf.Print(squared, [squared], '\nSQUARED: ', summarize=1000)
    pred_sum = tf.add(pred_sum, squared)


    #y_pred_float = tf.Print(y_pred_float, [y_pred_float], '\nPRED_INPU: ', summarize=1000)
    #pred_sum = tf.Print(pred_sum, [pred_sum], 'pred_sum6: ')
    #y_true_float = tf.Print(y_true_float, [y_true_float], '\nTRUE_INPU: ', summarize=1000)

    #true_diff = y_pred_float - (y_true_float * 2)
    #true_diff = tf.Print(true_diff, [true_diff], 'TRUE DIFF: ')
    #true_diff = tf.square(true_diff)
    #true_diff = tf.Print(true_diff, [true_diff], 'TRUE DIFF: ')
    #true_diff = tf.square(true_diff * 10)
    #true_diff = tf.Print(true_diff, [true_diff], 'TRUE DIFF: ')
    #pred_sum = pred_sum + true_diff
    #pred_sum = tf.Print(pred_sum, [pred_sum], 'pred_sum final: ')


    #pred_sum = tf.Print(pred_sum, [pred_sum], 'pred_sum 99999: ')
    pred_sum = tf.reduce_mean(pred_sum, axis=2)
    pred_sum = tf.Print(pred_sum, [pred_sum], 'pred_sum final: ')

    print(pred_sum.get_shape())

    categorical_loss = categorical_crossentropy(y_true_float, y_pred_float)
    categorical_loss = tf.Print(categorical_loss, [categorical_loss], 'CATLOSS: ')
    print(categorical_loss.get_shape())

    loss = pred_sum + categorical_loss
    loss = tf.Print(loss, [loss], 'LOSS: ')

    return categorical_loss

    #max = tf.reduce_max(y_own_pred, axis=0)
    # sum = sum - 1
    # sum = tf.maximum(sum, 0)
    # sum = tf.Print(sum, [sum], 'SUM: ')
    #
    # print(sum.get_shape())
    #
    # return sum
    #
    # factor = (tf.reduce_sum(sum, axis=1))
    # factor = tf.Print(factor, [factor], message='Factor: ')
    # factor = factor * 0.6
    #
    # result = softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred) + factor
    # result = tf.Print(result, [result], 'RESULT:')
    #
    # return result
