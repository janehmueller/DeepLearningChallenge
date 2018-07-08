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

    #y_pred1 = tf.Print(y_pred[0, :], [y_pred[0, :]], '\n1PRED: ')
    #y_pred2 = tf.Print(y_pred[1, :], [y_pred[1, :]], '\n2PRED: ')
    y_pred_float = tf.cast(y_pred, tf.float32)
    y_true_float = tf.cast(y_true, tf.float32)
    #y_pred_max = tf.argmax(y_own_pred, axis=2)

    #one_hot = tf.one_hot(y_pred_max, tf.shape(y_own_pred)[2])
    #return one_hot
    #one_hot = tf.tanh(y_pred_max)
    #sum = tf.reduce_sum(one_hot, axis=0)# + (y_pred1[0] * 0) + (y_pred2[0] * 0)
    sum = tf.reduce_mean(y_pred_float, axis=1)# + (y_pred1[0] * 0) + (y_pred2[0] * 0)
    #sum = tf.Print(sum, [sum], 'SUM------|: ', summarize=10000)
    print(sum.get_shape())

    #max_sum = tf.reduce_max(sum, axis=1, keepdims=True)
    #max_sum = tf.Print(max_sum, [max_sum], 'MAX: ')
    #max_sum = max_sum - .1
    #max_sum = tf.Print(max_sum, [max_sum], 'MAX: ')
    #print(max_sum.get_shape())
    #max_sum = tf.map_fn(lambda value_1: tf.map_fn(lambda value_2: tf.tile(value_2, 23683), value_1), max_sum)
    #max_sum = tf_repeat(max_sum, [[1, 64] * tf.shape(max_sum)[0]])
    #max_sum = tf_repeat(max_sum, [1, 1, 23683])
    #max_sum = tf.Print(max_sum, [max_sum], 'MAX: ')
    #print(max_sum.get_shape())
    #pred_sum = y_pred - max_sum
    #pred_sum = tf.Print(pred_sum, [pred_sum], 'pred_sum1: ')
    #pred_sum = tf.maximum(pred_sum, 0)
    #pred_sum = tf.Print(pred_sum, [pred_sum], 'pred_sum2: ')
    #pred_sum = pred_sum * 1000 * 100
    #pred_sum = tf.Print(pred_sum, [pred_sum], 'pred_sum3: ')
    #pred_sum = tf.reduce_max(pred_sum, axis=1)
    #pred_sum = tf.Print(pred_sum, [pred_sum], 'pred_sum4: ')

    pred_sum = y_pred_float - 1000
    pred_sum = tf.maximum(pred_sum, 0)
    pred_sum = tf.Print(pred_sum, [pred_sum], 'pred_sum5: ')
    pred_sum = pred_sum + tf.square(sum) * 500
    #y_pred_float = tf.Print(y_pred_float, [y_pred_float], '\nPRED_INPU: ', summarize=1000)
    pred_sum = tf.Print(pred_sum, [pred_sum], 'pred_sum6: ')
    #y_true_float = tf.Print(y_true_float, [y_true_float], '\nTRUE_INPU: ', summarize=1000)

    true_diff = y_pred_float - (y_true_float * 100)
    #true_diff = tf.Print(true_diff, [true_diff], 'TRUE DIFF: ')
    #true_diff = tf.square(true_diff)
    #true_diff = tf.Print(true_diff, [true_diff], 'TRUE DIFF: ')
    true_diff = true_diff * 10
    #true_diff = tf.Print(true_diff, [true_diff], 'TRUE DIFF: ')
    pred_sum = pred_sum + true_diff
    #pred_sum = tf.Print(pred_sum, [pred_sum], 'pred_sum final: ')


    print(pred_sum.get_shape())
    return pred_sum

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
