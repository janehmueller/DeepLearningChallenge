import os

import tensorflow
from tensorflow.python.ops.nn_ops import softmax_cross_entropy_with_logits
from tensorflow import reshape, reduce_all, boolean_mask, reduce_mean, cast, argmax, float32

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor import meteor
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge


def categorical_crossentropy_from_logits(y_true, y_pred):
    """
    Discarding the EOS token is still needed even though it was set to all zero arrays in the "preprocess_batch" method
    since the mean of the losses is still different (even though the sum is the same).
    :param y_true: the correct value
    :param y_pred: the predicted value
    :return: the softmax cross entropy loss with logits (from tensorflow)
    """
    # Discard the one-hot encoded EOS token
    y_true = y_true[:, :-1, :]
    y_pred = y_pred[:, :-1, :]
    return softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)


def categorical_accuracy_with_variable_timestep(y_true, y_pred):
    # Discard the one-hot encoded EOS token (not needed since they are all zero arrays)
    # y_true = y_true[:, :-1, :]
    # y_pred = y_pred[:, :-1, :]

    # Flatten the timestep dimension
    shape = tensorflow.shape(y_true)
    y_true = reshape(y_true, [-1, shape[-1]])
    y_pred = reshape(y_pred, [-1, shape[-1]])

    # Discard rows that are all zeros since they are EOS tokens or padding
    y_true_is_zero = tensorflow.equal(y_true, 0)
    y_true_is_zero_row = reduce_all(y_true_is_zero, axis=-1)
    y_true = boolean_mask(y_true, ~y_true_is_zero_row)  # ~ is bitwise not
    y_pred = boolean_mask(y_pred, ~y_true_is_zero_row)

    accuracy = reduce_mean(cast(tensorflow.equal(argmax(y_true, axis=1), argmax(y_pred, axis=1)), dtype=float32))
    return accuracy


# As Keras stores a function's name as its metric's name
categorical_accuracy_with_variable_timestep.__name__ = 'categorical_accuracy_wvt'


class Score(object):
    """
    Subclasses wrap pycocoevalcap.
    """
    def __init__(self, score_name, implementation):
        self.score_name = score_name
        self.implementation = implementation

    def calculate(self, id_to_prediction, id_to_references):
        id_to_prediciton_list = dict(map(lambda kv: (kv[0], [kv[1]]), id_to_prediction.items()))
        average_score, scores = self.implementation.compute_score(id_to_references, id_to_prediciton_list)
        if isinstance(average_score, (list, tuple)):
            average_score = map(float, average_score)
        else:
            average_score = float(average_score)
        return {
            self.score_name: average_score
        }


class BLEU(Score):
    def __init__(self, n=4):
        super(BLEU, self).__init__("bleu", Bleu(n))
        self.n = n

    def calculate(self, id_to_prediction, id_to_references):
        name_to_score = super(BLEU, self).calculate(id_to_prediction, id_to_references)
        scores = name_to_score.values()[0]
        result = {}
        for index, score in enumerate(scores, start=1):
            name = f"{self.score_name}_{index}"
            result[name] = score


class CIDEr(Score):
    def __init__(self):
        super(CIDEr, self).__init__("cider", Cider())


class METEOR(Score):
    def __init__(self):
        super(METEOR, self).__init__("meteor", Meteor())

    def calculate(self, id_to_prediction, id_to_references):
        if self.data_downloaded():
            return super(METEOR, self).calculate(id_to_prediction, id_to_references)
        else:
            return {
                self.score_name: 0.0
            }

    def data_downloaded(self):
        meteor_dir = os.path.dirname(meteor.__file__)
        jar_exists = os.path.isfile(os.path.join(meteor_dir, "meteor-1.5.jar"))
        zip_exists = os.path.isfile(os.path.join(meteor_dir, "data", "paraphrase-en.gz"))
        return jar_exists and zip_exists


class ROUGE(Score):
    def __init__(self):
        super(ROUGE, self).__init__("rouge", Rouge())
