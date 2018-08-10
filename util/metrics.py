from os import path
from typing import Dict, List

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor import meteor
from pycocoevalcap.rouge.rouge import Rouge


class Score(object):
    """A subclass of this class is an adapter of pycocoevalcap."""

    def __init__(self, name: str, implementation):
        self._name = name
        self._implementation = implementation

    def calculate(self, image_id_to_prediction: Dict[int, str], image_id_to_captions: Dict[int, List[str]]) -> dict:
        # wrap prediction in a list since that is the format the implementations expect
        image_id_to_predictions_list = {image_id: [caption] for image_id, caption in image_id_to_prediction.items()}
        avg_score, scores = self._implementation.compute_score(image_id_to_captions, image_id_to_predictions_list)

        if isinstance(avg_score, (list, tuple)):
            avg_score = [float(score) for score in avg_score]
        else:
            avg_score = float(avg_score)

        return {self._name: avg_score}


class CIDEr(Score):
    def __init__(self):
        super(CIDEr, self).__init__('CIDEr', Cider())


class ROUGE(Score):
    def __init__(self):
        super(ROUGE, self).__init__('ROUGE', Rouge())


class BLEU(Score):
    def __init__(self, n: int = 4):
        super(BLEU, self).__init__('BLEU', Bleu(n))
        self._n = n

    def calculate(self,
                  image_id_to_prediction: Dict[int, str],
                  image_id_to_captions: Dict[int, List[str]]) -> Dict[str, float]:
        name_to_score = super(BLEU, self).calculate(image_id_to_prediction, image_id_to_captions)

        scores = name_to_score[self._name]
        result = {}
        for index, score in enumerate(scores, start=1):
            name = '{}({})'.format(self._name, index)
            result[name] = score

        return result


class METEOR(Score):
    def __init__(self):
        super(METEOR, self).__init__('METEOR', meteor.Meteor())

    def calculate(self,
                  image_id_to_prediction: Dict[int, str],
                  image_id_to_captions: Dict[int, List[str]]) -> Dict[str, float]:
        if self._data_downloaded():
            return super(METEOR, self).calculate(image_id_to_prediction, image_id_to_captions)

        return {self._name: 0.0}

    def _data_downloaded(self):
        meteor_dir = path.dirname(meteor.__file__)
        jar_exists = path.isfile(path.join(meteor_dir, 'meteor-1.5.jar'))
        data_exists = path.isfile(path.join(meteor_dir, 'data', 'paraphrase-en.gz'))
        return jar_exists and data_exists
