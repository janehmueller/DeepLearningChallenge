from typing import Dict, List
from pycocoevalcap.cider.cider import Cider

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
        super(CIDEr, self).__init__('cider', Cider())
