from keras.layers import RepeatVector, CuDNNGRU, GRU, Bidirectional
from keras.backend.tensorflow_backend import _is_current_explicit_device, _get_available_gpus

from src.config import base_configuration


class RNNNet:
    @property
    def layers(self) -> list:
        layers = []

        layers.append(RepeatVector(
            base_configuration['sizes']['repeat_vector_length']
        ))

        layers.append(Bidirectional(self.GRUclass(
            base_configuration['sizes']['rnn_output'],
            return_sequences=True
        )))

        return layers

    @property
    def GRUclass(self):
        if not _is_current_explicit_device('CPU') and len(_get_available_gpus()) > 0:
            print('On GPU, using CuDNNGRU layer')
            return CuDNNGRU
        else:
            print('On CPU, using GRU layer')
            return GRU
