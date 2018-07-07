from keras.layers import RepeatVector, CuDNNGRU, GRU, Bidirectional
from keras.backend.tensorflow_backend import _is_current_explicit_device, _get_available_gpus

from src.config import base_configuration
from util.checkGPU import onGPU


class RNNNet:
    @property
    def layers(self) -> list:
        layers = []

        layers.append(RepeatVector(
            base_configuration['sizes']['repeat_vector_length']
        ))

        layers.append(self.GRUclass(
            base_configuration['sizes']['rnn_output'],
            return_sequences=True
            #dropout=.2,
            #recurrent_dropout=.2
        ))

        return layers

    @property
    def GRUclass(self):
        if onGPU:
            print('On GPU, using CuDNNGRU layer')
            return CuDNNGRU
        else:
            print('On CPU, using GRU layer')
            return GRU
