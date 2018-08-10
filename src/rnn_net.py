from keras import regularizers
from keras.layers import RepeatVector, CuDNNGRU, GRU, Bidirectional, BatchNormalization, LSTM
from keras.backend.tensorflow_backend import _is_current_explicit_device, _get_available_gpus

from src.config import base_configuration
from util.checkGPU import on_GPU


class RNNNet:
    @property
    def layers(self) -> list:
        layers = [
            BatchNormalization(),
            self.GRUclass(
                base_configuration['sizes']['rnn_output'],
                return_sequences=True,
                dropout=base_configuration['sizes']['dropout_rate'],
                recurrent_dropout=base_configuration['sizes']['dropout_rate'],
                # kernel_regularizer=regularizers.l2(0.01)
            )
        ]
        return layers

    @property
    def GRUclass(self):
        if on_GPU:
            # print('On GPU, using CuDNNGRU layer')
            print('On GPU, not using CuDNNGRU layer since we use dropout')
            return GRU
            # return CuDNNGRU
        else:
            print('On CPU, using GRU layer')
            return GRU
