from keras.layers import RepeatVector, GRU

from src.config import base_configuration


class RNNNet:
    @property
    def layers(self) -> list:
        layers = []

        layers.append(RepeatVector(
            base_configuration['sizes']['repeat_vector_length']
        ))

        layers.append(GRU(
            base_configuration['sizes']['rnn_output'],
            return_sequences=True,
            stateful=True
        ))

        return layers
