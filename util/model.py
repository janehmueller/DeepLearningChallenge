import tensorflow as tf
from keras import Input
from keras.models import Model as KerasModel
from keras.applications import InceptionV3
from keras.initializers import RandomUniform
from keras.layers import BatchNormalization, Dense, RepeatVector, Embedding, GRU, LSTM, Bidirectional, TimeDistributed, \
    Concatenate, Lambda
from keras.optimizers import Adam, SGD
from keras.regularizers import l1_l2
from keras.utils import multi_gpu_model

from .config import base_configuration
from .metrics import categorical_crossentropy_from_logits, categorical_accuracy_with_variable_timestep
from .word_vectors import WordVector


class Model(object):
    def __init__(self,
                 learning_rate=None,
                 vocab_size=None,
                 embedding_size=None,
                 rnn_output_size=None,
                 dropout_rate=None,
                 bidirectional_rnn=None,
                 rnn_type=None,
                 rnn_layers=None,
                 l1_reg=None,
                 l2_reg=None,
                 initializer=None,
                 word_vector_init=None):
        self.keras_model = None

        self.learning_rate = learning_rate or base_configuration["params"]["learning_rate"]
        self.vocab_size = vocab_size or base_configuration["params"]["vocab_size"]
        self.embedding_size = embedding_size or base_configuration["params"]["embedding_size"]
        self.rnn_output_size = rnn_output_size or base_configuration["params"]["rnn_output_size"]
        self.dropout_rate = dropout_rate or base_configuration["params"]["dropout_rate"]
        self.rnn_type = rnn_type or base_configuration["params"]["rnn_type"]
        self.rnn_layers = rnn_layers or base_configuration["params"]["rnn_layers"]
        self.word_vector_init = word_vector_init or base_configuration["params"]["word_vector_init"]

        self.initializer = initializer or base_configuration["params"]["initializer"]
        if self.initializer == 'vinyals_uniform':
            self.initializer = RandomUniform(-0.08, 0.08)

        self.bidirectional_rnn = bidirectional_rnn or base_configuration["params"]["bidirectional_rnn"]

        self.regularizer = l1_l2()  # (l1_reg, l2_reg)

        if self.vocab_size is None:
            raise ValueError('config.active_config().vocab_size cannot be None! You should check your config or you can'
                             ' explicitly pass the vocab_size argument.')

        if self.rnn_type not in ('lstm', 'gru'):
            raise ValueError('rnn_type must be either "lstm" or "gru"!')

        if self.rnn_layers < 1:
            raise ValueError('rnn_layers must be >= 1!')

        if self.word_vector_init is not None and self.embedding_size != 300:
            raise ValueError('If word_vector_init is not None, embedding_size must be 300')

    def build(self, vocabulary=None):
        if self.keras_model:
            return
        # if not vocabulary and self.word_vector_init:
        image_input, image_embedding = self.build_image_embedding()
        sentence_input, word_embedding = self.build_word_embedding(vocabulary)

        if base_configuration['print_layer_outputs']:
            image_embedding = Lambda(self.lambda_print_layer("Image Embedding output: "))(image_embedding)
            word_embedding = Lambda(self.lambda_print_layer("Word Embedding output: "))(word_embedding)

        rnn_input = Concatenate(axis=1)([image_embedding, word_embedding])

        if base_configuration['print_layer_outputs']:
            rnn_input = Lambda(self.lambda_print_layer("RNN input output: "))(rnn_input)

        rnn_output = self.build_rnn_model(rnn_input)
        if base_configuration['print_layer_outputs']:
            rnn_output = Lambda(self.lambda_print_layer("RNN output output: "))(rnn_output)

        model = KerasModel(inputs=[image_input, sentence_input], outputs=rnn_output)
        print('LEARNING_RATE: {}'.format(self.learning_rate))
        model = multi_gpu_model(model)
        model.compile(
            #optimizer=Adam(lr=self.learning_rate, clipnorm=5.0),  # Gradients will be clipped when L2 norm exceeds value
            optimizer=SGD(lr=self.learning_rate),  # Gradients will be clipped when L2 norm exceeds value
            loss=categorical_crossentropy_from_logits,
            metrics=[categorical_accuracy_with_variable_timestep]
        )
        self.keras_model = model

    def lambda_print_layer(self, message):
        def print_results(tensor):
            return tf.Print(tensor, [tensor], message)
        return print_results

    def build_image_embedding(self):
        # Initialize with imagenet weights
        image_model = InceptionV3(include_top=False, weights="imagenet", pooling="avg")

        # Fix weights
        for layer in image_model.layers:
            layer.trainable = False

        dense_input = BatchNormalization(axis=-1)(image_model.output)
        if base_configuration['print_layer_outputs']:
            dense_input = Lambda(self.lambda_print_layer("Batch Normalization output: "))(dense_input)

        # TODO: regularizer and initializer
        # kernel_regularizer=self.regularizer,
        # kernel_initializer=self.initializer
        dense_image = Dense(
            units=self.embedding_size
        )(dense_input)

        if base_configuration['print_layer_outputs']:
            dense_image = Lambda(self.lambda_print_layer("Dense Image output: "))(dense_image)

        # Add timestep dimension to fit the RNN dimensions
        image_embedding = RepeatVector(1)(dense_image)

        image_input = image_model.input
        return image_input, image_embedding

    def build_word_embedding(self, vocabulary):
        sentence_input = Input(shape=[None])
        self.vocab_size = len(vocabulary)
        if base_configuration['print_layer_outputs']:
            sentence_input = Lambda(self.lambda_print_layer("Sentence Input output: "))(sentence_input)

        if not self.word_vector_init:
            print("Not using word vector init...")
            # TODO: regularizer
            # embeddings_regularizer=self.regularizer,
            word_embedding = Embedding(
                input_dim=self.vocab_size,
                output_dim=self.embedding_size
            )(sentence_input)
        else:
            print("Using word vector init...")
            word_vector = WordVector(vocabulary, self.initializer, self.word_vector_init)
            embedding_weights = word_vector.vectorize_words(vocabulary)
            # TODO: regularizer
            # embeddings_regularizer=self.regularizer,
            word_embedding = Embedding(
                input_dim=self.vocab_size,
                output_dim=self.embedding_size,
                weights=[embedding_weights]
            )(sentence_input)

        if base_configuration['print_layer_outputs']:
            word_embedding = Lambda(self.lambda_print_layer("Word Embedding output: "))(word_embedding)
        return sentence_input, word_embedding

    def rnn(self):
        RNN = GRU if self.rnn_type == "gru" else LSTM
        # TODO: regularizer and initializer
        # kernel_regularizer=self.regularizer,
        # kernel_initializer=self.initializer
        rnn = RNN(
            units=self.rnn_output_size,
            return_sequences=True,
            dropout=self.dropout_rate,
            recurrent_dropout=self.dropout_rate,
            implementation=2
        )
        if self.bidirectional_rnn:
            rnn = Bidirectional(rnn)
        return rnn

    def build_rnn_model(self, sequence_input):
        layer_input = sequence_input
        for _ in range(0, self.rnn_layers):
            layer_input = BatchNormalization(axis=-1)(layer_input)
            # rnn_output = self.rnn()(layer_input)
            # TODO: regularizer and initializer
            # kernel_regularizer=self.regularizer,
            # kernel_initializer=self.initializer
            rnn_output = LSTM(
                units=self.rnn_output_size,
                return_sequences=True,
                dropout=self.dropout_rate,
                recurrent_dropout=self.dropout_rate
            )(layer_input)
            layer_input = rnn_output

        dense_time_distributed_layer = TimeDistributed(Dense(units=self.vocab_size))(rnn_output)
        return dense_time_distributed_layer
