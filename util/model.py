from keras import Input
from keras.models import Model as KerasModel
from keras.applications import InceptionV3
from keras.initializers import RandomUniform
from keras.layers import BatchNormalization, Dense, RepeatVector, Embedding, GRU, LSTM, Bidirectional, TimeDistributed, \
    Concatenate
from keras.optimizers import Adam
from keras.regularizers import l1_l2

from util.config import base_configuration
from util.metrics import categorical_crossentropy_from_logits, categorical_accuracy_with_variable_timestep
from util.word_vectors import WordVector


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

        self.regularizer = l1_l2(l1_reg, l2_reg)

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
        rnn_input = Concatenate(axis=1)([image_embedding, word_embedding])
        rnn_output = self.build_rnn_model(rnn_input)

        model = KerasModel(inputs=[image_input, sentence_input], outputs=rnn_output)
        model.compile(
            optimizer=Adam(lr=self.learning_rate, clipnorm=5.0),  # Gradients will be clipped when L2 norm exceeds value
            loss=categorical_crossentropy_from_logits,
            metrics=[categorical_accuracy_with_variable_timestep]
        )
        self.keras_model = model

    def build_image_embedding(self):
        # Initialize with imagenet weights
        image_model = InceptionV3(include_top=False, weights="imagenet", pooling="avg")

        # Fix weights
        for layer in image_model.layers:
            layer.trainable = False

        dense_input = BatchNormalization(axis=-1)(image_model.output)
        dense_image = Dense(
            units=self.embedding_size,
            kernel_regularizer=self.regularizer,
            kernel_initializer=self.initializer)(dense_input)

        # Add timestep dimension to fit the RNN dimensions
        image_embedding = RepeatVector(1)(dense_image)

        image_input = image_model.input
        return image_input, image_embedding

    def build_word_embedding(self, vocabulary):
        sentence_input = Input(shape=[None])

        if not self.word_vector_init:
            word_embedding = Embedding(
                input_dim=self.vocab_size,
                output_dim=self.embedding_size,
                embeddings_regularizer=self.regularizer
            )(sentence_input)
        else:
            word_vector = WordVector(vocabulary, self.initializer, self.word_vector_init)
            embedding_weights = word_vector.vectorize_words(vocabulary)
            word_embedding = Embedding(
                input_dim=self.vocab_size,
                output_dim=self.embedding_size,
                embeddings_regularizer=self.regularizer,
                weights=[embedding_weights]
            )(sentence_input)
        return sentence_input, word_embedding

    def build_rnn_model(self, sequence_input):
        RNN = GRU if self.rnn_type == "gru" else LSTM

        def rnn():
            rnn = RNN(
                units=self.rnn_output_size,
                return_sequences=True,
                dropout=self.dropout_rate,
                recurrent_dropout=self.dropout_rate,
                kernel_regularizer=self.regularizer,
                kernel_initializer=self.initializer,
                implementation=2
            )
            if self.bidirectional_rnn:
                rnn = Bidirectional(rnn)
            return rnn

        layer_input = sequence_input
        for _ in range(0, self.rnn_layers):
            layer_input = BatchNormalization(axis=-1)(layer_input)
            rnn_output = rnn()(layer_input)
            layer_input = rnn_output

        dense_time_distributed_layer = TimeDistributed(Dense(units=self.vocab_size))(rnn_output)
        return dense_time_distributed_layer
