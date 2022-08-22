from tensorflow.python.keras import backend as K
from tensorflow.keras import initializers, regularizers, constraints
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import *
from utils.loss import *
from configuration import *

class Attention(Layer):

    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')
        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)
        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]
        if self.bias:
            self.b = self.add_weight(shape=(input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None
        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim
        e = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))), (-1, step_dim))  # e = K.dot(x, self.W)
        if self.bias:
            e += self.b
        e = K.tanh(e)
        a = K.exp(e)
        if mask is not None:
            a *= K.cast(mask, K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        a = K.expand_dims(a)

        c = K.sum(a * x, axis=1)
        return c

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.features_dim

class my_han(object):

    def __init__(self):
        self.maxlen_sentence = max_len
        self.max_features = voc_size
        self.embedding_dims = vec_len

    def build_model(self,embeding_matrix):
        input = Input(shape=(self.maxlen_sentence,))
        x_sentence = Embedding(input_dim=self.max_features, output_dim=self.embedding_dims,
                      weights=[embeding_matrix], input_length=max_len)(input)
        x_sentence = GRU(64, return_sequences = True)(x_sentence)  # LSTM or GRU
        x_sentence = Attention(self.maxlen_sentence)(x_sentence)
        output = Dense(1, activation='sigmoid')(x_sentence)
        model = Model(inputs=input, outputs=output)
        # model.summary()
        model.compile('adam', loss = fixed_binary_loss,  metrics=['accuracy'])
        return model
