from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras import initializers, regularizers, constraints
from configuration import *
from utils.loss import  *

class MyAttention(Layer):

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
        super(MyAttention, self).__init__(**kwargs)

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

    def get_config(self):
        config = {
            "W_regularizer": self.W_regularizer,
            "b_regularizer": self.b_regularizer,
            "W_constraint": self.W_constraint,
            "b_constraint": self.b_constraint,
            "bias": self.bias,
            "step_dim": self.step_dim
        }
        base_config = super(MyAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class myModel():
    def __init__(self):
        self.maxlen_sentence = max_len
        self.maxlen_syn = max_len_syn
        self.max_features = voc_size
        self.embedding_dims = vec_len

    def build_model(self, embedding_matrix =[]):
        # textual part
        # Sentence part
        input2 = Input(shape=(self.maxlen_sentence,))
        x_sentence = Embedding(input_dim=self.max_features, output_dim=self.embedding_dims,
                               weights=[embedding_matrix], input_length=max_len)(input2)
        x_sentence = Bidirectional(GRU(64, return_sequences=True))(x_sentence)
        x_sentence = MyAttention(self.maxlen_sentence)(x_sentence)

        input1 = Input(shape=(self.maxlen_syn,))
        x_syn = Embedding(input_dim=self.max_features, output_dim=self.embedding_dims,
                               weights=[embedding_matrix], input_length=max_len)(input1)
        x_syn = Bidirectional(GRU(64, return_sequences=True))(x_syn)
        x_syn = MyAttention(self.maxlen_syn)(x_syn)
        concate = Concatenate()([x_sentence,x_syn])
        output = Dense(1, activation='sigmoid')(concate)
        model = Model(inputs=[input1,input2], outputs=output)
        # model.summary()
        model.compile('adam', loss = CB_focal_loss, metrics=['accuracy'])
        return model


class myModel_log():
    def __init__(self):
        self.maxlen_sentence = max_len
        self.maxlen_syn = max_len_syn
        self.max_features = voc_size
        self.embedding_dims = vec_len

    def build_model(self, embedding_matrix =[]):
        # textual part
        # Sentence part
        input2 = Input(shape=(self.maxlen_sentence,))
        x_sentence = Embedding(input_dim=self.max_features, output_dim=self.embedding_dims,
                                weights=[embedding_matrix],input_length=max_len)(input2)
        x_sentence = Bidirectional(GRU(64, return_sequences=True))(x_sentence)
        x_sentence = Reshape((400,128,1), input_shape=(400,128))(x_sentence)
        x_sentence = Conv2D(filters=64, kernel_size=(5,128), padding='valid', activation='relu')(x_sentence)
        x_sentence = Dropout(0.2)(x_sentence)
        x_sentence = BatchNormalization()(x_sentence)
        x_sentence = Reshape((396,64), input_shape=(396,1,64))(x_sentence)
        x_sentence = MyAttention(396)(x_sentence)

        input1 = Input(shape=(self.maxlen_syn,))
        x_syn = Embedding(input_dim=self.max_features, output_dim=self.embedding_dims,
                          weights=[embedding_matrix], input_length=max_len_syn)(input1)
        x_syn = Bidirectional(GRU(64, return_sequences=True))(x_syn)
        x_syn = Reshape((100,128,1), input_shape=(100,128))(x_syn)
        x_syn = Conv2D(filters=64, kernel_size=(5,128), padding='valid', activation='relu')(x_syn)
        x_syn = Dropout(0.2)(x_syn)
        x_syn = BatchNormalization()(x_syn)
        x_syn = Reshape((96,64), input_shape=(96,1,64))(x_syn)
        x_syn = MyAttention(96)(x_syn)

        concate = Concatenate()([x_sentence,x_syn])

        output = Dense(6, activation='softmax')(concate)
        model = Model(inputs=[input1,input2], outputs=output)
        # model.summary()
        model.compile('adam', loss = multi_category_focal_loss2_fixed, metrics=['accuracy'])
        return model
