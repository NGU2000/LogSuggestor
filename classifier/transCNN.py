import numpy as np
from tensorflow.keras.layers import *
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.models import Sequential,Model

class PositionalLayer(Layer):
    def __init__(self, input_units):
        super().__init__()
        assert input_units % 2 == 0, "Input_units should be even."
        self.base = K.constant((1 / 10000) ** (np.arange(input_units / 2) * 2 / input_units))

    def call(self, x):
        length = K.shape(x)[1]
        angles = K.transpose(K.tile(self.base[:, None], [1, length]) * K.arange(0, length, dtype='float32'))
        positional_encoding = K.concatenate([K.sin(angles), K.cos(angles)], axis=1)
        return x + positional_encoding


class MultiHeadAttention(Layer):
    def __init__(self, input_units, head_units, transform_units, **kargs):
        super().__init__()
        self.head_units = head_units
        self.dense_q = TimeDistributed(Dense(transform_units * head_units))
        self.dense_k = TimeDistributed(Dense(transform_units * head_units))
        self.dense_v = TimeDistributed(Dense(transform_units * head_units))
        self.attention = Attention(**kargs)
        self.dense_output = TimeDistributed(Dense(input_units))

    def _split_and_concat(self, x):
        return K.concatenate(tf.split(x, self.head_units, axis=-1), axis=0)

    def call(self, q, v, q_mask, v_mask):
        k = v
        q_transform = self._split_and_concat(self.dense_q(q))
        v_transform = self._split_and_concat(self.dense_v(v))
        k_transform = self._split_and_concat(self.dense_k(k))

        head_concat = K.concatenate(
            tf.split(
                self.attention(
                    [q_transform, v_transform, k_transform],
                    mask=[
                        K.tile(q_mask, [self.head_units, 1]),
                        K.tile(v_mask, [self.head_units, 1])
                    ]
                ),
                self.head_units,
                axis=0
            ),
            axis=-1
        )
        return self.dense_output(head_concat)


class ResNorm(Layer):
    def __init__(self, sequential):
        super().__init__()
        self.sequential = sequential
        self.layer_norm = LayerNormalization()

    def call(self, x):
        return self.layer_norm(x + self.sequential(x))


class ResNormAttention(Layer):
    def __init__(self, attention_layer):
        super().__init__()
        self.attention_layer = attention_layer
        self.layer_norm = LayerNormalization()

    def call(self, q, v, q_mask, v_mask):
        return self.layer_norm(q + self.attention_layer(q, v, q_mask, v_mask))


class EncoderLayer(Layer):
    def __init__(self, input_units, head_units, transform_units, dropout, ffn_units):
        super().__init__()
        self.attention = ResNormAttention(
            MultiHeadAttention(
                input_units,
                head_units,
                transform_units,
                use_scale=True,
                dropout=dropout
            )
        )
        self.ffn = ResNorm(Sequential([
            TimeDistributed(Dense(ffn_units, activation='relu')),
            TimeDistributed(Dense(input_units)),
        ]))

    def call(self, encoding, padding):
        return self.ffn(self.attention(encoding, encoding, padding, padding))


class Encoder(Layer):
    def __init__(self, embedding_input_dim, embedding_output_dim, layer_units, head_units, transform_units, dropout,
                 ffn_units):
        super().__init__()
        self.embedding_output_dim = embedding_output_dim
        self.embedding_layer = Embedding(embedding_input_dim, embedding_output_dim)
        self.pos_layer = PositionalLayer(embedding_output_dim)
        self.encoder_layers = [EncoderLayer(embedding_output_dim, head_units, transform_units, dropout, ffn_units) for _
                               in range(layer_units)]

    def call(self, embedding_input, padding):
        encoding = self.embedding_layer(embedding_input) * K.sqrt(K.constant(self.embedding_output_dim))
        encoding = self.pos_layer(encoding)
        for layer in self.encoder_layers:
            encoding = layer(encoding, padding)
        return encoding
class DecoderLayer(Layer):
    def __init__(self, input_units, head_units, transform_units, dropout, ffn_units):
        super().__init__()
        self.attention1 = ResNormAttention(
            MultiHeadAttention(
                input_units,
                head_units,
                transform_units,
                use_scale=True,
                causal=True,
                dropout=dropout
            )
        )
        self.attention2 = ResNormAttention(
            MultiHeadAttention(
                input_units,
                head_units,
                transform_units,
                use_scale=True,
                dropout=dropout
            )
        )
        self.ffn = ResNorm(Sequential([
            TimeDistributed(Dense(ffn_units, activation='relu')),
            TimeDistributed(Dense(input_units)),
        ]))


    def call(self, encoding, decoding, encoding_padding, decoding_padding):
        return self.ffn(
            self.attention2(
                self.attention1(decoding, decoding, decoding_padding, decoding_padding),
                encoding,
                decoding_padding,
                encoding_padding
            )
        )


class Decoder(Layer):
    def __init__(self, embedding_input_dim, embedding_output_dim, layer_units, head_units, transform_units, dropout,
                 ffn_units):
        super().__init__()
        self.embedding_output_dim = embedding_output_dim
        self.embedding_layer = Embedding(embedding_input_dim, embedding_output_dim)
        self.pos_layer = PositionalLayer(embedding_output_dim)
        self.decoder_layers = [DecoderLayer(embedding_output_dim, head_units, transform_units, dropout, ffn_units) for _
                               in range(layer_units)]
        self.final_layer = TimeDistributed(Dense(embedding_input_dim))

    def call(self, encoding, embedding_input, encoding_padding, decoding_padding):
        decoding = self.embedding_layer(embedding_input) * K.sqrt(K.constant(self.embedding_output_dim))
        decoding = self.pos_layer(decoding)
        for layer in self.decoder_layers:
            decoding = layer(encoding, decoding, encoding_padding, decoding_padding)
        decoding = self.final_layer(decoding)
        return decoding

# ENCODER_EMBEDDING_INPUT_DIM = 10783 + 2
# ENCODER_EMBEDDING_OUTPUT_DIM = 128
# DECODER_EMBEDDING_INPUT_DIM = 10783 + 2
# DECODER_EMBEDDING_OUTPUT_DIM = 128
# LAYER_UNITS = 4
# HEAD_UNITS = 8
# TRANSFORM_UNITS = ENCODER_EMBEDDING_OUTPUT_DIM // HEAD_UNITS
# FFN_UNITS = 512
# DROPOUT = 0.1
#
# encoder = Encoder(
#     ENCODER_EMBEDDING_INPUT_DIM,
#     ENCODER_EMBEDDING_OUTPUT_DIM,
#     LAYER_UNITS,
#     HEAD_UNITS,
#     TRANSFORM_UNITS,
#     DROPOUT,
#     FFN_UNITS
# )
#
# decoder = Decoder(
#     DECODER_EMBEDDING_INPUT_DIM,
#     DECODER_EMBEDDING_OUTPUT_DIM,
#     LAYER_UNITS,
#     HEAD_UNITS,
#     TRANSFORM_UNITS,
#     DROPOUT,
#     FFN_UNITS
# )
#
# def loss_func(decoding_real, decoding_pred):
#     mask = K.not_equal(decoding_real, 0)
#     # from_logits=True表示预测的解码向量没有经过softmax
#     loss = tf.keras.losses.sparse_categorical_crossentropy(decoding_real, decoding_pred, from_logits=True)
#     mask = tf.cast(mask, dtype=loss.dtype)
#     loss *= mask
#     return K.mean(loss)
#
#
#
# encoder_embedding_input = Input([None], dtype='int64')
# decoder_embedding_input = Input([None], dtype='int64')
# # 遮挡编码为0的位置，编码0在分词器中为空串，不会出现在句子中间
# encoding_padding = K.not_equal(encoder_embedding_input, 4)
# decoding_padding = K.not_equal(decoder_embedding_input, 4)
# encoding = encoder(encoder_embedding_input, encoding_padding)
# decoding = decoder(encoding, decoder_embedding_input, encoding_padding, decoding_padding)
# output = Dense(1)(decoding)
# transformer = Model(
#     inputs=[
#         encoder_embedding_input,
#         decoder_embedding_input
#     ],
#     outputs=output
# )
# transformer.summary()