import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow import nn
class SelfAttnModel(tf.keras.Model):

    def __init__(self, input_dims, data_format='channels_last', **kwargs):
        super(SelfAttnModel, self).__init__(**kwargs)
        self.attn = _Attention(data_format=data_format)
        self.query_conv = tf.keras.layers.Conv2D(filters=input_dims // 8,
                                                 kernel_size=1,
                                                 data_format=data_format)
        self.key_conv = tf.keras.layers.Conv2D(filters=input_dims // 8,
                                               kernel_size=1,
                                               data_format=data_format)
        self.value_conv = tf.keras.layers.Conv2D(filters=input_dims,
                                                 kernel_size=1,
                                                 data_format=data_format)

    def call(self, inputs, training=False):
        q = self.query_conv(inputs)
        k = self.key_conv(inputs)
        v = self.value_conv(inputs)
        return self.attn([q, k, v, inputs])


class _Attention(tf.keras.layers.Layer):

    def __init__(self, data_format='channels_last', **kwargs):
        super(_Attention, self).__init__(**kwargs)
        self.data_format = data_format

    def build(self, input_shapes):
        self.gamma = self.add_weight(self.name + '_gamma',
                                     shape=(),
                                     initializer=tf.initializers.Zeros)

    def call(self, inputs):
        if len(inputs) != 4:
            raise Exception('an attention layer should have 4 inputs')
        query_tensor = inputs[0]
        key_tensor = inputs[1]
        value_tensor = inputs[2]
        origin_input = inputs[3]
        input_shape = tf.shape(query_tensor)
        if self.data_format == 'channels_first':
            height_axis = 2
            width_axis = 3
        else:
            height_axis = 1
            width_axis = 2

        batchsize = input_shape[0]
        height = input_shape[height_axis]
        width = input_shape[width_axis]

        if self.data_format == 'channels_first':
            proj_query = tf.transpose(
                tf.reshape(query_tensor, (batchsize, -1, height * width)), (0, 2, 1))
            proj_key = tf.reshape(key_tensor, (batchsize, -1, height * width))
            proj_value = tf.reshape(value_tensor, (batchsize, -1, height * width))
        else:
            proj_query = tf.reshape(query_tensor, (batchsize, height * width, -1))
            proj_key = tf.transpose(
                tf.reshape(key_tensor, (batchsize, height * width, -1)), (0, 2, 1))
            proj_value = tf.transpose(
                tf.reshape(value_tensor, (batchsize, height * width, -1)), (0, 2, 1))

        energy = tf.matmul(proj_query, proj_key)
        attention = tf.nn.softmax(energy)
        out = tf.matmul(proj_value, tf.transpose(attention, (0, 2, 1)))

        if self.data_format == 'channels_first':
            out = tf.reshape(out, (batchsize, -1, height, width))
        else:
            out = tf.reshape(
                tf.transpose(out, (0, 2, 1)), (batchsize, height, width, -1))

        return tf.add(tf.multiply(out, self.gamma), origin_input), attention

class channelAttention(Layer):
    def __init__(self , in_planes , ration = 16):
        super(channelAttention, self).__init__()
        # inputsz = np.array(alist.shape[1:])
        # outputsz = np.array([2, 3])
        #
        # stridesz = np.floor(inputsz / outputsz).astype(np.int32)
        #
        # kernelsz = inputsz - (outputsz - 1) * stridesz
        #
        # adp = t.nn.AdaptiveAvgPool2d(list(outputsz))
        # avg = tf.nn.AvgPool2d(kernel_size=[1, kernelsz, kernelsz, 1], stride=[1, stridesz, stridesz, 1])
        #
        self.avg_pool = nn.avg_pool2d(1) # 通道数不变，H*W变为1*1
        self.max_pool = nn.avg_pool2d(1) #
        self.fc1 = Conv2D(in_planes , in_planes // 16 , 1 , bias = False)
        self.relu1 = ReLU()
        self.fc2 = Conv2D(in_planes//16 , in_planes ,1, bias = False)
        self.sigmoid = tf.sigmoid()

    def forward(self , x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        #两层神经网络共享
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)