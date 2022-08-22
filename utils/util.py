from sklearn.model_selection import train_test_split
from utils.fileloader import txt2tfrec, recReader
import numpy as np
import tensorflow as tf
import os

#设置
def settings():
    # 设置参数
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

#划分数据集
def train_test_builder(data, label, train_rate):
    x_train, x_test, y_train, y_test = train_test_split(data, label,
                                                        test_size = 1 - train_rate, random_state=5, shuffle=True)
    return x_train, y_train, x_test, y_test

#填充向量维度
def zero_padding_vec(vec, pad_size, channel = 2):
    """
    vec:向量
    pad_size:目标维度大小
    channel: 填充维度，默认第2维度
    """
    vec = np.pad(vec,((0,0),(0,pad_size - vec.shape[1])),mode = 'constant')
    return vec

#填充向量维度
def one_padding_vec(vec, pad_size, channel = 2):
    """
    vec:向量
    pad_size:目标维度大小
    channel: 填充维度，默认第2维度
    """
    vec = np.pad(vec,((0,0),(0,pad_size - vec.shape[1])),mode = 'constant',constant_values = (1))
    return vec


#写入读取tfrecords数据
def records_rw(filename):
    # 转换数据集
    txt2tfrec("Dataset/Tomcat.txt")
    # 读取数据为张量
    blocksmt, label = recReader("tomcat.tfrecords")
    return blocksmt,label
