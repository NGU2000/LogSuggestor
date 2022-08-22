#定义损失函数，标签为（0,1）
#需要全部使用张量计算
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

#mse损失
def mse(y_true,y_pred):
    return tf.reduce_mean(tf.square(y_pred - tf.cast(y_true, dtype = tf.float32)))

#focal loss去不平衡用,输入都为张量
def focal_loss(y_true, y_pred, gamma=2., alpha=.25):
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1 + K.epsilon())) - K.mean(
        (1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0 + K.epsilon()))

#二分类
def cb_binary_focal_loss_fixed(y_true, y_pred, gamma=2., alpha=.75, beta=0.999):
    """
    :param y_true: A tensor of the same shape as `y_pred`
    :param y_pred:  A tensor resulting from a sigmoid
    :return: Output tensor.
    """
    truth_num = tf.cast(tf.math.count_nonzero(y_true), dtype=tf.float32)
    false_num = tf.cast(tf.math.count_nonzero(y_true - 1), dtype=tf.float32)
    cb = (1 - beta) / (1 - tf.pow(beta, tf.cast(y_true, dtype=tf.float32) * (truth_num - false_num) + false_num))
    y_true = tf.cast(y_true, tf.float32)
    # Define epsilon so that the back-propagation will not result in NaN for 0 divisor case
    epsilon = K.epsilon()
    # Add the epsilon to prediction value
    # y_pred = y_pred + epsilon
    # Clip the prediciton value
    y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
    # Calculate p_t
    p_t = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
    # Calculate alpha_t
    alpha_factor = K.ones_like(y_true) * alpha
    alpha_t = tf.where(K.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
    loss = alpha_t * K.pow((1 - p_t), gamma) * (-K.log(p_t))
    # Sum the losses in mini_batch
    loss = K.mean(K.sum( loss * cb, axis=-1))
    return 10 * loss

#mymodel_log损失函数
def multi_category_focal_loss2_fixed(y_true, y_pred,alpha =.5, gamma = 2):
    epsilon = 1.e-7
    gamma = float(gamma)
    alpha = tf.constant(alpha, dtype=tf.float32)
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
    alpha_t = y_true * alpha + (tf.ones_like(y_true) - y_true) * (1 - alpha)
    y_t = tf.multiply(y_true, y_pred) + tf.multiply(1 - y_true, 1 - y_pred)
    ce = -K.log(y_t)
    weight = tf.pow(tf.subtract(1., y_t), gamma)
    fl = tf.multiply(tf.multiply(weight, ce), alpha_t)
    loss = tf.reduce_mean(fl)
    return loss



#修正的二分交叉熵
def fixed_binary_loss(y_true, y_pred,margin = 0.6):
    y_true = tf.cast(y_true, dtype = tf.float32)
    return - (1 - theta(y_true - margin) * theta(y_pred - margin) - theta(1 - margin - y_true) *
              theta(1 - margin - y_pred)) * (y_true * K.log(y_pred + 1e-8) + (1 - y_true) * K.log(1 - y_pred + 1e-8))

def theta(t):
    return (K.sign(t)+1.)/2.

#二分类cb_loss
def CB_focal_loss(y_true, y_pred, beta = 0.999):
    margin = tf.constant(0.675)
    y_true = tf.cast(y_true, dtype= tf.float32)
    truth_num = tf.cast(tf.math.count_nonzero(y_true), dtype = tf.float32)
    false_num = tf.cast(tf.math.count_nonzero(y_true - 1), dtype = tf.float32)
    cb = (1 - beta)/(1 - tf.pow(beta, tf.cast(y_true, dtype= tf.float32) * (truth_num - false_num) + false_num))
    return 1000*(1 - theta(y_true - margin) * theta(y_pred - margin) - theta(1 - margin - y_true) * theta(1 - margin - y_pred)) \
           *cb* focal_loss(y_true,y_pred)

#二分类cb_loss
def CB_loss(y_true, y_pred, beta = 0.999):
    truth_num = tf.cast(tf.math.count_nonzero(y_true), dtype = tf.float32)
    false_num = tf.cast(tf.math.count_nonzero(y_true - 1), dtype = tf.float32)
    cb = (1 - beta)/(1 - tf.pow(beta, tf.cast(y_true, dtype= tf.float32) * (truth_num - false_num) + false_num))
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    return cb * bce(y_true,y_pred)

#多分类cbloss
def CB_log_loss(y_true, y_pred, beta = 0.999):
    truth_num = tf.cast(tf.math.count_nonzero(y_true), dtype = tf.float32)
    false_num = tf.cast(tf.math.count_nonzero(y_true - 1), dtype = tf.float32)
    cb = (1 - beta)/(1 - tf.pow(beta, tf.cast(y_true, dtype= tf.float32) * (truth_num - false_num) + false_num))
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    return cb * bce(y_true,y_pred)

def GHMC_Loss(Bins=10, momentum=0, batch_size=100):
    shape = batch_size
    bins = Bins
    edges = [float(x) / bins for x in range(bins + 1)]
    edges[-1] += 1e-6
    mmt = momentum
    def GHMC_Loss_fixed(y_true, y_pred):
        if momentum > 0:
            acc_sum = [0.0 for _ in range(bins)]
        y_true = tf.cast(y_true, tf.float32)
        weights = tf.zeros(shape)
        g = tf.abs(y_true - y_pred)
        tot = len(y_pred)
        n = 0
        for i in range(bins):
            inds = tf.cast((g >= edges[i]) & (g < edges[i + 1]), dtype=tf.int32)
            num_in_bin = tf.reduce_sum(inds)
            if num_in_bin > 0:
                if momentum > 0:
                    acc_sum[i] = momentum * acc_sum[i] + (1 - mmt) * num_in_bin
                    weights[inds.flatten()] = tot / acc_sum[i]
                else:
                    weights[inds.flatten()] = tot / num_in_bin
                n += 1
        if n > 0:
            weights = weights / n
        weights = tf.convert_to_tensor(weights.astype('float32'))
        weights = tf.reshape(weights, [shape])
        return tf.nn.weighted_cross_entropy_with_logits(labels=y_true, logits=y_pred, pos_weight=weights)
    return GHMC_Loss_fixed


