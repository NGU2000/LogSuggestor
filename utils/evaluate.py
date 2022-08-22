import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score,accuracy_score,roc_curve

#画出曲线
def draw_log_roc_auc():
    pass

#log type的评估函数
def evaluate_print_log(y_true,y_pred):
    pred_res = np.argmax(y_pred,axis = -1)
    one_hot = np.zeros((len(y_true),6))
    for i in range(len(y_true)):
        one_hot[i][y_true[i]] = 1
    one_hot = np.vstack((one_hot,[0,0,0,0,1,0]))
    y_pred = np.vstack((y_pred,[0,0,0,0,1,0]))
    try:
        roc = roc_auc_score(list(one_hot), list(y_pred), multi_class='ovo')
        acc = accuracy_score(y_true, pred_res)
        print("roc值是{:.3f}\n"
              "准确率是{:.3f}".format(roc, acc))
    except:
        print("Error")
    # 找样本分布


#评估
def evaluate(y_true, y_pred):
    """
    参数为list形式，1xn型
    :param predict:
    :param truth:
    :return:
    """
    #转化为numpy数组
    sampling_num = len(y_true)
    y_pred = np.array(y_pred).copy()
    truth = np.array(y_true).copy()
    T_num = np.sum(truth)
    y_pred[y_pred > 0.5] = 1
    y_pred[y_pred <= 0.5] =0
    #打印正例样本分布
    print("测试集正例个数{}, 占比{:.3f}".format(T_num, float(T_num/sampling_num)))
    #各项指标
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    #计算
    for i in range(truth.shape[0]):
        if y_pred[i] == 1 and truth[i] == 1:
            TP += 1
        elif y_pred[i] == 1 and truth[i] == 0:
            FP += 1
        elif y_pred[i] == 0 and truth[i] == 1:
            FN += 1
        else:
            TN += 1
    accuracy = 0
    precision =0
    recall =0
    F_1 =0
    ba = 0
    #计算各项指标
    try:
        accuracy = (TP + TN) / (TP + FP + FN + TN)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        F_1 = 2 * precision * recall / (precision + recall)
        ba = (TP / (TP + FN) + TN / (TN + FP)) / 2
    except:
        pass
    return accuracy,precision,recall,F_1,ba,[TP,FN,FP,TN]

#评估打印
def evaluate_print(y_true,y_pred):
    """
    打印各个指标结果，并打印结果值
    :param predict:
    :param truth:
    :return:
    """
    accuracy,precision,recall,F_1,ba,others = evaluate(y_true,y_pred)
    #数据打印
    print("准确率：{:.3f}\n"
          "精确率：{:.3f}\n"
          "召回率：{:.3f}\n"
          "F_measure:{:.3f}\n"
          "Ba:{:.3f}\n".format(accuracy, precision, recall, F_1,ba))
    #数据图表
    return others
#AUC
def auc_tf(y_true, y_pred,n_bins = 1000):
    """
    n_bins:最细粒度
    """
    # 分类别
    postive_len = tf.reduce_sum(y_true)
    if postive_len != 0:
        negative_len = tf.shape(y_true)[0] - postive_len
        total_case = postive_len * negative_len
        pos_histogram = [0 for _ in range(n_bins)]
        neg_histogram = [0 for _ in range(n_bins)]
        bin_width = 1.0 / n_bins
        # 遍历所有标签
        for i in range(tf.shape(y_true)[0]):
            nth_bin = int(y_pred[i] / bin_width)
            if y_true[i] == 1:
                pos_histogram[nth_bin] += 1
            else:
                neg_histogram[nth_bin] += 1
        # 划分类型
        accumulated_neg = 0
        satisfied_pair = 0
        # 分类
        for i in range(n_bins):
            satisfied_pair += (pos_histogram[i] * accumulated_neg + pos_histogram[i] * neg_histogram[i] * 0.5)
            accumulated_neg += neg_histogram[i]
        return satisfied_pair / float(total_case)
    else:
        return 0

#AUC
def auc_np(y_true, y_pred,n_bins = 1000):
    """
    n_bins:最细粒度
    """
    # 分类别
    postive_len = np.sum(y_true)
    if postive_len != 0:
        negative_len = y_true.shape[0] - postive_len
        total_case = postive_len * negative_len
        pos_histogram = [0 for _ in range(n_bins)]
        neg_histogram = [0 for _ in range(n_bins)]
        bin_width = 1.0 / n_bins
        # 遍历所有标签
        for i in range(y_true.shape[0]):
            nth_bin = int(y_pred[i] / bin_width)
            if y_true[i] == 1:
                pos_histogram[nth_bin] += 1
            else:
                neg_histogram[nth_bin] += 1
        # 划分类型
        accumulated_neg = 0
        satisfied_pair = 0
        # 分类
        for i in range(n_bins):
            satisfied_pair += (pos_histogram[i] * accumulated_neg + pos_histogram[i] * neg_histogram[i] * 0.5)
            accumulated_neg += neg_histogram[i]
        return satisfied_pair / float(total_case)
    else:
        return 0