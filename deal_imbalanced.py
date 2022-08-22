#处理不平衡数据，正负样本比例约为1：7
import numpy as np
from imblearn.over_sampling import BorderlineSMOTE,ADASYN

class Balanced():

    #类似于mixup，通过已有样本的线性采样来进行处理
    def overSMOTE(self, data, label):
        over_data = data.copy()
        over_label = label.copy()
        smo = BorderlineSMOTE(kind = "borderline-1",random_state = 42)
        data_smo, label_smo = smo.fit_resample(over_data, over_label)
        return data_smo, label_smo

    #过采样
    def overADASYN(self, data, label):
        over_data = data.copy()
        over_label = label.copy()
        smo = ADASYN(random_state = 42)
        data_ada, label_ada = smo.fit_resample(over_data, over_label)
        return  data_ada, label_ada

    # 欠采样，减少多数类的样本数量
    def underSampling(self, data, label):
        #转为numpy格式
        data = np.array(data)
        label = np.array(label)
        pos_num = len(label[label == 1])
        #提取数组的索引
        pos_indices = np.argwhere(label == 1).squeeze()
        neg_indices = np.argwhere(label == 0).squeeze()
        #随机选择多数类样本
        random_neg_indices = np.array(np.random.choice(neg_indices, pos_num, replace = False))
        under_sample_indices = np.concatenate([pos_indices,random_neg_indices])
        under_sample_data = data[under_sample_indices]
        under_sample_label = label[under_sample_indices]
        return under_sample_data, under_sample_label

    # 过采样,重复少数类样本数量
    def overSampling(self, data, label):
        #转为numpy格式
        data = np.array(data)
        label = np.array(label)
        neg_num =len(label[label == 0])
        # 提取数组的索引
        pos_indices = np.argwhere(label == 1).squeeze()
        neg_indices = np.argwhere(label == 0).squeeze()
        #随机选择少数类样本进行复制
        random_pos_indices = np.array(np.random.choice(pos_indices,neg_num,replace = True))
        over_sample_indices= np.concatenate([random_pos_indices,neg_indices])
        over_sample_data = data[over_sample_indices]
        over_sample_label = label[over_sample_indices]
        return over_sample_data, over_sample_label

