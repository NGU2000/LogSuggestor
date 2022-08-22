#查看数据分布分析和作图
from utils.fileloader import txt2list
from configuration import *
import matplotlib.pyplot as plt
import numpy as np
import os
import pygal

file_path =  raw_data_path
paths = os.listdir(file_path)
# 这两行代码解决 plt 中文显示的问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
axis_name = []
tol_tf = []
tol_tf_rate = []
tol_txt_len = []
true_nums = []
false_nums = []

#遍历所有结果
for i in paths:
    axis_name.append(i.replace(".txt",""))
    combined_data, labels, [true_num,false_num] = txt2list(os.path.join(file_path,i))
    true_nums.append(100*true_num/(true_num+false_num))
    false_nums.append(-100*false_num/(true_num+false_num))
    tol_tf_rate.append(100*true_num/false_num)
    for z in combined_data[1]:
        tol_txt_len.append(len(z.split(' ')))

#文本长度分量
tol_txt_len = np.array(tol_txt_len)
sample_num = tol_txt_len.shape[0]
#划分区间
distr_area = []
dis = 150
for i in range(6):
    distr_area.append(i * dis)
distr_name = []
for i in range(5):
    distr_name.append('{}-{}'.format(distr_area[i],distr_area[i+1]))
distr_name.append('>{}'.format(distr_area[-1]))
area_num = []
for x in range(5):
    tmp = tol_txt_len[tol_txt_len >= x * dis]
    tmp = tmp[tmp < (x + 1) * dis]
    area_num.append(100*tmp.shape[0]/sample_num)
area_num.append(100*tol_txt_len[tol_txt_len >= distr_area[5]].shape[0]/sample_num)

#正负样本分布图
def TF_plot():
    plt.figure(1)
    plt.xlabel("数据集", fontsize=15)
    plt.ylabel("样本比例(%)", fontsize=15)
    bar_width = 0.4
    index_true = np.arange(len(axis_name))
    plt.bar(index_true, height=true_nums, width=bar_width, color='#1F77B4', label='正样本')
    plt.bar(index_true, height=false_nums, width=bar_width, color='#FF7F0E', label='负样本')
    plt.xticks(index_true, axis_name)
    for x, y in zip(index_true, true_nums):
        plt.text(x, y + 3, '{:.2f}'.format(y), ha='center', va='bottom')

    for x, y in zip(index_true, false_nums):
        plt.text(x, y - 4.5, '{:.2f}'.format(-y), ha='center', va='bottom')
    plt.legend(loc='best', fontsize=10)
    plt.ylim(-100, 40)
    plt.title("各数据集正负样本分布图", fontsize=20)
    plt.show()

# 语义向量长度分布图
def txtlen_plot():
    plt.figure(2)
    bar_width = 0.5
    index = np.arange(len(distr_area))
    plt.xticks(index, distr_name)
    for x in range(len(index)):
        plt.bar(index[x], area_num[x], width = bar_width, color = '#05B9E2',alpha = 1 - 0.05 * x)
    for x, y in zip(index, area_num):
        plt.text(x, y + 3, '{:.2f}'.format(y), ha='center', va='bottom')
    plt.xlabel("向量包含词个数", fontsize = 15)
    plt.ylabel("比例(%)", fontsize = 15)
    plt.title("语义向量长度分布图", fontsize = 20)
    plt.ylim(0,100)
    plt.show()

#日志类别分布图
def log_type_distr():
    root_path = "Dataset2\\raw"
    datasets = os.listdir(root_path)
    tol_num = []
    sum_tol = []
    for dataset_name in datasets:
        file_path = os.path.join(root_path,dataset_name)
        combined_data, labels, _ = txt2list(file_path)
        nums = []
        tol = 0
        for i in range(6):
            num = labels.count(i)
            nums.append(num)
            tol += num
        sum_tol.append(tol)
        tol_num.append(nums)
        print(dataset_name)
    index = np.arange(len(distr_area))
    plt.xticks(index, axis_name)
    bar_width = 0.1
    for x in range(len(index)):
        for y in range(6):
            plt.bar(index[x] + (y-2) * bar_width, 100 * (tol_num[x][y])/sum_tol[x], width=bar_width, color='#05B9E2')
    print(tol_num)
    plt.xlabel("数据集", fontsize=15)
    plt.ylabel("比例(%)", fontsize=15)
    plt.title("日志输出类型分布图", fontsize=20)
    plt.ylim(0,100)
    plt.figure(3)
    plt.show()

#分析主函数
# def analysis_main():
#     # TF_plot()
#     # txtlen_plot()
#     log_type_distr()
#
# analysis_main()