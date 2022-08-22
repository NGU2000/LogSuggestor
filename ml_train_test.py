from utils.fileloader import txt2list,txt2tfrec,recReader
import time
from sklearn import svm
from utils.evaluate import evaluate_print
from classifier.plattoSvm import PlattSMO
from sklearn.ensemble import RandomForestClassifier
from preprocessing import preprocess
import pickle
from utils.util import train_test_builder, zero_padding_vec
from word_embedding import tf_idf, load_w2vec, pretrain_w2vec
from deal_imbalanced import Balanced
from sklearn.decomposition import PCA
from classifier.word2vec import  load_w2v_model
from configuration import  *
from sklearn.preprocessing import StandardScaler
import numpy as np
import random
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import  pydot

class ML:

    def __init__(self):
        pass

    # SVM模型
    def SVM_train(self, x_train, y_train, name=""):
        start_time = time.time()
        # 使用GA算法进行参数优化
        model = svm.SVC(C=100,max_iter = 80000, tol=1e-2, kernel='rbf', gamma="auto", cache_size=6060,
                        class_weight="balanced", verbose=2)
        model.fit(x_train, y_train)
        end_time = time.time()
        print("分类器训练耗时{:.3f}s".format(end_time - start_time))
        # 保存参数
        with open('variables/svm_{}.pickle'.format(name), 'wb') as f:
            pickle.dump(model, f)

    def SVM_test(self,x_test,y_test,name):
        with open('variables/svm_{}.pickle'.format(name), 'rb') as f:
            model = pickle.load(f)
        # 结果预测与评估
        predicted = model.predict(x_test)
        evaluate_print(y_test,predicted)

    #随机森林模型
    def RF_train(self,x_train, y_train, name=""):
        rf = RandomForestClassifier()
        rf.fit(x_train,y_train)
        # 保存参数
        with open('variables/RandomForest_{}.pickle'.format(name), 'wb') as f:
            pickle.dump(rf, f)

    def RF_test(self,x_test, y_test, name=""):
        with open('variables/RandomForest_{}.pickle'.format(name), 'rb') as f:
            model = pickle.load(f)
            # 结果预测与评估
        predicted = model.predict(x_test)
        evaluate_print( y_test,predicted)

    #决策树模型
    def DT_train(self,x_train, y_train, name=""):
        tree = DecisionTreeClassifier(max_depth=100,random_state=0)
        tree.fit(x_train,y_train)
        # 保存参数
        with open('variables/DecisionTree_{}.pickle'.format(name), 'wb') as f:
            pickle.dump(tree, f)

    def DT_test(self,x_test, y_test, name=""):
        with open('variables/DecisionTree_{}.pickle'.format(name), 'rb') as f:
            model = pickle.load(f)
            # 结果预测与评估
        predicted = model.predict(x_test)
        evaluate_print(y_test,predicted)

    # plattoSVM
    def svm_plattsmo(self,x_train, y_train, x_test, y_test):
        start_time = time.time()
        smo = PlattSMO(x_train, y_train, 200, 0.0001, 10000, name='rbf', theta=20)
        smo.smoP()
        end_time = time.time()
        print("分类器训练耗时{:.3f}s".format(end_time - start_time))
        predicted = smo.predict(x_test)
        evaluate_print(predicted, y_test)

    # 分类器选择训练函数
    def classifier_train(self,x_train, y_train, name=""):
        # print("SVM训练")
        # self.SVM_train(x_train.copy(), y_train.copy(), name)
        # print("RF训练")
        # self.RF_train(x_train.copy(), y_train.copy(), name)
        print("DT训练")
        self.DT_train(x_train.copy(), y_train.copy(), name)


    # 分类器选择测试函数，测试不同的值
    def classifier_test(self, x_test, y_test, name):
        print("SVM")
        self.SVM_test(x_test.copy(), y_test.copy(), name)
        print("RF")
        self.RF_test(x_test.copy(), y_test.copy(), name)
        print("DT")
        self.DT_test(x_test.copy(), y_test.copy(), name)
    # 机器学习训练
    def ML_train(self, data_path, sampling_choice = 1, need_pre = False, splited = True, balanced = False):
        # 读取本地数据 combined_data : [syn_vec,semi_vec,fus_vec]
        combined_data, labels, tf_rate = txt2list(data_path)
        verboses = [1, 0, 0]
        print("开始数据处理")
        # 一次处理三种向量
        for i in range(3):
            dataset_name = data_path.split('\\')[-1].replace(".txt", "")
            output_name = dataset_name + str(i)
            #判断是否需要预处理
            if need_pre == True:
                pass
            # 判断是否划分过数据集
            if splited == True:
                train_filename = "{}/{}_{}.txt".format(train_data_path, dataset_name, i)
                test_filename = "{}/{}_{}.txt".format(test_data_path, dataset_name, i)
                with open(train_filename, "r", encoding="utf-8") as f:
                    train_data = f.read()
                    train_datas = train_data.split('\n')
                    x_train = []
                    y_train = []
                    for j in range(int((len(train_datas) - 1) / 2)):
                        x_train.append(train_datas[2 * j])
                        y_train.append(int(train_datas[2 * j + 1]))
                    f.close()
                with open(test_filename, "r", encoding="utf-8") as f:
                    test_data = f.read()
                    test_datas = test_data.split('\n')
                    x_test = []
                    y_test = []
                    for x in range(int((len(test_datas) - 1) / 2)):
                        x_test.append(test_datas[2 * x])
                        y_test.append(int(test_datas[2 * x + 1]))
                    f.close()
            # 不平衡处理
            if balanced == True:
                balance = Balanced()
                # 欠采样
                if sampling_choice == 0:
                    x_train, y_train = balance.underSampling(x_train, y_train)
                # 过采样
                elif sampling_choice == 1:
                    x_train, y_train = balance.overSampling(x_train, y_train)
            # 导入词向量模型
            w2v_model, _ = load_w2v_model(w2v_model100_path)
            x_train = self.w2v_transform(w2v_model,x_train)
            x_test = self.w2v_transform(w2v_model,x_test)
            print("向量转化完毕")
            #特征提取，降维
            x_train = PCA(100).fit_transform(x_train)
            x_test = PCA(100).fit_transform(x_test)
            print("降维完毕")
            x_train = StandardScaler().fit_transform(x_train)
            x_test = StandardScaler().fit_transform(x_test)
            print("归一化完成")
            # 分类器模型训练和测试
            self.classifier_train(x_train, y_train, name = output_name)

    # 机器学习预测
    def ML_test(self,data_path, splited = True):
        # 读取本地数据 combined_data : [syn_vec,semi_vec,fus_vec]
        verboses = [1, 0, 0]
        print("开始数据处理")
        # 一次处理三种向量
        for i in range(3):
            dataset_name = data_path.split('\\')[-1].replace(".txt", "")
            output_name = dataset_name + str(i)
            print(output_name)
            # 判断是否划分过数据集
            if splited == True:
                test_filename = "{}\\{}_{}.txt".format(test_data_path, dataset_name, i)
                print(test_filename)
                with open(test_filename, "r", encoding="utf-8") as f:
                    test_data = f.read()
                    test_datas = test_data.split('\n')
                    x_test = []
                    y_test = []
                    for x in range(int((len(test_datas) - 1) / 2)):
                        x_test.append(test_datas[2 * x])
                        y_test.append(int(test_datas[2 * x + 1]))
                    f.close()
            # 导入词向量模型
            w2v_model, _ = load_w2v_model(w2v_model100_path)
            x_test = self.w2v_transform(w2v_model,x_test)
            print("向量转化完毕")
            #特征提取，降维
            x_test = PCA(100).fit_transform(x_test)
            print("降维完毕")
            x_test = StandardScaler().fit_transform(x_test)
            print("归一化完成")
            #测试与评估
            self.classifier_test(x_test, y_test, name = output_name)

    # 转换为向量
    def w2v_transform(self,model,data_vec):
        word2v = np.zeros((len(data_vec), max_len_ml, vec_len))
        for i in range(len(data_vec)):
            words = data_vec[i].split(" ")
            if len(words) <= max_len_ml:
                for j in range(len(words)):
                    word2v[i, j,:] = model.wv[words[j]]
            #否则从后往前随机裁剪
            else:
                dis = len(words) - max_len_ml
                random_len = random.randint(0, int(dis / 4))
                start_pos = max(dis - random_len,0)
                end_pos = len(words) - random_len
                #遍历所有文件
                for j in range(start_pos,end_pos):
                    word2v[i, j - (dis - random_len),:] = model.wv[words[j]]
        return word2v.reshape((len(data_vec), max_len_ml * vec_len))