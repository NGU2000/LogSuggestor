#深度学习训练
import random
from utils.fileloader import txt2list
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from wordbag_builder import read_wordbag
from preprocessing import preprocess
from utils.util import train_test_builder
from configuration import *
from classifier.txtCnn import TextCNN
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, LearningRateScheduler, CSVLogger
from classifier.word2vec import load_w2v_model
from utils.evaluate import evaluate, evaluate_print
from functools import partial
import math
import numpy as np
from deal_imbalanced import Balanced
from classifier.HAN import my_han
from classifier.myModel import myModel,myModel_log
import tensorflow as tf
class DL:

    def __init__(self):
        pass

    #词袋模型的训练函数
    def DL_train_bog(self, data_path, if_balanced = False, need_pre = False):
        # 读取本地数据 combined_data : [syn_vec,semi_vec,fus_vec]
        combined_data, labels, tf_rate = txt2list(data_path)
        verboses = [1, 0, 0]
        print("开始数据处理")
        word2num, num2word = self.words2dict_bow()
        vocad_size = len(word2num) + 2
        for i in range(3):
            # 不需要预处理
            if need_pre == False:
                dataset_name = data_path.split('/')[-1].replace(".txt", "")
                dataset_path = "{}/{}_pre{}.txt".format(train_data_path,dataset_name, i)
                combined_data[i] = self.read_preprocessed_feature(dataset_path)
            # 需要预处理
            else:
                combined_data[i] = preprocess(combined_data[i], verboses[i])
            # 划分数据集
            x_train, y_train, x_test, y_test = train_test_builder(combined_data[i], labels, train_rate)
            # 找到最长长度的句子
            maxlen = max(max([len(s.split(' ')) for s in x_train]),max([len(s.split(' ')) for s in x_test]))
            # 向量化
            x_train_vec = self.words2vec_bow(x_train, maxlen)
            x_test_vec = self.words2vec_bow(x_test, maxlen)
            #判断是否进行平衡
            if if_balanced == True:
                pass
            textcnn = TextCNN()
            # 创建模型
            model = textcnn.build_model(maxlen)
            print(vocad_size)
            print(maxlen)
            # self.train(model,x_train_vec, np.array(y_train),str(i))
            #预测结果并评估
            y_predict = model.predict(x_test_vec)
            evaluate(y_test, y_predict)

    #word2vec训练函数
    def DL_train_w2v(self, data_path, need_pre = False, splited = True, balanced = False, c_model = 2):
        # 读取本地数据 combined_data : [syn_vec,semi_vec,fus_vec]
        combined_data, labels, tf_rate = txt2list(data_path)
        verboses = [1, 0, 0]
        print("开始数据处理")
        #处理好syn和textual数据进行运算
        if c_model == 2:
            dataset_name = data_path.split('\\')[-1].replace(".txt", "")
            train_data = []
            train_label = []
            train_filename_0 = "{}/{}_{}.txt".format(train_data_path, dataset_name, 0)
            train_filename_1 = "{}/{}_{}.txt".format(train_data_path, dataset_name, 1)
            test_filename_0 = "{}/{}_{}.txt".format(test_data_path, dataset_name, 0)
            test_filename_1 = "{}/{}_{}.txt".format(test_data_path, dataset_name, 1)
            # 读取数据
            if splited == True:
                x_train_0 = []
                x_train_1 = []
                y_train = []
                x_test_0 = []
                x_test_1 = []
                y_test = []
                self.read_txt(train_filename_0,x_train_0,y_train)
                self.read_txt(train_filename_1,x_train_1,[])
                self.read_txt(test_filename_0,x_test_0,y_test)
                self.read_txt(test_filename_1,x_test_1,[])
            # 导入模型权重
            w2v_model, weight = load_w2v_model(w2v_model100_path)
            # 变量向量化
            x_train_vec0 = self.words2vec_w2v(x_train_0, w2v_model, max_len_syn)
            x_test_vec0 = self.words2vec_w2v(x_test_0, w2v_model, max_len_syn)
            x_train_vec1 = self.words2vec_w2v(x_train_1, w2v_model, max_len)
            x_test_vec1 = self.words2vec_w2v(x_test_1, w2v_model, max_len)
            model = myModel().build_model(embedding_matrix = weight)
            self.model_train(model, [x_train_vec0,x_train_vec1], np.array(y_train), dataset_name)
            # 预测结果并评估
            y_predict = model.predict([x_test_vec0,x_test_vec1])
            evaluate_print(y_test, y_predict)
        else:
            for i in range(3):
                train_data = []
                dataset_name = data_path.split('\\')[-1].replace(".txt", "")
                # 不需要预处理
                if need_pre == False:
                    dataset_path = "{}/{}_pre{}.txt".format(pre_data_path, dataset_name, i)
                    train_data = self.read_preprocessed_feature(dataset_path)
                # 需要预处理
                else:
                    train_data = preprocess(combined_data[i], verboses[i])
                # 判断是否需要平衡数据
                if balanced == True:
                    b = Balanced()
                    train_data, labels = b.overSampling(train_data, labels)
                train_filename = "{}/{}_{}.txt".format(train_data_path, dataset_name, i)
                test_filename = "{}/{}_{}.txt".format(test_data_path, dataset_name, i)

                # 判断是否划分过数据集
                if splited == False:
                    # 划分数据集
                    x_train, y_train, x_test, y_test = train_test_builder(train_data, labels, train_rate)
                    # 存储数据到文件夹内
                    self.write_txt(train_filename, x_train, y_train)
                    self.write_txt(test_filename, x_test, y_test)
                # 划分过，则从文件夹直接导入
                else:
                    x_train = []
                    y_train = []
                    x_test = []
                    y_test = []
                    self.read_txt(train_filename, x_train, y_train)
                    self.read_txt(test_filename, x_test, y_test)
                # 导入模型权重
                w2v_model, weight = load_w2v_model(w2v_model100_path)
                # 向量化
                x_train_vec = self.words2vec_w2v(x_train, w2v_model, max_len)
                x_test_vec = self.words2vec_w2v(x_test, w2v_model, max_len)
                model = []
                # 根据所选参数创建模型
                if c_model == 0:
                    model = TextCNN().build_model(embeding_matrix=weight)
                elif c_model == 1:
                    model = my_han().build_model(embeding_matrix=weight)
                # print(maxlen)
                self.model_train(model, x_train_vec, np.array(y_train), dataset_name)
                # 预测结果并评估
                y_predict = model.predict(x_test_vec)
                evaluate_print(y_test, y_predict)

    #多分类log
    def DL_train_log(self, data_path):
        dataset_name = data_path.split('\\')[-1].replace(".txt", "")
        train_filename_0 = "{}/{}_{}.txt".format(train_data_path2, dataset_name, 0)
        train_filename_1 = "{}/{}_{}.txt".format(train_data_path2, dataset_name, 1)
        x_train_0 = []
        x_train_1 = []
        y_train = []
        self.read_txt(train_filename_0, x_train_0, y_train)
        self.read_txt(train_filename_1, x_train_1, [])
        # 导入模型权重
        w2v_model, weight = load_w2v_model(w2v_model100_path)
        # 向量化
        x_train_vec0 = self.words2vec_w2v(x_train_0, w2v_model, max_len_syn)
        x_train_vec1 = self.words2vec_w2v(x_train_1, w2v_model, max_len)
        model = myModel_log().build_model(embedding_matrix=weight)
        self.model_train(model, [x_train_vec0, x_train_vec1], tf.one_hot(np.array(y_train), depth=6), dataset_name)

    #读取
    def read_txt(self,filename,data,label):
        with open(filename, "r", encoding="utf-8") as f:
            data_tol = f.read()
            datas = data_tol.split('\n')
            for i in range(int((len(datas) - 1) / 2)):
                data.append(datas[2 * i])
                label.append(int(datas[2 * i + 1]))
            f.close()

    #写入
    def write_txt(self,filename,data,label):
        with open(filename, "w", encoding="utf-8") as f:
            for i in range(len(data)):
                f.write(data[i] + "\n")
                f.write(str(label[i]))
                f.write('\n')
            f.close()

    # w2v向量化01
    def words2vec_w2v(self, data, w2v_model, max_len):
        #依据模型的参数，进行向量化操作，对短的向量进行补长，对长的向量进行裁剪
        key2index = w2v_model.wv.key_to_index
        word2v = np.zeros((len(data), max_len))
        for i in range(len(data)):
            words = data[i].split(" ")
            #先判断向量长度是否越界
            #不越界则正常补长
            if len(words) <= max_len:
                for j in range(len(words)):
                    word2v[i, j] = key2index[words[j]]
            #否则从后往前随机裁剪
            else:
                dis = len(words) - max_len
                random_len = random.randint(0, int(dis / 4))
                start_pos = max(dis - random_len,0)
                end_pos = len(words) - random_len
                #遍历所有文件
                for j in range(start_pos,end_pos):
                    word2v[i, j - (dis - random_len)] = key2index[words[j]]
        return word2v


    #训练的回调函数
    def model_callbacks(self, model, name, early_stopping = True):
        callback_list = list()
        #模型名称
        output_path = "saved_models/%s-{epoch:03d}-{val_loss:.3f}.h5"%(name)
        #模型可视化
        callback_list.append(ModelCheckpoint(output_path, monitor='val_loss',save_best_only= False, save_weights_only= True,
                                             verbose= 1 ,mode = 'min'))
        callback_list.append(CSVLogger("checkpoint\\train.log", append = True))
        #tensorboard
        callback_list.append(TensorBoard(log_dir="checkpoint\\logs",histogram_freq= 1))
        #学习率调整
        callback_list.append(LearningRateScheduler(partial(self.step_decay, intial_lr = lr_rate, drop_lr = drop_rate,
                                                           epoch_drop = lr_patience)))
        #输出结果
        if early_stopping:
            callback_list.append(EarlyStopping(verbose= 1, patience= early_stopping_patience))

        return callback_list

    # 学习率下降迭代函数
    def step_decay(self, epoch, intial_lr, drop_lr, epoch_drop):
        return intial_lr * math.pow(drop_lr,(epoch // epoch_drop))

    # 训练函数
    def model_train(self, model, x_train, y_train, name):
        # 设置训练参数
        model.fit(x_train,
                  y_train,
                  batch_size = batch_size,
                  epochs = epochs,
                  validation_split= val_rate,
                  callbacks= self.model_callbacks(
                      model = model,
                      name = name
                  ))

    # 读取预处理特征,返回两重列表
    def read_preprocessed_feature(self, feature_path):
        with open(feature_path, "r",encoding= 'utf-8') as f:
            total_data = f.read()
            datas = total_data.split("\n")
            features = []
            for i in range(len(datas) - 1):
                    features.append(datas[i])
            f.close()
            return features

    # bow读取字典进行单词映射
    def words2dict_bow(self):
        words = read_wordbag("word_bags.txt")#返回的是列表
        word2num = dict()
        num2word = dict()
        for i in range(len(words)):
            word2num[words[i]] = i + 1
            num2word[i + 1] = words[i]
        return word2num, num2word

    # bow向量化
    def words2vec_bow(self,words_list, word2num, maxlen):
        vec = np.zeros((len(words_list),maxlen))
        for i in range(len(words_list)):
            words = words_list[i].split(" ")
            for j in range(len(words)):
                vec[i, j] = word2num[words[j]]
        return vec

    # 单词-整数映射
    def pad_sentence(self, tokenizer, max_length, docs):
        encoded = tokenizer.texts_to_sequences(docs)
        padded = pad_sequences(encoded, maxlen=max_length, padding='post')
        return padded

    # 单词整数映射
    def word2Token(self,code_txt):
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(code_txt)
        return tokenizer


def train_dl(txt_path):
    # 训练
    test = DL()
    # test.DL_train_w2v(txt_path)
    test.DL_train_log(txt_path)