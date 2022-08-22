#代码文本处理主函数
import os
from ml_train_test import ML
from test import Test
from wordbag_builder import build_sentences
from classifier.word2vec import train_word2vec
from dl_train import train_dl
from utils.util import  settings
from configuration import *
from test_func import  *

def main(verbose = [0,0,0,0,1]):
    """
    主函数
    sampling_choice：选择采样策略
    处理顺序：
    """
    #设置项目参数
    settings()

    # 预处理
    if verbose[0] == 1:
        # 创建预处理的句表
        build_sentences(raw_data_path)

    # 训练w2v模型
    if verbose[1] == 1:
        train_word2vec(pre_data_path)

    #训练深度学习模型
    if verbose[2] == 1:
        #遍历所有预处理好的数据进行训练
        paths = os.listdir(raw_data_path2)
        for i in paths:
            filename = os.path.join(raw_data_path2,i)
            train_dl(filename)

    #训练机器学习模型
    if verbose[3] == 1:
        # 遍历所有预处理好的数据进行训练
        paths = os.listdir(raw_data_path)
        for i in paths:
            filename = os.path.join(raw_data_path, i)
            ML().ML_train(filename)

    #测试
    if verbose[4] == 1:
        # 进行深度学习和机器模型测试和评估
        # 遍历所有预处理好的数据进行测试
        paths = os.listdir(raw_data_path)
        s = 1
        #遍历数据集
        for dataset in paths:
            filename = os.path.join(raw_data_path, dataset)
            if s == 1 and filename.count('Cassandra') > 0:
                s = 0
            else:
                continue
            ML().ML_test(filename)
            # dataset_name = dataset.replace(".txt","")
            # #遍历模型
            # model_name = os.listdir("saved_models")
            # #模型地址
            # for model in model_name:
            #     models = os.path.join("saved_models",model)
            #     model_loss = os.listdir(models)
                #遍历损失
                # for loss in model_loss:
                #     if model != "mymodel":
                        # loss_path = os.path.join(models, loss)
                        # data_012s = os.listdir(loss_path)
                        # # 遍历数据集012
                        # for data_012 in data_012s:
                        #     dataset012 = os.path.join(loss_path, data_012)
                        #     model_names = os.listdir(dataset012)
                        #     # 遍历所有模型
                        #     for model_total in model_names:
                        #         if model_total.count(dataset_name) > 0:
                        #             print(os.path.join(dataset012, model_total))
                        #             if model.count('text_cnn') > 0:
                        #                 Test().dl_test(filename, os.path.join(dataset012, model_total), 0)
                        #             elif model.count('HAN') > 0:
                        #                 Test().dl_test(filename, os.path.join(dataset012, model_total), 1)


#运行主函数
if __name__ == "__main__":
    main()

    # settings()

    # level_train()
    # level_test("saved_models\\mylogmodel")


    # root_model_path = "saved_models\\mymodel\\cb_loss"
    # paths = os.listdir(root_model_path)
    # datasets = ['Cassandra',
    #             'ElasticSearch',
    #             'Flink',
    #             'Hadoop',
    #             'HBase',
    #             'Kafka',
    #             'Tomcat',
    #             'Wicket',
    #             'Zookeeper'
    #             ]
    #
    # for dataset in datasets:
    #     another = datasets.copy()
        # for i in paths:
        #     if i.count('HBase') == 0:
        #         continue
        #     if i.count(dataset) > 0:
        #         print(i)
        #         for j in another:
        #             print(j)
        #             Test().dl_test("Dataset\\raw\\{}.txt".format(j), os.path.join(root_model_path, i),2)

    # dataset_name = ["Cassandra","ElasticSearch","Flink","HBase","Kafka","Tomcat","Wicket","Zookeeper","Hadoop"]
    # for dataset in dataset_name:
    #     for i in range(3):
    #         model_path = "saved_models\\HAN\\cb_loss\\{}".format(i)
    #         for path in os.listdir(model_path):
    #             if path.count(dataset) > 0:
    #                 print(path)
    #                 Test().dl_test("Dataset\\raw\\{}.txt".format(dataset), os.path.join(model_path, path), 1)
