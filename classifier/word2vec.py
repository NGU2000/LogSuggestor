from gensim.models import Word2Vec
import numpy as np
from configuration import *
import os
import sys

def load_w2v_model(model_path):
    curPath = os.path.abspath(os.path.dirname(__file__))
    rootPath = os.path.split(curPath)[0]
    sys.path.append(rootPath)
    model = Word2Vec.load(model_path)
    model.init_sims(replace=True)
    vocab = model.wv.key_to_index
    embeding_matrix = np.zeros((len(vocab) + 1, 100))
    for word, i in vocab.items():
        try:
            embeding_vector = model.wv[str(word)]
            embeding_matrix[i] = embeding_vector
        except KeyError:
            continue
    # print(embeding_matrix)
    return model, embeding_matrix


# #读取所有分词句函数
# def read_sentences(datasets_path):
#     #存储所有句子给word2vec备用
#     total_data = []
#     paths = os.listdir(datasets_path)
#     #遍历所有文件
#     for filename in paths:
#         filepath = os.path.join(datasets_path, filename)
#         #打开文件进行遍历和记录
#         with open(filepath, "r", encoding="utf-8") as f:
#             # 读取结果
#             all_data = f.read()
#             sentences = all_data.split("\n")
#             # 分割词汇
#             for sentence in sentences:
#                 words = sentence.split(" ")
#                 total_data.append(words)
#             f.close()
#     return total_data
#
# # 训练word2vec模型
# def train_word2vec(data_path):
#     # 读取sentences
#     sentences = read_sentences(data_path)
#     # 训练模型，词向量的长度设置为100 ， 迭代次数为8# ，采用skip-gram模型# ,采用负采样# 窗口选择6# 最小词频是1# ，模型保存为pkl格式
#     w2v_model = Word2Vec(sentences, vector_size=100, epochs=20, sg=0, window=5, min_count=1)
#     w2v_model.save(w2v_model100_path)




# _,t =load_w2v_model("../word2vec_100.model")
# print(1)

# from builtins import bytes, range
#
# import pandas as pd
#
# pd.options.mode.chained_assignment = None
# from sklearn.manifold import TSNE
# import gensim
# import matplotlib.pyplot as plt
# from matplotlib.font_manager import FontProperties
#
# font = FontProperties(fname="himalaya.ttf", size=20)
#
#
# def tsne_plot(model, words_num):
#     labels = []
#     tokens = []
#     for word in model.wv.key_to_index:
#         tokens.append(model.wv[word])
#         labels.append(word)
#
#     tsne_model = TSNE(perplexity=30, n_components=2, init='pca', n_iter=1000, random_state=23)
#     new_values = tsne_model.fit_transform(tokens)
#     x = []
#     y = []
#     for value in new_values:
#         x.append(value[0])
#         y.append(value[1])
#     plt.figure(figsize=(10, 10))
#     for i in range(100,words_num):
#         plt.scatter(x[i], y[i])
#         if b'\xe0' in bytes(labels[i], encoding="utf-8"):
#             this_font = font
#         else:
#             this_font = 'SimHei'
#         plt.annotate(labels[i],
#                      Fontproperties=this_font,
#                      xy=(x[i], y[i]),
#                      xytext=(5, 2),
#                      textcoords='offset points',
#                      ha='right',
#                      va='bottom')
#     plt.show()
#
#
# if __name__ == '__main__':
#     model = gensim.models.Word2Vec.load("../word2vec_100.model")
#     # print(f'There are {len(model.wv.index2word)} words in vocab')
#     word_num = int(input('please input how many words you want to plot:'))
#     tsne_plot(model, word_num)