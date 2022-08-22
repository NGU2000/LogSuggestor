#预处理
from tqdm import tqdm
from configuration import java_keywords
from utils.PorterStem import PorterStemmer
from nltk.stem import SnowballStemmer,PorterStemmer,LancasterStemmer
from nltk.corpus import stopwords
from spiral import ronin,samurai
import time
import re

# 洗词
def clean_words(words):
    """
    1、删除所有符号
    2、将所有数字用特殊符号替换
    """
    punctuation_regx = "[{().+\\-<>=;:!?{}&|*%/\\\\'\"\\[\\]~;$_@,#}]"
    number_regx = r'[1-9]+\.?[0-9]*'
    del_punc = re.sub(r'[{}]+'.format(punctuation_regx), ' ', words).strip()
    cleaned_words = re.sub(number_regx, '@',del_punc).strip()
    return  cleaned_words

# 词干提取
def stem_words(code_str, verbose = 0):
    stemmer = ""
    if verbose == 0:
        stemmer = SnowballStemmer('english')
    elif verbose == 1:
        stemmer = PorterStemmer()
        return stemmer.stem(code_str)
    elif verbose == 2:
        stemmer = LancasterStemmer()
    return stemmer.stem(code_str)

# 分词
def split_words(code_str,verbose = 0):
    words = []
    if verbose == 0:
        words = ronin.split(code_str)
    elif verbose == 1:
        words = samurai.split(code_str)
    return words

#超时判断
def txt_deal(code_txt,verbose,stop_words):
    combined = ""
    # 如果处理文本，则进行分词和停词处理
    if verbose == 0:
        # 分词
        splited = split_words(code_txt, 0)
        # print("分词")
        # 转为小写
        for i in range(len(splited)):
            splited[i] = splited[i].lower()
        # 去停词
        file_words = [word for word in splited if word not in stop_words]
        # print("去停词")
        # 连接结果
        for strtmp in file_words:
            combined = combined + stem_words(strtmp,1) + " "
        # print("连接")
    else:
        # 分词
        splited = code_txt.split(" ")
        for i in splited:
            combined = combined + stem_words(i, 1) + " "
        combined = code_txt.lower()
    return combined

#预处理主函数
def preprocess(data, verbose = 0):
    """
    verbose = 0:进行分词、小写和去停词操作
    verbose = 1:不进行分词、小写和去停词操作
    """
    data_copy = data.copy()
    # 设置停词表
    stop_words = stopwords.words('english')
    # 停词表内加入关键字,符号和数字已经提前处理
    for i in java_keywords:
        stop_words.append(i)
    #存储最终结果
    x_data_true = []
    # 记录时间
    start_time = time.time()
    index = 0
    # 显示处理进度
    with tqdm(total = len(data_copy), leave = True, ncols = 100, desc = "Analyzing", unit_scale = True) as pbar:
        for code_txt in data_copy:
            # 如果不为字符串则转为字符串
            if isinstance(code_txt, str) == False:
                code_txt = str(code_txt)
            index += 1
            # print(index)
            # 更新处理进度
            pbar.update(1)
            # 洗词在java中进行过就不用做了
            if code_txt == None:
                continue
            combined = txt_deal(code_txt, verbose, stop_words)
            x_data_true.append(combined)
    end_time = time.time()
    print("数据处理耗时{:.2f}s".format(end_time - start_time))
    return x_data_true