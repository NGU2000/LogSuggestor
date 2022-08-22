#设置参数

#训练测试比例
train_rate = 0.8
val_rate = 0.25
batch_size = 48

#原始数据文件夹地址
raw_data_path = "Dataset\\raw"
raw_data_path2 = "Dataset2\\raw"
#预处理数据文件夹地址
pre_data_path = "Dataset\\pre\\all"
pre_data_path2 = "Dataset2\\pre\\all"
#划分完的数据地址
train_data_path = "Dataset\\pre\\train"
test_data_path = "Dataset\\pre\\test"
train_data_path2 = "Dataset2\\pre\\train"
test_data_path2 = "Dataset2\\pre\\test"

#迭代次数
epochs = 100
lr_patience = 2

#前停次数
early_stopping_patience = 10

#深度学习学习率
lr_rate = 3e-4
drop_rate = 0.7

#生成词袋模型的批处理大小
wordbag_batch = 5000

#word2vec模型保存地址
w2v_model100_path = r"E:\document\project\finalPaper\code\mycode\pythonProject\word2vec_100.model"
w2v_model200_path = r"E:\document\project\finalPaper\code\mycode\pythonProject\word2vec_200.model"

#词向量最长长度
max_len = 400
max_len_syn = 100
#机器学习的长度
max_len_ml = 100
#word2vec模型长度及大小
voc_size = 10783
vec_len = 100

#Java符号
dots = ['{', '}', '/', '@', ':', '.', '"', '<', '>', '=', '-', '+', '%', '[', ']', '!', '|','(', ')', ';' , ',', '_', '%']

#Java关键字
java_keywords = [
    "abstract",
    "assert",
    "boolean",
    "break",
    "byte",
    "case",
    "catch",
    "char",
    "class",
    "const",
    "continue",
    "default",
    "do",
    "double",
    "else",
    "enum",
    "extends",
    "final",
    "finally",
    "float",
    "for",
    "goto",
    "if",
    "implements",
    "import",
    "instanceof",
    "int",
    "interface",
    "long",
    "native",
    "new",
    "package",
    "private",
    "protected",
    "public",
    "return",
    "short",
    "static",
    "strictfp",
    "super",
    "switch",
    "synchronized",
    "this",
    "throw",
    "throws",
    "transient",
    "try",
    "void",
    "volatile",
    "while",
    "null",
    "int",
    "String",
    "List",
    "ArrayList"
]