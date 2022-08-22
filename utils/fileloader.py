# 导入文件模块
import tensorflow as tf
import re
import time

def txt2list(filename):
    """
    将txt文件转为列表
    :param filename:
    :return:
    """
    print("开始文档读取")
    start_time = time.time()
    #存储标签和代码段
    syn_vec = []#语义
    semi_vec = []#句法
    fus_vec = []#混合
    labels = []#标签
    # 统计个数
    sample_num = 0
    true_num = 0
    false_num = 0
    #提取样本
    with open(filename,"r",encoding= "utf-8") as f:
        loc = f.read()
        data = loc.split('\n')
        i = 0
        total_num = len(data) - 2
        #循环提取样本信息
        while i < total_num:
            #提取label
            label = int(re.findall("(?<=Label:).+",data[i])[0])
            if label == 0:
                false_num += 1
            else:
                true_num += 1
            i += 1
            #提取代码文本
            semi = data[i][9:]
            i += 1
            while data[i][:5]!= "AstP:":
                semi += data[i]
                i += 1
                if len(data[i]) < 5:
                   semi += data[i]
                   i += 1
            #提取AST树
            syn = data[i][6:]
            i += 1
            while data[i][:5] != "FusP:":
                syn += data[i]
                i += 1
                if len(data[i]) < 5:
                    syn += data[i]
                    i += 1
            #提取混合文本
            fus = data[i][6:]
            i += 1
            while data[i][:3] != "Id:":
                fus += data[i]
                i += 1
                #判断是否到达界限值
                if i >= total_num:
                    break
                if len(data[i]) < 3:
                    fus += data[i]
                    i+= 1
            #样本输出
            labels.append(label)
            semi_vec.append(semi)
            fus_vec.append(fus)
            syn_vec.append(syn)
            sample_num += 1
    end_time = time.time()
    print("共{}个样本".format(sample_num))
    print("文档读取完成，耗时{:.3f}s".format(end_time - start_time))
    return [syn_vec,semi_vec,fus_vec], labels, [true_num,false_num]

def txt2tfrec(filename):
    """
    将txt文件读取为tfrecords文件并保存
    :param filename:
    :return:
    """

    outpath = "Dataset/{}".format(filename.split('/')[-1]).replace("txt","tfrecords")#输出路径
    with open(filename, "r",encoding = "utf-8") as f:
        #获得代码块文本
        loc = f.read()
        data = loc.split('\n')
        num = len(data)#样本数
        writer = tf.io.TFRecordWriter(outpath)
        for i in data:
            blocksmt = re.findall("(?<=blocksmt:).+(?=label:)",i)
            label = re.findall("(?<=label:).+(?=spos)",i)
            #判断退出条件
            if blocksmt == []:
                break
            feature = tf.train.Features(feature = {
                'blocksmt': tf.train.Feature(bytes_list = tf.train.BytesList(value = [blocksmt[0].encode(encoding = "utf-8")])),
                'ast': tf.train.Feature(bytes_list = tf.train.BytesList(value = [blocksmt[1].encode(encoding="utf-8")])),
                'fus': tf.train.Feature(bytes_list = tf.train.BytesList(value = [blocksmt[2].encode(encoding="utf-8")])),
                'label':tf.train.Feature(int64_list = tf.train.Int64List(value = [int(label[0][0])]))
            })
            example = tf.train.Example(features = feature)
            writer.write(example.SerializeToString())
        writer.close()

def reader(data):
    #定义格式
    feature_extraction = {
        'blocksmt': tf.io.FixedLenFeature([],tf.string),
        'ast': tf.io.FixedLenFeature([], tf.string),
        'fus': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([],tf.int64)
    }
    feature_dict = tf.io.parse_single_example(data,feature_extraction)
    blocksmt = tf.io.decode_raw(feature_dict['blocksmt'],tf.uint8)
    label = feature_dict['label']
    return blocksmt,label

def recReader(filename):
    """
    将tfrecords文件导入到程序
    :param filename:
    :return:
    """
    dataset = tf.data.TFRecordDataset(filename)
    dataset = dataset.map(reader)
    blocksmt = []
    label = []
    for block_tmp,label_tmp in dataset:
        blocksmt.append(block_tmp)
        label.append(label_tmp)
    return blocksmt,label
