import shutil

import os

from only_web.cluster.text_analysis_tools.api.text_cluster.dbscan import DbscanClustering
import time

# from PyQt5.QtWidgets import QFileDialog

# from .Vocab import SecondVocab, ThirdVocab
# import matplotlib.pyplot as plt

def get_all_info(triple_info):
    text_all, aspect_all, opinion_all, triple_all, sen_polarity_all, target_all ,express_all= [], [], [], [], [], [] , []

    for i, single_info in enumerate(triple_info):
        text = str(i + 1) + ' ' + single_info['text']
        # aspect = single_info['aspect']
        # opinion = single_info['opinion']
        triples = single_info['triples']
        sen_polarity = single_info['sen_polarity']
        target = single_info['target']
        # express = single_info['express']

        text_all.append(text)
        # aspect_all.append(aspect)
        # opinion_all.append(opinion)
        triple_all.append(triples)
        sen_polarity_all.append(sen_polarity)
        target_all.append(target)
        # express_all.append(express)

    return text_all, triple_all, sen_polarity_all, target_all


def get_a_and_o(aspect_all, opinion_all):
    res = []
    assert len(aspect_all) == len(opinion_all)

    for i in range(len(aspect_all)):
        single_info = str(i + 1) + ' ' + '实体: '

        for aspect in aspect_all[i]:
            if aspect != aspect_all[i][-1]:
                add_board = '<' + aspect + '>' + '、'
            else:
                add_board = '<' + aspect + '>' + ' '
            single_info += add_board

        single_info += '观点: '
        for opinion in opinion_all[i]:

            if opinion != opinion_all[i][-1]:
                add_board = '<' + opinion + '>' + '、'
            else:
                add_board = '<' + opinion + '>' + ' '
            single_info += add_board

        res.append(single_info)

    return res


def get_RE_tri(triple_all):
    ao_pair, ao_tri = [], []

    for i, triplets in enumerate(triple_all):
        single_info_pair = str(i + 1) + ': '
        single_info_tri = str(i + 1) + ': '
        for triplet in triplets:
            aspect, opinion, polarity = triplet
            if triplet != triplets[-1]:
                single_info_pair += '(' + '<' + aspect + '>' + ', ' + '<' + opinion + '>' + ')' + '、'
                single_info_tri += '(' + '<' + aspect + '>' + ', ' + '<' + opinion + '>' + ', ' + polarity + ')' + '、'
            else:
                single_info_pair += '(' + '<' + aspect + '>' + ', ' + '<' + opinion + '>' + ')'
                single_info_tri += '(' + '<' + aspect + '>' + ', ' + '<' + opinion + '>' + ', ' + polarity + ')'

        ao_pair.append(single_info_pair)
        ao_tri.append(single_info_tri)
    return ao_pair, ao_tri


def get_target_info(target_all, SecondVocab, ThirdVocab):
    # target:(ap,second_name,third_name,polarity,(ap,op,polarity))
    all_target = []
    for one_sentence in target_all:
        for target in one_sentence:
            all_target.append(target)

    total_POS, total_NEG = 0, 0
    second_info = {}
    second_list = []
    third_list = []
    third_info = {}
    for single_info in all_target:
        ap, second_name, third_name, polarity, tri = single_info
        second_name = SecondVocab.w2i[second_name]
        third_name = ThirdVocab.w2i[third_name]
        if second_name not in second_list:
            second_list.append(second_name)
        if (second_name, third_name) not in third_list and polarity!='NEU':
            third_list.append((second_name, third_name))
        if polarity == 'POS':
            total_POS += 1
            second_info[second_name + 'POS'] = second_info.get((second_name + 'POS'), 0) + 1
            _ = third_info.setdefault((third_name + 'POS'), [])
            third_info[third_name + 'POS'].append((polarity, tri))
        if polarity == 'NEG':
            total_NEG += 1
            second_info[second_name + 'NEG'] = second_info.get((second_name + 'NEG'), 0) + 1
            _ = third_info.setdefault((third_name + 'NEG'), [])
            third_info[third_name + 'NEG'].append((polarity, tri))

    # 总的情感极性比例
    total_p, total_n = compute_prop(total_POS, total_NEG)
    chart1 = {}
    chart1['positive'] = total_p
    chart1['negative'] = total_n

    # second类情感比例
    chart2 = {}

    for second_name in second_list:
        pos = second_info.setdefault((second_name + 'POS'), 0)
        neg = second_info.setdefault((second_name + 'NEG'), 0)
        p, n = compute_prop(pos, neg)
        chart2[second_name] = {}
        chart2[second_name]['positive'] = p
        chart2[second_name]['negative'] = n

    # third类处理
    chart3 = {}

    for second_name in second_list:
        chart3[second_name] = {}

    for name in third_list:
        third_pos, third_neg = 0, 0
        second_name, third_name = name
        pos_list = third_info.setdefault(third_name + 'POS', [])
        neg_list = third_info.setdefault(third_name + 'NEG', [])
        for sample in pos_list:
            if sample:
                polarity, tri = sample
                third_pos += 1
            else:
                tri = ''
            _ = chart3[second_name].setdefault(third_name, {})
            _ = chart3[second_name][third_name].setdefault('positive_instances', [])
            _ = chart3[second_name][third_name].setdefault('negative_instances', [])
            chart3[second_name][third_name]['positive_instances'].append(tri)

        for sample in neg_list:
            if sample:
                polarity, tri = sample
                third_neg += 1
            else:
                tri = ''
            _ = chart3[second_name].setdefault(third_name, {})
            _ = chart3[second_name][third_name].setdefault('negative_instances', [])
            _ = chart3[second_name][third_name].setdefault('positive_instances', [])
            chart3[second_name][third_name]['negative_instances'].append(tri)

        p, n = compute_prop(third_pos, third_neg)
        chart3[second_name][third_name]['positive'] = p
        chart3[second_name][third_name]['negative'] = n

    start_cluster_time = time.time()
    # 对每个小类中的aspect进行聚类，把相同内容显示到一块
    for second_name in chart3:
        second_total = 0
        third_len_list = []
        for third_name in chart3[second_name]:
            pos_aspect, neg_aspect = [], []  # 聚类的输入
            pos_instances = chart3[second_name][third_name]['positive_instances']  # List<Tuple>
            neg_instances = chart3[second_name][third_name]['negative_instances']

            chart3[second_name][third_name]['positive_instances'] = []
            for instance in pos_instances:
                pos_aspect.append(instance[0])
            pos_len = len(pos_instances)
            neg_len = len(neg_instances)

            total_len = pos_len+neg_len #记录小类中所有的评价数，用于计算用户较关注哪个大类
            third_len_list.append([third_name,total_len])
            second_total+=total_len
            #start cluster
            if pos_len!=0:
                result = dbscan_cluster(pos_aspect,eps=0.05)  # 聚类结束的结果
                for key in result:
                    cluster = result[key]

                    new_aspect_list = list(map(lambda x: pos_instances[x][0], cluster))
                    new_opinion_list = list(map(lambda x: pos_instances[x][1], cluster))
                    new_aspect = max(new_aspect_list,key=new_aspect_list.count)#选择一个出现次数最多的aspect
                    new_opinion = list(set(new_opinion_list))#去除掉重复的

                    rate = (len(new_aspect_list) / (pos_len + 1e-5)) * 100
                    chart3[second_name][third_name]['positive_instances'].append([new_aspect,new_opinion,rate])

            chart3[second_name][third_name]['negative_instances'] = []
            for instance in neg_instances:
                neg_aspect.append(instance[0])

            #start cluster
            if neg_len!=0:
                result = dbscan_cluster(neg_aspect,eps=0.05)  # 聚类结束的结果
                for key in result:
                    cluster = result[key]

                    new_aspect_list = list(map(lambda x: neg_instances[x][0], cluster))
                    new_opinion_list = list(map(lambda x: neg_instances[x][1], cluster))
                    new_aspect = max(new_aspect_list,key=new_aspect_list.count)
                    new_opinion = list(set(new_opinion_list))

                    rate = (len(new_aspect_list) / (neg_len + 1e-5) )* 100
                    chart3[second_name][third_name]['negative_instances'].append([new_aspect,new_opinion,rate])

        for info in third_len_list:
            third_type,number = info
            chart3[second_name][third_type]['rate'] = (number / (second_total+1e-5))* 100

    end_cluster_time = time.time()
    print('cluster cost time %.3f'%(end_cluster_time-start_cluster_time))

    return chart1, chart2, chart3

def get_express(express_all,opinion_all):

    assert len(express_all) == len(opinion_all)
    chart4 = {}

    for i in range(len(express_all)):
        for info in express_all[i]:
            if info:
                express,type = info
                _ = chart4.setdefault(type,[])
                # chart4[type].append((opinion_all[i],express))
                chart4[type].append(express)

    return chart4

def compute_prop(pos, neg):
    p = pos / (pos + neg + 1e-5)
    n = neg / (pos + neg + 1e-5)
    return p, n


def get_text(input_path):
    text = []
    f_text = open(input_path, 'r', encoding='utf-8')
    lines = f_text.readlines()
    for line in lines:
        text.append(line.strip())
    f_text.close()
    return text


def get_batch(text, batch_size):
    batch_text = []
    if len(text) > batch_size:
        for i in range(0, len(text), batch_size):
            batch_text.append(text[i:(i + batch_size)])
    else:
        batch_text.append(text)
    return batch_text


def dbscan_cluster(data_path, eps=0.005, min_samples=0, fig=False):
    """
    基于DBSCAN进行文本聚类
    :param data_path: 文本路径，每行一条
    :param eps: DBSCA中半径参数
    :param min_samples: DBSCAN中半径eps内最小样本数目
    :param fig: 是否对降维后的样本进行画图显示，默认False
    :return: {'cluster_0': [0, 1, 2, 3, 4], 'cluster_1': [5, 6, 7, 8, 9]}   0,1,2....为文本的行号
    """
    dbscan = DbscanClustering()
    result = dbscan.dbscan(corpus_path=data_path, eps=eps, min_samples=min_samples, fig=fig)
    return result
    # print("dbscan result: {}\n".format(result))



