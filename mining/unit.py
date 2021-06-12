import shutil

import os

from only_web.cluster.text_analysis_tools.api.text_cluster.dbscan import DbscanClustering
import time

# from PyQt5.QtWidgets import QFileDialog

# from .Vocab import SecondVocab, ThirdVocab
# import matplotlib.pyplot as plt

def get_all_info(triple_info):
    text_all, aspect_all, opinion_all, triple_all, sen_polarity_all, target_all = [], [], [], [], [], []

    for i, single_info in enumerate(triple_info):
        text = str(i + 1) + ' ' + single_info['text']
        aspect = single_info['aspect']
        opinion = single_info['opinion']
        triples = single_info['triples']
        sen_polarity = single_info['sen_polarity']
        target = single_info['target']

        text_all.append(text)
        aspect_all.append(aspect)
        opinion_all.append(opinion)
        triple_all.append(triples)
        sen_polarity_all.append(sen_polarity)
        target_all.append(target)

    return text_all, aspect_all, opinion_all, triple_all, sen_polarity_all, target_all


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
        if (second_name, third_name) not in third_list:
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


def create_list(first_name):
    create = locals()
    last_name = {}
    for name in first_name:
        create[name] = list()

    for idx in create:
        if isinstance(create[idx], list):
            last_name[idx] = create[idx]
    return last_name


def read_first_data(file):
    second_name = set()
    second_name_list = set()
    all_data = []
    with open(file, encoding='utf-8') as f:
        for line in f.readlines():
            if line != '\n':
                line = line.strip()
                line = line.split()
                line = line[1:-1]
                for idx in range(0, len(line), 3):
                    second_name.add(line[idx + 1][0])
    vocab = SecondVocab()
    for name in second_name:
        second_name_list.add(vocab.word2id(name))

    last_name = create_list(second_name_list)

    with open(file, encoding='utf-8') as f:
        for line in f.readlines():
            if line != '\n':
                line = line.strip()
                line = line.split()
                line = line[0:-1]
                for idx in range(0, len(line), 3):
                    last_name[vocab.word2id(line[idx + 2][0])].append([line[idx], line[idx + 2], line[idx + 1]])

    last_second_data = make_last(last_name)

    for aspect in last_name:
        third_aspect = read_aspect_third_data(last_name[aspect])
        all_data.append([aspect, third_aspect])

    return last_second_data, all_data


def make_last(last_name):
    last_second_data = []
    for name in last_name:
        bad, neu, good = 0, 0, 0
        for units in last_name[name]:
            if units[-1] == str(0):
                bad += 1
            elif units[-1] == str(1):
                neu += 1
            elif units[-1] == str(2):
                good += 1
        last_second_data.append([name, bad, neu, good])
    return last_second_data


def read_second_data(file):
    third_name = set()
    third_name_list = set()
    with open(file, encoding='utf-8') as f:
        for line in f.readlines():
            if line != '\n':
                line = line.strip()
                line = line.split(' ')
                line = line[1:-1]
                for idx in range(0, len(line), 3):
                    third_name.add(line[idx + 1])
    vocab = ThirdVocab()
    for name in third_name:
        third_name_list.add(vocab.word2id(name))

    last_name = create_list(third_name_list)

    with open(file, encoding='utf-8') as f:
        for line in f.readlines():
            if line != '\n':
                line = line.strip()
                line = line.split(' ')
                line = line[1:-1]
                for idx in range(0, len(line), 3):
                    last_name[vocab.word2id(line[idx + 1])].append([line[idx], line[idx + 1], line[idx + 2]])
    last_third_data = make_last(last_name)

    return last_third_data


def create_father_item(data):
    good = 0
    bad = 0
    for line in data:
        good += line[3]
        bad += line[1]
    bad1 = bad / (bad + good)
    good1 = good / (bad + good)
    return {'positive': good1, 'negative': bad1}


def create_all_item(data):
    result = {}
    for inspects in data:
        father_name = inspects[0]
        node = {}
        for inspect in inspects[1]:
            son_name = inspect[1]
            bad = int(inspect[2])
            good = int(inspect[4])
            if bad + good == 0:
                good = 1
                bad = 1
            bad1 = bad / (bad + good)
            good1 = good / (bad + good)
            node[son_name] = {'positive': good1, 'negative': bad1}
        result[father_name] = node
    return result


def create_item(data):
    result = {}
    for inspect in data:
        name = inspect[0]
        bad = int(inspect[1])
        good = int(inspect[3])
        if bad + good == 0:
            good = 1
            bad = 1
        bad1 = bad / (bad + good)
        good1 = good / (bad + good)
        result[name] = {'positive': good1, 'negative': bad1}
    return result


def read_aspect_third_data(file):
    third_name = set()
    third_name_list = set()
    for line in file:
        for idx in range(0, len(line), 3):
            third_name.add(line[idx + 1])
    vocab2 = SecondVocab()
    vocab3 = ThirdVocab()
    for name in third_name:
        third_name_list.add(vocab3.word2id(name))

    last_name = create_list(third_name_list)

    for line in file:
        for idx in range(0, len(line), 3):
            last_name[vocab3.word2id(line[idx + 1])].append([vocab2.word2id(line[idx + 1][0]), line[idx], line[idx + 1],
                                                             line[idx + 2]])

    last_third_data = make_aspect_last(last_name)

    return last_third_data


def make_aspect_last(last_name):
    last_second_data = []
    second_name = ''
    for name in last_name:
        bad, neu, good = 0, 0, 0
        for units in last_name[name]:
            second_name = units[0]
            if units[-1] == str(0):
                bad += 1
            elif units[-1] == str(1):
                neu += 1
            elif units[-1] == str(2):
                good += 1
        last_second_data.append([second_name, name, bad, neu, good])
    return last_second_data


def extract_info1(path):
    insts = []
    with open(path, 'r', encoding='utf8') as f:
        inst = []
        for line in f.readlines():
            line = line.strip()
            if len(line) == 0 and len(inst) != 0:
                insts.append(inst)
                inst = []
            elif len(line) == 1:
                inst.append(line)
            else:
                line = line.split()
                line[1] = line[1].split('-')
                inst.append(line)
        if len(inst) != 0:
            insts.append(inst)

    info_list = []
    with open('info.txt', 'w', encoding='utf8') as f:
        for inst in insts:
            # 及 O
            # 周 B - a1 - 2 13
            # 边 I - a1 - 2 13
            # 设 I - a1 - 2 13
            # 施 E - a1 - 2 13
            # 完 B - e0 - 2 - a0 - a1
            # 善 E - e0 - 2 - a0 - a1
            sent_info = []
            e_info = []
            info = []
            e = []

            for elem in inst:
                if len(elem) == 1 or len(elem[1]) == 1:
                    continue
                if 'a' in elem[1][1] and len(elem) == 3:
                    if elem[1][0] == 'S':
                        info.append(elem[0])  # 周
                        info.append(elem[1][1])  # a1
                        info.append([])  # e0
                        info.append(elem[1][2])  # 2
                        info.append(elem[2])  # 13
                        info.append(inst[-1])  # 1
                        sent_info.append(info)
                        info = []
                    elif elem[1][0] == 'B':
                        info.append(elem[0])  # 周
                        info.append(elem[1][1])  # a1
                        info.append([])  # e0
                        info.append(elem[1][2])  # 2
                        info.append(elem[2])  # 13
                        info.append(inst[-1])  # 1
                    elif elem[1][0] == 'I':
                        info[0] += elem[0]  # 周
                    elif elem[1][0] == 'E':
                        info[0] += elem[0]  # 周
                        sent_info.append(info)
                        info = []
                elif 'exp' in elem[1][1] and len(elem[1]) > 3:
                    if elem[1][0] == 'S':
                        e.append('exp-' + elem[1][1][1] + '@@@' + elem[0])
                        e.append(elem[1][3:])
                        e_info.append(e)
                        e = []
                    elif elem[1][0] == 'B':
                        e.append('exp-' + elem[1][1][1] + '@@@' + elem[0])
                        e.append(elem[1][3:])
                    elif elem[1][0] == 'I':
                        e[0] += elem[0]
                    elif elem[1][0] == 'E':
                        e[0] += elem[0]
                        e_info.append(e)
                        e = []
                elif 'e' in elem[1][1] and len(elem[1]) > 2:
                    if elem[1][0] == 'S':
                        e.append('e' + '@@@' + elem[0])
                        e.append(elem[1][3:])
                        e_info.append(e)
                        e = []
                    elif elem[1][0] == 'B':
                        e.append('e' + '@@@' + elem[0])
                        e.append(elem[1][3:])
                    elif elem[1][0] == 'I':
                        e[0] += elem[0]
                    elif elem[1][0] == 'E':
                        e[0] += elem[0]
                        e_info.append(e)
                        e = []

            for e in e_info:  # [['我们',[a1, a2]]...]
                a_list = e[1]
                for a in a_list:
                    for sub_info in sent_info:
                        if a in sub_info:
                            sub_info[2] = e[0]
            info_list.append(sent_info)
        for sent_info in info_list:
            if sent_info == []:
                continue
            for idx, info in enumerate(sent_info):
                if idx == 0:
                    f.write(info[0] + '\t' + info[2] + '\t' + info[3] + '\t' + info[4] + '\t' + info[5])
                else:
                    f.write(
                        '\t' + '###' + '\t' + info[0] + '\t' + info[2] + '\t' + info[3] + '\t' + info[4] + '\t' + info[
                            5])
            f.write('\n')


# def open_file():
#     path, _ = QFileDialog.getOpenFileName()
#     file = 'input_path.txt'
#     writer = open(file, encoding='utf-8', mode='w')
#     writer.write(path)
#     writer.close()


def aspect_text(text, name):
    data_bad = []
    data_good = []
    vocab3 = ThirdVocab()
    for line in text:
        line = line[0]
        if vocab3.word2id(line[0][1]) == name and line[3] == "0":
            data_bad.append(line)
        elif vocab3.word2id(line[0][1]) == name and line[3] == "2":
            data_good.append(line)
    FGSA_str_bad = ['(<' + gfsa[0][0] + '>, ' + gfsa[1] + ', <' + gfsa[2] + '>, ' + str(gfsa[3]) + ')' for gfsa in
                    data_bad]
    FGSA_str_good = ['(<' + gfsa[0][0] + '>, ' + gfsa[1] + ', <' + gfsa[2] + '>, ' + str(gfsa[3]) + ')' for gfsa in
                     data_good]
    return FGSA_str_good, FGSA_str_bad


def extract_info(path, path1):
    insts = []
    with open(path, 'r', encoding='utf8') as f:
        inst = []
        for line in f.readlines():
            line = line.strip()
            if len(line) == 0 and len(inst) != 0:
                insts.append(inst)
                inst = []
            elif len(line) == 1:
                inst.append(line)
            else:
                line = line.split()
                line[1] = line[1].split('-')
                inst.append(line)
        if len(inst) != 0:
            insts.append(inst)

    ### extract a e exp
    a_e_exp_list = []  # [(我们, a), (xx, e), (x, exp-fac)..]
    for inst in insts:
        a_e_exp_sent = []
        word = ''
        label = ''
        for line in inst:
            if len(line) == 1:
                a_e_exp_list.append(a_e_exp_sent)
            else:
                if line[1][0] == 'S':
                    word = line[0]
                    label = line[1][1]
                    if 'exp' in label:
                        label = '-'.join(line[1][1:3])
                    elif 'e' in label:
                        label = 'e'
                    elif 'a' in label:
                        label = 'a'
                    a_e_exp_sent.append((word, label))
                elif line[1][0] == 'B':
                    word = line[0]
                    label = line[1][1]
                    if 'exp' in label:
                        label = '-'.join(line[1][1:3])
                    elif 'e' in label:
                        label = 'e'
                    elif 'a' in label:
                        label = 'a'
                elif line[1][0] == 'I':
                    word += line[0]
                elif line[1][0] == 'E':
                    word += line[0]
                    a_e_exp_sent.append((word, label))
        # a_e_exp_list.append(a_e_exp_sent)
    ###

    info_list = []
    with open('info.txt', 'w', encoding='utf8') as f:
        for inst in insts:
            # 及 O
            # 周 B - a1 - 2 13
            # 边 I - a1 - 2 13
            # 设 I - a1 - 2 13
            # 施 E - a1 - 2 13
            # 完 B - e0 - 2 - a0 - a1
            # 善 E - e0 - 2 - a0 - a1
            sent_info = []
            e_info = []
            info = []
            e = []
            for elem in inst:
                if len(elem) == 1 or len(elem[1]) == 1:
                    continue

                if 'a' in elem[1][1] and len(elem) == 3:
                    if elem[1][0] == 'S':
                        info.append(elem[0])  # 周
                        info.append(elem[1][1])  # a1
                        info.append([])  # e0
                        info.append(elem[1][2])  # 2
                        info.append(elem[2])  # 13
                        info.append(inst[-1])  # 1
                        sent_info.append(info)
                        info = []
                    elif elem[1][0] == 'B':
                        info.append(elem[0])  # 周
                        info.append(elem[1][1])  # a1
                        info.append([])  # e0
                        info.append(elem[1][2])  # 2
                        info.append(elem[2])  # 13
                        info.append(inst[-1])  # 1
                    elif elem[1][0] == 'I':
                        info[0] += elem[0]  # 周
                    elif elem[1][0] == 'E':
                        info[0] += elem[0]  # 周
                        sent_info.append(info)
                        info = []
                elif 'exp' in elem[1][1] and len(elem[1]) > 3:
                    if elem[1][0] == 'S':
                        e.append('exp-' + elem[1][1][1] + '@@@' + elem[0])
                        e.append(elem[1][3:])
                        e_info.append(e)
                        e = []
                    elif elem[1][0] == 'B':
                        e.append('exp-' + elem[1][1][1] + '@@@' + elem[0])
                        e.append(elem[1][3:])
                    elif elem[1][0] == 'I':
                        e[0] += elem[0]
                    elif elem[1][0] == 'E':
                        e[0] += elem[0]
                        e_info.append(e)
                        e = []
                elif 'e' in elem[1][1] and len(elem[1]) > 2:
                    if elem[1][0] == 'S':
                        e.append('e' + '@@@' + elem[0])
                        e.append(elem[1][3:])
                        e_info.append(e)
                        e = []
                    elif elem[1][0] == 'B':
                        e.append('e' + '@@@' + elem[0])
                        e.append(elem[1][3:])
                    elif elem[1][0] == 'I':
                        e[0] += elem[0]
                    elif elem[1][0] == 'E':
                        e[0] += elem[0]
                        e_info.append(e)
                        e = []

            for e in e_info:  # [['我们',[a1, a2]]...]
                a_list = e[1]
                for a in a_list:
                    for sub_info in sent_info:
                        if a in sub_info:
                            sub_info[2] = e[0]
            info_list.append(sent_info)

        with open(path1, 'w', encoding='utf8') as f1:
            for sent_info in info_list:
                if sent_info == []:
                    continue
                for idx, info in enumerate(sent_info):
                    if idx == 0:
                        f.write(info[0] + '\t' + info[2] + '\t' + info[3] + '\t' + info[4] + '\t' + info[5])
                        f1.write(info[0] + '\t' + info[3] + '\t' + info[4])
                    else:
                        f.write(
                            '\t' + '###' + '\t' + info[0] + '\t' + info[2] + '\t' + info[3] + '\t' + info[4] + '\t' +
                            info[5])
                        f1.write('\t' + info[0] + '\t' + info[3] + '\t' + info[4])
                f1.write('\t' + info[5])
                f.write('\n')
                f1.write('\n')
    return a_e_exp_list
