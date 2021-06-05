import json
from .unit import *

data_path = 'test3.txt'
path = 'info.txt'


def getText(path):
    sent_list = []
    with open(path, 'r', encoding='utf8') as f:
        sent = []
        for line in f.readlines():
            line = line.strip()
            if len(line) == 0:
                sent_list.append(''.join(sent))
                sent = []
            elif len(line) == 1:
                continue
            else:
                line = line.split()
                sent.append(line[0])
        if len(sent) != 0:
            sent_list.append(''.join(sent))
    return sent_list


def getInfo():
    aspect_list = []
    e_list = []
    exp_list = []
    re_list = []
    fgsa_list = []
    NER_str_list = []
    with open(path, 'r', encoding='utf8') as f:
        for line in f.readlines():
            elem_list = line.strip().split('###')
            sent_aspect = []
            sent_e = []
            sent_exp = []
            sent_re = []
            sent_fgsa = []
            for elem in elem_list:
                elem = elem.strip().split()
                aspect = elem[0]
                e_exp_words = elem[1].split('@@@')
                e_exp = e_exp_words[0]
                aspect_id = elem[3]
                re = (aspect, e_exp_words[0], e_exp_words[1])
                fgsa = ((aspect, aspect_id), e_exp_words[0], e_exp_words[1], elem[2])
                sent_aspect.append('<' + aspect + '>')
                sent_re.append(re)
                sent_fgsa.append(fgsa)
                if 'exp' in e_exp:
                    sent_exp.append('<' + e_exp_words[1] + '>')
                else:
                    sent_e.append('<' + e_exp_words[1] + '>')
            if len(sent_exp) == 0:
                ner_str = '实体：' + '、'.join(sent_aspect) + '\t情感：' + '、'.join(sent_e)
            elif len(sent_e) == 0:
                ner_str = '实体：' + '、'.join(sent_aspect) + '\t描述：' + '、'.join(sent_exp)
            else:
                ner_str = '实体：' + '、'.join(sent_aspect) + '\t情感：' + '、'.join(sent_e) + '\t描述：' + '、'.join(sent_exp)
            NER_str_list.append(ner_str)

            re_list.append(sent_re)
            fgsa_list.append(sent_fgsa)

    re_str_list = []
    for idx, sent_re in enumerate(re_list):
        sent_re = str(idx + 1) + '：' + '、'.join(['(<' + re[0] + '>, ' + re[1] + ', <' + re[2] + '>)' for re in sent_re])
        re_str_list.append(sent_re)
    fgsa_str_list = []
    for idx, sent_fgsa in enumerate(fgsa_list):
        sent_fgsa = str(idx + 1) + '：' + '、'.join(
            ['(<' + fgsa[0][0] + '>, ' + fgsa[1] + ', <' + fgsa[2] + '>, ' + str(fgsa[3]) + ')' for fgsa in sent_fgsa])
        fgsa_str_list.append(sent_fgsa)
    return NER_str_list, re_str_list, fgsa_str_list, fgsa_list


def mining(decode_path):
    a_e_exp_list = extract_info(path=decode_path, path1='test3.txt')
    a_e_exp_str_list = []
    for a_e_exp_sent in a_e_exp_list:
        a = []
        e = []
        exp = []
        for word, label in a_e_exp_sent:
            if label == 'a':
                a.append(word)
            elif label == 'e':
                e.append(word)
            elif label == 'exp':
                exp.append(word)
            else:
                raise RuntimeError
        a_e_exp_str = ''
        if len(a) != 0:
            a_str = '、'.join(['<' + word + '>' for word in a])
            a_e_exp_str += '实体：' + a_str + '\t'
        if len(e) != 0:
            e_str = '、'.join(['<' + word + '>' for word in e])
            a_e_exp_str += '情感：' + e_str + '\t'
        if len(exp) != 0:
            exp_str = '、'.join(['<' + word + '>' for word in exp])
            a_e_exp_str += '描述：' + exp_str
        a_e_exp_str_list.append(a_e_exp_str)

    first_name, second_name = read_first_data(data_path)
    chart1 = create_father_item(first_name)
    chart2 = create_item(first_name)
    chart3 = create_all_item(second_name)


    sent_list = getText(decode_path)
    TEXT_str = [str(idx + 1) + ' ' + sent for idx, sent in enumerate(sent_list)]
    NER_str_list, re_str_list, fgsa_str_list, fgsa_list = getInfo()
    NER_str = [str(idx + 1) + ' ' + ner_str for idx, ner_str in enumerate(a_e_exp_str_list)]
    RE_str = re_str_list
    FGSA_str = fgsa_str_list

    vocab = ThirdVocab()
    for out_key, out_value in chart3.items():
        for in_key, in_value in out_value.items():
            positive, negative = aspect_text(fgsa_list, in_key)
            in_value['positive_instances'] = positive
            in_value['negative_instances'] = negative
    result = json.dumps({'text1': TEXT_str,
                         'text2': NER_str,
                         'text3': RE_str,
                         'text4': FGSA_str,
                         'chart1': chart1,
                         'chart2': chart2,
                         'chart3': chart3})
    return result

