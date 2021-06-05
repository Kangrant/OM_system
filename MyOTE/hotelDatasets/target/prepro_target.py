import os
import string

def target_map(dataset_path, type):
    #所有内容放到一个列表
    # read_file = os.path.join(dataset_path + type + '_target.txt')
    # fr = open(read_file, 'r', encoding='utf-8')
    write_file = os.path.join(dataset_path + type + '_target_map.txt')
    fw = open(write_file, 'r', encoding='utf-8')

    lines = fw.readlines()
    lines = eval(lines[0])
    print(len(lines))
    data = []
    # for line in lines:
    #     text, label = line.strip().split()
    #     data.append((text, label))

    # fw.write(str(data))

    # fr.close()
    fw.close()


def add_target(dataset_path, type):
    fr = open(os.path.join(dataset_path + type + '.pair'),'r',encoding='utf-8')

    fTarget = open(os.path.join(dataset_path + type +'_target_map.txt'),'r',encoding='utf-8')
    target = fTarget.readlines()[0]
    targets = eval(target)

    lines = fr.readlines()
    sentence_num = 0
    total = 0
    all_text = []
    all_label = []
    for i in range(0,len(lines),4):
        labels = []
        sentence_num+=1
        sentence = ''.join(lines[i].strip().split(' '))
        triplets = lines[i+2].strip().split(';')
        is_exit = False
        for triplet in triplets:
            ap_start,ap_end = eval(triplet)[0]
            target = sentence[ap_start:ap_end+1]

            for t in targets:
                text,id = t
                if text == target:
                    is_exit = True
                    res =  (ap_start,ap_end,text,id)
                    if res not in labels:
                        labels.append(res)
                    break

        if is_exit:
            all_text.append(sentence)
            all_label.append(labels)


    #将短语也放进去训练和验证

    f = open(os.path.join(dataset_path + type +'_target.txt'),'r',encoding='utf-8')
    for line in f.readlines():
        text,label = line.strip().split()
        all_text.append(text)
        end = len(text)-1
        all_label.append([(0,end,text,label)])



    fw = open(os.path.join(dataset_path + type + '.target'),'w',encoding='utf-8')
    for i in range(len(all_text)):
        fw.write(all_text[i]+'\n')
        fw.write(str(all_label[i])+'\n')


    fr.close()
    fw.close()
    f.close()

def sentence_all():
    f_train = open('train.pair','r',encoding='utf-8')
    f_dev = open('dev.pair', 'r', encoding='utf-8')
    f_all_sentence = open('all_sentece.txt', 'w', encoding='utf-8')

    lines_train = f_train.readlines()
    lines_dev = f_dev.readlines()

    sen_all = []
    ap_label = []
    ap_op_label = []
    sen_po = []
    triples_all = []
    for i in range(0,len(lines_train),4):
        ap_spans = []
        sentence = ''.join(lines_train[i].strip().split(' '))
        sen_all.append(sentence)
        triples = lines_train[i+2].strip().split(';')
        ap_op_label.append(lines_train[i+1].strip())
        triples_all.append(lines_train[i+2].strip())
        sen_po.append(lines_train[i+3].strip())

        for tri in triples:
            ap_span,_,_=eval(tri)
            ap_spans.append(ap_span)
        ap_label.append(ap_spans)

    for i in range(0,len(lines_dev),4):
        ap_spans = []
        sentence = ''.join(lines_dev[i].strip().split(' '))
        sen_all.append(sentence)
        triples = lines_dev[i+2].strip().split(';')
        ap_op_label.append(lines_dev[i+1].strip())
        triples_all.append(lines_dev[i+2].strip())
        sen_po.append(lines_dev[i+3].strip())

        for tri in triples:
            ap_span,_,_=eval(tri)
            ap_spans.append(ap_span)
        ap_label.append(ap_spans)

    assert len(sen_all) == len(ap_label)

    for i in range(len(sen_all)):
        f_all_sentence.write(sen_all[i]+'\n')
        f_all_sentence.write(str(ap_op_label[i])+'\n')
        f_all_sentence.write(str(triples_all[i])+'\n')
        f_all_sentence.write(str(sen_po[i])+'\n')
        f_all_sentence.write(str(ap_label[i])+'\n')
    print(i)
    f_train.close()
    f_dev.close()
    f_all_sentence.close()

def all_static():

    fTarget = open('train_target_map.txt', 'r', encoding='utf-8')
    f_all_sentence = open('all_sentece.txt', 'r', encoding='utf-8')

    target_list = eval(fTarget.readlines()[0])
    lines = f_all_sentence.readlines()

    train_sen ,dev_sen = [],[]
    train_label ,dev_label = [],[]
    train_ap_op_label = []
    train_tri=[]
    train_sen_po=[]

    for i in range(0,len(lines),5):
        sentence = lines[i].strip()
        ap_op_label = lines[i+1].strip()
        trip = lines[i+2].strip()
        sen_po = lines[i+3].strip()
        ap_label = eval(lines[i+4].strip())
        board = len(ap_label)
        temp_label = []

        for span in ap_label:
            beg,end = span
            aspect = sentence[beg:end+1]
            for target in target_list:
                text,type = target
                if text == aspect:
                    temp_label.append((beg,end,aspect,type))
                    break


        if len(temp_label) == board:
            train_sen.append(' '.join(list(sentence)))
            train_label.append(temp_label)
            train_sen_po.append(sen_po)
            train_tri.append(trip)
            train_ap_op_label.append(ap_op_label)
        # elif len(temp_label)>0:
        #     dev_sen.append(sentence)
        #     dev_label.append(temp_label)

    print(len(train_label))#17894
    # print(len(dev_label))  #298

    f_train = open('train.all', 'w', encoding='utf-8')
    f_dev = open('dev.all', 'w', encoding='utf-8')

    for i in range(len(train_label)):
        if i < 14315:
            f_train.write(train_sen[i]+'\n')
            f_train.write(train_ap_op_label[i]+'\n')
            f_train.write(train_tri[i]+'\n')
            f_train.write(train_sen_po[i]+'\n')
            f_train.write(str(train_label[i])+'\n')
        else:
            f_dev.write(train_sen[i]+'\n')
            f_dev.write(train_ap_op_label[i]+'\n')
            f_dev.write(train_tri[i]+'\n')
            f_dev.write(train_sen_po[i]+'\n')
            f_dev.write(str(train_label[i])+'\n')

    # for i in range(len(dev_label)):
    #     f_dev.write(dev_sen[i]+'\n')
    #     f_dev.write(str(dev_label[i])+'\n')


    fTarget.close()
    f_all_sentence.close()
    f_train.close()
    f_dev.close()
if __name__ == '__main__':
    dataset_path = r'hotelDatasets/target/'
    type = 'train'
    # target_map(dataset_path, type)
    #add_target(dataset_path,type) #train 15451  15382
                                  #dev   2766   1624
    #sentence_all() #train+dev 18216
    all_static()