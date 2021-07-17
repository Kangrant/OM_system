import torch
from only_web.MyOTE.infer import Inferer
from only_web.MyOTE.models.ote import OTE
import time

def get_opt():

    dataset = 'hotel'
    # set your trained models here
    model_state_dict_paths = {
        #'ote': 'state_dict/ote_' + dataset + '.pkl',
        'ote_Bert': 'only_web/MyOTE/state_dict/ote_Bert_' + dataset + '.pkl',
        'ote_LSTM': 'only_web/MyOTE/state_dict/ote_LSTM_' + dataset + '.pkl',
    }
    model_classes = {
        'ote_Bert': OTE,
        'ote_LSTM': OTE,
    }
    input_colses = {
        'ote_Bert': ['text_indices', 'text_mask'],
        'ote_LSTM': ['text_indices', 'text_mask'],
    }
    target_colses = {
        'ote_Bert': ['ap_indices', 'op_indices', 'triplet_indices', 'text_mask', 'sentece_polarity','target_indices'],
        'ote_LSTM': ['ap_indices', 'op_indices', 'triplet_indices', 'text_mask', 'sentece_polarity','target_indices'],
    }
    data_dirs = {
        'hotel': 'only_web/MyOTE/hotelDatasets/hotel',

    }


    class Option(object):
        pass

    opt = Option()
    opt.useBert = True
    opt.dataset = dataset
    if opt.useBert:
        opt.model_name = 'ote_Bert'
    else:
        opt.model_name = 'ote_LSTM'
    opt.eval_cols = ['ap_spans', 'op_spans', 'triplets', 'sentece_polarity','targets']
    opt.model_class = model_classes[opt.model_name]
    opt.input_cols = input_colses[opt.model_name]
    opt.target_cols = target_colses[opt.model_name]
    opt.state_dict_path = model_state_dict_paths
    opt.embed_dim = 300
    opt.hidden_dim = 300
    opt.polarities_dim = 4
    opt.batch_size = 32
    opt.data_dir = data_dirs[opt.dataset]
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    return opt

def tripleModel(text,model,tokenizer):

    opt = get_opt()

    inf = Inferer(opt,model,tokenizer)

    polarity_map = {0: 'N', 1: 'NEU', 2: 'NEG', 3: 'POS'}

    #text = ['朋友一行合肥打球,选择这家酒店,房间干净整洁,前台小妹妹很热情,退房时因天气热,还送了瓶水给我,感觉很好,下次有机会去,还会住这家酒店']

    pred_out_all = []
    st_time = time.time()

    opt.useBert = True if len(text) <=10000 else False
    for batch_text in text:
        pred_out = inf.evaluate(batch_text,opt)
        pred_out_all.append(pred_out)
    en_time = time.time()

    s_time = time.time()
    triple_info = []
    # aspect_all, opinion_all = [], []
    for j in range(len(pred_out_all)):
        pred_out = pred_out_all[j]
        for i in range(len(pred_out[0])):
            info = {}
            ap_span, op_span = [], []
            #ap and op
            ap_pred = pred_out[0][i]
            op_pred = pred_out[1][i]
            for ap in ap_pred:
                ap_beg, ap_end = ap
                aspect = text[j][i][ap_beg:ap_end + 1]
                ap_span.append(aspect)
            info['aspect'] = ap_span
            for op in op_pred:
                op_beg, op_end = op
                opinion = text[j][i][op_beg:op_end + 1]
                op_span.append(opinion)
            info['opinion'] = op_span
            # assert len(aspect_all) == len(opinion_all)
            info['text'] = text[j][i]
            #句子极性
            s_p = pred_out[3][i]
            sen_polarity = polarity_map[s_p]
            info['sen_polarity'] = sen_polarity
            #三元组
            triplets = pred_out[2][i]
            target_info = pred_out[4][i]
            tri,target_temp = [],[]

            _target_info = []
            for target in target_info:
                tar_beg,tar_end ,_ = target
                for tri_ in triplets:
                    tri_beg,tri_end,_,_,_ = tri_
                    if tar_beg == tri_beg and tar_end == tri_end:
                        _target_info.append(target)


            for triplet in triplets:
                ap_beg, ap_end, op_beg, op_end, p = triplet
                ap = text[j][i][ap_beg:ap_end + 1]
                op = text[j][i][op_beg:op_end + 1]
                polarity = polarity_map[p]
                tri.append((ap,op,polarity))
                for _target in _target_info:
                    a_beg,a_end ,third_name= _target
                    aspect = text[j][i][a_beg:a_end + 1]
                    second_name = third_name[0]
                    if(aspect == ap):
                        target_temp.append((ap,second_name,third_name,polarity,(ap,op,polarity)))
                        break

            info['triples'] = tri
            info['target'] = target_temp

            express_single = pred_out[5][i]
            exp = []
            if express_single:
                for express in express_single:
                    beg,end ,type= express
                    exp.append((text[j][i][beg:end+1],type))
            info['express'] = exp

            triple_info.append(info)

    model_end_time = time.time()

    print("triple_model time: %.3f" % (en_time-st_time))
    print("triple_precess pred out time: %.3f" % (model_end_time-s_time))
    return triple_info

