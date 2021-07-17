from only_web.mining.triple_model  import tripleModel,get_opt
from only_web.mining import Vocab
from only_web.mining.unit import *
from only_web.MyOTE.models.ote import OTE
from only_web.MyOTE.data_utils import ABSADataReader,build_tokenizer,build_embedding_matrix
import time
import os
import json
import threading
import torch
from transformers import BertTokenizer
import numpy as np

np.seterr(divide='ignore',invalid='ignore')
# triple_info = [{'text': '我是中国人','aspect':['我','中国人'],'opinion':['我','是中国人'],'triples':[('我','33','POS'),('你','吗','POS')],'sen_polarity':'POS'},
#                {'text': '我是中国人','aspect':['我'],'opinion':['是中国人'],'triples':[('他','22','POS')],'sen_polarity':'POS'}]
#
# target_info = [[('我','24','国籍','人种'),('你','24','国籍','人种')],[('他','45','a','人asfa')]]



class MyThread(threading.Thread):

    def __init__(self, func, args=()):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args

    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        try:
            return self.result  # 如果子线程不使用join方法，此处可能会报没有self.result的错误
        except Exception:
            return None

def load_model():
    start_time = time.time()
    opt = get_opt()

    #bert
    tokenizer_bert = BertTokenizer.from_pretrained("only_web/MyOTE/bert-base-chinese")
    embedding_matrix_bert = torch.tensor([0])
    #LSTM
    tokenizer_LSTM = build_tokenizer(data_dir=opt.data_dir)
    embedding_matrix_LSTM = build_embedding_matrix(opt.data_dir, tokenizer_LSTM.word2idx, opt.embed_dim, opt.dataset)

    absa_data_reader = ABSADataReader(data_dir=opt.data_dir,opt=opt)
    idx2tag,idx2polarity,idx2target,id2express =  absa_data_reader.reverse_tag_map, \
                                                  absa_data_reader.reverse_polarity_map, \
                                                  absa_data_reader.reverse_target_map,\
                                                  absa_data_reader.reverse_express_map


    model_bert = OTE(embedding_matrix=embedding_matrix_bert,
                opt=opt,
                idx2tag=idx2tag,
                idx2polarity=idx2polarity,
                idx2target = idx2target,
                idx2express =id2express,
                ).to(opt.device)

    model_bert.load_state_dict(torch.load(opt.state_dict_path['ote_Bert'], map_location=lambda storage, loc: storage))
    model_bert.eval()

    model_LSTM = OTE(embedding_matrix=embedding_matrix_LSTM,
                     opt=opt,
                     idx2tag=idx2tag,
                     idx2polarity=idx2polarity,
                     idx2target = idx2target,
                     idx2express =id2express,
                     ).to(opt.device)

    model_LSTM.load_state_dict(torch.load(opt.state_dict_path['ote_LSTM'], map_location=lambda storage, loc: storage))
    model_LSTM.eval()
    # model_LSTM,tokenizer_LSTM = [],[]

    return model_bert,tokenizer_bert,model_LSTM,tokenizer_LSTM


def decode(input_path):
    start_time = time.time()

    SecondVocab = Vocab.SecondVocab()
    ThirdVocab = Vocab.ThirdVocab()
    # text = get_text(os.path.join(os.path.dirname(os.path.abspath('.')), input_path))
    text = get_text(input_path)

    batch_text = get_batch(text=text,batch_size=100)

    model_bert,tokenizer_bert,model_LSTM,tokenizer_LSTM = load_model()


    model = model_bert if len(text) <=10000 else model_LSTM
    tokenizer = tokenizer_bert if len(text) <= 10000 else model_LSTM

    triple_info = tripleModel(batch_text,model,tokenizer)  #List<Dict>    # [{text: str}
                                                    # {aspect :[]},
                                                    # {opinion:[]},
                                                    # {triples:[(),()]}
                                                    # {sen_polarity: str}]
                                                    # {target:[(),()]}

    s_time = time.time()
    text_all,aspect_all,opinion_all,triple_all,sen_polarity_all,target_all ,express_all = get_all_info(triple_info)

    aspect_and_opinion = get_a_and_o(aspect_all,opinion_all)
    ao_pair,ao_tri = get_RE_tri(triple_all)

    chart1,chart2,chart3 = get_target_info(target_all,SecondVocab,ThirdVocab)

    express_info = get_express(express_all,opinion_all)

    e_time = time.time()
    print("Process Post time: %.3f" % (e_time-s_time))
    print('total cost time: %.3f' % (e_time-start_time))
    result = ( {'text1': text_all,
                         'text2': aspect_and_opinion,
                         'text3': ao_pair,
                         'text4': ao_tri,
                         'chart1': chart1,
                         'chart2': chart2,
                         'chart3': chart3,
                         'chart4': express_info,
                })
    return result


if __name__ == '__main__':
    input_path = r'input_100.txt'
    res = decode(input_path)
    print(res)

