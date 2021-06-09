from only_web.mining.triple_model  import tripleModel,get_opt
from only_web.mining import Vocab
from only_web.mining.unit import *
from only_web.MyOTE.models.ote import OTE
from only_web.MyOTE.data_utils import ABSADataReader
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
    tokenizer = BertTokenizer.from_pretrained('only_web/MyOTE/bert-base-chinese')

    absa_data_reader = ABSADataReader(data_dir=opt.data_dir)
    idx2tag,idx2polarity,idx2target = absa_data_reader.reverse_tag_map, \
                                      absa_data_reader.reverse_polarity_map, \
                                      absa_data_reader.reverse_target_map
    model = OTE(
                opt=opt,
                idx2tag=idx2tag,
                idx2polarity=idx2polarity,
                idx2target = idx2target
                ).to(opt.device)
    print('loading model {0} ...'.format('OTE'))
    model.load_state_dict(torch.load(opt.state_dict_path, map_location=lambda storage, loc: storage))
    end_time = time.time()
    model.eval()
    print("Load model time:%.3f" % (end_time-start_time))
    return model,tokenizer


def decode(input_path):
    start_time = time.time()

    SecondVocab = Vocab.SecondVocab()
    ThirdVocab = Vocab.ThirdVocab()
    # text = get_text(os.path.join(os.path.dirname(os.path.abspath('.')), input_path))
    text = get_text(input_path)

    batch_text = get_batch(text=text,batch_size=5)

    model,tokenizer = load_model()
    triple_info = tripleModel(batch_text,model,tokenizer)  #List<Dict>    # [{text: str}
                                                    # {aspect :[]},
                                                    # {opinion:[]},
                                                    # {triples:[(),()]}
                                                    # {sen_polarity: str}]
                                                    # {target:[(),()]}

    s_time = time.time()
    text_all,aspect_all,opinion_all,triple_all,sen_polarity_all,target_all = get_all_info(triple_info)

    aspect_and_opinion = get_a_and_o(aspect_all,opinion_all)
    ao_pair,ao_tri = get_RE_tri(triple_all)

    chart1,chart2,chart3 = get_target_info(target_all,SecondVocab,ThirdVocab)
    e_time = time.time()
    print("Process Post time: %.3f" % (e_time-s_time))
    print('total cost time: %.3f' % (e_time-start_time))
    result = ( {'text1': text_all,
                         'text2': aspect_and_opinion,
                         'text3': ao_pair,
                         'text4': ao_tri,
                         'chart1': chart1,
                         'chart2': chart2,
                         'chart3': chart3})
    return result


if __name__ == '__main__':
    input_path = r'input_100.txt'
    res = decode(input_path)
    print(res)

