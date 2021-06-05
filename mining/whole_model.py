from only_web.mining.triple_model  import tripleModel
from only_web.mining import Vocab
from only_web.mining.unit import *
import time
import os
import json
import threading

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




def decode(input_path):
    SecondVocab = Vocab.SecondVocab()
    ThirdVocab = Vocab.ThirdVocab()
    # text = get_text(os.path.join(os.path.dirname(os.path.abspath('.')), input_path))
    text = get_text(input_path)
    start_time = time.time()

    batch_text = get_batch(text=text,batch_size=5)



    triple_info = tripleModel(batch_text)  #List<Dict>    # [{text: str}
    # {aspect :[]},
    # {opinion:[]},
    # {triples:[(),()]}
    # {sen_polarity: str}]
    # {target:[(),()]}
    end_time = time.time()
    print('cost time:%s '%(end_time-start_time))

    text_all,aspect_all,opinion_all,triple_all,sen_polarity_all,target_all = get_all_info(triple_info)

    aspect_and_opinion = get_a_and_o(aspect_all,opinion_all)
    ao_pair,ao_tri = get_RE_tri(triple_all)
    s_time = time.time()

    chart1,chart2,chart3 = get_target_info(target_all,SecondVocab,ThirdVocab)
    e_time = time.time()
    print("Process target time:",str(e_time-s_time))

    result = ( {'text1': text_all,
                'text2': aspect_and_opinion,
                'text3': ao_pair,
                'text4': ao_tri,
                'chart1': chart1,
                'chart2': chart2,
                'chart3': chart3})
    return result


if __name__ == '__main__':
    input_path = r'input.txt'
    res = decode(input_path)
    print(res)

