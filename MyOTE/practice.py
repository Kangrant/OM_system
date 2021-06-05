from transformers import BertTokenizer
#
import torch
#
t = BertTokenizer.from_pretrained('bert-base-chinese')

# res = t.encode([],add_special_tokens=False,is_split_into_words=True)
res = t(text=['我是中国人','你是谁'],add_special_tokens=False)
print(res)


