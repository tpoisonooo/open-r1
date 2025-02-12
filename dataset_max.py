from datasets import load_dataset
import pandas as pd
import pdb
import tiktoken

ENCODER = None
# modified from https://github.com/HKUDS/LightRAG
def encode_string(content: str, model_name: str = "gpt-4o"):
    global ENCODER
    if ENCODER is None:
        tiktoken.get_encoding("cl100k_base")
        ENCODER = tiktoken.encoding_for_model(model_name)
    tokens = ENCODER.encode(content)
    return tokens

# 加载数据集
dataset_name = "/data/share/Bespoke-Stratos-17k"  # 替换为你的Hugging Face数据集名称
dataset = load_dataset(dataset_name)

train = dataset['train']
max_len = 0
min_len = 200000
lens = []

num_columns=10
for item in train:
    # cur_len = len(encode_string(str(item)))
    cur_len = len(str(item))
    lens.append(cur_len)
    if cur_len > max_len:
        max_len = cur_len
    if cur_len < min_len:
        min_len = cur_len

slot = [0] * (num_columns + 1)
step = (max_len - min_len) // num_columns
for val in lens:
    index = round((val - min_len) / step)
    slot[index] += 1

print(min_len, max_len, step, num_columns)
for i, count in enumerate(slot):
    print('{} \t\t {}'.format(min_len + i * step, count))
