from datasets import load_dataset, Dataset, DatasetDict
import random

def random_indexs(code_dataset, random_count: int = 7500):
    max_len = len(code_dataset)
    assert random_count < max_len
    return random.sample(range(max_len), random_count)

code_dataset = load_dataset("/data/share/coding-problems-python")['train']
indexes = random_indexs(code_dataset)
code_dataset = code_dataset.select(indexes)

# 加代码数据
# 'source', 'task_type', 'in_source_id', 'problem_statement', 'gold_standard_solution', 
# 'problem_id', 'metadata', 'verification_info'
merged_code = [{
    'task_type':code['task_type'], 
    'problem_statement':code['problem_statement'], 
    'gold_standard_solution':code['gold_standard_solution'], 
    'verification_info': code['verification_info'],
    'source_type': 'code_python',
    'system': 'You are a helpful Assistant.'
} for code in code_dataset]

# 'problem', 'level', 'solution', 'type', source_type
# 7500
# problem -> problem_statement
# solution -> gold_standard_solution
# task -> task_type
# 新增 source_type
math_dataset = load_dataset("/home/data/share/MATH-lighteval")

merged_math = [{}] * len(math_dataset['train'])
for index, m in enumerate(math_dataset['train']):
    item = {
        'problem_statement':m['problem'],
        'gold_standard_solution':m['solution'],
        'task_type':m['type'],
        'source_type':'math',
        'verification_info': None,
        'system': "You are a helpful AI Assistant that provides well-reasoned and detailed responses. You first think about the reasoning process as an internal monologue and then provide the user with the answer. Respond in the following format: <think>\n...\n</think>\n<answer>\n...\n</answer>"
    }
    merged_math[index] = item

print(len(merged_code))
print(len(merged_math))

test_math = [{'problem_statement':m['problem'], 
              'gold_standard_solution':m['solution'], 
              'task_type':m['type'], 
              'source_type':'math', 
              'verification_info': None,
              'system': "You are a helpful AI Assistant that provides well-reasoned and detailed responses. You first think about the reasoning process as an internal monologue and then provide the user with the answer. Respond in the following format: <think>\n...\n</think>\n<answer>\n...\n</answer>"} for m in math_dataset['test']]

train_merged = Dataset.from_list(merged_math + merged_code)
# train_merged = Dataset.from_list(merged_math)

test_merged = Dataset.from_list(test_math)

dataset_dict = DatasetDict({"train": train_merged, "test": test_merged})
for split, dataset in dataset_dict.items():
    # 定义保存路径，例如：path/to/save/parquet_files/train.parquet
    save_path = f"/data/share/lighteval-code/{split}.parquet"
    # 保存为 Parquet 文件
    dataset.to_parquet(save_path)
    print(f"Saved {split} dataset to {save_path}")

# df = dataset_dict.to_pandas()
# df.to_parquet("/data/share/lighteval-code/train.parquet", engine="pyarrow", index=False)

# from datasets import load_dataset, Dataset
# import numpy as np

# def random_indexs(code_dataset, random_count: int = 15000):
#     max_len = len(code_dataset)
#     assert random_count < max_len
#     return np.random.choice(max_len, random_count, replace=False)

# code_dataset = load_dataset("/data/share/coding-problems-python")['train']
# indexes = random_indexs(code_dataset)
# code_dataset = code_dataset.select(indexes)

# merged_code = [{**code, 'source_type': 'code_python'} for code in code_dataset]

# math_dataset = load_dataset("/home/data/share/MATH-lighteval")['train']
# merged_math = [
#     {
#         'problem_statement': m['problem'],
#         'gold_standard_solution': m['solution'],
#         'task_type': m['type'],
#         'source_type': 'math',
#         'system': "You are a helpful AI Assistant that provides well-reasoned and detailed responses. You first think about the reasoning process as an internal monologue and then provide the user with the answer. Respond in the following format: <think>\n...\n</think>\n<answer>\n...\n</answer>"
#     } for m in math_dataset
# ]

# merged = Dataset.from_dict({"train": merged_code + merged_math})
# merged.to_parquet("/data/share/lighteval-code/train.parquet", engine="pyarrow", index=False)