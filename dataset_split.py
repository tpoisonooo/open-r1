from datasets import load_dataset
import pandas as pd
import tiktoken

# 加载数据集
dataset_name = "/data/share/NuminaMath-TIR/"  # 替换为你的Hugging Face数据集名称
dataset = load_dataset(dataset_name)


# 检查数据集是否包含train和test分割
if "train" in dataset.keys() and "test" in dataset.keys():
    # 随机抽取训练集和测试集的25%
    train_df = dataset["train"].to_pandas().sample(frac=0.5, random_state=42)
    test_df = dataset["test"].to_pandas().sample(frac=0.5, random_state=42)
    print(len(train_df))
    print(len(test_df))


    # 保存为新的Parquet文件
    train_df.to_parquet("/data/share/TIR-tiny/data/train-00000-of-00001.parquet", index=False)
    test_df.to_parquet("/data/share/TIR-tiny/data/test-00000-of-00001.parquet", index=False)

    print("样本数据已保存为 train_sampled.parquet 和 test_sampled.parquet")
else:
    print("数据集未包含train和test分割，请检查数据集结构。")