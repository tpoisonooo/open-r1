import unittest

from datasets import load_dataset
from open_r1.rewards import code_reward
from tqdm import tqdm

class TestCodeRewards(unittest.TestCase):
    def test_code_reward(self):
        code_dataset = load_dataset("/data/share/verifiable-coding-problems-python")
        samples = code_dataset["train"]
        keep_index = []
        for index, sample in tqdm(enumerate(samples)):
            test_completions = [[{"content": sample["gold_standard_solution"]}]]
            reward_kwargs = {"verification_info": [sample["verification_info"]]}
            # print(test_completions, reward_kwargs)
            rewards = code_reward(test_completions, **reward_kwargs)

            if rewards[0] == 1.0:
                keep_index.append(index)

        import pdb
        pdb.set_trace()
        save_samples = samples.select(keep_index)
        df = save_samples.to_pandas()
        # 3. 保存为 Parquet 文件
        df.to_parquet("/data/share/coding-problems-python/train.parquet", engine="pyarrow", index=False)

if __name__ == "__main__":
    unittest.main()
