"""Reward functions for GRPO training."""

import json
import math
import re
from typing import Dict, Tuple

from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify

def accuracy_reward(completions, gold_standard_solution, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""
    train_data_type = kwargs.get('source_type', None)
    if train_data_type == 'code_python':
        return 0
    
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, sol in zip(contents, gold_standard_solution):
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        if len(gold_parsed) != 0:
            # We require the answer to be provided in correct latex (no malformed operators)
            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed="all",
                            units=True,
                        ),
                        # Ensures that boxed is tried first
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )
            # Reward 1 if the content is the same as the ground truth, 0 otherwise
            try:
                reward = float(verify(answer_parsed, gold_parsed))
            except Exception as e:
                print(f"verify failed: {e}, answer: {answer_parsed}, gold: {gold_parsed}")
                reward = 0.0
        else:
            # If the gold solution is not parseable, we reward 1 to skip this example
            reward = 1.0
            print("Failed to parse gold solution: ", sol)
        rewards.append(reward)
    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the reasoning process is enclosed within <think> and </think> tags, while the final answer is enclosed within <answer> and </answer> tags."""
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


def tag_count_reward(completions, source_type, **kwargs) -> list[float]:
    """Reward function that checks if we produce the desired number of think and answer tags associated with `format_reward()`.

    Adapted from: https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb#file-grpo_demo-py-L90
    """
    
    # 正向匹配
    def count_positive(text: str) -> float:
        count = 0.0
        positive_pattern = [r'^<think>', r'</think>', r'<answer>', r'</answer>$']
        for pattern in positive_pattern:
            match = re.search(pattern, text, re.MULTILINE)
            if match:
                count += 0.25
                text = text[match.end():]
            else:
                break
        return count
    
    def is_meaningful(content:str) ->  bool:
        pattern = r'[a-zA-Z\u4e00-\u9fff]'
        if re.search(pattern, content):
            return True
        else:
            return False

    def count_negative(text: str) -> float:
        count = 0.0
        negative_pattern = [r'<think>(.*?)</think>', r'<answer>(.*?)</answer>']
        for pattern in negative_pattern:
            match = re.search(pattern, text)
            if not match:
                continue
            content = match.group(1)
            if not is_meaningful(content=content):
                count -= 0.5
        return count

    rewards = []
    for completion, _type in zip(completions, source_type):
        if 'code_python' in _type:
            rewards.append(0.0)
            continue

        c = completion[0]["content"]
        score = count_positive(c) + count_negative(c)
        rewards.append(score)
    return rewards


def reasoning_steps_reward(completions, source_type, **kwargs):
    r"""Reward function that checks for clear step-by-step reasoning.
    Regex pattern:
        Step \d+: - matches "Step 1:", "Step 2:", etc.
        ^\d+\. - matches numbered lists like "1.", "2.", etc. at start of line
        \n- - matches bullet points with hyphens
        \n\* - matches bullet points with asterisks
        First,|Second,|Next,|Finally, - matches transition words
    """
    rewards = []
    pattern = r"(Step \d+:|^\d+\.|\n-|\n\*|First,|Second,|Next,|Finally,|So,|Now,|Since)"
    for completion, _type in zip(completions, source_type):
        if 'code_python' in _type:
            rewards.append(0.0)
            continue
        
        completion_content = completion[0]["content"]
        count = len(re.findall(pattern, completion_content))
        rewards.append(min(1.0, count / 3))
    return rewards


# 第一个 epoch 不开 len_reward
def len_reward(completions: list[Dict[str, str]], gold_standard_solution: list[str], source_type, **kwargs) -> float:
    """Compute length-based rewards to discourage overthinking and promote token efficiency.

    Taken from the Kimi 1.5 tech report: https://arxiv.org/abs/2501.12599

    Args:
        completions: List of model completions
        solution: List of ground truth solutions

    Returns:
        List of rewards where:
        - For correct answers: reward = 0.5 - (len - min_len)/(max_len - min_len)
        - For incorrect answers: reward = min(0, 0.5 - (len - min_len)/(max_len - min_len))
    """

    contents = [completion[0]["content"] for completion in completions]

    # First check correctness of answers
    correctness = []
    for content, sol, _type in zip(contents, gold_standard_solution, source_type):
        if 'code_python' == _type:
            rewards.append(0.0)
            continue

        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        if len(gold_parsed) == 0:
            # Skip unparseable examples
            correctness.append(True)  # Treat as correct to avoid penalizing
            print("Failed to parse gold solution: ", sol)
            continue

        answer_parsed = parse(
            content,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        equations=True,
                        boxed=True,
                        units=True,
                    ),
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )
        correctness.append(verify(answer_parsed, gold_parsed))

    # Calculate lengths
    lengths = [len(content) for content in contents]
    min_len = min(lengths)
    max_len = max(lengths)

    # If all responses have the same length, return zero rewards
    if max_len == min_len:
        return [0.0] * len(completions)

    rewards = []
    for length, is_correct in zip(lengths, correctness):
        lambda_val = 0.5 - (length - min_len) / (max_len - min_len)

        if is_correct:
            reward = lambda_val
        else:
            reward = min(0, lambda_val)

        rewards.append(float(reward))

    return rewards


def get_cosine_scaled_reward(
    min_value_wrong: float = -1.0,
    max_value_wrong: float = -0.5,
    min_value_correct: float = 0.5,
    max_value_correct: float = 1.0,
    max_len: int = 4096,
):
    def cosine_scaled_reward(completions, gold_standard_solution, source_type, **kwargs):
        """Reward function that scales based on completion length using a cosine schedule.

        Shorter correct solutions are rewarded more than longer ones.
        Longer incorrect solutions are penalized less than shorter ones.

        Args:
            completions: List of model completions
            solution: List of ground truth solutions

        This function is parameterized by the following arguments:
            min_value_wrong: Minimum reward for wrong answers
            max_value_wrong: Maximum reward for wrong answers
            min_value_correct: Minimum reward for correct answers
            max_value_correct: Maximum reward for correct answers
            max_len: Maximum length for scaling
        """
        contents = [completion[0]["content"] for completion in completions]
        rewards = []

        for content, sol, _type in zip(contents, gold_standard_solution, source_type):
            if 'code_python' == _type:
                rewards.append(float(0.0))
                continue

            gold_parsed = parse(sol, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
            if len(gold_parsed) == 0:
                rewards.append(1.0)  # Skip unparseable examples
                import pdb
                pdb.set_trace()
                print("Failed to parse gold solution: ", sol)
                continue

            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed=True,
                            units=True,
                        ),
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )

            is_correct = verify(answer_parsed, gold_parsed)
            gen_len = len(content)

            # Apply cosine scaling based on length
            progress = gen_len / max_len
            cosine = math.cos(progress * math.pi)

            if is_correct:
                min_value = min_value_correct
                max_value = max_value_correct
            else:
                # Swap min/max for incorrect answers
                min_value = max_value_wrong
                max_value = min_value_wrong

            reward = min_value + 0.5 * (max_value - min_value) * (1.0 + cosine)
            rewards.append(float(reward))

        return rewards

    return cosine_scaled_reward


def get_repetition_penalty_reward(ngram_size: int, max_penalty: float, **kwargs):
    """
    Computes N-gram repetition penalty as described in Appendix C.2 of https://arxiv.org/abs/2502.03373.
    Reference implementation from: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

    Args:
    ngram_size: size of the n-grams
    max_penalty: Maximum (negative) penalty for wrong answers
    """

    if max_penalty > 0:
        raise ValueError(f"max_penalty {max_penalty} should not be positive")

    def zipngram(text: str, ngram_size: int):
        words = text.lower().split()
        return zip(*[words[i:] for i in range(ngram_size)])

    def repetition_penalty_reward(completions, **kwargs) -> float:
        """
        reward function the penalizes repetitions
        ref implementation: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

        Args:
            completions: List of model completions
        """

        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        for completion in contents:
            if completion == "":
                rewards.append(0.0)
                continue
            if len(completion.split()) < ngram_size:
                rewards.append(0.0)
                continue

            ngrams = set()
            total = 0
            for ng in zipngram(completion, ngram_size):
                ngrams.add(ng)
                total += 1

            scaling = 1 - len(ngrams) / total
            reward = scaling * max_penalty
            rewards.append(reward)
        return rewards

    return repetition_penalty_reward


def extract_code(completion: str) -> str:
    pattern = re.compile(r"```python\n(.*?)```", re.DOTALL)
    matches = pattern.findall(completion)
    if len(matches) >=1:
        extracted_answer = matches[-1]
    elif 'input' in completion:
        extracted_answer = completion
    else:
        extracted_answer = ''
    return extracted_answer

def code_reward(completions, source_type, verification_info, **kwargs) -> list[float]:
    evaluation_script_template = """

def evaluate():
    import subprocess
    import json
    import math

    code = {code}
    test_cases = json.loads({test_cases})
    passed = 0
    total = len(test_cases)
    exec_timeout = 10

    for case in test_cases:
        process = subprocess.run(
            ["python3", "-c", code],
            input=case["input"],
            text=True,
            capture_output=True,
            timeout=exec_timeout
        )

        if process.returncode != 0:  # Error in execution
            continue

        output = process.stdout.strip()

        ##
        all_correct = True
        for line1, line2 in zip(output.split('\\n'), case['output'].split('\\n')):
            line1 = line1.strip()
            line2 = line2.strip()
            if line1 == line2:
                continue
            # print((line1, line2))
            try:
                num1 = float(line1)
                num2 = float(line2)
                all_correct = all_correct and math.isclose(num1, num2)
            except Exception as e:
                all_correct = False

        if all_correct:
            passed += 1

    success_rate = (passed / total)
    return success_rate

result=evaluate()
    """
    code_snippets = [extract_code(completion[-1]["content"]) for completion in completions]
    rewards = []
        
    for code, info, _type in zip(code_snippets, verification_info, source_type):
        if _type != 'code_python':
            rewards.append(0.0)
            continue

        script = evaluation_script_template.format(code=json.dumps(code), test_cases=json.dumps(json.dumps(info["test_cases"])))

        local_vars = {}
        try:
            exec(script, globals(), local_vars)
            score = local_vars.get('result')
            rewards.append(score)
        except Exception as e:
            print(f"Error from code sandbox: {e}")
            rewards.append(-0.1)

    return rewards

def get_code_format_reward(language: str = "python", **kwargs):
    """Format reward function specifically for code responses.

    Args:
        language: Programming language supported by E2B https://e2b.dev/docs/code-interpreting/supported-languages
    """

    pattern = r".*```python\n.*\n```.*"

    def code_format_reward(completions, source_type, **kwargs):

        rewards = []
        for completion, _type in zip(completions, source_type):
            if _type != 'code_python':
                rewards.append(0.0)
                continue
            content = completion[0]["content"]
            match = re.match(pattern, content, re.DOTALL | re.MULTILINE)
            if match:
                rewards.append(1.0)
            else:
                rewards.append(-0.5)

        return rewards
    return code_format_reward
