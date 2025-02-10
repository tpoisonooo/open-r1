"""Script to test format rewards for different models using vLLM."""

import argparse
from typing import List

import torch
from datasets import load_dataset

from open_r1.grpo import SYSTEM_PROMPT
from open_r1.rewards import format_reward
from vllm import LLM, SamplingParams


def format_prompt(question: str) -> List[dict]:
    """Format the prompt as a conversation."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
        {"role": "assistant", "content": "Let me solve this step by step.\n<think>"},
    ]


def apply_chat_template(messages: List[dict], tokenizer) -> str:
    """Apply the model's chat template if available, otherwise use our fixed template."""
    if hasattr(tokenizer, "apply_chat_template"):
        # Temporarily override the model's chat template
        original_template = tokenizer.chat_template
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, continue_final_message=True)
        finally:
            # Restore the original template
            tokenizer.chat_template = original_template

    # Fallback to simple template if no tokenizer chat template support
    formatted = ""
    for msg in messages:
        if msg["role"] == "system":
            formatted += f"System: {msg['content']}\n\n"
        elif msg["role"] == "user":
            formatted += f"User: {msg['content']}\n"
        elif msg["role"] == "assistant":
            formatted += f"Assistant: {msg['content']}\n"
    return formatted


def main():
    parser = argparse.ArgumentParser()
    # Model arguments
    parser.add_argument(
        "--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", help="Model ID or path"
    )
    parser.add_argument("--model_revision", type=str, default="main", help="Model revision to use")
    parser.add_argument(
        "--torch_dtype", type=str, default="bfloat16", help="PyTorch dtype (float16, bfloat16, float32)"
    )

    # Dataset arguments
    parser.add_argument("--dataset_name", type=str, default="open-r1/LIMO", help="Dataset to use for testing")
    parser.add_argument("--dataset_split", type=str, default="test", help="Dataset split to use")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to test")

    # Generation arguments
    parser.add_argument("--max_tokens", type=int, default=4096, help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--max_prompt_length", type=int, default=768, help="Maximum length for prompts")

    # vLLM arguments
    parser.add_argument("--vllm_device", type=str, default="auto", help="Device to use for vLLM")
    parser.add_argument(
        "--vllm_gpu_memory_utilization", type=float, default=0.7, help="GPU memory utilization for vLLM"
    )

    args = parser.parse_args()

    # Set torch dtype
    if args.torch_dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    elif args.torch_dtype == "float16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    # Initialize vLLM
    print(f"Loading model {args.model}...")
    llm = LLM(
        model=args.model,
        revision=args.model_revision,
        dtype=torch_dtype,
        gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        device=args.vllm_device,
    )
    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    # Load dataset
    print(f"Loading dataset {args.dataset_name}...")
    dataset = load_dataset(args.dataset_name, split=args.dataset_split)

    # Sample questions from dataset
    if args.num_samples:
        dataset = dataset.select(range(min(args.num_samples, len(dataset))))

    # Test each question
    for example in dataset:
        print("\n" + "=" * 80)
        print(f"Question: {example['problem']}")
        print("-" * 80)

        # Format prompt
        messages = format_prompt(example["problem"])
        prompt = apply_chat_template(messages, tokenizer)

        # Generate completion
        outputs = llm.generate(prompt, sampling_params)
        completion = outputs[0].outputs[0].text
        print(f"Completion:\n{completion}")

        # Check format reward
        reward = format_reward([[{"content": completion}]])
        print(f"\nFormat reward: {reward[0]}")

        # Print analysis
        if reward[0] == 0:
            print("\nAnalysis: Format check failed. Looking for pattern:")
            print("- Must contain <think>...</think> followed by <answer>...</answer>")
            print("- Check if tags are properly closed and in correct order")
        else:
            print("\nAnalysis: Format check passed!")

        print(f"\nGround truth answer: {example.get('answer', 'N/A')}")


if __name__ == "__main__":
    main()
