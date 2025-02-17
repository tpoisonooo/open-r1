ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero2.yaml     --num_processes=7 src/open_r1/grpo.py     --config recipes/Qwen2.5-Math-7B/grpo/config_simple_rl.yaml
lighteval vllm "pretrained=/home/khj/workspace/open-r1/data/Qwen-2.5-7B-Simple-RL,dtype=bfloat16,max_model_length=4096,gpu_memory_utilisation=0.8,tensor_parallel_size=1" "custom|math_500|0|0"     --custom-tasks src/open_r1/evaluate.py     --use-chat-template     --output-dir data/evals/distill
lighteval vllm "pretrained=/home/khj/workspace/open-r1/data/Qwen-2.5-7B-Simple-RL,dtype=bfloat16,max_model_length=4096,gpu_memory_utilisation=0.8,tensor_parallel_size=1" "custom|gpqa_diamond|0|0"     --custom-tasks src/open_r1/evaluate.py     --use-chat-template     --output-dir data/evals/distill

# Qwen2.5-Math-Instruct Series
# Qwen2.5-Math-1.5B-Instruct
MODEL_NAME_OR_PATH="/home/khj/workspace/open-r1/data/Qwen-2.5-7B-Simple-RL"

export CUDA_VISIBLE_DEVICES="7"
PROMPT_TYPE="qwen25-math-cot"
MODEL_NAME_OR_PATH="/data/share/Qwen2.5-Math-7B"
bash sh/eval_math.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH
