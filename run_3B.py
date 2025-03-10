ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero2.yaml     --num_processes=7 src/open_r1/grpo.py     --config recipes/Qwen2.5-Math-7B/grpo/config_simple_rl.yaml
lighteval vllm "pretrained=/home/khj/workspace/open-r1/data/Qwen-2.5-7B-Simple-RL,dtype=bfloat16,max_model_length=4096,gpu_memory_utilisation=0.8,tensor_parallel_size=1" "custom|math_500|0|0"     --custom-tasks src/open_r1/evaluate.py     --use-chat-template     --output-dir data/evals/distill
lighteval vllm "pretrained=/home/khj/workspace/open-r1/data/Qwen-2.5-7B-Simple-RL,dtype=bfloat16,max_model_length=4096,gpu_memory_utilisation=0.8,tensor_parallel_size=1" "custom|gpqa_diamond|0|0"     --custom-tasks src/open_r1/evaluate.py     --use-chat-template     --output-dir data/evals/distill

# 评测
PROMPT_TYPE="qwen25-math-cot"
# Qwen2.5-Math-1.5B-Instruct
export CUDA_VISIBLE_DEVICES="0"
MODEL_NAME_OR_PATH="/home/khj/workspace/open-r1/data/Qwen-2.5-7B-Simple-RL"
export MODEL_NAME_OR_PATH="/home/data/share/Qwen2.5-3B-Instruct"
bash sh/eval_math.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH

/home/khj/workspace/open-r1/data/Qwen-2.5-3B-Simple-RL

export CUDA_VISIBLE_DEVICES="0"
PROMPT_TYPE="qwen25-math-cot"
MODEL_NAME_OR_PATH="/home/khj/workspace/open-r1/data/Qwen-2.5-3B-Simple-RL"
bash sh/eval_math.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH


# 训练
export OMP_NUM_THREADS=4
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero2.yaml     --num_processes=5 src/open_r1/grpo.py     --config recipes/Qwen2.5-3B-Instruct/grpo/config_simple_rl.yaml
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero2.yaml     --num_processes=5 src/open_r1/grpo.py     --config recipes/Qwen2.5-3B-Instruct/grpo/config_simple_rl.yaml


ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero2.yaml     --num_processes=2 src/open_r1/grpo.py     --config recipes/Qwen2.5-3B-Instruct/grpo/config_simple_rl.yaml


Return your final response within \\boxed{}. The graph, $G$ of $y=\\log_{10}x$ is rotated $90^{\\circ}$ counter-clockwise about the origin to obtain a new graph $G'$. Which of the following is an equation for $G'$?\n(A) $y=\\log_{10}\\left(\\frac{x+90}{9}\\right)$ (B) $y=\\log_{x}10$ (C) $y=\\frac{1}{x+1}$ (D) $y=10^{-x}$ (E) $y=10^x$

trl chat --model_name_or_path /home/khj/workspace/open-r1/data/Qwen-2.5-3B-Simple-RL --system_prompt "You are a helpful AI Assistant that provides well-reasoned and detailed responses. You first think about the reasoning process as an internal monologue and then provide the user with the answer. Respond in the following format: <think>\n...\n</think>\n<answer>\n...\n</answer>" --max_new_tokens 8192

vllm serve /home/khj/workspace/open-r1/data/Qwen-2.5-3B-Simple-RL/checkpoint-117 --enable-prefix-caching --served-model-name test --port 8001 --tensor-parallel-size 1
vllm complete --url http://127.0.0.1:8001/v1 --model-name test --api-key EMPTY
<|im_start|>You are a helpful AI Assistant that provides well-reasoned and detailed responses. You first think about the reasoning process as an internal monologue and then provide the user with the answer. Respond in the following format: <think>\n...\n</think>\n<answer>\n...\n</answer>\n<|im_end|>\n<|im_start|>user\nDetermine if the graph of the equation below is a parabola, circle, ellipse, hyperbola, point, line, two lines, or empty. $x^2 + 2y^2 - 6x - 8y + 21 = 0$<|im_end|>\n<|im_start|>assistant\n

export VLLM_LOGGING_LEVEL=DEBUG
export CUDA_VISIBLE_DEVICES="7"
PROMPT_TYPE="qwen25-math-cot"
MODEL_NAME_OR_PATH="/home/khj/workspace/open-r1/data/Qwen-2.5-3B-Simple-RL/checkpoint-117"
bash sh/eval_math.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH
