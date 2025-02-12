# 1.5B
TASK=gpqa:diamond
lighteval vllm "pretrained=/home/khj/workspace/open-r1/data/Qwen2.5-1.5B-Open-R1-GRPO,dtype=bfloat16,max_model_length=32768,gpu_memory_utilisation=0.8" "custom|$TASK|0|0" \
    --custom-tasks src/open_r1/evaluate.py \
    --use-chat-template \
    --output-dir data/evals/grpo


lighteval vllm "pretrained=/data/share/Qwen2.5-1.5B-Instruct,dtype=bfloat16,max_model_length=32768,gpu_memory_utilisation=0.8" "custom|$TASK|0|0" \
    --custom-tasks src/open_r1/evaluate.py \
    --use-chat-template \
    --output-dir data/evals/base

/data/khj/workspace/open-r1/data/Qwen2.5-1.5B-Open-R1-Distill
lighteval vllm "pretrained=/data/khj/workspace/open-r1/data/Qwen2.5-1.5B-Open-R1-Distill,dtype=bfloat16,max_model_length=32768,gpu_memory_utilisation=0.8" "custom|$TASK|0|0" \
    --custom-tasks src/open_r1/evaluate.py \
    --use-chat-template \
    --output-dir data/evals/distill

# 3B
# 


TASK=gpqa:diamond
lighteval vllm "pretrained=/data/share/Qwen2.5-3B-Instruct,dtype=bfloat16,max_model_length=32768,gpu_memory_utilisation=0.8" "custom|$TASK|0|0" \
    --custom-tasks src/open_r1/evaluate.py \
    --use-chat-template \
    --output-dir data/evals/distill

# 0.2576
TASK=gpqa:diamond
lighteval vllm "pretrained=/data/khj/workspace/open-r1/data/Qwen2.5-3B-Open-R1-GRPO,dtype=bfloat16,max_model_length=32768,gpu_memory_utilisation=0.8" "custom|$TASK|0|0" \
    --custom-tasks src/open_r1/evaluate.py \
    --use-chat-template \
    --output-dir data/evals/distill


/data/khj/workspace/open-r1/data/Qwen2.5-3B-Open-R1-Distill

TASK=math_500
TASK=aime24
lighteval vllm "pretrained=/data/khj/workspace/open-r1/data/Qwen2.5-3B-Open-R1-Distill,dtype=bfloat16,max_model_length=32768,gpu_memory_utilisation=0.8" "custom|$TASK|0|0" \
    --custom-tasks src/open_r1/evaluate.py \
    --use-chat-template \
    --output-dir data/evals/distill

TASK=aime24
TASK=math_500
lighteval vllm "pretrained=/data/share/Qwen2.5-3B-Instruct,dtype=bfloat16,max_model_length=32768,gpu_memory_utilisation=0.8" "custom|$TASK|0|0" \
    --custom-tasks src/open_r1/evaluate.py \
    --use-chat-template \
    --output-dir data/evals/distill


MODEL_ARGS="pretrained=/home/khj/workspace/open-r1/data/Qwen2.5-3B-Open-R1-Distill,dtype=bfloat16,max_model_length=32768,gpu_memory_utilisation=0.8,tensor_parallel_size=1"
lighteval vllm $MODEL_ARGS "custom|gpqa:diamond|0|0" \
    --custom-tasks src/open_r1/evaluate.py \
    --use-chat-template \
    --output-dir data/evals/distill

MODEL_ARGS="pretrained=/home/khj/workspace/open-r1/data/Qwen2.5-1.5B-Open-R1-Distill,dtype=bfloat16,max_model_length=32768,gpu_memory_utilisation=0.8,tensor_parallel_size=1"
lighteval vllm "pretrained=/home/khj/workspace/open-r1/data/Qwen2.5-1.5B-Open-R1-Distill,dtype=bfloat16,max_model_length=32768,gpu_memory_utilisation=0.8,tensor_parallel_size=1" "custom|gpqa:diamond|0|0" \
    --custom-tasks src/open_r1/evaluate.py \
    --use-chat-template \
    --output-dir data/evals/distill

# oc
opencompass --models /data/share/Qwen2.5-3B-Instruct --datasets demo_math_chat_gen -a vllm

opencompass --datasets demo_math_chat_gen  --hf-type chat --hf-path /data/share/Qwen2.5-3B-Instruct


python run.py     --datasets gpqa_gen     --hf-type chat     --hf-path /data/khj/workspace/open-r1/data/Qwen2.5-3B-Open-R1-Distill --debug  --max-out-len  8192 -a vllm  --hf-num-gpus 7  --max-num-workers 8
| dataset | version | metric | mode | Qwen2.5-3B-Instruct_hf-vllm |
|----- | ----- | ----- | ----- | -----|
| GPQA_diamond | 5aeece | accuracy | gen | 29.80 |

| dataset | version | metric | mode | Qwen2.5-3B-Open-R1-Distill_hf-vllm |
|----- | ----- | ----- | ----- | -----|
| GPQA_diamond | 5aeece | accuracy | gen | 27.78 |


