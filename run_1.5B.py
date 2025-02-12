Return your final response within \\boxed{}. The graph, $G$ of $y=\\log_{10}x$ is rotated $90^{\\circ}$ counter-clockwise about the origin to obtain a new graph $G'$. Which of the following is an equation for $G'$?\n(A) $y=\\log_{10}\\left(\\frac{x+90}{9}\\right)$ (B) $y=\\log_{x}10$ (C) $y=\\frac{1}{x+1}$ (D) $y=10^{-x}$ (E) $y=10^x$

export OMP_NUM_THREADS=2
accelerate launch --config_file recipes/accelerate_configs/zero3.yaml src/open_r1/sft.py     --config recipes/Qwen2.5-1.5B-Instruct/sft/config_demo.yaml
lighteval vllm "pretrained=/home/khj/workspace/open-r1/data/Qwen2.5-1.5B-Open-R1-Distill,dtype=bfloat16,max_model_length=32768,gpu_memory_utilisation=0.8,tensor_parallel_size=1" "custom|gpqa:diamond|0|0"     --custom-tasks src/open_r1/evaluate.py     --use-chat-template     --output-dir data/evals/distill
lighteval vllm "pretrained=/home/data/share/Qwen2.5-1.5B-Instruct,dtype=bfloat16,max_model_length=32768,gpu_memory_utilisation=0.8,tensor_parallel_size=1" "custom|gpqa:diamond|0|0"     --custom-tasks src/open_r1/evaluate.py     --use-chat-template     --output-dir data/evals/distill

lighteval vllm "pretrained=/home/data/share/DeepSeek-R1-Distill-Qwen-1.5B,dtype=bfloat16,max_model_length=32768,gpu_memory_utilisation=0.8,tensor_parallel_size=1" "custom|gpqa:diamond|0|0"     --custom-tasks src/open_r1/evaluate.py     --use-chat-template     --output-dir data/evals/distill
5min49s
lighteval vllm "pretrained=/home/data/share/DeepSeek-R1-Distill-Qwen-1.5B,dtype=bfloat16,max_model_length=32768,gpu_memory_utilisation=0.8,tensor_parallel_size=1" "custom|math_500|0|0"     --custom-tasks src/open_r1/evaluate.py     --use-chat-template     --output-dir data/evals/distill

|        Task         |Version|     Metric     |Value |   |Stderr|
|---------------------|------:|----------------|-----:|---|-----:|
|all                  |       |extractive_match|0.3283|±  |0.0335|
|custom:gpqa:diamond:0|      1|extractive_match|0.3283|±  |0.0335|
|      Task       |Version|     Metric     |Value|   |Stderr|
|-----------------|------:|----------------|----:|---|-----:|
|all              |       |extractive_match| 0.83|±  |0.0168|
|custom:math_500:0|      1|extractive_match| 0.83|±  |0.0168|

lighteval vllm "pretrained=/data/khj/workspace/open-r1/data/Qwen2.5-1.5B-Open-R1-Distill,dtype=bfloat16,max_model_length=32768,gpu_memory_utilisation=0.8,tensor_parallel_size=1" "custom|math_500|0|0"     --custom-tasks src/open_r1/evaluate.py     --use-chat-template     --output-dir data/evals/distill

27.78
python run.py --datasets gpqa_gen --hf-type chat --hf-path /data/khj/workspace/open-r1/data/Qwen2.5-1.5B-Open-R1-Distill --debug  --max-out-len 32700 -a vllm  --hf-num-gpus 7  --max-num-workers 8
