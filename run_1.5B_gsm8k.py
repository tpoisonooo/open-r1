Return your final response within \\boxed{}. The graph, $G$ of $y=\\log_{10}x$ is rotated $90^{\\circ}$ counter-clockwise about the origin to obtain a new graph $G'$. Which of the following is an equation for $G'$?\n(A) $y=\\log_{10}\\left(\\frac{x+90}{9}\\right)$ (B) $y=\\log_{x}10$ (C) $y=\\frac{1}{x+1}$ (D) $y=10^{-x}$ (E) $y=10^x$

export OMP_NUM_THREADS=2
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero2.yaml \
    --num_processes=7 src/open_r1/grpo_only_acc.py \
    --config recipes/Qwen2.5-1.5B-Instruct/grpo/config_gsm8k.yaml