ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero3.yaml \
    --num_processes=7 src/open_r1/grpo.py \
    --config recipes/Qwen2.5-3B/grpo/config_demo.yaml

# zero2 14 小时, zero3 12 小时 (zero3 在源码里没加载 )
# 
        # if is_deepspeed_zero3_enabled():
        #     self.ref_model = AutoModelForCausalLM.from_pretrained(model_id, **model_init_kwargs)
        # elif peft_config is None:
        #     # If PEFT configuration is not provided, create a reference model based on the initial model.
        #     self.ref_model = create_reference_model(model)
# zero2+tf32 预期 13h

# 换 tiny data， 4h
# 
