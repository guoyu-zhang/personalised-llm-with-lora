import torch
from huggingface_hub import login
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import AutoPeftModelForCausalLM


if __name__ == "__main__":
    access_token_read = "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"
    access_token_write = "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"
    login(token = access_token_write)

    bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    )
    
    # 1. load a pretrained model
    model = AutoPeftModelForCausalLM.from_pretrained(
        './refined_user1',
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        quantization_config=bnb_config,
    )
    
    model.push_to_hub("guoyu-zhang/user1")