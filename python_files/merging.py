# 0. imports
import torch
from huggingface_hub import login
from transformers import AutoModelForCausalLM
from peft import PeftModel


if __name__ == "__main__":
    access_token_read = "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"
    access_token_write = "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"
    login(token = access_token_write)
    base_model_name = "meta-llama/Llama-2-7b-chat-hf"
    adapter_model_name = "guoyu-zhang/Llama2-7b-chat-dpo-hh-lora"
    
    # 1. load a pretrained model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        load_in_4bit=True,
    )
    model.config.use_cache = False
    
    model = PeftModel.from_pretrained(model, adapter_model_name)

    model = model.merge_and_unload()
    model.push_to_hub("guoyu-zhang/Llama2-7b-chat-dpo-hh")
    # model.save_pretrained("merged_model")
   