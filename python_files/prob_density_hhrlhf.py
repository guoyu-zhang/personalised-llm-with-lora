import os
import sys
from typing import Dict

import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
from peft import AutoPeftModelForCausalLM


def get_anthropic_hh_dataset(
    data_dir: str = "data",
    sanity_check: bool = False,
    cache_dir: str = None,
    num_proc=24,
    split="train",
) -> Dataset:
    """Load the Anthropic helpfulness and harmlessness dataset from Hugging Face and convert it to the necessary format.

    The dataset is converted to following structure:
    {
        'prompt': List[str],
        'chosen': List[str],
        'rejected': List[str],
    }
    """
    dataset = load_dataset(
        "Dahoas/full-hh-rlhf",
        split=split,
        cache_dir=cache_dir,
        data_dir=data_dir,
    )
    
    original_columns = dataset.column_names

    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 1000)))
        
    def return_prompt_and_responses(samples) -> Dict[str, str]:
        return {
            "prompt": samples["prompt"],
            "chosen": samples["chosen"],
            "rejected": samples["rejected"],
        }

    return dataset.map(
        return_prompt_and_responses,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )

if __name__ == "__main__": 
    results_file = "proper_prob_density_merged_test.txt"
    access_token_read = "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"
    access_token_write = "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"
    login(token = access_token_write)
    # -------------- PRETRAINED MODEL --------------
    
    # Load the model
    pretrained_model = AutoModelForCausalLM.from_pretrained(
        "guoyu-zhang/Llama2-7b-chat-dpo-hh",
        low_cpu_mem_usage=True,
        torch_dtype=torch.float32,
        load_in_4bit=True,
    )

    # Load the tokenizer
    pretrained_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

    with open(results_file, 'a') as file: 
        file.write("Loaded model\n")

    # 3. Load evaluation dataset
    eval_dataset = get_anthropic_hh_dataset(data_dir="data", sanity_check=False, split="test")
    eval_dataset = eval_dataset.filter(  
        lambda x: len(x["prompt"]) + len(x["chosen"]) <= 1024
        and len(x["prompt"]) + len(x["rejected"]) <= 1024
    )

    with open(results_file, 'a') as file: 
        file.write("Loaded dataset\n")   
    
    # Loop over the dataset, get the prompt, generate a response with both PRETRAINED and FINE TUNED models.
    # Get the log probs for each of these for the JUDGE model
    win_rate_count = 0
    counter = 0
    sum_nll = 0
    for example in eval_dataset:
        log_likelihood = 0
        counter += 1 
            
        prompt = example['prompt']
        chosen = example['chosen']
        
        combined_text = prompt + chosen
        tokenized_text = pretrained_tokenizer.encode(combined_text, return_tensors="pt")

        # Calculate log probability
        with torch.no_grad():
            outputs = pretrained_model(tokenized_text, labels=tokenized_text)
            log_likelihood = outputs.loss.item() * -1
            sum_nll += log_likelihood
            
        with open(results_file, 'a') as file: 
            file.write(str(log_likelihood) + "\n") 

    with open(results_file, 'a') as file: 
        file.write(str(sum_nll) + "\n") 
