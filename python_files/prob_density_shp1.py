import os
import sys
from typing import Dict
import pandas as pd

import torch
from datasets import Dataset, load_dataset, DatasetDict
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
from peft import AutoPeftModelForCausalLM



def get_shp_dataset(
    dataset_name: str = "",
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
        dataset_name,
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

    access_token_read = "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"
    access_token_write = "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"
    login(token = access_token_write)

    # Load the model
    
    model_name = "guoyu-zhang/hh_shp_1"
    dataset_name = "guoyu-zhang/shp_1"
    results_file = "hh_shp1_dpo1.txt"
    
    pretrained_model = AutoPeftModelForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float32,
        load_in_4bit=True,
    )

    # Load the tokenizer
    pretrained_tokenizer = AutoTokenizer.from_pretrained(model_name)

    with open(results_file, 'a') as file: 
        file.write("Loaded model\n")

    # 2. Load the dataset
    train_dataset = get_shp_dataset(dataset_name=dataset_name, data_dir="data", split="train")    
    train_dataset = train_dataset.filter(
        lambda x: len(x["prompt"]) + len(x["chosen"]) <= 4096
        and len(x["prompt"]) + len(x["rejected"]) <= 4096
    )
    
    full_dataset = train_dataset.train_test_split(test_size=0.4)
    ff = full_dataset['test'].train_test_split(test_size=0.75)
    eval_dataset = ff['test']

    with open(results_file, 'a') as file: 
        file.write("Loaded dataset\n")   
    
    combined_chosen = []
    combined_rejected = []
    for d in eval_dataset:
        c_c = d['prompt'] + d['chosen']
        c_r = d['prompt'] + d['rejected']
        combined_chosen.append(c_c)
        combined_rejected.append(c_r)
    # Loop over the dataset, get the prompt, generate a response with both PRETRAINED and FINE TUNED models.
    # Get the log probs for each of these for the JUDGE model
    
    input_texts = combined_chosen

    all = []
    for i in input_texts:
        pretrained_tokenizer.pad_token = pretrained_tokenizer.eos_token
        input_ids = pretrained_tokenizer(i, padding=True, return_tensors="pt").input_ids
        outputs = pretrained_model(input_ids, labels=input_ids)
        all.append(outputs.loss.item())
        with open(results_file, 'a') as file: 
            file.write(str(outputs.loss.item()) + "\n") 
    
    
    with open(results_file, 'a') as file: 
        file.write(str(sum(all)) + "\n") 
    
    
    # win_rate_count = 0
    # counter = 0
    # sum_nll = 0
    # for example in eval_dataset:
    #     log_likelihood = 0
    #     counter += 1 
            
    #     prompt = example['prompt']
    #     chosen = example['chosen']
        
    #     combined_text = prompt + chosen
    #     tokenized_text = pretrained_tokenizer.encode(combined_text, return_tensors="pt")

    #     # Calculate log probability
    #     with torch.no_grad():
    #         outputs = pretrained_model(tokenized_text, labels=tokenized_text)
    #         log_likelihood = outputs.loss.item() * -1
    #         sum_nll += log_likelihood
            
    #     with open(results_file, 'a') as file: 
    #         file.write(str(log_likelihood) + "\n") 
            

    # with open(results_file, 'a') as file: 
    #     file.write(str(sum_nll) + "\n") 



