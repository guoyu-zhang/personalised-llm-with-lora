import os
import sys
from typing import Dict

import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
from peft import AutoPeftModelForCausalLM


def get_user_dataset(
    data_dir: str = "data",
    sanity_check: bool = False,
    cache_dir: str = None,
    num_proc=24,
    split="train",
    user_number='1',
) -> Dataset:
    """Load dataset for one user and convert it to the necessary format.

    The dataset is converted to following structure:
    {
        'prompt': List[str],
        'chosen': List[str],
        'rejected': List[str],
    }
    """
    def load_user(user_number, split):
        train_path = "../data/parquets/user" + user_number + "_train-00000-of-00001.parquet"
        valid_path = "../data/parquets/user" + user_number + "_valid-00000-of-00001.parquet"
        test_path = "../data/parquets/user" + user_number + "_test-00000-of-00001.parquet"
        user_dataset = load_dataset("parquet", data_files={'train': train_path, 'valid': valid_path, 'test': test_path})
        return user_dataset[split]
    
    dataset = load_user(user_number, split)
    original_columns = dataset.column_names

    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 1000)))
        
    def return_prompt_and_responses(samples) -> Dict[str, str]:
        return {
            "prompt": "Human: " + samples["history"] + " Assistant: ",
            "chosen": samples["human_ref_A"] if samples["labels"] == 1 else samples["human_ref_B"],
            "rejected": samples["human_ref_B"] if samples["labels"] == 1 else samples["human_ref_A"],
        }

    return dataset.map(
        return_prompt_and_responses,
        num_proc=num_proc,
        remove_columns=original_columns,
    )


if __name__ == "__main__": 
    results_file = "prob_density_user2_base_llama2.txt"
    access_token_read = "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"
    access_token_write = "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"
    login(token = access_token_write)
    # -------------- PRETRAINED MODEL --------------
    
    # Load the model
    pretrained_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-chat-hf",
        low_cpu_mem_usage=True,
        torch_dtype=torch.float32,
        load_in_4bit=True,
    )

    # Load the tokenizer
    pretrained_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

    with open(results_file, 'a') as file: 
        file.write("Loaded model\n")

    # 3. Load evaluation dataset
    eval_dataset = get_user_dataset(data_dir="data", sanity_check=False, split="test", user_number='2')


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



