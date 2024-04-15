import os
import sys
from typing import Dict
import pandas as pd

import torch
from datasets import Dataset, load_dataset, DatasetDict
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
from peft import AutoPeftModelForCausalLM


def get_user_dataset(
    num_proc=24,
    split="train",
    user_number="1",
) -> Dataset:
    """Load dataset for one user and convert it to the necessary format."""
    
    train_path = "../generate_prompt/user" + user_number + ".json"
    df = pd.read_json(train_path)[:200]
    df = Dataset.from_pandas(df)
    # 80% train, 20% test + validation
    train_testvalid = df.train_test_split(test_size=0.2)
    # Split the 20% test + valid in 10:30
    test_valid = train_testvalid['test'].train_test_split(test_size=30/40)
    # gather everyone if you want to have a single DatasetDict
    train_test_valid_dataset = DatasetDict({
        'train': train_testvalid['train'],
        'test': test_valid['test'],
        'valid': test_valid['train']})
    dataset = train_test_valid_dataset[split]
    
    original_columns = dataset.column_names
        
    def return_prompt_and_responses(samples) -> Dict[str, str]:
        return {
            "prompt": "Human: " + samples["instruction"] + " Assistant: ",
            "chosen": samples["response_1"] if samples["chosen"] == 1 else samples["response_2"],
            "rejected": samples["response_2"] if samples["chosen"] == 1 else samples["response_1"],
        }

    return dataset.map(
        return_prompt_and_responses,
        num_proc=num_proc,
        remove_columns=original_columns,
    )



if __name__ == "__main__": 
    results_file = "prob_density_user1_base.txt"
    access_token_read = "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"
    access_token_write = "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"
    login(token = access_token_write)
    # -------------- PRETRAINED MODEL --------------
    # Load the model
    # pretrained_model = AutoModelForCausalLM.from_pretrained(
    #     "meta-llama/Llama-2-7b-chat-hf",
    #     low_cpu_mem_usage=True,
    #     torch_dtype=torch.float32,
    #     load_in_4bit=True,
    # )
    
    # Load the model
    pretrained_model = AutoPeftModelForCausalLM.from_pretrained(
        "./refined_user1",
        low_cpu_mem_usage=True,
        torch_dtype=torch.float32,
        load_in_4bit=True,
    )

    # Load the tokenizer
    pretrained_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

    with open(results_file, 'a') as file: 
        file.write("Loaded model\n")

    # 3. Load evaluation dataset
    eval_dataset = get_user_dataset(split="test", user_number='1')


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
        chosen = example['rejected']
        
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



