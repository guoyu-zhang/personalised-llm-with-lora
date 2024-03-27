import os
import sys
from typing import Dict

import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
from peft import AutoPeftModelForCausalLM


def get_dataset(
    num_proc=24,
) -> Dataset:
    """Load dataset for one user and convert it to the necessary format.

    The dataset is converted to following structure:
    {
        'prompt': List[str],
        'chosen': List[str],
        'rejected': List[str],
    }
    """

    dataset = load_dataset(
        "allenai/pref-test-sets",
        split='pku_better',
    )
    
    original_columns = dataset.column_names

    def return_prompt_and_responses(samples):
      prompt_content = samples['prompt']
     
      return {
          "prompt": "Human: " + prompt_content[0]['content'] + " Assistant: ",
          "chosen": samples['chosen'],
          "rejected": samples['rejected'],
        }

    return dataset.map(
        return_prompt_and_responses,
        num_proc=num_proc,
        remove_columns=original_columns,
    )


if __name__ == "__main__": 
    results_file = "prob_density_pku_1200.txt"
    access_token_read = "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"
    access_token_write = "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"
    login(token = access_token_write)
    # -------------- PRETRAINED MODEL --------------
    
    # Load the model
    pretrained_model = AutoPeftModelForCausalLM.from_pretrained(
        "./user_pku_better_1200",
        low_cpu_mem_usage=True,
        torch_dtype=torch.float32,
        load_in_4bit=True,
    )

    # Load the tokenizer
    pretrained_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

    with open(results_file, 'a') as file: 
        file.write("Loaded model\n")

    # 2. Load the dataset
    all_dataset = get_dataset()   
    eval_dataset = all_dataset.select(range(8500,9000))

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



