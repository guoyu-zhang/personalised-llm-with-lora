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
    sanity_check: bool = True,
    cache_dir: str = None,
    num_proc=24,
    split="train",
) -> Dataset:
    """Load the Anthropic helpfulness and harmlessness dataset from Hugging Face and convert it to the necessary format.

    The dataset is converted to a dictionary with the following structure:
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
    results_file = "winrate_pretrained_vs_falcon_by_openchat.txt"
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
        file.write("Loaded pretrained model\n")

    # -------------- JUDGE MODEL --------------
    # "mistralai/Mixtral-8x7B-Instruct-v0.1"

    # Load the model
    judge_model = AutoModelForCausalLM.from_pretrained(
        "openchat/openchat_3.5",
        low_cpu_mem_usage=True,
        torch_dtype=torch.float32,
        load_in_4bit=True,
    )

    # Load the tokenizer
    judge_tokenizer = AutoTokenizer.from_pretrained("openchat/openchat_3.5")
    
    with open(results_file, 'a') as file: 
        file.write("Loaded judge model\n")  
    
    # -------------- REFERENCE MODEL --------------
    # openchat/openchat_3.5 - 3% fine tuned model wins
    
    
    # tiiuae/falcon-7b-instruct - 26.5% pretrained model wins
    # tiiuae/falcon-7b-instruct - 100% fine tuned model wins
    # berkeley-nest/Starling-LM-7B-alpha - 100% fine tuned model wins
    # openchat/openchat-3.5-0106 - 75.18% pretrained model wins
    # openchat/openchat-3.5-0106 - 100% fine tuned model wins

    # Load the model
    ref_model = AutoModelForCausalLM.from_pretrained(
        "tiiuae/falcon-7b-instruct",
        low_cpu_mem_usage=True,
        torch_dtype=torch.float32,
        load_in_4bit=True,
    )

    # Load the tokenizer
    ref_tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b-instruct")

    with open(results_file, 'a') as file: 
        file.write("Loaded reference model\n")   
    

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
    for example in eval_dataset:

        with open(results_file, 'a') as file: 
            file.write("Iteration: " + str(counter) + "\n")   
            
        log_probs_pretrained, log_probs_fine_tuned = 0, 0
        prompt = example['prompt']
        
        # -------------- PRETRAINED MODEL --------------
        inputs = pretrained_tokenizer.encode(prompt, return_tensors="pt")
        inputs = inputs.to('cuda')

        # Generate a response
        outputs = pretrained_model.generate(input_ids=inputs, max_length=500, num_return_sequences=1)

        # Decode the generated response
        generated_text = pretrained_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        with open(results_file, 'a') as file: 
            file.write("PRETRAINED" + "\n" + generated_text + "\n" + "--------------" + "\n") 
        
        tokenized_text = judge_tokenizer.encode(generated_text, return_tensors="pt")
        with torch.no_grad():
            outputs = judge_model(tokenized_text, labels=tokenized_text)
            log_probs_pretrained = outputs.loss.item() * -1
            
            with open(results_file, 'a') as file: 
                file.write("pretrained: " + str(log_probs_pretrained) + "\n") 
            
        # -------------- REFRENCE MODEL --------------
        inputs = ref_tokenizer.encode(prompt, return_tensors="pt")
        inputs = inputs.to('cuda')

        # Generate a response
        outputs = ref_model.generate(input_ids=inputs, max_length=500, num_return_sequences=1)

        # Decode the generated response
        generated_text = ref_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        with open(results_file, 'a') as file: 
                file.write("REFERENCE" + "\n" + generated_text + "\n" + "--------------" + "\n") 
        
        tokenized_text = judge_tokenizer.encode(generated_text, return_tensors="pt")
        with torch.no_grad():
            outputs = judge_model(tokenized_text, labels=tokenized_text)
            log_probs_ref = outputs.loss.item() * -1
            with open(results_file, 'a') as file: 
                file.write("reference: " + str(log_probs_ref) + "\n") 
    
        if log_probs_ref < log_probs_pretrained:
            win_rate_count += 1
        
        counter += 1
        # Calculate the ongoing winrate
        win_rate = win_rate_count / counter
        
        with open(results_file, 'a') as file: 
            file.write(f"Win Rate: {win_rate:.2%}") 
            file.write("\n") 
            
        if counter == 100:
            break
        

    # Calculate the winrate
    win_rate = win_rate_count / len(eval_dataset)
    sys.stdout.write(f"Win Rate: {win_rate:.2%}")
    
    

