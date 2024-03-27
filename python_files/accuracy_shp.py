from typing import Dict

from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
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
        split='shp',
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
    
    # Load the model
    model = AutoPeftModelForCausalLM.from_pretrained(
        "./user_shp",
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        load_in_4bit=True,
    )

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

    # 2. Load the dataset
    all_dataset = get_dataset()   
    eval_dataset = all_dataset.select(range(1400,1700))

    # Loop over the dataset and calculate the log probabilities for 'chosen' and 'rejected' responses
    correct_count = 0
    for example in eval_dataset:
        prompt = example['prompt']
        chosen = example['chosen']
        rejected = example['rejected']
        log_probs_chosen, log_probs_rejected = 0, 0
        
        tokenized_text_chosen = tokenizer.encode(prompt + chosen, return_tensors="pt")
        with torch.no_grad():
            outputs = model(tokenized_text_chosen, labels=tokenized_text_chosen)
            log_probs_chosen = outputs.loss * -1
            
        tokenized_text_rejected = tokenizer.encode(prompt + rejected, return_tensors="pt")
        with torch.no_grad():
            outputs = model(tokenized_text_rejected, labels=tokenized_text_rejected)
            log_probs_rejected = outputs.loss * -1

        
        if log_probs_chosen > log_probs_rejected:
            correct_count += 1

    # Calculate the accuracy
    accuracy = correct_count / len(eval_dataset)
    print(f"Accuracy: {accuracy:.2%}")






