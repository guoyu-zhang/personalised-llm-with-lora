from typing import Dict

from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import AutoPeftModelForCausalLM


def get_anthropic_hh_dataset(
    data_dir: str = "data",
    sanity_check: bool = False,
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
    
    # Load the model
    model = AutoPeftModelForCausalLM.from_pretrained(
        "./results/cfinal_checkpoint",
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        load_in_4bit=True,
    )

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

    # 3. Load evaluation dataset
    eval_dataset = get_anthropic_hh_dataset(data_dir="data", sanity_check=True, split="test")
    eval_dataset = eval_dataset.filter(
        lambda x: len(x["prompt"]) + len(x["chosen"]) <= 1024
        and len(x["prompt"]) + len(x["rejected"]) <= 1024
    )
    
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






