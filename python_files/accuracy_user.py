from typing import Dict
import pandas as pd

from datasets import Dataset, load_dataset, DatasetDict
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
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
    
    # Load the model
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-chat-hf",
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        load_in_4bit=True,
    )
    
    # # Load the model
    # model = AutoPeftModelForCausalLM.from_pretrained(
    #     "./refined_user1",
    #     low_cpu_mem_usage=True,
    #     torch_dtype=torch.float16,
    #     load_in_4bit=True,
    # )

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

    # 3. Load evaluation dataset
    eval_dataset = get_user_dataset(split="test", user_number='1')

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






