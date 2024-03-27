from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


if __name__ == "__main__":
    
    # DPO FINE TUNED MODEL ON ANTHROPIC HH DATASET    
    
    # Load the model
    model = AutoModelForCausalLM.from_pretrained(
        "./results/checkpoint-1000",
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        load_in_4bit=True,
    )

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("./results/checkpoint-1000")

    # Prepare the prompt
    prompt = "How can I start saving money to pay off my credit card? Assistant:"
    
    # --- Chosen
    chosen = "This is an excellent question! It sounds like you’ve been using your credit card for a while, and are facing debt that you want to pay down. What do you think is a good way to keep track of your spending, so that you can make sure you’re saving? How should you decide which purchases to “save” for, and when?"
        
    combined_text = prompt + ' ' + chosen
    tokenized_text = tokenizer.encode(combined_text, return_tensors="pt")

    # Calculate log probability
    with torch.no_grad():
        outputs = model(tokenized_text, labels=tokenized_text)
        log_likelihood = outputs.loss * -1

    print(f"chosen, Log likelihood of the response: {log_likelihood.item()}")
    
    # --- Rejected
    rejected = "Great! If you’re looking to save money, we can put your current cash flow on a chart to see where it might be useful to spend less. Here’s what your net income looked like in the last 6 months: Assistants? “What’s this?”"

    combined_text = prompt + ' ' + rejected
    tokenized_text = tokenizer.encode(combined_text, return_tensors="pt")

    # Calculate log probability
    with torch.no_grad():
        outputs = model(tokenized_text, labels=tokenized_text)
        log_likelihood = outputs.loss * -1

    print(f"rejected, Log likelihood of the response: {log_likelihood.item()}")
    
    # --- Random
    random = "What are there twenty flour"

    combined_text = prompt + ' ' + random
    tokenized_text = tokenizer.encode(combined_text, return_tensors="pt")

    # Calculate log probability
    with torch.no_grad():
        outputs = model(tokenized_text, labels=tokenized_text)
        log_likelihood = outputs.loss * -1

    print(f"random, Log likelihood of the response: {log_likelihood.item()}")