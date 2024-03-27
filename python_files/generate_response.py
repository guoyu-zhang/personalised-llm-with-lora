from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import AutoPeftModelForCausalLM
import torch


if __name__ == "__main__":
    
    # -------------- PRETRAINED MODEL --------------
    
    # Load the model
    pretrained_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-chat-hf",
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        load_in_4bit=True,
    )

    # Load the tokenizer
    pretrained_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

    # -------------- FINE TUNED MODEL --------------

    # Load the model
    fine_tuned_model = AutoPeftModelForCausalLM.from_pretrained(
        "./results_user1",
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        load_in_4bit=True,
    )

    # Load the tokenizer
    fine_tuned_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    

    # Prepare the prompt
    prompt = "How can I start saving money to pay off my credit card? Assistant:"
    
    # -------------- PRETRAINED MODEL --------------
    inputs = pretrained_tokenizer.encode(prompt, return_tensors="pt")
    inputs = inputs.to('cuda')
    # Generate a response
    outputs = pretrained_model.generate(inputs, max_length=500, num_return_sequences=1)

    # Decode the generated response
    generated_text = pretrained_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print(generated_text)
    print("--------------")    
    # -------------- FINE TUNED MODEL --------------
    inputs = fine_tuned_tokenizer.encode(prompt, return_tensors="pt")
    inputs = inputs.to('cuda')
    # Generate a response
    outputs = fine_tuned_model.generate(input_ids=inputs)

    # Decode the generated response
    generated_text = fine_tuned_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print(generated_text)
