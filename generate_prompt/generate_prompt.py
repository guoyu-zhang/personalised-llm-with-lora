import time
import json
from openai import OpenAI

assistant_id = "asst_lIA99J4dFfBPYhS5Z0fEDOU6"
prompts = ["What to do on a rainy day?", "How do I get fit?"]

def get_key():
    # Configure your OpenAI API key
    with open('openai_key.txt', 'r') as file:
        for l in file:
            key = l
    return key


def parse_response(client, thread):
    messages = client.beta.threads.messages.list(
        thread_id=thread.id
    )
    messages = messages.data
    responses = []
    
    for m in messages:
        responses.append(m.content[0].text.value)
    
    return responses
        

def get_responses(client, thread, prompt):
    message = client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=prompt
            )
        
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant_id,
        # instructions="Please address the user as Jane Doe. The user has a premium account."
        )
    
    
    while run.status in ['queued', 'in_progress', 'cancelling']:
        time.sleep(1) # Wait for 1 second
        run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id
        )
        
    return thread


def main(): 
    key = get_key() 
    client = OpenAI(api_key=key)
    thread = client.beta.threads.create()

    for p in prompts:
        thread = get_responses(client, thread, p)
        

    responses = parse_response(client, thread)
    
    with open("prompt_responses.txt", "w") as file:
        for r in responses:
            file.write(r)
            file.write("\n")

    
if __name__ == "__main__":
    main()
