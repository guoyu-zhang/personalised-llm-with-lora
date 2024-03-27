import json

def filter_json_entries(input_file, output_file):
    # Load the JSON data
    with open(input_file, 'r') as file:
        data = json.load(file)

    # Open the output file initially to write the opening bracket for the JSON array
    with open(output_file, 'w') as file:
        file.write('[')

    first_entry = True

    # Loop through each entry in the JSON data
    for i, entry in enumerate(data):
        print("\n\n\n\n\n\n\n\n")
        print("--------------------", i+1, "/ 200")
        # print(f"--------------------{i}/ 200")
        print("Instruction:", entry['instruction'], '\n')
        
        output = entry['output']
        responses = output.split("\n")
        print("1:", responses[0][3:], "\n")
        print("2:", responses[1][3:])

        # Prompt the user to choose between the responses
        user_input = input("Prefer 1 or 2: ")

        while user_input not in ['1', '2']:
            user_input = input("Please input 1 or 2: ")

        # If the user chooses to keep the entry, add it to the filtered data
        entry['chosen'] = user_input

        del entry['avg_similarity_score']
        del entry['most_similar_instructions']
        del entry['input']
        del entry['output']
        
        entry['response_1'] = responses[0][3:]
        entry['response_2'] = responses[1][3:]
        

        # Write the current entry to the file
        with open(output_file, 'a') as file:
            # If it's not the first entry, prepend a comma to separate JSON objects
            if not first_entry:
                file.write(',')
            else:
                first_entry = False

            json.dump(entry, file, indent=4)
        print("\n\n\n\n\n\n\n\n")

    # Write the closing bracket for the JSON array
    with open(output_file, 'a') as file:
        file.write(']')

# Example usage
input_file = 'generations.json'
output_file = 'user3.json'
filter_json_entries(input_file, output_file)
