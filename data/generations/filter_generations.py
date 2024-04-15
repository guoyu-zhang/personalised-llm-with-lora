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
    removed = 0
    for i, entry in enumerate(data):
        print("\n\n\n\n\n\n\n\n")
        print("--------------------", i+1, "/",len(data), "----------", " removed:", removed)
        print("Instruction:", entry['instruction'], '\n')
        
        output = entry['output']
        responses = output.split("\n")
        print("1:", responses[0][3:], "\n")
        print("2:", responses[1][3:])
        print(entry['theme'])

        # # Prompt the user to choose between the responses
        # user_input = input("x or pass: ")

        # while user_input and user_input[-1] not in ['x']:
        #     user_input = input("x or pass: ")
        
        # if user_input == 'x':
        #     removed += 1
        # # If the user chooses to keep the entry, add it to the filtered data
        # entry['chosen'] = user_input
        entry['chosen'] = ""

        most_sim = entry['most_similar_instructions']
        avg_sim = entry['avg_similarity_score']
        del entry['output']
        del entry['most_similar_instructions']
        del entry['avg_similarity_score']
        
        entry['response_1'] = responses[0][3:]
        entry['response_2'] = responses[1][3:]
        entry['most_similar_instructions'] = most_sim
        entry['avg_similarity_score'] = avg_sim
        
        

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
input_file = 'regen_0.9.json'
output_file = 'regen_filtered_0.9.json'
filter_json_entries(input_file, output_file)
