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
        print("--------------------", i+1)
        print("History:", entry['history'], '\n')
        
        print("1:", entry['human_ref_A'], "\n")
        print("2:", entry['human_ref_B'])

        # Prompt the user to choose between the responses
        user_input = input("Prefer 1 or 2, or del: ")

        while user_input not in ['1', '2', 'del']:
            user_input = input("Please input 1 or 2 or del: ")

        # If the user chooses to keep the entry, add it to the filtered data
        entry['labels'] = user_input
     

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
input_file = 'shp_cleared.json'
output_file = 'nikki.json'
filter_json_entries(input_file, output_file)
