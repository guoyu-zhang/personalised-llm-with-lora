import json

def filter_json_entries(input_file):
    # Load the JSON data
    counter = 0
    with open(input_file, 'r') as file:
        data = json.load(file)
        
    filtered = []
    for d in data:
        if d['chosen'] == "":
            counter += 1
            filtered.append(d)

    with open("1000.json", 'w') as f:
        json.dump(filtered, f, indent=4)
    
filter_json_entries("regen_filtered923.json")
