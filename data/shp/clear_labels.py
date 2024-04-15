import json

def clear_labels_in_json(input_file_path, output_json_file_path=None):
    # Columns to be removed from each JSON object
    columns_to_remove = ["post_id", "upvote_ratio", "c_root_id_A", "c_root_id_B", "created_at_utc_A", "created_at_utc_B", "score_A", "score_B", "seconds_difference", "score_ratio"]

    with open(input_file_path, 'r', encoding='utf-8') as infile, open(output_json_file_path, 'w', encoding='utf-8') as outfile:
        # Load the entire JSON data from the input file
        data = json.load(infile)

        # Process each item in the JSON data
        for item in data:
            # Clear the 'labels' field
            item['labels'] = ''

            # Remove specified columns if they exist
            for column in columns_to_remove:
                item.pop(column, None)

        # Write the updated JSON data to the output file
        json.dump(data, outfile, indent=4)

if __name__ == '__main__':
    json_file_path = 'shp.json'  # Input JSON file path
    output_json_file_path = 'shp_cleared.json'  # Output JSON file path

    clear_labels_in_json(json_file_path, output_json_file_path)
