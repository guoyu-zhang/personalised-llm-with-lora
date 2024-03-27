import csv

# Specify the path to your CSV file
csv_file_path = 'all_data_cleared.csv'
output_csv_file_path = 'all_data_user1_2nd_iter.csv'

# Open the CSV file for reading and an output file for writing
with open(csv_file_path, mode='r', newline='') as infile, open(output_csv_file_path, mode='w', newline='') as outfile:
    reader = csv.DictReader(infile)
    # Ensure the output CSV has the same column headers as the input
    writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
    writer.writeheader()

    # Iterate over each row in the CSV
    counter = 0
    for row in reader:
        counter+=1
        # Print the desired fields
        print("**************************\n\n\n**************************\n**************************")
        print(counter)
        print(f"History: {row['history']}")
        print("------------------------")
        print(f"Human Reference A: {row['human_ref_A']}")
        print("------------------------")
        print(f"Human Reference B: {row['human_ref_B']}")

        # Prompt the user for input
        user_input = input("Enter 1 or 0 for the label: ")

        # Validate input and re-prompt if necessary
        while user_input not in ['1', '0']:
            print("Invalid input. Please enter either 1 or 0.")
            user_input = input("Enter 1 or 0 for the label: ")

        # Write the user input to the 'labels' column
        row['labels'] = user_input

        # Write the updated row to the output file
        writer.writerow(row)
        

print("CSV file has been updated.")
