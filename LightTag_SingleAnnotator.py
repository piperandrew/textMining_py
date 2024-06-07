import json
import pandas as pd

# Adjust the file path below to match the location of your JSON file
file_path = '/Users/akpiper/Desktop/[INSERT HERE]'

# Load the JSON file
with open(file_path, 'r') as file:
    data = json.load(file)

# Initialize an empty list to store the data for DataFrame creation
data_for_df = []

# Iterate over the examples in the data
for example in data['examples']:
    content = example.get('content', '')
    # Loop through classifications to get classname and annotator details
    for classification in example.get('classifications', []):
        classname = classification.get('classname', '')
        # The provided code seems to assume multiple annotators for a single classification
        # This could be a source of duplication if the data does not match this structure
        # We'll collect unique annotator-classname pairs to avoid duplication
        classified_by = classification.get('classified_by', [])
        for annotator_info in classified_by:
            annotator = annotator_info.get('annotator', '')
            # Append the information as a tuple to maintain uniqueness
            data_for_df.append((content, annotator, classname))

# Convert the list of tuples to a DataFrame and drop duplicates if any
df = pd.DataFrame(list(set(data_for_df)), columns=['Content', 'annotator', 'classname'])

# Save the DataFrame to a CSV file on your desktop
csv_file_path = '/Users/akpiper/Desktop/extracted_information.csv'  # Adjust as needed
df.to_csv(csv_file_path, index=False)

print(f"CSV file has been saved to: {csv_file_path}")



