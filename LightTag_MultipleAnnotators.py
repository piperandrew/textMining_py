import json
import pandas as pd

# Load the JSON data from the uploaded file
file_path = '/Users/akpiper/Desktop/gpt-comparison-validation_annotations.json'

# Load the JSON file
with open(file_path, 'r') as file:
    data = json.load(file)

# Extract unique annotator names to create columns for each
annotator_names = set()
for example in data['examples']:
    for classification in example.get('classifications', []):
        for annotator_info in classification.get('classified_by', []):
            annotator_names.add(annotator_info['annotator'])

# Sort the annotator names to ensure consistent column order
sorted_annotator_names = sorted(list(annotator_names))

# Initialize a dictionary to hold the content and annotations
data_for_df = {'Content': []}
data_for_df.update({annotator: [] for annotator in sorted_annotator_names})

# Populate the dictionary with content and corresponding annotations
for example in data['examples']:
    data_for_df['Content'].append(example['content'])
    annotations = {annotator: '' for annotator in sorted_annotator_names}
    
    for classification in example.get('classifications', []):
        for annotator_info in classification.get('classified_by', []):
            annotator = annotator_info['annotator']
            classname = classification['classname']
            annotations[annotator] = classname
    
    for annotator, annotation in annotations.items():
        data_for_df[annotator].append(annotation)

# Convert the dictionary to a DataFrame
df = pd.DataFrame(data_for_df)

# Save the DataFrame to a CSV file
csv_file_path = '/Users/akpiper/Desktop/extracted_information.csv'  # Adjust as needed
df.to_csv(csv_file_path, index=False)

print(f"CSV file has been saved to: {csv_file_path}")
