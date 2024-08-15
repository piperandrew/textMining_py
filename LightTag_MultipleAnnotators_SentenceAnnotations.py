import json
import pandas as pd

# Load the JSON file
# Replace 'your_json_file_path.json' with the actual path to your JSON file
file_path = 'your_json_file_path.json'
with open(file_path, 'r') as file:
    data = json.load(file)

# Extract relevant information for the CSV including annotator details
records = []

for example in data.get("examples", []):
    example_id = example.get("example_id")
    content = example.get("content")
    metadata = example.get("metadata", {})
    filename = metadata.get("Filename")
    annotations = example.get("annotations", [])
    
    for annotation in annotations:
        annotators = annotation.get("annotated_by", [])
        for annotator in annotators:
            record = {
                "example_id": example_id,
                "content": content,
                "filename": filename,
                "annotation_start": annotation.get("start"),
                "annotation_end": annotation.get("end"),
                "annotation_tag": annotation.get("tag"),
                "annotation_value": annotation.get("value"),
                "annotator": annotator.get("annotator"),
            }
            records.append(record)

# Convert to DataFrame
df = pd.DataFrame(records)

# Save to CSV
# Replace 'your_output_csv_file_path.csv' with the desired path for the output CSV file
csv_file_path = 'your_output_csv_file_path.csv'
df.to_csv(csv_file_path, index=False)

print(f"CSV file has been saved to {csv_file_path}")
