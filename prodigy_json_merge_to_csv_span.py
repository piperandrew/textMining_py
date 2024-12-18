#this script takes .json files from Prodigy span annotation as input
#it merges as many individual span annotations into a single .csv
#the final table includes: FileID, Text, TextSpan, Begin and End Token of the span, annotatorID and the label

import json
import csv
import os

# Files provided
json_files = ["mode_AS.json", "mode_EA.json", "mode_NW.json", "mode_AL.json"]

# Define the output file
output_csv = "merged_output_with_span.csv"

# Columns for the CSV
columns = ["FileID", "AnnotatorID", "Text", "Start", "End", "Label", "Span"]

# Open the CSV for writing
with open(output_csv, mode="w", newline="", encoding="utf-8") as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=columns)
    writer.writeheader()  # Write the header row

    # Process each file
    for json_file in json_files:
        annotator_id = os.path.splitext(json_file)[0]  # Use filename as AnnotatorID

        # Load JSON data (handling multiple JSON objects)
        with open(json_file, "r", encoding="utf-8") as file:
            for line in file:
                if line.strip():  # Ignore empty lines
                    item = json.loads(line.strip())
                    
                    # Extract spans
                    file_id = item.get("fileID")
                    text = item.get("text")
                    spans = item.get("spans", [])

                    for span in spans:
                        # Reconstruct the span text
                        span_text = text[span.get("start"):span.get("end")]

                        # Write the row
                        writer.writerow({
                            "FileID": file_id,
                            "AnnotatorID": annotator_id,
                            "Text": text,
                            "Start": span.get("start"),
                            "End": span.get("end"),
                            "Label": span.get("label"),
                            "Span": span_text
                        })

print(f"CSV with spans has been created: {output_csv}")
