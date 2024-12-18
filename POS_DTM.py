###################### POS Tagging ###########################################
#### Makes a document term matrix where terms = Part-of-speech tags using Spacy
#### Normalizes tag frequency by total number of words in the document
#### Takes as input a directory of .txt files
#### Outputs a .csv file where rows = filenames, columsn = POS tags

import os
import spacy
from collections import Counter
import pandas as pd

# Load SpaCy's English model
nlp = spacy.load("en_core_web_sm")

# Function to calculate normalized fine-grained POS frequency using SpaCy
def calculate_pos_frequencies_spacy(text):
    # Process the text with SpaCy to obtain fine-grained POS tags
    doc = nlp(text)
    
    # Get the total number of words
    total_words = len(doc)
    
    # Count the frequency of each fine-grained POS tag
    pos_counts = Counter(token.tag_ for token in doc)
    
    # Normalize the frequencies by dividing by the total number of words
    normalized_pos_freq = {tag: count / total_words for tag, count in pos_counts.items()}
    
    return normalized_pos_freq

# Function to process all .txt files in a directory
def process_text_files_in_directory(directory_path):
    # List all .txt files in the directory
    text_files = [f for f in os.listdir(directory_path) if f.endswith('.txt')]
    
    # Dictionary to store the POS frequencies for each file
    data = {}
    
    # Process each file
    for file_name in text_files:
        file_path = os.path.join(directory_path, file_name)
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
            pos_frequencies = calculate_pos_frequencies_spacy(text)
            data[file_name] = pos_frequencies
    
    # Create a DataFrame from the results
    df = pd.DataFrame.from_dict(data, orient='index').fillna(0)
    
    # Reorder the columns alphabetically for easier comparison
    df = df.reindex(sorted(df.columns), axis=1)
    
    # Save the DataFrame to a CSV file
    df.to_csv('fine_grained_pos_document_term_matrix_spacy.csv')
    
    # Optionally, display the DataFrame
    print(df)

# Set the directory path where the .txt files are located
directory_path = ''  # Replace with your actual directory path

# Process the text files
process_text_files_in_directory(directory_path)



