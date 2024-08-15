#This script takes as input a table with two columns: "text" "id"
#Text == a passage of any length
#id == a unique identifier for each passage
#It tokenizes the sentences of the passage and then further splits on independent clauses
#It outputs a two columned table with clauses and id from the primary table

import pandas as pd
import spacy
from nltk.tokenize import PunktSentenceTokenizer

# Load the English tokenizer, POS tagger, parser, and NER from SpaCy
nlp = spacy.load("en_core_web_sm")

# Load the uploaded CSV file
input_csv_path = 'NarraDetect_Clauses.csv'
input_df = pd.read_csv(input_csv_path)

# Function to split sentences into independent clauses
def split_into_independent_clauses(sentence):
    """
    Split a sentence into independent clauses based on the presence of conjunctions
    and the presence of a subject and verb in the clause.
    """
    doc = nlp(sentence)
    independent_clauses = []
    clause = []
    has_subject = False
    has_verb = False

    for token in doc:
        clause.append(token.text)

        # Check if the current token is a subject or a verb
        if token.dep_ in ("nsubj", "nsubjpass"):
            has_subject = True
        if token.pos_ in ("VERB", "AUX"):
            has_verb = True

        # Check if the token is a coordinating conjunction (cc) or punctuation that can split independent clauses
        if token.dep_ == "cc" or (token.dep_ == "punct" and token.text in {".", ";"}):
            # Split if the current clause has both a subject and a verb
            if has_subject and has_verb:
                independent_clauses.append(" ".join(clause).strip())
                clause = []
                has_subject = False
                has_verb = False

    # Add the last clause if any tokens remain and it forms an independent clause
    if clause and has_subject and has_verb:
        independent_clauses.append(" ".join(clause).strip())

    return independent_clauses

# Prepare the output DataFrame
output_data = []

for index, row in input_df.iterrows():
    text_id = row["id"]
    text = row["text"]
    
    # Tokenize text into sentences
    punkt_tokenizer = PunktSentenceTokenizer()
    sentences = punkt_tokenizer.tokenize(text)
    
    # Tokenize each sentence into independent clauses
    for sentence in sentences:
        independent_clauses = split_into_independent_clauses(sentence)
        
        # Add each clause with its corresponding text_id to the output data
        for clause in independent_clauses:
            output_data.append({"text": clause, "id": text_id})

# Create the output DataFrame
output_df = pd.DataFrame(output_data)

# Save the output DataFrame to a CSV file
output_csv_path = 'independent_clauses_output.csv'
output_df.to_csv(output_csv_path, index=False)

output_csv_path
