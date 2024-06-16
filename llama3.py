####### Runs llama3:8b model on your laptop through ollama installation
####### Takes as input a dataframe with a column called "sentence"
####### Produces a new dataframe with a new column (here called "moral") which is a response to the prompt

import pandas as pd
import ollama
import subprocess

# Pull the model if it's not already available
subprocess.run(["ollama", "pull", "llama3:8b"], check=True)

# Function to generate the moral of the sentence
def get_moral_of_sentence(sentence):
    prompt = f'What is the moral of this sentence? State your answer as a single keyword or phrase {sentence}'
    response = ollama.generate(model='llama3:8b', prompt=prompt)
    return response['response']

# Read the table of sentences
df = pd.read_csv('sentences.csv')

# Create a new column for the morals
df['moral'] = df['sentence'].apply(get_moral_of_sentence)

# Save the results to a new CSV file
df.to_csv('sentences_with_morals.csv', index=False)

print("Processing complete. Results saved to sentences_with_morals.csv.")

