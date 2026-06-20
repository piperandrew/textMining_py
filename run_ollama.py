####### Runs a model on your laptop through ollama installation
####### Takes as input a dataframe with a column called "text"
####### Produces a new dataframe with a new column named after the model, which is a response to the prompt
import pandas as pd
import ollama
import subprocess
import time

model_name = "qwen3:30b"

temperature = 0

#prompt_template = """Here is a portion of a novel. Please summarize it IN ENGLISH by listing the TEN most significant events and plot developments in the order they appear. 
#    Write your answer IN ENGLISH. Here is the passage {sentence}"""

prompt_template = """I will give you a summary whose plot arc I would like you to summarize IN ENGLISH.

For each section, provide 2–3 sentences (no more than 60 words) IN ENGLISH.
DO NOT use character names or specific place names. Always describe characters by their role
(main character, secondary character, partner, spouse, sibling, helper, antagonist, etc.).

Provide:
- Beginning: What are the major events that initiate the story?
- Conflict: What is the central problem, obstacle, or tension that drives the story forward? What must the characters overcome?
- Ending: What are the major events that conclude the story? Provide a neutral description.

Here is the novel: {sentence}"""

# Pull the model if it's not already available
subprocess.run(["ollama", "pull", model_name], check=True)

# Function to generate the moral of the sentence
def get_moral_of_sentence(sentence):
    prompt = prompt_template.format(sentence=sentence)
    response = ollama.generate(model=model_name, prompt=prompt, options={"temperature": temperature})
    return response['response']

# Read the table of sentences
df = pd.read_csv('text.csv')
start_time = time.time()

# Create a new column for the morals
df[model_name] = df['text'].apply(get_moral_of_sentence)

# Save the results to a new CSV file
df.to_csv('text_with_answers.csv', index=False)
print(f"Processing complete in {round(time.time() - start_time, 2)} seconds.")

print('\a')

