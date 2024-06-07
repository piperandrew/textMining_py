########### Extracts Nouns from Column of Sentences in Table #########

# You can turn "stem" on/off
stem = True  # Set to True to enable stemming, False to disable

# Lowercases and removes punctuation prior to extracting nouns.
# Creates new columns called "nouns_noStem" and "nouns_withStem" to the same table and outputs as output.csv

import pandas as pd
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.stem import PorterStemmer

# Ensure you have the required NLTK data files
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Initialize the PorterStemmer
stemmer = PorterStemmer()

# Function to preprocess and extract nouns from a sentence
def extract_nouns(sentence, stem=False):
    if pd.isna(sentence):
        return ""
    # Convert to string and lowercase
    sentence = str(sentence).lower()
    # Remove punctuation
    sentence = re.sub(r'[^\w\s]', '', sentence)
    # Tokenize
    words = word_tokenize(sentence)
    # POS tagging
    pos_tags = pos_tag(words)
    
    # Extract nouns
    if stem:
        nouns = [stemmer.stem(word) for word, pos in pos_tags if pos.startswith('NN')]
        return ' '.join(nouns)
    else:
        nouns = [word for word, pos in pos_tags if pos.startswith('NN')]
        return ' '.join(nouns)

# Read the input CSV file
input_file = 'CHICAGO_GENERICS_ALL_Cleaned.csv'
df = pd.read_csv(input_file)

# Apply the noun extraction function to the "sentences" column without stemming
df['nouns_noStem'] = df['sentences'].apply(lambda x: extract_nouns(x, stem=False))

# Apply the noun extraction function to the "sentences" column with stemming if stemming is enabled
if stem:
    df['nouns_withStem'] = df['sentences'].apply(lambda x: extract_nouns(x, stem=True))

# Save the modified dataframe to a new CSV file
output_file = 'output.csv'
df.to_csv(output_file, index=False)

print("POS tagging, preprocessing, and extraction of nouns completed. The output is saved to", output_file)
