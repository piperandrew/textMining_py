import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import math

# ============================================================
# CONFIGURATION - Edit these as needed
# ============================================================
INPUT_CSV = "input.csv"           # Your input file
OUTPUT_CSV = "output.csv"         # Results will be saved here
TEXT_COLUMN = "text"              # Column name containing the text
ID_COLUMN = "id"                  # Column name for passage ID (change if different)
MODEL_NAME = "EleutherAI/gpt-neo-1.3B"
TOKEN_LENGTHS = [10, 20, 30, 40, 50]

# ============================================================
# LOAD MODEL AND TOKENIZER
# ============================================================
print(f"Loading model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
print(f"Model loaded on: {device}")

# ============================================================
# PERPLEXITY FUNCTION
# ============================================================
def calculate_perplexity(text, tokenizer, model, device):
    """Calculate perplexity of a text passage."""
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
    
    perplexity = math.exp(loss.item())
    return perplexity

# ============================================================
# MAIN PROCESSING
# ============================================================
print(f"Reading input CSV: {INPUT_CSV}")
df = pd.read_csv(INPUT_CSV)

results = []

for idx, row in df.iterrows():
    passage_id = row[ID_COLUMN]
    full_text = row[TEXT_COLUMN]
    
    # Tokenize the full text once
    full_tokens = tokenizer.encode(full_text)
    
    for length in TOKEN_LENGTHS:
        # Extract first N tokens
        truncated_tokens = full_tokens[:length]
        
        # Skip if the passage doesn't have enough tokens
        if len(truncated_tokens) < length:
            print(f"  Passage {passage_id}: only {len(truncated_tokens)} tokens, skipping length {length}")
            continue
        
        # Decode back to text
        truncated_text = tokenizer.decode(truncated_tokens)
        
        # Calculate perplexity
        ppl = calculate_perplexity(truncated_text, tokenizer, model, device)
        
        results.append({
            "passage_id": passage_id,
            "token_length": length,
            "passage_text": truncated_text,
            "perplexity": ppl
        })
    
    # Progress indicator
    if (idx + 1) % 10 == 0:
        print(f"Processed {idx + 1}/{len(df)} passages")

# ============================================================
# SAVE RESULTS
# ============================================================
results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_CSV, index=False)
print(f"Results saved to: {OUTPUT_CSV}")
print(f"Total rows in output: {len(results_df)}")