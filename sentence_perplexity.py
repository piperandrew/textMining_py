# ============================================================
# MULTI-MODEL PERPLEXITY CALCULATOR
# ============================================================
#
# SUMMARY:
# This script calculates perplexity scores for text passages using multiple
# language models. Perplexity measures how "surprised" a model is by text,
# where lower perplexity = more predictable/typical language patterns.
#
# WHAT IT DOES:
#   1. Loads each specified language model in sequence
#   2. For each passage in the input CSV:
#      - Truncates to various token lengths (10, 20, 30, 40, 50 tokens)
#      - Calculates perplexity for each truncated version
#   3. Outputs results with model name, passage ID, token length, and perplexity
#
# INPUT FORMAT (CSV):
#   - id: Unique identifier for each passage
#   - text: The text passage to analyze
#
# OUTPUT FORMAT (CSV):
#   - model: Name of the language model used
#   - passage_id: ID from input file
#   - token_length: Number of tokens used (10, 20, 30, 40, or 50)
#   - passage_text: The actual truncated text analyzed
#   - perplexity: The perplexity score (lower = more predictable)
#
# MODELS USED:
#   - EleutherAI/gpt-neo-1.3B: 1.3 billion parameter GPT-Neo model
#   - HuggingFaceTB/SmolLM2-1.7B: 1.7 billion parameter SmolLM model  
#   - Qwen/Qwen2.5-0.5B: 0.5 billion parameter Qwen model
#
# USAGE:
#   1. Set up a clean Python environment with torch, transformers, pandas
#   2. Place your input CSV in the same directory
#   3. Run: python perplexity_script.py
#   4. Results saved to output CSV
#
# ============================================================

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
ID_COLUMN = "id"                  # Column name for passage ID

# List of models to run
# Each model will process all passages, then be unloaded before the next
MODEL_NAMES = [
    "EleutherAI/gpt-neo-1.3B",    # 1.3B parameter GPT-Neo
    "HuggingFaceTB/SmolLM2-1.7B", # 1.7B parameter SmolLM
    "Qwen/Qwen2.5-0.5B",          # 0.5B parameter Qwen
]

# Token lengths to test
# For each passage, we extract the first N tokens and measure perplexity
TOKEN_LENGTHS = [10, 20, 30, 40, 50]

# ============================================================
# PERPLEXITY FUNCTION
# ============================================================

def calculate_perplexity(text, tokenizer, model, device):
    """
    Calculate perplexity of a text passage.
    
    Perplexity = exp(cross-entropy loss)
    Lower values indicate the model finds the text more predictable.
    
    Args:
        text: The text string to evaluate
        tokenizer: The model's tokenizer
        model: The language model
        device: torch device (cuda or cpu)
    
    Returns:
        float: The perplexity score
    """
    # Tokenize the text
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)
    
    # Calculate loss (model predicts each token given previous tokens)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
    
    # Convert loss to perplexity
    perplexity = math.exp(loss.item())
    return perplexity

# ============================================================
# MAIN PROCESSING
# ============================================================

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Models to process: {len(MODEL_NAMES)}")
print(f"Token lengths: {TOKEN_LENGTHS}")
print("=" * 60)

# Load input data
print(f"\nReading input CSV: {INPUT_CSV}")
df = pd.read_csv(INPUT_CSV)
print(f"Found {len(df)} passages to process")

# Store all results across all models
all_results = []

# Process each model
for model_idx, model_name in enumerate(MODEL_NAMES):
    print("\n" + "=" * 60)
    print(f"MODEL {model_idx + 1}/{len(MODEL_NAMES)}: {model_name}")
    print("=" * 60)
    
    # --------------------------------------------------------
    # Load model and tokenizer
    # --------------------------------------------------------
    print(f"Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print(f"Loading model...")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Move model to GPU
    model.to(device)
    model.eval()  # Set to evaluation mode (disables dropout, etc.)
    print(f"Model loaded on: {device}")
    
    # --------------------------------------------------------
    # Process all passages with this model
    # --------------------------------------------------------
    for idx, row in df.iterrows():
        passage_id = row[ID_COLUMN]
        full_text = row[TEXT_COLUMN]
        
        # Tokenize the full text once (more efficient than re-tokenizing)
        full_tokens = tokenizer.encode(full_text)
        
        # Process each token length
        for length in TOKEN_LENGTHS:
            # Extract first N tokens
            truncated_tokens = full_tokens[:length]
            
            # Skip if the passage doesn't have enough tokens
            if len(truncated_tokens) < length:
                print(f"  Passage {passage_id}: only {len(truncated_tokens)} tokens, skipping length {length}")
                continue
            
            # Decode back to text (this is what the model actually sees)
            truncated_text = tokenizer.decode(truncated_tokens)
            
            # Calculate perplexity
            ppl = calculate_perplexity(truncated_text, tokenizer, model, device)
            
            # Store result
            all_results.append({
                "model": model_name,
                "passage_id": passage_id,
                "token_length": length,
                "passage_text": truncated_text,
                "perplexity": ppl
            })
        
        # Progress indicator every 10 passages
        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{len(df)} passages")
    
    print(f"Completed {model_name}")
    
    # --------------------------------------------------------
    # Unload model to free GPU memory before loading next model
    # --------------------------------------------------------
    print(f"Unloading model to free memory...")
    del model
    del tokenizer
    torch.cuda.empty_cache()  # Clear GPU memory

# ============================================================
# SAVE RESULTS
# ============================================================

print("\n" + "=" * 60)
print("SAVING RESULTS")
print("=" * 60)

results_df = pd.DataFrame(all_results)
results_df.to_csv(OUTPUT_CSV, index=False)

print(f"Results saved to: {OUTPUT_CSV}")
print(f"Total rows in output: {len(results_df)}")
print(f"  - {len(MODEL_NAMES)} models")
print(f"  - {len(df)} passages")
print(f"  - {len(TOKEN_LENGTHS)} token lengths")
print("\nDone!")