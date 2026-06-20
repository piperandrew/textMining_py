cat > run_qwen3_transformers.py <<'PY'
import time
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Run:
#   python run_qwen3_transformers.py

MODEL_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507"

INFILE = "Validation_Multilingual_BookSum_PAS_Qwen.csv"
OUTFILE = "Validation_Multilingual_BookSum_PAS_Qwen_with_answers.csv"
TEXT_COL = "summary"
OUT_COL = "qwen3_30b_a3b_2507_pas"

BATCH_SIZE = 4
MAX_NEW_TOKENS = 260
DO_SAMPLE = False
TEMPERATURE = 0.0

PROMPT_TEMPLATE = """I will give you a summary whose plot arc I would like you to summarize IN ENGLISH.

For each section, provide 2–3 sentences (no more than 60 words) IN ENGLISH.
DO NOT use character names or specific place names. Always describe characters by their role
(main character, secondary character, partner, spouse, sibling, helper, antagonist, etc.).

Provide:
- Beginning: What are the major events that initiate the story?
- Conflict: What is the central problem, obstacle, or tension that drives the story forward? What must the characters overcome?
- Ending: What are the major events that conclude the story? Provide a neutral description.

Here is the novel: {sentence}"""

def make_prompt(tokenizer: AutoTokenizer, sentence: str) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": PROMPT_TEMPLATE.format(sentence=sentence)},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

def main() -> None:
    df = pd.read_csv(INFILE)

    if OUT_COL not in df.columns:
        df[OUT_COL] = pd.NA

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        padding_side="left"
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,
        device_map="auto"
    )
    model.eval()

    start = time.time()

    # Only process rows missing output, so reruns resume safely.
    idxs = [i for i in range(len(df)) if pd.isna(df.at[i, OUT_COL])]

    if len(idxs) == 0:
        print("Nothing to do: all rows already have outputs.")
        return

    for s in range(0, len(idxs), BATCH_SIZE):
        batch_idxs = idxs[s:s + BATCH_SIZE]
        sentences = [str(df.at[i, TEXT_COL]) for i in batch_idxs]
        prompts = [make_prompt(tokenizer, t) for t in sentences]

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)

        # Per-example prompt lengths (handles padding correctly)
        prompt_lens = attention_mask.sum(dim=1).tolist()

        with torch.no_grad():
            gen = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=DO_SAMPLE,
                temperature=(TEMPERATURE if DO_SAMPLE else None),
                eos_token_id=tokenizer.eos_token_id
            )

        # Strip prompt tokens per row, then decode
        decoded = []
        for row_gen, plen in zip(gen, prompt_lens):
            decoded.append(
                tokenizer.decode(row_gen[int(plen):], skip_special_tokens=True).strip()
            )

        for i, text in zip(batch_idxs, decoded):
            df.at[i, OUT_COL] = text

        # Checkpoint each batch
        df.to_csv(OUTFILE, index=False)
        done = min(s + len(batch_idxs), len(idxs))
        print(f"Saved {done}/{len(idxs)} completed rows")

    print(f"Done. Total time: {round(time.time() - start, 2)} seconds")

if __name__ == "__main__":
    main()
PY
