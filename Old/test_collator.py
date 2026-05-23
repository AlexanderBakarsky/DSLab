from data import CLCollator, tokenize, format, source_lang, target_lang
import data
from transformers import AutoTokenizer

MODEL = "swiss-ai/Apertus-8B-2509"  # replace with actual model name

def test_cl_collator():
    data.tokenizer = AutoTokenizer.from_pretrained(MODEL)
    data.source_lang = "de"
    data.target_lang = "en"

    # Minimal fake translation pairs
    pairs = [
        {"translation": {"de": "Das ist ein Test.", "en": "This is a test."}},
        {"translation": {"de": "Guten Morgen.", "en": "Good morning."}},
        {"translation": {"de": "Wie geht es dir?", "en": "How are you?"}},
        {"translation": {"de": "Das Wetter ist schön.", "en": "The weather is nice."}},
    ]

    formatted  = [format(p) for p in pairs]
    tokenized  = [tokenize(f) for f in formatted]

    collator = CLCollator(data.tokenizer)
    batch    = collator(tokenized)

    print("=== CL ===")
    print(f"input_ids shape:      {batch['input_ids'].shape}")       # [bs, 2, cl_seq_len]
    print(f"attention_mask shape: {batch['attention_mask'].shape}")  # [bs, 2, cl_seq_len]

    print("\n=== CLM ===")
    print(f"clm_input_ids shape:      {batch['clm_input_ids'].shape}")      # [bs, clm_seq_len]
    print(f"clm_target_ids shape:     {batch['clm_target_ids'].shape}")     # [bs, clm_seq_len]
    print(f"clm_attention_mask shape: {batch['clm_attention_mask'].shape}") # [bs, clm_seq_len]

    # Verify prompt is masked in target
    for i in range(len(pairs)):
        input_ids   = batch["clm_input_ids"][i].tolist()
        target_ids  = batch["clm_target_ids"][i].tolist()
        prompt_mask = [t == -100 for t in target_ids]
        first_target = next((j for j, m in enumerate(prompt_mask) if not m), None)
        print(f"\nExample {i}: prompt_len={first_target}, total_len={sum(t != -100 for t in target_ids)} target tokens")
        print(f"  clm_input decoded:  {data.tokenizer.decode([t for t in input_ids if t != data.tokenizer.pad_token_id])}")
        print(f"  target decoded:     {data.tokenizer.decode([t for t in target_ids if t != -100])}")

    # Verify CL src/tgt are separate
    print("\n=== CL pair check ===")
    for i in range(len(pairs)):
        src_ids = batch["input_ids"][i][0].tolist()
        tgt_ids = batch["input_ids"][i][1].tolist()
        print(f"Example {i} src: {data.tokenizer.decode([t for t in src_ids if t != data.tokenizer.pad_token_id])}")
        print(f"Example {i} tgt: {data.tokenizer.decode([t for t in tgt_ids if t != data.tokenizer.pad_token_id])}")

if __name__ == "__main__":
    test_cl_collator()
