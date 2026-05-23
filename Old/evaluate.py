from datasets import load_dataset
from huggingface_hub import login
import sacrebleu
from tqdm import tqdm
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

login()

device = "cuda"

SRC_LANG = "eng_Latn"
TGT_LANG = "fra_Latn"
SPLIT = "devtest" 

model_name = "swiss-ai/Apertus-8B-2509"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
).to(device)

src_data = load_dataset("openlanguagedata/flores_plus", SRC_LANG, split=SPLIT)
tgt_data = load_dataset("openlanguagedata/flores_plus", TGT_LANG, split=SPLIT)

src_dict = {ex["id"]: ex["text"] for ex in src_data}
tgt_dict = {ex["id"]: ex["text"] for ex in tgt_data}

pairs = [(src_dict[i], tgt_dict[i]) for i in src_dict if i in tgt_dict]

print(f"Loaded {len(pairs)} sentence pairs")

def translate_fn(texts):
    prompts = [
        f"English: {text}\nFrench:"
        for text in texts
    ]

    # Tokenize as a batch
    model_inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(model.device)

    # Generate in batch
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=128,
        do_sample=False
    )

    outputs = []

    # Extract only the generated continuation per sample
    for i in range(len(texts)):
        input_len = model_inputs.input_ids[i].shape[0]
        output_ids = generated_ids[i][input_len:]

        decoded = tokenizer.decode(output_ids, skip_special_tokens=True)

        # Clean output (truncate at first newline)
        cleaned = decoded.strip().split("\n")[0]
        outputs.append(cleaned)

    return outputs


BATCH_SIZE = 16
hypotheses = []
references = []

for i in tqdm(range(0, len(pairs), BATCH_SIZE)):
    batch = pairs[i:i+BATCH_SIZE]
    
    src_batch = [x[0] for x in batch]
    ref_batch = [x[1] for x in batch]
    
    preds = translate_fn(src_batch)
    
    hypotheses.extend(preds)
    references.extend(ref_batch)

bleu = sacrebleu.corpus_bleu(hypotheses, [references])

print("\n=== BLEU SCORE ===")
print(bleu.score)