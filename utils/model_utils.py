import numpy as np
import transformers, sklearn, os, random, torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from train.contrastive_learning import ContrastiveLoss
from tqdm import tqdm

device = "cuda"



def get_embeddings(model, tokenizer, data_loader, pooler_type, max_context_len):
    """A function that takes in a model, tokenizer, parallel evaluation dataset 
    and returns a dictionary containing the per-sequence pooled embeddings of both the source and target
    language. 
    """

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    cl_criterion = ContrastiveLoss(
        pooler_type,
        None, # we don't use hard negatives pooler
        temp = -1 # we don't use the contrastive loss so set dummy value
    )

    cl_embeddings = {"src_embeddings": [] ,"target_embeddings": []}

    model.eval()
    with torch.no_grad():
        for batch in tqdm(data_loader, desc = "Extracting embeddings"):

            # --- CL loss ---
            src_enc = tokenizer(batch["src"], return_tensors="pt", padding="max_length", truncation=True, max_length=max_context_len).to(device)
            tgt_enc = tokenizer(batch["target"], return_tensors="pt", padding="max_length", truncation=True, max_length=max_context_len).to(device)
            src_outputs, tgt_outputs = cl_criterion.backbone_forward(
                get_decoder_backbone(model),
                src_enc["input_ids"], src_enc["attention_mask"], tgt_enc["input_ids"],
                 tgt_enc["attention_mask"]
            )

            cl_embeddings["src_embeddings"].append(cl_criterion.pooler(src_enc["attention_mask"], src_outputs))
            cl_embeddings["target_embeddings"].append(cl_criterion.pooler(tgt_enc["attention_mask"], tgt_outputs))

    cl_embeddings["src_embeddings"] =  torch.cat(cl_embeddings["src_embeddings"], dim = 0)
    cl_embeddings["target_embeddings"] =  torch.cat(cl_embeddings["target_embeddings"], dim = 0)

    return cl_embeddings["src_embeddings"], cl_embeddings["target_embeddings"]



def load_model_and_tokenizer(model_path, cache_dir):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        cache_dir=cache_dir,
        torch_dtype=torch.bfloat16,
    ).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        cache_dir=cache_dir,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def save_model_and_tokenizer(model, tokenizer, output_dir):
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def load_peft_model(base_model_path, adapter_dir, cache_dir):
    base_model, tokenizer = load_model_and_tokenizer(base_model_path, cache_dir)
    model = PeftModel.from_pretrained(base_model, adapter_dir).to("cuda")
    return model, tokenizer


def get_decoder_backbone(model):
    """Return the inner transformer stack (no lm_head) for either a PeftModel or a bare CausalLM."""
    from peft import PeftModel as _PeftModel
    if isinstance(model, _PeftModel):
        # PeftModel → LoraModel → HF CausalLM → inner decoder
        return model.base_model.model.model
    return model.model
