import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, StoppingCriteria, StoppingCriteriaList
from peft import PeftModel
import torch
from datasets import load_dataset
import datasets
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from abc import ABC, abstractmethod
import numpy as np
import json, os
import datetime
from sacrebleu.metrics import BLEU, CHRF
from utils import get_decoder_backbone



def _evaluate_pair(model, data_loader, tokenizer, max_prompt_len, max_generation_context_len, use_BLEU=True, use_chrF=False, use_cl=False, cl_criterion=None, print_examples=True):
    """
    Evaluate on a string-level data_loader with batches of {"prompt": str, "target": str, "src": str}.
    Tokenizes on the fly for CLM loss, CL loss (if use_cl), and generation/BLEU/chrF.
    """
    device = model.device
    bleu = BLEU()
    chrf = CHRF()
    use_generation = use_BLEU or use_chrF
    hypotheses, references = [], []
    total_clm_loss_tokens, total_clm_tokens = 0.0, 0
    total_cl_loss, total_batches = 0.0, 0

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id


    model.eval()
    warned_generation_len = False
    warned_prompt_len = False
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for batch in tqdm(data_loader, desc= "Evaluating on a pair."):
            batch_size = len(batch["prompt"])

            # --- CLM loss: manually build right-padded [prompt + target] sequences.
            # The prompt string already starts with BOS (added by format_dataset),
            # and the target gets EOS appended below — matching make_tokenize. ---
            clm_sequences, clm_labels = [], []
            for prompt, target in zip(batch["prompt"], batch["target"]):
                prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
                target_ids = tokenizer(target, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
                if len(target_ids) > max_generation_context_len:
                    print(f"Warning: target length {len(target_ids)} exceeds max_generation_context_len {max_generation_context_len}")
                    warned_generation_len = True
                if len(prompt_ids) > max_prompt_len:
                    print(f"Warning: prompt length {len(prompt_ids)} exceeds max_prompt_len {max_prompt_len}, prompt will be truncated.")
                    warned_prompt_len = True
                clm_sequences.append(prompt_ids + target_ids)
                clm_labels.append([-100] * len(prompt_ids) + target_ids)

            max_clm_len = min(max(len(s) for s in clm_sequences), max_prompt_len + max_generation_context_len)
            clm_input_ids = torch.full((batch_size, max_clm_len), pad_id, dtype=torch.long)
            clm_attention_mask = torch.zeros(batch_size, max_clm_len, dtype=torch.long)
            clm_target_ids = torch.full((batch_size, max_clm_len), -100, dtype=torch.long)
            for i, (seq, lbl) in enumerate(zip(clm_sequences, clm_labels)):
                seq, lbl = seq[:max_clm_len], lbl[:max_clm_len]
                clm_input_ids[i, :len(seq)] = torch.tensor(seq)
                clm_attention_mask[i, :len(seq)] = 1
                clm_target_ids[i, :len(lbl)] = torch.tensor(lbl)

            out = model(
                input_ids=clm_input_ids.to(device),
                attention_mask=clm_attention_mask.to(device),
                labels=clm_target_ids.to(device),
            )
            # HF CausalLM loss is mean over scored (label != -100) tokens after
            # the internal shift; weight by that count for an unbiased corpus mean.
            n_label_tokens = (clm_target_ids[:, 1:] != -100).sum().item()
            total_clm_loss_tokens += out.loss.item() * n_label_tokens
            total_clm_tokens += n_label_tokens

            # --- CL loss ---
            if use_cl:
                # Pad src and tgt jointly so both end up with the same seq length
                # (required for the stack below).
                joint_enc = tokenizer(
                    batch["src"] + batch["target"],
                    return_tensors="pt", padding=True, truncation=True, max_length=max_prompt_len,
                    add_special_tokens=False,
                ).to(device)
                bs_cl = len(batch["src"])
                src_enc = {
                    "input_ids": joint_enc["input_ids"][:bs_cl],
                    "attention_mask": joint_enc["attention_mask"][:bs_cl],
                }
                tgt_enc = {
                    "input_ids": joint_enc["input_ids"][bs_cl:],
                    "attention_mask": joint_enc["attention_mask"][bs_cl:],
                }
                src_outputs, tgt_outputs = cl_criterion.backbone_forward(
                    get_decoder_backbone(model),
                    src_enc["input_ids"],
                    src_enc["attention_mask"],
                    tgt_enc["input_ids"],
                    tgt_enc["attention_mask"],
                )
                cl_batch = {"attention_mask": torch.stack([src_enc["attention_mask"], tgt_enc["attention_mask"]], dim=1)}
                total_cl_loss += cl_criterion(src_outputs, tgt_outputs, cl_batch).item()

            total_batches += 1

            # --- Generation for BLEU/chrF (left-padded prompts) ---
            if use_generation:
                tokenizer.padding_side = "left"
                tokenizer.truncation_side = "left"
                inputs = tokenizer(
                    batch["prompt"],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    add_special_tokens=False,
                    max_length=max_prompt_len,
                ).to(device)
                tokenizer.padding_side = "right"
                tokenizer.truncation_side = "right"
                output_ids = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=max_generation_context_len,
                    pad_token_id=pad_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

                prompt_len = inputs["input_ids"].shape[1]
                for i in range(output_ids.size(0)):
                    gen_ids = output_ids[i, prompt_len:]
                    eos_positions = (gen_ids == tokenizer.eos_token_id).nonzero()
                    if len(eos_positions) > 0:
                        gen_ids = gen_ids[: eos_positions[0].item()]
                    hyp = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                    hypotheses.append(hyp)
                    references.append(batch["target"][i].removesuffix(tokenizer.eos_token).strip())

    if print_examples:
        print("\n--- Sample generations ---")
        for i, (hyp, ref) in enumerate(zip(hypotheses[:10], references[:10])):
            print(f"[{i}] REF: {ref}")
            print(f"[{i}] HYP: {hyp}")

    torch.cuda.empty_cache()
    model.train()
    return {
        "val_clm_loss": total_clm_loss_tokens / max(total_clm_tokens, 1),
        "val_cl_loss": total_cl_loss / total_batches if use_cl else 0.0,
        "val_bleu": bleu.corpus_score(hypotheses, [references]).score if use_BLEU else 0.0,
        "val_chrf": chrf.corpus_score(hypotheses, [references]).score if use_chrF else 0.0,
        "samples": list(zip(references[:10], hypotheses[:10])) if use_generation else [],
    }


def evaluate(model, loaders, tokenizer, max_prompt_len, max_generation_context_len, use_BLEU=True, use_chrF=False, use_cl=False, cl_criterion=None, print_examples=True, aggregate = True):
    """Calls run_evaluation() once per language pair and aggregates results.

    loaders: a single DataLoader or a dict mapping lang-pair strings to DataLoaders.
    With a single loader the result is returned as-is. With multiple loaders,
    val_clm_loss / val_cl_loss / val_bleu are averaged and up to
    max(1, 10 // #pairs) samples from each language pair are concatenated.
    """
    from data.loaders import MultilingualDataLoader
    if isinstance(loaders, dict):
        loaders_dict = loaders
    elif isinstance(loaders, MultilingualDataLoader):
        loaders_dict = loaders.loaders
    else:
        loaders_dict = {"default": loaders}

    n_pairs = len(loaders_dict)
    samples_per_pair = 10

    per_pair = {
        lp: _evaluate_pair(
            model, loader, tokenizer, max_prompt_len, max_generation_context_len,
            use_BLEU=use_BLEU, use_chrF=use_chrF, use_cl=use_cl, cl_criterion=cl_criterion,
            print_examples=False,
        )
        for lp, loader in loaders_dict.items()
    }

    # Tag each sample with its language pair so downstream consumers can show
    # which pair a (ref, hyp) came from. Always keep up to 10 per pair.
    combined_samples = []
    for lp, r in per_pair.items():
        for ref, hyp in r["samples"][:samples_per_pair]:
            combined_samples.append((lp, ref, hyp))

    # aggregate=True: training-loop view (single mean across pairs).
    # aggregate=False: final-eval view (per-pair prefixed keys), regardless of n_pairs.
    if aggregate:
        result = {
            "val_clm_loss": sum(r["val_clm_loss"] for r in per_pair.values()) / n_pairs,
            "val_cl_loss": sum(r["val_cl_loss"] for r in per_pair.values()) / n_pairs,
            "val_bleu": sum(r["val_bleu"] for r in per_pair.values()) / n_pairs,
            "val_chrf": sum(r["val_chrf"] for r in per_pair.values()) / n_pairs,
            "samples": combined_samples,
        }
    else:
        result = {"samples": combined_samples}
        for lang_pair in per_pair.keys():
            result.update({
                f"{lang_pair}/val_clm_loss": per_pair[lang_pair]["val_clm_loss"],
                f"{lang_pair}/val_cl_loss": per_pair[lang_pair]["val_cl_loss"],
                f"{lang_pair}/val_bleu": per_pair[lang_pair]["val_bleu"],
                f"{lang_pair}/val_chrf": per_pair[lang_pair]["val_chrf"],
                f"{lang_pair}/samples": per_pair[lang_pair]["samples"][:samples_per_pair],
            })

    if print_examples:
        print("\n--- Sample generations ---")
        for i, (lp, ref, hyp) in enumerate(result["samples"]):
            print(f"[{i}] [{lp}] REF: {ref}")
            print(f"[{i}] [{lp}] HYP: {hyp}")

    return result