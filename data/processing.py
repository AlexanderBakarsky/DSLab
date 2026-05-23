import os
import random
from typing import Optional

import numpy as np
import datasets
from torch.utils.data import DataLoader
from .attributes import _LANG_NAME, INSTRUCT_TEMPLATES, BARE_TEMPLATES
from .loaders import DATASET_LOADERS


def _sample_example_ids(i: int, n: int, num_examples: int) -> np.ndarray:
    """Sample num_examples indices from [0, n) excluding i."""
    ids = np.random.choice(n - 1, num_examples, replace=False)
    ids[ids >= i] += 1
    return ids


def prepare_prompt(source_lang, target_lang, source_sentence):
    return f"{_LANG_NAME[source_lang]}:{source_sentence} = {_LANG_NAME[target_lang]}:"


def format_dataset(ds: datasets.Dataset, source_lang: str, target_lang: str,
                   tokenizer, num_examples: int = 0, instruct: bool = False,
                   icl_examples: dict = None, training: bool = False,
                   mix_prompt_formats: bool = True) -> datasets.Dataset:
    """
    Format a raw {"source", "target"} dataset into {"prompt", "target", "src"}.
    Works for both training (num_examples=0) and evaluation (num_examples>=0).

    instruct=False, num_examples=0 : base prompt, no ICL
    instruct=False, num_examples>0 : base prompt with prepended ICL examples
    instruct=True,  num_examples=0 : chat-template prompt, no ICL
    instruct=True,  num_examples>0 : chat-template prompt with few-shot turns

    BOS is embedded in the prompt string and EOS is appended to target so that
    make_tokenize and evaluate never need to add special tokens themselves.

    If icl_examples is provided (dict with "source" and "target" lists), those
    are used as fixed ICL examples for every row instead of sampling from ds.

    If training=True, per-example sampling from SYSTEM_MSGS/USER_WRAPPERS (instruct)
    or BARE_TEMPLATES (non-instruct) diversifies the prompt format. If training=False
    (eval), index 0 of each pool is always used so results are deterministic and
    match a single canonical format.
    """
    n = len(ds)
    srcs, tgts = ds["source"], ds["target"]
    eos = tokenizer.eos_token
    sl_name, tl_name = _LANG_NAME[source_lang], _LANG_NAME[target_lang]

    if icl_examples is not None:
        icl_srcs, icl_tgts = icl_examples["source"], icl_examples["target"]
        get_icl = lambda _: zip(icl_srcs, icl_tgts)
    else:
        get_icl = lambda i: ((srcs[j], tgts[j]) for j in _sample_example_ids(i, n, num_examples))

    rows = []
    sample_format = training and mix_prompt_formats
    if instruct:
        for i in range(n):
            tmpl = random.choice(INSTRUCT_TEMPLATES) if sample_format else INSTRUCT_TEMPLATES[0]

            sys_msg, _ = tmpl("", sl_name, tl_name)
            messages = []
            if sys_msg:
                messages.append({"role": "system", "content": sys_msg})
            for src_ex, tgt_ex in get_icl(i):
                _, user_ex = tmpl(src_ex, sl_name, tl_name)
                messages += [
                    {"role": "user",      "content": user_ex},
                    {"role": "assistant", "content": tgt_ex},
                ]
            _, user_msg = tmpl(srcs[i], sl_name, tl_name)
            messages.append({"role": "user", "content": user_msg})
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            rows.append({"prompt": prompt, "target": tgts[i], "src": srcs[i]})
    else:
        bos = tokenizer.bos_token
        for i in range(n):
            tmpl = random.choice(BARE_TEMPLATES) if sample_format else BARE_TEMPLATES[0]
            prompt = bos
            for src_ex, tgt_ex in get_icl(i):
                prompt += tmpl(sl_name, tl_name, src_ex) + tgt_ex.rstrip("\n") + eos + "\n"
            prompt += tmpl(sl_name, tl_name, srcs[i])
            rows.append({"prompt": prompt, "target": tgts[i], "src": srcs[i]})

    result = datasets.Dataset.from_list(rows)
    print(f"Formatted dataset with {len(result)} examples. Sample rows:")
    for _ in range(3):
        print(random.choice(rows))
    return result


def make_tokenize(tokenizer, max_len):
    """Tokenize a dataset formatted by format_dataset. BOS is already in the prompt
    string; EOS is appended only to the CLM target so the model learns to stop.
    CL inputs are tokenized without any special tokens."""
    def tokenize(pair):
        # CLM: prompt already contains BOS, append EOS to the target ids.
        prompt_ids = tokenizer(pair["prompt"], add_special_tokens=False)["input_ids"]
        target_ids = tokenizer(pair["target"], add_special_tokens=False)["input_ids"]
        target_ids = target_ids + [tokenizer.eos_token_id]
        clm_input_ids = prompt_ids + target_ids
        clm_prompt_len = len(prompt_ids)

        # CL: no special tokens — the BOS/EOS hidden states would otherwise dominate
        # the pooled sentence embedding and compress all cosine similarities.
        src_enc = tokenizer(pair["src"], truncation=True, max_length=max_len, add_special_tokens=False)
        tgt_enc = tokenizer(pair["target"], truncation=True, max_length=max_len, add_special_tokens=False)

        return {
            "input_ids":      [src_enc["input_ids"],      tgt_enc["input_ids"]],
            "attention_mask": [src_enc["attention_mask"],  tgt_enc["attention_mask"]],
            "clm_input_ids":  clm_input_ids,
            "clm_prompt_len": clm_prompt_len,
        }
    return tokenize





def load_raw_dataset(name: str, split: str, source_lang: str, target_lang: str, max_size: int, cache_dir: str) -> datasets.Dataset:
    """Load any supported dataset as a HF Dataset with {"source": str, "target": str} columns.
    Language codes must be keys from lang_dict._LANG_NAME."""
    if name not in DATASET_LOADERS:
        raise ValueError(f"Unknown dataset '{name}'. Choose from: {list(DATASET_LOADERS)}")
    return DATASET_LOADERS[name](split, source_lang, target_lang, max_size, cache_dir)


def filter_parallel_data(
    ds: datasets.Dataset,
    source_lang: str,
    target_lang: str,
    min_chars: int = 2,
    max_chars: int = 2000,
    min_len_ratio: float = 0.5,
    max_len_ratio: float = 2.0,
    drop_identical: bool = True,
    dedup: bool = True,
    lang_id: bool = False,
    lang_id_model_path: Optional[str] = None,
    lang_id_min_confidence: float = 0.5,
    verbose: bool = True,
) -> datasets.Dataset:
    """
    Clean a parallel {'source','target'} HF Dataset.

    Applies (in order):
      1. Empty / whitespace-only removal on either side.
      2. Character-length bounds [min_chars, max_chars] on both sides.
      3. Character-length ratio filter: keep if len(tgt)/len(src) in [min_len_ratio, max_len_ratio].
      4. (Optional) drop rows where source and target strings are byte-identical.
      5. (Optional) exact deduplication on (source, target).
      6. (Optional) fastText lid.176 language-ID filter: keep only rows where both sides
         are predicted as the expected language with confidence >= lang_id_min_confidence.
         Requires the `fasttext` package and a path to lid.176.bin via lang_id_model_path
         (env var FASTTEXT_LID_PATH is also checked as a fallback).

    Returns the filtered dataset. Prints per-stage drop counts if verbose=True.
    """
    def _log(stage, before, after):
        if verbose:
            print(f"[filter_parallel_data] {stage}: {before} -> {after} ({before - after} dropped)")

    n0 = len(ds)

    def _basic_filter(ex):
        s, t = ex["source"], ex["target"]
        if s is None or t is None:
            return False
        s = s.strip()
        t = t.strip()
        if len(s) < min_chars or len(t) < min_chars:
            return False
        if len(s) > max_chars or len(t) > max_chars:
            return False
        ratio = len(t) / len(s)
        if ratio < min_len_ratio or ratio > max_len_ratio:
            return False
        if drop_identical and s == t:
            return False
        return True

    ds = ds.filter(_basic_filter)
    _log("length/empty/identical", n0, len(ds))
    n1 = len(ds)

    if dedup:
        df = ds.to_pandas().drop_duplicates(subset=["source", "target"])
        ds = datasets.Dataset.from_pandas(df, preserve_index=False)
        _log("dedup", n1, len(ds))
        n1 = len(ds)

    if lang_id:
        try:
            import fasttext
        except ImportError:
            print("[filter_parallel_data] WARNING: fasttext not installed; skipping language-ID filter.")
            return ds

        model_path = lang_id_model_path or os.environ.get("FASTTEXT_LID_PATH")
        if not model_path or not os.path.exists(model_path):
            print(f"[filter_parallel_data] WARNING: fastText lid.176.bin not found "
                  f"(lang_id_model_path={model_path}). Skipping language-ID filter. "
                  f"Download from https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin")
            return ds

        lid = fasttext.load_model(model_path)
        src_label = f"__label__{source_lang}"
        tgt_label = f"__label__{target_lang}"

        def _lang_ok(text, expected_label):
            # fasttext expects single-line input
            text = text.replace("\n", " ").strip()
            if not text:
                return False
            labels, probs = lid.predict(text, k=1)
            return labels[0] == expected_label and probs[0] >= lang_id_min_confidence

        def _lang_filter(ex):
            return _lang_ok(ex["source"], src_label) and _lang_ok(ex["target"], tgt_label)

        ds = ds.filter(_lang_filter)
        _log(f"lang_id ({source_lang}/{target_lang} ≥ {lang_id_min_confidence})", n1, len(ds))

    if verbose:
        print(f"[filter_parallel_data] final: {len(ds)} rows (started with {n0}, kept {100*len(ds)/max(n0,1):.1f}%)")
    return ds

