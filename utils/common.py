
import torch, transformers, os
import numpy as np
import random 

def fix_seeds(seed):
    """Seed common RNG sources so training and analysis remain reproducible."""
    transformers.set_seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def build_lang_pairs(src, tgt):
    src_is_list = isinstance(src, list)
    tgt_is_list = isinstance(tgt, list)

    if src_is_list and tgt_is_list and len(src) != len(tgt):
        raise ValueError(
            f"source_lang and target_lang lists must have the same length, "
            f"got {len(src)} and {len(tgt)}"
        )
    
    if src_is_list and tgt_is_list:
        pairs = list(zip(src, tgt))
    elif src_is_list:
        pairs = [(s, tgt) for s in src]
    elif tgt_is_list:
        pairs = [(src, t) for t in tgt]
    else:
        pairs = [(src, tgt)]
    return pairs