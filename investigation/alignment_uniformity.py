import torch
import torch.nn.functional as F

from train.contrastive_learning import ContrastiveLoss
from utils import get_decoder_backbone

from tqdm import tqdm


def _compute_alignment(src_emb, tgt_emb):
    assert tgt_emb.dim() == 2
    tgt_emb = F.normalize(tgt_emb, dim=-1)
    src_emb = F.normalize(src_emb, dim=-1)
    sim = torch.norm(src_emb - tgt_emb, p=2, dim=-1) ** 2
    return sim.mean().item()


def _compute_uniformity(emb):
    assert emb.dim() == 2
    emb = F.normalize(emb, dim=-1)
    sim = -2 * torch.norm(emb[None, :, :] - emb[:, None, :], dim=-1, p=2) ** 2
    sim = torch.exp(sim)
    return torch.log(sim.mean()).item()


def log_alignment_uniformity(model, tokenizer, investigation_loader, pooler_type, max_context_len, wandb_run=None):
    """Compute SimCSE alignment and uniformity for each (src_lang, en) pair.
    investigation_loader is a dict of {pair_key: DataLoader} where each batch
    has "src" and "target" string fields (same format as get_final_evaluation_loader).
    Embeddings are extracted via ContrastiveLoss.encode + Pooler, matching the
    training pooling path exactly.
    """
    device = next(model.parameters()).device
    cl = ContrastiveLoss(pooler_type, hard_neg_pooler_type=None, temp=-1)
    backbone = get_decoder_backbone(model)
    model.eval()

    with torch.no_grad():
        for pair_key, loader in investigation_loader.items():
            src_lang, tgt_lang = pair_key.split("-", 1)
            src_embs, tgt_embs = [], []

            for batch in tqdm(loader, desc= f"Investigating {src_lang}, {tgt_lang}"):
                src_enc = tokenizer(
                    batch["src"], return_tensors="pt", padding=True,
                    truncation=True, max_length=max_context_len, add_special_tokens=False,
                ).to(device)
                tgt_enc = tokenizer(
                    batch["target"], return_tensors="pt", padding=True,
                    truncation=True, max_length=max_context_len, add_special_tokens=False,
                ).to(device)

                src_out = cl.encode(backbone, src_enc["input_ids"], src_enc["attention_mask"])
                src_pooled = cl.pooler(src_enc["attention_mask"], src_out).float().cpu()

                tgt_out = cl.encode(backbone, tgt_enc["input_ids"], tgt_enc["attention_mask"])
                tgt_pooled = cl.pooler(tgt_enc["attention_mask"], tgt_out).float().cpu()

                src_embs.append(src_pooled)
                tgt_embs.append(tgt_pooled)

            src_embs = torch.cat(src_embs, dim=0)
            tgt_embs = torch.cat(tgt_embs, dim=0)

            alignment = _compute_alignment(src_embs, tgt_embs)
            uniformity = _compute_uniformity(torch.cat([src_embs, tgt_embs], dim=0))

            print(f"{src_lang}-{tgt_lang}: alignment={alignment:.4f}, uniformity={uniformity:.4f}")

            if wandb_run is not None:
                wandb_run.log({
                    f"{src_lang}-{tgt_lang}/alignment": alignment,
                    f"{src_lang}-{tgt_lang}/uniformity": uniformity,
                })