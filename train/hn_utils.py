import torch
from sacrebleu.metrics import BLEU


@torch.no_grad()
def _build_clm_hard_negatives(logits, clm_target_ids, pad_id):
    """
    From a teacher-forced CLM forward, extract the model's argmax predictions at
    label positions (i.e. predicted target tokens) and re-pack them as a padded
    batch of sequences suitable for a CL-style backbone forward.

    Args:
        logits: (bs, seq_len, vocab) — CLM logits.
        clm_target_ids: (bs, seq_len) — labels with -100 on prompt/pad positions.
        pad_id: int — padding token id for the resulting sequences.

    Returns:
        neg_input_ids: (bs, max_tgt_len) long tensor on the same device as logits.
        neg_attention_mask: (bs, max_tgt_len) long tensor.
    """
    predicted = logits.argmax(dim=-1)  # (bs, seq_len)
    # logits at position i predict token at position i+1; shift right so
    # shifted[:, i] is the prediction for token at position i.
    shifted = torch.cat(
        [torch.full_like(predicted[:, :1], pad_id), predicted[:, :-1]], dim=1
    )
    target_mask = clm_target_ids != -100  # (bs, seq_len)

    bs = predicted.size(0)
    per_example = []
    max_len = 0
    for b in range(bs):
        idx = target_mask[b].nonzero(as_tuple=True)[0]
        toks = shifted[b, idx]
        per_example.append(toks)
        if toks.size(0) > max_len:
            max_len = toks.size(0)

    device = predicted.device
    if max_len == 0:
        # Degenerate batch (shouldn't happen due to the all-masked guard upstream).
        max_len = 1
    neg_input_ids = torch.full(
        (bs, max_len), pad_id, device=device, dtype=predicted.dtype
    )
    neg_attention_mask = torch.zeros((bs, max_len), device=device, dtype=torch.long)
    for b, toks in enumerate(per_example):
        L = toks.size(0)
        if L > 0:
            neg_input_ids[b, :L] = toks
            neg_attention_mask[b, :L] = 1
    return neg_input_ids, neg_attention_mask


@torch.no_grad()
def _cos_filter_mask(cl_criterion, backbone, neg_outputs, neg_attention_mask,
                     tgt_input_ids, tgt_attention_mask, tgt_outputs_main, threshold):
    """
    Returns a bool tensor (bs,) where True means the hard negative's cosine
    similarity to the target (under hard_neg_pooler) meets or exceeds
    `threshold` and should be excluded from the CL loss.
    """
    z_neg = cl_criterion.hard_neg_pooler(neg_attention_mask, neg_outputs)
    if cl_criterion.hard_neg_pooler is cl_criterion.pooler:
        z_tgt = cl_criterion.pooler(tgt_attention_mask, tgt_outputs_main)
    else:
        tgt_outputs_hn = cl_criterion.encode(
            backbone, tgt_input_ids, tgt_attention_mask, use_hard_neg_pooler=True,
        )
        z_tgt = cl_criterion.hard_neg_pooler(tgt_attention_mask, tgt_outputs_hn)
    cos = torch.nn.functional.cosine_similarity(z_neg, z_tgt, dim=-1)
    return cos >= threshold


@torch.no_grad()
def _bleu_filter_mask(neg_input_ids, tgt_input_ids, tokenizer, threshold):
    """
    Returns a bool tensor (bs,) where True means the hard negative's BLEU score
    against the target meets or exceeds `threshold` and should be excluded from
    the CL loss.
    """
    neg_texts = tokenizer.batch_decode(neg_input_ids, skip_special_tokens=True)
    tgt_texts = tokenizer.batch_decode(tgt_input_ids, skip_special_tokens=True)

    bleu = BLEU(effective_order=True)
    mask = [bleu.sentence_score(hyp, [ref]).score >= threshold
            for hyp, ref in zip(neg_texts, tgt_texts)]
    return torch.tensor(mask, dtype=torch.bool, device=neg_input_ids.device)

