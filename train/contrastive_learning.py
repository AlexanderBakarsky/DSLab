import contextlib
import re
from types import SimpleNamespace
import torch
import torch.nn as nn




class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp

class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_bottom': average of the bottom layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_bottom2': average of the first two layers.
    'avg_first_last': average of the first and the last layers.
    'last': the hidden state of the last token in the last layers.
    'last_top2': average of the hidden state of the last token in the last two layers.
    'last_first_last': average of the hidden state of the last token in the first and last layers.
    """
    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        self.pooler_re = re.compile(r'^(avg|last|max)-[0-9]+$')
        assert pooler_type in [
            "cls", "cls_before_pooler", 
            "avg", "avg_top2", "avg_first_last", "avg_bottom", "avg_bottom2", "avg_all",
            "max",
            "last", "last_top2", "last_first_last",
            ] or self.pooler_re.match(pooler_type), "unrecognized pooling type %s" % self.pooler_type

    def need_out_hiddens(self, pooler_type):
        assert pooler_type in [
            "cls", "cls_before_pooler", 
            "avg", "avg_top2", "avg_first_last", "avg_bottom", "avg_bottom2", "avg_all",
            "max",
            "last", "last_top2", "last_first_last",
            ] or self.pooler_re.match(pooler_type), "unrecognized pooling type %s" % self.pooler_type

        return pooler_type in [
            'avg_top2', 'avg_first_last', 
            "avg_bottom", "avg_bottom2", "avg_all",
            'last_top2', 'last_first_last'] or self.pooler_re.match(pooler_type)
        

    def average_embed(self, hidden_states:torch.tensor, attention_mask:torch.tensor):
        """
        hidden_states.shape: (bs, sent_len, hidden_size)
        attention_mask: (bs, sent_len) (1: attent, 0: pad)
        """
        return ((hidden_states * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))

    def max_embed(self, hidden_states:torch.tensor, attention_mask:torch.tensor):
        """
        hidden_states.shape: (bs, sent_len, hidden_size)
        attention_mask: (bs, sent_len) (1: attent, 0: pad)
        """
        return torch.max((hidden_states * attention_mask.unsqueeze(-1)), dim=1)[0]

    def last_embed(self, hidden_states:torch.tensor, attention_mask:torch.tensor):
        """
        hidden_states.shape: (bs, sent_len, hidden_size)
        attention_mask: (bs, sent_len) (1: attent, 0: pad)
        """
        sequence_lengths = attention_mask.sum(-1) - 1
        batch_size = hidden_states.shape[0]
        return hidden_states[torch.arange(batch_size, device=hidden_states.device), sequence_lengths]

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state
        # pooler_output = outputs.pooler_output
        hidden_states = outputs.hidden_states

        if self.pooler_type in ['cls_before_pooler', 'cls']:
            return last_hidden[:, 0]
        elif self.pooler_type == "avg":
            return self.average_embed(last_hidden, attention_mask)
        elif "avg-" in self.pooler_type:
            l_idx = int(self.pooler_type.split("-")[1])
            return self.average_embed(hidden_states[l_idx], attention_mask)
        elif "max-" in self.pooler_type:
            l_idx = int(self.pooler_type.split("-")[1])
            return self.max_embed(hidden_states[l_idx], attention_mask)
        elif "last-" in self.pooler_type:
            l_idx = int(self.pooler_type.split("-")[1])
            return self.last_embed(hidden_states[l_idx], attention_mask)
        elif self.pooler_type == "avg_first_last":
            first_hidden = hidden_states[1]
            last_hidden = hidden_states[-1]
            return self.average_embed((first_hidden + last_hidden) / 2.0, attention_mask)
        elif self.pooler_type == "avg_bottom":
            first_hidden = hidden_states[1]
            return self.average_embed(first_hidden, attention_mask)
        elif self.pooler_type == "avg_bottom2":
            first_hidden = hidden_states[1]
            second_hidden = hidden_states[2]
            return self.average_embed((first_hidden + second_hidden) / 2.0, attention_mask)
        elif self.pooler_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            return self.average_embed((last_hidden + second_last_hidden) / 2.0, attention_mask)
        elif self.pooler_type == "avg_all":
            num_layer = len(hidden_states)-1
            hidden_sum = 0
            for l in range(num_layer):
                hidden_sum += hidden_states[l+1]
            return self.average_embed(hidden_sum / float(num_layer), attention_mask)
        elif self.pooler_type == "last":
            return self.last_embed(last_hidden, attention_mask)
        elif "last-" in self.pooler_type:
            l_idx = int(self.pooler_type.split("-")[1])
            return self.last_embed(hidden_states[l_idx], attention_mask)
        elif self.pooler_type == "last_first_last":
            first_layer_last = self.last_embed(hidden_states[1], attention_mask)
            last_layer_last = self.last_embed(hidden_states[-1], attention_mask)
            return (first_layer_last + last_layer_last) / 2.0
        elif self.pooler_type == "last_top2":
            second_last_layer_last = self.last_embed(hidden_states[-2], attention_mask)
            last_layer_last = self.last_embed(hidden_states[-1], attention_mask)
            return (second_last_layer_last + last_layer_last) / 2.0
        else:
            raise NotImplementedError
class ContrastiveLoss(nn.Module):
    _LAYER_RE = re.compile(r'^(?:avg|last|max)-(\d+)$')

    def __init__(self, pooler_type, hard_neg_pooler_type, temp=0.05, center_mode="off", center_momentum=0.99):
        super().__init__()
        assert center_mode in ("off", "joint", "per_side"), f"unknown center_mode: {center_mode}"
        self.pooler = Pooler(pooler_type)
        # If None, hard negatives use the same pooler as the all-vs-all CL.
        self.hard_neg_pooler = Pooler(hard_neg_pooler_type) if hard_neg_pooler_type is not None else self.pooler
        self.sim = Similarity(temp=temp)
        self.loss = nn.CrossEntropyLoss()
        self.center_mode = center_mode
        # EMA momentum for the running mean(s) used by centering.
        # Higher = slower update; 0.99 → effective window ~100 batches.
        self.center_momentum = center_momentum
        # Running means are lazily allocated on the first forward pass so we don't
        # need to know hidden_size at construction time.
        self._ema_initialized = False

    def encode(self, backbone, input_ids, attention_mask, use_hard_neg_pooler = False):
        """Run a single backbone forward pass and return a SimpleNamespace compatible with the pooler."""
        pooler = self.hard_neg_pooler if use_hard_neg_pooler else self.pooler
        pooler_type = pooler.pooler_type
        m = self._LAYER_RE.match(pooler_type)
        if m:
            layer_idx = int(m.group(1))
            assert layer_idx >= 1, f"fast-path layer index must be >= 1 (got {layer_idx}); use output_hidden_states for embedding-layer pooling"
            captured = [None]
            def _hook(_module, _inp, out):
                captured[0] = out[0] if isinstance(out, tuple) else out
            handle = backbone.layers[layer_idx - 1].register_forward_hook(_hook)
            out = backbone(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=False)
            handle.remove()
            h = captured[0]
            return SimpleNamespace(
                last_hidden_state=out.last_hidden_state,
                hidden_states=(None,) * layer_idx + (h,),
            )
        else:
            out = backbone(input_ids=input_ids, attention_mask=attention_mask,
                           output_hidden_states=pooler.need_out_hiddens(pooler_type))
            hs = out.hidden_states
            return SimpleNamespace(
                last_hidden_state=out.last_hidden_state,
                hidden_states=tuple(h for h in hs) if hs else None,
            )

    def backbone_forward(self, backbone, src_input_ids, src_attention_mask, tgt_input_ids, tgt_attention_mask,
                         grad_src: bool = True, grad_tgt: bool = True):
        """Encode src and tgt separately. Returns (src_outputs, tgt_outputs)."""
        with torch.no_grad() if not grad_src else contextlib.nullcontext():
            src_outputs = self.encode(backbone, src_input_ids, src_attention_mask)
        with torch.no_grad() if not grad_tgt else contextlib.nullcontext():
            tgt_outputs = self.encode(backbone, tgt_input_ids, tgt_attention_mask)
        return src_outputs, tgt_outputs

    def forward(self, src_outputs, tgt_outputs, batch, neg_outputs=None, neg_attention_mask=None, hn_mask=None,
                return_sim_stats=False):
        z1 = self.pooler(batch["attention_mask"][:, 0, :], src_outputs)  # (bs, hidden)
        z2 = self.pooler(batch["attention_mask"][:, 1, :], tgt_outputs)  # (bs, hidden)

        # Mean-center the pooled embeddings via an EMA running mean.
        # A per-batch mean would constrain residuals to sum to zero; with small
        # batches that creates an artificial bipolar cosine axis. EMA across
        # batches leaves per-batch residuals unconstrained.
        # Modes:
        #   "joint":    one shared mean for (z1, z2). CL must train the model to
        #               make the language axis disappear in the LM's hidden space.
        #   "per_side": separate means for src and tgt. CL sees content-only
        #               cosines; the language axis is hidden by the metric.
        if self.center_mode != "off":
            with torch.no_grad():
                if self.center_mode == "joint":
                    mu_batch = ((z1.mean(0) + z2.mean(0)) / 2).detach()
                    if not self._ema_initialized:
                        self.register_buffer("mu_ema", mu_batch.clone())
                        self._ema_initialized = True
                    elif self.training:
                        m = self.center_momentum
                        self.mu_ema.mul_(m).add_(mu_batch, alpha=1 - m)
                else:  # "per_side"
                    mu1_batch = z1.mean(0).detach()
                    mu2_batch = z2.mean(0).detach()
                    if not self._ema_initialized:
                        self.register_buffer("mu1_ema", mu1_batch.clone())
                        self.register_buffer("mu2_ema", mu2_batch.clone())
                        self._ema_initialized = True
                    elif self.training:
                        m = self.center_momentum
                        self.mu1_ema.mul_(m).add_(mu1_batch, alpha=1 - m)
                        self.mu2_ema.mul_(m).add_(mu2_batch, alpha=1 - m)
            if self.center_mode == "joint":
                z1 = z1 - self.mu_ema
                z2 = z2 - self.mu_ema
            else:
                z1 = z1 - self.mu1_ema
                z2 = z2 - self.mu2_ema

        # cos_sim[i, j] = similarity between source i and target j
        cos_sim = self.sim(z1.unsqueeze(1), z2.unsqueeze(0))  # (bs, bs)
        labels = torch.arange(cos_sim.size(0)).long().to(z1.device)

        # Stats use raw cosine similarity (undo the temperature scaling in self.sim)
        # so distributions are bounded in [-1, 1] and comparable across temps.
        sim_stats = None
        if return_sim_stats:
            cos_raw = cos_sim.detach() * self.sim.temp
            eye = torch.eye(cos_raw.size(0), dtype=torch.bool, device=cos_raw.device)
            sim_stats = {
                "pos": cos_raw[eye].float().cpu(),
                "neg": cos_raw[~eye].float().cpu(),
                "hn": None,
            }

        if neg_outputs is not None:
            # Per-row hard negative: sim(z1_i, z_neg_i). Add as extra column on the
            # src->tgt direction only; reverse direction keeps the original (bs, bs) matrix
            # since hard negatives don't have a natural anchor there.
            z_neg = self.hard_neg_pooler(neg_attention_mask, neg_outputs)  # (bs, hidden)
            if self.center_mode == "joint":
                z_neg = z_neg - self.mu_ema
            elif self.center_mode == "per_side":
                # Hard negatives are model-predicted targets (same language as z2),
                # so center them with the target-side EMA mean.
                z_neg = z_neg - self.mu2_ema
            hard_sim = self.sim(z1, z_neg).unsqueeze(1)  # (bs, 1)

            if return_sim_stats:
                hn_raw = (hard_sim.detach() * self.sim.temp).squeeze(1)
                if hn_mask is not None:
                    hn_raw = hn_raw[~hn_mask]
                sim_stats["hn"] = hn_raw.float().cpu()

            if hn_mask is not None:
                hard_sim = hard_sim.masked_fill(hn_mask.unsqueeze(1), -1e9)

            cos_sim_fwd = torch.cat([cos_sim, hard_sim], dim=1)  # (bs, bs+1)
            loss = 0.5 * (self.loss(cos_sim_fwd, labels) + self.loss(cos_sim.T, labels))
        else:
            loss = 0.5 * (self.loss(cos_sim, labels) + self.loss(cos_sim.T, labels))

        if return_sim_stats:
            return loss, sim_stats
        return loss

    def _apply_centering(self, Z1, Z2, Zneg=None):
        """Update EMA mean(s) on the full pooled batch and return centered tensors.
        Mirrors the centering logic in `forward` but operates on a single full
        pooled batch (used by GradCache so the EMA update sees the same batch
        statistics as a one-shot forward would)."""
        if self.center_mode == "off":
            return Z1, Z2, Zneg
        with torch.no_grad():
            if self.center_mode == "joint":
                mu_batch = ((Z1.mean(0) + Z2.mean(0)) / 2).detach()
                if not self._ema_initialized:
                    self.register_buffer("mu_ema", mu_batch.clone())
                    self._ema_initialized = True
                elif self.training:
                    m = self.center_momentum
                    self.mu_ema.mul_(m).add_(mu_batch, alpha=1 - m)
            else:
                mu1_batch = Z1.mean(0).detach()
                mu2_batch = Z2.mean(0).detach()
                if not self._ema_initialized:
                    self.register_buffer("mu1_ema", mu1_batch.clone())
                    self.register_buffer("mu2_ema", mu2_batch.clone())
                    self._ema_initialized = True
                elif self.training:
                    m = self.center_momentum
                    self.mu1_ema.mul_(m).add_(mu1_batch, alpha=1 - m)
                    self.mu2_ema.mul_(m).add_(mu2_batch, alpha=1 - m)
        if self.center_mode == "joint":
            Z1c = Z1 - self.mu_ema
            Z2c = Z2 - self.mu_ema
            Znegc = (Zneg - self.mu_ema) if Zneg is not None else None
        else:
            Z1c = Z1 - self.mu1_ema
            Z2c = Z2 - self.mu2_ema
            Znegc = (Zneg - self.mu2_ema) if Zneg is not None else None
        return Z1c, Z2c, Znegc

    def _loss_from_pooled(self, Z1, Z2, Zneg=None, hn_mask=None, return_sim_stats=False):
        """Compute the CL loss from pre-pooled (and pre-centered) embeddings.
        Used by GradCache after stitching together micro-batched pooled vectors."""
        cos_sim = self.sim(Z1.unsqueeze(1), Z2.unsqueeze(0))
        labels = torch.arange(cos_sim.size(0)).long().to(Z1.device)

        sim_stats = None
        if return_sim_stats:
            cos_raw = cos_sim.detach() * self.sim.temp
            eye = torch.eye(cos_raw.size(0), dtype=torch.bool, device=cos_raw.device)
            sim_stats = {
                "pos": cos_raw[eye].float().cpu(),
                "neg": cos_raw[~eye].float().cpu(),
                "hn": None,
            }

        if Zneg is not None:
            hard_sim = self.sim(Z1, Zneg).unsqueeze(1)
            if return_sim_stats:
                hn_raw = (hard_sim.detach() * self.sim.temp).squeeze(1)
                if hn_mask is not None:
                    hn_raw = hn_raw[~hn_mask]
                sim_stats["hn"] = hn_raw.float().cpu()
            if hn_mask is not None:
                hard_sim = hard_sim.masked_fill(hn_mask.unsqueeze(1), -1e9)
            cos_sim_fwd = torch.cat([cos_sim, hard_sim], dim=1)
            loss = 0.5 * (self.loss(cos_sim_fwd, labels) + self.loss(cos_sim.T, labels))
        else:
            loss = 0.5 * (self.loss(cos_sim, labels) + self.loss(cos_sim.T, labels))

        if return_sim_stats:
            return loss, sim_stats
        return loss

    def gradcache_step(self, backbone, batch, micro_bs,
                        neg_input_ids=None, neg_attention_mask=None, hn_mask=None,
                        grad_src=True, grad_tgt=True,
                        autocast_dtype=None, loss_scale=1.0,
                        return_sim_stats=False):
        """GradCache-style CL: forward in micro-batches twice to give gradients
        identical to a single full-batch forward, while only holding one
        micro-batch's encoder activations at a time.

        Pass 1: no-grad encode every micro-batch, collect pooled `z`s.
        Mid:    compute loss on the stitched pooled batch (fp32, tiny memory)
                and get dL/dz for each example.
        Pass 2: re-encode each micro-batch with grad and call
                z_chunk.backward(g_chunk). Param grads accumulate across chunks.

        The `loss_scale` is applied to dL/dz (so callers can pre-scale by
        cl_weight / grad_accum_steps the same way they would for a one-shot
        backward).

        Returns (loss_value_detached, sim_stats_or_None). Param grads are left
        accumulated on the model — no .backward() needed by the caller.
        """
        src_ids = batch["input_ids"][:, 0, :]
        src_mask = batch["attention_mask"][:, 0, :]
        tgt_ids = batch["input_ids"][:, 1, :]
        tgt_mask = batch["attention_mask"][:, 1, :]
        bs = src_ids.size(0)
        has_neg = neg_input_ids is not None

        chunks = [(s, min(s + micro_bs, bs)) for s in range(0, bs, micro_bs)]

        def ac():
            return torch.autocast("cuda", dtype=autocast_dtype) if autocast_dtype is not None else contextlib.nullcontext()

        # --- Pass 1: no-grad encode and pool each micro-batch.
        z1_list, z2_list, zneg_list = [], [], []
        for s, e in chunks:
            with torch.no_grad(), ac():
                so = self.encode(backbone, src_ids[s:e], src_mask[s:e])
                z1_list.append(self.pooler(src_mask[s:e], so).float())
                del so
                to = self.encode(backbone, tgt_ids[s:e], tgt_mask[s:e])
                z2_list.append(self.pooler(tgt_mask[s:e], to).float())
                del to
                if has_neg:
                    no = self.encode(backbone, neg_input_ids[s:e], neg_attention_mask[s:e],
                                     use_hard_neg_pooler=True)
                    zneg_list.append(self.hard_neg_pooler(neg_attention_mask[s:e], no).float())
                    del no

        Z1 = torch.cat(z1_list, dim=0).detach().requires_grad_(True)
        Z2 = torch.cat(z2_list, dim=0).detach().requires_grad_(True)
        Zneg = torch.cat(zneg_list, dim=0).detach().requires_grad_(True) if has_neg else None

        # --- Loss on cached pooled vectors. Centering + EMA happen here so the
        # update sees the full-batch statistics (matches one-shot forward).
        Z1c, Z2c, Znegc = self._apply_centering(Z1, Z2, Zneg)
        out = self._loss_from_pooled(Z1c, Z2c, Znegc, hn_mask=hn_mask,
                                     return_sim_stats=return_sim_stats)
        if return_sim_stats:
            loss, sim_stats = out
        else:
            loss, sim_stats = out, None

        scaled = loss * loss_scale
        grad_inputs = [Z1, Z2] + ([Zneg] if has_neg else [])
        grads = torch.autograd.grad(scaled, grad_inputs)
        g1, g2 = grads[0], grads[1]
        gneg = grads[2] if has_neg else None
        loss_val = loss.detach()

        # --- Pass 2: re-encode each micro-batch with grad and inject dL/dz as
        # the upstream gradient. Param grads accumulate across micro-batches.
        for s, e in chunks:
            with ac():
                if grad_src:
                    so = self.encode(backbone, src_ids[s:e], src_mask[s:e])
                    z1_chunk = self.pooler(src_mask[s:e], so)
                    z1_chunk.float().backward(g1[s:e])
                    del so, z1_chunk
                if grad_tgt:
                    to = self.encode(backbone, tgt_ids[s:e], tgt_mask[s:e])
                    z2_chunk = self.pooler(tgt_mask[s:e], to)
                    z2_chunk.float().backward(g2[s:e])
                    del to, z2_chunk
                if has_neg:
                    no = self.encode(backbone, neg_input_ids[s:e], neg_attention_mask[s:e],
                                     use_hard_neg_pooler=True)
                    zneg_chunk = self.hard_neg_pooler(neg_attention_mask[s:e], no)
                    zneg_chunk.float().backward(gneg[s:e])
                    del no, zneg_chunk

        return loss_val, sim_stats