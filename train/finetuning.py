from train.contrastive_learning import ContrastiveLoss
from transformers import get_cosine_schedule_with_warmup
import json
import os
import shutil
import torch
import bitsandbytes as bnb
from tqdm import tqdm
from evaluate import evaluate
from utils import get_decoder_backbone
import wandb

from train.hn_utils import _bleu_filter_mask, _build_clm_hard_negatives, _cos_filter_mask


class Trainer:

    def __init__(self, model, tokenizer, data_loader, val_loader, training_args, model_args, run,):

        #init atributes
        self.training_args = training_args
        self.model_args = model_args
        self.run = run

        #init model and tokenizer
        self.model = model
        self.tokenizer = tokenizer

        # dataloaders
        self.data_loader = data_loader
        self.val_loader= val_loader
         

        # CL
        self.cl_criterion = ContrastiveLoss(
            self.training_args.pooler_type,
            self.training_args.hn_pooler_type,
            temp= self.training_args.cl_temperature,
            center_mode= self.training_args.cl_center_mode,
            center_momentum= self.training_args.cl_center_momentum,
        ).to(model.device)

        self.use_cl = training_args.use_cl
        self.cl_hard_negatives = training_args.cl_hard_negatives

        self.use_cl_hard_neg = self.use_cl and self.cl_hard_negatives
        if self.use_cl_hard_neg:
            assert not training_args.pack_clm, "cl_hard_negatives requires pack_clm=false"
            self.pad_id = tokenizer.pad_token_id

        # Micro-batch sizes for in-batch chunking. The loader batch is the
        # effective batch; CLM is chunked by clm_micro_bs (gradients scaled by
        # 1/n_clm_chunks and accumulated), CL runs once per loader batch via
        # gradcache_step with micro_bs=cl_micro_bs.
        B = training_args.batch_size
        assert B % training_args.clm_micro_bs == 0, \
            f"batch_size ({B}) must be a multiple of clm_micro_bs ({training_args.clm_micro_bs})"
        assert B % training_args.cl_micro_bs == 0, \
            f"batch_size ({B}) must be a multiple of cl_micro_bs ({training_args.cl_micro_bs})"
        self.clm_micro_bs = training_args.clm_micro_bs
        self.cl_micro_bs = training_args.cl_micro_bs

        self.sim_buf = {"pos": [], "neg": [], "hn": []}
        self.sample_rows = []  # accumulates (epoch, step, idx, ref, hyp) across evals

        # loss per epoch
        self.clm_loss_per_epoch = 0.0
        self.cl_loss_per_epoch = 0.0

        # loss since eval
        self.clm_loss_since_eval = 0.0
        self.cl_loss_since_eval = 0.0
        self.steps_since_eval = 0
        
        #optimizer
        self.optimizer = bnb.optim.AdamW8bit(
            [param for _, param in model.named_parameters() if param.requires_grad],
            lr=training_args.lr,
            weight_decay=training_args.weight_decay
        )

        total_steps = training_args.epochs * len(data_loader)
        warmup_steps = int(training_args.warmup_fraction * total_steps)

        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        
        )

        # checkpointing
        self.best_score = None
        self.best_ckpt_dir = os.path.join(model_args.training_output_dir, "best_checkpoint")
        
        #wandb session
        # Make `samples_seen` the global x-axis for every metric/chart in the run.
        self.run.define_metric("samples_seen")
        self.run.define_metric("*", step_metric="samples_seen")

    def train(self):

        self.run_eval(epoch=0, step=0, global_step=0)

        for e in range(self.training_args.epochs):

            
            for i, batch in tqdm(enumerate(self.data_loader), total=len(self.data_loader), desc = "Training."):
                batch = {k: v.to(self.model.device) for k, v in batch.items()}

                # CLM: forward+backward in clm_micro_bs chunks; collect
                # CLM-derived hard negatives (concatenated across chunks).
                clm_step, neg_input_ids, neg_attention_mask = self.compute_clm_loss(batch)

                # CL: one gradcache_step over the loader batch with
                # micro_bs=cl_micro_bs. Param grads accumulate onto the model.
                cl_step, sim_stats = self.compute_cl_loss(batch, neg_input_ids, neg_attention_mask)

                if sim_stats:
                    self.sim_buf["pos"].append(sim_stats["pos"])
                    self.sim_buf["neg"].append(sim_stats["neg"])
                    if sim_stats["hn"] is not None:
                        self.sim_buf["hn"].append(sim_stats["hn"])

                self.clm_loss_per_epoch += clm_step
                self.cl_loss_per_epoch += cl_step
                self.clm_loss_since_eval += clm_step
                self.cl_loss_since_eval += cl_step
                self.steps_since_eval += 1

                self.opt_step()

                if (i + 1) % self.training_args.eval_steps == 0:
                    self.eval_save_step(i, e)

            #end of epoch logging
            epoch_log = {
                "samples_seen": (e + 1) * len(self.data_loader) * self.training_args.batch_size,
                "epoch": e,
                "epoch_clm_loss": self.clm_loss_per_epoch / len(self.data_loader),
            }
            if self.use_cl:
                epoch_log["epoch_cl_loss"] = self.cl_loss_per_epoch / len(self.data_loader)
            self.run.log(epoch_log)

        self.promote_best_model()
        

    def compute_clm_loss(self, batch):
        """CLM forward+backward in clm_micro_bs chunks. Each chunk's loss is
        scaled by 1/n_chunks so the accumulated gradient matches a single
        full-batch backward. Also extracts CLM-derived hard negatives per chunk
        and stitches them into full-batch tensors for downstream CL use."""
        B = batch["clm_input_ids"].size(0)
        chunks = [(s, min(s + self.clm_micro_bs, B)) for s in range(0, B, self.clm_micro_bs)]
        n = len(chunks)

        clm_loss_token_sum = 0.0
        clm_token_count = 0
        neg_id_parts, neg_mask_parts = [], []
        for s, e in chunks:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                clm_out = self.model(
                    input_ids=batch["clm_input_ids"][s:e],
                    attention_mask=batch["clm_attention_mask"][s:e],
                    labels=batch["clm_target_ids"][s:e],
                    use_cache=False,
                )
                clm_loss_raw = clm_out.loss
                clm_loss = clm_loss_raw * self.training_args.clm_weight

                if self.use_cl_hard_neg:
                    nid, nmask = _build_clm_hard_negatives(
                        clm_out.logits, batch["clm_target_ids"][s:e], self.pad_id
                    )
                    neg_id_parts.append(nid)
                    neg_mask_parts.append(nmask)
                del clm_out

            (clm_loss / n).backward()
            n_tok = (batch["clm_target_ids"][s:e, 1:] != -100).sum().item()
            clm_loss_token_sum += clm_loss_raw.item() * n_tok
            clm_token_count += n_tok

        avg_clm = clm_loss_token_sum / max(clm_token_count, 1)

        if self.use_cl_hard_neg:
            # Each chunk pads to its own per-chunk max target length, so
            # restitch by padding to the global max before concatenating.
            def _pad_cat(tensors, pad_value):
                max_len = max(t.size(1) for t in tensors)
                out = []
                for t in tensors:
                    if t.size(1) < max_len:
                        pad = torch.full((t.size(0), max_len - t.size(1)),
                                         pad_value, dtype=t.dtype, device=t.device)
                        t = torch.cat([t, pad], dim=1)
                    out.append(t)
                return torch.cat(out, dim=0)
            neg_input_ids = _pad_cat(neg_id_parts, pad_value=self.pad_id)
            neg_attention_mask = _pad_cat(neg_mask_parts, pad_value=0)
        else:
            neg_input_ids, neg_attention_mask = None, None

        return avg_clm, neg_input_ids, neg_attention_mask

    def compute_cl_loss(self, batch, neg_input_ids, neg_attention_mask):
        """One GradCache CL forward+backward over the loader batch, chunked
        internally by cl_micro_bs."""
        if not self.use_cl:
            return 0.0, None

        backbone = get_decoder_backbone(self.model)

        hn_mask = None
        if self.use_cl_hard_neg:
            if self.training_args.hn_bleu_threshold is not None:
                hn_mask = _bleu_filter_mask(
                    neg_input_ids, batch["input_ids"][:, 1, :],
                    self.tokenizer, self.training_args.hn_bleu_threshold,
                )
            if self.training_args.hn_cos_threshold is not None:
                # Chunked no-grad cos mask, matching GradCache memory.
                cos_mask_parts = []
                B = batch["input_ids"].size(0)
                for s in range(0, B, self.cl_micro_bs):
                    e = min(s + self.cl_micro_bs, B)
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                        neg_o = self.cl_criterion.encode(
                            backbone, neg_input_ids[s:e], neg_attention_mask[s:e], True,
                        )
                        tgt_o_main = self.cl_criterion.encode(
                            backbone, batch["input_ids"][s:e, 1, :],
                            batch["attention_mask"][s:e, 1, :],
                        )
                        cos_mask_parts.append(_cos_filter_mask(
                            self.cl_criterion, backbone,
                            neg_o, neg_attention_mask[s:e],
                            batch["input_ids"][s:e, 1, :],
                            batch["attention_mask"][s:e, 1, :],
                            tgt_o_main, self.training_args.hn_cos_threshold,
                        ))
                cos_mask = torch.cat(cos_mask_parts, dim=0)
                hn_mask = cos_mask if hn_mask is None else (hn_mask | cos_mask)

        cl_loss_raw, sim_stats = self.cl_criterion.gradcache_step(
            backbone, batch, self.cl_micro_bs,
            neg_input_ids=neg_input_ids, neg_attention_mask=neg_attention_mask,
            hn_mask=hn_mask,
            grad_src=self.training_args.cl_grad_src,
            grad_tgt=self.training_args.cl_grad_tgt,
            autocast_dtype=torch.bfloat16,
            loss_scale=self.training_args.cl_weight,
            return_sim_stats=True,
        )
        return cl_loss_raw.item(), sim_stats

    def opt_step(self):
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad],
            self.training_args.max_grad_norm
        )
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)

    def eval_save_step(self, i, e):
        train_clm_loss = self.clm_loss_since_eval / self.steps_since_eval
        train_cl_loss = self.cl_loss_since_eval / self.steps_since_eval if self.use_cl else None

        self.clm_loss_since_eval = 0.0
        self.cl_loss_since_eval = 0.0
        self.steps_since_eval = 0
        results = self.run_eval(
            train_clm_loss=train_clm_loss,
            train_cl_loss=train_cl_loss,
            epoch=e,
            step=i,
            global_step=(e * len(self.data_loader) + i + 1) * self.training_args.batch_size,
        )
        if self.training_args.save_best_checkpoint:
            self.maybe_save_best(self.model, results, f"epoch{e}_step{i}")
    
    def run_eval(self, train_clm_loss=None, train_cl_loss=None, epoch=None, step=None, global_step=0):
        torch.cuda.empty_cache()
        results = evaluate(
            self.model,
            self.val_loader,
            self.tokenizer,
            self.model_args.max_context_len,
            self.model_args.max_generation_context_len,
            use_BLEU=self.training_args.use_BLEU,
            use_chrF=self.training_args.use_chrF,
            use_cl= self.training_args.use_cl,
            cl_criterion=self.cl_criterion,
            aggregate=True #Get performance across all langauges. Irrelevant for a single language pair.
        )
        log_dict = {"samples_seen": global_step, "val_clm_loss": results["val_clm_loss"]}
        if train_clm_loss is not None:
            log_dict["train_clm_loss"] = train_clm_loss
        if self.use_cl:
            log_dict["val_cl_loss"] = results["val_cl_loss"]
            if train_cl_loss is not None:
                log_dict["train_cl_loss"] = train_cl_loss
        if self.training_args.use_BLEU:
            log_dict["val_bleu"] = results["val_bleu"]
        if self.training_args.use_chrF:
            log_dict["val_chrf"] = results["val_chrf"]
        if self.training_args.use_BLEU or self.training_args.use_chrF:
            for idx, (lp, ref, hyp) in enumerate(results.get("samples", [])):
                self.sample_rows.append([epoch if epoch is not None else 0,
                                    step if step is not None else 0,
                                    idx, lp, ref, hyp])
            if self.sample_rows:
                log_dict["val_samples"] = wandb.Table(
                    columns=["epoch", "step", "idx", "lang_pair", "ref", "hyp"],
                    data=self.sample_rows,
                )
        if self.use_cl and self.sim_buf["pos"]:
            pos = torch.cat(self.sim_buf["pos"]).numpy()
            neg = torch.cat(self.sim_buf["neg"]).numpy()
            log_dict["sim/pos"] = wandb.Histogram(pos, num_bins=64)
            log_dict["sim/neg"] = wandb.Histogram(neg, num_bins=64)
            if self.sim_buf["hn"]:
                hn = torch.cat(self.sim_buf["hn"]).numpy()
                log_dict["sim/hn"] = wandb.Histogram(hn, num_bins=64)
            self.sim_buf["pos"].clear()
            self.sim_buf["neg"].clear()
            self.sim_buf["hn"].clear()
        self.run.log(log_dict)
        prefix = "Initial eval" if epoch is None else f"Epoch {epoch}, Step {step}"
        train_str = ""
        if train_clm_loss is not None:
            train_str = f"Train CLM: {train_clm_loss:.4f}, "
            if train_cl_loss is not None:
                train_str += f"Train CL: {train_cl_loss:.4f}, "
        print(
            f"{prefix}, {train_str}"
            f"Val CLM: {results['val_clm_loss']:.4f}, "
            f"Val CL: {results['val_cl_loss']:.4f}, "
            f"Val BLEU: {results['val_bleu']:.2f}, "
            f"Val chrF: {results['val_chrf']:.2f}"
        )
        return results
    
    def maybe_save_best(self, model, results, step_label):

        if self.training_args.use_BLEU:
            score = results["val_bleu"]
            is_better = self.best_score is None or score > self.best_score
        else:
            score = results["val_clm_loss"]
            is_better = self.best_score is None or score < self.best_score
        if not is_better:
            return
        self.best_score = score
        if os.path.isdir(self.best_ckpt_dir):
            shutil.rmtree(self.best_ckpt_dir)
        os.makedirs(self.best_ckpt_dir, exist_ok=True)
        model.save_pretrained(self.best_ckpt_dir)
        with open(os.path.join(self.best_ckpt_dir, "results.json"), "w") as f:
            json.dump({**results, "step_label": step_label}, f)
        print(f"New best checkpoint: {step_label} (score={score:.4f})")

    def promote_best_model(self):
        # Promote the single retained best checkpoint to training_output_dir.
        if self.training_args.save_best_checkpoint and os.path.isdir(self.best_ckpt_dir):
            for item in os.listdir(self.best_ckpt_dir):
                if item == "results.json":
                    continue
                src = os.path.join(self.best_ckpt_dir, item)
                dst = os.path.join(self.model_args.training_output_dir, item)
                if os.path.isdir(src):
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                else:
                    shutil.copy2(src, dst)
            print(f"Best checkpoint score: {self.best_score:.4f}")
            shutil.rmtree(self.best_ckpt_dir)

    