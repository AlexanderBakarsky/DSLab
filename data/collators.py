import torch
from dataclasses import dataclass
from transformers import PreTrainedTokenizerBase


@dataclass
class CLCollator:
    tokenizer: PreTrainedTokenizerBase
    max_len: int
    pack_clm: bool = False
    clm_block_size: int = None  # defaults to max_len if None

    def __call__(self, features):
        bs = len(features)
        num_sent = len(features[0]["input_ids"])  # 2: src + tgt

    # CL:
        # Flatten and truncate
        flat_features = [
            {"input_ids": f["input_ids"][i][:self.max_len], "attention_mask": f["attention_mask"][i][:self.max_len]}
            for f in features
            for i in range(num_sent)
        ]

        #pad
        cl_batch = self.tokenizer.pad(flat_features, padding=True, return_tensors="pt")
        #reshape
        batch = {k: v.view(bs, num_sent, -1) for k, v in cl_batch.items()}



    # CLM:
        for f in features:
            if f["clm_prompt_len"] > self.max_len:
                print(f"[WARN] CLM prompt truncated: prompt_len={f['clm_prompt_len']} > max_len={self.max_len}")

        #truncate
        clm_input_ids = [f.pop("clm_input_ids")[:self.max_len] for f in features]

        # clamp prompt_len to truncated length - avoids all-(-100) label rows and NaN loss
        clm_prompt_len = [min(f.pop("clm_prompt_len"), len(ids)) for f, ids in zip(features, clm_input_ids)]

        pad_id  = self.tokenizer.pad_token_id

        if self.pack_clm:
            block_size = self.clm_block_size or self.max_len
            bos_id = self.tokenizer.bos_token_id
            eos_id = self.tokenizer.eos_token_id

            # Per-example label masks (1 = score, 0 = ignore). Prompt positions get 0;
            # target positions get 1. BOS/EOS are always scored (matches CrossLingualAlignment).
            flat_ids, flat_mask = [], []
            for ids, pl in zip(clm_input_ids, clm_prompt_len):
                mask = [0] * pl + [1] * (len(ids) - pl)
                for j, tid in enumerate(ids):
                    if tid == bos_id or tid == eos_id:
                        mask[j] = 1
                flat_ids.extend(ids)
                flat_mask.extend(mask)

            # Chunk into fixed-size blocks; pad the tail block with pad tokens (label -100)
            # so we keep the remainder instead of dropping it.
            total = len(flat_ids)
            n_blocks = (total + block_size - 1) // block_size
            pad_needed = n_blocks * block_size - total
            flat_ids += [pad_id] * pad_needed
            flat_mask += [0] * pad_needed

            packed_ids = torch.tensor(flat_ids).view(n_blocks, block_size)
            packed_mask = torch.tensor(flat_mask).view(n_blocks, block_size)
            packed_labels = torch.where(packed_mask.bool(), packed_ids, torch.full_like(packed_ids, -100))

            # NaN guard: if a row has no scored positions, unmask its last token.
            all_masked = (packed_labels == -100).all(dim=1)
            if all_masked.any():
                packed_labels[all_masked, -1] = packed_ids[all_masked, -1]

            # Attention mask is all 1s (no padding within packed blocks except the tail,
            # which is masked out via labels=-100). Default causal mask spans the block,
            # so tokens can attend across example boundaries -- matches CrossLingualAlignment
            # and provides implicit few-shot signal.
            batch["clm_input_ids"] = packed_ids
            batch["clm_target_ids"] = packed_labels
            # Full blocks contain no padding; only the tail block is pad-padded.
            # Mask out those tail pads so real tokens don't attend to them.
            attn = torch.ones_like(packed_ids)
            if pad_needed > 0:
                attn[-1, block_size - pad_needed:] = 0
            batch["clm_attention_mask"] = attn
        else:
            max_len_clm = max(len(ids) for ids in clm_input_ids)

            #pad
            batch["clm_input_ids"] = torch.tensor([
                ids + [pad_id] * (max_len_clm - len(ids)) for ids in clm_input_ids
            ])
            batch["clm_target_ids"] = torch.tensor([
                [-100] * pl + ids[pl:] + [-100] * (max_len_clm - len(ids))
                for ids, pl in zip(clm_input_ids, clm_prompt_len)
            ])
            batch["clm_attention_mask"] = torch.tensor([
                [1] * len(ids) + [0] * (max_len_clm - len(ids)) for ids in clm_input_ids
            ])

        return batch


