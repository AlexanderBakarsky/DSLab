from .processing import load_raw_dataset, filter_parallel_data, format_dataset, make_tokenize
from .collators import CLCollator

from torch.utils.data import DataLoader
import numpy as np
from .loaders import MultilingualDataLoader

def _get_evaluation_data(split, max_size, source_l, target_l, bs, cache_folder,
                        num_examples=0, instruct=False, tokenizer=None,
                        num_workers=0, dataset="opus100", icl_split=None, icl_seed=42,
                        icl_dataset=None):
    ds = load_raw_dataset(dataset, split, source_l, target_l, max_size, cache_folder)
    if icl_split is not None:
        icl_ds_name = icl_dataset if icl_dataset is not None else dataset
        icl_ds = load_raw_dataset(icl_ds_name, icl_split, source_l, target_l, max_size=None, cache_dir=cache_folder)
        rng = np.random.default_rng(icl_seed)
        indices = rng.choice(len(icl_ds), num_examples, replace=False)
        icl_examples = {"source": [icl_ds["source"][i] for i in indices],
                        "target": [icl_ds["target"][i] for i in indices]}
        print(f"[ICL] Selected {num_examples} demos from '{icl_ds_name}/{icl_split}' (seed={icl_seed}, indices={indices.tolist()})")
        for j, (s, t) in enumerate(zip(icl_examples["source"], icl_examples["target"])):
            print(f"  [{j}] SRC: {s}")
            print(f"  [{j}] TGT: {t}")
    else:
        icl_examples = None
    formatted = format_dataset(ds, source_l, target_l, tokenizer, num_examples=num_examples, instruct=instruct, icl_examples=icl_examples, training=False)
    return DataLoader(formatted, batch_size=bs, shuffle=False, num_workers=num_workers)


def _get_training_data(split, max_size, source_l, target_l, tokenizer, bs, max_len, cache_folder, num_workers=0, dataset="opus100", instruct=False, pack_clm=False, clm_block_size=None, filter_data=False, filter_lang_id=False, filter_lang_id_model_path=None, mix_prompt_formats=True):

    ds = load_raw_dataset(dataset, split, source_l, target_l, max_size, cache_folder)
    if filter_data:
        ds = filter_parallel_data(
            ds, source_l, target_l,
            lang_id=filter_lang_id,
            lang_id_model_path=filter_lang_id_model_path,
        )
    formatted = format_dataset(ds, source_l, target_l, tokenizer, num_examples=0, instruct=instruct, training=True, mix_prompt_formats=mix_prompt_formats)
    tokenized = formatted.map(make_tokenize(tokenizer, max_len)).remove_columns(["prompt", "target", "src"])

    collator = CLCollator(tokenizer, max_len, pack_clm=pack_clm, clm_block_size=clm_block_size or max_len)

    data_loader = DataLoader(
        tokenized,
        batch_size=bs,
        shuffle=True,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=False,  # GB10 has shared CPU/GPU memory; pinning gives no benefit
    )

    return data_loader

def get_training_loader(lang_pairs, training_args, model_args, tokenizer):

    loaders = {
        f"{s}-{t}": _get_training_data(
            training_args.train_split, training_args.size_training_data,
            s, t, tokenizer, training_args.batch_size,
            model_args.max_context_len, model_args.cache_dir,
            training_args.num_workers, dataset=training_args.train_dataset,
            instruct=model_args.use_instruct, pack_clm=training_args.pack_clm,
            clm_block_size=training_args.clm_block_size,
            filter_data=training_args.filter_data,
            filter_lang_id=training_args.filter_lang_id,
            filter_lang_id_model_path=training_args.filter_lang_id_model_path,
        )
        for s, t in lang_pairs
    }
    return MultilingualDataLoader(loaders, seed=model_args.seed)

def get_loop_evaluation_loader(lang_pairs, training_args, eval_args, model_args, tokenizer):
    return {
        f"{s}-{t}": _get_evaluation_data(
            training_args.train_val_split,
            training_args.size_eval_data,
            s, t,
            training_args.batch_size,
            model_args.cache_dir,
            num_examples=training_args.train_eval_num_examples,
            instruct=model_args.use_instruct,
            tokenizer=tokenizer,
            num_workers=training_args.num_workers,
            dataset=training_args.train_val_dataset or training_args.train_dataset,
            icl_split=eval_args.icl_split if training_args.train_eval_num_examples > 0 else None,
            icl_dataset=eval_args.icl_dataset if training_args.train_eval_num_examples > 0 else None,
        )
        for s, t in lang_pairs
    }

    

def get_final_evaluation_loader(lang_pairs, eval_args, model_args, eval_tokenizer):
    return  {
        f"{s}-{t}": _get_evaluation_data(
            eval_args.split,
            eval_args.max_size,
            s, t, 
            eval_args.eval_batch_size, 
            model_args.cache_dir,
            num_examples=eval_args.num_examples,
            instruct=model_args.use_instruct,
            tokenizer=eval_tokenizer,
            num_workers=1,
            dataset=eval_args.dataset,
            icl_split=eval_args.icl_split,
            icl_dataset=eval_args.icl_dataset,
        )
        for s, t in lang_pairs
    }