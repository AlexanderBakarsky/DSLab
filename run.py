import logging
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional
import torch
import wandb
import numpy as np
import transformers
from transformers import HfArgumentParser, AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from train.finetuning import Trainer
from utils import fix_seeds, load_model_and_tokenizer, save_model_and_tokenizer, load_peft_model, build_lang_pairs
from evaluate import evaluate
from data import get_training_loader, get_final_evaluation_loader, get_loop_evaluation_loader
from investigation import log_alignment_uniformity, plot_embeddings

assert os.environ.get("HF_TOKEN") is not None, "Huggingface token not found. Please set the HF_TOKEN environment variable."

@dataclass 
class ModelArguments:

    #Defined here becuase they are shared between training and evaluation logic
    source_lang:str = field(
        metadata = {"help": "Language to translate from."}
    )

    target_lang:str = field(
        metadata={"help":"Language to translate from"}

    )
    #model to load for training
    model_name: str = field(
        metadata={"help": "Model to use as base model. Give either Huggingface ID or path."}
    )

    cache_dir: str = field(
        default = os.getenv("SCRATCH_DIR"),
        metadata={"help":"Cache dir for tokenizer, model, datasets etc."}
    )

    # Evaluation
    eval_model: Optional[str] = field( 
        default = None,
        metadata =  {"help":"Huggingface ID or path to model to evaluate."}
    )

    

    # Use instruction tuning
    use_instruct: bool = field(
        default=False,
        metadata={"help": "Whether to use instruction tuned model."}
    )

    #where to store trained model
    training_output_dir: str = field(
        default = os.getenv("TEAM_DIR") + "/apertus_ft",
        metadata={"help":"Base directory where trained models are stored."}
    )
    training_output_subdir: Optional[str] = field(
        default=None,
        metadata={"help": "Optional sub-directory under training_output_dir for this run. Useful for sweeps so each run gets its own folder under a shared base."}
    )

    
    run_name: Optional[str] = field(
        default=None,
        metadata={"help": "WandB run name. Defaults to 'apertus-{src}-{tgt}-{CL/no-CL}-lora'."}
    )
    group: Optional[str] = field(
        default = None,
        metadata = {"help":"Group in project to add the run to."}
    )

    # Random seed
    seed: int = field(
        default=0,
        metadata={"help": "Random seed for reproducibility."}
    )

    # Maximum context length for evaluation
    max_context_len: int = field(
        default=512,
        metadata={"help": "Maximum context length for input sequences."}
    )

    max_generation_context_len: int = field(
        default=128,
        metadata={"help": "Maximum context length for generated sequences."}
    )


@dataclass
class TrainingArguments:
    """
    Arguments for training and evaluation, including hyperparameters and model-specific settings.
    """
    #train model befor eval

    do_train: bool = field(
        default = True,
        metadata={"help": "Is the model trained before evaluation."}
    )

    train_dataset: str = field(
        default="opus100",
        metadata={"help": "Dataset to use for training."}
    )

    train_split: str = field(
        default="train",
        metadata={"help": "Split to use for training."}
    )

    train_val_split: str = field(
        default="validation",
        metadata={"help": "Split to use for validation during training."}
    )

    train_val_dataset: Optional[str] = field(
        default=None,
        metadata={"help": "Dataset to use for validation during training. Defaults to train_dataset if unset."}
    )

    # Size of the parallel dataset to train on
    size_training_data: int = field(
        default = 5000,
        metadata={"help":"Size of the parallel training dataset."}
    )

    size_eval_data: int = field(
        default = 200,
        metadata={"help":"Size of the parallel evaluation dataset. Used to monitor training, differentiate from the full evaluation set."}
    ) 

    # Learning rate
    lr: float = field(
        default=2e-4,
        metadata={"help": "Learning rate for the optimizer."}
    )

    # LoRA-specific arguments
    lora_rank: int = field(
        default=8,
        metadata={"help": "Rank for LoRA (Low-Rank Adaptation) layers."}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "Dropout rate for LoRA layers."}
    )

    # Training epochs
    epochs: int = field(
        default=1,
        metadata={"help": "Number of training epochs."}
    )

    # Batch size
    batch_size: int = field(
        default=8,
        metadata={"help": "Batch size for training and evaluation."}
    )

    # Micro-batch sizes for in-batch chunking. The loader batch_size is the
    # effective batch; CLM is forwarded in clm_micro_bs chunks (gradients
    # accumulated, scaled by 1/n_chunks), CL runs once per loader batch via
    # GradCache with internal chunking by cl_micro_bs. batch_size must be a
    # multiple of both.
    clm_micro_bs: int = field(
        default=1,
        metadata={"help": "Micro-batch size for CLM forward+backward chunks. batch_size must be a multiple of this."}
    )
    cl_micro_bs: int = field(
        default=1,
        metadata={"help": "Micro-batch size for CL GradCache chunks. batch_size must be a multiple of this."}
    )

    # Evaluation steps
    eval_steps: int = field(
        default=50,
        metadata={"help": "Number of evaluation steps."}
    )

    # Maximum gradient norm
    max_grad_norm: float = field(
        default=1.0,
        metadata={"help": "Maximum gradient norm for gradient clipping."}
    )

    # Number of workers for data loading
    num_workers: int = field(
        default=8,
        metadata={"help": "Number of workers for data loading."}
    )

    # Weight decay
    weight_decay: float = field(
        default=0.1,
        metadata={"help": "Weight decay for the optimizer."}
    )

    # Pooler type
    pooler_type: str = field(
        default="avg-1",
        metadata={"help": "Type of pooler to use for all-vs-all CL (e.g., 'avg-10' for averaging the sequence embeddings after the 10-th layer)."}
    )
    hn_pooler_type: Optional[str] = field(
        default=None,
        metadata={"help": "Type of pooler to use for hard negatives CL. If None, reuses pooler_type."}
    )

    # Use contrastive learning
    use_cl: bool = field(
        default=True,
        metadata={"help": "Whether to use contrastive learning."}
    )

    # Contrastive learning weight
    cl_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for the contrastive learning loss."}
    )

    # Causal language modeling weight
    clm_weight: float = field(
        default=1.5,
        metadata={"help": "Weight for the causal language modeling loss."}
    )

    # Contrastive learning temperature
    cl_temperature: float = field(
        default=0.05,
        metadata={"help": "Temperature for contrastive learning."}
    )

    # Warmup fraction
    warmup_fraction: float = field(
        default=0.05,
        metadata={"help": "Fraction of training steps for warmup."}
    )

    use_BLEU: bool = field(
        default=True,
        metadata={"help": "Whether to compute BLEU during training-loop evaluation."}
    )

    use_chrF: bool = field(
        default=False,
        metadata={"help": "Whether to compute chrF during training-loop evaluation. chrF is always computed in the final evaluation regardless of this flag."}
    )

    train_eval_num_examples: int = field(
        default=0,
        metadata={"help": "Number of few-shot ICL demos to use in the training-loop evaluation. When >0, reuses icl_split/icl_dataset from EvaluationArguments so demos match the final eval."}
    )

    mix_prompt_formats: bool = field(
        default=True,
        metadata={"help": "If true, sample a random template from BARE_TEMPLATES/INSTRUCT_TEMPLATES per training example (regularizes against prompt-format overfitting and improves few-shot ICL format-following). If false, always use index 0 (the canonical eval format) — gives 100% training signal on the eval format at the cost of format-following flexibility."}
    )

    overwrite_output_dir:bool = field(
        default = False,
        metadata= {"help": "A switch for safety."}
    )

    pack_clm: bool = field(
        default=False,
        metadata={"help": "Pack CLM examples into fixed-size blocks (GPT-style sequence packing). No document-boundary attention masking, so tokens attend across packed examples."}
    )

    clm_block_size: Optional[int] = field(
        default=None,
        metadata={"help": "Block size for CLM sequence packing. Defaults to max_context_len if not set. Only used when pack_clm is true."}
    )

    cl_hard_negatives: bool = field(
        default=False,
        metadata={"help": "Use CLM argmax predictions as per-example hard negatives in the CL loss. Requires use_cl=true and pack_clm=false."}
    )

    cl_grad_src: bool = field(
        default=True,
        metadata={"help": "Whether to differentiate through the source embeddings in the CL loss."}
    )

    cl_grad_tgt: bool = field(
        default=True,
        metadata={"help": "Whether to differentiate through the target embeddings in the CL loss."}
    )

    hn_bleu_threshold: Optional[float] = field(
        default=None,
        metadata={"help": "BLEU score threshold (0-100) for filtering hard negatives. Hard negatives scoring >= threshold against the target are excluded from the CL loss. Disabled if None."}
    )

    hn_cos_threshold: Optional[float] = field(
        default=None,
        metadata={"help": "Cosine similarity threshold (-1 to 1) for filtering hard negatives. Hard negatives whose pooled embedding has cosine similarity >= threshold against the target are excluded from the CL loss. Uses hn_pooler_type. Disabled if None. Combined with hn_bleu_threshold via OR."}
    )

    cl_center_mode: str = field(
        default="off",
        metadata={"help": "Mean-centering of pooled embeddings before cosine similarity. 'off': no centering. 'joint': subtract one shared EMA mean of (z1, z2) — cosine measures content alignment minus residual language drift, so CL has to push the language axis to zero structurally (preferred when you want a language-agnostic encoder). 'per_side': subtract separate EMA means for src and tgt — cosine measures content-only alignment, language axis hidden by the metric (cleaner diagnostic histograms but does not train the model to be language-agnostic)."}
    )

    cl_center_momentum: float = field(
        default=0.99,
        metadata={"help": "EMA momentum for the running mean(s) used by cl_center_mode. Higher = slower update. 0.99 → effective averaging window ~100 batches."}
    )

    save_best_checkpoint: bool = field(
        default=True,
        metadata={"help": "If true, save the best-scoring intermediate checkpoint during training (by val_bleu if use_BLEU else val_clm_loss) and promote it to training_output_dir at the end. If false, only the final post-training save is kept."}
    )


    use_lora: bool = field(
        default=True,
        metadata={"help": "If true, wrap the base model with LoRA adapters and save/load only adapter weights. If false, do full fine-tuning of all weights and save/load the full model."}
    )

    filter_data: bool = field(
        default=False,
        metadata={"help": "If true, apply length/empty/ratio/identical/dedup filtering to the training parallel data after loading."}
    )

    filter_lang_id: bool = field(
        default=False,
        metadata={"help": "If true (and filter_data=true), additionally apply fastText lid.176 language-ID filtering. Requires the fasttext package and lid.176.bin (path from filter_lang_id_model_path or FASTTEXT_LID_PATH env var)."}
    )

    filter_lang_id_model_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to fastText lid.176.bin for language-ID filtering. If unset, falls back to the FASTTEXT_LID_PATH environment variable."}
    )


@dataclass
class EvaluationArguments:
    """
    Arguments for evaluation, including paths, batch sizes, and context lengths.
    """

    #dataset to evaluate on 
    dataset: str = field(
        metadata= {"help":"Dataset to evaluate on."}
    )
    split: str = field(
        metadata={"help": "Split from the dataset to evaluate on. See splits specific to the dataset."}
    )

    do_evaluate:bool = field(
        default = True,
        metadata = {"help":"Run evaluation on the model."}
    )

    do_investigate:bool = field(
        default = False,
        metadata= {"help":"Run alignment_uniformity studies and plot t-SNE of language representation."}
    )
    
    do_plot:bool = field(
        default= False,
        metadata = {"help":"Produce t-SNE plots of language representation."}
    )

    investigation_languages: List[str] = field(
        default_factory=list,
        metadata={"help": "Non-English language codes to investigate alignment/uniformity against English. Must be non-empty and contain no 'en' entries when do_investigate=True."}
    )

    # Directory for evaluation outputs
    eval_output_dir: str = field(
        default="./evals",
        metadata={"help": "Directory to save evaluation results."}
    )

    # Evaluation batch size
    eval_batch_size: int = field(
        default=4,
        metadata={"help": "Batch size for evaluation."}
    )

    max_size: Optional[int] = field(
        default = None,
        metadata={"help":"Possibly truncate evaluation set. If not set, the entire dataset is used."}
    )

    num_examples: int = field(
        default = 0,
        metadata={"help":"Number of examples for multi-shot evalluation."}
    )

    icl_split: Optional[str] = field(
        default = None,
        metadata={"help": "Dataset split to sample ICL examples from. If not set, examples are sampled from the evaluation split."}
    )

    icl_dataset: Optional[str] = field(
        default = None,
        metadata={"help": "Dataset to draw ICL examples from. If not set, defaults to the evaluation dataset. Use this to e.g. evaluate on opus100 with FLORES dev demos."}
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    # create logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    sh.setFormatter(formatter)

    logger.addHandler(sh)

    
    logger.info("****Parsing arguments****")
    parser = HfArgumentParser((ModelArguments, TrainingArguments, EvaluationArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, training_args, eval_args = parser.parse_yaml_file(yaml_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, training_args, eval_args  = parser.parse_args_into_dataclasses()

    # PyYAML's YAML 1.1 resolver parses unquoted scientific notation like `5e-5` as a
    # string rather than a float (it requires a decimal point: `5.0e-5`). HfArgumentParser
    # does not coerce string values to the declared field type, so silently-stringified
    # floats propagate until something downstream (e.g. the optimizer) crashes. Coerce
    # any `float`-typed field whose value came in as a string.
    import dataclasses
    for args_obj in (model_args, training_args, eval_args):
        for f in dataclasses.fields(args_obj):
            if f.type is float and isinstance(getattr(args_obj, f.name), str):
                setattr(args_obj, f.name, float(getattr(args_obj, f.name)))

    # Resolve `training_output_dir / training_output_subdir` once so all
    # downstream code can keep reading `model_args.training_output_dir` as the
    # final path.
    if model_args.training_output_subdir:
        model_args.training_output_dir = os.path.join(
            model_args.training_output_dir, model_args.training_output_subdir
        )

    if (
        model_args.training_output_dir is not None
        and os.path.exists(model_args.training_output_dir)
        and os.listdir(model_args.training_output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.overwrite_output_dir}) already exists and is not empty."
            "Use --overwrite_output_dir to overcome."
        )
    if(
        model_args.eval_model is None and model_args.model_name is None
        or training_args.do_train and model_args.model_name is None
        ):
        raise ValueError(
            f"Functinality requires a model name supplied.\
                  Supply at least one model and turn off the corresponding switch if train or eval should be skipped."
        )

    logger.info("****Instantiating WandB****")

    run = wandb.init(
    entity="**your wandb entity**",
    group = model_args.group,
    project="**your wandb project**",
    name=f"{model_args.run_name or 'apertus-' + ('CL' if training_args.use_cl else 'no-CL') + '-lora'}-{model_args.source_lang}-{model_args.target_lang}",
    config={
        "source language": model_args.source_lang,
        "target language": model_args.target_lang,
        "model": model_args.model_name,
        "eval model": model_args.eval_model,
        "use instruct": model_args.use_instruct,
        "train dataset": training_args.train_dataset,
        "train split": training_args.train_split,
        "train val split": training_args.train_val_split,
        "train val dataset": training_args.train_val_dataset or training_args.train_dataset,
        "size training data": training_args.size_training_data,
        "size eval data": training_args.size_eval_data,
        "eval dataset": eval_args.dataset,
        "eval split": eval_args.split,
        "eval max size": eval_args.max_size,
        "eval num examples": eval_args.num_examples,
        "eval icl split": eval_args.icl_split,
        "learning rate": training_args.lr,
        "lora rank": training_args.lora_rank,
        "lora dropout": training_args.lora_dropout,
        "epochs": training_args.epochs,
        "batch size": training_args.batch_size,
        "clm micro bs": training_args.clm_micro_bs,
        "cl micro bs": training_args.cl_micro_bs,
        "eval steps": training_args.eval_steps,
        "max grad norm": training_args.max_grad_norm,
        "weight decay": training_args.weight_decay,
        "pooler type": training_args.pooler_type,
        "hn pooler type": training_args.hn_pooler_type or training_args.pooler_type,
        "use contrastive learning": training_args.use_cl,
        "contrastive loss weight": training_args.cl_weight,
        "CLM loss weight": training_args.clm_weight,
        "contrastive temperature": training_args.cl_temperature,
        "cl grad src": training_args.cl_grad_src,
        "cl grad tgt": training_args.cl_grad_tgt,
        "cl center mode": training_args.cl_center_mode,
        "cl center momentum": training_args.cl_center_momentum,
        "hn bleu threshold": training_args.hn_bleu_threshold,
        "hn cos threshold": training_args.hn_cos_threshold,
        "warmup fraction": training_args.warmup_fraction,
        "seed": model_args.seed,
        "max context len": model_args.max_context_len,
        "max generation context len": model_args.max_generation_context_len,
        "pack clm": training_args.pack_clm,
        "clm block size": training_args.clm_block_size,
        "cl clm hard negatives": training_args.cl_hard_negatives,
        "use lora": training_args.use_lora,
        "use BLEU": training_args.use_BLEU,
        "use chrF": training_args.use_chrF,
        "train eval num examples": training_args.train_eval_num_examples,
        "mix prompt formats": training_args.mix_prompt_formats,
        "save best checkpoint": training_args.save_best_checkpoint,
        "filter data": training_args.filter_data,
        "filter lang id": training_args.filter_lang_id,
        "eval batch size": eval_args.eval_batch_size,
        "eval icl dataset": eval_args.icl_dataset,
    })

    #fix seeds
    fix_seeds(model_args.seed)

    logger.info("****Loading model and tokenizer****")
    model, tokenizer = load_model_and_tokenizer(model_args.model_name, model_args.cache_dir)
    
    # Construct language pairs for train and evaluation
    lang_pairs = build_lang_pairs(model_args.source_lang, model_args.target_lang)

    #Training
    logger.info("****Training****")
    if training_args.do_train:
        if training_args.use_lora:
            lora_config = LoraConfig(
                r=training_args.lora_rank,
                lora_alpha=training_args.lora_rank * 2,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
                lora_dropout=training_args.lora_dropout,
                bias="none",
                layers_to_transform=list(range(10)) + [30, 31]
            )
            train_model = get_peft_model(model, lora_config)
        else:
            # Full fine-tuning: all parameters are trainable by default.
            train_model = model
        
        #prepare data loaders
        train_loader = get_training_loader(lang_pairs, training_args, model_args, tokenizer)
        loop_eval_loader = get_loop_evaluation_loader(lang_pairs, training_args, eval_args,model_args, tokenizer)

        # Train on all languages simultaneously
        Trainer(train_model, tokenizer, train_loader, loop_eval_loader, training_args, model_args, run).train()
        save_model_and_tokenizer(train_model, tokenizer, model_args.training_output_dir)
        # Snapshot the parameter yaml next to the saved model so the run is
        # reproducible from the output dir alone. Tolerate the source being
        # gone (e.g. sweep-generated yamls cleaned up mid-run).
        if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
            yaml_src = os.path.abspath(sys.argv[1])
            if os.path.isfile(yaml_src):
                import shutil
                shutil.copy2(yaml_src, model_args.training_output_dir)
            else:
                logger.warning(f"Parameter yaml not found at end of training, skipping snapshot: {yaml_src}")

    
    
    eval_path = model_args.eval_model or model_args.training_output_dir
    is_adapter_dir = os.path.exists(os.path.join(eval_path, "adapter_config.json"))
    model_args.eval_model = eval_path

    if is_adapter_dir:
        logger.info(f"evaluate on LoRA adapter at {eval_path}")
        eval_model, eval_tokenizer = load_peft_model(model_args.model_name, eval_path, model_args.cache_dir)
    else:
        logger.info(f"evaluate on full model at {eval_path}")
        eval_model, eval_tokenizer = load_model_and_tokenizer(eval_path, model_args.cache_dir)
    
    if eval_args.do_evaluate:
        logger.info("****Evaluating****")
        eval_loader = get_final_evaluation_loader(
            lang_pairs,
            eval_args,
            model_args,
            eval_tokenizer=eval_tokenizer
        )
        results = evaluate(
            eval_model,
            eval_loader,
            eval_tokenizer,
            model_args.max_context_len,
            model_args.max_generation_context_len,
            use_chrF=True,
            use_cl=False,
            aggregate=False
        )

        # Output an evaluation result for every language in the
        for s, t in lang_pairs:
            pair = f"{s}-{t}"
            run.summary[f"{pair}-eval/BLEU"] = results[f"{pair}/val_bleu"]
            run.summary[f"{pair}-eval/chrF"] = results[f"{pair}/val_chrf"]

    if eval_args.do_investigate:
        logger.info("****Investigation****")
        if not eval_args.investigation_languages:
            raise ValueError("do_investigate=True requires at least one language in investigation_languages.")
        if any(lang == "en" for lang in eval_args.investigation_languages):
            raise ValueError("investigation_languages must not contain 'en'.")
        investigation_pairs = [(lang, "en") for lang in eval_args.investigation_languages]
        investigation_loader = get_final_evaluation_loader(
            investigation_pairs,
            eval_args,
            model_args,
            eval_tokenizer=eval_tokenizer,
        )
        log_alignment_uniformity(
            eval_model, eval_tokenizer, investigation_loader,
            training_args.pooler_type, model_args.max_context_len, run,
        )
        if eval_args.do_plot:
            plot_embeddings(
                eval_model, eval_tokenizer, investigation_loader,
                training_args.pooler_type, model_args.max_context_len,
                output_dir=eval_args.eval_output_dir,
                wandb_run=run,
            )



    run.finish()


if __name__ == "__main__":
    main()