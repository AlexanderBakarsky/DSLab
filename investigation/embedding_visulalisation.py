import os
import torch
import numpy as np
import wandb
from tqdm import tqdm
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

from train.contrastive_learning import ContrastiveLoss
from utils import get_decoder_backbone


def plot_embeddings(model, tokenizer, investigation_loader, pooler_type, max_context_len, output_dir, wandb_run=None):
    """Encode all languages in investigation_loader, run a single joint t-SNE, and
    plot each language in a distinct colour. Saves the figure as SVG and uploads
    it to wandb as an HTML panel so the vector graphic is preserved.

    investigation_loader is the same dict produced by get_final_evaluation_loader
    for investigation pairs: {"{src_lang}-{tgt_lang}": DataLoader}, where each
    batch has "src" and "target" string fields.
    """
    device = next(model.parameters()).device
    cl = ContrastiveLoss(pooler_type, hard_neg_pooler_type=None, temp=-1)
    backbone = get_decoder_backbone(model)
    model.eval()

    # Accumulate embeddings keyed by language code.
    # English embeddings from all pairs are merged into one "en" bucket.
    lang_embeddings: dict[str, list] = {}

    with torch.no_grad():
        for pair_key, loader in investigation_loader.items():
            src_lang, tgt_lang = pair_key.split("-", 1)
            lang_embeddings.setdefault(src_lang, [])
            lang_embeddings.setdefault(tgt_lang, [])

            # Collect English only from the first pair that provides it so it
            # appears with the same number of points as each source language.
            collect_tgt = not lang_embeddings[tgt_lang]

            for batch in tqdm(loader, desc=f"Encoding {pair_key}"):
                src_enc = tokenizer(
                    batch["src"], return_tensors="pt", padding=True,
                    truncation=True, max_length=max_context_len, add_special_tokens=False,
                ).to(device)

                src_out = cl.encode(backbone, src_enc["input_ids"], src_enc["attention_mask"])
                lang_embeddings[src_lang].append(
                    cl.pooler(src_enc["attention_mask"], src_out).float().cpu()
                )

                if collect_tgt:
                    tgt_enc = tokenizer(
                        batch["target"], return_tensors="pt", padding=True,
                        truncation=True, max_length=max_context_len, add_special_tokens=False,
                    ).to(device)
                    tgt_out = cl.encode(backbone, tgt_enc["input_ids"], tgt_enc["attention_mask"])
                    lang_embeddings[tgt_lang].append(
                        cl.pooler(tgt_enc["attention_mask"], tgt_out).float().cpu()
                    )

    # Stack into one array and track per-point language labels.
    languages = list(lang_embeddings.keys())
    stacked, point_labels = [], []
    for lang in languages:
        embs = torch.cat(lang_embeddings[lang], dim=0).numpy()
        stacked.append(embs)
        point_labels.extend([lang] * len(embs))

    all_embeddings = np.concatenate(stacked, axis=0)
    point_labels = np.array(point_labels)

    print(f"Running t-SNE on {len(all_embeddings)} embeddings across {len(languages)} languages…")
    tsne = TSNE(n_components=2, random_state=42, perplexity=20, learning_rate=200)
    embeddings_2d = tsne.fit_transform(all_embeddings)

    # One colour per language drawn from tab10 / tab20 so colours are distinct.
    cmap = plt.cm.tab10 if len(languages) <= 10 else plt.cm.tab20
    colors = cmap(np.linspace(0, 1, len(languages)))

    fig, ax = plt.subplots(figsize=(10, 8))
    for lang, color in zip(languages, colors):
        mask = point_labels == lang
        ax.scatter(
            embeddings_2d[mask, 0], embeddings_2d[mask, 1],
            alpha=0.6, label=lang, s=5, color=color,
        )

    ax.set_xlabel("t-SNE axis 1")
    ax.set_ylabel("t-SNE axis 2")
    ax.legend(markerscale=3)
    fig.tight_layout()

    svg_path = os.path.join(output_dir, "tsne_embeddings.svg")
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved t-SNE plot to {svg_path}")

    if wandb_run is not None:
        with open(svg_path) as f:
            svg_content = f.read()
        wandb_run.log({"investigation/tsne": wandb.Html(svg_content)})

