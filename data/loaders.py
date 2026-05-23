import os
import random
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import datasets
from datasets import load_dataset
from torch.utils.data import DataLoader
from .attributes import _LANG_NAME, _FLORES_CODE



class DatasetLoader(ABC):
    # Seed for all dataloaders, fixed by the fix_seeds(seed) method in utils/common.py
    seed = int(np.random.randint(0, 2**31 - 1))
    @abstractmethod
    def __call__(self, split: str, source_lang: str, target_lang: str, max_size: Optional[int], cache_dir: str) -> datasets.Dataset: ...


class Opus100Loader(DatasetLoader):
    """Streams from OPUS-100 with a buffered shuffle so the subsample is not biased
    toward the head of the shard. Seed is drawn from the global numpy RNG, which
    `fix_seeds(model_args.seed)` initializes earlier in run.py — so subsamples are
    reproducible across runs that share the same `seed` config value."""
    

    def __call__(self, split: str, source_lang: str, target_lang: str, max_size: int, cache_dir: str) -> datasets.Dataset:
        assert source_lang in _LANG_NAME and target_lang in _LANG_NAME, "Language codes must be keys from lang_dict._LANG_NAME"
        assert split in ["train", "validation", "test"], "Split must be one of 'train', 'validation', 'test'"
        # OPUS-100 ships one directional config per pair, with the lang codes in
        # alphabetical order (e.g. "en-zh", never "zh-en"). Load the canonical config and
        # let the per-example dict lookup below pick the correct side for the requested
        # direction — this makes both "en-zh" and "zh-en" work transparently.
        a, b = sorted([source_lang, target_lang])
        config = f"{a}-{b}"
        ds = load_dataset("Helsinki-NLP/opus-100", config, split=split, cache_dir=cache_dir, streaming=True)
        if max_size is not None:
            shuffle_seed = DatasetLoader.seed
            buffer_size = max(10 * max_size, 10_000)
            ds = ds.shuffle(seed=shuffle_seed, buffer_size=buffer_size)
        sources, targets = [], []
        for ex in ds:
            t = ex["translation"]
            sources.append(t[source_lang])
            targets.append(t[target_lang])
            if max_size is not None and len(sources) >= max_size:
                break
        return datasets.Dataset.from_dict({
            "source": sources,
            "target": targets,
        })

class WMT14Loader(DatasetLoader):
    def __call__(self, split: str, source_lang: str, target_lang: str, max_size: int, cache_dir: str) -> datasets.Dataset:
        assert source_lang in _LANG_NAME and target_lang in _LANG_NAME, "Language codes must be keys from lang_dict._LANG_NAME"
        assert split in ["train", "validation", "test"], "Split must be one of 'train', 'validation', 'test'"
        config = f"{source_lang}-{target_lang}"
        ds = load_dataset("wmt/wmt14", config, split=split, cache_dir=cache_dir, streaming=True)
        sources, targets = [], []
        for ex in ds:
            t = ex["translation"]
            sources.append(t[source_lang])
            targets.append(t[target_lang])
            if max_size is not None and len(sources) >= max_size:
                break
        return datasets.Dataset.from_dict({
            "source": sources,
            "target": targets,
        })

class Flores200Loader(DatasetLoader):
    data_dir = os.path.join(os.environ["TEAM_DIR"], "datasets/flores200/flores200_dataset")

    def __call__(self, split: str, source_lang: str, target_lang: str, max_size: int, cache_dir: str) -> datasets.Dataset:
        assert source_lang in _FLORES_CODE and target_lang in _FLORES_CODE, "Language codes must be keys from lang_dict._FLORES_CODE"
        assert split in ["dev", "devtest"], "Split must be one of 'dev', 'devtest'"
        src_code = _FLORES_CODE[source_lang]
        tgt_code = _FLORES_CODE[target_lang]
        def read_lines(lang_code):
            path = os.path.join(self.data_dir, split, f"{lang_code}.{split}")
            with open(path) as f:
                return [line.rstrip("\n") for line in f]
        return datasets.Dataset.from_dict({
            "source": read_lines(src_code)[:max_size] if max_size is not None else read_lines(src_code),
            "target": read_lines(tgt_code)[:max_size] if max_size is not None else read_lines(tgt_code),
        })

class NLLBLoader(DatasetLoader):
    """Loads NLLB bitext from OPUS (https://opus.nlpl.eu/NLLB/). LASER scores are
    not preserved in the OPUS distribution, so no score filtering is applied."""

    def __call__(self, split: str, source_lang: str, target_lang: str, max_size: int, cache_dir: str) -> datasets.Dataset:
        assert source_lang in _LANG_NAME and target_lang in _LANG_NAME, "Language codes must be keys from lang_dict._LANG_NAME"
        assert split == "train", "NLLB loader only provides a 'train' split"

        # OPUS uses lang-code pairs sorted alphabetically, e.g. "en-kk".
        a, b = sorted([source_lang, target_lang])
        pair = f"{a}-{b}"
        url = f"https://object.pouta.csc.fi/OPUS-NLLB/v1/moses/{pair}.txt.zip"

        import urllib.request, zipfile, io
        cache_subdir = os.path.join(cache_dir, "opus_nllb", pair)
        os.makedirs(cache_subdir, exist_ok=True)
        src_file = os.path.join(cache_subdir, f"NLLB.{pair}.{source_lang}")
        tgt_file = os.path.join(cache_subdir, f"NLLB.{pair}.{target_lang}")

        if not (os.path.exists(src_file) and os.path.exists(tgt_file)):
            with urllib.request.urlopen(url) as resp:
                data = resp.read()
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                zf.extractall(cache_subdir)

        sources, targets = [], []
        with open(src_file, encoding="utf-8") as fs, open(tgt_file, encoding="utf-8") as ft:
            if max_size is None:
                for s_line, t_line in zip(fs, ft):
                    sources.append(s_line.rstrip("\n"))
                    targets.append(t_line.rstrip("\n"))
            else:
                import random
                rng = random.Random(0)
                for i, (s_line, t_line) in enumerate(zip(fs, ft)):
                    if i < max_size:
                        sources.append(s_line.rstrip("\n"))
                        targets.append(t_line.rstrip("\n"))
                    else:
                        j = rng.randint(0, i)
                        if j < max_size:
                            sources[j] = s_line.rstrip("\n")
                            targets[j] = t_line.rstrip("\n")
        return datasets.Dataset.from_dict({"source": sources, "target": targets})


class AllenAINLLBLoader(DatasetLoader):
    """Loads NLLB bitext from `allenai/nllb` on the HuggingFace Hub, which
    preserves per-pair LASER margin scores (the OPUS-NLLB redistribution strips
    them). Filters by LASER margin score >= `laser_score_min` (default 1.06, the
    threshold used in the NLLB paper for high-quality bitext).

    Requires `datasets<3.0` because `allenai/nllb` is a script-based dataset and
    `datasets>=3.0` removed script loading. Install with:
        pip install 'datasets==2.20.0'
    The other loaders in this module are 2.x-compatible, so this downgrade is
    safe for the rest of the codebase.
    """
    laser_score_min: float = 1.06

    def __call__(self, split: str, source_lang: str, target_lang: str, max_size: int, cache_dir: str) -> datasets.Dataset:
        assert source_lang in _FLORES_CODE and target_lang in _FLORES_CODE, \
            "Language codes must be keys from lang_dict._FLORES_CODE (NLLB uses BCP-47 with script)"
        assert split == "train", "allenai/nllb only provides a 'train' split"

        src_bcp = _FLORES_CODE[source_lang]
        tgt_bcp = _FLORES_CODE[target_lang]

        # AllenAI configs are named "{src}-{tgt}" but not always alphabetical
        # (NLLB_PAIRS is mostly sorted, CCMATRIX_PAIRS is not). Try both orders.
        last_err = None
        for a, b in [(src_bcp, tgt_bcp), (tgt_bcp, src_bcp)]:
            try:
                ds = load_dataset(
                    "allenai/nllb", f"{a}-{b}", split="train",
                    cache_dir=cache_dir, streaming=True, trust_remote_code=True,
                )
                file_src = a
                break
            except (ValueError, FileNotFoundError) as e:
                last_err = e
        else:
            raise RuntimeError(f"No allenai/nllb config for ({src_bcp}, {tgt_bcp}): {last_err}")

        ds = ds.filter(lambda ex: ex.get("laser_score") is not None and ex["laser_score"] >= self.laser_score_min)

        if max_size is not None:
            shuffle_seed = DatasetLoader.seed
            buffer_size = max(10 * max_size, 10_000)
            ds = ds.shuffle(seed=shuffle_seed, buffer_size=buffer_size)

        sources, targets = [], []
        for ex in ds:
            t = ex["translation"]
            s_text, t_text = t[file_src], t[tgt_bcp if file_src == src_bcp else src_bcp]
            if file_src != src_bcp:
                s_text, t_text = t_text, s_text
            sources.append(s_text)
            targets.append(t_text)
            if max_size is not None and len(sources) >= max_size:
                break

        return datasets.Dataset.from_dict({"source": sources, "target": targets})


class WMT24PPRMLoader(DatasetLoader):
    def __call__(self, split: str, source_lang: str, target_lang: str, max_size: int, cache_dir: str) -> datasets.Dataset:
        assert source_lang == "de" and target_lang.startswith("rm"), "WMT24PP-RM loader only supports German to Romansh translation"
        assert split=="test", "WMT24PP-RM loader only supports test split"
        config = f"de_DE-{target_lang}"
        ds = load_dataset("ZurichNLP/wmt24pp-rm", config, split=split, cache_dir=cache_dir)
        if max_size is not None:
            ds = ds.select(range(max_size))
        return datasets.Dataset.from_dict({
            "source": ds["source"],
            "target": ds["target"],
        })
    
DATASET_LOADERS: dict[str, DatasetLoader] = {
    "opus100":    Opus100Loader(),
    "wmt14":      WMT14Loader(),
    "flores200":  Flores200Loader(),
    "wmt24pp-rm": WMT24PPRMLoader(),
    "nllb":       NLLBLoader(),
    "nllb-laser": AllenAINLLBLoader(),
}


class MultilingualDataLoader:
    """Interleaves multiple per-language-pair DataLoaders so each batch contains
    examples from exactly one language pair, while across batches different pairs
    appear according to the chosen sampling strategy.

    Args:
        loaders:  dict mapping a language-pair string (e.g. "sw-en") to a DataLoader
                  produced by get_training_data.
        sampling: "proportional" — draw a language pair with probability proportional
                  to its dataset size (default; prevents over-fitting to small corpora).
                  "uniform"      — each language pair is equally likely per batch.

    Iteration semantics: one pass exhausts every language pair exactly once.
    Batches from exhausted pairs are dropped; the remaining active pairs are
    re-weighted at each step so sampling stays well-defined until all are done.

    Yields: batch dicts, identical to what a plain DataLoader would yield.
    """

    def __init__(self, loaders: dict[str, DataLoader], sampling: str = "uniform", seed: int = None):
        assert sampling in ("proportional", "uniform"), \
            f"sampling must be 'proportional' or 'uniform', got '{sampling}'"
        self.loaders = loaders
        if sampling == "proportional":
            sizes = {lp: len(dl.dataset) for lp, dl in loaders.items()}
            total = sum(sizes.values())
            self.weights = {lp: s / total for lp, s in sizes.items()}
        else:
            self.weights = {lp: 1.0 / len(loaders) for lp in loaders}
        # Use a dedicated RNG so pair-sampling order is reproducible under fix_seeds
        # and independent of other random.* consumers.
        self._rng = random.Random(seed if seed is not None else int(np.random.randint(0, 2**31 - 1)))

    def __len__(self):
        """Total number of batches across all language pairs."""
        return sum(len(dl) for dl in self.loaders.values())

    def __iter__(self):
        iterators = {lp: iter(dl) for lp, dl in self.loaders.items()}
        active = set(self.loaders.keys())

        while active:
            norm = sum(self.weights[lp] for lp in active)
            lp = self._rng.choices(
                list(active),
                weights=[self.weights[lp] / norm for lp in active],
                k=1,
            )[0]
            try:
                yield next(iterators[lp])
            except StopIteration:
                active.discard(lp)
