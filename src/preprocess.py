import numpy as np
import torch
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets as tv_datasets, transforms
from datasets import load_dataset
from transformers import DistilBertTokenizerFast

# -----------------------------------------------------------------------------
# Collate utilities
# -----------------------------------------------------------------------------

def _vision_collate(batch):
    imgs = torch.stack([b[0]["pixel_values"] for b in batch])
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return {"pixel_values": imgs}, labels


def _text_collate(batch):
    max_len = max(x[0]["input_ids"].size(0) for x in batch)
    input_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    attn = torch.zeros(len(batch), max_len, dtype=torch.long)
    labels = torch.tensor([x[1] for x in batch]).long()
    for i, (inp, _) in enumerate(batch):
        seq_len = inp["input_ids"].size(0)
        input_ids[i, :seq_len] = inp["input_ids"]
        attn[i, :seq_len] = inp["attention_mask"]
    return {"input_ids": input_ids, "attention_mask": attn}, labels

# -----------------------------------------------------------------------------
# Dataset wrappers
# -----------------------------------------------------------------------------

class CIFAR10Wrapper(torch.utils.data.Dataset):
    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        return {"pixel_values": img}, label


class TextWrapper(torch.utils.data.Dataset):
    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        row = self.ds[idx]
        return {
            "input_ids": torch.tensor(row["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(row["attention_mask"], dtype=torch.long),
        }, int(row["label"])

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def prepare_tokenizer(cfg):
    if cfg.dataset.name == "alpaca-cleaned":
        tok_name = cfg.model.name if "distilbert" in cfg.model.name else "distilbert-base-uncased"
        return DistilBertTokenizerFast.from_pretrained(tok_name, cache_dir=".cache/")
    return None


def _split_dataset(ds, splits):
    total = len(ds)
    train_sz = int(total * splits.train)
    val_sz = int(total * splits.val)
    test_sz = total - train_sz - val_sz
    train_ds = ds.select(range(train_sz))
    val_ds = ds.select(range(train_sz, train_sz + val_sz))
    test_ds = ds.select(range(train_sz + val_sz, total))
    return train_ds, val_ds, test_ds


def prepare_dataloaders(cfg, tokenizer, trial_mode=False):
    """Return train/val/test dataloaders for the given run configuration."""

    # --------------------------- Vision (CIFAR-10) -----------------------
    if cfg.dataset.name == "cifar10":
        aug_list = []
        if cfg.dataset.augmentation.random_flip:
            aug_list.append(transforms.RandomHorizontalFlip())
        if cfg.dataset.augmentation.random_crop:
            aug_list.append(transforms.RandomCrop(cfg.dataset.image_size, padding=4))
        aug_list.append(transforms.ToTensor())
        transform = transforms.Compose(aug_list)

        full = tv_datasets.CIFAR10(root=".cache/", train=True, download=True, transform=transform)
        total_len = len(full)
        train_len = int(total_len * cfg.dataset.splits.train)
        val_len = int(total_len * cfg.dataset.splits.val)
        test_len = total_len - train_len - val_len
        train_ds, val_ds, test_ds = random_split(full, [train_len, val_len, test_len], generator=torch.Generator().manual_seed(42))

        if trial_mode:
            train_ds = Subset(train_ds, list(range(min(64, len(train_ds)))))
            val_ds = Subset(val_ds, list(range(min(64, len(val_ds)))))
            test_ds = Subset(test_ds, list(range(min(64, len(test_ds)))))

        train_loader = DataLoader(CIFAR10Wrapper(train_ds), batch_size=cfg.dataset.batch_size, shuffle=True, collate_fn=_vision_collate)
        val_loader = DataLoader(CIFAR10Wrapper(val_ds), batch_size=cfg.dataset.batch_size, shuffle=False, collate_fn=_vision_collate)
        test_loader = DataLoader(CIFAR10Wrapper(test_ds), batch_size=cfg.dataset.batch_size, shuffle=False, collate_fn=_vision_collate)
        return train_loader, val_loader, test_loader

    # --------------------------- Text (Alpaca cleaned) -------------------
    elif cfg.dataset.name == "alpaca-cleaned":
        assert tokenizer is not None, "Tokenizer required for text dataset"
        raw = load_dataset("yahma/alpaca-cleaned", cache_dir=".cache/")
        ds = raw["train"]  # dataset is not pre-split

        # Determine threshold for binary classification based on output length
        label_col = cfg.dataset.label_column
        lengths = [len(str(example[label_col])) for example in ds]  # modest size so memory OK
        threshold = int(np.median(lengths))

        # ---------------- Label assignment ------------------------------
        def add_label(example):
            example_len = len(str(example[label_col]))
            example["label"] = int(example_len > threshold)
            return example

        ds = ds.map(add_label, desc="Assigning binary labels from output length")

        # ---------------- Tokenisation ----------------------------------
        text_cols = cfg.dataset.text_columns

        def tok_fn(example):
            concat_text = " ".join(str(example[c]) for c in text_cols if example[c] is not None)
            enc = tokenizer(concat_text, truncation=True, max_length=cfg.dataset.max_seq_length)
            example["input_ids"] = enc["input_ids"]
            example["attention_mask"] = enc["attention_mask"]
            return example

        keep_cols = {"input_ids", "attention_mask", "label"}
        ds = ds.map(tok_fn, batched=False, remove_columns=[c for c in ds.column_names if c not in keep_cols])

        train_ds, val_ds, test_ds = _split_dataset(ds, cfg.dataset.splits)

        if trial_mode:
            train_ds = train_ds.select(range(min(128, len(train_ds))))
            val_ds = val_ds.select(range(min(128, len(val_ds))))
            test_ds = test_ds.select(range(min(128, len(test_ds))))

        train_loader = DataLoader(TextWrapper(train_ds), batch_size=cfg.dataset.batch_size, shuffle=True, collate_fn=_text_collate)
        val_loader = DataLoader(TextWrapper(val_ds), batch_size=cfg.dataset.batch_size, shuffle=False, collate_fn=_text_collate)
        test_loader = DataLoader(TextWrapper(test_ds), batch_size=cfg.dataset.batch_size, shuffle=False, collate_fn=_text_collate)
        return train_loader, val_loader, test_loader

    else:
        raise ValueError(f"Unsupported dataset {cfg.dataset.name}")