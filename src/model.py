import torch
import torch.nn as nn
from transformers import DistilBertModel

# -----------------------------------------------------------------------------
# Adapter module (task-specific bottleneck)
# -----------------------------------------------------------------------------

class Adapter(nn.Module):
    def __init__(self, hidden_dim: int, adapter_size: int, non_linearity: str = "relu"):
        super().__init__()
        act = nn.ReLU() if non_linearity.lower() == "relu" else nn.GELU()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, adapter_size),
            act,
            nn.Linear(adapter_size, hidden_dim),
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1e-3)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x) + x  # residual

# -----------------------------------------------------------------------------
# Simple patch embedding for vision input
# -----------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    def __init__(self, img_size: int, patch_size: int, in_ch: int, embed_dim: int):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # B, D, H/ps, W/ps
        x = x.flatten(2).transpose(1, 2)  # B, N, D
        return x

# -----------------------------------------------------------------------------
# Unified DistilBERT classifier (works for both text & images)
# -----------------------------------------------------------------------------

class DistilClassifier(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.is_text = cfg.dataset.name != "cifar10"

        self.backbone = DistilBertModel.from_pretrained(cfg.model.name, cache_dir=".cache/")
        hidden_dim = self.backbone.config.hidden_size

        # Vision patch embedding (used only for CIFAR-10)
        if not self.is_text:
            self.patch_embed = PatchEmbed(cfg.dataset.image_size, cfg.dataset.patch_size, 3, hidden_dim)
        else:
            self.patch_embed = None

        # Optional adapter
        if cfg.model.get("adapter") and cfg.model.adapter.enabled:
            self.adapter = Adapter(hidden_dim, cfg.model.adapter.adapter_size, cfg.model.adapter.non_linearity)
        else:
            self.adapter = None

        self.dropout = nn.Dropout(cfg.model.classification_head.dropout)
        self.cls_head = nn.Linear(hidden_dim, cfg.model.classification_head.num_labels)

        if cfg.model.freeze_base_model:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, input_ids=None, attention_mask=None, pixel_values=None):
        if self.is_text:
            outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        else:
            embeds = self.patch_embed(pixel_values)
            attn_mask = torch.ones(embeds.size()[:2], dtype=torch.long, device=embeds.device)
            outputs = self.backbone(inputs_embeds=embeds, attention_mask=attn_mask)

        pooled = outputs.last_hidden_state[:, 0]  # CLS token/first patch
        if self.adapter:
            pooled = self.adapter(pooled)
        logits = self.cls_head(self.dropout(pooled))
        return logits

# -----------------------------------------------------------------------------
# Factory
# -----------------------------------------------------------------------------

def build_model(cfg):
    return DistilClassifier(cfg)