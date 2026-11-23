"""Frozen vision foundation models that map images [B, 3, H, W] to patch tokens [B, N, C]."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import torch
from torch import nn
from torchvision import transforms

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# Patch tokens [B, N, C] and the (h, w) grid they lie on.
Tokens = tuple[torch.Tensor, tuple[int, int]]


class Backbone(ABC):
    """A frozen model (`model`) that returns patch tokens for a batch of images."""

    model: nn.Module
    feature_dim: int
    # Whether the dataset should apply ImageNet normalization before the backbone.
    imagenet_normalize: bool = False

    def __init__(self, device: str) -> None:
        self.device = device

    @abstractmethod
    def tokens(self, images: torch.Tensor) -> Tokens: ...

    def freeze(self) -> None:
        for p in self.model.parameters():
            p.requires_grad = False


def _square_grid(n: int) -> tuple[int, int]:
    side = int(math.sqrt(n))
    return side, side


class DINOv3(Backbone):
    model_id = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    imagenet_normalize = True

    def __init__(self, device: str) -> None:
        super().__init__(device)
        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(self.model_id).to(device).eval()
        self.feature_dim = self.model.config.hidden_size

    def tokens(self, images: torch.Tensor) -> Tokens:
        out = self.model(pixel_values=images.to(self.device))[0]
        tokens = out[:, 1 + self.model.config.num_register_tokens :]
        ps = self.model.config.patch_size
        return tokens, (images.shape[2] // ps, images.shape[3] // ps)


class DermLIP(Backbone):
    """ViT-B/16 image tower of DermLIP (open_clip), without the projection head."""

    model_id = "hf-hub:redlessone/DermLIP_ViT-B-16"

    def __init__(self, device: str) -> None:
        super().__init__(device)
        import open_clip

        clip_model, _, _ = open_clip.create_model_and_transforms(self.model_id)
        self.model = clip_model.visual.to(device).eval()
        self.model.output_tokens = True
        size = self.model.image_size
        self.image_size = size if isinstance(size, int) else size[0]
        ps = self.model.patch_size
        self.patch_size = int(ps[0] if isinstance(ps, tuple) else ps)
        self.feature_dim = 768
        self.preprocess = transforms.Compose(
            [
                transforms.CenterCrop((self.image_size, self.image_size)),
                transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
            ]
        )

    def tokens(self, images: torch.Tensor) -> Tokens:
        images = self.preprocess(images.to(self.device))
        out = self.model(images)
        hidden = out[1] if isinstance(out, tuple) else out
        grid = (images.shape[2] // self.patch_size, images.shape[3] // self.patch_size)
        n = grid[0] * grid[1]
        hidden = hidden[:, 1:] if hidden.shape[1] == n + 1 else hidden[:, :n]
        return hidden, grid


class SigLIP(Backbone):
    model_id = "google/siglip-base-patch16-224"

    def __init__(self, device: str) -> None:
        super().__init__(device)
        from transformers import SiglipImageProcessor, SiglipVisionModel

        self.processor = SiglipImageProcessor.from_pretrained(self.model_id)
        self.model = SiglipVisionModel.from_pretrained(self.model_id).to(device).eval()
        self.feature_dim = self.model.config.hidden_size

    def tokens(self, images: torch.Tensor) -> Tokens:
        pixel_values = self.processor(images=images, return_tensors="pt")["pixel_values"]
        pixel_values = pixel_values.to(self.device)
        out = self.model(pixel_values=pixel_values)[0]
        ps = self.model.config.patch_size
        return out, (pixel_values.shape[2] // ps, pixel_values.shape[3] // ps)


class PanDerm(Backbone):
    """PanDerm ViT-B/16 image tower from the DermLIP-PanDerm open_clip checkpoint (via timm)."""

    repo_id = "redlessone/DermLIP_PanDerm-base-w-PubMed-256"

    def __init__(self, device: str) -> None:
        super().__init__(device)
        self.model = load_panderm(self.repo_id).to(device).eval()
        self.feature_dim = 768
        self.preprocess = transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD)

    def tokens(self, images: torch.Tensor) -> Tokens:
        out = self.model(self.preprocess(images).to(self.device))[:, 1:]
        return out, (images.shape[2] // 16, images.shape[3] // 16)


class SAM3(Backbone):
    model_id = "facebook/sam3"

    def __init__(self, device: str) -> None:
        super().__init__(device)
        from transformers import Sam3Model, Sam3Processor

        self.processor = Sam3Processor.from_pretrained(self.model_id)
        self.model = Sam3Model.from_pretrained(self.model_id).vision_encoder.to(device).eval()
        self.feature_dim = 1024

    def tokens(self, images: torch.Tensor) -> Tokens:
        pixel_values = self.processor(
            images=[img.detach().cpu() for img in images], return_tensors="pt"
        )["pixel_values"].to(self.device)
        out = self.model(pixel_values=pixel_values, output_hidden_states=False)
        hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out
        return hidden, _square_grid(hidden.shape[1])


class MedGemma(Backbone):
    """Vision tower of MedGemma; tokens are reshaped to a square grid (dropping a CLS token)."""

    model_id = "google/medgemma-4b-it"

    def __init__(self, device: str) -> None:
        super().__init__(device)
        from transformers import AutoImageProcessor, AutoModel

        self.processor = AutoImageProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        full = AutoModel.from_pretrained(self.model_id, trust_remote_code=True)
        for name in ("language_model", "text_model", "lm_head"):
            if hasattr(full, name):
                setattr(full, name, None)
        if getattr(full, "vision_tower", None) is not None:
            self.model = full.vision_tower
        elif getattr(full, "vision_model", None) is not None:
            self.model = full.vision_model
        else:
            self.model = full
        self.model = self.model.to(device).eval()
        self.feature_dim = self.model.config.hidden_size

    def tokens(self, images: torch.Tensor) -> Tokens:
        pixel_values = self.processor(
            images=[img.detach().cpu() for img in images], return_tensors="pt"
        )["pixel_values"].to(self.device)
        if hasattr(self.model, "get_image_features"):
            feats = self.model.get_image_features(pixel_values=pixel_values)
        else:
            feats = self.model(pixel_values=pixel_values, output_hidden_states=True)
            feats = feats.last_hidden_state
        n = feats.shape[1]
        if math.isqrt(n) ** 2 != n:
            feats = feats[:, 1:]
        return feats, _square_grid(feats.shape[1])


BACKBONES: dict[str, type[Backbone]] = {
    "dinov3": DINOv3,
    "dermlip": DermLIP,
    "siglip": SigLIP,
    "panderm": PanDerm,
    "sam3": SAM3,
    "medgemma": MedGemma,
}


def load_backbone(name: str, device: str) -> Backbone:
    """Load and freeze a backbone by its config name."""
    if name not in BACKBONES:
        raise ValueError(f"Unknown backbone {name!r}; choose from {sorted(BACKBONES)}")
    backbone = BACKBONES[name](device)
    backbone.freeze()
    return backbone


def load_panderm(repo_id: str) -> nn.Module:
    """timm ViT-B/16 with PanDerm weights.

    The checkpoint stores separate q/v biases and LayerScale gammas; q/v biases are merged into
    timm's qkv bias (k bias = 0) and gammas and the head are dropped, as in the paper runs.
    """
    import timm
    from huggingface_hub import hf_hub_download

    ckpt = torch.load(
        hf_hub_download(repo_id=repo_id, filename="open_clip_pytorch_model.bin"),
        map_location="cpu",
    )
    visual = {k.removeprefix("visual."): v for k, v in ckpt.items() if k.startswith("visual.")}
    state = {
        k: v
        for k, v in visual.items()
        if "gamma_1" not in k and "gamma_2" not in k and not k.startswith("head.")
    }
    for i in range(12):
        q, v = f"blocks.{i}.attn.q_bias", f"blocks.{i}.attn.v_bias"
        if q in visual and v in visual:
            state[f"blocks.{i}.attn.qkv.bias"] = torch.cat(
                [visual[q], torch.zeros_like(visual[q]), visual[v]]
            )
            state.pop(q, None)
            state.pop(v, None)
    model = timm.create_model(
        "vit_base_patch16_224", pretrained=False, num_classes=0, global_pool="", img_size=224
    )
    result = model.load_state_dict(state, strict=False)
    if result.missing_keys:
        print(f"PanDerm: {len(result.missing_keys)} missing keys")
    return model
