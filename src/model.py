import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

import timm


__all__ = [
    "ContrastiveBackbone",
    "ProjectionHead",
    "SegmentationHead",
    "FloodDetectionModel",
]


class ContrastiveBackbone(nn.Module):
    """
    Backbone network that extracts feature maps for both contrastive learning
    and segmentation. It wraps a model from `timm` and returns the last
    convolutional feature map.

    Parameters
    ----------
    backbone_name : str
        Name of the timm model (e.g., ``"resnet50"``).
    pretrained : bool, default True
        Whether to load ImageNet‑pretrained weights.
    out_channels : int, optional
        If provided, adds a 1×1 convolution to adapt the backbone output
        channels to ``out_channels``. Useful when the downstream heads expect a
        fixed channel dimension.
    """

    def __init__(
        self,
        backbone_name: str = "resnet50",
        pretrained: bool = True,
        out_channels: Optional[int] = None,
    ) -> None:
        super().__init__()
        # ``features_only=True`` returns a list of feature maps from each stage.
        # The last element is the highest‑level representation.
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, features_only=True
        )
        self.out_channels = out_channels

        if out_channels is not None:
            # The number of channels of the last feature map varies per backbone.
            in_ch = self.backbone.feature_info[-1]["num_chs"]
            self.adapt_conv = nn.Conv2d(in_ch, out_channels, kernel_size=1, bias=False)
        else:
            self.adapt_conv = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input image tensor of shape ``(B, C, H, W)``.

        Returns
        -------
        torch.Tensor
            Feature map of shape ``(B, C', H', W')`` where ``C'`` is either the
            original backbone output channels or ``out_channels`` if adaptation
            was requested.
        """
        feats = self.backbone(x)  # list of feature maps
        last_feat = feats[-1]  # highest‑level representation
        return self.adapt_conv(last_feat)


class ProjectionHead(nn.Module):
    """
    MLP projection head used in contrastive learning (e.g., SimCLR, MoCo).

    The architecture is ``Linear -> BN -> ReLU -> Linear``. The output is
    L2‑normalized to lie on the unit hypersphere.

    Parameters
    ----------
    in_dim : int
        Dimensionality of the input feature vector.
    hidden_dim : int
        Dimensionality of the hidden layer.
    out_dim : int
        Dimensionality of the final embedding.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 512, out_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim, bias=False),
            nn.BatchNorm1d(out_dim, affine=False),  # only for normalization
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(B, C)``.

        Returns
        -------
        torch.Tensor
            Normalized embedding of shape ``(B, out_dim)``.
        """
        x = self.net(x)
        return F.normalize(x, dim=1)


class SegmentationHead(nn.Module):
    """
    Simple segmentation head that upsamples the backbone feature map to the
    original image resolution and predicts per‑pixel class scores.

    The upsampling strategy uses a series of bilinear interpolations followed
    by 3×3 convolutions to refine the predictions.

    Parameters
    ----------
    in_channels : int
        Number of channels coming from the backbone.
    num_classes : int
        Number of segmentation classes (e.g., 1 for binary flood mask).
    hidden_channels : int, default 256
        Number of channels in the intermediate convolution layers.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        hidden_channels: int = 256,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        self.classifier = nn.Conv2d(hidden_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Feature map from the backbone of shape ``(B, C, H', W')``.
        target_size : tuple[int, int]
            Desired spatial size ``(H, W)`` of the output mask.

        Returns
        -------
        torch.Tensor
            Logits of shape ``(B, num_classes, H, W)``.
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        # Upsample to the target resolution using bilinear interpolation.
        x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        logits = self.classifier(x)
        return logits


class FloodDetectionModel(nn.Module):
    """
    End‑to‑end model for self‑supervised flood detection.

    During training the model can be used in two modes:
    * **contrastive** – returns a normalized embedding for a global average‑pooled
      backbone feature map.
    * **segmentation** – returns per‑pixel logits for flood mask prediction.

    The forward method decides the mode based on the ``mode`` argument.

    Parameters
    ----------
    backbone_name : str, default "resnet50"
        Name of the timm backbone.
    embed_dim : int, default 256
        Dimension of the contrastive embedding.
    proj_hidden_dim : int, default 512
        Hidden dimension of the projection head.
    num_classes : int, default 1
        Number of segmentation classes.
    pretrained_backbone : bool, default True
        Whether to load ImageNet weights for the backbone.
    """

    def __init__(
        self,
        backbone_name: str = "resnet50",
        embed_dim: int = 256,
        proj_hidden_dim: int = 512,
        num_classes: int = 1,
        pretrained_backbone: bool = True,
    ) -> None:
        super().__init__()
        # Use a fixed channel size after the backbone to simplify head design.
        backbone_out_ch = 512  # ResNet‑50 final stage has 2048 channels; we compress.
        self.backbone = ContrastiveBackbone(
            backbone_name=backbone_name,
            pretrained=pretrained_backbone,
            out_channels=backbone_out_ch,
        )

        # Projection head operates on a pooled vector.
        self.proj_head = ProjectionHead(
            in_dim=backbone_out_ch, hidden_dim=proj_hidden_dim, out_dim=embed_dim
        )

        self.seg_head = SegmentationHead(
            in_channels=backbone_out_ch, num_classes=num_classes, hidden_channels=256
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract backbone features without any head processing.

        Returns
        -------
        torch.Tensor
            Feature map of shape ``(B, C, H', W')``.
        """
        return self.backbone(x)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """
        Produce a contrastive embedding from an input image.

        The image is first encoded, globally average‑pooled, then passed through
        the projection head.

        Returns
        -------
        torch.Tensor
            Normalized embedding of shape ``(B, embed_dim)``.
        """
        feats = self.encode(x)  # (B, C, H', W')
        # Global average pooling
        pooled = F.adaptive_avg_pool2d(feats, 1).flatten(1)  # (B, C)
        return self.proj_head(pooled)

    def segment(self, x: torch.Tensor) -> torch.Tensor:
        """
        Produce segmentation logits for an input image.

        Returns
        -------
        torch.Tensor
            Logits of shape ``(B, num_classes, H, W)`` where ``H`` and ``W`` match
            the spatial dimensions of ``x``.
        """
        feats = self.encode(x)
        logits = self.seg_head(feats, target_size=x.shape[-2:])
        return logits

    def forward(
        self, x: torch.Tensor, mode: str = "segmentation"
    ) -> torch.Tensor:
        """
        Forward pass dispatcher.

        Parameters
        ----------
        x : torch.Tensor
            Input image tensor ``(B, C, H, W)``.
        mode : {"contrastive", "segmentation"}
            Determines which head to apply.

        Returns
        -------
        torch.Tensor
            Either a normalized embedding (contrastive) or per‑pixel logits
            (segmentation).
        """
        if mode == "contrastive":
            return self.project(x)
        if mode == "segmentation":
            return self.segment(x)
        raise ValueError(f"Unsupported mode '{mode}'. Use 'contrastive' or 'segmentation'.")


# --------------------------------------------------------------------------- #
# Simple sanity‑check that can be executed when the module is run directly.
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    torch.manual_seed(42)
    model = FloodDetectionModel()
    model.eval()

    dummy = torch.randn(2, 3, 256, 256)  # batch of two RGB tiles

    # Contrastive path
    embed = model(dummy, mode="contrastive")
    assert embed.shape == (2, 256), f"Unexpected embedding shape: {embed.shape}"
    # Embedding should be L2‑normalized
    norms = embed.norm(p=2, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), "Embeddings not normalized"

    # Segmentation path
    logits = model(dummy, mode="segmentation")
    assert logits.shape == (2, 1, 256, 256), f"Unexpected logits shape: {logits.shape}"

    print("Self‑test passed.")