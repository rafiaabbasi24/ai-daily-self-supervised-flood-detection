import os
import argparse
import logging
import datetime
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
import rasterio
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------
def setup_logging(log_dir: Path) -> None:
    """Configure root logger to output to console and a file."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"train_{datetime.datetime.now():%Y%m%d_%H%M%S}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, mode="w"),
        ],
    )


def save_checkpoint(
    state: dict,
    checkpoint_dir: Path,
    epoch: int,
    is_best: bool = False,
    max_keep: int = 5,
) -> None:
    """
    Save training state.

    Parameters
    ----------
    state: dict
        Dictionary containing model state, optimizer state, epoch, etc.
    checkpoint_dir: Path
        Directory where checkpoints are stored.
    epoch: int
        Current epoch number.
    is_best: bool, default=False
        If True, also save a copy named ``best.pth``.
    max_keep: int, default=5
        Maximum number of recent checkpoints to keep.
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoint_dir / f"epoch_{epoch:04d}.pth"
    torch.save(state, ckpt_path)
    logging.info(f"Saved checkpoint: {ckpt_path}")

    if is_best:
        best_path = checkpoint_dir / "best.pth"
        torch.save(state, best_path)
        logging.info(f"Saved best model: {best_path}")

    # Remove old checkpoints
    ckpts = sorted(checkpoint_dir.glob("epoch_*.pth"))
    if len(ckpts) > max_keep:
        for old_ckpt in ckpts[:-max_keep]:
            try:
                old_ckpt.unlink()
                logging.debug(f"Removed old checkpoint: {old_ckpt}")
            except OSError:
                pass


def load_checkpoint(
    checkpoint_path: Path, model: nn.Module, optimizer: optim.Optimizer, scheduler: Optional[optim.lr_scheduler._LRScheduler] = None
) -> Tuple[int, dict]:
    """
    Load training state from a checkpoint.

    Returns
    -------
    start_epoch: int
        Epoch to resume from (last saved epoch + 1).
    metrics: dict
        Dictionary containing any additional logged metrics.
    """
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    if scheduler is not None and "scheduler_state" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state"])

    start_epoch = checkpoint["epoch"] + 1
    metrics = checkpoint.get("metrics", {})
    logging.info(f"Loaded checkpoint '{checkpoint_path}' (epoch {checkpoint['epoch']})")
    return start_epoch, metrics


# ----------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------
class SatelliteFloodDataset(Dataset):
    """
    Dataset for loading satellite images (GeoTIFF) and returning two
    augmented views for self‑supervised contrastive learning.

    Parameters
    ----------
    file_list : List[Path]
        List of image file paths.
    transform : torchvision.transforms.Compose
        Transform applied to each view.
    """

    def __init__(self, file_list: List[Path], transform: transforms.Compose):
        self.file_list = file_list
        self.transform = transform

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path = self.file_list[idx]
        with rasterio.open(path) as src:
            # Read all bands, convert to float32 and scale to [0, 1]
            img = src.read(out_dtype="float32")
            img = np.moveaxis(img, 0, -1)  # C,H,W -> H,W,C
            img = np.clip(img / np.iinfo(src.dtypes[0]).max, 0, 1)

        img = (img * 255).astype(np.uint8)  # torchvision expects uint8
        pil_img = transforms.functional.to_pil_image(img)

        # Two independent augmentations
        view1 = self.transform(pil_img)
        view2 = self.transform(pil_img)
        return view1, view2


def build_file_list(root_dir: Path, split: str) -> List[Path]:
    """
    Scan ``root_dir`` for image files belonging to a split.

    Expected directory layout:
        root_dir/
            train/
                *.tif
            val/
                *.tif

    Parameters
    ----------
    root_dir : Path
        Base directory containing split sub‑folders.
    split : str
        Either ``'train'`` or ``'val'``.

    Returns
    -------
    List[Path]
        Sorted list of file paths.
    """
    split_dir = root_dir / split
    if not split_dir.is_dir():
        raise ValueError(f"Split directory does not exist: {split_dir}")
    return sorted(split_dir.rglob("*.tif"))


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------
def get_backbone(backbone_name: str, pretrained: bool = True) -> nn.Module:
    """
    Load a backbone from timm.

    Parameters
    ----------
    backbone_name : str
        Name of the timm model, e.g. ``'resnet50'``.
    pretrained : bool, default=True
        Whether to load ImageNet weights.

    Returns
    -------
    nn.Module
        Feature extractor without the classification head.
    """
    model = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0, global_pool="avg")
    return model


class ProjectionHead(nn.Module):
    """
    MLP projection head used in SimCLR‑style contrastive learning.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 2048, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimCLRModel(nn.Module):
    """
    Wrapper that combines a backbone and a projection head.
    """

    def __init__(self, backbone_name: str, out_dim: int = 128):
        super().__init__()
        self.backbone = get_backbone(backbone_name)
        self.projector = ProjectionHead(self.backbone.feature_info[-1]["num_chs"], out_dim=out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        proj = self.projector(features)
        return proj


# ----------------------------------------------------------------------
# Contrastive loss
# ----------------------------------------------------------------------
def nt_xent_loss(z_i: torch.Tensor, z_j: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """
    Normalised Temperature‑scaled Cross Entropy Loss (InfoNCE) used in SimCLR.

    Parameters
    ----------
    z_i, z_j : torch.Tensor
        Normalised embeddings of shape (N, D).
    temperature : float, default=0.5
        Temperature scaling factor.

    Returns
    -------
    torch.Tensor
        Scalar loss.
    """
    batch_size = z_i.shape[0]
    z = torch.cat([z_i, z_j], dim=0)  # (2N, D)
    z = F.normalize(z, dim=1)

    # Compute similarity matrix
    sim = torch.matmul(z, z.T) / temperature  # (2N, 2N)

    # Mask to remove similarity with itself
    mask = torch.eye(2 * batch_size, device=z.device).bool()
    sim.masked_fill_(mask, -float("inf"))

    # Positive pairs are (i, i+N) and (i+N, i)
    positives = torch.cat([torch.diag(sim, batch_size), torch.diag(sim, -batch_size)], dim=0)
    logits = torch.cat([positives.unsqueeze(1), sim], dim=1)  # (2N, 2N+1)

    labels = torch.zeros(2 * batch_size, dtype=torch.long, device=z.device)  # positives are first column
    loss = F.cross_entropy(logits, labels, reduction="mean")
    return loss


# ----------------------------------------------------------------------
# Training / Validation loops
# ----------------------------------------------------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler,
    epoch: int,
    log_interval: int = 10,
) -> float:
    model.train()
    total_loss = 0.0
    for batch_idx, (x1, x2) in enumerate(loader, start=1):
        x1 = x1.to(device, non_blocking=True)
        x2 = x2.to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            z1 = model(x1)
            z2 = model(x2)
            loss = nt_xent_loss(z1, z2)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        if batch_idx % log_interval == 0:
            logging.info(
                f"Epoch [{epoch}] Batch [{batch_idx}/{len(loader)}] "
                f"Loss: {loss.item():.4f}"
            )
    avg_loss = total_loss / len(loader)
    logging.info(f"Epoch [{epoch}] Training loss: {avg_loss:.4f}")
    return avg_loss


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """
    Simple validation that computes the average NT‑Xent loss on the validation set.
    """
    model.eval()
    total_loss = 0.0
    for x1, x2 in loader:
        x1 = x1.to(device, non_blocking=True)
        x2 = x2.to(device, non_blocking=True)
        z1 = model(x1)
        z2 = model(x2)
        loss = nt_xent_loss(z1, z2)
        total_loss += loss.item()
    avg_loss = total_loss / len(loader)
    logging.info(f"Validation loss: {avg_loss:.4f}")
    return avg_loss


def plot_metrics(train_losses: List[float], val_losses: List[float], out_path: Path) -> None:
    """Save a simple loss curve plot."""
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training/Validation Loss")
    plt.legend()
    plt.grid(True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    logging.info(f"Saved loss plot to {out_path}")


# ----------------------------------------------------------------------
# Argument parser
# ----------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Self‑Supervised Flood Detection Training")
    parser.add_argument("--data-dir", type=Path, required=True, help="Root directory containing 'train' and 'val' sub‑folders")
    parser.add_argument("--backbone", type=str, default="resnet50", help="timm backbone name")
    parser.add_argument("--out-dim", type=int, default=128, help="Dimensionality of projection head output")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size per GPU")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.3, help="Base learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--temperature", type=float, default=0.5, help="Contrastive loss temperature")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("./checkpoints"), help="Directory to save checkpoints")
    parser.add_argument("--resume", type=Path, default=None, help="Path to a checkpoint to resume from")
    parser.add_argument("--log-interval", type=int, default=10, help="Batches between logging")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker count")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu)")
    parser.add_argument("--plot-path", type=Path, default=Path("./metrics/loss_curve.png"), help="Where to save loss plot")
    return parser.parse_args()


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = True

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    setup_logging(Path("./logs"))

    logging.info("Starting training with configuration:")
    for k, v in vars(args).items():
        logging.info(f"  {k}: {v}")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    train_files = build_file_list(args.data_dir, "train")
    val_files = build_file_list(args.data_dir, "val")

    # Common augmentations for SimCLR
    aug = transforms.Compose(
        [
            transforms.RandomResizedCrop(size=224, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = SatelliteFloodDataset(train_files, transform=aug)
    val_dataset = SatelliteFloodDataset(val_files, transform=aug)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # ------------------------------------------------------------------
    # Model, optimizer, scheduler
    # ------------------------------------------------------------------
    model = SimCLRModel(backbone_name=args.backbone, out_dim=args.out_dim).to(device)

    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    scaler = torch.cuda.amp.GradScaler()

    start_epoch = 0
    best_val_loss = float("inf")
    train_losses: List[float] = []
    val_losses: List[float] = []

    if args.resume:
        start_epoch, _ = load_checkpoint(args.resume, model, optimizer, scheduler)
        logging.info(f"Resuming from epoch {start_epoch}")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    for epoch in range(start_epoch, args.epochs):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            scaler,
            epoch + 1,
            log_interval=args.log_interval,
        )
        val_loss = validate(model, val_loader, device)

        scheduler.step()

        train_losses.append(train_loss)
        val_losses