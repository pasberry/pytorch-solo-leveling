"""
Phase 2 Lab 4: Custom Loss Functions
Implement custom loss functions for various tasks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    Focuses training on hard examples.

    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    Args:
        alpha: Weighting factor for class balance
        gamma: Focusing parameter (higher = more focus on hard examples)
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Logits of shape (batch_size, num_classes)
            targets: Ground truth labels of shape (batch_size,)
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        return focal_loss.mean()


class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing regularization.
    Prevents overfitting by softening hard labels.

    Args:
        num_classes: Number of classes
        smoothing: Smoothing factor (0.0 = no smoothing, 0.1 = typical)
    """
    def __init__(self, num_classes: int, smoothing: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Log probabilities of shape (batch_size, num_classes)
            targets: Ground truth labels of shape (batch_size,)
        """
        log_probs = F.log_softmax(inputs, dim=1)

        # Create smoothed labels
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), self.confidence)

        loss = -torch.sum(true_dist * log_probs, dim=1)
        return loss.mean()


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for similarity learning.
    Pulls similar pairs together, pushes dissimilar pairs apart.

    Args:
        margin: Margin for dissimilar pairs
    """
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        embedding1: torch.Tensor,
        embedding2: torch.Tensor,
        label: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            embedding1: First embedding (batch_size, embed_dim)
            embedding2: Second embedding (batch_size, embed_dim)
            label: 1 if similar, 0 if dissimilar (batch_size,)
        """
        distance = F.pairwise_distance(embedding1, embedding2)

        # Loss for similar pairs: distance^2
        # Loss for dissimilar pairs: max(0, margin - distance)^2
        loss = label * distance.pow(2) + \
               (1 - label) * F.relu(self.margin - distance).pow(2)

        return loss.mean()


class TripletLoss(nn.Module):
    """
    Triplet loss for metric learning.
    Ensures: d(anchor, positive) + margin < d(anchor, negative)

    Args:
        margin: Minimum distance between positive and negative
    """
    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            anchor: Anchor embeddings (batch_size, embed_dim)
            positive: Positive embeddings (batch_size, embed_dim)
            negative: Negative embeddings (batch_size, embed_dim)
        """
        pos_dist = F.pairwise_distance(anchor, positive)
        neg_dist = F.pairwise_distance(anchor, negative)

        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()


class DiceLoss(nn.Module):
    """
    Dice loss for segmentation tasks.
    Based on F1/Dice coefficient.

    Args:
        smooth: Smoothing constant to avoid division by zero
    """
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Predicted probabilities (batch_size, num_classes, H, W)
            targets: Ground truth (batch_size, num_classes, H, W)
        """
        inputs = torch.sigmoid(inputs)

        # Flatten
        inputs_flat = inputs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (inputs_flat * targets_flat).sum()
        dice_coeff = (2.0 * intersection + self.smooth) / \
                     (inputs_flat.sum() + targets_flat.sum() + self.smooth)

        return 1.0 - dice_coeff


class HuberLoss(nn.Module):
    """
    Huber loss - combines L1 and L2 loss.
    Less sensitive to outliers than MSE.

    Args:
        delta: Threshold where loss transitions from L2 to L1
    """
    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.delta = delta

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Predictions
            targets: Ground truth
        """
        diff = torch.abs(inputs - targets)
        loss = torch.where(
            diff < self.delta,
            0.5 * diff.pow(2),
            self.delta * (diff - 0.5 * self.delta)
        )
        return loss.mean()


class InfoNCELoss(nn.Module):
    """
    InfoNCE loss for contrastive learning (used in CLIP, SimCLR).

    Args:
        temperature: Temperature parameter for softmax
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        embeddings1: torch.Tensor,
        embeddings2: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            embeddings1: First set of embeddings (batch_size, embed_dim)
            embeddings2: Second set of embeddings (batch_size, embed_dim)
        """
        # Normalize embeddings
        embeddings1 = F.normalize(embeddings1, dim=1)
        embeddings2 = F.normalize(embeddings2, dim=1)

        # Compute similarity matrix
        logits = torch.matmul(embeddings1, embeddings2.T) / self.temperature

        # Labels: positive pairs are on diagonal
        batch_size = embeddings1.size(0)
        labels = torch.arange(batch_size, device=embeddings1.device)

        # Symmetric loss
        loss_i2j = F.cross_entropy(logits, labels)
        loss_j2i = F.cross_entropy(logits.T, labels)

        return (loss_i2j + loss_j2i) / 2


def demo_focal_loss():
    """Demonstrate focal loss for imbalanced classification."""
    print("=" * 70)
    print("DEMO: Focal Loss")
    print("=" * 70)

    torch.manual_seed(42)

    # Simulate imbalanced classification
    logits = torch.randn(100, 5)
    # Most samples from class 0
    targets = torch.cat([
        torch.zeros(80, dtype=torch.long),
        torch.randint(1, 5, (20,))
    ])

    # Compare with standard cross entropy
    ce_loss = F.cross_entropy(logits, targets)
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
    fl = focal_loss(logits, targets)

    print(f"Cross Entropy Loss: {ce_loss.item():.4f}")
    print(f"Focal Loss: {fl.item():.4f}")
    print("\nFocal loss focuses more on hard examples (minority classes)")

    print("=" * 70)


def demo_contrastive_losses():
    """Demonstrate contrastive and triplet losses."""
    print("\n" + "=" * 70)
    print("DEMO: Contrastive & Triplet Losses")
    print("=" * 70)

    torch.manual_seed(42)

    # Contrastive loss
    emb1 = F.normalize(torch.randn(32, 128), dim=1)
    emb2 = F.normalize(torch.randn(32, 128), dim=1)
    labels = torch.randint(0, 2, (32,)).float()  # Similar or dissimilar

    contrastive = ContrastiveLoss(margin=1.0)
    c_loss = contrastive(emb1, emb2, labels)

    print(f"Contrastive Loss: {c_loss.item():.4f}")

    # Triplet loss
    anchor = F.normalize(torch.randn(32, 128), dim=1)
    positive = F.normalize(torch.randn(32, 128), dim=1)
    negative = F.normalize(torch.randn(32, 128), dim=1)

    triplet = TripletLoss(margin=0.3)
    t_loss = triplet(anchor, positive, negative)

    print(f"Triplet Loss: {t_loss.item():.4f}")

    print("\nThese losses learn embeddings where similar items are close")

    print("=" * 70)


def demo_infonce_loss():
    """Demonstrate InfoNCE loss."""
    print("\n" + "=" * 70)
    print("DEMO: InfoNCE Loss (CLIP-style)")
    print("=" * 70)

    torch.manual_seed(42)

    # Simulate image and text embeddings
    batch_size = 64
    image_emb = torch.randn(batch_size, 512)
    text_emb = torch.randn(batch_size, 512)

    infonce = InfoNCELoss(temperature=0.07)
    loss = infonce(image_emb, text_emb)

    print(f"InfoNCE Loss: {loss.item():.4f}")
    print("\nInfoNCE is used in CLIP for image-text matching")
    print("It treats each row/column pair as a positive example")

    print("=" * 70)


def demo_label_smoothing():
    """Demonstrate label smoothing."""
    print("\n" + "=" * 70)
    print("DEMO: Label Smoothing")
    print("=" * 70)

    torch.manual_seed(42)

    logits = torch.randn(32, 10)
    targets = torch.randint(0, 10, (32,))

    ce_loss = F.cross_entropy(logits, targets)
    ls_loss = LabelSmoothingLoss(num_classes=10, smoothing=0.1)
    smoothed_loss = ls_loss(logits, targets)

    print(f"Cross Entropy Loss: {ce_loss.item():.4f}")
    print(f"Label Smoothing Loss: {smoothed_loss.item():.4f}")
    print("\nLabel smoothing prevents overconfident predictions")

    print("=" * 70)


if __name__ == "__main__":
    demo_focal_loss()
    demo_contrastive_losses()
    demo_infonce_loss()
    demo_label_smoothing()

    print("\n✓ All custom loss demonstrations complete!")
