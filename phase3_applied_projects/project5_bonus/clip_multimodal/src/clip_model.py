"""
CLIP: Contrastive Language-Image Pre-training
Build a multimodal model that learns joint embeddings for images and text
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import math


class ImageEncoder(nn.Module):
    """
    Image Encoder using Vision Transformer or ResNet

    Options:
    1. Vision Transformer (ViT)
    2. ResNet-50
    """

    def __init__(self, embed_dim=512, arch='resnet50', pretrained=True):
        super().__init__()

        if arch == 'resnet50':
            # ResNet-50 backbone
            resnet = models.resnet50(pretrained=pretrained)
            # Remove final FC layer
            self.backbone = nn.Sequential(*list(resnet.children())[:-1])
            backbone_dim = 2048

        elif arch == 'vit':
            # Vision Transformer (simplified)
            self.backbone = VisionTransformer(
                image_size=224,
                patch_size=16,
                embed_dim=768,
                num_heads=12,
                num_layers=12
            )
            backbone_dim = 768

        # Project to common embedding space
        self.projection = nn.Sequential(
            nn.Linear(backbone_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, images):
        """
        Args:
            images: (B, 3, H, W)

        Returns:
            embeddings: (B, embed_dim)
        """
        features = self.backbone(images)
        features = features.flatten(1)  # (B, backbone_dim)
        embeddings = self.projection(features)
        embeddings = F.normalize(embeddings, p=2, dim=-1)  # L2 normalize
        return embeddings


class TextEncoder(nn.Module):
    """
    Text Encoder using Transformer

    Architecture:
        Token Embedding + Positional Encoding
        → Transformer Encoder
        → [CLS] token or mean pooling
        → Projection head
    """

    def __init__(self, vocab_size, embed_dim=512, max_len=77, num_heads=8, num_layers=12):
        super().__init__()

        self.embed_dim = embed_dim

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)

        # Positional encoding
        self.positional_encoding = nn.Parameter(torch.randn(1, max_len, embed_dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, text):
        """
        Args:
            text: (B, L) - token indices

        Returns:
            embeddings: (B, embed_dim)
        """
        batch_size, seq_len = text.shape

        # Token embedding + positional encoding
        x = self.token_embedding(text)  # (B, L, D)
        x = x + self.positional_encoding[:, :seq_len, :]

        # Transformer encoding
        x = self.transformer(x)  # (B, L, D)

        # Pool: take [CLS] token (first token) or mean
        # Here we'll use the first token
        pooled = x[:, 0, :]  # (B, D)

        # Project
        embeddings = self.projection(pooled)
        embeddings = F.normalize(embeddings, p=2, dim=-1)  # L2 normalize

        return embeddings


class VisionTransformer(nn.Module):
    """
    Vision Transformer for image encoding
    """

    def __init__(self, image_size=224, patch_size=16, embed_dim=768, num_heads=12, num_layers=12):
        super().__init__()

        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        # Patch embedding
        self.patch_embedding = nn.Conv2d(
            3, embed_dim, kernel_size=patch_size, stride=patch_size
        )

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, images):
        """
        Args:
            images: (B, 3, H, W)

        Returns:
            features: (B, embed_dim)
        """
        batch_size = images.size(0)

        # Extract patches
        patches = self.patch_embedding(images)  # (B, D, H/P, W/P)
        patches = patches.flatten(2).transpose(1, 2)  # (B, N, D)

        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, patches], dim=1)  # (B, N+1, D)

        # Add positional encoding
        x = x + self.pos_encoding

        # Transformer
        x = self.transformer(x)

        # Take CLS token
        cls_output = x[:, 0, :]  # (B, D)

        return cls_output


class CLIP(nn.Module):
    """
    CLIP: Contrastive Language-Image Pre-training

    Training:
        - Batch of (image, text) pairs
        - Compute image embeddings and text embeddings
        - Maximize cosine similarity for matched pairs
        - Minimize for non-matched pairs (contrastive learning)
    """

    def __init__(
        self,
        vocab_size,
        embed_dim=512,
        image_arch='resnet50',
        text_max_len=77,
        text_num_heads=8,
        text_num_layers=12,
        temperature=0.07
    ):
        super().__init__()

        self.image_encoder = ImageEncoder(embed_dim, arch=image_arch)
        self.text_encoder = TextEncoder(
            vocab_size, embed_dim, text_max_len, text_num_heads, text_num_layers
        )

        # Learnable temperature parameter
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / temperature))

    def forward(self, images, texts):
        """
        Args:
            images: (B, 3, H, W)
            texts: (B, L)

        Returns:
            logits_per_image: (B, B)
            logits_per_text: (B, B)
        """
        # Encode images and texts
        image_embeddings = self.image_encoder(images)  # (B, D)
        text_embeddings = self.text_encoder(texts)  # (B, D)

        # Compute similarity matrix
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_embeddings @ text_embeddings.T
        logits_per_text = logits_per_image.T

        return logits_per_image, logits_per_text

    def encode_image(self, images):
        """Encode images to embeddings"""
        return self.image_encoder(images)

    def encode_text(self, texts):
        """Encode texts to embeddings"""
        return self.text_encoder(texts)


def contrastive_loss(logits_per_image, logits_per_text):
    """
    Contrastive loss (InfoNCE)

    Args:
        logits_per_image: (B, B) - similarity matrix
        logits_per_text: (B, B) - similarity matrix

    Returns:
        loss: Scalar loss
    """
    batch_size = logits_per_image.size(0)

    # Labels: diagonal elements are positive pairs
    labels = torch.arange(batch_size, device=logits_per_image.device)

    # Cross-entropy loss from both directions
    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)

    loss = (loss_i + loss_t) / 2

    return loss


def zero_shot_classification(model, images, text_prompts, tokenizer):
    """
    Zero-shot classification using CLIP

    Args:
        model: CLIP model
        images: (B, 3, H, W)
        text_prompts: List of text descriptions for each class
        tokenizer: Text tokenizer

    Returns:
        predictions: (B,) - predicted class indices
        probabilities: (B, num_classes) - softmax probabilities
    """
    model.eval()

    with torch.no_grad():
        # Encode images
        image_embeddings = model.encode_image(images)  # (B, D)

        # Encode all text prompts
        texts = tokenizer(text_prompts).to(images.device)
        text_embeddings = model.encode_text(texts)  # (num_classes, D)

        # Compute similarity
        logits = image_embeddings @ text_embeddings.T  # (B, num_classes)
        logits = logits * model.logit_scale.exp()

        # Softmax
        probabilities = F.softmax(logits, dim=-1)
        predictions = probabilities.argmax(dim=-1)

    return predictions, probabilities


def image_text_retrieval(model, images, texts, top_k=5):
    """
    Retrieve top-k most similar texts for each image

    Args:
        model: CLIP model
        images: (B_img, 3, H, W)
        texts: (B_text, L)
        top_k: Number of results to return

    Returns:
        top_indices: (B_img, top_k) - indices of top-k texts for each image
        top_scores: (B_img, top_k) - similarity scores
    """
    model.eval()

    with torch.no_grad():
        image_embeddings = model.encode_image(images)  # (B_img, D)
        text_embeddings = model.encode_text(texts)  # (B_text, D)

        # Similarity matrix
        similarity = image_embeddings @ text_embeddings.T  # (B_img, B_text)

        # Top-k
        top_scores, top_indices = similarity.topk(k=top_k, dim=-1)

    return top_indices, top_scores


# Example training loop
def train_clip():
    """Training loop skeleton for CLIP"""
    print("""
    # CLIP Training Loop

    model = CLIP(vocab_size=49408, embed_dim=512)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.2)
    scaler = torch.cuda.amp.GradScaler()  # Mixed precision

    for epoch in range(num_epochs):
        for batch in dataloader:
            images, texts = batch  # (B, 3, H, W), (B, L)

            with torch.cuda.amp.autocast():
                # Forward
                logits_per_image, logits_per_text = model(images, texts)

                # Compute loss
                loss = contrastive_loss(logits_per_image, logits_per_text)

            # Backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

    # Zero-shot evaluation
    text_prompts = ["a photo of a cat", "a photo of a dog", ...]
    predictions, probs = zero_shot_classification(model, test_images, text_prompts, tokenizer)
    accuracy = (predictions == labels).float().mean()
    print(f'Zero-shot accuracy: {accuracy:.2%}')
    """)


if __name__ == "__main__":
    print("=" * 60)
    print("CLIP: Contrastive Language-Image Pre-training")
    print("=" * 60)

    # Create model
    vocab_size = 49408  # Standard CLIP vocab size
    model = CLIP(vocab_size=vocab_size, embed_dim=512, image_arch='resnet50')

    print(f"\nModel created:")
    print(f"Image encoder: ResNet-50")
    print(f"Text encoder: Transformer")
    print(f"Embedding dimension: 512")

    # Example forward pass
    batch_size = 4
    images = torch.randn(batch_size, 3, 224, 224)
    texts = torch.randint(0, vocab_size, (batch_size, 77))

    logits_per_image, logits_per_text = model(images, texts)

    print(f"\nExample forward pass:")
    print(f"Images: {images.shape}")
    print(f"Texts: {texts.shape}")
    print(f"Logits per image: {logits_per_image.shape}")
    print(f"Logits per text: {logits_per_text.shape}")

    # Compute loss
    loss = contrastive_loss(logits_per_image, logits_per_text)
    print(f"\nContrastive loss: {loss.item():.4f}")

    # Example embeddings
    image_emb = model.encode_image(images)
    text_emb = model.encode_text(texts)

    print(f"\nEmbeddings:")
    print(f"Image embeddings: {image_emb.shape}")
    print(f"Text embeddings: {text_emb.shape}")
    print(f"Cosine similarity: {(image_emb[0] @ text_emb[0]).item():.4f}")

    print("\n" + "=" * 60)
    print("CLIP Model Ready!")
    print("=" * 60)
    print("\nKey Features:")
    print("✓ Joint image-text embedding space")
    print("✓ Contrastive learning (InfoNCE loss)")
    print("✓ Zero-shot classification")
    print("✓ Image-text retrieval")
    print("✓ Transfer learning for downstream tasks")

    # Show training skeleton
    print("\n" + "=" * 60)
    print("Training Setup:")
    print("=" * 60)
    train_clip()
