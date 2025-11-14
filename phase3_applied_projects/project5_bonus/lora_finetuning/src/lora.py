"""
LoRA: Low-Rank Adaptation of Large Language Models
Efficient finetuning by adapting low-rank matrices
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LoRALayer(nn.Module):
    """
    LoRA: Add low-rank update to frozen weights

    W' = W + ΔW = W + BA

    where:
    - W: Frozen pretrained weights (d_out × d_in)
    - B: Low-rank matrix (d_out × r)
    - A: Low-rank matrix (r × d_in)
    - r << min(d_out, d_in) (typically r = 4, 8, 16)

    Only B and A are trainable, reducing params by ~1000x
    """

    def __init__(self, in_features, out_features, rank=4, alpha=16, dropout=0.0):
        super().__init__()

        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Initialize A with kaiming_uniform, B with zeros
        # This ensures ΔW = BA = 0 at initialization
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x, original_output):
        """
        Args:
            x: Input tensor
            original_output: Output from frozen layer (Wx)

        Returns:
            output: original_output + low-rank adaptation
        """
        # Low-rank update: x @ A.T @ B.T
        lora_output = (self.dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return original_output + lora_output


class LoRALinear(nn.Module):
    """
    Linear layer with LoRA adaptation

    Combines frozen linear layer + LoRA update
    """

    def __init__(self, in_features, out_features, rank=4, alpha=16, dropout=0.0, bias=True):
        super().__init__()

        # Frozen linear layer
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        # LoRA adaptation
        self.lora = LoRALayer(in_features, out_features, rank, alpha, dropout)

        # Freeze original weights
        self.linear.weight.requires_grad = False
        if bias and self.linear.bias is not None:
            self.linear.bias.requires_grad = False

    def forward(self, x):
        # Original output (frozen)
        original_output = self.linear(x)

        # Add LoRA adaptation
        return self.lora(x, original_output)


class LoRAAttention(nn.Module):
    """
    Multi-head attention with LoRA applied to Q, K, V projections

    Typically, LoRA is applied to:
    - Query (Q) projection
    - Value (V) projection
    - Sometimes Key (K) projection

    Not applied to output projection (empirically works well)
    """

    def __init__(self, embed_dim, num_heads, rank=4, alpha=16, dropout=0.0):
        super().__init__()

        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Q, K, V projections with LoRA
        self.q_proj = LoRALinear(embed_dim, embed_dim, rank, alpha, dropout)
        self.k_proj = nn.Linear(embed_dim, embed_dim)  # No LoRA on K
        self.v_proj = LoRALinear(embed_dim, embed_dim, rank, alpha, dropout)

        # Output projection (no LoRA)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.scale = math.sqrt(self.head_dim)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        # Project Q, K, V
        Q = self.q_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.k_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        V = self.v_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)

        Q = Q.permute(0, 2, 1, 3)
        K = K.permute(0, 2, 1, 3)
        V = V.permute(0, 2, 1, 3)

        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, V)

        # Concatenate heads
        output = output.permute(0, 2, 1, 3).contiguous()
        output = output.reshape(batch_size, seq_len, self.embed_dim)

        # Output projection
        output = self.out_proj(output)

        return output


def apply_lora_to_model(model, rank=4, alpha=16, dropout=0.0, target_modules=None):
    """
    Apply LoRA to specific modules in a pretrained model

    Args:
        model: Pretrained model
        rank: LoRA rank
        alpha: LoRA alpha (scaling)
        dropout: Dropout probability
        target_modules: List of module names to apply LoRA (e.g., ['q_proj', 'v_proj'])

    Returns:
        model: Model with LoRA applied
    """
    if target_modules is None:
        target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj']

    for name, module in model.named_modules():
        if any(target in name for target in target_modules):
            if isinstance(module, nn.Linear):
                # Replace with LoRA linear
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]

                parent = model.get_submodule(parent_name) if parent_name else model

                lora_linear = LoRALinear(
                    module.in_features,
                    module.out_features,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout,
                    bias=module.bias is not None
                )

                # Copy weights
                lora_linear.linear.weight.data = module.weight.data.clone()
                if module.bias is not None:
                    lora_linear.linear.bias.data = module.bias.data.clone()

                setattr(parent, child_name, lora_linear)

    return model


def count_parameters(model, trainable_only=False):
    """Count model parameters"""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def print_trainable_parameters(model):
    """Print trainable parameters breakdown"""
    trainable = count_parameters(model, trainable_only=True)
    total = count_parameters(model, trainable_only=False)

    print(f"Trainable params: {trainable:,} ({100 * trainable / total:.2f}%)")
    print(f"Total params: {total:,}")


def merge_lora_weights(model):
    """
    Merge LoRA weights into base model for inference

    W' = W + BA

    After merging, no overhead during inference
    """
    for module in model.modules():
        if isinstance(module, LoRALinear):
            # Compute LoRA update
            lora_weight = (module.lora.lora_B @ module.lora.lora_A) * module.lora.scaling

            # Merge into base weights
            module.linear.weight.data += lora_weight

            # Zero out LoRA matrices to avoid double-counting
            module.lora.lora_A.data.zero_()
            module.lora.lora_B.data.zero_()

    print("LoRA weights merged into base model")


if __name__ == "__main__":
    print("=" * 60)
    print("LoRA: Low-Rank Adaptation")
    print("=" * 60)

    # Example 1: Basic LoRA layer
    print("\n1. Basic LoRA Linear Layer")
    print("-" * 60)

    in_features = 768
    out_features = 768
    rank = 8
    alpha = 16

    lora_linear = LoRALinear(in_features, out_features, rank, alpha)

    batch_size = 4
    seq_len = 10
    x = torch.randn(batch_size, seq_len, in_features)

    output = lora_linear(x)

    print(f"Input: {x.shape}")
    print(f"Output: {output.shape}")

    trainable = sum(p.numel() for p in lora_linear.parameters() if p.requires_grad)
    total = sum(p.numel() for p in lora_linear.parameters())

    print(f"Trainable params: {trainable:,} ({100 * trainable / total:.2f}%)")
    print(f"Total params: {total:,}")

    # Example 2: LoRA Attention
    print("\n2. LoRA Multi-Head Attention")
    print("-" * 60)

    embed_dim = 768
    num_heads = 12

    lora_attn = LoRAAttention(embed_dim, num_heads, rank=8, alpha=16)

    output = lora_attn(x)

    print(f"Input: {x.shape}")
    print(f"Output: {output.shape}")
    print_trainable_parameters(lora_attn)

    # Example 3: Parameter efficiency
    print("\n3. Parameter Efficiency Comparison")
    print("-" * 60)

    # Full finetuning
    full_linear = nn.Linear(768, 768)
    full_params = sum(p.numel() for p in full_linear.parameters())

    # LoRA finetuning
    lora_params = sum(p.numel() for p in lora_linear.parameters() if p.requires_grad)

    print(f"Full finetuning: {full_params:,} params")
    print(f"LoRA (r={rank}): {lora_params:,} params")
    print(f"Reduction: {full_params / lora_params:.1f}x")

    # Example 4: Different ranks
    print("\n4. Effect of Rank (r)")
    print("-" * 60)

    print(f"{'Rank':<8} {'Trainable Params':<20} {'Reduction':<12}")
    print("-" * 60)

    for r in [1, 2, 4, 8, 16, 32, 64]:
        lora_r = LoRALinear(768, 768, rank=r)
        params = sum(p.numel() for p in lora_r.parameters() if p.requires_grad)
        reduction = full_params / params

        print(f"{r:<8} {params:<20,} {reduction:<12.1f}x")

    print("\n" + "=" * 60)
    print("LoRA Complete!")
    print("=" * 60)
    print("\nKey Concepts:")
    print("✓ Low-Rank: ΔW = BA where B (d×r), A (r×d)")
    print("✓ Frozen base model: Only train B and A")
    print("✓ Efficient: ~0.1% trainable params vs full finetuning")
    print("✓ Scaling: alpha/r to control update magnitude")
    print("✓ Mergeable: W' = W + BA (no inference overhead)")
    print("✓ Multi-task: Different LoRA adapters per task")

    print("\n" + "=" * 60)
    print("Training Example:")
    print("=" * 60)
    print("""
    # 1. Load pretrained model
    model = AutoModelForCausalLM.from_pretrained("gpt2")

    # 2. Apply LoRA
    model = apply_lora_to_model(
        model,
        rank=8,
        alpha=16,
        target_modules=['q_proj', 'v_proj']
    )

    # 3. Print trainable params
    print_trainable_parameters(model)
    # Output: Trainable params: 294,912 (0.25%)

    # 4. Train (only LoRA params are updated)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=3e-4
    )

    for batch in dataloader:
        loss = model(**batch).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # 5. Merge for inference
    merge_lora_weights(model)

    # 6. Save adapter only (100x smaller than full model)
    torch.save({
        'lora_A': model.lora.lora_A,
        'lora_B': model.lora.lora_B
    }, 'lora_adapter.pt')
    """)

    print("\n" + "=" * 60)
    print("Benefits:")
    print("=" * 60)
    print("✓ Memory: Train 7B model on single GPU")
    print("✓ Storage: 100x smaller adapter files")
    print("✓ Speed: Faster training (fewer params)")
    print("✓ Modularity: Swap adapters for different tasks")
    print("✓ No degradation: Matches full finetuning performance")
