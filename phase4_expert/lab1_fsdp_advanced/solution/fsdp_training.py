"""
Advanced FSDP: Fully Sharded Data Parallel Training
Train billion-parameter models efficiently across multiple GPUs
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)
import functools


class TransformerBlock(nn.Module):
    """Single Transformer block for demonstration"""

    def __init__(self, dim, num_heads, ff_dim):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, dim)
        )

    def forward(self, x):
        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)

        # Feed-forward
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)

        return x


class LargeTransformer(nn.Module):
    """Large Transformer model for FSDP training"""

    def __init__(self, vocab_size, dim=2048, num_layers=24, num_heads=16, ff_dim=8192):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            TransformerBlock(dim, num_heads, ff_dim)
            for _ in range(num_layers)
        ])
        self.output = nn.Linear(dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        return self.output(x)


def setup_distributed():
    """Initialize distributed training"""
    if not dist.is_available():
        print("Distributed training not available")
        return False

    if not dist.is_initialized():
        # For single-node multi-GPU
        dist.init_process_group(backend='nccl')

    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)

    return True


def get_fsdp_config(strategy='full_shard', use_cpu_offload=False, use_activation_checkpointing=True):
    """
    Get FSDP configuration

    Sharding Strategies:
    - FULL_SHARD (ZeRO-3): Shard parameters, gradients, and optimizer states
    - SHARD_GRAD_OP (ZeRO-2): Shard gradients and optimizer states only
    - NO_SHARD (DDP): No sharding, just data parallelism
    """

    # Sharding strategy
    if strategy == 'full_shard':
        sharding_strategy = ShardingStrategy.FULL_SHARD
    elif strategy == 'shard_grad_op':
        sharding_strategy = ShardingStrategy.SHARD_GRAD_OP
    elif strategy == 'no_shard':
        sharding_strategy = ShardingStrategy.NO_SHARD
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Mixed precision
    mixed_precision_policy = MixedPrecision(
        param_dtype=torch.float16,
        reduce_dtype=torch.float16,
        buffer_dtype=torch.float16,
    )

    # CPU offloading
    cpu_offload = CPUOffload(offload_params=True) if use_cpu_offload else None

    # Auto wrap policy (wrap layers with >100M parameters)
    auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy,
        min_num_params=100_000_000
    )

    config = {
        'sharding_strategy': sharding_strategy,
        'mixed_precision': mixed_precision_policy,
        'cpu_offload': cpu_offload,
        'auto_wrap_policy': auto_wrap_policy,
        'backward_prefetch': BackwardPrefetch.BACKWARD_PRE,
        'forward_prefetch': True,
        'limit_all_gathers': True,
    }

    return config


def apply_fsdp(model, config):
    """Apply FSDP to model"""
    model = FSDP(model, **config)
    return model


def apply_activation_checkpointing_to_model(model):
    """
    Apply activation checkpointing to save memory

    Trade-off: Slower training (recompute activations) for lower memory usage
    """
    check_fn = lambda submodule: isinstance(submodule, TransformerBlock)

    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=functools.partial(
            checkpoint_wrapper,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        ),
        check_fn=check_fn,
    )


def train_step(model, batch, optimizer, scaler=None):
    """Single training step with FSDP"""
    input_ids, labels = batch

    if scaler is not None:
        # Mixed precision training
        with torch.cuda.amp.autocast():
            outputs = model(input_ids)
            loss = nn.functional.cross_entropy(
                outputs.view(-1, outputs.size(-1)),
                labels.view(-1)
            )

        # Backward with gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        # Standard training
        outputs = model(input_ids)
        loss = nn.functional.cross_entropy(
            outputs.view(-1, outputs.size(-1)),
            labels.view(-1)
        )
        loss.backward()
        optimizer.step()

    optimizer.zero_grad()

    return loss.item()


def get_model_size(model):
    """Calculate model size in GB"""
    param_size = 0
    for param in model.parameters():
        param_size += param.numel() * param.element_size()

    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.numel() * buffer.element_size()

    size_gb = (param_size + buffer_size) / 1024**3
    return size_gb


def print_memory_stats():
    """Print GPU memory statistics"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")


if __name__ == "__main__":
    import os

    print("=" * 60)
    print("Advanced FSDP Training")
    print("=" * 60)

    # Model configuration
    vocab_size = 50000
    dim = 2048
    num_layers = 24  # ~1B parameters
    num_heads = 16
    ff_dim = 8192

    print(f"\nModel Configuration:")
    print(f"Vocabulary: {vocab_size:,}")
    print(f"Dimension: {dim}")
    print(f"Layers: {num_layers}")
    print(f"Heads: {num_heads}")
    print(f"FFN dimension: {ff_dim}")

    # Create model
    model = LargeTransformer(vocab_size, dim, num_layers, num_heads, ff_dim)

    # Calculate model size
    model_size = get_model_size(model)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel size: {model_size:.2f}GB")
    print(f"Parameters: {num_params:,} ({num_params / 1e9:.2f}B)")

    # FSDP configuration comparison
    print("\n" + "=" * 60)
    print("FSDP Sharding Strategies:")
    print("=" * 60)

    strategies = {
        'FULL_SHARD (ZeRO-3)': 'Shard params, grads, optimizer states',
        'SHARD_GRAD_OP (ZeRO-2)': 'Shard grads and optimizer states only',
        'NO_SHARD (DDP)': 'No sharding, data parallelism only',
    }

    for strategy, description in strategies.items():
        print(f"{strategy}: {description}")

    print("\n" + "=" * 60)
    print("Memory Optimization Techniques:")
    print("=" * 60)
    print("✓ Mixed Precision (FP16/BF16): 2x memory reduction")
    print("✓ Activation Checkpointing: Trade compute for memory")
    print("✓ CPU Offloading: Offload params/grads to CPU")
    print("✓ Gradient Accumulation: Simulate larger batches")
    print("✓ FSDP Sharding: Distribute model across GPUs")

    print("\n" + "=" * 60)
    print("Training Setup Example:")
    print("=" * 60)
    print("""
    # 1. Initialize distributed
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)

    # 2. Create model
    model = LargeTransformer(vocab_size=50000, dim=2048, num_layers=24)

    # 3. Apply FSDP
    fsdp_config = get_fsdp_config(strategy='full_shard')
    model = FSDP(model, **fsdp_config)

    # 4. Apply activation checkpointing
    apply_activation_checkpointing_to_model(model)

    # 5. Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # 6. Training loop
    for batch in dataloader:
        loss = train_step(model, batch, optimizer)

    # 7. Save checkpoint
    from torch.distributed.fsdp import FullStateDictConfig, StateDictType

    with FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    ):
        state_dict = model.state_dict()
        if dist.get_rank() == 0:
            torch.save(state_dict, 'model.pt')
    """)

    print("\n" + "=" * 60)
    print("Launch Commands:")
    print("=" * 60)
    print("# Single node, 4 GPUs:")
    print("torchrun --nproc_per_node=4 fsdp_training.py")
    print("\n# Multi-node (2 nodes, 4 GPUs each):")
    print("# Node 0:")
    print("torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 \\")
    print("  --master_addr=NODE0_IP --master_port=29500 fsdp_training.py")
    print("# Node 1:")
    print("torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 \\")
    print("  --master_addr=NODE0_IP --master_port=29500 fsdp_training.py")

    print("\n" + "=" * 60)
    print("Key Takeaways:")
    print("=" * 60)
    print("1. FSDP enables training models that don't fit on single GPU")
    print("2. ZeRO-3 (FULL_SHARD) provides maximum memory savings")
    print("3. Activation checkpointing trades compute for memory")
    print("4. Mixed precision (FP16) speeds up training 2-3x")
    print("5. Proper sharding strategy depends on model size and GPUs")
    print("6. Monitor memory usage to optimize configuration")
