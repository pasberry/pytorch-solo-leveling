"""
Phase 1 Lab 3: Advanced Training Techniques
Learn advanced training techniques: LR scheduling, gradient clipping, early stopping
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR
from typing import Dict, Optional, Callable
import numpy as np
from pathlib import Path


class EarlyStopping:
    """
    Early stopping to stop training when validation loss stops improving.

    Args:
        patience: Number of epochs to wait before stopping
        min_delta: Minimum change to qualify as improvement
        mode: 'min' for loss, 'max' for accuracy
    """
    def __init__(self, patience: int = 5, min_delta: float = 0.0, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.

        Args:
            score: Current metric value

        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            return False

        # Check if score improved
        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:  # mode == 'max'
            improved = score > (self.best_score + self.min_delta)

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True

        return False


class GradientClipper:
    """
    Gradient clipping utility.

    Args:
        max_norm: Maximum norm for gradient clipping
        norm_type: Type of norm (2 for L2 norm)
    """
    def __init__(self, max_norm: float = 1.0, norm_type: float = 2.0):
        self.max_norm = max_norm
        self.norm_type = norm_type

    def clip(self, parameters) -> float:
        """
        Clip gradients and return total norm.

        Args:
            parameters: Model parameters

        Returns:
            Total gradient norm before clipping
        """
        return torch.nn.utils.clip_grad_norm_(parameters, self.max_norm, self.norm_type)


class MetricsTracker:
    """Track and compute training metrics."""
    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all metrics."""
        self.losses = []
        self.predictions = []
        self.targets = []

    def update(self, loss: float, predictions: torch.Tensor, targets: torch.Tensor):
        """Update metrics with batch results."""
        self.losses.append(loss)
        self.predictions.append(predictions.detach().cpu())
        self.targets.append(targets.detach().cpu())

    def compute(self) -> Dict[str, float]:
        """Compute final metrics."""
        # Average loss
        avg_loss = np.mean(self.losses)

        # Concatenate all predictions and targets
        all_preds = torch.cat(self.predictions)
        all_targets = torch.cat(self.targets)

        # Compute accuracy
        correct = (all_preds == all_targets).sum().item()
        total = all_targets.size(0)
        accuracy = 100.0 * correct / total

        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }


class AdvancedTrainer:
    """
    Advanced trainer with multiple features:
    - Learning rate scheduling
    - Gradient clipping
    - Early stopping
    - Mixed precision training (optional)
    - TensorBoard logging (optional)
    """
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device,
        scheduler: Optional[object] = None,
        gradient_clipper: Optional[GradientClipper] = None,
        early_stopping: Optional[EarlyStopping] = None,
        use_amp: bool = False
    ):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.gradient_clipper = gradient_clipper
        self.early_stopping = early_stopping
        self.use_amp = use_amp

        # Mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler() if use_amp else None

        # History
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': [],
            'grad_norm': []
        }

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        metrics_tracker = MetricsTracker()

        total_grad_norm = 0.0
        num_batches = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass with optional mixed precision
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)

                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()

                # Gradient clipping
                if self.gradient_clipper:
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = self.gradient_clipper.clip(self.model.parameters())
                    total_grad_norm += grad_norm

                # Optimizer step with gradient scaling
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard training
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                # Backward pass
                loss.backward()

                # Gradient clipping
                if self.gradient_clipper:
                    grad_norm = self.gradient_clipper.clip(self.model.parameters())
                    total_grad_norm += grad_norm

                # Optimizer step
                self.optimizer.step()

            # Update metrics
            _, predicted = outputs.max(1)
            metrics_tracker.update(loss.item(), predicted, targets)
            num_batches += 1

        # Compute final metrics
        metrics = metrics_tracker.compute()

        # Add average gradient norm
        if self.gradient_clipper and num_batches > 0:
            metrics['grad_norm'] = total_grad_norm / num_batches

        return metrics

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        metrics_tracker = MetricsTracker()

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                # Update metrics
                _, predicted = outputs.max(1)
                metrics_tracker.update(loss.item(), predicted, targets)

        return metrics_tracker.compute()

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        checkpoint_dir: Optional[Path] = None
    ) -> Dict:
        """
        Complete training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs
            checkpoint_dir: Directory to save checkpoints

        Returns:
            Training history
        """
        if checkpoint_dir:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

        best_val_loss = float('inf')

        print("Starting advanced training...")
        print("=" * 70)
        print(f"Device: {self.device}")
        print(f"Mixed Precision: {self.use_amp}")
        print(f"Gradient Clipping: {self.gradient_clipper is not None}")
        print(f"Early Stopping: {self.early_stopping is not None}")
        print(f"LR Scheduler: {type(self.scheduler).__name__ if self.scheduler else None}")
        print("=" * 70)
        print()

        for epoch in range(num_epochs):
            # Train
            train_metrics = self.train_epoch(train_loader)

            # Validate
            val_metrics = self.validate(val_loader)

            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']

            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['lr'].append(current_lr)
            if 'grad_norm' in train_metrics:
                self.history['grad_norm'].append(train_metrics['grad_norm'])

            # Print progress
            print(f"Epoch [{epoch+1}/{num_epochs}]")
            print(f"  Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.2f}%")
            print(f"  Val Loss:   {val_metrics['loss']:.4f} | Val Acc:   {val_metrics['accuracy']:.2f}%")
            print(f"  LR: {current_lr:.6f}", end="")

            if 'grad_norm' in train_metrics:
                print(f" | Grad Norm: {train_metrics['grad_norm']:.4f}", end="")
            print()

            # Save best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                if checkpoint_dir:
                    self.save_checkpoint(checkpoint_dir / "best_model.pt", epoch, val_metrics)
                    print("  ✓ Best model saved!")

            # Learning rate scheduling
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()

                # Check if LR changed
                new_lr = self.optimizer.param_groups[0]['lr']
                if new_lr != current_lr:
                    print(f"  → Learning rate reduced: {current_lr:.6f} → {new_lr:.6f}")

            # Early stopping
            if self.early_stopping:
                if self.early_stopping(val_metrics['loss']):
                    print(f"\n  Early stopping triggered at epoch {epoch+1}")
                    break

            print()

        print("=" * 70)
        print("Training complete!")
        return self.history

    def save_checkpoint(self, path: Path, epoch: int, metrics: Dict):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'history': self.history
        }
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        torch.save(checkpoint, path)


def demo_lr_schedulers():
    """Demonstrate different learning rate schedulers."""
    print("=" * 70)
    print("DEMO: Learning Rate Schedulers")
    print("=" * 70)
    print()

    # Create synthetic data
    train_data = TensorDataset(torch.randn(800, 20), torch.randint(0, 5, (800,)))
    val_data = TensorDataset(torch.randn(200, 20), torch.randint(0, 5, (200,)))
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Test different schedulers
    schedulers = {
        'StepLR': StepLR,
        'CosineAnnealing': CosineAnnealingLR,
        'ReduceLROnPlateau': ReduceLROnPlateau
    }

    for name, SchedulerClass in schedulers.items():
        print(f"\nTesting {name}:")
        print("-" * 70)

        # Create model and optimizer
        model = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Linear(64, 5)
        )
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        # Create scheduler
        if name == 'StepLR':
            scheduler = SchedulerClass(optimizer, step_size=3, gamma=0.5)
        elif name == 'CosineAnnealing':
            scheduler = SchedulerClass(optimizer, T_max=10)
        else:  # ReduceLROnPlateau
            scheduler = SchedulerClass(optimizer, mode='min', patience=2, factor=0.5)

        # Create trainer
        trainer = AdvancedTrainer(
            model=model,
            criterion=nn.CrossEntropyLoss(),
            optimizer=optimizer,
            device=device,
            scheduler=scheduler
        )

        # Train
        history = trainer.fit(train_loader, val_loader, num_epochs=10)

        print(f"\nLearning rate progression:")
        for epoch, lr in enumerate(history['lr']):
            print(f"  Epoch {epoch+1}: {lr:.6f}")


def demo_gradient_clipping():
    """Demonstrate gradient clipping."""
    print("\n" + "=" * 70)
    print("DEMO: Gradient Clipping")
    print("=" * 70)
    print()

    # Create synthetic data
    train_data = TensorDataset(torch.randn(800, 20), torch.randint(0, 5, (800,)))
    val_data = TensorDataset(torch.randn(200, 20), torch.randint(0, 5, (200,)))
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create model
    model = nn.Sequential(
        nn.Linear(20, 64),
        nn.ReLU(),
        nn.Linear(64, 5)
    )

    # Train with gradient clipping
    trainer = AdvancedTrainer(
        model=model,
        criterion=nn.CrossEntropyLoss(),
        optimizer=optim.Adam(model.parameters(), lr=0.001),
        device=device,
        gradient_clipper=GradientClipper(max_norm=1.0)
    )

    history = trainer.fit(train_loader, val_loader, num_epochs=5)

    print("\nGradient norms:")
    for epoch, norm in enumerate(history['grad_norm']):
        print(f"  Epoch {epoch+1}: {norm:.4f}")


def demo_early_stopping():
    """Demonstrate early stopping."""
    print("\n" + "=" * 70)
    print("DEMO: Early Stopping")
    print("=" * 70)
    print()

    # Create synthetic data
    train_data = TensorDataset(torch.randn(800, 20), torch.randint(0, 5, (800,)))
    val_data = TensorDataset(torch.randn(200, 20), torch.randint(0, 5, (200,)))
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create model
    model = nn.Sequential(
        nn.Linear(20, 64),
        nn.ReLU(),
        nn.Linear(64, 5)
    )

    # Train with early stopping
    trainer = AdvancedTrainer(
        model=model,
        criterion=nn.CrossEntropyLoss(),
        optimizer=optim.Adam(model.parameters(), lr=0.001),
        device=device,
        early_stopping=EarlyStopping(patience=3, mode='min')
    )

    history = trainer.fit(train_loader, val_loader, num_epochs=20)

    print(f"\nTraining stopped after {len(history['train_loss'])} epochs")


if __name__ == "__main__":
    torch.manual_seed(42)

    demo_lr_schedulers()
    demo_gradient_clipping()
    demo_early_stopping()

    print("\n✓ All advanced training demonstrations complete!")
