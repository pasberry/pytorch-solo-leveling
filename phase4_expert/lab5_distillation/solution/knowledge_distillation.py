"""
Knowledge Distillation: Compress Large Models
Train small student models to mimic large teacher models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TeacherModel(nn.Module):
    """Large teacher model"""
    def __init__(self, input_dim=100, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.net(x)


class StudentModel(nn.Module):
    """Smaller student model"""
    def __init__(self, input_dim=100, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)


def distillation_loss(student_logits, teacher_logits, true_labels, temperature=3.0, alpha=0.5):
    """
    Knowledge Distillation Loss

    L = α * L_hard + (1-α) * L_soft

    - L_hard: Standard cross-entropy with true labels
    - L_soft: KL divergence between student and teacher (softened)

    Args:
        student_logits: Student model outputs
        teacher_logits: Teacher model outputs (detached)
        true_labels: Ground truth labels
        temperature: Softening temperature (higher = softer)
        alpha: Weight for hard loss (0-1)

    Returns:
        Total distillation loss
    """
    # Hard loss: Standard cross-entropy
    hard_loss = F.cross_entropy(student_logits, true_labels)

    # Soft loss: KL divergence with temperature scaling
    soft_student = F.log_softmax(student_logits / temperature, dim=1)
    soft_teacher = F.softmax(teacher_logits / temperature, dim=1)

    soft_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (temperature ** 2)

    # Combined loss
    total_loss = alpha * hard_loss + (1 - alpha) * soft_loss

    return total_loss, hard_loss, soft_loss


def train_teacher(model, train_loader, num_epochs=10):
    """Train teacher model"""
    print("Training Teacher Model...")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0

        for batch in train_loader:
            x, y = batch
            optimizer.zero_grad()

            logits = model(x)
            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)

        accuracy = correct / total
        print(f"Epoch {epoch+1}/{num_epochs}: Loss={total_loss/len(train_loader):.4f}, Acc={accuracy:.4f}")

    return model


def train_student(student, teacher, train_loader, num_epochs=20, temperature=3.0, alpha=0.5):
    """Train student model with distillation"""
    print("\nTraining Student Model with Distillation...")
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)

    teacher.eval()  # Teacher in eval mode
    student.train()

    for epoch in range(num_epochs):
        total_loss = 0
        total_hard_loss = 0
        total_soft_loss = 0
        correct = 0
        total = 0

        for batch in train_loader:
            x, y = batch
            optimizer.zero_grad()

            # Student prediction
            student_logits = student(x)

            # Teacher prediction (no gradients)
            with torch.no_grad():
                teacher_logits = teacher(x)

            # Distillation loss
            loss, hard_loss, soft_loss = distillation_loss(
                student_logits, teacher_logits, y, temperature, alpha
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_hard_loss += hard_loss.item()
            total_soft_loss += soft_loss.item()
            correct += (student_logits.argmax(1) == y).sum().item()
            total += y.size(0)

        accuracy = correct / total
        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"Loss={total_loss/len(train_loader):.4f}, "
              f"Hard={total_hard_loss/len(train_loader):.4f}, "
              f"Soft={total_soft_loss/len(train_loader):.4f}, "
              f"Acc={accuracy:.4f}")

    return student


def evaluate(model, test_loader):
    """Evaluate model"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in test_loader:
            x, y = batch
            logits = model(x)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)

    accuracy = correct / total
    return accuracy


if __name__ == "__main__":
    print("=" * 60)
    print("Knowledge Distillation")
    print("=" * 60)

    # Create models
    teacher = TeacherModel(input_dim=100, num_classes=10)
    student = StudentModel(input_dim=100, num_classes=10)

    teacher_params = sum(p.numel() for p in teacher.parameters())
    student_params = sum(p.numel() for p in student.parameters())

    print(f"\nTeacher: {teacher_params:,} parameters")
    print(f"Student: {student_params:,} parameters")
    print(f"Compression: {teacher_params / student_params:.1f}x smaller")

    # Create synthetic dataset
    print("\n" + "=" * 60)
    print("Generating Synthetic Dataset")
    print("=" * 60)

    N_train = 10000
    N_test = 1000

    X_train = torch.randn(N_train, 100)
    y_train = torch.randint(0, 10, (N_train,))

    X_test = torch.randn(N_test, 100)
    y_test = torch.randint(0, 10, (N_test,))

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32)

    print(f"Train: {N_train} samples")
    print(f"Test: {N_test} samples")

    # Train teacher
    print("\n" + "=" * 60)
    train_teacher(teacher, train_loader, num_epochs=10)

    teacher_acc = evaluate(teacher, test_loader)
    print(f"\nTeacher Test Accuracy: {teacher_acc:.4f}")

    # Train student (baseline - without distillation)
    print("\n" + "=" * 60)
    print("Training Student (Baseline - No Distillation)")
    print("=" * 60)

    student_baseline = StudentModel(input_dim=100, num_classes=10)
    optimizer = torch.optim.Adam(student_baseline.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    student_baseline.train()
    for epoch in range(20):
        for batch in train_loader:
            x, y = batch
            optimizer.zero_grad()
            logits = student_baseline(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

    student_baseline_acc = evaluate(student_baseline, test_loader)
    print(f"Student (No Distillation) Test Accuracy: {student_baseline_acc:.4f}")

    # Train student with distillation
    print("\n" + "=" * 60)
    train_student(student, teacher, train_loader, num_epochs=20, temperature=3.0, alpha=0.5)

    student_acc = evaluate(student, test_loader)
    print(f"\nStudent (With Distillation) Test Accuracy: {student_acc:.4f}")

    # Summary
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)

    print(f"Teacher Accuracy:                 {teacher_acc:.4f}")
    print(f"Student (No Distillation):        {student_baseline_acc:.4f}")
    print(f"Student (With Distillation):      {student_acc:.4f}")
    print(f"\nImprovement from Distillation:    {(student_acc - student_baseline_acc):.4f}")
    print(f"Gap to Teacher:                   {(teacher_acc - student_acc):.4f}")

    print("\n" + "=" * 60)
    print("Key Insights:")
    print("=" * 60)
    print("✓ Student learns from teacher's soft targets (probability distributions)")
    print("✓ Soft targets contain more information than hard labels")
    print("✓ Temperature controls softness (typical: 2-5)")
    print("✓ Alpha balances hard loss and soft loss (typical: 0.3-0.7)")
    print("✓ Distillation often recovers 95%+ of teacher performance")
    print("✓ Benefits: Faster inference, smaller model, lower memory")

    print("\n" + "=" * 60)
    print("Advanced Techniques:")
    print("=" * 60)
    print("1. Feature Distillation: Match intermediate layer outputs")
    print("2. Self-Distillation: Teacher = Student (from previous epoch)")
    print("3. Progressive Distillation: Teacher → Medium → Small")
    print("4. Task-Specific Distillation: Distill for specific downstream task")
    print("5. Quantization-Aware Distillation: Distill to quantized model")
