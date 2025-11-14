"""
Fairness, Bias Detection & Mitigation
Ensure ML models are fair and ethical
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import confusion_matrix
from typing import Dict, List


class FairnessMetrics:
    """
    Calculate fairness metrics for binary classification

    Key Metrics:
    - Demographic Parity: P(Y_hat=1 | A=a) should be equal across groups
    - Equalized Odds: TPR and FPR should be equal across groups
    - Equal Opportunity: TPR should be equal across groups
    """

    @staticmethod
    def demographic_parity(predictions, sensitive_attr):
        """
        Demographic Parity: P(Y_hat=1 | A=a) equal for all groups

        Returns difference between max and min positive rates
        """
        groups = np.unique(sensitive_attr)
        positive_rates = []

        for group in groups:
            mask = sensitive_attr == group
            positive_rate = predictions[mask].mean()
            positive_rates.append(positive_rate)

        return max(positive_rates) - min(positive_rates)

    @staticmethod
    def equalized_odds(y_true, y_pred, sensitive_attr):
        """
        Equalized Odds: TPR and FPR equal across groups

        Returns max difference in TPR and FPR
        """
        groups = np.unique(sensitive_attr)
        tpr_diff = []
        fpr_diff = []

        for group in groups:
            mask = sensitive_attr == group
            if mask.sum() == 0:
                continue

            y_true_group = y_true[mask]
            y_pred_group = y_pred[mask]

            tn, fp, fn, tp = confusion_matrix(
                y_true_group, y_pred_group, labels=[0, 1]
            ).ravel()

            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

            tpr_diff.append(tpr)
            fpr_diff.append(fpr)

        return {
            'tpr_diff': max(tpr_diff) - min(tpr_diff),
            'fpr_diff': max(fpr_diff) - min(fpr_diff)
        }

    @staticmethod
    def equal_opportunity(y_true, y_pred, sensitive_attr):
        """
        Equal Opportunity: TPR equal across groups

        Focuses only on positive outcomes
        """
        groups = np.unique(sensitive_attr)
        tprs = []

        for group in groups:
            mask = (sensitive_attr == group) & (y_true == 1)
            if mask.sum() == 0:
                continue

            tpr = y_pred[mask].mean()
            tprs.append(tpr)

        return max(tprs) - min(tprs)


class BiasDetector:
    """Detect bias in datasets and model predictions"""

    def __init__(self, sensitive_attributes: List[str]):
        self.sensitive_attributes = sensitive_attributes

    def analyze_dataset(self, data, labels, attributes):
        """
        Analyze dataset for class imbalance across sensitive groups

        Args:
            data: Features
            labels: Target labels
            attributes: Dict of sensitive attributes
        """
        print("=" * 60)
        print("Dataset Bias Analysis")
        print("=" * 60)

        for attr_name, attr_values in attributes.items():
            print(f"\n{attr_name}:")

            groups = np.unique(attr_values)
            for group in groups:
                mask = attr_values == group
                group_size = mask.sum()
                positive_rate = labels[mask].mean()

                print(f"  {group}: {group_size} samples ({positive_rate:.1%} positive)")

    def analyze_predictions(self, y_true, y_pred, attributes):
        """
        Analyze model predictions for fairness

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            attributes: Dict of sensitive attributes
        """
        print("\n" + "=" * 60)
        print("Prediction Bias Analysis")
        print("=" * 60)

        metrics = FairnessMetrics()

        for attr_name, attr_values in attributes.items():
            print(f"\n{attr_name}:")

            # Demographic parity
            dp = metrics.demographic_parity(y_pred, attr_values)
            print(f"  Demographic Parity Diff: {dp:.3f}")

            # Equalized odds
            eo = metrics.equalized_odds(y_true, y_pred, attr_values)
            print(f"  TPR Difference: {eo['tpr_diff']:.3f}")
            print(f"  FPR Difference: {eo['fpr_diff']:.3f}")

            # Equal opportunity
            eop = metrics.equal_opportunity(y_true, y_pred, attr_values)
            print(f"  Equal Opportunity Diff: {eop:.3f}")

            # Interpretation
            if dp < 0.1 and eo['tpr_diff'] < 0.1:
                print(f"  ✓ Model appears fair on {attr_name}")
            else:
                print(f"  ⚠ Model may be biased on {attr_name}")


class FairnessConstrainedModel(nn.Module):
    """
    Model with fairness constraints

    Techniques:
    - Adversarial debiasing
    - Reweighting
    - Fairness regularization
    """

    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()

        # Main classifier
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Adversarial network (tries to predict sensitive attribute)
        self.adversary = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x, return_hidden=False):
        # Get hidden representation
        hidden = self.classifier[:-1](x)

        # Classification
        output = self.classifier[-1](hidden)

        if return_hidden:
            return output, hidden
        return output

    def adversarial_loss(self, x, sensitive_attr):
        """
        Adversarial debiasing: Encourage hidden representation
        to not encode sensitive attribute
        """
        _, hidden = self.forward(x, return_hidden=True)
        adv_pred = self.adversary(hidden)

        # Want adversary to fail (hidden shouldn't encode sensitive attr)
        adv_loss = nn.functional.binary_cross_entropy_with_logits(
            adv_pred, sensitive_attr
        )

        return -adv_loss  # Negative because we want to maximize adversary's error


def train_with_fairness(model, train_loader, num_epochs=10, lambda_fair=0.1):
    """
    Train model with fairness constraint

    Loss = Classification Loss + λ * Fairness Loss
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(num_epochs):
        for batch in train_loader:
            x, y, sensitive_attr = batch

            # Classification loss
            output = model(x)
            clf_loss = nn.functional.binary_cross_entropy_with_logits(output, y)

            # Fairness loss (adversarial debiasing)
            fair_loss = model.adversarial_loss(x, sensitive_attr)

            # Total loss
            loss = clf_loss + lambda_fair * fair_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")


if __name__ == "__main__":
    print("=" * 60)
    print("Fairness & Bias Audit")
    print("=" * 60)

    # Simulate dataset with potential bias
    np.random.seed(42)
    N = 1000

    # Two groups: A and B
    group = np.random.choice(['A', 'B'], size=N, p=[0.7, 0.3])

    # Features
    features = np.random.randn(N, 10)

    # Biased labels (group A has higher positive rate)
    base_rate_A = 0.6
    base_rate_B = 0.3
    labels = np.array([
        np.random.rand() < (base_rate_A if g == 'A' else base_rate_B)
        for g in group
    ]).astype(float)

    print(f"\nDataset: {N} samples")
    print(f"Group A: {(group == 'A').sum()} ({base_rate_A:.0%} positive)")
    print(f"Group B: {(group == 'B').sum()} ({base_rate_B:.0%} positive)")

    # Train biased model
    print("\n" + "=" * 60)
    print("Training Baseline Model (No Fairness Constraints)")
    print("=" * 60)

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(10, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )

        def forward(self, x):
            return self.net(x)

    model = SimpleModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    X_tensor = torch.FloatTensor(features)
    y_tensor = torch.FloatTensor(labels).unsqueeze(1)

    for epoch in range(50):
        output = model(X_tensor)
        loss = nn.functional.binary_cross_entropy_with_logits(output, y_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Get predictions
    with torch.no_grad():
        y_pred_proba = torch.sigmoid(model(X_tensor)).numpy()
        y_pred = (y_pred_proba > 0.5).astype(float).flatten()

    # Fairness audit
    detector = BiasDetector(['group'])

    # Analyze dataset
    detector.analyze_dataset(
        features,
        labels,
        {'group': group}
    )

    # Analyze predictions
    detector.analyze_predictions(
        labels,
        y_pred,
        {'group': group}
    )

    # Calculate metrics
    print("\n" + "=" * 60)
    print("Fairness Metrics:")
    print("=" * 60)

    metrics = FairnessMetrics()
    group_encoded = (group == 'B').astype(float)

    dp = metrics.demographic_parity(y_pred, group)
    print(f"Demographic Parity Difference: {dp:.3f}")
    print(f"  → Should be < 0.1 for fairness")

    eo = metrics.equalized_odds(labels, y_pred, group)
    print(f"Equalized Odds (TPR diff): {eo['tpr_diff']:.3f}")
    print(f"  → Should be < 0.1 for fairness")

    print("\n" + "=" * 60)
    print("Mitigation Strategies:")
    print("=" * 60)
    print("1. Reweighting: Weight training samples to balance groups")
    print("2. Adversarial Debiasing: Prevent model from encoding bias")
    print("3. Fairness Constraints: Add fairness regularization to loss")
    print("4. Post-processing: Adjust thresholds per group")
    print("5. Data Augmentation: Balance dataset across groups")

    print("\n" + "=" * 60)
    print("Key Takeaways:")
    print("=" * 60)
    print("✓ Always audit models for bias before deployment")
    print("✓ Check multiple fairness metrics (no single metric is perfect)")
    print("✓ Consider fairness-accuracy tradeoffs")
    print("✓ Document fairness analysis in model cards")
    print("✓ Monitor fairness metrics in production")
