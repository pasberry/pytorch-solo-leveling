# Lab 3: Model Fairness, Bias Detection & Mitigation ⚖️

> **Time:** 2-3 hours
> **Difficulty:** Expert
> **Goal:** Master fairness auditing and debiasing techniques to build ethical, responsible AI systems

---

## 📖 Why This Lab Matters

Your model achieves 95% accuracy. It's deployed to production. Millions of users rely on it for critical decisions. **Then disaster strikes:**

**Real-world fairness failures:**
- **Amazon's hiring AI (2018):** Penalized resumes containing "women's" → Scrapped after bias discovered
- **COMPAS recidivism (2016):** 45% false positive rate for Black defendants vs 23% for white defendants
- **Face recognition (2018):** Error rate 34% for dark-skinned women vs <1% for light-skinned men (MIT study)
- **Ad delivery (2019):** Google showed high-paying job ads more to men than women
- **Credit scoring:** Lower credit limits for same creditworthiness based on zip code (proxy for race)

**The stakes:**
- **Legal:** GDPR, EU AI Act, US Fair Credit Reporting Act
- **Reputational:** PR disasters, loss of user trust, boycotts
- **Financial:** Lawsuits, fines, lost revenue
- **Ethical:** Perpetuating historical discrimination, harming marginalized groups

**The reality:**
```python
# Your model might be biased even with 95% accuracy
overall_accuracy = 0.95

# Hidden bias:
accuracy_group_A = 0.98  # Privileged group
accuracy_group_B = 0.82  # Marginalized group

# 16 percentage point disparity!
# Legal threshold often: <10% disparity
```

**This lab teaches you:**
- How to **detect** bias in ML models (fairness metrics)
- How to **measure** disparate impact (demographic parity, equalized odds)
- How to **mitigate** bias (pre-processing, in-processing, post-processing)
- How to **audit** models for production (model cards, documentation)

**Master fairness, and you build AI that works for everyone—not just the majority.**

---

## 🧠 The Big Picture: Why ML Models Are Biased

### Sources of Bias in ML

**1. Historical Bias (in data collection)**
```
Problem: Training data reflects historical discrimination

Example: Hiring data from 1990-2020
  - 90% of engineers were men (historical hiring bias)
  - Model learns: "engineer" → "male" (proxy features)
  - Result: Discriminates against women applicants

Real case: Amazon scrapped hiring AI for this reason
```

**2. Representation Bias (sampling)**
```
Problem: Some groups underrepresented in training data

Example: Face recognition trained on:
  - 80% light-skinned faces
  - 20% dark-skinned faces

Result:
  - High accuracy for light skin (abundant data)
  - Low accuracy for dark skin (scarce data)

Real case: MIT Gender Shades study (2018)
```

**3. Measurement Bias (label quality)**
```
Problem: Labels systematically noisier for some groups

Example: Medical diagnosis
  - Doctors spend more time with English speakers
  - Less accurate labels for non-English speakers
  - Model learns noise → worse performance

Result: Health disparities amplified
```

**4. Aggregation Bias (one-size-fits-all)**
```
Problem: Single model for heterogeneous populations

Example: Diabetes risk prediction
  - Model trained on population average
  - Different risk factors by ethnicity (genetics, diet)
  - One model underperforms for minority groups

Solution: Group-specific models or feature engineering
```

**5. Evaluation Bias (metric choice)**
```
Problem: Overall accuracy hides group disparities

Example:
  Overall accuracy: 90%
  Majority group (90% of data): 95% accuracy
  Minority group (10% of data): 50% accuracy

Single metric (90%) looks good, but deeply unfair!
```

### The Fairness-Accuracy Tradeoff

**Uncomfortable truth:** Perfect fairness often requires sacrificing some accuracy.

```
Unconstrained model:
  Overall accuracy: 85%
  Group A accuracy: 90%
  Group B accuracy: 70%
  Disparity: 20 percentage points (unfair!)

Fair model (equalized odds):
  Overall accuracy: 82% (↓ 3%)
  Group A accuracy: 82%
  Group B accuracy: 82%
  Disparity: 0 percentage points (fair!)

Cost: -3% overall accuracy
Benefit: Equal treatment for all groups
```

**Production question:** How much accuracy is your organization willing to sacrifice for fairness?

---

## 🔬 Deep Dive: Fairness Metrics

### Protected Attributes

**Definition:** Characteristics that should NOT influence predictions (legally or ethically).

**Common protected attributes:**
- Race / ethnicity
- Gender / sex
- Age (especially >40 in US)
- Religion
- National origin
- Disability status
- Sexual orientation

**Legal context:**
- **US:** Title VII (employment), Fair Housing Act, ECOA (credit)
- **EU:** GDPR Article 9 (special category data)
- **EU AI Act:** High-risk AI systems must prove fairness

### Fairness Metric 1: Demographic Parity

**Definition:** Positive prediction rate should be equal across groups.

**Mathematical formulation:**
```
P(Ŷ=1 | A=0) = P(Ŷ=1 | A=1)

Where:
  Ŷ = prediction
  A = protected attribute (e.g., race: 0=white, 1=Black)

In words: "The model predicts positive at the same rate for all groups"
```

**Example: Loan approval**
```
Total applicants: 1000
  - Group A (500): 400 approved → 80% approval rate
  - Group B (500): 200 approved → 40% approval rate

Demographic parity violation: 80% ≠ 40% (40 point disparity!)

Legal threshold (80% rule):
  min_rate / max_rate ≥ 0.8
  40% / 80% = 0.5 < 0.8  → FAIL (likely illegal discrimination)
```

**Implementation:**
```python
def demographic_parity_difference(y_pred, sensitive_attr):
    """Calculate demographic parity difference."""
    groups = np.unique(sensitive_attr)
    rates = []

    for group in groups:
        mask = sensitive_attr == group
        positive_rate = y_pred[mask].mean()
        rates.append(positive_rate)

    # Difference between max and min rates
    return max(rates) - min(rates)

# Usage
dpd = demographic_parity_difference(predictions, race)
print(f"Demographic parity difference: {dpd:.2%}")
# Target: <10% for fairness
```

**When to use:**
- Hiring decisions (equal opportunity)
- Ad delivery (equal exposure)
- College admissions (access)

**Limitation:** Ignores actual qualification/merit. May be unfair if groups have different true base rates.

### Fairness Metric 2: Equalized Odds

**Definition:** True positive rate AND false positive rate should be equal across groups.

**Mathematical formulation:**
```
P(Ŷ=1 | Y=1, A=0) = P(Ŷ=1 | Y=1, A=1)  (Equal TPR)
P(Ŷ=1 | Y=0, A=0) = P(Ŷ=1 | Y=0, A=1)  (Equal FPR)

Where:
  Y = true label
  Ŷ = prediction
  A = protected attribute

In words: "The model has the same error rates for all groups"
```

**Example: Criminal recidivism prediction**
```
Group A (white defendants):
  True positives: 90/100 (90% TPR)
  False positives: 23/100 (23% FPR)

Group B (Black defendants):
  True positives: 70/100 (70% TPR)  ← Lower! Misses more reoffenders
  False positives: 45/100 (45% FPR) ← Higher! False accusations

Equalized odds violation:
  TPR: 90% ≠ 70% (20 point gap)
  FPR: 23% ≠ 45% (22 point gap)

Impact: Black defendants face both:
  - Less accurate identification of actual risk
  - More false accusations of risk
```

**Implementation:**
```python
def equalized_odds_difference(y_true, y_pred, sensitive_attr):
    """Calculate equalized odds difference."""
    groups = np.unique(sensitive_attr)
    tpr_diff = 0
    fpr_diff = 0

    tprs = []
    fprs = []

    for group in groups:
        mask = sensitive_attr == group

        # True positive rate
        tp = ((y_true[mask] == 1) & (y_pred[mask] == 1)).sum()
        fn = ((y_true[mask] == 1) & (y_pred[mask] == 0)).sum()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        tprs.append(tpr)

        # False positive rate
        fp = ((y_true[mask] == 0) & (y_pred[mask] == 1)).sum()
        tn = ((y_true[mask] == 0) & (y_pred[mask] == 0)).sum()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fprs.append(fpr)

    return max(tprs) - min(tprs), max(fprs) - min(fprs)

# Usage
tpr_diff, fpr_diff = equalized_odds_difference(y_true, y_pred, race)
print(f"TPR difference: {tpr_diff:.2%}")
print(f"FPR difference: {fpr_diff:.2%}")
# Target: Both <10%
```

**When to use:**
- High-stakes decisions (criminal justice, medical diagnosis)
- When both false positives AND false negatives have serious consequences
- When you have ground truth labels

**Advantage:** Captures error rates, not just prediction rates. More robust.

### Fairness Metric 3: Equal Opportunity

**Definition:** True positive rate should be equal across groups (relaxed version of equalized odds).

**Mathematical formulation:**
```
P(Ŷ=1 | Y=1, A=0) = P(Ŷ=1 | Y=1, A=1)

In words: "Among qualified candidates, selection rate is equal"
```

**Example: College admissions**
```
Group A: 80% of qualified students admitted (TPR)
Group B: 50% of qualified students admitted (TPR)

Equal opportunity violation: 80% ≠ 50%

Impact: Qualified students from Group B systematically rejected
```

**When to use:**
- When false negatives are more important than false positives
- Opportunities (admissions, hiring, loans)

**Advantage:** Simpler than equalized odds, focuses on opportunity for qualified individuals.

### Fairness Metric 4: Calibration

**Definition:** Among individuals with prediction score p, p% should actually be positive (across all groups).

**Mathematical formulation:**
```
P(Y=1 | Ŷ=p, A=0) = P(Y=1 | Ŷ=p, A=1) = p

In words: "The model's confidence scores mean the same thing for all groups"
```

**Example: Loan default risk prediction**
```
Model predicts 30% default risk for both:
  - Group A applicant
  - Group B applicant

Calibration requires:
  - 30% of Group A with this score actually default
  - 30% of Group B with this score actually default

Violation example:
  - Group A: 30% of those with score 0.3 default ✓
  - Group B: 50% of those with score 0.3 default ✗

Impact: Model under-estimates risk for Group B!
```

**Implementation:**
```python
def calibration_curve_by_group(y_true, y_prob, sensitive_attr, n_bins=10):
    """Plot calibration curves for each group."""
    groups = np.unique(sensitive_attr)

    for group in groups:
        mask = sensitive_attr == group

        # Bin predictions
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(y_prob[mask], bins) - 1

        # Calculate calibration
        calibration = []
        for i in range(n_bins):
            bin_mask = bin_indices == i
            if bin_mask.sum() > 0:
                mean_pred = y_prob[mask][bin_mask].mean()
                mean_true = y_true[mask][bin_mask].mean()
                calibration.append((mean_pred, mean_true))

        # Plot
        pred_vals, true_vals = zip(*calibration)
        plt.plot(pred_vals, true_vals, label=f'Group {group}')

    plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    plt.xlabel('Predicted probability')
    plt.ylabel('True probability')
    plt.legend()
    plt.show()
```

**When to use:**
- Risk scoring (credit, insurance, recidivism)
- When prediction scores are used for decision-making
- Medical diagnosis (probabilities must be trustworthy)

**Incompatibility theorem:** You cannot achieve calibration AND equalized odds simultaneously (except in trivial cases)!

### Impossibility Results

**Theorem (Kleinberg et al., 2016):** Except in degenerate cases, you cannot satisfy all of:
1. Demographic parity
2. Equalized odds
3. Calibration

**Implication:** Must choose which fairness metric matters most for your application!

**Practical guidance:**
| Use Case | Primary Metric | Reason |
|----------|----------------|--------|
| Hiring | Demographic parity | Equal opportunity by law |
| Criminal justice | Equalized odds | Both false positives and negatives matter |
| Medical diagnosis | Calibration | Risk scores must be trustworthy |
| Ad delivery | Demographic parity | Equal exposure |
| Credit scoring | Equalized odds + Calibration | Regulatory requirements |

---

## 🔧 Debiasing Techniques

### Pre-Processing: Fix the Data

**Idea:** Remove bias from training data before training.

**Technique 1: Reweighting**
```python
# Give higher weight to underrepresented groups
def compute_sample_weights(y, sensitive_attr):
    """Reweight samples to balance group representation."""
    weights = np.ones(len(y))

    for group in np.unique(sensitive_attr):
        for label in [0, 1]:
            mask = (sensitive_attr == group) & (y == label)
            count = mask.sum()
            if count > 0:
                # Weight inversely proportional to group-label frequency
                weights[mask] = 1.0 / count

    # Normalize
    weights /= weights.mean()
    return weights

# Use in training
sample_weights = compute_sample_weights(y_train, sensitive_train)
model.fit(X_train, y_train, sample_weight=sample_weights)
```

**Technique 2: Resampling**
```python
# Oversample minority groups, undersample majority
from imblearn.over_sampling import SMOTE

def fair_resampling(X, y, sensitive_attr):
    """Balance dataset across sensitive groups."""
    # For each sensitive group, balance positive/negative classes
    X_resampled = []
    y_resampled = []
    sensitive_resampled = []

    for group in np.unique(sensitive_attr):
        mask = sensitive_attr == group
        X_group = X[mask]
        y_group = y[mask]

        # Apply SMOTE to balance within group
        smote = SMOTE(random_state=42)
        X_balanced, y_balanced = smote.fit_resample(X_group, y_group)

        X_resampled.append(X_balanced)
        y_resampled.append(y_balanced)
        sensitive_resampled.append(np.full(len(y_balanced), group))

    return (np.vstack(X_resampled),
            np.concatenate(y_resampled),
            np.concatenate(sensitive_resampled))
```

**Technique 3: Fair Representation Learning**
```python
# Learn features that are independent of sensitive attributes
class FairAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )
        self.sensitive_predictor = nn.Linear(latent_dim, 1)  # Adversary

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        sensitive_pred = self.sensitive_predictor(z)
        return x_recon, z, sensitive_pred

# Training: Maximize reconstruction, minimize sensitive prediction
def train_fair_ae(model, X, y, sensitive_attr):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(100):
        x_recon, z, sensitive_pred = model(X)

        # Reconstruction loss
        recon_loss = F.mse_loss(x_recon, X)

        # Adversarial loss (make sensitive attribute unpredictable)
        adv_loss = F.binary_cross_entropy_with_logits(
            sensitive_pred.squeeze(),
            sensitive_attr.float()
        )

        # Total: Minimize reconstruction, maximize adversarial error
        loss = recon_loss - 0.5 * adv_loss  # Negative adv_loss!

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Use learned fair representation (z) for downstream tasks
    return model.encoder(X).detach()
```

### In-Processing: Constrain the Model

**Idea:** Add fairness constraints during training.

**Technique 1: Fairness Regularization**
```python
# Add fairness penalty to loss function
def fairness_regularized_loss(y_pred, y_true, sensitive_attr, lambda_fair=0.1):
    """Loss = task loss + fairness penalty."""
    # Task loss (binary cross-entropy)
    task_loss = F.binary_cross_entropy(y_pred, y_true)

    # Fairness penalty (demographic parity)
    groups = torch.unique(sensitive_attr)
    group_means = []
    for group in groups:
        mask = sensitive_attr == group
        group_means.append(y_pred[mask].mean())

    # Penalty: Variance of group means (0 if all equal)
    fairness_penalty = torch.var(torch.stack(group_means))

    # Total loss
    total_loss = task_loss + lambda_fair * fairness_penalty
    return total_loss

# Training
for batch in dataloader:
    X, y, sensitive = batch
    y_pred = model(X)
    loss = fairness_regularized_loss(y_pred, y, sensitive, lambda_fair=0.5)
    loss.backward()
    optimizer.step()
```

**Technique 2: Adversarial Debiasing**
```python
# Train predictor + adversary simultaneously
class FairClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.task_predictor = nn.Linear(32, 1)
        self.adversary = nn.Linear(32, 1)  # Predicts sensitive attribute

    def forward(self, x):
        features = self.feature_extractor(x)
        task_pred = self.task_predictor(features)
        adv_pred = self.adversary(features)
        return task_pred, adv_pred

# Training: Predictor wants to hide sensitive info from adversary
def train_adversarial(model, X, y, sensitive):
    optimizer_task = optim.Adam(
        list(model.feature_extractor.parameters()) +
        list(model.task_predictor.parameters()),
        lr=1e-3
    )
    optimizer_adv = optim.Adam(model.adversary.parameters(), lr=1e-3)

    for epoch in range(100):
        # Train adversary (maximize sensitive prediction)
        task_pred, adv_pred = model(X)
        adv_loss = F.binary_cross_entropy_with_logits(adv_pred.squeeze(), sensitive.float())
        optimizer_adv.zero_grad()
        adv_loss.backward(retain_graph=True)
        optimizer_adv.step()

        # Train task predictor (minimize task loss, fool adversary)
        task_pred, adv_pred = model(X)
        task_loss = F.binary_cross_entropy_with_logits(task_pred.squeeze(), y.float())
        adv_loss = F.binary_cross_entropy_with_logits(adv_pred.squeeze(), sensitive.float())

        # Total: Good task performance + bad adversary performance
        total_loss = task_loss - 0.5 * adv_loss  # Adversarial term!

        optimizer_task.zero_grad()
        total_loss.backward()
        optimizer_task.step()
```

**Technique 3: Constrained Optimization**
```python
# Enforce fairness as hard constraint
from scipy.optimize import minimize

def train_with_fairness_constraint(X, y, sensitive, max_disparity=0.1):
    """Train with demographic parity constraint."""
    # Define objective: Maximize accuracy
    def objective(weights):
        # Logistic regression predictions
        y_pred = 1 / (1 + np.exp(-X @ weights))
        # Negative log-likelihood
        return -np.mean(y * np.log(y_pred + 1e-8) +
                        (1 - y) * np.log(1 - y_pred + 1e-8))

    # Define fairness constraint
    def fairness_constraint(weights):
        y_pred = 1 / (1 + np.exp(-X @ weights))
        # Demographic parity difference
        group_rates = []
        for group in np.unique(sensitive):
            mask = sensitive == group
            group_rates.append(y_pred[mask].mean())
        disparity = max(group_rates) - min(group_rates)
        return max_disparity - disparity  # Constraint: disparity <= max_disparity

    # Optimize with constraint
    constraints = {'type': 'ineq', 'fun': fairness_constraint}
    result = minimize(
        objective,
        x0=np.zeros(X.shape[1]),
        constraints=constraints,
        method='SLSQP'
    )

    return result.x
```

### Post-Processing: Adjust Predictions

**Idea:** Modify predictions after training to satisfy fairness.

**Technique 1: Threshold Optimization**
```python
# Use different thresholds for different groups
def find_fair_thresholds(y_pred_proba, y_true, sensitive, target_metric='equalized_odds'):
    """Find group-specific thresholds for fairness."""
    groups = np.unique(sensitive)
    thresholds = {}

    if target_metric == 'demographic_parity':
        # Set thresholds to equalize positive prediction rates
        target_rate = y_pred_proba.mean()  # Overall rate

        for group in groups:
            mask = sensitive == group
            # Find threshold that gives target_rate for this group
            threshold = np.quantile(y_pred_proba[mask], 1 - target_rate)
            thresholds[group] = threshold

    elif target_metric == 'equalized_odds':
        # Set thresholds to equalize TPR and FPR
        from sklearn.metrics import roc_curve

        # Find threshold that balances TPR/FPR across groups
        # (Simplified: use threshold that equalizes TPR)
        tprs_at_thresholds = []

        for threshold in np.linspace(0, 1, 100):
            tprs = []
            for group in groups:
                mask = sensitive == group
                y_pred = (y_pred_proba[mask] >= threshold).astype(int)
                tp = ((y_true[mask] == 1) & (y_pred == 1)).sum()
                fn = ((y_true[mask] == 1) & (y_pred == 0)).sum()
                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                tprs.append(tpr)

            tpr_disparity = max(tprs) - min(tprs)
            tprs_at_thresholds.append((threshold, tpr_disparity))

        # Choose threshold with minimal TPR disparity
        best_threshold, _ = min(tprs_at_thresholds, key=lambda x: x[1])
        for group in groups:
            thresholds[group] = best_threshold  # Same threshold for all

    return thresholds

# Apply thresholds
def apply_fair_thresholds(y_pred_proba, sensitive, thresholds):
    y_pred = np.zeros(len(y_pred_proba))
    for group, threshold in thresholds.items():
        mask = sensitive == group
        y_pred[mask] = (y_pred_proba[mask] >= threshold).astype(int)
    return y_pred
```

**Technique 2: Reject Option Classification**
```python
# For predictions near decision boundary, flip to satisfy fairness
def reject_option_classification(y_pred_proba, sensitive, margin=0.1):
    """Flip predictions in uncertain region to improve fairness."""
    y_pred = (y_pred_proba >= 0.5).astype(int)

    # Identify "reject region" (near decision boundary)
    uncertain = (y_pred_proba >= 0.5 - margin) & (y_pred_proba <= 0.5 + margin)

    # Calculate group positive rates
    group_rates = []
    for group in np.unique(sensitive):
        mask = sensitive == group
        group_rates.append((group, y_pred[mask].mean()))

    # Identify privileged (highest rate) and unprivileged (lowest rate) groups
    privileged_group = max(group_rates, key=lambda x: x[1])[0]
    unprivileged_group = min(group_rates, key=lambda x: x[1])[0]

    # In uncertain region:
    # - Flip privileged group predictions 1 → 0
    # - Flip unprivileged group predictions 0 → 1
    mask_priv = uncertain & (sensitive == privileged_group) & (y_pred == 1)
    mask_unpriv = uncertain & (sensitive == unprivileged_group) & (y_pred == 0)

    y_pred[mask_priv] = 0
    y_pred[mask_unpriv] = 1

    return y_pred
```

**Comparison:**

| Approach | Pros | Cons | Use When |
|----------|------|------|----------|
| Pre-processing | Simple, model-agnostic | May lose information | Limited control over training |
| In-processing | Best performance | Requires custom training | You control training pipeline |
| Post-processing | Easy retrofit, model-agnostic | Suboptimal performance | Model already deployed |

---

## 📊 Mathematical Foundations

### Fairness Metrics Formalization

**Notation:**
- $X$: Features
- $Y$: True label
- $\hat{Y}$: Predicted label
- $A$: Sensitive attribute
- $S = f(X)$: Model score

**Independence (Demographic Parity):**
$$\hat{Y} \perp A$$
$$P(\hat{Y}=1|A=0) = P(\hat{Y}=1|A=1)$$

**Separation (Equalized Odds):**
$$\hat{Y} \perp A | Y$$
$$P(\hat{Y}=1|Y=y, A=0) = P(\hat{Y}=1|Y=y, A=1) \quad \forall y \in \{0,1\}$$

**Sufficiency (Calibration):**
$$Y \perp A | \hat{Y}$$
$$P(Y=1|\hat{Y}=\hat{y}, A=0) = P(Y=1|\hat{Y}=\hat{y}, A=1) \quad \forall \hat{y}$$

### Impossibility Theorem (Simplified Proof Sketch)

**Claim:** Cannot achieve Independence, Separation, and Sufficiency simultaneously.

**Proof sketch:**
Assume we have:
1. Independence: $P(\hat{Y}=1|A=0) = P(\hat{Y}=1|A=1)$
2. Separation: $P(\hat{Y}=1|Y=1,A=0) = P(\hat{Y}=1|Y=1,A=1)$

Now, by Bayes' rule:
$$P(\hat{Y}=1|A=a) = P(\hat{Y}=1|Y=1,A=a) \cdot P(Y=1|A=a) + P(\hat{Y}=1|Y=0,A=a) \cdot P(Y=0|A=a)$$

If base rates differ ($P(Y=1|A=0) \neq P(Y=1|A=1)$) and we want both Independence and Separation to hold, we need:
$$P(\hat{Y}=1|Y=0,A=0) \neq P(\hat{Y}=1|Y=0,A=1)$$

But this violates Separation for $Y=0$! Contradiction. ∎

**Implication:** Must choose which fairness definition to prioritize.

---

## 🏭 Production: Fairness Auditing at Scale

### Model Cards (Google, 2019)

**Purpose:** Transparent documentation of model performance, including fairness metrics.

**Required components:**
1. **Model details:** Architecture, training data, intended use
2. **Intended use:** Who should use it, for what purpose
3. **Factors:** Demographics, environment, instrumentation
4. **Metrics:** Performance metrics per demographic group
5. **Evaluation data:** Test set composition
6. **Training data:** Composition, collection methods, preprocessing
7. **Quantitative analysis:** Disaggregated metrics by subgroup
8. **Ethical considerations:** Fairness, privacy, security
9. **Caveats and recommendations:** Known failure modes, suggestions

**Example template:**
```markdown
# Model Card: Credit Risk Predictor v2.1

## Model Details
- **Model type:** Gradient Boosted Trees (XGBoost)
- **Input:** Credit history (36 features)
- **Output:** Default risk score (0-1)
- **Training data:** 5M credit applications (2018-2022)

## Intended Use
- **Primary use:** Pre-screening for loan applications
- **Users:** Credit analysts at XYZ Bank
- **Out-of-scope:** Not for use in hiring or insurance decisions

## Metrics
| Metric | Overall | Male | Female | White | Black | Hispanic | Asian |
|--------|---------|------|--------|-------|-------|----------|-------|
| AUC | 0.82 | 0.83 | 0.81 | 0.84 | 0.78 | 0.80 | 0.83 |
| Demographic parity | - | 0.65 | 0.58 | 0.68 | 0.52 | 0.55 | 0.70 |
| Equalized odds (TPR) | - | 0.72 | 0.68 | 0.74 | 0.66 | 0.69 | 0.73 |

## Fairness Analysis
- **Demographic parity violation:** Black applicants approved at 52% rate vs 68% for white applicants (16 point gap, above 10% threshold)
- **Mitigation:** Applied threshold adjustment to reduce gap to 8%
- **Residual disparities:** Asian and Hispanic groups within acceptable range

## Ethical Considerations
- **Known biases:** Historical lending discrimination reflected in training data
- **Mitigation efforts:** Reweighting, fairness regularization (λ=0.3)
- **Monitoring:** Monthly fairness audits, alert if disparity >12%

## Caveats
- **Do not use** for applicants with <6 months credit history (insufficient data)
- **Lower accuracy** for recent immigrants (sparse credit history)
- **Recalibrate** annually due to economic condition changes
```

### Continuous Fairness Monitoring

**Production system:**
```python
class FairnessMonitor:
    def __init__(self, protected_attributes, thresholds):
        self.protected_attributes = protected_attributes
        self.thresholds = thresholds  # {'demographic_parity': 0.1, 'equalized_odds': 0.1}
        self.alerts = []

    def check_fairness(self, y_true, y_pred, metadata):
        """Run fairness checks and trigger alerts if violations detected."""
        violations = []

        for attr in self.protected_attributes:
            sensitive = metadata[attr]

            # Check demographic parity
            dpd = demographic_parity_difference(y_pred, sensitive)
            if dpd > self.thresholds['demographic_parity']:
                violations.append({
                    'metric': 'demographic_parity',
                    'attribute': attr,
                    'value': dpd,
                    'threshold': self.thresholds['demographic_parity']
                })

            # Check equalized odds
            tpr_diff, fpr_diff = equalized_odds_difference(y_true, y_pred, sensitive)
            if tpr_diff > self.thresholds['equalized_odds']:
                violations.append({
                    'metric': 'equalized_odds_tpr',
                    'attribute': attr,
                    'value': tpr_diff,
                    'threshold': self.thresholds['equalized_odds']
                })

        # Trigger alerts
        if violations:
            self.trigger_alert(violations)

        return violations

    def trigger_alert(self, violations):
        """Send alerts to MLOps team."""
        alert_message = f"⚠️ Fairness violation detected!\n"
        for v in violations:
            alert_message += f"  - {v['metric']} for {v['attribute']}: {v['value']:.2%} (threshold: {v['threshold']:.2%})\n"

        # Send to monitoring system (e.g., PagerDuty, Slack)
        print(alert_message)
        # send_to_slack(alert_message)
        # create_pagerduty_incident(violations)

# Usage in production
monitor = FairnessMonitor(
    protected_attributes=['gender', 'race', 'age_group'],
    thresholds={'demographic_parity': 0.10, 'equalized_odds': 0.10}
)

# Check every batch of predictions
for batch in prediction_stream:
    violations = monitor.check_fairness(
        y_true=batch['labels'],
        y_pred=batch['predictions'],
        metadata=batch['metadata']
    )

    if violations:
        # Log to database
        log_fairness_violation(violations)
        # Potentially roll back model deployment
```

---

## 🎯 Learning Objectives

By the end of this lab, you will:

**Theory:**
- [ ] Understand sources of bias in ML (historical, representation, measurement, aggregation)
- [ ] Explain fairness metrics (demographic parity, equalized odds, calibration)
- [ ] Know the impossibility of satisfying all fairness definitions simultaneously
- [ ] Analyze fairness-accuracy tradeoffs
- [ ] Interpret model cards and fairness documentation

**Implementation:**
- [ ] Measure fairness metrics on real datasets
- [ ] Detect bias in trained models across demographic groups
- [ ] Apply debiasing techniques (pre/in/post-processing)
- [ ] Create model cards with disaggregated metrics
- [ ] Build fairness monitoring systems

**Production Skills:**
- [ ] Audit models for regulatory compliance (GDPR, Fair Lending)
- [ ] Choose appropriate fairness metrics for use case
- [ ] Implement continuous fairness monitoring
- [ ] Document model limitations and known biases
- [ ] Make informed fairness-accuracy tradeoff decisions

---

## 💻 Exercises

### Exercise 1: Audit a Biased Model (45 mins)

**What You'll Learn:**
- Detecting hidden bias in "accurate" models
- Computing fairness metrics across demographic groups
- Interpreting disparity and its real-world impact
- Recognizing when overall accuracy masks unfairness

**Why It Matters:**
Real deployed models (Amazon hiring AI, COMPAS, face recognition) all achieved high overall accuracy but had severe fairness violations. This exercise teaches you to look beyond aggregate metrics and examine subgroup performance—critical for responsible AI.

**Task:** Audit a loan approval model for demographic parity and equalized odds violations.

**Starter code:**
```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset (simulate)
np.random.seed(42)
n = 10000

# Features (credit score, income, etc.)
X = np.random.randn(n, 10)

# Sensitive attribute (race: 0=majority, 1=minority)
race = np.random.choice([0, 1], size=n, p=[0.8, 0.2])

# Simulate bias: minority group has lower credit scores (historical discrimination)
X[race == 1, 0] -= 0.5  # Lower credit score for minority group

# True labels (ability to repay loan - based on features)
y_true = (X[:, 0] + X[:, 1] > 0).astype(int)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X, y_true)
y_pred = model.predict(X)

# Overall accuracy (looks good!)
print(f"Overall accuracy: {accuracy_score(y_true, y_pred):.2%}")

# TODO: Compute fairness metrics
# 1. Demographic parity difference
# 2. Equalized odds (TPR and FPR by group)
# 3. Accuracy by group
# 4. Visualize disparities
```

**Expected findings:**
- Overall accuracy: ~85%
- Demographic parity: 15-20 point gap (minority group approved less)
- TPR disparity: 10-15 point gap
- Minority group accuracy: 10-15 points lower

### Exercise 2: Implement Debiasing Techniques (60 mins)

**What You'll Learn:**
- Applying reweighting to balance training data
- Using adversarial debiasing during training
- Post-processing predictions with fair thresholds
- Comparing debiasing approaches (accuracy vs fairness tradeoff)

**Why It Matters:**
Detecting bias is not enough—you must fix it. This exercise gives hands-on experience with the three main debiasing paradigms (pre/in/post-processing), teaching you which to use in different production scenarios.

**Task:** Apply all three debiasing techniques and compare results.

**Implementation:**
```python
# Baseline: Biased model (from Exercise 1)
baseline_accuracy = accuracy_score(y_true, y_pred)
baseline_dpd = demographic_parity_difference(y_pred, race)

# Technique 1: Reweighting (pre-processing)
sample_weights = compute_sample_weights(y_true, race)
model_reweighted = RandomForestClassifier(random_state=42)
model_reweighted.fit(X, y_true, sample_weight=sample_weights)
y_pred_reweighted = model_reweighted.predict(X)

# Technique 2: Adversarial debiasing (in-processing)
fair_model = FairClassifier(input_dim=10)
train_adversarial(fair_model, torch.FloatTensor(X), torch.FloatTensor(y_true), torch.FloatTensor(race))
y_pred_adversarial = (fair_model(torch.FloatTensor(X))[0].squeeze() > 0.5).numpy()

# Technique 3: Threshold optimization (post-processing)
y_pred_proba = model.predict_proba(X)[:, 1]
thresholds = find_fair_thresholds(y_pred_proba, y_true, race, target_metric='demographic_parity')
y_pred_postproc = apply_fair_thresholds(y_pred_proba, race, thresholds)

# Compare
results = []
for name, y_pred_method in [
    ('Baseline', y_pred),
    ('Reweighting', y_pred_reweighted),
    ('Adversarial', y_pred_adversarial),
    ('Post-processing', y_pred_postproc)
]:
    acc = accuracy_score(y_true, y_pred_method)
    dpd = demographic_parity_difference(y_pred_method, race)
    tpr_diff, fpr_diff = equalized_odds_difference(y_true, y_pred_method, race)

    results.append({
        'Method': name,
        'Accuracy': f"{acc:.2%}",
        'Demographic Parity Diff': f"{dpd:.2%}",
        'TPR Diff': f"{tpr_diff:.2%}",
        'FPR Diff': f"{fpr_diff:.2%}"
    })

print(pd.DataFrame(results))
```

**Expected results:**
| Method | Accuracy | DPD | TPR Diff | FPR Diff |
|--------|----------|-----|----------|----------|
| Baseline | 85% | 18% | 15% | 12% |
| Reweighting | 82% | 8% | 10% | 8% |
| Adversarial | 83% | 5% | 7% | 6% |
| Post-proc | 84% | 2% | 14% | 3% |

**Analysis:**
- All debiasing methods reduce disparity
- Trade-off: 2-3% accuracy loss for 10-16% fairness gain
- Adversarial debiasing best overall balance

### Exercise 3: Create a Model Card (30 mins)

**What You'll Learn:**
- Documenting model performance across demographics
- Identifying and disclosing limitations
- Writing clear ethical considerations
- Meeting transparency requirements (GDPR, AI Act)

**Why It Matters:**
Regulatory frameworks (EU AI Act, proposed US bills) require transparency. Model cards are becoming industry standard (Google, OpenAI use them). This skill is essential for production ML deployment.

**Task:** Create a comprehensive model card for your loan approval model.

**Template:** (Fill in based on your model from Exercise 2)

```markdown
# Model Card: [Your Model Name]

## Model Details
- Model type: ___
- Version: ___
- Training data size: ___
- Features: ___

## Intended Use
- Primary use case: ___
- Target users: ___
- Out-of-scope uses: ___

## Performance Metrics
| Metric | Overall | Group A | Group B |
|--------|---------|---------|---------|
| Accuracy | ___ | ___ | ___ |
| Precision | ___ | ___ | ___ |
| Recall | ___ | ___ | ___ |
| Demographic Parity Diff | ___ | - | - |
| Equalized Odds (TPR) | ___ | ___ | ___ |

## Fairness Analysis
- Identified disparities: ___
- Root causes: ___
- Mitigation applied: ___
- Residual issues: ___

## Ethical Considerations
- Known biases: ___
- Privacy risks: ___
- Potential harms: ___

## Caveats and Recommendations
- Known failure modes: ___
- Monitoring requirements: ___
- Update frequency: ___
```

### Exercise 4: Fairness-Accuracy Tradeoff Curve (45 mins)

**What You'll Learn:**
- Plotting Pareto frontier of fairness vs accuracy
- Quantifying cost of fairness (in accuracy points)
- Choosing optimal operating point for business requirements
- Communicating tradeoffs to stakeholders

**Why It Matters:**
Business stakeholders need to understand the tradeoff: "How much accuracy do we sacrifice for fairness?" This exercise teaches you to quantify and visualize this, enabling informed decision-making.

**Task:** Generate tradeoff curve by varying fairness regularization strength.

**Implementation:**
```python
import matplotlib.pyplot as plt

# Vary fairness penalty (lambda)
lambdas = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
accuracies = []
fairness_violations = []

for lambda_fair in lambdas:
    # Train with fairness regularization
    model = FairClassifier(input_dim=10)
    train_with_regularization(model, X_train, y_train, race_train, lambda_fair=lambda_fair)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    dpd = demographic_parity_difference(y_pred, race_test)

    accuracies.append(acc)
    fairness_violations.append(dpd)

# Plot Pareto frontier
plt.figure(figsize=(10, 6))
plt.plot(fairness_violations, accuracies, 'o-', linewidth=2, markersize=8)
plt.xlabel('Fairness Violation (Demographic Parity Difference)', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Fairness-Accuracy Tradeoff Curve', fontsize=14)
plt.grid(True, alpha=0.3)

# Annotate points
for i, lambda_val in enumerate(lambdas):
    plt.annotate(f'λ={lambda_val}', (fairness_violations[i], accuracies[i]),
                 textcoords="offset points", xytext=(0,10), ha='center')

# Mark legal threshold
plt.axvline(x=0.10, color='r', linestyle='--', label='Legal threshold (10%)')
plt.legend()
plt.show()

# Print recommendations
print("Recommendations:")
for i, lambda_val in enumerate(lambdas):
    if fairness_violations[i] <= 0.10:
        print(f"λ={lambda_val}: Accuracy={accuracies[i]:.2%}, DPD={fairness_violations[i]:.2%} ✓ (Compliant)")
    else:
        print(f"λ={lambda_val}: Accuracy={accuracies[i]:.2%}, DPD={fairness_violations[i]:.2%} ✗ (Non-compliant)")
```

**Expected output:**
- Clear Pareto frontier showing tradeoff
- Identification of minimum λ to achieve compliance
- Quantified accuracy cost of fairness (e.g., "3% accuracy for 15% fairness improvement")

### Exercise 5: Production Fairness Monitoring (60 mins)

**What You'll Learn:**
- Building continuous fairness monitoring system
- Setting up alerts for fairness violations
- Logging disaggregated metrics over time
- Detecting fairness drift in production

**Why It Matters:**
Fairness is not "one and done"—models drift over time, data distributions shift, regulations change. Production systems need continuous monitoring. This exercise simulates a real MLOps fairness monitoring pipeline.

**Task:** Build a monitoring system that tracks fairness metrics over time and alerts on violations.

**Implementation:**
```python
class ProductionFairnessMonitor:
    def __init__(self, model, protected_attrs, thresholds, log_db):
        self.model = model
        self.protected_attrs = protected_attrs
        self.thresholds = thresholds
        self.log_db = log_db
        self.violation_count = 0

    def monitor_batch(self, X, y_true, metadata, timestamp):
        """Monitor a batch of predictions."""
        y_pred = self.model.predict(X)

        # Compute metrics overall
        overall_acc = accuracy_score(y_true, y_pred)

        # Compute metrics per group
        metrics_by_group = {}
        for attr in self.protected_attrs:
            sensitive = metadata[attr]
            metrics_by_group[attr] = {}

            for group in np.unique(sensitive):
                mask = sensitive == group
                group_acc = accuracy_score(y_true[mask], y_pred[mask])
                metrics_by_group[attr][f'group_{group}_accuracy'] = group_acc

            # Fairness metrics
            dpd = demographic_parity_difference(y_pred, sensitive)
            tpr_diff, fpr_diff = equalized_odds_difference(y_true, y_pred, sensitive)

            metrics_by_group[attr]['demographic_parity_diff'] = dpd
            metrics_by_group[attr]['tpr_diff'] = tpr_diff
            metrics_by_group[attr]['fpr_diff'] = fpr_diff

            # Check thresholds
            if dpd > self.thresholds['demographic_parity']:
                self.send_alert(f"Demographic parity violation for {attr}: {dpd:.2%}", timestamp)
            if tpr_diff > self.thresholds['equalized_odds']:
                self.send_alert(f"Equalized odds (TPR) violation for {attr}: {tpr_diff:.2%}", timestamp)

        # Log to database
        self.log_metrics(timestamp, overall_acc, metrics_by_group)

        return metrics_by_group

    def send_alert(self, message, timestamp):
        """Send alert to ops team."""
        print(f"[ALERT {timestamp}] {message}")
        self.violation_count += 1
        # In production: send to Slack, PagerDuty, etc.

    def log_metrics(self, timestamp, overall_acc, metrics_by_group):
        """Log metrics to database for historical tracking."""
        log_entry = {
            'timestamp': timestamp,
            'overall_accuracy': overall_acc,
            'metrics_by_group': metrics_by_group
        }
        self.log_db.append(log_entry)

    def plot_fairness_over_time(self):
        """Visualize fairness drift."""
        timestamps = [entry['timestamp'] for entry in self.log_db]
        dpd_values = [entry['metrics_by_group']['race']['demographic_parity_diff']
                      for entry in self.log_db]

        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, dpd_values, 'o-', linewidth=2)
        plt.axhline(y=self.thresholds['demographic_parity'], color='r',
                    linestyle='--', label='Threshold')
        plt.xlabel('Time')
        plt.ylabel('Demographic Parity Difference')
        plt.title('Fairness Monitoring: Demographic Parity Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

# Simulate production monitoring
log_db = []
monitor = ProductionFairnessMonitor(
    model=model,
    protected_attrs=['race', 'gender'],
    thresholds={'demographic_parity': 0.10, 'equalized_odds': 0.10},
    log_db=log_db
)

# Simulate weekly batches
for week in range(12):
    # Simulate data drift (fairness degrades over time)
    X_batch = X_test + np.random.randn(*X_test.shape) * 0.1 * week
    y_batch = y_test
    race_batch = race_test

    metrics = monitor.monitor_batch(
        X_batch,
        y_batch,
        metadata={'race': race_batch, 'gender': np.random.choice([0, 1], len(race_batch))},
        timestamp=f'Week {week+1}'
    )

# Visualize drift
monitor.plot_fairness_over_time()
print(f"\nTotal fairness violations: {monitor.violation_count}")
```

**Expected behavior:**
- Fairness metrics logged every week
- Alerts triggered when thresholds exceeded
- Visualization showing fairness drift over time
- Clear indication when model needs retraining

---

## ⚠️ Common Pitfalls

### 1. Ignoring Intersectionality

**Symptom:** Model fair for race and gender separately, but unfair for Black women.

**Cause:** Fairness checked independently, not for intersecting groups.

**Solution:**
```python
# Check all intersections
for race_val in [0, 1]:
    for gender_val in [0, 1]:
        mask = (race == race_val) & (gender == gender_val)
        group_acc = accuracy_score(y_true[mask], y_pred[mask])
        print(f"Race={race_val}, Gender={gender_val}: Accuracy={group_acc:.2%}")
```

### 2. Optimizing Wrong Fairness Metric

**Symptom:** Model satisfies demographic parity but has wildly different error rates.

**Cause:** Demographic parity doesn't consider accuracy—only positive prediction rate.

**Solution:** Choose metric based on use case. For high-stakes decisions, use equalized odds or calibration.

### 3. Removing Sensitive Attributes Doesn't Remove Bias

**Symptom:** Dropped race/gender from features, but model still biased.

**Cause:** Proxy features (zip code → race, name → gender) still encode sensitive info.

**Solution:** Use fairness constraints or debiasing techniques, not just feature removal.

### 4. Training on Biased Labels

**Symptom:** Model perfectly replicates training labels but is still unfair.

**Cause:** Training labels themselves are biased (e.g., historical discrimination).

**Solution:** Pre-processing to correct label bias or use fairness-aware learning.

### 5. Forgetting to Monitor After Deployment

**Symptom:** Model fair at launch, becomes unfair 6 months later.

**Cause:** Data drift, distribution shift, no continuous monitoring.

**Solution:** Implement continuous fairness monitoring (Exercise 5).

---

## 🏆 Expert Checklist for Mastery

**Foundations:**
- [ ] Understand sources of bias (historical, representation, measurement)
- [ ] Can explain demographic parity, equalized odds, calibration
- [ ] Know the impossibility of satisfying all fairness definitions
- [ ] Recognize fairness-accuracy tradeoffs

**Implementation:**
- [ ] Computed fairness metrics on real dataset
- [ ] Applied debiasing (pre/in/post-processing)
- [ ] Created model card with disaggregated metrics
- [ ] Built fairness monitoring system

**Production:**
- [ ] Audited production model for bias
- [ ] Set up continuous fairness monitoring
- [ ] Made informed fairness-accuracy tradeoff decisions
- [ ] Documented model limitations and known biases

**Advanced:**
- [ ] Understand intersectionality in fairness
- [ ] Familiar with fairness regulations (GDPR, Fair Lending, EU AI Act)
- [ ] Can advise stakeholders on fairness requirements
- [ ] Know latest research (causal fairness, individual fairness)

---

## 🚀 Next Steps

After mastering fairness:

1. **Explainability (SHAP/LIME)**
   - Understand which features drive predictions
   - Detect proxy features encoding sensitive attributes
   - Explain model decisions to auditors

2. **Privacy-Preserving ML**
   - Differential privacy for training data protection
   - Federated learning for decentralized fairness
   - Secure multi-party computation

3. **Causal Fairness**
   - Counterfactual fairness (what would have happened?)
   - Path-specific causal effects
   - Avoiding feedback loops

4. **Regulatory Compliance**
   - GDPR (right to explanation)
   - EU AI Act (high-risk AI systems)
   - Sector-specific regulations (Fair Lending Act, HIPAA)

---

## 📚 References

**Papers:**
- [Fairness Through Awareness (Dwork et al., 2012)](https://arxiv.org/abs/1104.3913)
- [Equality of Opportunity in Supervised Learning (Hardt et al., 2016)](https://arxiv.org/abs/1610.02413)
- [Inherent Trade-Offs in the Fair Determination of Risk Scores (Kleinberg et al., 2016)](https://arxiv.org/abs/1609.05807)
- [Model Cards for Model Reporting (Mitchell et al., 2019)](https://arxiv.org/abs/1810.03993)

**Tools:**
- [AI Fairness 360 (IBM)](https://aif360.mybluemix.net/)
- [Fairlearn (Microsoft)](https://fairlearn.org/)
- [What-If Tool (Google)](https://pair-code.github.io/what-if-tool/)

**Regulations:**
- [EU AI Act](https://artificialintelligenceact.eu/)
- [GDPR Article 22 (Automated Decision-Making)](https://gdpr-info.eu/art-22-gdpr/)
- [Fair Credit Reporting Act (US)](https://www.ftc.gov/legal-library/browse/statutes/fair-credit-reporting-act)

---

## 🎯 Solution

Complete implementation: `solution/fairness_audit.py`

**What you'll build:**
- Comprehensive fairness auditing framework
- Multiple debiasing techniques (pre/in/post-processing)
- Model card generation
- Production monitoring system
- Fairness-accuracy tradeoff analysis
- Compliance checking (80% rule, disparate impact)

**Next: Lab 4 - Model Serving & Production Inference!**
