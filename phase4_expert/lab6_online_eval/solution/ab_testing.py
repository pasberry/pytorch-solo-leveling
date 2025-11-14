"""
Online Evaluation & A/B Testing
Measure real-world model performance with statistical rigor
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple


class ABTest:
    """
    A/B Testing Framework for ML Models

    Compares two models (Control vs Treatment) using statistical tests
    """

    def __init__(self, alpha=0.05):
        """
        Args:
            alpha: Significance level (typically 0.05 for 95% confidence)
        """
        self.alpha = alpha
        self.control_metrics = []
        self.treatment_metrics = []

    def log_outcome(self, group: str, metric_value: float):
        """
        Log single outcome

        Args:
            group: 'control' or 'treatment'
            metric_value: Metric value (e.g., CTR, accuracy, latency)
        """
        if group == 'control':
            self.control_metrics.append(metric_value)
        elif group == 'treatment':
            self.treatment_metrics.append(metric_value)
        else:
            raise ValueError(f"Invalid group: {group}")

    def t_test(self) -> Dict:
        """
        Two-sample t-test for continuous metrics

        Returns:
            Dictionary with test results
        """
        control = np.array(self.control_metrics)
        treatment = np.array(self.treatment_metrics)

        # Two-sample t-test
        t_stat, p_value = stats.ttest_ind(treatment, control)

        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(control)-1)*control.std()**2 + (len(treatment)-1)*treatment.std()**2) / (len(control) + len(treatment) - 2))
        cohens_d = (treatment.mean() - control.mean()) / pooled_std

        # Confidence interval
        diff = treatment.mean() - control.mean()
        se = np.sqrt(control.var()/len(control) + treatment.var()/len(treatment))
        ci_lower = diff - 1.96 * se
        ci_upper = diff + 1.96 * se

        significant = p_value < self.alpha

        return {
            'control_mean': control.mean(),
            'treatment_mean': treatment.mean(),
            'diff': diff,
            'diff_pct': (diff / control.mean()) * 100 if control.mean() != 0 else 0,
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': significant,
            'cohens_d': cohens_d,
            'ci_95': (ci_lower, ci_upper),
            'n_control': len(control),
            'n_treatment': len(treatment)
        }

    def z_test_proportions(self, control_successes: int, treatment_successes: int) -> Dict:
        """
        Z-test for binary metrics (e.g., CTR, conversion rate)

        Args:
            control_successes: Number of successes in control
            treatment_successes: Number of successes in treatment

        Returns:
            Dictionary with test results
        """
        n_control = len(self.control_metrics)
        n_treatment = len(self.treatment_metrics)

        p_control = control_successes / n_control
        p_treatment = treatment_successes / n_treatment

        # Pooled proportion
        p_pooled = (control_successes + treatment_successes) / (n_control + n_treatment)

        # Z-statistic
        se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n_control + 1/n_treatment))
        z = (p_treatment - p_control) / se

        # P-value (two-tailed)
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))

        # Confidence interval
        se_diff = np.sqrt(p_control*(1-p_control)/n_control + p_treatment*(1-p_treatment)/n_treatment)
        diff = p_treatment - p_control
        ci_lower = diff - 1.96 * se_diff
        ci_upper = diff + 1.96 * se_diff

        significant = p_value < self.alpha

        return {
            'control_rate': p_control,
            'treatment_rate': p_treatment,
            'diff': diff,
            'diff_pct': (diff / p_control) * 100 if p_control != 0 else 0,
            'z_statistic': z,
            'p_value': p_value,
            'significant': significant,
            'ci_95': (ci_lower, ci_upper),
            'n_control': n_control,
            'n_treatment': n_treatment
        }

    def required_sample_size(self, baseline_rate: float, mde: float, power: float = 0.8) -> int:
        """
        Calculate required sample size per group

        Args:
            baseline_rate: Current conversion rate (e.g., 0.10 for 10%)
            mde: Minimum detectable effect (relative, e.g., 0.05 for 5% improvement)
            power: Statistical power (typically 0.8)

        Returns:
            Required sample size per group
        """
        alpha = self.alpha
        z_alpha = stats.norm.ppf(1 - alpha/2)
        z_beta = stats.norm.ppf(power)

        p1 = baseline_rate
        p2 = p1 * (1 + mde)

        n = (z_alpha + z_beta)**2 * (p1*(1-p1) + p2*(1-p2)) / (p2 - p1)**2

        return int(np.ceil(n))


class MultiArmedBandit:
    """
    Multi-Armed Bandit for dynamic traffic allocation

    Instead of fixed 50/50 split, gradually shift traffic to better model
    """

    def __init__(self, n_arms: int = 2, epsilon: float = 0.1):
        """
        Args:
            n_arms: Number of models to test
            epsilon: Exploration rate (0-1)
        """
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)

    def select_arm(self) -> int:
        """
        Select which model to show user

        Uses epsilon-greedy strategy:
        - With probability epsilon: explore (random)
        - With probability 1-epsilon: exploit (best arm)
        """
        if np.random.rand() < self.epsilon:
            # Explore
            return np.random.randint(self.n_arms)
        else:
            # Exploit
            return np.argmax(self.values)

    def update(self, arm: int, reward: float):
        """
        Update arm statistics

        Args:
            arm: Which model was shown
            reward: Observed metric (1 for success, 0 for failure)
        """
        self.counts[arm] += 1
        n = self.counts[arm]

        # Incremental average
        value = self.values[arm]
        self.values[arm] = value + (reward - value) / n

    def get_statistics(self) -> Dict:
        """Get current statistics"""
        return {
            'counts': self.counts.tolist(),
            'values': self.values.tolist(),
            'best_arm': int(np.argmax(self.values))
        }


if __name__ == "__main__":
    print("=" * 60)
    print("Online Evaluation & A/B Testing")
    print("=" * 60)

    # Example 1: A/B Test for CTR
    print("\nExample 1: A/B Test for Click-Through Rate (CTR)")
    print("-" * 60)

    np.random.seed(42)

    # Simulate A/B test
    n_users = 10000
    control_ctr = 0.10  # 10% CTR
    treatment_ctr = 0.105  # 10.5% CTR (5% improvement)

    ab_test = ABTest(alpha=0.05)

    control_clicks = 0
    treatment_clicks = 0

    for i in range(n_users):
        if i < n_users // 2:
            # Control group
            clicked = np.random.rand() < control_ctr
            ab_test.log_outcome('control', float(clicked))
            control_clicks += clicked
        else:
            # Treatment group
            clicked = np.random.rand() < treatment_ctr
            ab_test.log_outcome('treatment', float(clicked))
            treatment_clicks += clicked

    results = ab_test.z_test_proportions(control_clicks, treatment_clicks)

    print(f"Control CTR:    {results['control_rate']:.4f} ({control_clicks}/{n_users//2})")
    print(f"Treatment CTR:  {results['treatment_rate']:.4f} ({treatment_clicks}/{n_users//2})")
    print(f"Difference:     {results['diff']:.4f} ({results['diff_pct']:.2f}%)")
    print(f"95% CI:         [{results['ci_95'][0]:.4f}, {results['ci_95'][1]:.4f}]")
    print(f"P-value:        {results['p_value']:.4f}")
    print(f"Significant:    {results['significant']}")

    if results['significant']:
        print("\n✓ Treatment is statistically significantly better!")
    else:
        print("\n✗ No statistically significant difference")

    # Example 2: Sample size calculation
    print("\n" + "=" * 60)
    print("Example 2: Sample Size Calculation")
    print("=" * 60)

    baseline_ctr = 0.10
    mde = 0.05  # Want to detect 5% relative improvement

    required_n = ab_test.required_sample_size(baseline_ctr, mde, power=0.8)

    print(f"Baseline CTR:              {baseline_ctr:.2%}")
    print(f"Minimum Detectable Effect: {mde:.2%}")
    print(f"Required sample size:      {required_n:,} per group")
    print(f"Total required:            {required_n * 2:,} users")

    # Example 3: Multi-Armed Bandit
    print("\n" + "=" * 60)
    print("Example 3: Multi-Armed Bandit")
    print("=" * 60)

    # Simulate 3 models with different CTRs
    true_ctrs = [0.10, 0.12, 0.11]  # Model 1: 10%, Model 2: 12%, Model 3: 11%
    n_rounds = 1000

    bandit = MultiArmedBandit(n_arms=3, epsilon=0.1)

    print("Running bandit for 1000 rounds...")

    for round in range(n_rounds):
        # Select model
        arm = bandit.select_arm()

        # Simulate user interaction
        reward = 1.0 if np.random.rand() < true_ctrs[arm] else 0.0

        # Update statistics
        bandit.update(arm, reward)

    stats = bandit.get_statistics()

    print(f"\nResults after {n_rounds} rounds:")
    print(f"True CTRs:      {true_ctrs}")
    print(f"Estimated CTRs: {[f'{v:.4f}' for v in stats['values']]}")
    print(f"Traffic split:  {[f'{c/n_rounds:.2%}' for c in stats['counts']]}")
    print(f"Best model:     Model {stats['best_arm'] + 1}")

    print("\n" + "=" * 60)
    print("Key Concepts:")
    print("=" * 60)
    print("✓ A/B Testing: Fixed traffic split (50/50), requires sample size")
    print("✓ Bandit: Dynamic traffic allocation, faster convergence")
    print("✓ Statistical Significance: p < 0.05 (95% confidence)")
    print("✓ Effect Size: Practical significance (is improvement meaningful?)")
    print("✓ Sample Size: Larger for smaller effects")
    print("✓ Multiple Testing: Bonferroni correction for multiple metrics")

    print("\n" + "=" * 60)
    print("Production Workflow:")
    print("=" * 60)
    print("1. Define success metric (CTR, revenue, latency)")
    print("2. Calculate required sample size")
    print("3. Run experiment for sufficient duration")
    print("4. Check guardrail metrics (no regressions)")
    print("5. Perform statistical test")
    print("6. If significant: Roll out to 100%")
    print("7. Monitor for long-term effects")

    print("\n" + "=" * 60)
    print("Common Pitfalls:")
    print("=" * 60)
    print("✗ Stopping test early (p-value hacking)")
    print("✗ Ignoring novelty effects")
    print("✗ Not checking guardrail metrics")
    print("✗ Testing multiple metrics without correction")
    print("✗ Insufficient sample size")
    print("✗ Ignoring long-term impact")
