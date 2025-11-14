"""
PPO: Proximal Policy Optimization
State-of-the-art reinforcement learning algorithm
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np


class ActorCritic(nn.Module):
    """
    Actor-Critic Network

    Actor: Policy network π(a|s) - outputs action probabilities
    Critic: Value network V(s) - estimates state value
    """

    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )

        # Actor head (policy)
        self.actor = nn.Linear(hidden_dim, action_dim)

        # Critic head (value function)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        features = self.shared(state)
        action_logits = self.actor(features)
        value = self.critic(features)
        return action_logits, value

    def act(self, state, deterministic=False):
        """
        Sample action from policy

        Args:
            state: Current state
            deterministic: If True, return argmax action

        Returns:
            action: Sampled action
            log_prob: Log probability of action
            value: State value
        """
        action_logits, value = self.forward(state)
        dist = Categorical(logits=action_logits)

        if deterministic:
            action = action_logits.argmax(dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)

        return action, log_prob, value

    def evaluate(self, states, actions):
        """
        Evaluate actions

        Returns:
            log_probs: Log probabilities of actions
            values: State values
            entropy: Policy entropy (for exploration bonus)
        """
        action_logits, values = self.forward(states)
        dist = Categorical(logits=action_logits)

        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_probs, values.squeeze(), entropy


class RolloutBuffer:
    """Buffer to store rollout experience"""

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []

    def store(self, state, action, reward, log_prob, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)

    def get(self):
        return (
            torch.stack(self.states),
            torch.tensor(self.actions),
            torch.tensor(self.rewards),
            torch.stack(self.log_probs),
            torch.stack(self.values),
            torch.tensor(self.dones, dtype=torch.float32)
        )

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
        self.dones.clear()


class PPO:
    """
    PPO Algorithm

    Key idea: Clip the policy ratio to prevent too large policy updates
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=64,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        # Actor-Critic network
        self.policy = ActorCritic(state_dim, action_dim, hidden_dim)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        # Rollout buffer
        self.buffer = RolloutBuffer()

    def select_action(self, state, deterministic=False):
        """Select action and store in buffer"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action, log_prob, value = self.policy.act(state_tensor, deterministic)

        return action.item(), log_prob, value

    def compute_gae(self, rewards, values, dones, next_value):
        """
        Compute Generalized Advantage Estimation (GAE)

        Advantage: A(s,a) = Q(s,a) - V(s)
        GAE: Exponentially weighted average of TD residuals
        """
        advantages = []
        gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_v = next_value
            else:
                next_v = values[t + 1]

            # TD residual: δ = r + γV(s') - V(s)
            delta = rewards[t] + self.gamma * next_v * (1 - dones[t]) - values[t]

            # GAE: A = δ + (γλ)δ_{t+1} + (γλ)^2 δ_{t+2} + ...
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        advantages = torch.tensor(advantages)
        returns = advantages + values

        return advantages, returns

    def update(self, epochs=10, batch_size=64):
        """
        Update policy using PPO

        Args:
            epochs: Number of epochs to train on collected data
            batch_size: Mini-batch size
        """
        # Get rollout data
        states, actions, rewards, old_log_probs, values, dones = self.buffer.get()

        # Compute advantages (requires final value for bootstrapping)
        with torch.no_grad():
            _, next_value = self.policy(states[-1].unsqueeze(0))
            next_value = next_value.squeeze()

        advantages, returns = self.compute_gae(rewards, values, dones, next_value)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Multiple epochs of optimization
        dataset_size = len(states)
        for _ in range(epochs):
            # Random mini-batches
            indices = np.random.permutation(dataset_size)

            for start in range(0, dataset_size, batch_size):
                end = min(start + batch_size, dataset_size)
                batch_indices = indices[start:end]

                # Get batch
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # Evaluate actions
                log_probs, values, entropy = self.policy.evaluate(batch_states, batch_actions)

                # Policy ratio: π_new / π_old
                ratios = torch.exp(log_probs - batch_old_log_probs)

                # Clipped surrogate objective
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (MSE)
                value_loss = F.mse_loss(values, batch_returns)

                # Entropy bonus (for exploration)
                entropy_loss = -entropy.mean()

                # Total loss
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

        # Clear buffer
        self.buffer.clear()

        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': -entropy_loss.item()
        }


def train_ppo_cartpole():
    """Example: Train PPO on CartPole"""
    try:
        import gym
    except ImportError:
        print("Install gym: pip install gym")
        return

    print("=" * 60)
    print("Training PPO on CartPole-v1")
    print("=" * 60)

    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Create PPO agent
    agent = PPO(state_dim, action_dim, hidden_dim=64, lr=3e-4)

    # Training parameters
    max_episodes = 500
    max_steps = 500
    update_every = 2048

    episode_rewards = []
    total_steps = 0

    for episode in range(max_episodes):
        state = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            # Select action
            action, log_prob, value = agent.select_action(state)

            # Environment step
            next_state, reward, done, _ = env.step(action)

            # Store in buffer
            agent.buffer.store(
                torch.FloatTensor(state),
                action,
                reward,
                log_prob,
                value,
                done
            )

            state = next_state
            episode_reward += reward
            total_steps += 1

            # Update policy
            if total_steps % update_every == 0:
                metrics = agent.update()

            if done:
                break

        episode_rewards.append(episode_reward)

        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {episode+1} | Avg Reward (last 10): {avg_reward:.2f}")

            if avg_reward >= 475:  # Solved threshold
                print(f"\nSolved in {episode+1} episodes!")
                break

    env.close()

    print("\n" + "=" * 60)
    print("PPO Training Complete!")
    print("=" * 60)


if __name__ == "__main__":
    print("=" * 60)
    print("PPO: Proximal Policy Optimization")
    print("=" * 60)

    # Demonstrate PPO components
    state_dim = 4
    action_dim = 2
    batch_size = 32

    # Create policy
    policy = ActorCritic(state_dim, action_dim, hidden_dim=64)

    print(f"\nActor-Critic Network:")
    total_params = sum(p.numel() for p in policy.parameters())
    print(f"Parameters: {total_params:,}")

    # Example forward pass
    states = torch.randn(batch_size, state_dim)
    actions, log_probs, values = policy.act(states)

    print(f"\nExample:")
    print(f"States: {states.shape}")
    print(f"Actions: {actions.shape}")
    print(f"Log probs: {log_probs.shape}")
    print(f"Values: {values.shape}")

    print("\n" + "=" * 60)
    print("Key Concepts:")
    print("=" * 60)
    print("✓ Actor-Critic: Learn policy and value function jointly")
    print("✓ PPO Clipping: Prevent too large policy updates")
    print("✓ GAE: Better advantage estimation")
    print("✓ Multiple epochs: Reuse collected data efficiently")
    print("✓ Entropy bonus: Encourage exploration")

    print("\n" + "=" * 60)
    print("To train on CartPole, uncomment below:")
    print("=" * 60)
    print("# train_ppo_cartpole()")
