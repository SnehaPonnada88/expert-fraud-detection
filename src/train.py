import numpy as np
import random
from stable_baselines3 import PPO
from env import ExpertEnv
import gymnasium as gym
from gymnasium import spaces


# -----------------------------
# SYNTHETIC DATA GENERATION
# -----------------------------
def generate_sample():
    """
    Simulate candidate features
    """

    is_fraud = random.random() > 0.5

    if not is_fraud:
        # Genuine
        profile = np.random.uniform(0.0, 0.4)
        depth = np.random.uniform(0.6, 1.0)
        consistency = np.random.uniform(0.6, 1.0)
        progression = np.random.uniform(0.6, 1.0)
        label = 0
    else:
        # Fraud
        profile = np.random.uniform(0.6, 1.0)
        depth = np.random.uniform(0.0, 0.4)
        consistency = np.random.uniform(0.0, 0.4)
        progression = np.random.uniform(0.0, 0.4)
        label = 1

    return profile, depth, consistency, progression, label


# -----------------------------
# TRAINING ENV (SELF-CONTAINED)
# -----------------------------
class TrainingEnv(gym.Env):
    def __init__(self):
        super().__init__()

        # Actions: PASS / FLAG / PROBE
        self.action_space = spaces.Discrete(3)

        # State: profile, depth, consistency, progression, uncertainty
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(5,), dtype=np.float32
        )

    def compute_uncertainty(self, depth, consistency):
        return abs(depth - consistency)

    def reset(self, seed=None, options=None):
        self.profile, self.depth, self.consistency, self.progression, self.label = generate_sample()

        # initial state (only profile visible)
        self.state = np.array([
            self.profile,
            0.0,
            0.0,
            0.0,
            0.0  # uncertainty
        ], dtype=np.float32)

        return self.state, {}

    def step(self, action):
        uncertainty = self.compute_uncertainty(self.depth, self.consistency)

        expertise = (
            0.5 * self.depth +
            0.3 * self.consistency +
            0.2 * self.progression
        )

        # -----------------------------
        # PROBE
        # -----------------------------
        if action == 2:
            # reveal full state
            self.state = np.array([
                self.profile,
                self.depth,
                self.consistency,
                self.progression,
                uncertainty
            ], dtype=np.float32)

            # reward shaping
            if 0.35 < expertise < 0.7 or uncertainty > 0.2:
                reward = 5   # good probe
            else:
                reward = -2  # unnecessary probe

            return self.state, reward, False, False, {}

        # -----------------------------
        # PASS
        # -----------------------------
        elif action == 0:
            reward = 10 if self.label == 0 else -10
            return self.state, reward, True, False, {}

        # -----------------------------
        # FLAG
        # -----------------------------
        else:
            reward = 10 if self.label == 1 else -10
            return self.state, reward, True, False, {}


# -----------------------------
# TRAINING
# -----------------------------
if __name__ == "__main__":

    env = TrainingEnv()

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64
    )

    model.learn(total_timesteps=20000)

    model.save("models/ppo_model")

    print("Training complete and model saved!")