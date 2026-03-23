import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import random

class ExpertEnv(gym.Env):
    def __init__(self, df):
        super().__init__()
        self.df = df.reset_index(drop=True)

        # Actions: 0 = PASS, 1 = FLAG, 2 = PROBE
        self.action_space = spaces.Discrete(3)

        # State: profile, depth, consistency, progression, uncertainty
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(5,), dtype=np.float32
        )

    def compute_uncertainty(self, depth, consistency):
        return abs(depth - consistency)

    def reset(self, seed=None):
        self.idx = random.randint(0, len(self.df) - 1)
        self.row = self.df.iloc[self.idx]

        # initial state (only profile known)
        profile = self.row["profile_score"]

        self.state = np.array([
            profile,
            0.0,
            0.0,
            0.0,
            0.0   # uncertainty
        ], dtype=np.float32)

        return self.state, {}

    def update_state(self, profile, depth, consistency, progression):
        uncertainty = self.compute_uncertainty(depth, consistency)

        self.state = np.array([
            profile,
            depth,
            consistency,
            progression,
            uncertainty
        ], dtype=np.float32)

        return self.state

    def step(self, action):
        label = self.row["label"]

        profile = self.row["profile_score"]
        depth = self.row["depth_score"]
        consistency = self.row["consistency_score"]
        progression = self.row["progression_score"]

        uncertainty = self.compute_uncertainty(depth, consistency)

        expertise = (
            0.5 * depth +
            0.3 * consistency +
            0.2 * progression
        )

        # -----------------------------
        # PROBE ACTION
        # -----------------------------
        if action == 2:
            # reward probing in uncertain cases
            if 0.35 < expertise < 0.7 or uncertainty > 0.2:
                reward = 5   # good probe
            else:
                reward = -2  # unnecessary probe

            # move to full state
            self.state = np.array([
                profile,
                depth,
                consistency,
                progression,
                uncertainty
            ], dtype=np.float32)

            return self.state, reward, False, False, {}

        # -----------------------------
        # PASS ACTION
        # -----------------------------
        elif action == 0:
            reward = 10 if label == 0 else -10
            return self.state, reward, True, False, {}

        # -----------------------------
        # FLAG ACTION
        # -----------------------------
        else:
            reward = 10 if label == 1 else -10
            return self.state, reward, True, False, {}