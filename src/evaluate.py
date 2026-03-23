
import pandas as pd
from stable_baselines3 import PPO
from env import ExpertEnv
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv("../data/synthetic.csv")
env = ExpertEnv(df)
model = PPO.load("../models/ppo_model")

y_true, y_pred = [], []

for _ in range(300):
    obs, _ = env.reset()
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)

    y_true.append(env.row["label"])
    y_pred.append(1 if action == 1 else 0)

print("Accuracy:", accuracy_score(y_true, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
