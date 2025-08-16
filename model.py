import numpy as np
import torch
import torch.nn as nn
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class LiquidCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fc = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, x, h):
        combined = torch.cat([x, h], dim=-1)
        u = torch.tanh(self.fc(combined))
        dh = u - h
        h = h + dh * torch.sigmoid(u)
        return h


class LiquidNetwork(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.cell = LiquidCell(input_size, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, x):
        h = torch.zeros(x.size(0), self.hidden_size, device=x.device)
        for t in range(x.size(1)):
            h = self.cell(x[:, t, :], h)
        return h


class LNNExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, seq_len=30, features_dim=64):
        self.seq_len = seq_len
        super().__init__(observation_space, features_dim)
        self.lnn = LiquidNetwork(2, features_dim)

    def forward(self, observations):
        batch = observations.view(observations.size(0), self.seq_len, 2)
        h = self.lnn(batch)
        return h


class MinerviniEnv(gym.Env):
    def __init__(self, df):
        super().__init__()
        self.df = df
        self.features = df[['Close', 'Volume']]
        self.returns = self.features['Close'].pct_change().fillna(0)
        self.vol_ratio = (self.features['Volume'] / self.features['Volume'].rolling(20).mean()).fillna(1)
        self.seq_len = 30
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.seq_len*2,), dtype=np.float32)
        self.reset()

    def _get_obs(self):
        start = self.step_index - self.seq_len
        seq_ret = self.returns.iloc[start:self.step_index].values
        seq_vol = self.vol_ratio.iloc[start:self.step_index].values
        obs = np.stack([seq_ret, seq_vol], axis=1).astype(np.float32).flatten()
        return obs

    def reset(self):
        self.step_index = self.seq_len
        self.position = 0
        self.entry_price = 0.0
        return self._get_obs()

    def step(self, action):
        reward = 0.0
        done = False
        price = self.features['Close'].iloc[self.step_index]
        ma50 = self.df['MA50'].iloc[self.step_index]
        if action == 1 and self.position == 0:
            self.position = 1
            self.entry_price = price
        elif action == 2 and self.position == 1:
            reward = (price - self.entry_price) / self.entry_price
            self.position = 0
        if self.position == 1 and price < ma50:
            reward -= 1.0
            self.position = 0
        self.step_index += 1
        if self.step_index >= len(self.df):
            done = True
        return self._get_obs(), reward, done, {}


def train_agent(df, timesteps=10000):
    env = MinerviniEnv(df)
    policy_kwargs = dict(features_extractor_class=LNNExtractor, features_extractor_kwargs=dict(seq_len=30, features_dim=64))
    model = PPO('MlpPolicy', env, policy_kwargs=policy_kwargs, verbose=0)
    model.learn(total_timesteps=timesteps)
    return model
