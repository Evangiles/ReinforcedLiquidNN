import numpy as np
import pandas as pd
from typing import Dict, List
from model import MinerviniEnv


def compute_rs_ranks(data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
    momentum = {}
    for ticker, df in data.items():
        if len(df) < 130:
            continue
        momentum[ticker] = df['Close'].iloc[-1] / df['Close'].iloc[-125] - 1
    ranks = {}
    if momentum:
        series = pd.Series(momentum)
        ranks = (series.rank(pct=True) - 0).to_dict()
    return ranks


def apply_minervini_rules(df: pd.DataFrame, market_df: pd.DataFrame, rs_rank: float) -> Dict[str, bool]:
    price = df['Close'].iloc[-1]
    ma50 = df['MA50'].iloc[-1]
    ma150 = df['MA150'].iloc[-1]
    ma200 = df['MA200'].iloc[-1]
    rules = {}
    rules['ma_alignment'] = price > ma50 > ma150 > ma200 and ma150 > ma200 and ma200 > df['MA200'].iloc[-20]
    rules['52w_proximity'] = price >= 1.25 * df['52w_low'].iloc[-1] and price <= 0.75 * df['52w_high'].iloc[-1]
    rules['rs'] = rs_rank >= 0.6
    rules['volume_surge'] = df['Volume'].iloc[-1] >= 1.5 * df['VOL20'].iloc[-1]
    rules['trend'] = price > df['Close'].iloc[-30]
    rules['market'] = market_df['Close'].iloc[-1] > market_df['MA200'].iloc[-1]
    return rules


def latest_observation(df: pd.DataFrame) -> np.ndarray:
    env = MinerviniEnv(df)
    env.step_index = len(df)
    return env._get_obs()


def generate_signals(agent, data: Dict[str, pd.DataFrame]) -> List[Dict[str, str]]:
    market_df = data.get('SPY')
    rs_ranks = compute_rs_ranks(data)
    signals = []
    for ticker, df in data.items():
        if ticker == 'SPY':
            continue
        obs = latest_observation(df)
        obs_tensor = agent.policy.obs_to_tensor(obs)[0]
        dist = agent.policy.get_distribution(obs_tensor)
        probs = dist.distribution.probs.detach().cpu().numpy()[0]
        action = np.argmax(probs)
        rules = apply_minervini_rules(df, market_df, rs_ranks.get(ticker, 0))
        all_rules = all(rules.values())
        price = df['Close'].iloc[-1]
        if action == 1 and all_rules:
            decision = 'Buy'
        elif price < df['MA50'].iloc[-1]:
            decision = 'Sell'
        else:
            decision = 'Hold'
        signals.append({
            'ticker': ticker,
            'action': decision,
            'confidence': float(probs[action]),
            'rules': rules
        })
    return signals
