import argparse
import os
from data_loader import load_data
from model import train_agent
from signals import generate_signals
from notifier import send_email


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--timesteps', type=int, default=1000)
    parser.add_argument('--no-train', action='store_true')
    args = parser.parse_args()

    tickers = ['AAPL', 'MSFT', 'NVDA', 'SPY']
    data = load_data(tickers)
    if 'AAPL' not in data:
        print('Data download failed')
        return
    if args.no_train and os.path.exists('ppo_minervini.zip'):
        from stable_baselines3 import PPO
        agent = PPO.load('ppo_minervini.zip')
    else:
        agent = train_agent(data['AAPL'], timesteps=args.timesteps)
        agent.save('ppo_minervini.zip')
    signals = generate_signals(agent, data)
    for s in signals:
        print(s)
    user = os.getenv('GMAIL_USER')
    password = os.getenv('GMAIL_PASS')
    to = os.getenv('TO_EMAIL')
    if user and password and to:
        send_email(signals, user, password, to)


if __name__ == '__main__':
    main()
