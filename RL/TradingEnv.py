import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from collections import deque
import time
import os,csv
from datetime import datetime, timedelta
from DataReader import DataReader

# Function to convert tick data to OHLCV
def resample_to_ohlcv(df, interval):
    print(f"Resampling data to {interval}...")
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df.set_index('Timestamp', inplace=True)
    ohlcv = df['Value'].resample(interval).ohlc()
    ohlcv['Volume'] = df['Value'].resample(interval).count()
    ohlcv.dropna(inplace=True)
    print(f"Resampling complete. Number of rows after resampling: {ohlcv.shape[0]}")
    return ohlcv

# Define the RL Environment
class TradingEnv:
    def __init__(self, minutes):
        print(f"Initializing Trading Environment with last {minutes} minutes of data...")
        self.data_reader = DataReader(minutes)
        self.state_size = 5  # [open, high, low, close, volume]
        self.action_size = 3  # [hold, up, down]
        self.memory = deque(maxlen=2000)
        self.model = self.build_model()
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 32
        self.model_path = "trading_model.pth"
        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path))
            print("Loaded pre-trained model.")

    def build_model(self):
        print("Building model...")
        model = nn.Sequential(
            nn.Linear(self.state_size, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, self.action_size)
        )
        print("Model built successfully.")
        return model

    def load_data(self):
        print("Loading data...")
        raw_data = self.data_reader.fetch_recent_data()
        if raw_data is not None:
            print("Raw data fetched successfully.")
            raw_data = resample_to_ohlcv(raw_data, '5s')
            self.live_data = raw_data.iloc[:-12]
            self.one_min_data = resample_to_ohlcv(raw_data.iloc[-12:], '1T')
        else:
            print("Failed to load data.")

    def get_state(self):
        if self.live_data is None or len(self.live_data) == 0:
            print("No live data available. Returning zero state.")
            return np.zeros(self.state_size)
        latest_data = self.live_data.iloc[-1]
        state = latest_data[["open", "high", "low", "close", "Volume"]].values
        print(f"Current state: {state}")
        return np.array(state, dtype=np.float32)

    def get_reward(self):
        future_close = self.one_min_data.iloc[-1]["Value_close"]
        last_close = self.live_data.iloc[-1]["close"]
        reward = (future_close - last_close)
        print(f"Calculated reward: {reward}")
        return reward

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            action = random.randrange(self.action_size)
            print(f"Taking random action: {action}")
        else:
            act_values = self.model(torch.FloatTensor(state))
            action = torch.argmax(act_values).item()
            print(f"Taking predicted action: {action}")
        return action

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))
        print(f"Stored experience in memory. Memory size: {len(self.memory)}")

    def replay(self):
        if len(self.memory) < self.batch_size:
            print(f"Not enough samples for training. Memory size: {len(self.memory)}")
            return
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state in minibatch:
            target = reward + self.gamma * torch.max(self.model(torch.FloatTensor(next_state))).item()
            target_f = self.model(torch.FloatTensor(state))
            target_f[action] = target
            self.model.zero_grad()
            criterion = nn.MSELoss()
            loss = criterion(self.model(torch.FloatTensor(state)), target_f)
            loss.backward()
            optim.SGD(self.model.parameters(), lr=0.001).step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            print(f"Updated epsilon: {self.epsilon}")

    def save_model(self):
        torch.save(self.model.state_dict(), self.model_path)
        print("Model saved to disk.")

def train_agent():
    print("Training agent started...")
    env = TradingEnv(minutes=21)

    while True:
        env.load_data()
        state = env.get_state()
        if len(state) == 0:
            print("No sufficient data for prediction. Retrying...")
            time.sleep(60)
            continue

        action = env.act(state)
        next_state = env.get_state()
        reward = env.get_reward()

        env.remember(state, action, reward, next_state)
        env.replay()
        env.save_model()

        print(f"Action: {action}, Reward: {reward}")
        time.sleep(60)

# Function to calculate the duration of data in minutes
def calculate_data_duration(csv_filename):
    try:
        with open(csv_filename, 'r') as file:
            reader = csv.reader(file)
            next(reader)

            rows = list(reader)
            if rows:
                first_row = rows[0]
                last_row = rows[-1]
                
                first_timestamp = datetime.fromisoformat(first_row[0])
                last_timestamp = datetime.fromisoformat(last_row[0])

                time_diff = last_timestamp - first_timestamp
                data_duration_minutes = time_diff.total_seconds() / 60
                print(f"Data duration: {data_duration_minutes} minutes.")
                return data_duration_minutes
            else:
                return 0
    except FileNotFoundError:
        print(f"Error: {csv_filename} not found.")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 0

if __name__ == "__main__":
    while True:
        data_duration = calculate_data_duration("live_data.csv")
        print(f"Live data consists of {int(data_duration)} minutes of data.")
        
        if data_duration >= 21:
            print("Sufficient data available. Starting training...")
            train_agent()  # Start training
            break  # Exit the loop once training starts
        else:
            print("Insufficient data. Checking again in 1 minute...")
        
        time.sleep(60)  # Wait for 1 minute before checking again