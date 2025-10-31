import gym
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import os,time
import matplotlib.pyplot as plt
import pyautogui
import signal
from DataReader import DataReader

class TradingEnv:
    def __init__(self):
        self.df = pd.DataFrame()
        self.current_step = 0
        self.balance = 10000
        self.position = None
        self.entry_price = 0
        self.manual_feedback = None
        self.cumulative_profit = 0
        self.total_trades = 0
        self.correct_trades = 0

        # ✅ Create logs folder
        if not os.path.exists("logs"):
            os.makedirs("logs")

        # ✅ AutoEvaluation.csv file
        self.evaluation_file = "logs/AutoEvaluation.csv"
        if not os.path.exists(self.evaluation_file):
            pd.DataFrame(columns=["Step", "Action", "TrueAction", "Profit", "CumulativeProfit", "Accuracy"]).to_csv(self.evaluation_file, index=False)

        # ✅ Checkpoint Paths
        self.checkpoint_path = "logs/model_checkpoint.pth"

    def update_data(self, new_data):
        self.df = new_data
        self.current_step = 0
    def step(self):
        self.current_step = 0

    def prepare_state(self):
        state = self.df.iloc[self.current_step:self.current_step+12].drop('Timestamp (IST)', axis=1)
        return state.values


    def get_action(self, model):
        state = self.prepare_state().flatten()


        # ✅ Normalize state to prevent high gradients
        state = (state - state.mean()) / (state.std() + 1e-9)

        action_probs = model(torch.FloatTensor(state))

        # ✅ Prevent NaN in action probs
        if torch.isnan(action_probs).any():
            print("⚠️ NaN Detected. Resetting probabilities...")
            action_probs = torch.ones(3) / 3

        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample().item()
        entropy = action_dist.entropy().mean()

        return action, entropy
    
    


    def validator(self, action,close_price,model):
        
        if self.entry_price < close_price:
            true_action=1
        else:
            true_action=0

        # true_action = self.pause_for_feedback()
        reward = 0

        # ✅ Automatically adjust entry price
        if self.position is None or action != 2:
            self.entry_price = close_price
            self.position = action

        # ✅ Reward logic
        if action == 2:  # HOLD
            price_change = close_price - self.entry_price
            reward = price_change * 0.3
        else:
            if action == true_action:
                reward = abs(close_price - self.entry_price)
                self.balance += reward
                self.correct_trades += 1
            else:
                reward = -abs(close_price - self.entry_price)
                self.balance += reward

        # ✅ Track Cumulative Profit
        self.cumulative_profit += reward
        self.total_trades += 1
        accuracy = (self.correct_trades / self.total_trades) * 100

        # ✅ Log Evaluation
        self.log_evaluation(action, true_action, reward, accuracy)

        # ✅ Auto-Save Model
        if self.total_trades % 10 == 0:
            self.save_model(model)

        print(f'✅ Validated: Action={action}, TrueAction={true_action}, Profit={reward}')
        return reward

    def log_evaluation(self, action, true_action, profit, accuracy):
        log_df = pd.read_csv(self.evaluation_file)
        new_row = {
            "Step": self.total_trades,
            "Action": action,
            "TrueAction": true_action,
            "Profit": profit,
            "CumulativeProfit": self.cumulative_profit,
            "Accuracy": accuracy
        }
        new_row_df = pd.DataFrame([new_row])
        log_df = pd.concat([log_df, new_row_df], ignore_index=True)
        log_df.to_csv(self.evaluation_file, index=False)

    def save_model(self, model):
        torch.save(model.state_dict(), self.checkpoint_path)
        print("💾 Model Checkpoint Saved! [EVERY TRADE]")

    def load_model(self, model):
        if os.path.exists(self.checkpoint_path):
            model.load_state_dict(torch.load(self.checkpoint_path))
            print("🔄 Model Checkpoint Loaded Successfully!")

    def render(self):
        print(f'Step: {self.current_step}, Balance: {self.balance}')


class PPO(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PPO, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.model(x)


# ✅ Convert Action to Click
positions = {"down": (2336, 690), "up": (2336, 531)}
def click_button(action):
    print("Trade click button executed...")

    if action == 1:
        pyautogui.click(*positions['up'])
    elif action == 2:
        pyautogui.click(*positions['down'])


# ✅ Handle Force Exit (Ctrl+C only)
def signal_handler(sig, frame):
    print("\n💾 Saving Model Before Force Exit...")
    env.save_model(model)
    print("✅ Model Saved. Exiting Now.")
    exit(0)

# ✅ Attach Signal Handler (Windows Compatible)
signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C


# ✅ Train the Agent
import torch.optim as optim
import time
from DataReader import DataReader

def train_agent(df):
    global env, model
    env = TradingEnv()
    model = PPO(input_dim=12*20, output_dim=3)
    dataReader = DataReader()

    # ✅ Auto-load model if exists
    env.load_model(model)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    env.update_data(df)

    state = env.prepare_state()
    action, entropy = env.get_action(model)

    # ✅ Click the button based on action
    click_button(action)
    time.sleep(60)

    # ✅ Fetch closing price
    closing_value = dataReader.getFutureValues()
    reward = env.validator(action, closing_value, model)

    # ✅ Policy loss
    optimizer.zero_grad()
    policy_loss = -torch.log(model(torch.FloatTensor(state).flatten())[action]) * reward
    entropy_loss = -0.01 * entropy
    loss = policy_loss + entropy_loss

    # ✅ Prevent gradient explosion
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
    
    env.step()

    env.render()
    print("✅ Waiting for new data...")
