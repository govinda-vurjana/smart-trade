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
from speakmesage import speak_in_background

import time
import pyautogui
import pyperclip

class TradingEnv:
    def __init__(self):
        self.df = pd.DataFrame()
        self.current_step = 0
        self.balance = 1000  # ✅ Starting balance
        self.target_balance = 10000  # ✅ Target balance
        self.limiting_balance = 500  # ✅ Limit balance, if balance falls below, stop program
        self.entry_price = 0
        self.manual_feedback = None
        self.cumulative_profit = 0
        self.total_trades = 0
        self.correct_trades = 0
        self.consecutive_successful_trades = 0  # ✅ Track consecutive successful trades
        self.price_values = [70, 100, 120, 150, 180, 200, 250, 300, 350, 400, 450, 500]  # ✅ Price list
        self.current_price_index = 0  # ✅ Index to track the current price value

        # ✅ Create logs folder
        if not os.path.exists("logs"):
            os.makedirs("logs")

        # ✅ AutoEvaluation.csv file
        self.evaluation_file = "logs/AutoEvaluation.csv"
        if not os.path.exists(self.evaluation_file):
            pd.DataFrame(columns=["Step", "Action", "TrueAction", "Profit", "CumulativeProfit", "Accuracy", "Balance"]).to_csv(self.evaluation_file, index=False)

        # ✅ Checkpoint Paths
        self.checkpoint_path = "logs/model_checkpoint.pth"
    
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

    def update_data(self, new_data):
        self.df = new_data
        self.current_step = 0
    def step(self):
        self.current_step = 0

    def prepare_state(self):
        state = self.df.iloc[self.current_step:self.current_step+12].drop('Timestamp (IST)', axis=1)
        return state.values

    def validator(self, action, close_price, model):
        """ Validate the trade and update balance accordingly """

        f=self.df.tail(1)
        self.entry_price=f["Close"].iloc[-1]
        print("Entry Price Updated:",self.entry_price)
        # Determine true action (whether the model was correct or not)
        true_action = 1 if self.entry_price < close_price else 0

        reward = 0
        
        # If there's no open position, or the action is not "hold", adjust entry price

        if action == 2:
            # If action is "hold", we don't change entry_price or position.
            # You can optionally add logic here to penalize for holding too long or reward some other criteria.
            print(f"Holding position: Entry Price: {self.entry_price}, Current Price: {close_price}")
            reward = -1  # No reward for holding the position (could be customized based on conditions)

        # Reward Logic for successful or unsuccessful trade
        if action == true_action:
            self.consecutive_successful_trades += 1  # Increment consecutive successful trades
            # Ensure it doesn't exceed list bounds for price values
            price_value_index = min(self.consecutive_successful_trades - 1, len(self.price_values) - 1)
            reward = self.price_values[price_value_index]

            self.balance += reward
            self.correct_trades += 1
            print("Price value index",price_value_index)

            # Call autoPriceUpdate function with correct index
            self.autoPriceUpdate(self.price_values[price_value_index])
            print(self.price_values[price_value_index])

        else:
            reward = -80  # Unsuccessful trade deducts 80
            self.balance -= 80
            self.consecutive_successful_trades = 0  # Reset consecutive count on failure

            # Reset price value to default 70 on failure
            self.autoPriceUpdate(70)

        # Ensure balance never goes negative
        if self.balance <= 0:
            print("❌ Model Lost! Balance reached 0.")
            exit(0)

        # Ensure balance does not fall below limiting balance
        if self.balance < self.limiting_balance:
            print(f"❌ Model Stopped! Balance below limiting balance of {self.limiting_balance}.")
            exit(0)

        # Check if model wins (target balance reached)
        if self.balance >= self.target_balance:
            print("🎉 Model Wins! Balance reached target.")
            self.save_model(model)
            print(f"✅ Model saved at {self.checkpoint_path}")
            exit(0)

        # Track Cumulative Profit
        self.cumulative_profit += reward
        self.total_trades += 1
        accuracy = (self.correct_trades / self.total_trades) * 100

        # Log Evaluation
        self.log_evaluation(action, true_action, reward, accuracy)
        if self.total_trades % 10 == 0:
            self.save_model(model)

        print(f'✅ Validated: Action={action}, TrueAction={true_action}, Profit={reward}, Balance={self.balance}, Consecutive Success={self.consecutive_successful_trades}')
        return reward


    def log_evaluation(self, action, true_action, profit, accuracy):
        """ Log trade details including updated balance """
        log_df = pd.read_csv(self.evaluation_file)
        new_row = {
            "Step": self.total_trades,
            "Action": action,
            "TrueAction": true_action,
            "Profit": profit,
            "CumulativeProfit": self.cumulative_profit,
            "Accuracy": accuracy,
            "Balance": self.balance  # ✅ Log balance
        }
        new_row_df = pd.DataFrame([new_row])
        log_df = pd.concat([log_df, new_row_df], ignore_index=True)
        log_df.to_csv(self.evaluation_file, index=False)

    def autoPriceUpdate(self, value):
        """
        Clicks on the specified (x, y) coordinates, selects the existing content,
        and pastes the given number value to replace it.
        
        Args:
            value (int/float): The number to be pasted.
        """
        print("Auto Price Updated  {}",value)
        x = 2309
        y = 418
        pyautogui.click(x, y)  # Click on the given coordinates
        time.sleep(0.2)  # Short delay to ensure the field is active
        pyautogui.hotkey('ctrl', 'a')  # Select all existing content
        pyperclip.copy(str(value))  # Copy the new value to clipboard
        pyautogui.hotkey('ctrl', 'v')  # Paste the new value
        print(f"✅ Price updated to {value}")


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

# Initialize env and model globally
env = TradingEnv()
model = PPO(input_dim=12*39, output_dim=3)

# ✅ Load model checkpoint if exists
env.load_model(model)

optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_agent(df):
    
    dataReader = DataReader("live_data.csv")

    # ✅ Auto-load model if exists

    env.update_data(df)

    state = env.prepare_state()
    action, entropy = env.get_action(model)

    # ✅ Click the button based on action
    click_button(action)
    speak_in_background("Waiting for reward validation")

    time.sleep(60)

    # ✅ Fetch closing price
    closing_value = dataReader.getFutureValues()
    reward = env.validator(action, closing_value, model)
    time.sleep(1)

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
