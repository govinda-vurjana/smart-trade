import torch
import numpy as np
import pandas as pd
import time
from DataReader import DataReader
import pyautogui
from TickTradingEnv import TradingEnv

# ✅ Load the trained model
class PPO(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PPO, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, output_dim),
            torch.nn.Softmax(dim=-1)
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


# ✅ Set up trading environment
class TradingInference:
    def __init__(self):
        self.model_path = "logs/model_checkpoint.pth"
        self.env = TradingEnv()
        self.model = PPO(input_dim=12*39, output_dim=3)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.dataReader = DataReader()
        self.load_model()
    
    def load_model(self):
        if torch.cuda.is_available():
            self.model.load_state_dict(torch.load(self.model_path))
        else:
            self.model.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu')))
        print("✅ Model Loaded Successfully!")
    
    def save_model(self):
        torch.save(self.model.state_dict(), self.model_path)
        print("💾 Model Updated & Saved!")
    
    def infer(self, df):
        self.env.update_data(df)
        state = self.env.prepare_state()
        state = (state - state.mean()) / (state.std() + 1e-9)  # Normalize
        
        action_probs = self.model(torch.FloatTensor(state.flatten()))
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample().item()
        entropy = action_dist.entropy().mean()
        
        # ✅ Execute action
        click_button(action)
        
        time.sleep(60)  # Wait for market update
        closing_value = self.dataReader.getFutureValues()
        reward = self.env.validator(action, closing_value, self.model)
        
        # ✅ Learn from the inference
        self.optimizer.zero_grad()
        policy_loss = -torch.log(self.model(torch.FloatTensor(state.flatten()))[action]) * reward
        entropy_loss = -0.01 * entropy
        loss = policy_loss + entropy_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
        self.optimizer.step()
        
        self.save_model()
        self.env.render()
        print("✅ Inference Complete!")

