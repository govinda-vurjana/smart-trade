import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import pyautogui
import time
import signal
from speakmessage import speak_in_background
from DataReader import DataReader



# ✅ Global Variables for Signal Handling
global_env = None
global_model = None


# ✅ Handle Force Exit (Ctrl+C)
def signal_handler(sig, frame):
    """Ensures the model is saved before exiting when Ctrl+C is pressed."""
    global global_env, global_model
    print("\n💾 Saving Model Before Force Exit...")

    if global_env and global_model:
        global_env.save_model(global_model)
        print("✅ Model Saved Successfully!")

    print("🚪 Exiting Program...")
    exit(0)

# ✅ Attach Signal Handler
signal.signal(signal.SIGINT, signal_handler)  # Handles Ctrl+C interruption

# ✅ Trading Environment
class TradingEnv:
    def __init__(self, market_identifier):
        self.market_identifier = market_identifier  # Store market identifier
        self.df = pd.DataFrame()  
        self.current_step = 0
        self.balance = 10000
        self.entry_price = 0
        self.cumulative_profit = 0
        self.total_trades = 0
        self.correct_trades = 0

        # ✅ Create logs folder
        if not os.path.exists("logs"):
            os.makedirs("logs")

        # ✅ Market-Specific Model Paths
        self.checkpoint_path = f"logs/model_{market_identifier}.pth"

        # ✅ AutoEvaluation.csv file (market-specific)
        self.evaluation_file = f"logs/AutoEvaluation_{market_identifier}.csv"
        if not os.path.exists(self.evaluation_file):
            pd.DataFrame(columns=["Step", "Action", "TrueAction", "Profit", "CumulativeProfit", "Accuracy"]).to_csv(self.evaluation_file, index=False)

    def step(self):
        """Moves to the next trading step."""
        self.current_step = 0

    def prepare_state(self):
        """Returns the state in (1, seq_length, 39) format."""
        state = self.df.iloc[self.current_step:].drop(columns=['Timestamp (IST)'], errors='ignore')
        state = torch.FloatTensor(state.values)
        return state.unsqueeze(0)

    def get_action(self, model):
        """Chooses an action based on LSTM model output."""
        state = self.prepare_state()

        # Get action probabilities
        action_probs = model(state)  # Should return (1,2)

        # 🔍 Debugging: Check the shape of action_probs
        print("🔍 action_probs shape before squeeze:", action_probs.shape)  

        # Ensure it's a 1D tensor
        action_probs = action_probs.squeeze(0)  # Convert from (1,2) to (2,)

        # 🔍 Debugging: Check shape again
        print("🔍 action_probs shape after squeeze:", action_probs.shape)  

        # Prevent NaN values
        if torch.isnan(action_probs).any():
            print("⚠️ NaN Detected. Resetting probabilities...")
            action_probs = torch.ones(2) / 2  # [0.5, 0.5]

        # ✅ Corrected indexing for 1D tensor
        print(f"Action Probabilities: Buy = {action_probs[0].item():.4f}, Sell = {action_probs[1].item():.4f}")

        # Sample an action
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample().item()  # Either 0 (Buy) or 1 (Sell)

        print(f"🎯 Selected Action: {action} (0=Buy, 1=Sell)")
        entropy = action_dist.entropy().mean()

        return action, entropy



    def append_trade_data_to_csv(entry_price, close_price, action, true_action, df, csv_filename="trade_data.csv"):
        """
        Appends trade data to a CSV file, along with the dataframe's necessary columns.
        
        Parameters:
        entry_price (float): The entry price for the trade.
        close_price (float): The closing price for the trade.
        action (int): The action taken (0 for Buy, 1 for Sell).
        true_action (int): The true action (0 for Buy, 1 for Sell).
        df (pd.DataFrame): The dataframe containing trade data.
        csv_filename (str): The filename of the CSV file where the data will be appended. Default is "trade_data.csv".
        """
        
        # Create a new DataFrame to include the necessary columns
        df_to_save = df[["Close", "Open", "Low", "Close", "Timestamp"]].copy()
        df_to_save["Entry Price"] = entry_price
        df_to_save["Closing Price"] = close_price
        df_to_save["Action"] = action
        df_to_save["True Action"] = true_action
        
        # Append the data to the CSV file
        df_to_save.to_csv(csv_filename, mode='a', header=False, index=False)
        
        print(f"Data appended to {csv_filename}")


    def validator(self, action, close_price, model):
        """Evaluates the chosen action and computes reward."""
        # Determine correct action
        #0 means up
        #1 means down
        
        if self.entry_price < close_price:
            true_action = 0
        else:
            true_action = 1

        #Debug entry price and closing price.
      

        # true_action = 1 if self.entry_price < close_price else 0
        speak_in_background(f"True Action is {true_action}")

        reward = 100 if action == true_action else -80

        # ✅ Track Cumulative Profit
        self.cumulative_profit += reward
        self.total_trades += 1
        accuracy = (self.correct_trades / self.total_trades) * 100 if self.total_trades > 0 else 0

        # ✅ Log Evaluation
        self.log_evaluation(action, true_action, reward, accuracy)
        if self.total_trades % 10 == 0:
            self.save_model(model)
        print(f"Entry Price: {self.entry_price}")
        print(f"Closing Price: {close_price}")
        print(f"Action: {action}")
        print(f"True Action: {true_action}")


        print(f'✅ Validated: Action={action}, TrueAction={true_action}, Profit={reward}')
        # self.append_trade_data_to_csv(self.entry_price, close_price, action, true_action, self.df)

        return reward

    def log_evaluation(self, action, true_action, profit, accuracy):
        """Logs trading actions and performance."""
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

    def update_data(self, new_data):
        self.df = new_data
        self.current_step = 0

    def save_model(self, model):
        torch.save(model.state_dict(), self.checkpoint_path)
        print(f"💾 Model Checkpoint Saved! [Market: {self.market_identifier}]")

    def load_model(self, model):
        if os.path.exists(self.checkpoint_path):
            model.load_state_dict(torch.load(self.checkpoint_path))
            print(f"🔄 Model Checkpoint Loaded Successfully for {self.market_identifier}")
   

class PPO(nn.Module):
    def __init__(self, input_dim, output_dim=2, hidden_dim=128):  # Ensure output_dim=2
        super(PPO, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),  # Only 2 outputs now
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  
        return self.fc(lstm_out[:, -1, :])  # Should output (1,2)


# ✅ Convert Action to Click
positions = {"down": (2336, 690), "up": (2336, 531)}
def click_button(action):
    print("Trade click button executed...")
    msg=action
    speak_in_background(f"Action of model is {msg}")
    if action == 0:
        pyautogui.click(*positions['up'])
    elif action == 1:
        pyautogui.click(*positions['down'])


# ✅ Train the Agent
def train_agent(df, env, model, optimizer):
    global global_env, global_model
    #you can get the entry price from df tail.
    f= df.tail(1)
    env.entry_price =  f["Close"].iloc[-1]
    #Make them accessible in signal handler

    # ✅ Assign global variables for signal handling
    global_env = env
    global_model = model

    dataReader = DataReader("live_data.csv")
    #divide the dataframe here and give to 3 analysers short term - 3min, Medium term - 9min, long term 15min.
    df=df.tail(36)
    env.update_data(df)

    # ✅ Get State
    state = env.prepare_state()

    # ✅ Get Action
    action, entropy = env.get_action(model)

    # ✅ Execute Action
    speak_in_background("Waiting 1 minute for reward validation")
    click_button(action)
    time.sleep(61)

    # ✅ Fetch closing price & compute reward
    closing_value = dataReader.getFutureValues()
    reward = env.validator(action, closing_value, model)

    # ✅ Compute Policy Loss
    optimizer.zero_grad()
    action_probs = model(state)
    log_prob = torch.log(action_probs[0, action] + 1e-8)  # Prevent log(0)
    loss = -log_prob * reward - 0.01 * entropy

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
    optimizer.step()

    env.step()
    # env.render()
    speak_in_background("Waiting 2 more minutes for new data.")
    time.sleep(119)
    print("✅ Waiting for new data...")
