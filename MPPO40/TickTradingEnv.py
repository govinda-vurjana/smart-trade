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
        self.mistake_count = 0  # Initialize mistake counter
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
        """Chooses an action and predicts time using the LSTM model output."""
        state = self.prepare_state()
        action_probs, time_prediction = model(state)  # Returns two outputs

        # Sample action
        action_dist = torch.distributions.Categorical(action_probs.squeeze(0))
        action = action_dist.sample().item()  # Either 0 (Buy) or 1 (Sell)
        
        # Convert time prediction to a positive number
        predicted_time = max(1, time_prediction.item())  # Ensure minimum 1 sec
        
        print(f"🎯 Selected Action: {action} (0=Buy, 1=Sell), Predicted Time: {predicted_time:.2f} seconds")

        return action, predicted_time




    def validator(self, action, predicted_time_period, close_price, actual_time_elapsed, model):
        """Evaluates the chosen action and predicted time period for reward calculation."""

        # Determine the true action (0 = up, 1 = down)
        true_action = 0 if self.entry_price < close_price else 1

        # Calculate price movement difference
        price_diff = abs(self.entry_price - close_price)

        # Time difference penalty (smaller is better)
        time_error = abs(predicted_time_period - actual_time_elapsed)

        # Assign reward based on action correctness and time accuracy
        if action == true_action:
            if time_error == 0:
                reward = 100 + (price_diff * 10)  # Max reward for perfect timing & correct action
            else:
                reward = 80 - (time_error * 2)  # Reduce reward if timing is off
        else:
            reward = -80 * price_diff  # Wrong action gets strong penalty

        # Apply extra penalty if time prediction is too far off
        if time_error > predicted_time_period * 0.5:  # If more than 50% deviation
            reward -= 30

        # Track mistake count and apply decay if repeated mistakes occur
        if action != true_action or time_error > predicted_time_period * 0.5:
            self.mistake_count += 1
        else:
            self.mistake_count = max(0, self.mistake_count - 1)  # Reduce mistakes over time

        reward *= 0.99 ** self.mistake_count  # Gradually decrease rewards for repeated mistakes

        # ✅ Track Cumulative Profit
        self.cumulative_profit += reward
        self.total_trades += 1
        self.correct_trades += 1 if action == true_action and time_error == 0 else 0
        accuracy = (self.correct_trades / self.total_trades) * 100 if self.total_trades > 0 else 0

        # ✅ Log Evaluation
        self.log_evaluation(action, true_action, reward, accuracy)
        
        # ✅ Save Model Every 10 Trades
        if self.total_trades % 10 == 0:
            self.save_model(model)

        # ✅ Debugging Information
        print(f"Entry Price: {self.entry_price}")
        print(f"Closing Price: {close_price}")
        print(f"Predicted Action: {action}, True Action: {true_action}")
        print(f"Predicted Time Period: {predicted_time_period}, Actual Time Elapsed: {actual_time_elapsed}")
        print(f"Time Error: {time_error}")
        print(f"Price Difference: {price_diff}")
        print(f"Reward: {reward}")
        print(f"Mistake Count: {self.mistake_count}")
        print(f'✅ Validated: Action={action}, TrueAction={true_action}, Profit={reward}')

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
   
# ✅ PPO Model with Self-Attention
class PPO(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_heads=4):
        super(PPO, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

        # ✅ Self-Attention Layer
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)

        # ✅ Action prediction (Up/Down)
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Softmax(dim=-1)
        )

        # ✅ Time prediction
        self.time_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 6),
            nn.Softmax(dim=-1)
        )

        self.time_classes = torch.tensor([5, 10, 15, 30, 60, 120], dtype=torch.float32)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # LSTM output

        # Apply Self-Attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)

        # Use the last time step output for prediction
        final_out = attn_out[:, -1, :]

        # Compute action probabilities
        action_probs = self.action_head(final_out)
        action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)  # Normalize

        # Compute time prediction
        time_logits = self.time_head(final_out)
        time_prediction = self.time_classes[torch.argmax(time_logits, dim=-1)]

        return action_probs, time_prediction
# ✅ Convert Action to Click
position = {"down": (2336, 690), "up": (2336, 531)}
def click_button(action):
    print("Trade click button executed...")
    msg=action
    speak_in_background(f"Action of model is {msg}")
    if action == 0:
        pyautogui.click(*position['up'])
    elif action == 1:
        pyautogui.click(*position['down'])


# Predefined positions for each value
positions = {
    5: (2265, 376),
    10: (2349, 385),
    15: (2443, 378),
    30: (2260, 453),
    60: (2340, 452),
    120: (2447, 459),
}
def click_position(value,action):
    # time.sleep(10)
    if value not in positions:
        print("Invalid value. Please use one of [5, 10, 15, 30, 60, 120].")
        return
    
    # Click on the initial position
    initial_x, initial_y = 2347, 303
    pyautogui.click(initial_x, initial_y)
    time.sleep(1)  # Delay before clicking the second position

    # Click on the corresponding position based on the value
    x, y = positions[value]
    pyautogui.click(x, y)
    print(f"Clicked on initial position ({initial_x}, {initial_y}) and then ({x}, {y}) for value {value}.")
    click_button(action)







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
    action, time_period = env.get_action(model)

    # ✅ Execute Action
    # speak_in_background("Waiting 1 minute for reward validation")
    start_time = time.time()  # Start timer before clicking the button
    speak_in_background(f"Model Time Period is {time_period}")
    # click_button(action,time_period)
    click_position(time_period,action)
    time.sleep(time_period)
    actual_time = time.time() - start_time  # Calculate elapsed time in seconds


    # time.sleep(61)

    # ✅ Fetch closing price & compute reward
    closing_value = dataReader.getFutureValues()
    reward = env.validator(action,time_period ,closing_value,actual_time, model)




        # ✅ Compute Policy Loss
    optimizer.zero_grad()
    action_probs, time_prediction = model(state)
    # ✅ Define entropy properly
    # entropy = -torch.sum(action_probs * torch.log(action_probs + 1e-8))

    # Compute losses
    log_prob = torch.log(action_probs[0, action] + 1e-8)
    policy_loss = -log_prob * reward  # Reinforcement Learning loss
    # MSE Loss for time prediction
    true_time = torch.tensor([actual_time], dtype=torch.float32)  # Replace with true movement time
    time_loss = nn.MSELoss()(time_prediction, true_time)

    # Combine losses
    total_loss = policy_loss + 0.1 * time_loss  # Weight time loss lightly

    # Backpropagation
    total_loss.backward()
    optimizer.step()

    env.step()
    # env.render()
    speak_in_background("Waiting 2 more minutes for new data.")
    time.sleep(1)
    print("✅ Waiting for new data...")
