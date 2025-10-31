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
        """Prepares the state using only timestamp and five other features."""
        
        selected_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        state = self.df[selected_columns].iloc[self.current_step:].copy()



        # Normalize timestamp (convert to seconds since epoch)
        state['timestamp'] = pd.to_datetime(state['timestamp']).astype(int) / 10**9  

        state = torch.FloatTensor(state.values)
        return state.unsqueeze(0)


    def get_prediction(self, model):
        """Predicts the next 1-minute closing price."""
        state = self.prepare_state()
        predicted_price = model(state).squeeze(0).item()  # Extract single float value
        print(f"📈 Predicted Closing Price: {predicted_price:.4f}")
        return predicted_price




    def validator(self, predicted_price, actual_price, model, tail):
        """Computes reward and evaluation metrics based on prediction accuracy."""
        error = abs(predicted_price - actual_price)  # Absolute error
        reward = -error  # Negative reward for larger errors

        # ✅ Compute Improvement Accuracy (relative error improvement)
        prev_error = getattr(self, "prev_error", None)
        improvement_accuracy = (prev_error - error) / prev_error * 100 if prev_error is not None and prev_error != 0 else None
        self.prev_error = error  # Store current error for next step comparison

        # ✅ Find Best Matching Row in Tail
        tail = tail.copy()

        tail['error'] = abs(tail['close'] - predicted_price)
        best_match_idx = tail['error'].idxmin()  # Index of closest match
        best_match_time_offset = (best_match_idx + 1) * 5 if best_match_idx is not None else 0

        # ✅ Track Cumulative Profit
        self.cumulative_profit += reward

        # ✅ Log Evaluation
        self.log_evaluation(predicted_price, actual_price, error, reward, improvement_accuracy, best_match_time_offset)

        print(f"Predicted Price: {predicted_price}")
        print(f"Actual Closing Price: {actual_price}")
        print(f"Prediction Error: {error:.4f}")
        print(f"Reward: {reward:.4f}")
        if improvement_accuracy is not None:
            print(f"Improvement Accuracy: {improvement_accuracy:.2f}%")
        print(f"Best Match Time Offset: {best_match_time_offset} sec")

        return reward


    def log_evaluation(self, predicted_price, actual_price, error, reward, improvement_accuracy, best_match_time_offset):
        """Logs trading predictions and performance metrics."""
        log_df = pd.read_csv(self.evaluation_file)

        new_row = {
            "Step": self.total_trades,
            "PredictedPrice": predicted_price,
            "ActualPrice": actual_price,
            "Error": error,
            "Reward": reward,
            "ImprovementAccuracy": improvement_accuracy if improvement_accuracy is not None else "N/A",
            "CumulativeProfit": self.cumulative_profit,
            "BestMatchTimeOffset": best_match_time_offset
        }

        new_row_df = pd.DataFrame([new_row])
        log_df = pd.concat([log_df, new_row_df], ignore_index=True)
        log_df.to_csv(self.evaluation_file, index=False)


    def update_data(self, new_data):
        """Updates the dataset and resets step counter, keeping only required features."""
        self.df = new_data[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
        self.current_step = 0


    def save_model(self, model):
        torch.save(model.state_dict(), self.checkpoint_path)
        print(f"💾 Model Checkpoint Saved! [Market: {self.market_identifier}]")

    def load_model(self, model):
        if os.path.exists(self.checkpoint_path):
            model.load_state_dict(torch.load(self.checkpoint_path))
            print(f"🔄 Model Checkpoint Loaded Successfully for {self.market_identifier}")
   

class PPO(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=128):
        super(PPO, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Output a single value (predicted closing price)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])  # Should output (1,)

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
def train_agent(df,tail, env, model, optimizer):
    global global_env, global_model
    #you can get the entry price from df tail.
    df = df.copy()  # Ensure we're working with a copy
    df['timestamp'] = pd.to_datetime(df['timestamp']).astype(int) / 10**9  # Convert to seconds
    f= df.tail(1)
    env.entry_price =  f["close"].iloc[-1]
    #Make them accessible in signal handler

    # ✅ Assign global variables for signal handling
    global_env = env
    global_model = model

    #divide the dataframe here and give to 3 analysers short term - 3min, Medium term - 9min, long term 15min.
    env.update_data(df)

    # ✅ Get State
    predicted_price = env.get_prediction(model)

    # ✅ Get Action

    # ✅ Execute Action
    speak_in_background("Waiting 1 minute for reward validation")
    # click_button(action)
    # time.sleep(61)

    # ✅ Fetch closing price & compute reward
    closing_value = tail['close'].iloc[-1]
    reward = env.validator(predicted_price, closing_value, model,tail)

    # ✅ Compute Policy Loss
    predicted_price_tensor = model(env.prepare_state()).squeeze(0)
    actual_price_tensor = torch.tensor(closing_value, dtype=torch.float32)

    loss = nn.MSELoss()(predicted_price_tensor, actual_price_tensor)  # Mean Squared Error

    loss.backward()
    print(f"Loss: {loss.item():.6f}")
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
    optimizer.step()
    env.step()
    print("✅ New Data Processed")
    # env.render()
    # speak_in_background("Waiting 2 more minutes for new data.")
    # time.sleep(119)
    env.save_model(model)
    print("Model saved successfully.")
    print("✅ New Data")
