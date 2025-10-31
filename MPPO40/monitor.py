import csv
import time
import keyboard
from datetime import datetime
import pandas as pd
from DataReader import DataReader
from TickTradingEnv import train_agent, TradingEnv, PPO
from speakmessage import speak_in_background
import torch.optim as optim
from TCPClient import TCPClient

# ✅ Available Markets
markets = {
    1: "usd_inr",
    2: "usd_pkr",
    3: "usd_brl",
    4: "nzd_cad",
    5: "usd_dzd",
    6: "eur_usd"
}

# ✅ Market Selection
def select_market():
    print("\nSelect a target market:")
    for num, market in markets.items():
        print(f"{num}. {market.upper()}")  
    
    while True:
        try:
            choice = int(input("Enter the market number (1-5): "))
            if choice in markets:
                print(f"✅ Selected Market: {markets[choice].upper()}")
                return markets[choice]
            else:
                print("❌ Invalid choice. Please enter a number between 1-5.")
        except ValueError:
            print("❌ Invalid input. Please enter a number.")

# ✅ Get Selected Market
market_identifier = select_market()

# ✅ Initialize Environment & Model **(Persistent)**
env = TradingEnv(market_identifier)

# Define model parameters
input_dim = 39
hidden_dim = 128

# Create PPO model instance
model = PPO(input_dim=input_dim, hidden_dim=128, num_heads=4)

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ✅ Load Model Checkpoint (if available)
env.load_model(model)

# ✅ Function Called When Data is Ready
def data_ready_for_prediction(min, dataReader, file):
    print("✅ Data is ready for RL.")
    print("📊 Now resampling and doing feature engineering...")
    
    df = dataReader.fetch_recent_data(min, file)
    #this df consists the last row entry price or open price.
    
    if not df.empty:
        print(df.columns)
        print(df.shape)
        print("🧠 Got Agent State set..")
        speak_in_background("Training Agent Started")
        
        # ✅ Train Agent with Persistent Environment & Model
        train_agent(df, env, model, optimizer)
    else:
        print("❌ No State Info received..")
        speak_in_background("No State Info received..")

# ✅ Calculate Data Duration in Minutes
def calculate_data_duration(csv_filename):
    try:
        with open(csv_filename, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header
            
            rows = list(reader)
            if rows:
                first_timestamp = datetime.fromisoformat(rows[0][0])
                last_timestamp = datetime.fromisoformat(rows[-1][0])

                return (last_timestamp - first_timestamp).total_seconds() / 60
            else:
                return 0
    except FileNotFoundError:
        print(f"❌ Error: {csv_filename} not found.")
        return 0
    except Exception as e:
        print(f"❌ Error: {e}")
        return 0

# ✅ Monitor the live_data.csv file
csv_filename = 'live_data.csv'
oneminfile = "sampled5s.csv"
dataReader = DataReader(csv_filename)

print(f"📡 Monitoring {csv_filename} for market {market_identifier.upper()}...")

while True:
    data_duration = calculate_data_duration(csv_filename)
    print(f"⏳ Live data consists of {int(data_duration)} minutes of data.")

    if data_duration >= 3:
        data_ready_for_prediction(3, dataReader, oneminfile)
