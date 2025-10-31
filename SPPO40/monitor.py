import time
import pandas as pd
from tqdm import tqdm  # Real-time progress bar
from DataReader import DataReader
from TickTradingEnv import train_agent, TradingEnv, PPO
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
            choice = int(input("Enter the market number (1-6): "))
            if choice in markets:
                print(f"✅ Selected Market: {markets[choice].upper()}")
                return markets[choice]
            print("❌ Invalid choice. Please enter a number between 1-6.")
        except ValueError:
            print("❌ Invalid input. Please enter a number.")

# ✅ Get Selected Market
market_identifier = select_market()

# ✅ Initialize Environment & Model (Persistent)
env = TradingEnv(market_identifier)
model = PPO(input_dim=6)  
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ✅ Load Model Checkpoint (if available)
env.load_model(model)

# ✅ Function Called When Data is Ready
def data_ready_for_prediction(min, dataReader, df):
    # print("✅ Data ready for RL. Resampling and feature engineering...")

    state, tail = dataReader.fetch_recent_data(min, df)

    if not state.empty:
        print("📊 Training RL Agent...")
        train_agent(state, tail, env, model, optimizer)
    else:
        print("⚠️ No State Info received.")

# ✅ Path to the CSV file
file_path = r"cleaned_file.csv"

# ✅ Read CSV
df = pd.read_csv(file_path)

# ✅ Define chunk size (48 rows = 4 minutes)
chunk_size = 48
dataReader = DataReader()

total_chunks = (len(df) + chunk_size - 1) // chunk_size  # Total number of chunks
start_time = time.time()

# ✅ Iterate through DataFrame in 48-row chunks
for start in tqdm(range(0, len(df), chunk_size), desc="📊 Processing Data", unit=" chunk"):
    chunk_df = df.iloc[start:start + chunk_size]  # Select 48-row chunk
    
    # Print detailed logs occasionally
    if start % (chunk_size * 5) == 0:  
        print(f"\n📌 Processing rows {start} to {start + chunk_size - 1}")
    
    # ✅ Process Data Chunk
    data_ready_for_prediction(3, dataReader, chunk_df)

    # Print elapsed time occasionally
    if start % (chunk_size * 10) == 0:  
        elapsed_time = time.time() - start_time
        print(f"⏳ Elapsed Time: {elapsed_time:.2f} sec | 🔄 Chunks Remaining: {total_chunks - (start // chunk_size + 1)}")

# ✅ Final Summary
total_elapsed_time = time.time() - start_time
print(f"\n🎉 All data processed successfully in {total_elapsed_time:.2f} seconds!")
