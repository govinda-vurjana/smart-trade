import csv
import time
import keyboard
from datetime import datetime
import pandas as pd
from DataReader import DataReader
from levelizer import extract_key_levels
from trade import detect_trend
import pyautogui
from speakmessage import speak_in_background


# # ✅ Available Markets
# markets = {
#     1: "usd_inr",
#     2: "usd_pkr",
#     3: "usd_brl",
#     4: "nzd_cad",
#     5: "usd_dzd"
# }

# # ✅ Market Selection
# def select_market():
#     print("\nSelect a target market:")
#     for num, market in markets.items():
#         print(f"{num}. {market.upper()}")  
    
#     while True:
#         try:
#             choice = int(input("Enter the market number (1-5): "))
#             if choice in markets:
#                 print(f"✅ Selected Market: {markets[choice].upper()}")
#                 return markets[choice]
#             else:
#                 print("❌ Invalid choice. Please enter a number between 1-5.")
#         except ValueError:
#             print("❌ Invalid input. Please enter a number.")

# # ✅ Get Selected Market
# market_identifier = select_market()


# ✅ Convert Action to Click
positions = {"Down": (2336, 690), "Up": (2336, 531)}

def click_button(action):
    print("Trade click button executed...")
    speak_in_background(f"Action of model is {action}")
    
    if action in positions:
        pyautogui.click(*positions[action])



# ✅ Function Called When Data is Ready
def data_ready_for_prediction(min, dataReader, file):
    print("✅ Data is ready for RL.")
    speak_in_background("DATA ready")
    
    df = dataReader.fetch_recent_data(min, file)
    #this df consists the last row entry price or open price.
    
    if not df.empty:
        print(df.columns)
        print(df.shape)
        hh, hl, support, resistance = extract_key_levels(df)
        # now give the above hh,hl etc data to the detect trend,
        time.sleep(2)
        live_tick=dataReader.getCurrentTicks()
        print(live_tick)
        result = detect_trend(hh, hl, support, resistance, live_tick)

        trend = result["trend"]
        probability = result["probability"]

        print(trend)       # Output: Down
        print(probability) # Output: 0.92

        speak_in_background(probability)
        # if probability> 0.5:
        click_button(trend)
        # time.sleep(60)
        # speak_in_background("Woke up")
        
        # ✅ Train Agent with Persistent Environment & Model
    else:
        print("❌ No State Info received..")

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

# print(f"📡 Monitoring {csv_filename} for market {market_identifier.upper()}...")

while True:
    data_duration = calculate_data_duration(csv_filename)
    print(f"⏳ Live data consists of {int(data_duration)} minutes of data.")

    if data_duration >= 3:
        data_ready_for_prediction(3, dataReader, oneminfile)
        time.sleep(60)
        

    
