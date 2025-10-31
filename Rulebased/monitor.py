import csv
import time
import keyboard
from datetime import datetime
import pandas as pd
from DataReader import DataReader
from TrendAnalyser import TrendAnalyzer
import pyautogui


# ✅ Convert Action to Click
positions = {"down": (2336, 690), "up": (2336, 531)}
def click_button(action):
    print("Trade click button executed...")
    if action == 0:
        pyautogui.click(*positions['up'])
    elif action == 1:
        pyautogui.click(*positions['down'])

# ✅ Function Called When Data is Ready
def data_ready_for_prediction(dataReader):
    print("✅ Data is ready for RL.")
    
    df = dataReader.fetch_recent_data()
    #this df consists the last row entry price or open price.
    
    if not df.empty:
        print(df.columns)
        print(df.shape)
        print("🧠 Got Agent State set..")
        #pass the df containing ohlcv data to the Trade Analyser
                
        # Convert to list of tuples (Open, High, Low, Close, Volume)
        candles = list(df[["Open", "High", "Low", "Close", "Volume"]].itertuples(index=False, name=None))

        # Call the trend predictor
        trend, probability = TrendAnalyzer.predict_trend(candles)

        # Print result
        print(f"Trend: {'Uptrend' if trend == 1 else 'Downtrend' if trend == 0 else 'Uncertain'}")
        print(f"Probability: {probability:.2f}%")
        click_button(trend)


      
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

dataReader=DataReader(csv_filename)


while True:
    data_duration = calculate_data_duration(csv_filename)
    # print(f"⏳ Live data consists of {int(data_duration)} minutes of data.")

    if data_duration >= 3:
        data_ready_for_prediction(dataReader)
        time.sleep(60)
