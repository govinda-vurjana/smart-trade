import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from trade import detect_trend


def extract_key_levels(ohlc):
    """
    Extracts Higher Highs (HH), Higher Lows (HL), Support, and Resistance levels from OHLC data.
    :param ohlc: DataFrame with 'timestamp', 'high', 'low', 'close' columns.
    :return: hh, hl, support_levels, resistance_levels (each as a list of (timestamp, price))
    """
    timestamps = np.array(ohlc['Timestamp'])
    high_series = np.array(ohlc['High'])
    low_series = np.array(ohlc['Low'])
    close_series = np.array(ohlc['Close'])
    
    # Auto-calculate order based on dataset size (5% of total rows, min 5)
    order = 20
    
    # Higher Highs (HH) and Higher Lows (HL)
    hh, hl = [], []
    for i in range(1, len(ohlc)):
        if high_series[i] > high_series[i - 1]:
            hh.append((timestamps[i], high_series[i]))
        if low_series[i] > low_series[i - 1]:
            hl.append((timestamps[i], low_series[i]))
    
    # Support & Resistance Levels using Local Extrema
    support_idx = argrelextrema(close_series, np.less, order=order)[0]
    resistance_idx = argrelextrema(close_series, np.greater, order=order)[0]
    
    support_levels = [(timestamps[i], close_series[i]) for i in support_idx]
    resistance_levels = [(timestamps[i], close_series[i]) for i in resistance_idx]
    
    return hh, hl, support_levels, resistance_levels

# Load OHLC data from CSV
# df = pd.read_csv("usd_inr_ohlc_5s.csv", parse_dates=["Timestamp"])

# # Ensure correct data types
# # df = df[['Timestamp', 'High', 'low', 'close']].dropna()

# # Call the function
# hh, hl, support, resistance = extract_key_levels(df)

# # Print results
# # print("Higher Highs:", hh)
# # print("Higher Lows:", hl)
# # print("Support Levels:", support)
# # print("Resistance Levels:", resistance)
# live_tick = (1, 91.0967)  # Latest tick data

# result = detect_trend(hh, hl, support, resistance, live_tick)
# print(result)  