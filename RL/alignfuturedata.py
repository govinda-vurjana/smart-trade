import pandas as pd

# Load live data (240 rows for 20 minutes = 5 sec candles)
live_data = pd.read_csv('live_data.csv', parse_dates=['Timestamp'])

# Load 1-minute future data (1 row for 21st minute)
one_min_data = pd.read_csv('1mindata.csv', parse_dates=['Timestamp'])


# Aligning future 1-min data with the live data
live_data['future_close'] = one_min_data['Value_close'].values[0]

# Calculate price change: 1 if future price is higher, 0 if future price is lower or equal
live_data['price_change'] = (live_data['future_close'] > live_data['Value_close']).astype(int)

# Show the aligned data with future close and price change
print(live_data[['Timestamp', 'Value_close', 'future_close', 'price_change']].head())
