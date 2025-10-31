import pandas as pd
import plotly.graph_objects as go

def resample_to_ohlc(input_csv, output_csv):
    """Resamples tick data into 1-minute OHLC format for the last 30 minutes."""
    # Load CSV file
    df = pd.read_csv(input_csv)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
    df.dropna(inplace=True)

    # Filter last 30 minutes
    last_timestamp = df['Timestamp'].max()
    start_time = last_timestamp - pd.Timedelta(minutes=20)
    df = df[df['Timestamp'] >= start_time]

    # Set timestamp as index
    df.set_index('Timestamp', inplace=True)
    
    # Resample to 1-minute OHLC
    ohlc = df['Value'].resample('5s').agg(['first', 'max', 'min', 'last'])
    ohlc.columns = ['Open', 'High', 'Low', 'Close']
    ohlc.dropna(inplace=True)
    
    # Save to CSV
    ohlc.to_csv(output_csv)
    return ohlc



# # Example usage
# ohlc_csv = "usd_inr_ohlc.csv"
# resample_to_ohlc("usd_inr.csv", "usd_inr_ohlc_5s.csv")
