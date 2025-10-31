# DataReader.py
import csv
import pandas as pd
from datetime import datetime, timedelta

class DataReader:
    def __init__(self,filename):
        """
        Initializes the DataReader instance.
        :param minutes: The duration in minutes to fetch the recent data.
        """
        self.filename = filename  # The file that stores the live data
     


    def resample_ohlc(self,dataframe, timeframe):
        """
        Reads a DataFrame with timestamps, resamples data into the specified timeframe, and calculates OHLCV values.
        """
        # Ensure the Timestamp column is correctly converted
        dataframe["Timestamp (IST)"] = pd.to_datetime(dataframe["Timestamp"], format="%Y-%m-%d %H:%M:%S.%f", errors="coerce")
        
        # Drop invalid timestamps
        invalid_timestamps = dataframe["Timestamp (IST)"].isna().sum()
        dataframe.dropna(subset=["Timestamp (IST)"], inplace=True)

        # Convert Value column to numeric
        dataframe["Value"] = pd.to_numeric(dataframe["Value"], errors="coerce")
        dataframe.dropna(subset=["Value"], inplace=True)

        # Set Timestamp as index
        dataframe.set_index("Timestamp (IST)", inplace=True)

        # Compute resampled OHLCV data
        ohlcv_df = dataframe["Value"].resample(f"{timeframe}s").agg(["first", "max", "min", "last", "count"])
        ohlcv_df.columns = ["Open", "High", "Low", "Close", "Volume"]
        ohlcv_df.dropna(inplace=True)

        # Append data to CSV (create header if file doesn't exist)
        # ohlcv_df.to_csv(output_csv, mode="a", header=not os.path.exists(output_csv))

        # Generate summary report
        total_data_points = len(dataframe)
        earliest_time = dataframe.index.min()
        latest_time = dataframe.index.max()
        total_minutes = (latest_time - earliest_time).total_seconds() / 60 if total_data_points > 0 else 0
        num_resampled_intervals = len(ohlcv_df)

        print("\n📊 **Data Summary Report**")
        print(f"✅ Total data points: {total_data_points}")
        print(f"❌ Invalid timestamps dropped: {invalid_timestamps}")
        print(f"📅 Earliest timestamp: {earliest_time}")
        print(f"⏳ Latest timestamp: {latest_time}")
        print(f"⌛ Total data duration: {total_minutes:.2f} minutes")
        print(f"📈 Number of resampled intervals ({timeframe}s): {num_resampled_intervals}")
        # print(f"\n✅ Resampled OHLCV data saved to {output_csv}")

        return ohlcv_df



    def fetch_recent_data(self):
        """
        Fetches data from the last `self.minutes` of data duration.
        :return: Pandas DataFrame containing the last `self.minutes` of data.
        """

        df = pd.read_csv(self.filename)  # Load raw tick data
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        df=self.resample_ohlc(df,5)
        df=df.reset_index()
        # print(df.head(5))
        print("Resampling Completed of Tick data..")
       
        return df
      
 




