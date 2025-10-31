# DataReader.py
import csv
import pandas as pd
from datetime import datetime, timedelta
from FeatureEngineer import FeatureEngineer

class DataReader:
    def __init__(self,filename):
        """
        Initializes the DataReader instance.
        :param minutes: The duration in minutes to fetch the recent data.
        """
        self.filename = filename  # The file that stores the live data
        self.feature_set= FeatureEngineer(window=30)
     


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


    

    def split_alternating(self, df):
        df1, df2 = pd.DataFrame(), pd.DataFrame()
        
        for i in range(0, len(df), 12):
            if (i // 12) % 2 == 0:
                df1 = pd.concat([df1, df.iloc[i:i+12]])
            else:
                df2 = pd.concat([df2, df.iloc[i:i+12]])

        # Ensure both have exactly 60 rows
        df1 = df1.iloc[-60:] if len(df1) > 60 else df1
        df2 = df2.iloc[-60:] if len(df2) > 60 else df2

        return df1.reset_index(drop=True), df2.reset_index(drop=True)



    def fetch_recent_data(self,min,newfile):
        """
        Fetches data from the last `self.minutes` of data duration.
        :return: Pandas DataFrame containing the last `self.minutes` of data.
        """

        df = pd.read_csv(self.filename)  # Load raw tick data
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        df=self.resample_ohlc(df,5)
        df=df.reset_index()
        print(df.head(5))
        print("Resampling Completed of Tick data..")
        # df["Timestamp"] = pd.to_datetime(df["Timestamp (IST)"])  # Convert to datetime

        # print(df.head())
        # Convert to OHLC and save

        if min==3:
            recent_data=df.tail(36)
            # df1,df2=self.split_alternating(recent_data)
            recent_data.reindex()
            print("Doing Feature Engineering..")
            state = self.feature_set.compute_features(recent_data)
            print("Computed Feature Engineering State..")
            print(state.columns)
            print(state.shape)
            state.reindex()
            state.to_csv(newfile, index=False, mode='w', header=True)
            return state
        else:
            return None

    def getFutureValues(self):
        df = pd.read_csv(self.filename)
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        df=self.resample_ohlc(df,5)
        df=df.reset_index()
        future=df.tail(12)
        #now my future variable consists of 12 rows each represents 5sec of candle,
        # I need to select smallest Close Value and return it.
        closing_value=future.tail(1)
        print(closing_value)
        print(closing_value["Close"])
        print(type(closing_value["Close"]))
        print(closing_value["Close"].iloc[-1])
        return closing_value["Close"].iloc[-1]





