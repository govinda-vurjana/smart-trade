import pandas as pd
from datetime import datetime

class Resampler:
    def __init__(self, df, minutes):
        """
        Initializes the Resampler class with a DataFrame and a minute value.
        :param df: The input DataFrame with 'Timestamp', 'Value', and 'Volume' columns.
        :param minutes: The number of minutes for the timeframe.
        """
        # Store the input dataframe and minutes value
        self.df = df
        self.minutes = minutes
        self.timestamp_format = "%Y-%m-%d %H:%M:%S"
        
        # Generate the current timestamp to append to filenames
        self.current_timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    def resample_to_ohlcv(self, timeframe):
        """
        Resample the DataFrame to OHLCV values (Open, High, Low, Close, Volume).
        :param timeframe: The timeframe for resampling, e.g., '5s', '15s', '30s', '1Min'.
        :return: A DataFrame with OHLCV data.



        """
        print(self.df.head(5))
        print(self.df.columns)
        print(self.df.dtypes)
        print(self.df.shape)
        # print(self.df['Timestamp'])
        

        
        

        # # Resample the data for both 'Value' and calculate 'Volume' as the number of ticks (size)
        ohlcv_df = self.df[['Value']].resample(timeframe).agg({
            'Value': 'ohlc'  # Open, High, Low, Close for 'Value'
        })
        
        # # Calculate volume as the count of ticks in each period (i.e., number of rows per timeframe)
        volume_df = self.df.resample(timeframe).size()

        # # Add the 'Volume' column to the ohlcv DataFrame
        ohlcv_df['Volume'] = volume_df

        # # Flatten the MultiIndex if present
        ohlcv_df.columns = ['Value_open', 'Value_high', 'Value_low', 'Value_close', 'Volume']


        return ohlcv_df
    


    def save_to_csv(self, df, timeframe):
        """
        Save the OHLCV DataFrame to a CSV file.
        :param df: The DataFrame to be saved.
        :param timeframe: The timeframe for the resampling.
        :return: The filename of the saved CSV.
        """
        # Define the filename based on provided criteria
        filename = f"{self.minutes}min_{self.current_timestamp}_{timeframe}_ohlcv.csv"
        
        # Save the DataFrame to a CSV file
        df.to_csv(filename)
        
        return filename

    def process_resampling(self):
        """
        Process resampling for multiple timeframes.
        :return: A list of filenames where the OHLCV data has been saved.
        """
        filenames = []
        
        # Define the timeframes you want to process
        timeframes = ['5S']
             # Ensure 'Timestamp' is in datetime format
        self.df['Timestamp'] = pd.to_datetime(self.df['Timestamp'])

        # # Set 'Timestamp' as the index for resampling
        self.df.set_index('Timestamp', inplace=True)
       

        # # Convert the 'Value' column to numeric, ensuring any errors are coerced to NaN
        self.df['Value'] = pd.to_numeric(self.df['Value'], errors='coerce')

        # Process each timeframe
        for timeframe in timeframes:
            # Resample data to OHLCV for the given timeframe
            ohlcv_data = self.resample_to_ohlcv(timeframe)
            
            # Save the OHLCV data to CSV
            filename = self.save_to_csv(ohlcv_data, timeframe)
            
            # Append the filename to the list of filenames
            filenames.append(filename)
        
        return filenames
