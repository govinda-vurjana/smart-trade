# DataReader.py
import csv
import pandas as pd
from datetime import datetime, timedelta
# from FeatureEngineer import FeatureEngineer

class DataReader:
    def __init__(self):
        """
        Initializes the DataReader instance.
        :param minutes: The duration in minutes to fetch the recent data.
        """
        # self.feature_set= FeatureEngineer(window=30)
     

    def fetch_recent_data(self,min,df):
        """
        Fetches data from the last `self.minutes` of data duration.
        :return: Pandas DataFrame containing the last `self.minutes` of data.
        """
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        # df=df.reset_index()
        # print(df.head(5))

        if min==3:
            first_36_rows = df.head(36)  # Selects first 36 rows
            # print("Doing Feature Engineering..")
            # state = self.feature_set.compute_features(first_36_rows)
            # print("Computed Feature Engineering State..")
            # print(state.columns)
            # print(state.shape)
            first_36_rows.reindex()
            # state.to_csv(newfile, index=False, mode='w', header=True)
            return first_36_rows,df.tail(12)
        else:
            return None

    




