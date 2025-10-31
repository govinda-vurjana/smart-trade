# DataReader.py
import csv
import pandas as pd
from datetime import datetime, timedelta

class DataReader:
    def __init__(self, minutes):
        """
        Initializes the DataReader instance.
        :param minutes: The duration in minutes to fetch the recent data.
        """
        self.minutes = minutes
        self.filename = 'live_data.csv'  # The file that stores the live data

    def fetch_recent_data(self):
        """
        Fetches data from the last `self.minutes` of data duration.
        :return: Pandas DataFrame containing the last `self.minutes` of data.
        """
        # Get the current time and the timestamp for the cutoff time
        current_time = datetime.now()
        cutoff_time = current_time - timedelta(minutes=self.minutes)

        recent_data = []

        try:
            with open(self.filename, 'r') as file:
                reader = csv.reader(file)
                # Skip the header row
                next(reader)

                # Read all rows and filter based on the timestamp
                for row in reader:
                    timestamp = datetime.fromisoformat(row[0])  # Convert timestamp string to datetime object
                    if timestamp >= cutoff_time:
                        recent_data.append(row)

            # If we have data from the last `self.minutes`, return it as a DataFrame
            if recent_data:
                df = pd.DataFrame(recent_data, columns=["Timestamp", "Value"])
                return df
            else:
                print(f"No data available in the last {self.minutes} minutes.")
                return None

        except FileNotFoundError:
            print(f"Error: {self.filename} not found.")
            return None
        except Exception as e:
            print(f"Error: {e}")
            return None



 