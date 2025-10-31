import csv
import time,keyboard
from datetime import datetime
import pandas as pd
from DataReader import DataReader
from TickTradingEnv import train_agent



# Function to be called when 20 minutes of data is ready for prediction
def data_ready_for_prediction(min,dataReader,file):
    print("data is ready for RL.")
    print("Now resampling and doing feature engineering...")
    df=dataReader.fetch_recent_data(min,file)
    if not df.empty:
        print("Got Agent State set..")
        print("Training Agent Started")
        train_agent(df)
    else:
        print("No State Info received..")
    



# Function to calculate the duration of data in minutes
def calculate_data_duration(csv_filename):
    try:
        with open(csv_filename, 'r') as file:
            reader = csv.reader(file)
            # Skip the header row
            next(reader)

            # Get the timestamp of the first and last entry in the file
            rows = list(reader)
            if rows:
                first_row = rows[0]
                last_row = rows[-1]
                
                first_timestamp = datetime.fromisoformat(first_row[0])
                last_timestamp = datetime.fromisoformat(last_row[0])

                # Calculate the difference in time between the first and last timestamp
                time_diff = last_timestamp - first_timestamp
                print(time_diff)

                # Convert time difference to minutes
                data_duration_minutes = time_diff.total_seconds() / 60
                return data_duration_minutes
            else:
                return 0
    except FileNotFoundError:
        print(f"Error: {csv_filename} not found.")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 0


# Continuously monitor the live_data.csv file
csv_filename = 'live_data.csv'
oneminfile="sampled5s.csv" #holds recent 1min data

print("Monitoring live_data.csv...")

# Establish socket connection to send statuses
dataReader = DataReader()


while True:
    # Get the duration of data available in minutes
    ready=False
    data_duration = calculate_data_duration(csv_filename)
    
    # Print the duration of the available data
    print(f"Live data consists of {int(data_duration)} minutes of data.")
    
    # Send live data status to the server
    # send_status_to_server(client_socket, f"⏳ Live data consists of {int(data_duration)} minutes of data.")

    # Check if 20 minutes of data is available
    if data_duration >=3:
        #fetch last 10min of data and write to tenminfile
        data_ready_for_prediction(3,dataReader,oneminfile)  # Call the function when 10 minutes of data is available
        

   


    
