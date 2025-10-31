import os
import pandas as pd

# Load the CSV file
dest_path = r"C:\Users\vurja\AI\SmartTrade\stark\Preprocessed\EURUSD"
output_file = os.path.join(dest_path, "cleaned_EURUSD.csv")
file_path = output_file  # Update with your actual file path
df = pd.read_csv(file_path)

# Print initial stats
initial_rows = len(df)
missing_rows = df[['open', 'high', 'low', 'close']].isna().sum().sum()
print(f"📊 Initial Rows: {initial_rows}")
print(f"❌ Missing OHLC Values: {missing_rows}")

# Drop rows where 'open', 'high', 'low', 'close' are NaN
df_cleaned = df.dropna(subset=['open', 'high', 'low', 'close'])

# Print final stats
final_rows = len(df_cleaned)
dropped_rows = initial_rows - final_rows
print(f"✅ Final Rows: {final_rows}")
print(f"🗑️ Dropped Rows: {dropped_rows}")

# Save the cleaned data
output_file = "new_cleaned_file.csv"  # Update with desired output file name
df_cleaned.to_csv(output_file, index=False)

print(f"✅ Cleaned file saved: {output_file}")
