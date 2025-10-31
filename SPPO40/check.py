import pandas as pd

# ✅ Read the CSV File
df = pd.read_csv('logs/AutoEvaluation_eur_usd.csv')

# ✅ Compare Action vs TrueAction
matched = (df['Action'] == df['TrueAction']).sum()
not_matched = (df['Action'] != df['TrueAction']).sum()

# ✅ Print the Results
print(f"✅ Total Matches: {matched}")
print(f"❌ Total Mismatches: {not_matched}")

# ✅ Calculate Accuracy Percentage
total = len(df)
accuracy = (matched / total) * 100

print(f"📊 Accuracy: {accuracy:.2f}%")
