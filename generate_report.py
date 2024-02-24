import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Read the CSV file into a DataFrame
df = pd.read_csv('attentiveness_data.csv')

# Convert 'Timestamp' column to datetime format
df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%H:%M', errors='coerce')

# Group by timestamp and calculate the percentage of "Attentive" predictions
grouped_df = df.groupby('Timestamp')['Prediction'].value_counts(normalize=True).unstack().fillna(0)

# Plot the line graph
plt.figure(figsize=(10, 6))
plt.plot(grouped_df.index, grouped_df['Attentive'] * 100, label='Attentive', marker='o')
plt.plot(grouped_df.index, grouped_df['Not Attentive'] * 100, label='Not Attentive', marker='o')

# Customize the plot
plt.title('Attentiveness Over Time')
plt.xlabel('Timestamp')
plt.ylabel('Percentage')
plt.legend()
plt.grid(True)

# Format x-axis ticks at 1-minute intervals
plt.gca().xaxis.set_major_locator(mdates.MinuteLocator(interval=1))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

plt.xticks(rotation=45)
plt.tight_layout()

# Show the plot
plt.show()
