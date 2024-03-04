import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os  # Import the 'os' module for file and path operations

# Load the attentiveness data
attentiveness_data = pd.read_csv('attentiveness_data.csv')

# Convert the 'Timestamp' column to datetime
attentiveness_data['Timestamp'] = pd.to_datetime(attentiveness_data['Timestamp'])

# Round the timestamps to the nearest minute
attentiveness_data['Timestamp'] = attentiveness_data['Timestamp'].dt.round('1min')

# Create a line plot using seaborn
plt.figure(figsize=(12, 6))
sns.lineplot(x='Timestamp', y='Prediction', data=attentiveness_data)
plt.title('Attentiveness Prediction Over Time')
plt.xlabel('Time')
plt.ylabel('Prediction')

# Create the "reports" folder if it doesn't exist
os.makedirs('reports', exist_ok=True)  # Handles existing folder gracefully

# Generate a unique filename based on current timestamp
filename = f"attentiveness_report_{pd.Timestamp.now().strftime('%Y-%m-%d_%H-%M-%S')}.png"

# Save the plot as a PNG image in the "reports" folder with tight bounding box and no padding
filepath = os.path.join('reports', filename)
plt.savefig(filepath, format='png', bbox_inches='tight', pad_inches=0)

print(f"Report saved to: {filepath}")
