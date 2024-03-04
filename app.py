import cv2
from flask import Flask, render_template, Response, redirect, url_for
import mediapipe as mp
import datetime
import csv
from utils import mediapipe_detection, predict_attentiveness, draw_styled_landmarks
import io
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os
import seaborn as sns
matplotlib.use('Agg')  # Set the backend to Agg

import matplotlib.pyplot as plt
import io
import pandas as pd
# Customize these based on your setup
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480

# Initialize MediaPipe Holistic model (replace with your loading method if using pre-trained)
mp_holistic = mp.solutions.holistic.Holistic()

# Flask app setup
app = Flask(__name__)

# Initialize video capture
cap = None

# Initialize global variables
recording = False
timestamps = []
attentiveness_levels = []
detect = False

# Route for the main web page
@app.route('/')
def index():
    return render_template('index.html', recording=recording, timestamps=timestamps, attentiveness_levels=attentiveness_levels)

# Route for video streaming
def generate_frames():
    global recording, timestamps, attentiveness_levels, detect

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if detect:

        # Process frame with MediaPipe (including attentiveness prediction and visualization)
            image, results = mediapipe_detection(frame, mp_holistic)
            prediction = predict_attentiveness(results)

        # Draw styled landmarks on the frame with specified font scale
            image = draw_styled_landmarks(image, results, font_scale=2) 

        # Capture timestamps and attentiveness levels if recording
            if recording:
                timestamps.append(datetime.datetime.now().strftime("%H:%M:%S"))
                attentiveness_levels.append(prediction)

        # Encode frame as JPEG bytestream
            frame = cv2.imencode('.jpg', image)[1].tobytes()
            yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Start recording button handler
@app.route('/start_recording', methods=['POST'])
def start_recording():
    global recording,detect,cap
    recording = True
    detect = True
    cap= cv2.VideoCapture(0)
    return redirect(url_for('index'))
# Stop recording button handler
@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    global recording, timestamps, attentiveness_levels, detect,cap

    recording = False
    detect = False
    cap.release()
    # Save timestamps and attentiveness levels to a local CSV file (optional)
    if timestamps and attentiveness_levels:
        with open("attentiveness_data.csv", "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            for timestamp, attentiveness in zip(timestamps, attentiveness_levels):
                writer.writerow([timestamp, attentiveness])

    timestamps = []
    attentiveness_levels = []

    # Release the camera capture object
    

    return redirect(url_for('index'))

@app.route('/download_report')
def download_report():
  """
  Generates report image and returns a downloadable response.
  """
  try:
    # Generate report image using your existing logic
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

    # Save the plot as a PNG image in the "reports" folder
    filepath = os.path.join('reports', filename)
    plt.savefig(filepath, format='png', bbox_inches='tight', pad_inches=0)

    # Read the image data into a byte stream
    with open(filepath, 'rb') as f:
      image_bytes = f.read()

    # Set response headers for download
    response = Response(image_bytes, mimetype="image/png",
                        headers={"Content-Disposition": "attachment; filename=attentiveness_report.png"})
    return response

  except FileNotFoundError:
    return "Report image not found.", 404

@app.route('/generate_report', methods=['POST'])
def generate_report():
    global timestamps, attentiveness_levels

    try:
        df = pd.read_csv('attentiveness_data.csv')
        timestamps = df['Timestamp'].tolist()
        attentiveness_levels = df['Prediction'].tolist()
    except FileNotFoundError:
        return "Error: Attentiveness data file not found.", 404
    image_bytes = generate_report_image(timestamps, attentiveness_levels)

    if not timestamps or not attentiveness_levels:
        return "No data available for report generation.", 404

    # Load data from attentiveness_data.csv
    print(f"Image Bytes: {image_bytes}")
    return render_template('report.html', image_bytes=image_bytes)

def generate_report_image(timestamps, attentiveness_levels):
    """
    Generates a line graph of attentiveness over time and returns the image data as bytes.

    Args:
        timestamps (list): List of timestamps.
        attentiveness_levels (list): List of attentiveness levels (e.g., "Attentive", "Not Attentive").

    Returns:
        bytes: Image data in PNG format.
    """

    # Convert timestamps to datetime if needed
    if isinstance(timestamps[0], str):
        timestamps = pd.to_datetime(timestamps, format='%H:%M:%S')

    # Create a DataFrame from timestamps and attentiveness_levels
    data = pd.DataFrame({'Timestamp': timestamps, 'Prediction': attentiveness_levels})

    # Resample data at 1-minute intervals, count occurrences, and fill missing values
    resampled_data = data.resample('T', on='Timestamp').count().unstack().fillna(0)

    # Create new figure and plot (instead of showing)
    fig, ax = plt.subplots(figsize=(12, 6))
    resampled_data.plot(kind='line', marker='o', ax=ax)
    plt.title('Attentiveness Over Time')
    plt.xlabel('Time')
    plt.ylabel('Count')
    plt.grid(True)

    # Save the plot as a PNG image in memory
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    plt.close(fig)  # Close the figure to release resources

    # Return the image data as bytes
    return buffer.getvalue()

# Route for video streaming
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
