import streamlit as st
import cv2
import torch
from ultralytics import YOLO
import csv
import os

# Load the YOLO model (ensure it's a model capable of detecting drones)
model = YOLO("yolov8n.pt")  # Replace with your model file (YOLO model trained to detect drones)

# Streamlit configuration
st.set_page_config(page_title="Drone Detection in Video", layout="wide")

# Initialize CSV file for logging drone detections
csv_file = 'top_drone_detections.csv'
if not os.path.isfile(csv_file):
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Frame Number', 'Timestamp (Seconds)', 'Class Name', 'Confidence'])  # Write header


# Function to process video and detect drone
def process_video(uploaded_video):
    # Open the input video
    cap = cv2.VideoCapture(uploaded_video)
    
    # Define the codec and create VideoWriter object to save the output video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output_video.mp4', fourcc, 20.0, (640, 480))

    # Get the model's class names (from the model itself, not from the results)
    class_names = model.names  # This gives you a list of class names

    # Initialize frame number
    frame_number = 0

    # Process each frame in the video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run the object detection on the current frame
        results = model(frame)

        # Render the results on the frame
        annotated_frame = results[0].plot()

        # Check if drone is detected by looking at the classes detected in the frame
        drone_detected = False
        for result in results[0].boxes:
            # Access class ID and class name
            class_id = int(result.cls)  # The class ID of the detected object
            class_name = class_names[class_id]  # Get the class name from the model

            # Check if the detected object is a drone (adjust the class name as per your model's label)
            if "drone" in class_name.lower() or "aeroplane" in class_name.lower() or "airplane" in class_name.lower():
                drone_detected = True
                break

        # If drone is detected, log the detection in the CSV file
        if drone_detected:
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000  # Get timestamp in seconds
            with open(csv_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([frame_number, timestamp, class_name])  # Log frame number, timestamp, and class name

            # Display message on frame
            cv2.putText(annotated_frame, "Drone Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # Display the frame in Streamlit
        st.image(annotated_frame, channels="BGR", caption="Drone Detection Frame")

        # Write the frame to output video
        out.write(annotated_frame)

        # Increment frame number
        frame_number += 1

    # Release everything when done
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    return 'output_video.mp4'


# Streamlit UI
st.title("Micro Doppler Based Target Classification in realtime ")

# File uploader for video
uploaded_video = st.file_uploader("Upload a video for drone detection", type=["mp4", "avi", "mov", "mkv"])

if uploaded_video is not None:
    # Display uploaded video
    st.video(uploaded_video, format="video/mp4")

    if st.button("Run Micro Doppler target  Detection"):
        with st.spinner("Processing the video..."):
            # Save the uploaded video temporarily
            temp_video_path = f"temp_video.{uploaded_video.name.split('.')[-1]}"
            with open(temp_video_path, "wb") as f:
                f.write(uploaded_video.getbuffer())

            # Process the video and detect drones
            output_video_path = process_video(temp_video_path)

            # Provide download link for the output video
            st.success(" detection completed!")
            st.video(output_video_path, caption="Output Video with Drone Detection")
            
            # Provide download link for the CSV file
            st.download_button(
                label="Download Detection Log (CSV)",
                data=open(csv_file, "rb").read(),
                file_name=csv_file,
                mime="text/csv"
            )

            # Clean up the temporary files
            os.remove(temp_video_path)
            os.remove(output_video_path)

