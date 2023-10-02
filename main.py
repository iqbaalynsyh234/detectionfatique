import cv2
import numpy as np
import csv
import pandas as pd

# Function to calculate the eye fit ratio
def calculate_eye_fit_ratio(eye):
    a = distance(eye[1], eye[5])
    b = distance(eye[2], eye[4])
    c = distance(eye[0], eye[3])
    eye_fit_ratio = (a + b) / (2.0 * c)
    return eye_fit_ratio

# Function to calculate the eye aspect ratio
def calculate_eye_aspect_ratio(eye):
    a = distance(eye[1], eye[5])
    b = distance(eye[2], eye[4])
    c = distance(eye[0], eye[3])
    ear = (a + b) / (2.0 * c)
    return ear

# Function to calculate the distance between two points
def distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

# Initialize video capture
video_capture = cv2.VideoCapture('VID-1-fatique.mp4')

# Get video properties
frame_width = int(video_capture.get(3))
frame_height = int(video_capture.get(4))

# Initialize video writer for the output video in MP4 format
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_video.avi', fourcc, 20.0, (frame_width, frame_height))

# Initialize variables
eye_threshold = 0.20
fit_ratio_threshold = 0.3  # Define a threshold for eye fit ratio

# Initialize text_y_closed and text_y_open variables
text_y_closed = frame_height - 20
text_y_open = text_y_closed - 30

# Create a CSV file for storing open eyes ratio, eye fit ratio, and eyes closed status
csv_filename = 'hasil_ratio.csv'
with open(csv_filename, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Eyes Closed Status', 'Eye Fit Ratio'])  # Write header row

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Get the region of interest (ROI) for the face
            roi_gray = gray[y:y + h, x:x + w]

            # Detect eyes in the ROI
            eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            eyes = eye_cascade.detectMultiScale(roi_gray)

            # Initialize lists to store eye fit ratios and eye closed statuses for each frame
            eye_fit_ratios = []
            eye_closed_statuses = []

            for (ex, ey, ew, eh) in eyes:
                eye_roi = roi_gray[ey:ey + eh, ex:ex + ew]

                # Calculate the eye aspect ratio
                ear = calculate_eye_aspect_ratio(eye_roi)

                # Calculate the eye fit ratio
                fit_ratio = calculate_eye_fit_ratio(eye_roi)

                # Detect closed eyes
                if ear < eye_threshold or fit_ratio < fit_ratio_threshold:
                    eye_status = "Tired Eyes"
                else:
                    eye_status = "Alert Eyes"

                # Append the data for eye fit ratio and eyes closed status to the lists
                eye_fit_ratios.append(fit_ratio)
                eye_closed_statuses.append(eye_status)

                # Display the EAR and fit ratio values on the frame (top-right corner)
                cv2.putText(frame, f"Rasio: {ear:.2f} | Fit Ratio: {fit_ratio:.2f}", (frame_width - 120, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            # Calculate eye closed ratio and eye fit ratio for this frame
            total_eyes = len(eye_closed_statuses)
            closed_eyes = eye_closed_statuses.count("Tired Eyes")
            open_eyes = total_eyes - closed_eyes

            eye_closed_ratio = closed_eyes / total_eyes if total_eyes > 0 else 0
            eye_fit_ratio = sum(eye_fit_ratios) / total_eyes if total_eyes > 0 else 0

            # Add text displaying eye closed ratio and eye fit ratio to the frame
            text_eye_closed = f"Eye Closed Ratio: {eye_closed_ratio:.2f}"
            text_eye_fit = f"Eye Fit Ratio: {eye_fit_ratio:.2f}"

            cv2.putText(frame, text_eye_closed, (frame_width - 220, text_y_closed), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            cv2.putText(frame, text_eye_fit, (frame_width - 220, text_y_open), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            # Write the eye fit ratio and eye closed status to the CSV file
            csv_writer.writerow([eye_status, eye_fit_ratio])

            # Draw a frame around the detected face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Write the frame to the output video
        out.write(frame)

        # Show the frame
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release video capture and writer, and close the window
video_capture.release()
out.release()
cv2.destroyAllWindows()
