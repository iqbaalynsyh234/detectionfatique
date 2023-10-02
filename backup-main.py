import cv2
import numpy as np
import csv
import pandas as pd

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
frame_count = 0
eye_closed_frames = 0
total_eyes = 0
open_eyes = 0

# Initialize text_y_closed and text_y_open variables
text_y_closed = frame_height - 20
text_y_open = text_y_closed - 30

# Initialize an empty list to store data for open eyes ratio and eyes closed status
data_list = []

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

        total_eyes += len(eyes)

        for (ex, ey, ew, eh) in eyes:
            eye_roi = roi_gray[ey:ey + eh, ex:ex + ew]

            # Calculate the eye aspect ratio
            ear = calculate_eye_aspect_ratio(eye_roi)

            # Detect closed eyes
            if ear < eye_threshold:
                eye_closed_frames += 1
                eye_status = "Tired Eyes"
            else:
                open_eyes += 1
                eye_closed_frames = 0
                eye_status = "Alert Eyes"

            # If eyes are closed for several consecutive frames, mark as closed
            if eye_closed_frames >= 1:
                text_closed = f"Eyes Closed (Total: {total_eyes})"
                text_size_closed = cv2.getTextSize(text_closed, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                text_x_closed = frame_width - text_size_closed[0] - 20
                cv2.putText(frame, text_closed, (text_x_closed, text_y_closed), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            # Display the open eyes ratio above the "Eyes Closed" text
            text_open = f"Open Eyes Ratio: {open_eyes / total_eyes:.2f}"
            text_size_open = cv2.getTextSize(text_open, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            text_x_open = frame_width - text_size_open[0] - 20
            cv2.putText(frame, text_open, (text_x_open, text_y_open), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            # Display the EAR value on the frame (top-right corner)
            cv2.putText(frame, f"Rasio: {ear:.2f}", (frame_width - 120, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw a frame around the detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Append the data for open eyes ratio and eyes closed status to the list for each frame
    open_eyes_ratio = open_eyes / total_eyes if total_eyes > 0 else 0
    data_list.append([eye_status, open_eyes_ratio])

    # Write the frame to the output video
    out.write(frame)

    # Show the frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Convert the list to a Pandas DataFrame
data_df = pd.DataFrame(data_list, columns=['Eyes Closed Status', 'Open Eyes Ratio'])

# Save the DataFrame as a CSV file
csv_filename = 'ear_values.csv'
data_df.to_csv(csv_filename, index=False)

# Release video capture and writer, and close the window
video_capture.release()
out.release()
cv2.destroyAllWindows()
