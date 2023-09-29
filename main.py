import cv2
import numpy as np
import csv

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
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Use 'XVID' codec for AVI format
out = cv2.VideoWriter('output_video.avi', fourcc, 20.0, (frame_width, frame_height))

# Initialize variables
eye_threshold = 0.20
frame_count = 0
eye_closed_frames = 0

# Create a CSV file for storing eye status
csv_filename = 'eye_status.csv'
with open(csv_filename, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Frame', 'Eye Status'])  # Write header row

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

            for (ex, ey, ew, eh) in eyes:
                eye_roi = roi_gray[ey:ey + eh, ex:ex + ew]

                # Calculate the eye aspect ratio
                ear = calculate_eye_aspect_ratio(eye_roi)

                # Debugging: Print EAR values
                print(f"EAR: {ear}")

                # Detect closed eyes
                if ear < eye_threshold:
                    eye_closed_frames += 1
                    eye_status = "Tired Eyes"
                else:
                    eye_closed_frames = 0
                    eye_status = "Alert Eyes"

                # If eyes are closed for several consecutive frames, mark as closed
                if eye_closed_frames >= 1:
                    text = "Eyes Closed"
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    text_x = frame_width - text_size[0] - 20
                    text_y = frame_height - 20
                    cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                # Display the EAR value on the frame (top-right corner)
                cv2.putText(frame, f"Rasio: {ear:.2f}", (frame_width - 120, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Write the eye status to the CSV file
                csv_writer.writerow([frame_count, eye_status])  # Write frame number and eye status

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
