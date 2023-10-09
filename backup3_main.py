# import libraries #
import cv2
import numpy as np
import csv
import collections
import time

# Function to calculate the eye fit ratio #
def calculate_eye_fit_ratio(eye):
    a = distance(eye[1], eye[5])
    b = distance(eye[2], eye[4])
    c = distance(eye[0], eye[3])
    
    epsilon = 1e-5
    if c < epsilon:
        return 0.0
    eye_fit_ratio = (a + b) / (2.0 * c)
    return eye_fit_ratio

# Function to calculate the eye aspect ratio 
def calculate_eye_aspect_ratio(eye):
    a = distance(eye[1], eye[5])
    b = distance(eye[2], eye[4])
    c = distance(eye[0], eye[3])
    
    epsilon = 1e-5
    if c < epsilon:
        return 0.0
    ear = (a + b) / (2.0 * c)
    return ear

# Function to calculate the distance between two points
def distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

video_capture = cv2.VideoCapture('VID-1-fatique.mp4') # input capt
frame_width = int(video_capture.get(3))
frame_height = int(video_capture.get(4))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_video.avi', fourcc, 20.0, (frame_width, frame_height))

eye_threshold = 0.20 # eye threshold
fit_ratio_threshold = 0.8  # Mengubah threshold rasio mata
eye_closed_ratio_threshold = 0.8

text_y_closed = frame_height - 20
text_y_open = text_y_closed - 30

threshold_to_record = 0.8
csv_filename = 'hasil_ratio_coba.csv' # export to csv file
with open(csv_filename, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Eyes Status', 'Eye Ratio'])

    timestamp_filename = 'timestamps.txt'  # record timestamp format txt
    with open(timestamp_filename, 'w') as timestamp_file:
        fit_ratio_history = collections.deque(maxlen=10)
        last_eye_closed_time = time.time()  # Track the time of last eye closure
        eye_close_duration = 3  # 3 seconds
        eye_open_duration = 1   # 1 second
        eye_status = "Eye open"  # Initialize eye status

        while True:
            ret, frame = video_capture.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') # function load model 
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)) # batas faces frame min size frame ear

            for (x, y, w, h) in faces:
                roi_gray = gray[y:y + h, x:x + w]

                eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
                eyes = eye_cascade.detectMultiScale(roi_gray)

                eye_fit_ratios = []
                eye_closed_statuses = []

                for (ex, ey, ew, eh) in eyes:
                    eye_roi = roi_gray[ey:ey + eh, ex:ex + ew]

                    ear = calculate_eye_aspect_ratio(eye_roi)
                    fit_ratio = calculate_eye_fit_ratio(eye_roi)

                    if ear < eye_threshold or fit_ratio < fit_ratio_threshold:
                        eye_status = "Eye close"
                    else:
                        eye_status = "Fit Eye"

                    eye_fit_ratios.append(fit_ratio)
                    eye_closed_statuses.append(eye_status)

                    cv2.putText(frame, f"Rasio: {ear:.2f} | Fit Ratio: {fit_ratio:.2f}", (frame_width - 120, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                total_eyes = len(eye_closed_statuses)
                closed_eyes = eye_closed_statuses.count("Eye close")
                open_eyes = total_eyes - closed_eyes

                eye_closed_ratio = closed_eyes / total_eyes if total_eyes > 0 else 0
                eye_fit_ratio = sum(eye_fit_ratios) / total_eyes if total_eyes > 0 else 0

                text_eye_status = f"Eye Status: {eye_status}"
                cv2.putText(frame, text_eye_status, (frame_width - 220, text_y_closed), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                csv_writer.writerow([eye_status, eye_fit_ratio])

                if eye_fit_ratio > threshold_to_record:
                    timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
                    timestamp_file.write(f'{timestamp}: Eye Fit Ratio crossed the threshold ({eye_fit_ratio:.2f})\n')

                fit_ratio_history.append(eye_fit_ratio)
                fit_ratio_threshold = np.mean(fit_ratio_history)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            out.write(frame)
            cv2.imshow('Video', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

video_capture.release()
out.release()
cv2.destroyAllWindows()
