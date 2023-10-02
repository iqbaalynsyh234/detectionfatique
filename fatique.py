import cv2
import numpy as np
import pandas as pd

# Fungsi untuk mendeteksi rasio mata terbuka atau tertutup
def detect_eye_ratio(eye_landmarks):
    if len(eye_landmarks) == 6:  # Pastikan ada 6 titik landmark mata yang terdeteksi
        # Hitung jarak vertikal antara poin atas dan bawah mata
        vertical_distance = np.linalg.norm(eye_landmarks[1] - eye_landmarks[4])

        # Hitung jarak horizontal antara poin kiri dan kanan mata
        horizontal_distance = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])

        # Hitung rasio mata terbuka
        eye_ratio = horizontal_distance / vertical_distance

        return eye_ratio
    else:
        return None

# Inisialisasi classifier untuk deteksi wajah
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Inisialisasi classifier untuk deteksi mata
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Inisialisasi video capture
cap = cv2.VideoCapture('VID-1-fatique.mp4')  # Ganti 'nama_video.mp4' dengan nama video Anda

eye_ratios = []  # List untuk menyimpan rasio mata terbuka
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)

        for (ex, ey, ew, eh) in eyes:
            eye_landmarks = np.array([(x + ex, y + ey), (x + ex + ew, y + ey),
                                      (x + ex, y + ey + eh), (x + ex + ew, y + ey + eh),
                                      (x + ex + (ew // 2), y + ey), (x + ex + (ew // 2), y + ey + eh)])
            
            eye_ratio = detect_eye_ratio(eye_landmarks)
            
            if eye_ratio is not None:
                eye_ratios.append(eye_ratio)

    frame_count += 1

cap.release()
cv2.destroyAllWindows()

# Menyimpan hasil rasio mata ke dalam file CSV
df = pd.DataFrame({'Eye Ratio': eye_ratios})
df.to_csv('eye_ratios.csv', index=False)

print(f"Total frames: {frame_count}")
print(f"Rasio mata terbuka dan tertutup telah disimpan dalam 'eye_ratios.csv'")
