import cv2
import numpy as np
import os
import pandas as pd
from datetime import datetime

# ===== LOAD DATASET =====
path = "dataset"
images = []
names = []

for file in os.listdir(path):
    img = cv2.imread(f"{path}/{file}")
    if img is not None:
        img = cv2.resize(img, (100, 100))  # resize ngay từ đầu
        images.append(img)
        names.append(os.path.splitext(file)[0])

print("Loaded:", names)

# ===== SO SÁNH ẢNH (CẢI TIẾN) =====
def compare(img1, img2):
    img1 = cv2.resize(img1, (100, 100))
    
    # chuyển về grayscale → ổn định hơn
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(img1, img2)
    score = np.mean(diff)

    return score

# ===== GHI ĐIỂM DANH =====
def mark_attendance(name):
    file = "attendance.csv"
    
    if not os.path.exists(file) or os.stat(file).st_size == 0:
        df = pd.DataFrame(columns=["Name", "DateTime"])
        df.to_csv(file, index=False)

    df = pd.read_csv(file)
    today = datetime.now().strftime("%Y-%m-%d")

    if not ((df["Name"] == name) & (df["DateTime"].str.startswith(today))).any():
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        df.loc[len(df)] = [name, now]
        df.to_csv(file, index=False)

        print(f"[ĐIỂM DANH] {name} - {now}")

# ===== LOAD FACE DETECTOR =====
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# ===== MỞ CAMERA =====
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ⚠️ cải tiến detect nhiều mặt hơn
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=4,
        minSize=(60, 60)
    )

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]

        best_match = "Unknown"
        min_score = 9999

        for img, name in zip(images, names):
            score = compare(face_img, img)

            if score < min_score:
                min_score = score
                best_match = name

        # ===== DEBUG SCORE =====
        print(best_match, ":", min_score)

        # ⚠️ tăng threshold để nhận nhiều người hơn
        if min_score < 75:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(frame, best_match, (x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
            mark_attendance(best_match)
        else:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 2)
            cv2.putText(frame, "Unknown", (x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

    cv2.imshow("Attendance", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()