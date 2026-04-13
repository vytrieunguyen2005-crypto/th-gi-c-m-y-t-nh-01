import cv2
import os

path = "dataset"

cap = cv2.VideoCapture(0)

print("Nhấn SPACE để chụp | Q để thoát")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Add User", frame)

    key = cv2.waitKey(1) & 0xFF

    # 👉 Chụp ảnh
    if key == 32:  # SPACE
        name = input("Nhập tên: ")
        cv2.imwrite(f"{path}/{name}.jpg", frame)
        print(f"✅ Đã lưu {name}.jpg")

    # 👉 Thoát
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()