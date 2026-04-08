import cv2 as cv
import easyocr
import matplotlib.pyplot as plt 

img_path = 'data/bienso.jpg'
def crop_plate(img, x,y,w,h, pad=10):
    x1 = max(x - pad, 0)
    y1 = max(y - pad, 0)
    x2 = min(x + w + pad, img.shape[1])
    y2 = min(y + h + pad, img.shape[0])
    return img[y1:y2, x1:x2]


def preprocess_plate(plate_img):
    h,w = plate_img.shape[:2]
    scale = 60.0 / h if h < 60 else 1.0
    resize = cv.resize(plate_img, None, fx=scale,fy= scale, interpolation=cv.INTER_CUBIC)
    gray = cv.cvtColor(resize, cv.COLOR_BGR2GRAY)
    _, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    denoised = cv.fastNlMeansDenoising(binary, h=10)
    return denoised
# load image 


img = cv.imread(img_path, cv.IMREAD_COLOR)
# convert to gray 
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) #0 - 255

#chuyển sang ảnh nhị phân để dễ dàng phát hiện các cạnh
_, binary = cv.threshold(gray, 90, 255, cv.THRESH_BINARY) #0 - 1 
# khử nhiễu
denoise = cv.GaussianBlur(binary, (5, 5), 0) # chọn kỹ thuật khác

#detect các cạnh có trong ảnh
edges = cv.Canny(denoise, 20, 200)

kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
closed = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel)

contours, _ = cv.findContours(closed.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

candidate_plates = []
img_size = img.shape[0] * img.shape[1] 

for cnt in contours:
    x, y, w, h = cv.boundingRect(cnt)
    aspect_ratio = w / h
    area_ratio = (w * h)/ img_size
    if (2.0 < aspect_ratio < 6.0) and (0.005 < area_ratio < 0.15):
        candidate_plates.append((x, y, w, h))
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

for (x, y, w, h) in candidate_plates:
    plate_img = crop_plate(img, x, y, w, h)
    reader = easyocr.Reader(['en'], gpu=True) 
    plate_ready = preprocess_plate(plate_img)
    result = reader.readtext(plate_ready, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', detail=1)
    for (bbox, text, conf) in result:
        if conf > 0.5:  # Chỉ hiển thị kết quả có độ tin cậy > 0.5
            print(f"Detected Text: {text}, Confidence: {conf:.2f}")