from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import cv2
import json
import matplotlib.pyplot as plt
from ultralytics import YOLO
from PIL import Image

model = YOLO('Detect_for_ocr.pt')

config = Cfg.load_config_from_name('vgg_transformer')
config['cnn']['pretrained'] = True
config['device'] = 'cpu'
vietocr_model = Predictor(config)

image_path = './test_image.jpg'
image = cv2.imread(image_path)

image = cv2.resize(image, (640, 640))

results = model(image)
labels = {
    "Họ tên": "",
    "Ngày sinh": "",
    "Giới tính": "",
    "Số CCCD": "",
    "Quốc tịch": "",
    "Quê quán": "",
    "Nơi thường trú": "",
    "Ngày hết hạn": "",
}
current_places = []
# Lặp qua từng bounding box
for result in results[0].boxes:
    x_min, y_min, x_max, y_max = map(
        int, result.xyxy[0])  # Lấy tọa độ bounding box
    # Cắt ảnh theo bounding box
    cropped_image = image[y_min:y_max, x_min:x_max]

    # Chuyển đổi sang RGB vì VietOCR yêu cầu
    cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)

    # Chuyển numpy.ndarray sang PIL.Image
    cropped_image = Image.fromarray(cropped_image)
    class_id = int(result.cls[0])  # Get class ID
    label = model.names[class_id]
    # Trích xuất text bằng VietOCR
    text = vietocr_model.predict(cropped_image)

    if label == 'name':
        labels["Họ tên"] = text
    elif label == 'dob':
        labels["Ngày sinh"] = text
    elif label == 'gender':
        labels["Giới tính"] = text
    elif label == 'id':
        labels["Số CCCD"] = text
    elif label == 'nationality':
        labels["Quốc tịch"] = text
    elif label == 'origin_place':
        labels["Quê quán"] = text
    elif label == 'current_place':
        current_places.append(text)
    elif label == 'expire_date':
        labels["Ngày hết hạn"] = text
    else:
        continue
    if current_places:
        current_places = sorted(current_places, key=len)
        labels["Nơi thường trú"] = ", ".join(current_places)
    # Vẽ bounding box và text lên ảnh gốc
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
labels = json.dumps(labels, indent=4, ensure_ascii=False)
print(labels)
# Hiển thị ảnh với bounding boxes và text
plt.figure(figsize=(12, 6))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
