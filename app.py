import streamlit as st
from ultralytics import YOLO
import cv2
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import numpy as np
from qrdet import QRDetector
import json
import matplotlib.pyplot as plt
from ultralytics import YOLO
from PIL import Image
import tempfile


# Tải mô hình YOLO
model_idcard = YOLO("Detect_card.pt")
model_face = YOLO("Detect_face.pt")
model_for_ocr =  YOLO("Detect_for_ocr.pt")
config = Cfg.load_config_from_name('vgg_transformer')
config['cnn']['pretrained'] = True
config['device'] = 'cpu'
vietocr_model = Predictor(config)

def retrieve_documents_from_image(image):
    # image = cv2.imread(image_path)
    results = model_idcard(image)

    cropped_image = []

    for result in results[0].boxes:
        x_min, y_min, x_max, y_max = map(int, result.xyxy[0])
        cropped_image = image[y_min:y_max, x_min:x_max]

        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)

    # Chuyển đổi danh sách thành mảng NumPy nếu các ảnh có cùng kích thước
    return cropped_image


# Hàm tìm QR code
def find_qrcode(image):
    detectors = QRDetector(model_size="s")
    detections = detectors.detect(image=image, is_bgr=True)
    bboxes = []
    for detection in detections:
        x1, y1, x2, y2 = detection["bbox_xyxy"]
        confidence = detection["confidence"]
        bboxes.append([x1, y1, x2, y2, confidence])
    return list(map(int, bboxes[0][:-1])) if bboxes else None


# Hàm tìm khuôn mặt
def find_face(image):
    results = model_face(image)
    if results and len(results[0].boxes) > 0:
        box = results[0].boxes[0]
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        return [int(x1), int(y1), int(x2), int(y2)]
    return None



def align_bboxes(image):
    if image is None:
        return None, "Không thể đọc ảnh."

    h, w, _ = image.shape
    qr_bbox = find_qrcode(image)
    face_bbox = find_face(image)

    if not qr_bbox or not face_bbox:
        return None, "Không tìm thấy QR code hoặc khuôn mặt."

    # Tính toán góc xoay
    face_center = ((face_bbox[0] + face_bbox[2]) // 2, (face_bbox[1] + face_bbox[3]) // 2)
    qr_center = ((qr_bbox[0] + qr_bbox[2]) // 2, (qr_bbox[1] + qr_bbox[3]) // 2)

    delta_x = qr_center[0] - face_center[0]
    delta_y = face_center[1] - qr_center[1]
    angle = np.degrees(np.arctan2(delta_y, delta_x))

    # Nếu góc xoay gần 0, không cần xoay ảnh
    if abs(angle) < 1:  # Ngưỡng là 1 độ
        return image, None

    # Mở rộng ảnh để xoay
    diagonal = int(np.sqrt(h**2 + w**2))
    expanded_image = np.zeros((diagonal, diagonal, 3), dtype=np.uint8)
    x_offset = (diagonal - w) // 2
    y_offset = (diagonal - h) // 2
    expanded_image[y_offset:y_offset+h, x_offset:x_offset+w] = image

    # Xoay ảnh theo góc tính toán
    center = (diagonal // 2, diagonal // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)
    aligned_image = cv2.warpAffine(expanded_image, rotation_matrix, (diagonal, diagonal))

    # Xoay thêm 20 độ ngược chiều kim đồng hồ
    rotation_matrix_20 = cv2.getRotationMatrix2D(center, 22, 1.0)
    aligned_image = cv2.warpAffine(aligned_image, rotation_matrix_20, (diagonal, diagonal))

    # Tự động cắt lại vùng không phải pixel đen
    gray = cv2.cvtColor(aligned_image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])
        aligned_image = aligned_image[y:y+h, x:x+w]

    return aligned_image


# Hàm trích xuất thông tin từ ảnh
def extract_info_from_image(image_path, model, vietocr_model):
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
    image = cv2.imread(image_path)
    image = align_bboxes(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = retrieve_documents_from_image(image)
    image = cv2.resize(image,(640,640))
    vis_image = image.copy()
    # print(image.shape)
    results = model(image)
    for result in results[0].boxes:
        x_min, y_min, x_max, y_max = map(int, result.xyxy[0])
        cropped_image = image[y_min:y_max, x_min:x_max]

        # cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
        cropped_image = Image.fromarray(cropped_image)
        class_id = int(result.cls[0])
        label = model.names[class_id]

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
        color = (0, 255, 0)  # Green for bounding box
        cv2.rectangle(vis_image, (x_min, y_min), (x_max, y_max), color, 2)
        cv2.putText(vis_image, f"{label}: {text}", (x_min, y_min - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    if current_places:
        current_places = sorted(current_places, key=len)
        labels["Nơi thường trú"] = ", ".join(current_places)

    output_path = "output_with_bboxes.jpg"
    cv2.imwrite(output_path, vis_image)
    print(f"Bounding box image saved to {output_path}")
    # Optional: Display the image
    # cv2.imshow("Result", vis_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return json.dumps(labels, indent=4, ensure_ascii=False), output_path

st.title("Trích xuất thông tin từ ảnh CCCD")
uploaded_file = st.file_uploader("Tải lên ảnh CCCD", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as temp:
        temp.write(uploaded_file.read())
        temp_path = temp.name

    st.image(uploaded_file, caption="Ảnh đã tải lên", use_column_width=True)

    with st.spinner("Đang xử lý ảnh..."):
        info, output_path = extract_info_from_image(temp_path, model_for_ocr, vietocr_model)

    st.subheader("Thông tin trích xuất")
    st.json(json.loads(info))

    st.subheader("Ảnh với bounding boxes")
    st.image(output_path, caption="Ảnh sau khi xử lý", use_column_width=True)