import cv2
import torch
import numpy as np
from Utils.Lanedetect import detect_lanes
from Utils.predict_sign import predict_sign

# Load YOLOv5 model
model = torch.hub.load("E:/Simu Drive -AI/yolov5-master", 'yolov5s', source='local')
model.eval()

# Load video
cap = cv2.VideoCapture(r"C:\Users\Hp\Downloads\videoplayback (1).mp4")
if not cap.isOpened():
    print("Error: Couldn't open video.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ========== Lane Detection ==========
    lane_output = detect_lanes(frame.copy())  # returns annotated frame

    # ========== Traffic Sign Detection ==========
    # crop potential sign area — you can improve this
    sign_crop = frame[0:150, 0:150]
    label, confidence = predict_sign(sign_crop)

    # Show label if confident
    if confidence > 0.7:
        cv2.putText(lane_output, f"Sign: {label} ({confidence*100:.1f}%)", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # ========== YOLO Object Detection ==========
    results = model(frame)
    detections = results.pandas().xyxy[0]  # get predictions as pandas dataframe

    for _, row in detections.iterrows():
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        conf, cls = row['confidence'], row['name']
        if conf > 0.4:
            cv2.rectangle(lane_output, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(lane_output, f"{cls} ({conf:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show all in one window
    lane_output = cv2.resize(lane_output, (1280, 720))  # or (1920, 1080) for Full HD
    cv2.imshow("AutoPilot - X", lane_output)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
