import torch
import cv2

# Load the YOLOv5s model from your local cloned repo
model = torch.hub.load("E:/Simu Drive -AI/yolov5-master", 'yolov5s', source='local')
model.conf = 0.4  # Set confidence threshold

# Load your test video
cap = cv2.VideoCapture(r"C:\Users\Hp\Downloads\Untitled video - Made with Clipchamp.mp4")  # Change path if needed

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run object detection
    results = model(frame)
    result_img = results.render()[0]

    # Show output
    cv2.imshow("YOLOv5 Object Detection", result_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
