import cv2
from Utils.Lanedetect import detect_lanes

cap = cv2.VideoCapture(r"C:\Users\Hp\Downloads\Untitled video - Made with Clipchamp.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    processed = detect_lanes(frame)
    cv2.imshow("Enhanced Lane Detection", processed)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
