import cv2
import numpy as np

def region_of_interest(img):
    height, width = img.shape[:2]
    mask = np.zeros_like(img)

    polygon = np.array([[
        (int(0.1 * width), height),
        (int(0.45 * width), int(0.6 * height)),
        (int(0.55 * width), int(0.6 * height)),
        (int(0.9 * width), height)
    ]], np.int32)

    cv2.fillPoly(mask, polygon, 255)
    masked = cv2.bitwise_and(img, mask)
    return masked

def average_slope_intercept(lines):
    left_lines, right_lines = [], []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 == x1:
            continue  # skip vertical lines
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        if slope < -0.5:
            left_lines.append((slope, intercept))
        elif slope > 0.5:
            right_lines.append((slope, intercept))

    def make_line(slope_intercepts):
        if len(slope_intercepts) == 0:
            return None
        slope, intercept = np.mean(slope_intercepts, axis=0)
        y1 = 720
        y2 = int(0.6 * y1)
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        return np.array([[x1, y1, x2, y2]])

    left_avg = make_line(left_lines)
    right_avg = make_line(right_lines)

    lines_out = []
    if left_avg is not None:
        lines_out.append(left_avg)
    if right_avg is not None:
        lines_out.append(right_avg)

    return lines_out

def detect_lanes(frame):
    # Color thresholding in HLS
    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    
    # Detect white
    lower_white = np.array([0, 200, 0])
    upper_white = np.array([255, 255, 255])
    white_mask = cv2.inRange(hls, lower_white, upper_white)

    # Detect yellow
    lower_yellow = np.array([15, 30, 115])
    upper_yellow = np.array([35, 204, 255])
    yellow_mask = cv2.inRange(hls, lower_yellow, upper_yellow)

    mask = cv2.bitwise_or(white_mask, yellow_mask)
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

    # Convert to grayscale, blur, edge detection
    gray = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    roi = region_of_interest(edges)

    lines = cv2.HoughLinesP(
        roi,
        rho=1,
        theta=np.pi/180,
        threshold=50,
        minLineLength=50,
        maxLineGap=150
    )

    line_image = np.zeros_like(frame)
    if lines is not None:
        averaged_lines = average_slope_intercept(lines)
        for line in averaged_lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 8)

    combined = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    return combined
