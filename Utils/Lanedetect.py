import cv2
import numpy as np

def detect_lanes(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur to reduce noise and smooth the image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect edges using Canny edge detector
    edges = cv2.Canny(blurred, 50, 150)

    # Define region of interest (ROI) to focus on the road area
    height, width = edges.shape
    mask = np.zeros_like(edges)
    polygon = np.array([[
        (100, height),  # Bottom-left point
        (width - 100, height),  # Bottom-right point
        (width // 2, height // 2)  # Top-center point
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    # Detect lines in the masked edges using Hough Transform
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, threshold=50, minLineLength=100, maxLineGap=50)

    # Create an output image to display the lane lines
    output_image = np.copy(image)

    # If lines are detected, draw them on the image
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(output_image, (x1, y1), (x2, y2), (0, 255, 0), 5)

    return output_image