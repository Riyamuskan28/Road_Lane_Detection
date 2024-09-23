import cv2
import numpy as np

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lane_lines(img, lines, color=[255, 0, 0], thickness=10):
    line_image = np.zeros_like(img)
    
    if lines is None:
        return img
    
    left_lines = []
    right_lines = []
    left_weights = []
    right_weights = []
    
    for line in lines:
        for x1, y1, x2, y2 in line:
            if x2 - x1 == 0:  # Avoid divide by zero error
                continue  # Skip vertical lines
            
            slope = (y2 - y1) / (x2 - x1)
            length = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
            if slope < 0:  # Left lane
                left_lines.append((x1, y1, x2, y2))
                left_weights.append(length)
            else:  # Right lane
                right_lines.append((x1, y1, x2, y2))
                right_weights.append(length)
    
    left_lane = np.average(left_lines, axis=0, weights=left_weights) if len(left_weights) > 0 else None
    right_lane = np.average(right_lines, axis=0, weights=right_weights) if len(right_weights) > 0 else None
    
    y1 = img.shape[0]
    y2 = int(y1 * 0.6)
    
    if left_lane is not None:
        x1_left, y1_left, x2_left, y2_left = left_lane
        if x2_left - x1_left != 0:  # Check again to avoid divide by zero
            slope_left = (y2_left - y1_left) / (x2_left - x1_left)
            x1_left = int((y1 - y1_left) / slope_left + x1_left)
            x2_left = int((y2 - y2_left) / slope_left + x2_left)
            cv2.line(line_image, (x1_left, y1), (x2_left, y2), color, thickness)
    
    if right_lane is not None:
        x1_right, y1_right, x2_right, y2_right = right_lane
        if x2_right - x1_right != 0:  # Check again to avoid divide by zero
            slope_right = (y2_right - y1_right) / (x2_right - x1_right)
            x1_right = int((y1 - y1_right) / slope_right + x1_right)
            x2_right = int((y2 - y2_right) / slope_right + x2_right)
            cv2.line(line_image, (x1_right, y1), (x2_right, y2), color, thickness)
    
    combined_image = cv2.addWeighted(img, 0.8, line_image, 1, 1)
    return combined_image

def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    
    height, width = frame.shape[:2]
    roi_vertices = np.array([[(50, height), (width//2 - 50, height//2 + 50), 
                              (width//2 + 50, height//2 + 50), (width - 50, height)]])
    
    cropped_edges = region_of_interest(edges, roi_vertices)
    lines = cv2.HoughLinesP(cropped_edges, rho=1, theta=np.pi/180, threshold=50, minLineLength=100, maxLineGap=50)
    frame_with_lanes = draw_lane_lines(frame, lines)
    
    return frame_with_lanes

def lane_detection(video_path):
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_with_lanes = process_frame(frame)
        cv2.imshow("Lane Detection", frame_with_lanes)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "C:/Users/dell/Desktop/imagerecog/test.mp4"  # Specify the path to the video
    lane_detection(video_path)
