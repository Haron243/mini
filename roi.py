import cv2
import numpy as np
import os
import json

# File to store ROI coordinates
ROI_CONFIG_FILE = "roi_config.json"

def select_roi(cap):
    """
    Allow user to select a quadrilateral ROI by clicking 4 points
    Handles perspective correction and saves ROI to a config file
    """
    print("ROI SELECTION: Click 4 points to define the conveyor belt region")
    
    # Capture a frame for selection
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame during ROI selection.")
        return None
        
    # Make a copy for drawing
    selection_frame = frame.copy()
    
    # Initialize ROI coordinates
    points = []
    
    # Define selection callback
    def mouse_callback(event, x, y, flags, param):
        nonlocal points, selection_frame
        
        if event == cv2.EVENT_LBUTTONDOWN:
            # Add point on click
            points.append((x, y))
            # Update display
            selection_frame = frame.copy()
            
            # Draw all points collected so far
            for i, pt in enumerate(points):
                cv2.circle(selection_frame, pt, 5, (0, 255, 0), -1)
                cv2.putText(selection_frame, str(i+1), 
                          (pt[0]+10, pt[1]-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Draw lines between points
            if len(points) > 1:
                for i in range(len(points) - 1):
                    cv2.line(selection_frame, points[i], points[i+1], (0, 255, 0), 2)
                
                # Close the polygon if we have 4 points
                if len(points) == 4:
                    cv2.line(selection_frame, points[3], points[0], (0, 255, 0), 2)
                    cv2.putText(selection_frame, "Press ENTER to confirm or ESC to reset", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Set up the window
    cv2.namedWindow("Select Conveyor Belt Region")
    cv2.setMouseCallback("Select Conveyor Belt Region", mouse_callback)
    
    # Display instructions
    cv2.putText(selection_frame, "Click 4 points to define the conveyor region", 
              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Wait for user selection
    while True:
        cv2.imshow("Select Conveyor Belt Region", selection_frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == 13 and len(points) == 4:  # ENTER key and 4 points selected
            break
        elif key == 27:  # ESC key
            points = []
            selection_frame = frame.copy()
            cv2.putText(selection_frame, "Selection reset. Click 4 points.", 
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    cv2.destroyWindow("Select Conveyor Belt Region")
    
    if len(points) == 4:
        # Sort points in order: top-left, top-right, bottom-right, bottom-left
        # (This is important for perspective transform)
        points = order_points(np.array(points))
        
        # Compute width and height of the perspective-corrected region
        width_a = np.sqrt(((points[1][0] - points[0][0]) ** 2) + ((points[1][1] - points[0][1]) ** 2))
        width_b = np.sqrt(((points[2][0] - points[3][0]) ** 2) + ((points[2][1] - points[3][1]) ** 2))
        max_width = max(int(width_a), int(width_b))
        
        height_a = np.sqrt(((points[3][0] - points[0][0]) ** 2) + ((points[3][1] - points[0][1]) ** 2))
        height_b = np.sqrt(((points[2][0] - points[1][0]) ** 2) + ((points[2][1] - points[1][1]) ** 2))
        max_height = max(int(height_a), int(height_b))
        
        # Convert points to list of tuples for JSON serialization
        points_list = [(int(x), int(y)) for x, y in points]
        
        # Create ROI configuration
        roi_config = {
            "points": points_list,
            "width": max_width,
            "height": max_height
        }
        
        # Save to file
        with open(ROI_CONFIG_FILE, 'w') as f:
            json.dump(roi_config, f)
            
        print(f"ROI configuration saved to {ROI_CONFIG_FILE}")
        return roi_config
    
    return None

def order_points(pts):
    """
    Order points in: top-left, top-right, bottom-right, bottom-left order
    Important for perspective transform
    """
    # Initialize result
    rect = np.zeros((4, 2), dtype="float32")
    
    # Top-left will have the smallest sum, bottom-right will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # Top-right will have the smallest difference, bottom-left will have the largest
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect

def load_roi_config():
    """
    Load ROI configuration from file
    Returns None if config doesn't exist
    """
    if os.path.exists(ROI_CONFIG_FILE):
        try:
            with open(ROI_CONFIG_FILE, 'r') as f:
                roi_config = json.load(f)
            print(f"Loaded ROI configuration from {ROI_CONFIG_FILE}")
            return roi_config
        except Exception as e:
            print(f"Error loading ROI configuration: {e}")
    return None

def get_perspective_transform(roi_config):
    """
    Get perspective transform matrix from ROI config
    """
    if not roi_config:
        return None
    
    # Source points
    src_pts = np.array(roi_config["points"], dtype="float32")
    
    # Destination points (rectangle)
    dst_pts = np.array([
        [0, 0],
        [roi_config["width"] - 1, 0],
        [roi_config["width"] - 1, roi_config["height"] - 1],
        [0, roi_config["height"] - 1]
    ], dtype="float32")
    
    # Compute perspective transform matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    
    return M, roi_config["width"], roi_config["height"]

def warp_frame(frame, roi_config):
    """
    Apply perspective transform to extract and rectify the ROI
    """
    if not roi_config or frame is None:
        return frame, None
    
    # Get transform matrix
    M, width, height = get_perspective_transform(roi_config)
    
    # Warp the frame
    warped = cv2.warpPerspective(frame, M, (width, height))
    
    return warped, M

def unwarp_coordinates(point, inverse_matrix):
    """
    Convert a point from the warped (rectified) coordinates back to original frame coordinates
    """
    if inverse_matrix is None:
        return point
        
    px, py = point
    # Convert to homogeneous coordinates
    p_array = np.array([[[px, py]]], dtype=np.float32)
    # Apply inverse perspective transform
    original_p = cv2.perspectiveTransform(p_array, inverse_matrix)
    
    return (int(original_p[0][0][0]), int(original_p[0][0][1]))

def draw_roi(frame, roi_config):
    """
    Draw the ROI boundary on the original frame
    """
    if not roi_config or frame is None:
        return frame
        
    # Draw the quadrilateral
    points = np.array(roi_config["points"], np.int32)
    cv2.polylines(frame, [points], True, (255, 0, 0), 2)
    
    return frame

# Test function to run this module standalone
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()
    
    print("\n== ROI CALIBRATION TOOL ==")
    print("Running roi.py directly - will force recalibration.")
    
    # When run directly, always force a new ROI calibration
    print("Select 4 points to define the conveyor belt region.")
    roi_config = select_roi(cap)
    
    if roi_config:
        print("New ROI calibration saved successfully.")
        print("Testing new calibration with live video...")
        
        # Test the ROI with live video
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Draw ROI on original frame
            original_with_roi = draw_roi(frame.copy(), roi_config)
            
            # Apply perspective transform
            warped, M = warp_frame(frame, roi_config)
            
            # Display both frames
            cv2.imshow("Original with ROI", original_with_roi)
            if warped is not None:
                cv2.imshow("Perspective Corrected ROI", warped)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    else:
        print("ROI calibration canceled or failed.")
    
    cap.release()
    cv2.destroyAllWindows()