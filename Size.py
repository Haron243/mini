import cv2
import numpy as np
import time

def enhance_image(img):
    """Pre-process the image to better detect edges"""
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Apply blur to reduce noise
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    
    return blurred

def calculate_size(img, bbox, pixels_per_mm=10):
    """Calculate the size of an object from its bounding box"""
    x1, y1, x2, y2 = [int(coord) for coord in bbox[:4]]
    
    # Extract the region of interest
    roi = img[y1:y2, x1:x2]
    if roi.size == 0:  # Check if ROI is empty
        return 0, 0
        
    # Process the ROI for edge detection
    processed = enhance_image(roi)
    
    # Calculate dimensions in pixels
    height_px = y2 - y1
    width_px = x2 - x1
    
    # Convert to mm (or cm, based on calibration)
    width_mm = width_px / pixels_per_mm
    height_mm = height_px / pixels_per_mm
    
    return width_mm, height_mm

def calibrate_system(cap, known_size_mm=150.0):
    """Calibrate using a reference object of known size in millimeters"""
    print("CALIBRATION MODE: Place a reference object of exactly", known_size_mm, "mm width")
    print("Press 'c' to capture calibration image or 'q' to skip calibration")
    
    pixels_per_mm = 10.0  # Default fallback value
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Draw calibration guide
        height, width = frame.shape[:2]
        cv2.putText(frame, "Place reference object and press 'c'", (30, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow('Calibration', frame)
        
        key = cv2.waitKey(1)
        if key == ord('c'):  # Capture calibration image
            # Let user draw a line across the reference object
            print("Click and drag to measure reference object")
            
            # Create a copy of the frame for drawing
            temp_frame = frame.copy()
            ref_points = []
            
            def mouse_callback(event, x, y, flags, param):
                nonlocal temp_frame, ref_points
                
                if event == cv2.EVENT_LBUTTONDOWN:
                    ref_points = [(x, y)]
                    temp_frame = frame.copy()
                
                elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON and ref_points:
                    temp_frame = frame.copy()
                    cv2.line(temp_frame, ref_points[0], (x, y), (0, 255, 0), 2)
                
                elif event == cv2.EVENT_LBUTTONUP and ref_points:
                    ref_points.append((x, y))
                    cv2.line(temp_frame, ref_points[0], ref_points[1], (0, 255, 0), 2)
                    
                    # Calculate the pixel distance
                    distance_px = np.sqrt((ref_points[1][0] - ref_points[0][0])**2 + 
                                        (ref_points[1][1] - ref_points[0][1])**2)
                    
                    # Calculate pixels per mm
                    if distance_px > 0:
                        pixels_per_mm = distance_px / known_size_mm
                        cv2.putText(temp_frame, f"Calibration: {pixels_per_mm:.2f} px/mm", 
                                    (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            cv2.setMouseCallback('Calibration', mouse_callback)
            
            # Wait for user to complete the measurement
            while len(ref_points) < 2:
                cv2.imshow('Calibration', temp_frame)
                if cv2.waitKey(1) == ord('q'):
                    break
            
            if len(ref_points) == 2:
                # Calculate the pixel distance
                distance_px = np.sqrt((ref_points[1][0] - ref_points[0][0])**2 + 
                                    (ref_points[1][1] - ref_points[0][1])**2)
                
                # Calculate pixels per mm
                pixels_per_mm = distance_px / known_size_mm
                print(f"System calibrated: {pixels_per_mm:.2f} pixels per mm")
                
                # Display the calibration result for confirmation
                cv2.putText(temp_frame, f"Calibration: {pixels_per_mm:.2f} px/mm", 
                            (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow('Calibration', temp_frame)
                cv2.waitKey(2000)  # Show calibration result for 2 seconds
                break
        
        elif key == ord('q'):
            break
    
    cv2.destroyWindow('Calibration')
    return pixels_per_mm

# For standalone testing of the sizing module
if __name__ == "__main__":
    # Test the sizing functionality
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()
    
    # Calibrate
    pixels_per_mm = calibrate_system(cap)
    print(f"Using calibration value: {pixels_per_mm:.2f} pixels per mm")
    
    # Test measurements with manual bounding boxes
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        height, width = frame.shape[:2]
        # Create a test bounding box in the center (for demonstration)
        center_x, center_y = width // 2, height // 2
        box_width, box_height = width // 4, height // 4
        bbox = [center_x - box_width//2, center_y - box_height//2, 
                center_x + box_width//2, center_y + box_height//2]
        
        # Draw the test bounding box
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        
        # Calculate and display size
        obj_width, obj_height = calculate_size(frame, bbox, pixels_per_mm)
        label = f"Size: {obj_width:.1f}x{obj_height:.1f}mm"
        cv2.putText(frame, label, (bbox[0], bbox[1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display calibration info
        cv2.putText(frame, f"Calibration: {pixels_per_mm:.2f} px/mm", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        cv2.imshow("Size Measurement Test", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            pixels_per_mm = calibrate_system(cap)
    
    cap.release()
    cv2.destroyAllWindows()