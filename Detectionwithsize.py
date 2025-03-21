import cv2
import os
import numpy as np
from ultralytics import YOLO
import Size  # Import the Size module we created
import roi   # Import the new ROI module

def main():
    # Load YOLO model
    model_path = os.path.join('.', 'runs', 'detect', 'train29', 'weights', 'last.pt')
    model = YOLO(model_path)  # Load the custom model

    # Open webcam
    cap = cv2.VideoCapture(0)  # 0 refers to the default webcam

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    # Try loading existing ROI configuration
    roi_config = roi.load_roi_config()
    
    # If no ROI is configured or user wants to reconfigure
    if roi_config is None:
        print("No ROI configuration found. Please select ROI points...")
        roi_config = roi.select_roi(cap)
        if roi_config is None:
            print("No ROI selected, using full frame.")
    else:
        print("Using existing ROI configuration.")

    # Calibrate the system using the Size module
    pixels_per_mm = Size.calibrate_system(cap)
    print(f"Using calibration value: {pixels_per_mm:.2f} pixels per mm")

    # Detection settings
    threshold = 0.45  # Confidence threshold
    
    # Choose size units (mm or cm)
    use_cm = True  # Set to False to display in mm
    unit = "cm" if use_cm else "mm"
    scale_factor = 0.1 if use_cm else 1.0  # Convert mm to cm if needed

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        original_frame = frame.copy()
        
        # Draw ROI on original frame
        if roi_config:
            frame_with_roi = roi.draw_roi(frame.copy(), roi_config)
            
            # Apply perspective transform to get rectified ROI
            warped_roi, transform_matrix = roi.warp_frame(original_frame, roi_config)
            inverse_matrix = np.linalg.inv(transform_matrix) if transform_matrix is not None else None
            
            # Run detection on the warped (perspective corrected) ROI
            frame_for_detection = warped_roi
        else:
            frame_with_roi = frame
            frame_for_detection = frame
            inverse_matrix = None

        # Perform object detection with YOLO
        results = model(frame_for_detection)[0]

        for result in results.boxes.data.tolist():
            box_x1, box_y1, box_x2, box_y2, score, class_id = result

            if score > threshold:
                # If using ROI, need to convert coordinates back to original frame
                if roi_config and inverse_matrix is not None:
                    # Convert box corners back to original frame coordinates
                    orig_x1, orig_y1 = roi.unwarp_coordinates((box_x1, box_y1), inverse_matrix)
                    orig_x2, orig_y2 = roi.unwarp_coordinates((box_x2, box_y2), inverse_matrix)
                    
                    # Use these coordinates for display on original frame
                    display_x1, display_y1, display_x2, display_y2 = orig_x1, orig_y1, orig_x2, orig_y2
                    
                    # Calculate size based on warped frame (which has consistent scale)
                    width_mm, height_mm = Size.calculate_size(warped_roi, (box_x1, box_y1, box_x2, box_y2), pixels_per_mm)
                else:
                    # If not using ROI, just use the original coordinates
                    display_x1, display_y1, display_x2, display_y2 = int(box_x1), int(box_y1), int(box_x2), int(box_y2)
                    width_mm, height_mm = Size.calculate_size(frame, (box_x1, box_y1, box_x2, box_y2), pixels_per_mm)
                
                # Convert to selected unit
                width_unit = width_mm * scale_factor
                height_unit = height_mm * scale_factor
                
               
                
                # Get object name
                object_name = results.names[int(class_id)].upper()
                
                # Display object name and size
                label = f"{object_name}: {width_unit:.1f}x{height_unit:.1f}{unit}"
                
                # Add dimension lines
                # Add dimension lines
                cv2.line(frame_with_roi, (display_x1, display_y2 + 20), (display_x2, display_y2 + 20), (255, 0, 255), 2)
                cv2.line(frame_with_roi, (display_x2 + 20, display_y1), (display_x2 + 20, display_y2), (255, 0, 255), 2)
                
                # Display text with background for better visibility
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                cv2.rectangle(frame_with_roi, (display_x1, display_y1 - 25), 
                             (display_x1 + text_size[0], display_y1), (0, 255, 0), -1)
                cv2.putText(frame_with_roi, label, (display_x1, display_y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

        # Display calibration information
        cv2.putText(frame_with_roi, f"Calibration: {pixels_per_mm:.2f} px/mm", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Show both original frame with ROI and the warped view (for debugging)
        cv2.imshow("Object Detection with Size", frame_with_roi)
        if roi_config and 'warped_roi' in locals() and warped_roi is not None:
            cv2.imshow("Perspective Corrected ROI", warped_roi)

        # Press 'q' to exit, 'c' to recalibrate, 'r' to redefine ROI
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):  # Option to recalibrate
            pixels_per_mm = Size.calibrate_system(cap)
        elif key == ord('r'):  # Option to redefine ROI
            roi_config = roi.select_roi(cap)

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()