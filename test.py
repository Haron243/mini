import cv2

cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Camera not accessible")
else:
    ret, frame = cap.read()
    if ret:
        cv2.imshow("Test Frame", frame)
        cv2.waitKey(1)
    cap.release()
    cv2.destroyAllWindows()
