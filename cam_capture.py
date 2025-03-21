import cv2
import os
import tkinter as tk
from tkinter import messagebox
import threading

class CameraApp:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)  # Open the default webcam (0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "No webcam detected!")
            return
        
        self.is_capturing = False
        self.img_counter = 0
        self.output_dir = "captured_images"
        os.makedirs(self.output_dir, exist_ok=True)

        # GUI Setup
        self.root = tk.Tk()
        self.root.title("Webcam Controller")

        self.status_label = tk.Label(self.root, text="Ready", font=("Arial", 12))
        self.status_label.pack(pady=5)

        self.counter_label = tk.Label(self.root, text="Images: 0", font=("Arial", 12))
        self.counter_label.pack(pady=5)

        self.capture_button = tk.Button(self.root, text="📸 Capture Image", command=self.start_capturing, bg="green", fg="white", font=("Arial", 14))
        self.capture_button.pack(pady=10)

        self.pause_button = tk.Button(self.root, text="⏸️ Pause Capture", command=self.pause_capturing, bg="orange", font=("Arial", 14))
        self.pause_button.pack(pady=10)

        self.exit_button = tk.Button(self.root, text="❌ Exit", command=self.exit_program, bg="red", fg="white", font=("Arial", 14))
        self.exit_button.pack(pady=10)

        self.show_camera_feed()

    def show_camera_feed(self):
        """ Continuously update the live camera feed inside the Tkinter loop. """
        ret, frame = self.cap.read()
        if ret:
            cv2.imshow("Webcam Feed - Press 'q' to Exit", frame)

            if self.is_capturing:
                img_name = f"{self.output_dir}/image_{self.img_counter}.jpg"
                cv2.imwrite(img_name, frame)
                self.img_counter += 1
                self.counter_label.config(text=f"Images: {self.img_counter}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.exit_program()
        else:
            self.root.after(10, self.show_camera_feed)

    def start_capturing(self):
        self.is_capturing = True
        self.status_label.config(text="Status: Capturing", fg="green")
        print("📸 Capturing started...")

    def pause_capturing(self):
        self.is_capturing = False
        self.status_label.config(text="Status: Paused", fg="orange")
        print("⏸️ Capture Paused.")

    def exit_program(self):
        self.is_capturing = False
        self.cap.release()
        cv2.destroyAllWindows()
        self.root.destroy()
        print("❌ Program Exited.")

if __name__ == "__main__":
    app = CameraApp()
    app.root.mainloop()

