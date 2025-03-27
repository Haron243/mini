import cv2
import tkinter as tk
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage, Label, Frame
from pathlib import Path
import os
from ultralytics import YOLO
from roi import select_roi, load_roi_config, warp_frame, draw_roi, unwarp_coordinates
from Size import calibrate_system, calculate_size, load_calibration_config
from PIL import Image, ImageTk  # Import PIL for creating placeholder images
import numpy as np

LOG_FILE = os.path.join(os.path.dirname(__file__), "assets", "frame4", "log.txt")

def write_log(message):
    """Append a message to the log file with a newline."""
    with open(LOG_FILE, "a") as f:
        f.write(message + "\n")


class CVISApplication(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("CVIS Application")
        self.geometry("1126x739")
        self.configure(bg="#FFFFFF")
        self.resizable(False, False)
        self.cap = cv2.VideoCapture(0)
        # Path setup for assets
        self.output_path = Path(__file__).parent
        self.assets_path = {
            "main": self.output_path / "assets" / "frame0",
            "specification": self.output_path / "assets" / "frame1",
            "calibration1": self.output_path / "assets" / "frame2",
            "calibration2": self.output_path / "assets" / "frame3",
            "logs": self.output_path / "assets" / "frame4"
        }
        
        # Create asset directories if they don't exist
        for path in self.assets_path.values():
            os.makedirs(path, exist_ok=True)
        
        # Dictionary to store all frames
        self.frames = {}
        
        # Create and configure the container
        self.container = tk.Frame(self)
        self.container.pack(side="top", fill="both", expand=True)
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)
        
        # Create all frames
        frame_classes = (
            CameraPreviewFrame, 
            SpecificationFrame, 
            CalibrationFrame1, 
            CalibrationFrame2, 
            LogsFrame
        )
        
        for F in frame_classes:
            frame = F(self.container, self)
            self.frames[F.__name__] = frame
            frame.grid(row=0, column=0, sticky="nsew")
        
        # Show initial frame
        self.show_frame("CameraPreviewFrame")
    
    def show_frame(self, frame_name):
        """Show the specified frame and update button highlighting"""
        frame = self.frames[frame_name]
        frame.tkraise()
        
        # Update button highlighting in all frames
        for f in self.frames.values():
            f.update_button_highlights(frame_name)
    
    def relative_to_assets(self, frame_type, path):
        """Get the relative path to an asset for a specific frame"""
        return self.assets_path[frame_type] / Path(path)

class BaseFrame(tk.Frame):
    """Base frame with common elements for all frames"""
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        
        # Create canvas
        self.canvas = Canvas(
            self,
            bg="#FFFFFF",
            height=739,
            width=1126,
            bd=0,
            highlightthickness=0,
            relief="ridge"
        )
        self.canvas.place(x=0, y=0)
        
        # Try to load background image (blue gradient)
        try:
            self.bg_image = PhotoImage(
                file=controller.relative_to_assets("main", "image_1.png")
            )
            self.bg = self.canvas.create_image(
                563.0,
                369.0,
                image=self.bg_image
            )
        except:
            # Fallback if image not found
            write_log(f"Fallback image did not load! ")
            self.canvas.create_rectangle(0, 0, 1126, 739, fill="#0047AB")
        
        # Try to load logo
        try:
            self.logo_image = PhotoImage(
                file=controller.relative_to_assets("main", "image_2.png")
            )
            self.logo = self.canvas.create_image(
                181.0,
                65.0,
                image=self.logo_image
            )
        except:
            pass
        
        # Create CVIS title
        self.canvas.create_text(
            37.0,
            6.0,
            anchor="nw",
            text="CVIS",
            fill="#000000",
            font=("Montserrat Bold", 96 * -1)
        )
        
        # Create navigation buttons
        self.create_navigation_buttons()
    
    def create_navigation_buttons(self):
        """Create the navigation buttons on the left side"""
        self.button_data = [
            {
                "name": "button_1", 
                "command": lambda: self.controller.show_frame("CameraPreviewFrame"),
                "y": 130.0,
                "text": "Camera Preview",
                "frame": "CameraPreviewFrame"
            },
            {
                "name": "button_2", 
                "command": lambda: self.controller.show_frame("SpecificationFrame"),
                "y": 222.0,
                "text": "Specification",
                "frame": "SpecificationFrame"
            },
            {
                "name": "button_3", 
                "command": lambda: self.controller.show_frame("CalibrationFrame1"),
                "y": 314.0,
                "text": "Calibration",
                "frame": "CalibrationFrame"  # Changed from list to string
            },
            {
                "name": "button_4", 
                "command": lambda: self.controller.show_frame("LogsFrame"),
                "y": 406.0,
                "text": "Logs",
                "frame": "LogsFrame"
            }
        ]
        
        # Create a mapping of which frames should highlight which buttons
        self.frame_to_button_map = {
            "CameraPreviewFrame": "CameraPreviewFrame",
            "SpecificationFrame": "SpecificationFrame",
            "CalibrationFrame1": "CalibrationFrame",  # Both calibration frames map to the same button
            "CalibrationFrame2": "CalibrationFrame",
            "LogsFrame": "LogsFrame"
        }
        
        self.button_images = []  # Keep references to prevent garbage collection
        self.buttons = {}
        
        for btn in self.button_data:
            try:
                # Try to load button image
                button_image = PhotoImage(
                    file=self.controller.relative_to_assets("main", f"{btn['name']}.png")
                )
                self.button_images.append(button_image)
                
                button = Button(
                    self,
                    image=button_image,
                    borderwidth=0,
                    highlightthickness=0,
                    command=btn["command"],
                    relief="flat"
                )
                
                # Create active/highlighted button image
                highlighted_image = self.create_highlighted_button(btn["text"])
                self.button_images.append(highlighted_image)
                
            except:
                # Fallback if image not found


                button = Button(
                    self,
                    text=btn["text"],
                    borderwidth=0,
                    highlightthickness=0,
                    command=btn["command"],
                    relief="flat",
                    bg="#333333",
                    fg="#FFFFFF",
                    font=("Montserrat Medium", 16)
                )
            
            button.place(
                x=0.0,
                y=btn["y"],
                width=362.0,
                height=92.0
            )
            
            # Store both normal and active states
            self.buttons[btn["frame"]] = {
                "button": button,
                "normal_bg": "#333333",
                "highlighted_bg": "#0066CC",  # Highlighted color
                "normal_fg": "#FFFFFF",
                "highlighted_fg": "#FFFFFF",
                "text": btn["text"]
            }

    def update_button_highlights(self, current_frame):
        """Update the highlighting of buttons based on the current frame"""
        # Get the button ID that should be highlighted for this frame
        button_to_highlight = self.frame_to_button_map.get(current_frame)
        
        # Update each button's appearance
        for btn_id, btn_data in self.buttons.items():
            button = btn_data["button"]
            
            # Check if this button should be highlighted
            is_current = (btn_id == button_to_highlight)
            
            # Update button styling
            try:
                if is_current:
                    button.config(
                        bg=btn_data["highlighted_bg"],
                        fg=btn_data["highlighted_fg"]
                    )
                else:
                    button.config(
                        bg=btn_data["normal_bg"],
                        fg=btn_data["normal_fg"]
                    )
            except:
                write_log(f" Button configuration failed ! ")
                pass  # Skip if configuration fails


class CameraPreviewFrame(BaseFrame):
    """Main camera preview frame (live detection feed)"""
    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        # Use the shared camera capture from the controller
        self.cap = self.controller.cap
        
        # Load the YOLO detection model (update model_path as needed)
        model_path = os.path.join('.', 'runs', 'detect', 'train29', 'weights', 'last.pt')
        self.model = YOLO(model_path)
        
        # Use shared ROI configuration and load calibration value from JSON if not set
        self.roi_config = getattr(controller, "roi_config", None)
        self.pixels_per_mm = getattr(controller, "pixels_per_mm", load_calibration_config())
        
        # Create a canvas image placeholder (400x400 black image)
        placeholder = Image.new("RGB", (400, 400), (0, 0, 0))
        self.camera_image = ImageTk.PhotoImage(placeholder)
        self.camera = self.canvas.create_image(745.0, 327.0, image=self.camera_image)
        
        # Try to load status display image; if not, draw a fallback rectangle.
        try:
            self.status_image = PhotoImage(
                file=controller.relative_to_assets("main", "image_4.png")
            )
            self.status = self.canvas.create_image(745.0, 645.0, image=self.status_image)
        except:
            write_log(f" Error display image for camera preview ! ")
            self.canvas.create_rectangle(545, 600, 945, 690, fill="#555555", outline="")
        
        # Create product count displays (keeping the same style)
        self.canvas.create_text(
            448.0,633.0, anchor="nw", text="Nut",
            fill="#FFFFFF",
            font=("Montserrat Medium", 18 * -1)
        )
        self.canvas.create_text(
            816.0, 633.0, anchor="nw", text="Bolt",
            fill="#FFFFFF",
            font=("Montserrat Medium", 18 * -1)
        )
        
        # dynamically changing nuts or bolts detected
        self.nut_count_text=self.canvas.create_text(
            622.0, 633.0, anchor="nw", text="00",
            fill="#FFFFFF",
            font=("Montserrat Medium", 18 * -1)
        )
        self.bolt_count_text =self.canvas.create_text(
            1021.0, 633.0, anchor="nw", text="00",
            fill="#FFFFFF",
            font=("Montserrat Medium", 18 * -1)
        )
        
        # Start the live detection update loop
        self.update_detection_feed()
    
    def update_detection_feed(self):
        ret, frame = self.cap.read()
        if ret:
            original_frame = frame.copy()
            # Use ROI calibration if available
            if self.roi_config:
                # Draw the ROI on the original frame for visualization
                frame_with_roi = draw_roi(frame.copy(), self.roi_config)
                # Warp the frame to obtain the perspective-corrected ROI
                warped, transform_matrix = warp_frame(original_frame, self.roi_config)
                inverse_matrix = np.linalg.inv(transform_matrix) if transform_matrix is not None else None
                frame_for_detection = warped if warped is not None else frame
            else:
                frame_with_roi = frame
                frame_for_detection = frame
                inverse_matrix = None
            
            # (Optionally, update the calibration value from the controller if it has changed)
            self.pixels_per_mm = getattr(self.controller, "pixels_per_mm", load_calibration_config())
            
            # Run detection using the YOLO model
            results = self.model(frame_for_detection)[0]
            threshold = 0.45
            object_counts = {"NUT": 0, "BOLT": 0}
            use_cm = True
            unit = "cm" if use_cm else "mm"
            scale_factor = 0.1 if use_cm else 1.0
            
            # Loop through detected boxes
            for result in results.boxes.data.tolist():
                box_x1, box_y1, box_x2, box_y2, score, class_id = result
                if score > threshold:
                    # to get how many nuts and bolts are detected
                    object_name = results.names[int(class_id)].upper()
                    if object_name in object_counts:
                        object_counts[object_name] += 1


                    if self.roi_config and inverse_matrix is not None:
                        # Convert box coordinates from warped ROI to original frame coordinates
                        orig_x1, orig_y1 = unwarp_coordinates((box_x1, box_y1), inverse_matrix)
                        orig_x2, orig_y2 = unwarp_coordinates((box_x2, box_y2), inverse_matrix)
                        display_coords = (orig_x1, orig_y1, orig_x2, orig_y2)
                        # Calculate size based on warped frame (consistent scale)
                        width_mm, height_mm = calculate_size(warped, (box_x1, box_y1, box_x2, box_y2), self.pixels_per_mm)
                    else:
                        display_coords = (int(box_x1), int(box_y1), int(box_x2), int(box_y2))
                        width_mm, height_mm = calculate_size(frame, (box_x1, box_y1, box_x2, box_y2), self.pixels_per_mm)
                    
                    width_unit = width_mm * scale_factor
                    height_unit = height_mm * scale_factor
                    object_name = results.names[int(class_id)].upper()
                    label = f"{object_name}: {width_unit:.1f}x{height_unit:.1f}{unit}"
                    
                    

                    # Draw bounding box and dimension lines on frame_with_roi
                    cv2.rectangle(frame_with_roi, (display_coords[0], display_coords[1]), 
                                  (display_coords[2], display_coords[3]), (0, 255, 0), 2)
                    cv2.line(frame_with_roi, (display_coords[0], display_coords[3] + 20), 
                             (display_coords[2], display_coords[3] + 20), (255, 0, 255), 2)
                    cv2.line(frame_with_roi, (display_coords[2] + 20, display_coords[1]), 
                             (display_coords[2] + 20, display_coords[3]), (255, 0, 255), 2)
                    
                    # Draw label with background for readability
                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    cv2.rectangle(frame_with_roi, (display_coords[0], display_coords[1] - 25), 
                                  (display_coords[0] + text_size[0], display_coords[1]), (0, 255, 0), -1)
                    cv2.putText(frame_with_roi, label, (display_coords[0], display_coords[1] - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
            
            # updating the count of nut and bolt after each loop
            self.canvas.itemconfig(self.nut_count_text, text=f"{object_counts['NUT']:02d}")
            self.canvas.itemconfig(self.bolt_count_text, text=f"{object_counts['BOLT']:02d}")

            # Overlay calibration info on the frame
            cv2.putText(frame_with_roi, f"Calibration: {self.pixels_per_mm:.2f} px/mm", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            # Resize final frame to 400x400 for display
            display_frame = cv2.resize(frame_with_roi, (400, 400))
            display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(display_frame)
            self.camera_image = ImageTk.PhotoImage(image=img)
            self.canvas.itemconfig(self.camera, image=self.camera_image)
        
        self.after(30, self.update_detection_feed)
    
    def destroy(self):
        if self.cap.isOpened():
            self.cap.release()
        super().destroy()


class SpecificationFrame(BaseFrame):
    """Product specification frame (based on gui1.py)"""
    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        
        # Create specification title
        self.canvas.create_text(
            579.0,
            65.0,
            anchor="nw",
            text="Specify the product detail",
            fill="#FFFFFF",
            font=("Montserrat Medium", 24 * -1)
        )
        
        # Product selection section
        self.canvas.create_text(
            428.0,
            180.0,
            anchor="nw",
            text="Select the product",
            fill="#FFFFFF",
            font=("Montserrat Medium", 18 * -1)
        )
        
        # Try to load product selection image
        try:
            self.product_image = PhotoImage(
                file=controller.relative_to_assets("specification", "image_3.png")
            )
            self.product = self.canvas.create_image(
                550.0,
                245.0,
                image=self.product_image
            )
        except:
            # Fallback if image not found
            write_log(f" Failed to load Fallback image! ")
            self.canvas.create_rectangle(450, 225, 650, 265, fill="#555555", outline="")
        
        self.canvas.create_text(
            455.0,
            234.0,
            anchor="nw",
            text="Nut",
            fill="#FFFFFF",
            font=("Montserrat Medium", 18 * -1)
        )
        
        # Try to load product dropdown button
        try:
            self.product_dropdown_image = PhotoImage(
                file=controller.relative_to_assets("specification", "image_5.png")
            )
            self.product_dropdown = self.canvas.create_image(
                640.5,
                245.0,
                image=self.product_dropdown_image
            )
        except:
            pass
        
        # Size selection section
        self.canvas.create_text(
            428.0,
            350.0,
            anchor="nw",
            text="Select the size",
            fill="#FFFFFF",
            font=("Montserrat Medium", 18 * -1)
        )
        
        # Try to load size selection image
        try:
            self.size_image = PhotoImage(
                file=controller.relative_to_assets("specification", "image_4.png")
            )
            self.size = self.canvas.create_image(
                550.0,
                415.0,
                image=self.size_image
            )
        except:
            # Fallback if image not found
            write_log(f" Error not Fallback image! ")
            self.canvas.create_rectangle(450, 395, 650, 435, fill="#555555", outline="")
        
        self.canvas.create_text(
            455.0,
            404.0,
            anchor="nw",
            text="00.0 mm",
            fill="#FFFFFF",
            font=("Montserrat Medium", 18 * -1)
        )
        
        # Try to load size dropdown button
        try:
            self.size_dropdown_image = PhotoImage(
                file=controller.relative_to_assets("specification", "image_6.png")
            )
            self.size_dropdown = self.canvas.create_image(
                640.5,
                415.0,
                image=self.size_dropdown_image
            )
        except:
            pass

class CalibrationFrame1(BaseFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        self.cap = self.controller.cap
        self.roi_config = None  # To store ROI calibration result

        # Create a canvas image placeholder for the live feed (400x400 black image)
        placeholder = Image.new("RGB", (400, 400), (0, 0, 0))
        self.camera_image = ImageTk.PhotoImage(placeholder)
        self.camera = self.canvas.create_image(745.0, 327.0, image=self.camera_image)
        
        # Instruction text
        self.canvas.create_text(
            423.0,
            633.0,
            anchor="nw",
            text="1/2, Click 4 points for Marking the Region of interest",
            fill="#FFFFFF",
            font=("Montserrat Medium", 18 * -1)
        )
        
        # Add a "Select ROI" button
        self.roi_button = Button(
            self,
            text="Select ROI",
            command=self.run_roi_calibration,
            bg="#0066CC",
            fg="#FFFFFF",
            font=("Montserrat Medium", 16)
        )
        self.roi_button.place(x=50, y=650, width=150, height=40)
        
        # (If you already have a Next button in your design, it can remain here.)
        # For example, if desired:
        self.next_button = Button(
            self,
            text="Next",
            command=lambda: controller.show_frame("CalibrationFrame2"),
            bg="#0066CC",
            fg="#FFFFFF",
            font=("Montserrat Medium", 16)
        )
        self.next_button.place(x=993, y=612, width=96, height=66)
        
        # Start the live feed update loop
        self.update_roi_feed()

    def run_roi_calibration(self):
        """Run the ROI selection process using the current camera capture."""
        roi_config = select_roi(self.cap)
        if roi_config:
            print("ROI calibration completed:", roi_config)
            self.roi_config = roi_config
            self.controller.roi_config = roi_config  # Optionally save globally
        else:
            print("ROI calibration canceled or failed.")

    def update_roi_feed(self):
        """Grab a frame, apply ROI warp if available, and update the canvas."""
        ret, frame = self.cap.read()
        if ret:
            if self.roi_config:
                warped, _ = warp_frame(frame, self.roi_config)
                display_frame = warped if warped is not None else frame
            else:
                display_frame = frame

            display_frame = cv2.resize(display_frame, (400, 400))
            display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(display_frame)
            self.camera_image = ImageTk.PhotoImage(image=img)
            self.canvas.itemconfig(self.camera, image=self.camera_image)
        self.after(30, self.update_roi_feed)

    def destroy(self):
        if self.cap.isOpened():
            self.cap.release()
        super().destroy()


class CalibrationFrame2(BaseFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        self.cap = self.controller.cap
        self.calibrated_pixels = None  # Store calibration value
        
        # Create a canvas image placeholder for the live feed (400x400 black image)
        placeholder = Image.new("RGB", (400, 400), (0, 0, 0))
        self.camera_image = ImageTk.PhotoImage(placeholder)
        self.camera = self.canvas.create_image(745.0, 327.0, image=self.camera_image)
        
        self.canvas.create_text(
            423.0,
            633.0,
            anchor="nw",
            text="2/2, Measure 15 cm using a scale",
            fill="#FFFFFF",
            font=("Montserrat Medium", 18 * -1)
        )
        
        # Add a "Calibrate Size" button
        self.size_button = Button(
            self,
            text="Calibrate Size",
            command=self.run_size_calibration,
            bg="#0066CC",
            fg="#FFFFFF",
            font=("Montserrat Medium", 16)
        )
        self.size_button.place(x=50, y=650, width=150, height=40)
        
        # Add a "Finish" button to proceed after calibration
        self.finish_button = Button(
            self,
            text="Finish",
            command=lambda: controller.show_frame("CameraPreviewFrame"),
            bg="#0066CC",
            fg="#FFFFFF",
            font=("Montserrat Medium", 16)
        )
        self.finish_button.place(x=998, y=612, width=96, height=66)
        
        # Start updating the live feed on this panel
        self.update_size_feed()
        
    def run_size_calibration(self):
        """Run the size calibration process."""
        pixels_per_mm = calibrate_system(self.cap, known_size_mm=150.0)
        print(f"Size calibration completed: {pixels_per_mm:.2f} px/mm")
        self.calibrated_pixels = pixels_per_mm
        self.controller.pixels_per_mm = pixels_per_mm

    def update_size_feed(self):
        """Grab a frame and update the panel."""
        ret, frame = self.cap.read()
        if ret:
            display_frame = cv2.resize(frame, (400, 400))
            display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(display_frame)
            self.camera_image = ImageTk.PhotoImage(image=img)
            self.canvas.itemconfig(self.camera, image=self.camera_image)
        self.after(30, self.update_size_feed)
    
    def destroy(self):
        if self.cap.isOpened():
            self.cap.release()
        super().destroy()

class LogsFrame(BaseFrame):
    """Logs frame (based on gui4.py)"""
    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        
        # Create log display area
        self.log_text = Text(
            self,
            width=50,
            height=20,
            font=("Montserrat Medium", 12),
            bg="#333333",
            fg="#FFFFFF",
            bd=0
        )
           

        self.log_text.place(x=400, y=150)

        log_file_path = controller.relative_to_assets("logs", "log.txt")
        if os.path.exists(log_file_path):
            with open(log_file_path, "r") as f:
                log_content = f.read()
        else:
            log_content = "No logs found.\n"
        
        self.log_text.insert("end", log_content)
        self.log_text.config(state='disabled') 

        # Try to load status display image
        try:
            self.status_image = PhotoImage(
                file=controller.relative_to_assets("logs", "image_4.png")
            ) 
            self.status = self.canvas.create_image(
                745.0,
                645.0,
                image=self.status_image
            )
        except:
            # Fallback if image not found
            write_log(f" Log image is not found! ")
            self.canvas.create_rectangle(545, 600, 945, 690, fill="#555555", outline="")
        
        # Create product count displays
        self.canvas.create_text(
            448.0,
            633.0,
            anchor="nw",                
            text="Nut",
            fill="#FFFFFF",
            font=("Montserrat Medium", 18 * -1)
        )
        
        self.canvas.create_text(
            816.0,
            633.0,
            anchor="nw",
            text="Bolt",
            fill="#FFFFFF",
            font=("Montserrat Medium", 18 * -1)
        )
        
        self.canvas.create_text(
            622.0,
            633.0,
            anchor="nw",
            text="00",
            fill="#FFFFFF",
            font=("Montserrat Medium", 18 * -1)
        )
        
        self.canvas.create_text(
            1021.0,
            633.0,
            anchor="nw",
            text="00",
            fill="#FFFFFF",
            font=("Montserrat Medium", 18 * -1)
        )

if __name__ == "__main__":
    app = CVISApplication()
    app.mainloop()
