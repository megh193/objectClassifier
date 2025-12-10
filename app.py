import tkinter as tk
from tkinter import simpledialog
import cv2 as cv #Computer Vision Library
import os
from PIL import Image, ImageTk

import model
import camera


class App:
    def __init__(self, window_title="Camera Classifier"):
        # Create main window
        self.window = tk.Tk()
        self.window_title = window_title
        self.window.title(self.window_title)

        # Image counters for class 1 and 2
        self.counters = [1, 1]

        # Model and camera
        self.model = model.Model()
        self.auto_predict = False
        self.camera = camera.Camera()

        # Build GUI
        self.init_gui()

        # Update loop delay (ms)
        self.delay = 15
        self.update()

        # Keep window on top
        self.window.attributes("-topmost", True)
        self.window.mainloop()

    def init_gui(self):
        # Canvas to show camera feed
        self.canvas = tk.Canvas(
            self.window,
            width=int(self.camera.width),
            height=int(self.camera.height)
        )
        self.canvas.pack()

        # Auto prediction toggle
        self.btn_toggleauto = tk.Button(
            self.window,
            text="Auto Prediction",
            width=50,
            command=self.auto_predict_toggle
        )
        self.btn_toggleauto.pack(anchor=tk.CENTER, expand=True)

        # Ask for class names
        self.classname_one = simpledialog.askstring(
            "Classname One", "Enter the name of the first class:", parent=self.window
        )
        self.classname_two = simpledialog.askstring(
            "Classname Two", "Enter the name of the second class:", parent=self.window
        )

        # Fallbacks if user closes dialog / presses cancel
        if not self.classname_one:
            self.classname_one = "Class 1"
        if not self.classname_two:
            self.classname_two = "Class 2"

        # Buttons to capture images for each class
        self.btn_class_one = tk.Button(
            self.window,
            text=self.classname_one,
            width=50,
            command=lambda: self.save_for_class(1)
        )
        self.btn_class_one.pack(anchor=tk.CENTER, expand=True)

        self.btn_class_two = tk.Button(
            self.window,
            text=self.classname_two,
            width=50,
            command=lambda: self.save_for_class(2)
        )
        self.btn_class_two.pack(anchor=tk.CENTER, expand=True)

        # Train model button
        self.btn_train = tk.Button(
            self.window,
            text="Train Model",
            width=50,
            command=lambda: self.model.train_model(self.counters)
        )
        self.btn_train.pack(anchor=tk.CENTER, expand=True)

        # Single prediction button
        self.btn_predict = tk.Button(
            self.window,
            text="Predict",
            width=50,
            command=self.predict
        )
        self.btn_predict.pack(anchor=tk.CENTER, expand=True)

        # Reset button
        self.btn_reset = tk.Button(
            self.window,
            text="Reset",
            width=50,
            command=self.reset
        )
        self.btn_reset.pack(anchor=tk.CENTER, expand=True)

        # Label to show predicted class
        self.class_label = tk.Label(self.window, text="CLASS")
        self.class_label.config(font=("Arial", 20))
        self.class_label.pack(anchor=tk.CENTER, expand=True)

    def auto_predict_toggle(self):
        self.auto_predict = not self.auto_predict
        print("Auto prediction:", self.auto_predict)

    def save_for_class(self, class_num: int):
        """
        Capture current frame, convert to grayscale, resize to 150x150,
        and save to folder '1' or '2' depending on class_num.
        """
        ret, frame = self.camera.get_frame()
        if not ret or frame is None:
            print("Failed to capture frame for saving.")
            return

        # Ensure class folders exist
        if not os.path.exists("1"):
            os.mkdir("1")
        if not os.path.exists("2"):
            os.mkdir("2")

        filename = f'{class_num}/frame{self.counters[class_num - 1]}.jpg'

        # Convert to grayscale and resize to fixed size (same as model)
        gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
        gray_resized = cv.resize(
            gray,
            (self.model.img_width, self.model.img_height)
        )

        cv.imwrite(filename, gray_resized)

        print(f"Saved: {filename}")
        self.counters[class_num - 1] += 1

    def reset(self):
        """
        Delete all saved images for both classes and reset counters + model.
        """
        for folder in ['1', '2']:
            if os.path.isdir(folder):
                for file in os.listdir(folder):
                    file_path = os.path.join(folder, file)
                    if os.path.isfile(file_path):
                        os.unlink(file_path)

        self.counters = [1, 1]
        self.model = model.Model()
        self.class_label.config(text="CLASS")
        print("Reset completed: images deleted and model reinitialized.")

    def update(self):
        """
        Periodically called to update camera frame on the canvas
        and (optionally) run auto prediction.
        """
        if self.auto_predict:
            prediction = self.predict()
            # prediction already prints via predict(); here we just trigger it

        ret, frame = self.camera.get_frame()

        if ret and frame is not None:
            img = Image.fromarray(frame)
            self.photo = ImageTk.PhotoImage(image=img)
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.window.after(self.delay, self.update)

    def predict(self):
        """
        Capture a frame from the camera and use the model to predict the class.
        """
        ret, frame = self.camera.get_frame()
        if not ret or frame is None:
            print("Failed to capture frame for prediction.")
            return

        prediction = self.model.predict(frame)

        if prediction == 1:
            self.class_label.config(text=self.classname_one)
            print("Predicted:", self.classname_one)
            return self.classname_one
        elif prediction == 2:
            self.class_label.config(text=self.classname_two)
            print("Predicted:", self.classname_two)
            return self.classname_two
        else:
            self.class_label.config(text="Unknown")
            print("Predicted: Unknown")
            return "Unknown"
