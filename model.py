from sklearn.svm import LinearSVC
import numpy as np
import cv2 as cv


class Model:
    def __init__(self):
        self.model = LinearSVC() #Support vector machines (SVMs), used for classification and regression tasks.
        self.is_trained = False

        # Fixed image size for all training & prediction
        self.img_width = 150
        self.img_height = 150
        self.flat_size = self.img_width * self.img_height

    def _load_and_flatten(self, path):
        # Load image, convert to grayscale, resize to fixed size,and return flattened 1D numpy array.
        img = cv.imread(path, cv.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not read image: {path}")

        img = cv.resize(img, (self.img_width, self.img_height))
        return img.reshape(self.flat_size)

    def train_model(self, counters):
        #Train SVM using images in folders '1' and '2'.
        #counters: [count_for_class1, count_for_class2]

        img_list = []
        class_list = []

        # Class 1 images
        for i in range(1, counters[0]):
            path = f'1/frame{i}.jpg'
            try:
                img_flat = self._load_and_flatten(path)
            except ValueError as e:
                print(e)
                continue
            img_list.append(img_flat)
            class_list.append(1)

        # Class 2 images
        for i in range(1, counters[1]):
            path = f'2/frame{i}.jpg'
            try:
                img_flat = self._load_and_flatten(path)
            except ValueError as e:
                print(e)
                continue
            img_list.append(img_flat)
            class_list.append(2)

        if not img_list:
            print("No training images found! Capture some images first.")
            return

        img_array = np.array(img_list)
        class_array = np.array(class_list)

        self.model.fit(img_array, class_array)
        self.is_trained = True
        print("Model successfully trained!")
        print("Samples:", img_array.shape[0], "Feature length:", img_array.shape[1])

    def predict(self, frame):
        """
        Predict class for a given frame from the camera.
        frame: numpy array, RGB (from camera) or grayscale.
        """
        if not self.is_trained:
            print("Warning: Model not trained yet!")
            return 0  # "Unknown"

        # If color image, convert to grayscale
        if len(frame.shape) == 3:
            # frame is assumed RGB from camera.get_frame()
            gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
        else:
            gray = frame

        # Resize to fixed size and flatten
        gray_resized = cv.resize(gray, (self.img_width, self.img_height))
        img_flat = gray_resized.reshape(1, self.flat_size)

        prediction = self.model.predict(img_flat)
        return prediction[0]
