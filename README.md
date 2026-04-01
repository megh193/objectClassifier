# Object Classifier

A real-time object classifier built with Python that uses your webcam to capture images, train a machine learning model, and predict object classes live — all through a simple desktop GUI.

## How It Works

1. Launch the app and enter names for two object classes
2. Point your webcam at an object and click the class button to capture training images
3. Click **Train Model** to train an SVM classifier on the captured images
4. Click **Predict** or enable **Auto Prediction** to classify objects in real time

## Tech Stack

- **GUI** — Tkinter
- **Computer Vision** — OpenCV
- **Machine Learning** — scikit-learn (LinearSVC)
- **Image Processing** — Pillow, NumPy

## Project Structure

```
objectClassifier/
├── main.py       # Entry point
├── app.py        # GUI and application logic
├── camera.py     # Webcam capture handler
├── model.py      # SVM model — training and prediction
└── .gitignore
```

## Requirements

- Python 3.8+
- Webcam

Install dependencies:

```bash
pip install opencv-python pillow scikit-learn numpy
```

## Running the App

```bash
python main.py
```

## Usage

| Button | Action |
|---|---|
| Class 1 / Class 2 | Capture a training image for that class |
| Train Model | Train the SVM on captured images |
| Predict | Predict the current frame once |
| Auto Prediction | Toggle continuous real-time prediction |
| Reset | Clear all images and reset the model |

## Notes

- Captured images are saved as 150×150 grayscale JPGs in folders `1/` and `2/` (excluded from git)
- At least a few images per class are recommended before training for accurate predictions
