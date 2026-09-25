# Finger Digit Recognition

A computer vision and machine learning project that allows you to draw a digit in the air using your index finger and recognize it using a trained machine learning model.

## Demo

The application uses your webcam to:

1. Detect your hand using MediaPipe.
2. Track the tip of your index finger.
3. Draw the digit as you move your finger.
4. Process the drawing into a 28×28 image.
5. Predict the digit using a trained MNIST model.

## How It Works

```text
Webcam
   ↓
Hand Detection
   ↓
Index Finger Tracking
   ↓
Air Drawing
   ↓
Image Preprocessing
   ↓
ML Model
   ↓
Digit Prediction
