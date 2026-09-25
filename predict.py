import cv2
import mediapipe as mp
import joblib
import numpy as np


# =========================
# Load trained model
# =========================

model = joblib.load("models/digit_model.pkl")


# =========================
# MediaPipe setup
# =========================

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# =========================
# Camera
# =========================

cap = cv2.VideoCapture(0)


# =========================
# Drawing variables
# =========================

canvas = None
previous_point = None

smooth_x = None
smooth_y = None

smoothing = 0.25


# =========================
# Prediction variables
# =========================

last_prediction = None
last_confidence = None


# =========================
# Check if index finger is up
# =========================

def is_index_finger_up(hand_landmarks):

    index_tip = hand_landmarks.landmark[8]
    index_pip = hand_landmarks.landmark[6]

    return index_tip.y < index_pip.y


# =========================
# Main loop
# =========================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Could not access the camera.")
        break

    # Mirror camera
    frame = cv2.flip(frame, 1)

    # Create canvas
    if canvas is None:
        canvas = np.zeros_like(frame)


    # =========================
    # Hand tracking
    # =========================

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)


    drawing = False


    if results.multi_hand_landmarks:

        hand_landmarks = results.multi_hand_landmarks[0]

        # Draw hand landmarks
        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS
        )


        # =========================
        # Index finger position
        # =========================

        index_tip = hand_landmarks.landmark[8]

        raw_x = int(index_tip.x * frame.shape[1])
        raw_y = int(index_tip.y * frame.shape[0])


        # =========================
        # Smooth movement
        # =========================

        if smooth_x is None:
            smooth_x = raw_x
            smooth_y = raw_y

        smooth_x = int(
            smooth_x + smoothing * (raw_x - smooth_x)
        )

        smooth_y = int(
            smooth_y + smoothing * (raw_y - smooth_y)
        )


        current_point = (smooth_x, smooth_y)


        # =========================
        # Drawing control
        # =========================

        drawing = is_index_finger_up(hand_landmarks)


        if drawing:

            if previous_point is not None:

                cv2.line(
                    canvas,
                    previous_point,
                    current_point,
                    (255, 255, 255),
                    6,
                    cv2.LINE_AA
                )

            previous_point = current_point

        else:

            previous_point = None


        # Draw fingertip
        cv2.circle(
            frame,
            current_point,
            8,
            (0, 255, 0),
            -1
        )


    else:

        previous_point = None


    # =========================
    # Show drawing on camera
    # =========================

    frame_with_drawing = cv2.add(
        frame,
        canvas
    )


    # =========================
    # Status
    # =========================

    if drawing:

        cv2.putText(
            frame_with_drawing,
            "DRAWING",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    else:

        cv2.putText(
            frame_with_drawing,
            "RAISE INDEX FINGER TO DRAW",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )


    # =========================
    # Show prediction
    # =========================

    if last_prediction is not None:

        cv2.putText(
            frame_with_drawing,
            f"Prediction: {last_prediction}",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 255, 0),
            3
        )

        cv2.putText(
            frame_with_drawing,
            f"Confidence: {last_confidence * 100:.1f}%",
            (20, 135),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )


    # =========================
    # Controls
    # =========================

    cv2.putText(
        frame_with_drawing,
        "C: Clear   P: Predict   Q: Quit",
        (20, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    # =========================
    # Show window
    # =========================

    cv2.imshow(
        "Finger Digit Recognition",
        frame_with_drawing
    )


    # =========================
    # Keyboard controls
    # =========================

    key = cv2.waitKey(1) & 0xFF


    # Clear drawing
    if key == ord("c"):

        canvas = np.zeros_like(frame)

        previous_point = None

        last_prediction = None
        last_confidence = None


    # Predict digit
    elif key == ord("p"):

        # Convert drawing to grayscale
        gray = cv2.cvtColor(
            canvas,
            cv2.COLOR_BGR2GRAY
        )


        # Threshold
        _, threshold = cv2.threshold(
            gray,
            50,
            255,
            cv2.THRESH_BINARY
        )


        # Find contours
        contours, _ = cv2.findContours(
            threshold,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )


        if contours:

            # Largest contour
            contour = max(
                contours,
                key=cv2.contourArea
            )


            x, y, w, h = cv2.boundingRect(
                contour
            )


            # Crop digit
            digit = threshold[
                y:y+h,
                x:x+w
            ]


            # Keep aspect ratio
            scale = min(
                20 / w,
                20 / h
            )


            new_w = max(
                1,
                int(w * scale)
            )

            new_h = max(
                1,
                int(h * scale)
            )


            digit = cv2.resize(
                digit,
                (new_w, new_h)
            )


            # Create 28x28 image
            final_image = np.zeros(
                (28, 28),
                dtype=np.uint8
            )


            # Center digit
            x_offset = (28 - new_w) // 2
            y_offset = (28 - new_h) // 2


            final_image[
                y_offset:y_offset + new_h,
                x_offset:x_offset + new_w
            ] = digit


            # Normalize
            input_data = final_image / 255.0


            # Reshape for model
            input_data = input_data.reshape(
                1,
                784
            )


            # =========================
            # Prediction
            # =========================

            last_prediction = model.predict(
                input_data
            )[0]


            probabilities = model.predict_proba(
                input_data
            )


            last_confidence = probabilities.max()


            print(
                f"Prediction: {last_prediction}"
            )

            print(
                f"Confidence: {last_confidence * 100:.1f}%"
            )


        else:

            print("No digit detected.")


    # Quit
    elif key == ord("q"):

        break


# =========================
# Cleanup
# =========================

cap.release()

hands.close()

cv2.destroyAllWindows()