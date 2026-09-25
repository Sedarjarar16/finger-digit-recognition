import cv2
import mediapipe as mp


# -----------------------------
# MediaPipe Hand Tracking
# -----------------------------

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# -----------------------------
# Camera
# -----------------------------

cap = cv2.VideoCapture(0)


# -----------------------------
# Drawing Canvas
# -----------------------------

canvas = None
previous_point = None


# -----------------------------
# Main Loop
# -----------------------------

while True:

    ret, frame = cap.read()

    if not ret:
        print("Could not access the camera.")
        break

    # Create black canvas
    if canvas is None:
        canvas = frame.copy()
        canvas[:] = 0

    # Convert BGR → RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hand
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            # Index fingertip = landmark 8
            index_finger = hand_landmarks.landmark[8]

            h, w, _ = frame.shape

            # Convert normalized coordinates to pixels
            x = int(index_finger.x * w)
            y = int(index_finger.y * h)

            current_point = (x, y)

            # Draw line
            if previous_point is not None:

                cv2.line(
                    canvas,
                    previous_point,
                    current_point,
                    (255, 255, 255),
                    8
                )

            previous_point = current_point

            # Show fingertip
            cv2.circle(
                frame,
                current_point,
                10,
                (0, 255, 0),
                -1
            )

    else:

        # Stop drawing when hand disappears
        previous_point = None


    # -----------------------------
    # Combine Camera + Drawing
    # -----------------------------

    result = cv2.addWeighted(
        frame,
        0.7,
        canvas,
        0.3,
        0
    )

    cv2.imshow(
        "Finger Digit Recognition",
        result
    )


    # -----------------------------
    # Keyboard Controls
    # -----------------------------

    key = cv2.waitKey(1) & 0xFF


    # Q → Quit
    if key == ord("q"):
        break


    # C → Clear
    if key == ord("c"):

        canvas[:] = 0
        previous_point = None


    # S → Save Drawing
    if key == ord("s"):

        cv2.imwrite(
            "drawing.png",
            canvas
        )

        print("Drawing saved as drawing.png")


# -----------------------------
# Cleanup
# -----------------------------

cap.release()
hands.close()
cv2.destroyAllWindows()