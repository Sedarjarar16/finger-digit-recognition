import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

# Canvas for drawing
canvas = None
previous_point = None

while True:
    ret, frame = cap.read()

    if not ret:
        print("Could not access the camera.")
        break

    # Create canvas after knowing frame size
    if canvas is None:
        canvas = frame.copy()
        canvas[:] = 0

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            # Index fingertip
            index_finger = hand_landmarks.landmark[8]

            h, w, _ = frame.shape

            x = int(index_finger.x * w)
            y = int(index_finger.y * h)

            current_point = (x, y)

            # Draw line from previous point to current point
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
            cv2.circle(frame, current_point, 10, (0, 255, 0), -1)

    else:
        previous_point = None

    # Combine camera and drawing
    result = cv2.addWeighted(frame, 0.7, canvas, 0.3, 0)

    cv2.imshow("Finger Digit Recognition", result)

    key = cv2.waitKey(1) & 0xFF

    # Q = quit
    if key == ord("q"):
        break

    # C = clear drawing
    if key == ord("c"):
        canvas[:] = 0
        previous_point = None

cap.release()
hands.close()
cv2.destroyAllWindows()