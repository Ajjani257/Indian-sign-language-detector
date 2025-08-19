import cv2
import numpy as np
import mediapipe as mp
import joblib


model = joblib.load("isl_landmark_model_2hands.pkl")
label_encoder = joblib.load("label_encoder_2hands.pkl")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Start webcam
cap = cv2.VideoCapture(0)
print(" Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Webcam frame not captured.")
        break

    # Flip and convert
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hands
    results = hands.process(rgb_frame)
    landmarks_flattened = []

    if results.multi_hand_landmarks:
        # Extract landmarks for up to 2 hands
        for hand_landmarks in results.multi_hand_landmarks[:2]:
            for lm in hand_landmarks.landmark:
                landmarks_flattened.extend([lm.x, lm.y, lm.z])

        # Pad to 126 (21 landmarks × 3 coords × 2 hands)
        while len(landmarks_flattened) < 126:
            landmarks_flattened.extend([0.0, 0.0, 0.0])

        landmarks_flattened = landmarks_flattened[:126]

        # Predict
        data_np = np.array(landmarks_flattened).reshape(1, -1)
        y_pred = model.predict(data_np)
        pred_label = label_encoder.inverse_transform(y_pred)[0]
        prob = np.max(model.predict_proba(data_np))

        # Display prediction
        text = f"Pred: {pred_label} ({prob * 100:.1f}%)"
        cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 255, 0), 2, cv2.LINE_AA)

        # Draw hand landmarks
        for hand_landmarks in results.multi_hand_landmarks[:2]:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    else:
        cv2.putText(frame, "No hand detected", (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow("ISL Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
hands.close()
