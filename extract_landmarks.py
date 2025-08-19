import cv2
import mediapipe as mp
import os
import pandas as pd
from tqdm import tqdm

# Setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

data = []
labels = []
classes = sorted(os.listdir("Indian"))

for label in tqdm(classes):
    folder = os.path.join("Indian", label)
    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)

        if result.multi_hand_landmarks:
            sample = []
            for hand_landmarks in result.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    sample.extend([lm.x, lm.y, lm.z])

            # Pad with zeros if only 1 hand
            while len(sample) < 126:
                sample.extend([0.0] * 3)

            data.append(sample)
            labels.append(label)


df = pd.DataFrame(data)
df['label'] = labels
df.to_csv("isl_landmarks_2hands.csv", index=False)
print("Saved new 2-hand landmark CSV.")
