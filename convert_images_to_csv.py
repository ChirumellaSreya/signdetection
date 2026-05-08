import cv2
import os
import pandas as pd
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Load model
base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1
)

detector = vision.HandLandmarker.create_from_options(options)

data = []
DATASET_PATH = "dataset"

for folder in os.listdir(DATASET_PATH):
    folder_path = os.path.join(DATASET_PATH, folder)

    if not os.path.isdir(folder_path):
        continue

    label = folder.split('-')[0]
    print(f"Processing: {folder}")

    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)

        image = cv2.imread(img_path)
        if image is None:
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        result = detector.detect(mp_image)

        if result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                row = []
                for lm in hand_landmarks:
                    row.extend([lm.x, lm.y, lm.z])

                row.append(label)
                data.append(row)

# Create columns
columns = []
for i in range(21):
    columns += [f'x{i}', f'y{i}', f'z{i}']
columns.append('label')

df = pd.DataFrame(data, columns=columns)
df.to_csv("dataset.csv", index=False)

print("✅ dataset.csv created!")
print("Total samples:", len(df))