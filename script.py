import cv2
import mediapipe as mp
import json
import os

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Directory containing mudra images
IMAGE_FOLDER = r"C:\Users\sreeb\OneDrive\Documents\GitHub\Open-CV-experimentations\Mudras Images"
OUTPUT_FILE = "mudras_coordinates.json"

hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

data = {}

for filename in os.listdir(IMAGE_FOLDER):
    if not (filename.lower().endswith(".jpg") or filename.lower().endswith(".png") or filename.lower().endswith(".jpeg")):
        continue

    mudra_name = os.path.splitext(filename)[0]  # filename without extension
    image_path = os.path.join(IMAGE_FOLDER, filename)

    # Read and convert image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        coords = []

        for lm in hand_landmarks.landmark:
            coords.extend([lm.x, lm.y, lm.z])

        data[mudra_name] = coords
        print(f"[✅] Extracted landmarks for: {mudra_name}")
    else:
        print(f"[⚠️] No hand detected in: {mudra_name}")

# Save all mudras to JSON
with open(OUTPUT_FILE, "w") as f:
    json.dump(data, f, indent=4)

print(f"\nAll mudra coordinates saved to: {OUTPUT_FILE}")
hands.close()
