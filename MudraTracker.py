import cv2 as cv
import mediapipe as mp
import numpy as np
import json
import time
import math
import os

# ---------- SETUP ----------

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5)

DATA_FILE = "mudras_coordinates.json"

# Load saved mudras if available
if os.path.exists(DATA_FILE):
    with open(DATA_FILE, "r") as f:
        mudra_data = json.load(f)
else:
    mudra_data = {}

# ---------- UTILITY FUNCTIONS ----------

def get_finger_states(hand_landmarks):
    """Detect which fingers are up (1=open, 0=closed)."""
    finger_states = []
    
    # Thumb
    if hand_landmarks[4].x < hand_landmarks[3].x:
        finger_states.append(1)
    else:
        finger_states.append(0)

    # Index, Middle, Ring, Pinky
    for tip in [8, 12, 16, 20]:
        if hand_landmarks[tip].y < hand_landmarks[tip - 2].y:
            finger_states.append(1)
        else:
            finger_states.append(0)
    
    return finger_states


def normalize_landmarks(hand_landmarks):
    """Convert hand landmarks to normalized (x, y) list relative to wrist."""
    wrist = hand_landmarks[0]
    landmarks = []
    for lm in hand_landmarks:
        landmarks.append([lm.x - wrist.x, lm.y - wrist.y])
    return np.array(landmarks).flatten().tolist()


def compare_pose(current_pose, saved_pose):
    """Compare two poses using Euclidean distance."""
    current_pose = np.array(current_pose)
    saved_pose = np.array(saved_pose)
    return np.linalg.norm(current_pose - saved_pose)


def recognize_mudra(current_pose):
    """Return mudra name with minimum distance."""
    if not mudra_data:
        return "No Mudras Stored"

    best_match = None
    best_score = float('inf')

    for mudra_name, pose in mudra_data.items():
        score = compare_pose(current_pose, pose)
        if score < best_score:
            best_score = score
            best_match = mudra_name

    # You can tune this threshold
    if best_score < 0.6:
        return best_match
    else:
        return "Unknown"


def save_mudra(name, pose):
    """Save a new mudra to JSON file."""
    mudra_data[name] = pose
    with open(DATA_FILE, "w") as f:
        json.dump(mudra_data, f, indent=4)
    print(f"[✅] Mudra '{name}' saved successfully.")


# ---------- MAIN LOOP ----------

capture = cv.VideoCapture(0)
print("Press 'r' to record a new mudra, 'q' to quit.")

recording = False
record_name = ""
start_time = 0

while True:
    ret, frame = capture.read()
    if not ret:
        print("Failed to capture frame.")
        break

    frame = cv.flip(frame, 1)
    results = hands.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
    mudra_name = "No Hand"

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            current_pose = normalize_landmarks(hand_landmarks.landmark)
            finger_states = get_finger_states(hand_landmarks.landmark)

            # Recognition mode
            if not recording:
                mudra_name = recognize_mudra(current_pose)
            else:
                # During recording mode
                elapsed = time.time() - start_time
                cv.putText(frame, f"Recording '{record_name}'... {int(elapsed)}s", (20, 60),
                           cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                if elapsed > 2:
                    save_mudra(record_name, current_pose)
                    recording = False
                    record_name = ""

    # Display mudra name
    if not recording:
        cv.putText(frame, f"Mudra: {mudra_name}", (20, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv.imshow("Kuchipudi Mudra Recognition", frame)

    key = cv.waitKey(20) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('r'):
        # Trigger recording mode
        record_name = input("Enter mudra name to record: ").strip()
        start_time = time.time()
        recording = True
        print(f"[🎥] Recording new mudra: {record_name} (hold your hand still for 2 seconds)")

capture.release()
cv.destroyAllWindows()