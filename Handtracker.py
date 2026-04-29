import cv2 as cv
import mediapipe as mp
import pyautogui as pg
import time
import numpy as np
import math
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL
import ctypes

devices = AudioUtilities.GetSpeakers()
volume_interface = devices.EndpointVolume

last_action_time = 0
is_playing = False
last_right_gesture = None
last_left_gesture = None

# Finger state detection
def fingers_up(hand_landmarks):
    finger_states = []
    
    # Thumb
    if hand_landmarks[4].x < hand_landmarks[3].x:
        finger_states.append(1)     # up
    else:
        finger_states.append(0)     # down
    
    tips = [8, 12, 16, 20]  
    for tip in tips:
        if hand_landmarks[tip].y < hand_landmarks[tip - 2].y:
            finger_states.append(1)
        else:
            finger_states.append(0)

    return finger_states


# Gesture control (Optimized with Edge-Triggering)
def gesture_control(finger_states):
    global last_action_time, is_playing, last_right_gesture
    now = time.time()
    
    current_gesture = tuple(finger_states)
    
    # Require gesture change to trigger again (edge trigger)
    if current_gesture == last_right_gesture:
        return "Holding Action"
        
    # Small cooldown
    if now - last_action_time < 0.5:
        return "Cooldown"
        
    last_right_gesture = current_gesture
    
    # Play/Pause optimized: toggles only when gesture is entered
    if finger_states == [0,0,0,0,0] or finger_states == [1,1,1,1,1]:
        pg.press('playpause')
        is_playing = not is_playing
        last_action_time = now
        return "Play / Pause"
    # Prev track remains on right hand
    elif finger_states == [0,1,0,0,0]:
        pg.press('prevtrack')
        last_action_time = now
        return "Previous Track"
    else:
        return "No Action"


# Distance utility
def distance(p1, p2, frame):
    x1, y1 = int(p1.x * frame.shape[1]), int(p1.y * frame.shape[0])
    x2, y2 = int(p2.x * frame.shape[1]), int(p2.y * frame.shape[0])
    return math.hypot(x2 - x1, y2 - y1), (x1, y1), (x2, y2)


# MediaPipe Tasks setup
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.8,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
hands = HandLandmarker.create_from_options(options)

# Hand connections for drawing
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

capture = cv.VideoCapture(0)

while True:
    ret, frame = capture.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame = cv.flip(frame, 1)
    
    # Process for Mediapipe
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    results = hands.detect(mp_image)

    action = "No Hand"
    left_hand_landmarks, right_hand_landmarks = None, None
    volume_active = False

    if results.hand_landmarks:
        # Pre-scan for left hand to evaluate volume_active
        for idx, hand_landmarks in enumerate(results.hand_landmarks):
            if results.handedness[idx][0].category_name == "Left":
                l_states = fingers_up(hand_landmarks)
                # Block volume mode if performing Next Track gesture (Index only)
                if l_states == [0, 1, 0, 0, 0]:
                    volume_active = False
                elif l_states[4] == 0:
                    volume_active = True

        for idx, hand_landmarks in enumerate(results.hand_landmarks):
            # Draw landmarks and connections
            for connection in HAND_CONNECTIONS:
                p1 = hand_landmarks[connection[0]]
                p2 = hand_landmarks[connection[1]]
                x1, y1 = int(p1.x * frame.shape[1]), int(p1.y * frame.shape[0])
                x2, y2 = int(p2.x * frame.shape[1]), int(p2.y * frame.shape[0])
                cv.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
            
            for landmark in hand_landmarks:
                x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                cv.circle(frame, (x, y), 4, (0, 0, 255), -1)

            hand_label = results.handedness[idx][0].category_name
            
            # Separate the Left Hand completely
            if hand_label == "Left":
                left_hand_landmarks = hand_landmarks
                left_finger_states = fingers_up(hand_landmarks)
                
                # Check for Next Track gesture (Only index up)
                if left_finger_states == [0, 1, 0, 0, 0]:
                    current_left_gesture = tuple(left_finger_states)
                    if current_left_gesture != last_left_gesture:
                        if time.time() - last_action_time > 0.5:
                            pg.press('nexttrack')
                            last_action_time = time.time()
                            action = "Next Track (Left)"
                    last_left_gesture = current_left_gesture
                else:
                    last_left_gesture = tuple(left_finger_states)
                    
                    # Volume code
                    thumb_tip = left_hand_landmarks[4]
                    index_tip = left_hand_landmarks[8]
                    vol_dist, (x1, y1), (x2, y2) = distance(thumb_tip, index_tip, frame)

                    cv.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv.circle(frame, (x1, y1), 6, (255, 0, 0), -1)
                    cv.circle(frame, (x2, y2), 6, (255, 0, 0), -1)

                    if volume_active:
                        volume = np.interp(vol_dist, [30, 200], [0.0, 1.0])
                        volume_interface.SetMasterVolumeLevelScalar(volume, None)
                    else:
                        volume = volume_interface.GetMasterVolumeLevelScalar()

                    cv.putText(frame, f"Volume: {int(volume*100)}%", (10, 170),
                            cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # ONLY update finger_states if it is the RIGHT hand
            elif hand_label == "Right":
                right_hand_landmarks = hand_landmarks
                finger_states = fingers_up(hand_landmarks)
                # Only perform track changes if the left hand is not in "Volume mode"
                if not volume_active:
                    action = gesture_control(finger_states)
                else:
                    action = "Volume Active"

    if left_hand_landmarks and right_hand_landmarks:
        left_index_tip = left_hand_landmarks[8]
        right_index_tip = right_hand_landmarks[8]

        lx, ly = int(left_index_tip.x * frame.shape[1]), int(left_index_tip.y * frame.shape[0])
        rx, ry = int(right_index_tip.x * frame.shape[1]), int(right_index_tip.y * frame.shape[0])

        if is_playing: 
            num_points = 50
            for i in range(num_points):
                t1 = i / num_points
                t2 = (i+1) / num_points

                x1 = int(lx + (rx - lx) * t1)
                x2 = int(lx + (rx - lx) * t2)
                
                amplitude = 20
                freq = 5
                offset1 = int(amplitude * math.sin(2*math.pi*freq*t1 + time.time()*5))
                offset2 = int(amplitude * math.sin(2*math.pi*freq*t2 + time.time()*5))

                y1 = int(ly + (ry - ly) * t1 + offset1)
                y2 = int(ly + (ry - ly) * t2 + offset2)

                cv.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        else:  
            cv.line(frame, (lx, ly), (rx, ry), (200, 200, 200), 3)

        cv.circle(frame, (lx, ly), 6, (0, 255, 0), -1)
        cv.circle(frame, (rx, ry), 6, (0, 255, 0), -1)

    cv.putText(frame, action, (10, 70),
               cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv.imshow("Hand Music Control", frame)

    if cv.waitKey(20) & 0xFF == ord("q"):
        break

capture.release()
cv.destroyAllWindows()