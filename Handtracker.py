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
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume_interface = ctypes.cast(interface, ctypes.POINTER(IAudioEndpointVolume))

last_action_time = 0
is_playing = False

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


# Gesture control
def gesture_control(finger_states):
    global last_action_time, is_playing
    now = time.time()
    
    if now - last_action_time < 1:
        return "Cooldown"
    
    if finger_states == [0,0,0,0,0]:
        pg.press('playpause')
        is_playing = not is_playing
        last_action_time = now
        return "Play / Pause"
    elif finger_states == [1,1,1,1,1]:
        pg.press('playpause')
        is_playing = not is_playing
        last_action_time = now
        return "Play"
    elif finger_states == [1,0,0,0,0]:
        pg.press('nexttrack')
        last_action_time = now
        return "Next Track"
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


# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5)

capture = cv.VideoCapture(0)

while True:
    ret, frame = capture.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame = cv.flip(frame, 1)
    results = hands.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB))

    action = "No Hand"
    left_hand_landmarks, right_hand_landmarks = None, None

    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Left/Right classification
            hand_label = results.multi_handedness[idx].classification[0].label
            if hand_label == "Left":
                left_hand_landmarks = hand_landmarks
            else:
                right_hand_landmarks = hand_landmarks

            # Gesture control
            if hand_label == "Right":
                finger_states = fingers_up(hand_landmarks.landmark)
                action = gesture_control(finger_states)



    # Volume - Left hand only
    if left_hand_landmarks:
        thumb_tip = left_hand_landmarks.landmark[4]
        index_tip = left_hand_landmarks.landmark[8]
        vol_dist, (x1, y1), (x2, y2) = distance(thumb_tip, index_tip, frame)

        cv.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv.circle(frame, (x1, y1), 6, (255, 0, 0), -1)
        cv.circle(frame, (x2, y2), 6, (255, 0, 0), -1)

        
        left_finger_states = fingers_up(left_hand_landmarks.landmark)

        if left_finger_states[4] == 0:
            volume = np.interp(vol_dist, [30, 200], [0.0, 1.0])
            volume_interface.SetMasterVolumeLevelScalar(volume, None)
            frozen_volume = volume
        else: 
            volume = frozen_volume if 'frozen_volume' in locals() else 0.5

        cv.putText(frame, f"Volume: {int(volume*100)}%", (10, 170),
                cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Volume (Left hand only)
        if left_hand_landmarks:
            finger_states_left = fingers_up(left_hand_landmarks.landmark)  
            pinky_open = finger_states_left[4]

            thumb_tip = left_hand_landmarks.landmark[4]
            index_tip = left_hand_landmarks.landmark[8]
            vol_dist, (x1, y1), (x2, y2) = distance(thumb_tip, index_tip, frame)
            
            cv.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv.circle(frame, (x1, y1), 6, (255, 0, 0), -1)
            cv.circle(frame, (x2, y2), 6, (255, 0, 0), -1)

            if not pinky_open:
                volume = np.interp(vol_dist, [30, 200], [0.0, 1.0])
                volume_interface.SetMasterVolumeLevelScalar(volume, None)
            else:
                volume = volume_interface.GetMasterVolumeLevelScalar()

            cv.putText(frame, f"Volume: {int(volume*100)}%", (10, 170),
                    cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
    if left_hand_landmarks and right_hand_landmarks:
        left_index_tip = left_hand_landmarks.landmark[8]
        right_index_tip = right_hand_landmarks.landmark[8]

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