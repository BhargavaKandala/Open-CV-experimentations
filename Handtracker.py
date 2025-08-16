import cv2 as cv
import mediapipe as mp
import pyautogui as pg
import time

last_action_time = 0

def fingers_up(hand_landmarks):
    finger_states = []
    
    # Thumb finger is sideways
    if hand_landmarks[4].x < hand_landmarks[3].x:
        finger_states.append(1)     # Finger is up 
    else:
        finger_states.append(0)     # Finger down
    
    tips = [8, 12, 16, 20]
    
    for tip in tips:
        if hand_landmarks[tip].y < hand_landmarks[tip - 2].y:
            finger_states.append(1)
        else:
            finger_states.append(0)     # Same logic for all finger tips

    return finger_states

def gesture_control(finger_states):
    global last_action_time
    now = time.time()
    
    if now - last_action_time < 1:
        return "Cooldown"
    
    if finger_states == [0,0,0,0,0]:
        pg.press('playpause')
        last_action_time = now
        return "Play / Pause"
    elif finger_states == [1,1,1,1,1]:
        pg.press('playpause')
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
    

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence = 0.8, min_tracking_confidence=0.5)

capture = cv.VideoCapture(0)

while True:
    ret, frame = capture.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Flip & process
    frame = cv.flip(frame, 1)
    results = hands.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB))

    action = "No Hand"

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            finger_states = fingers_up(hand_landmarks.landmark)
            action = gesture_control(finger_states)

    # Show current action text always
    cv.putText(frame, action, (10,70), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv.imshow("Hand Music Control", frame)

    if cv.waitKey(1) & 0xFF == ord("q"):
        break

capture.release()
cv.destroyAllWindows()

cv.waitKey(0)