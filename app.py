# import cv2
# import mediapipe as mp
# import numpy as np
# import random
# import colorsys

# # Initialize video capture
# cap = cv2.VideoCapture(0)

# if not cap.isOpened():
#     print("Error: Camera not accessible")
#     exit()

# # Initialize Mediapipe Hands module
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(static_image_mode=False,
#                        max_num_hands=1,
#                        min_detection_confidence=0.7,
#                        min_tracking_confidence=0.7)

# # Particle class
# class Particle:
#     def __init__(self, position):
#         self.position = np.array(position, dtype=np.float32)
#         self.velocity = np.random.uniform(-1, 1, size=2) * 2
#         self.lifetime = random.randint(30, 60)
#         self.max_lifetime = self.lifetime
#         self.hue = random.uniform(0, 1)

#     def update(self):
#         self.position += self.velocity
#         self.lifetime -= 1
#         self.hue += 0.01
#         if self.hue > 1:
#             self.hue -= 1

#     def get_color(self):
#         rgb = colorsys.hsv_to_rgb(self.hue, 1.0, 1.0)
#         alpha = self.lifetime / self.max_lifetime
#         return (np.array(rgb) * 255 * alpha).astype(np.uint8)

# # Particle list
# particles = []

# # Previous hand center
# prev_center = None

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     frame = cv2.flip(frame, 1)
#     h, w, _ = frame.shape

#     # Convert frame to RGB
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     # Process frame
#     results = hands.process(rgb_frame)

#     # Create overlay for particles
#     overlay = frame.copy()
#     overlay = cv2.addWeighted(overlay, 0.5, overlay, 0, 0)

#     # Check for hands
#     if results.multi_hand_landmarks:
#         for hand_landmarks in results.multi_hand_landmarks:
#             # Get center of palm (landmark 9)
#             lm = hand_landmarks.landmark[9]
#             cx, cy = int(lm.x * w), int(lm.y * h)

#             # Calculate hand speed
#             if prev_center is not None:
#                 speed = np.linalg.norm(np.array([cx, cy]) - prev_center)
#             else:
#                 speed = 0

#             prev_center = np.array([cx, cy])

#             # Generate particles based on speed
#             num_new_particles = int(min(10, speed * 5)) + 3
#             for _ in range(num_new_particles):
#                 particles.append(Particle((cx, cy)))

#     # Update and draw particles
#     for particle in particles[:]:
#         particle.update()
#         if particle.lifetime <= 0:
#             particles.remove(particle)
#             continue

#         color = particle.get_color().tolist()
#         cv2.circle(overlay,
#                    (int(particle.position[0]), int(particle.position[1])),
#                    10,
#                    color,
#                    -1)

#     # Blend overlay with original frame
#     frame = cv2.addWeighted(frame, 0.8, overlay, 0.2, 0)

#     # Show frame
#     cv2.imshow('Hand AR Effect', frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Cleanup
# cap.release()
# cv2.destroyAllWindows()


import cv2
import mediapipe as mp
import pyautogui
import time

# Mediapipe हैंड डिटेक्शन सेटअप
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# कमांड भेजने के बीच cooldown समय (सेकंड में)
command_cooldown = 1
last_command_time = 0

# फिंगर उठने की स्थिति पता करने वाला फंक्शन
def fingers_up(hand_landmarks, img_width, img_height):
    landmarks = hand_landmarks.landmark
    finger_tips_ids = [8, 12, 16, 20]
    fingers = []

    for tip_id in finger_tips_ids:
        tip_y = landmarks[tip_id].y * img_height
        pip_y = landmarks[tip_id - 2].y * img_height
        fingers.append(1 if tip_y < pip_y else 0)

    # Thumb के लिए X coordinate से चेकिंग (सिंपल तरीका)
    thumb_tip_x = landmarks[4].x * img_width
    thumb_ip_x = landmarks[3].x * img_width
    fingers.insert(0, 1 if thumb_tip_x < thumb_ip_x else 0)

    return fingers  


def map_gesture_to_command(fingers, index_finger_tip_x, frame_center_x):
    total_fingers = sum(fingers)

    if total_fingers >= 4:
        return "up"    # Jump
    elif total_fingers == 0:
        return "down"  # Slide
    elif total_fingers == 1 and fingers[1] == 1:
        if index_finger_tip_x < frame_center_x:
            return "left"
        else:
            return "right"
    return None

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_height, frame_width, _ = frame.shape
    frame_center_x = frame_width // 2

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    command = None

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            fingers = fingers_up(hand_landmarks, frame_width, frame_height)
            index_finger_tip_x = hand_landmarks.landmark[8].x * frame_width

            command = map_gesture_to_command(fingers, index_finger_tip_x, frame_center_x)

            cv2.putText(frame, f"Fingers: {fingers}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            if command:
                cv2.putText(frame, f"Command: {command}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    cv2.imshow("Hand Gesture Control", frame)


    if command and (time.time() - last_command_time > command_cooldown):
        pyautogui.press(command)
        print(f"Executed command: {command}")
        last_command_time = time.time()


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
