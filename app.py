import cv2
import mediapipe as mp
import numpy as np
import random
import colorsys

# Initialize video capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Camera not accessible")
    exit()

# Initialize Mediapipe Hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)

# Particle class
class Particle:
    def __init__(self, position):
        self.position = np.array(position, dtype=np.float32)
        self.velocity = np.random.uniform(-1, 1, size=2) * 2
        self.lifetime = random.randint(30, 60)
        self.max_lifetime = self.lifetime
        self.hue = random.uniform(0, 1)

    def update(self):
        self.position += self.velocity
        self.lifetime -= 1
        self.hue += 0.01
        if self.hue > 1:
            self.hue -= 1

    def get_color(self):
        rgb = colorsys.hsv_to_rgb(self.hue, 1.0, 1.0)
        alpha = self.lifetime / self.max_lifetime
        return (np.array(rgb) * 255 * alpha).astype(np.uint8)

# Particle list
particles = []

# Previous hand center
prev_center = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame
    results = hands.process(rgb_frame)

    # Create overlay for particles
    overlay = frame.copy()
    overlay = cv2.addWeighted(overlay, 0.5, overlay, 0, 0)

    # Check for hands
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get center of palm (landmark 9)
            lm = hand_landmarks.landmark[9]
            cx, cy = int(lm.x * w), int(lm.y * h)

            # Calculate hand speed
            if prev_center is not None:
                speed = np.linalg.norm(np.array([cx, cy]) - prev_center)
            else:
                speed = 0

            prev_center = np.array([cx, cy])

            # Generate particles based on speed
            num_new_particles = int(min(10, speed * 5)) + 3
            for _ in range(num_new_particles):
                particles.append(Particle((cx, cy)))

    # Update and draw particles
    for particle in particles[:]:
        particle.update()
        if particle.lifetime <= 0:
            particles.remove(particle)
            continue

        color = particle.get_color().tolist()
        cv2.circle(overlay,
                   (int(particle.position[0]), int(particle.position[1])),
                   10,
                   color,
                   -1)

    # Blend overlay with original frame
    frame = cv2.addWeighted(frame, 0.8, overlay, 0.2, 0)

    # Show frame
    cv2.imshow('Hand AR Effect', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
