import cv2
import mediapipe as mp

class GestosMenu:
    class GestureLibrary:
        def identify_fingers_raised(self, hand_landmarks, hand_type):
            finger_tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky TIP landmarks
            finger_pips = [2, 6, 10, 14, 18]  # Thumb IP, Index PIP, Middle PIP, Ring PIP, Pinky PIP landmarks

            fingers_raised = 0

            for tip, pip in zip(finger_tips[1:], finger_pips[1:]):
                if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y:
                    fingers_raised += 1

            if hand_type == "Right":
                if hand_landmarks.landmark[finger_tips[0]].x < hand_landmarks.landmark[finger_pips[0]].x:
                    fingers_raised += 1
            else:  # Left hand
                if hand_landmarks.landmark[finger_tips[0]].x > hand_landmarks.landmark[finger_pips[0]].x:
                    fingers_raised += 1

            return fingers_raised

        def detect_ok_gesture(self, hand_landmarks):
            thumb_tip = 4
            index_tip = 8
            middle_pip = 6
            ring_pip = 10
            pinky_pip = 14

            thumb_index_distance = self.calculate_distance(
                hand_landmarks.landmark[thumb_tip],
                hand_landmarks.landmark[index_tip]
            )
            touching = thumb_index_distance < 0.05

            middle_up = hand_landmarks.landmark[12].y < hand_landmarks.landmark[middle_pip].y
            ring_up = hand_landmarks.landmark[16].y < hand_landmarks.landmark[ring_pip].y
            pinky_up = hand_landmarks.landmark[20].y < hand_landmarks.landmark[pinky_pip].y

            return touching and middle_up and ring_up and pinky_up

        @staticmethod
        def calculate_distance(point1, point2):
            return ((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2) ** 0.5

    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_draw = mp.solutions.drawing_utils
        self.gesture_library = self.GestureLibrary()

    def start(self):
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = self.hands.process(rgb_frame)

            left_fingers = 0
            right_fingers = 0
            left_ok = False
            right_ok = False

            if results.multi_hand_landmarks:
                for hand_landmarks, hand_handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    hand_type = hand_handedness.classification[0].label

                    wrist = hand_landmarks.landmark[0]
                    pinky_mcp = hand_landmarks.landmark[17]
                    distance = self.gesture_library.calculate_distance(wrist, pinky_mcp)

                    if distance < 0.1:
                        continue

                    is_ok = self.gesture_library.detect_ok_gesture(hand_landmarks)

                    if is_ok:
                        fingers_raised = 0
                    else:
                        fingers_raised = self.gesture_library.identify_fingers_raised(hand_landmarks, hand_type)

                    if hand_type == "Left":
                        left_fingers = fingers_raised
                        left_ok = is_ok
                    else:
                        right_fingers = fingers_raised
                        right_ok = is_ok

                    self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

            if left_fingers:
                cv2.putText(frame, f"Left Hand: {left_fingers}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if right_fingers:
                cv2.putText(frame, f"Right Hand: {right_fingers}", (w - 300, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if left_ok:
                cv2.putText(frame, "Left OK", (10, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            if right_ok:
                cv2.putText(frame, "Right OK", (w - 300, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            cv2.imshow('Gesture Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
