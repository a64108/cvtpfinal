import cv2
import mediapipe as mp

class GestosJogo:
    class GestureLibrary:
        def detect_ok_gesture(self, hand_landmarks):
            """
            Detect if a hand is making the OK gesture.
            Returns True if the gesture is detected, otherwise False.
            """
            if not hand_landmarks:
                return False

            # Access landmarks correctly
            thumb_tip = hand_landmarks.landmark[4]
            index_tip = hand_landmarks.landmark[8]

            distance = ((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2) ** 0.5

            # Threshold for OK gesture (adjust based on testing)
            return distance < 0.05

        def detect_hand_state(self, hand_landmarks):
            """
            Detect if the hand is open or closed.
            Returns a tuple with hand state and hand side:
            (0 if not detected, 1 if closed, 2 if open) and hand side ('left' or 'right').
            """
            if not hand_landmarks:
                return 0, 'unknown'

            # Thumb tip and other finger tips for comparison
            thumb_tip = hand_landmarks.landmark[4]
            index_tip = hand_landmarks.landmark[8]
            middle_tip = hand_landmarks.landmark[12]
            ring_tip = hand_landmarks.landmark[16]
            pinky_tip = hand_landmarks.landmark[20]

            # Calculate distances between thumb and other fingers to check if the hand is open or closed
            distances = [
                ((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2) ** 0.5,
                ((thumb_tip.x - middle_tip.x) ** 2 + (thumb_tip.y - middle_tip.y) ** 2) ** 0.5,
                ((thumb_tip.x - ring_tip.x) ** 2 + (thumb_tip.y - ring_tip.y) ** 2) ** 0.5,
                ((thumb_tip.x - pinky_tip.x) ** 2 + (thumb_tip.y - pinky_tip.y) ** 2) ** 0.5
            ]

            # Threshold for open/closed hand
            open_threshold = 0.1  # You can adjust this based on your tests
            closed_threshold = 0.03

            if all(distance > open_threshold for distance in distances):
                hand_state = 2  # Hand is open
            elif all(distance < closed_threshold for distance in distances):
                hand_state = 1  # Hand is closed
            else:
                hand_state = 0  # Not clearly open or closed

            # Check which hand it is (left or right)
            hand_side = 'right' if hand_landmarks.landmark[0].x > 0.5 else 'left'

            return hand_state, hand_side

    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False,
                                         max_num_hands=2,
                                         min_detection_confidence=0.5,
                                         min_tracking_confidence=0.5)
        self.mp_draw = mp.solutions.drawing_utils
        self.gesture_library = self.GestureLibrary()  # Initialize GestureLibrary

    def process_frame(self, frame):
        """
        Process a video frame to detect gestures.
        Returns a boolean indicating whether both hands are making the OK gesture.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            ok_gesture_detected = False
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                ok_gesture_detected |= self.gesture_library.detect_ok_gesture(hand_landmarks)

            # Return whether both hands are making the OK gesture
            return ok_gesture_detected

        return False
