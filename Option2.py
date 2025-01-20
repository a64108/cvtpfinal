import cv2
from GestosJogo import GestosJogo  # Import GestosJogo for gesture detection

def run(cap):
    gestos_jogo = GestosJogo()  # Create an instance of GestosJogo
    gamemode = 2  # Set gamemode to 1 when in Option 1
    menu = 1  # Set menu to 1 (indicating Menu 1)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        # Detect hands and gestures
        results = gestos_jogo.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        hands_landmarks = results.multi_hand_landmarks if results.multi_hand_landmarks else None

        left_ok = False
        right_ok = False

        if hands_landmarks:
            for hand_landmarks, hand_handedness in zip(hands_landmarks, results.multi_handedness):
                hand_type = hand_handedness.classification[0].label
                is_ok = gestos_jogo.gesture_library.detect_ok_gesture(hand_landmarks)

                if hand_type == "Left":
                    left_ok = is_ok
                else:
                    right_ok = is_ok

                gestos_jogo.mp_draw.draw_landmarks(frame, hand_landmarks, gestos_jogo.mp_hands.HAND_CONNECTIONS)

        # Display "Modo Objetos" at the top middle of the frame
        height, width, _ = frame.shape
        text = "Modo Objetos"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_x = (width - text_size[0]) // 2
        text_y = 50  # Position the text near the top middle

        cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # If both hands make the OK gesture, return to Menu 1 and set gamemode to 0
        if left_ok and right_ok:
            print("Both hands detected OK gesture, returning to Menu 1")
            gamemode = 0  # Set gamemode back to 0 (Menu)
            menu = 1  # Ensure we are in Menu 1 state
            return gamemode, menu  # Return gamemode and menu state to handle transitions


        # Update the existing window in Start.py
        cv2.imshow('Gesture Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # After exiting the loop, you should have gamemode = 0, indicating return to Menu 1
    return gamemode, menu  # Return gamemode and menu state to handle transitions

