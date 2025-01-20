import cv2
import mediapipe as mp
from gesture_library import GestureLibrary

def main():
    """
    Função principal que inicializa o reconhecimento de gestos e captura de vídeo.
    Entrada:
        Nenhuma
    Saída:
        Nenhuma
    """
    # Inicializa MediaPipe Hands e a biblioteca de gestos
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_draw = mp.solutions.drawing_utils

    gesture_library = GestureLibrary()

    # Inicia a captura de vídeo
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Inverte o quadro horizontalmente para efeito espelho
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Converte o quadro para RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Processa o quadro com MediaPipe Hands
        results = hands.process(rgb_frame)

        # Variáveis para rastrear os estados das mãos
        left_fingers = 0
        right_fingers = 0
        left_ok = False
        right_ok = False

        # Verifica se existem marcos de mãos
        if results.multi_hand_landmarks:
            for hand_landmarks, hand_handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                hand_type = hand_handedness.classification[0].label  # 'Esquerda' ou 'Direita'

                # Calcula a distância entre o pulso e o MCP do dedo mínimo
                wrist = hand_landmarks.landmark[0]  # Pulso
                pinky_mcp = hand_landmarks.landmark[17]  # MCP do dedo mínimo

                distance = gesture_library.calculate_distance(wrist, pinky_mcp)

                # Ignora a mão se a distância for muito pequena
                if distance < 0.1:  # Ajuste o limite conforme necessário
                    continue

                # Verifica se o gesto "OK" está sendo realizado
                is_ok = gesture_library.detect_ok_gesture(hand_landmarks)

                # Se for o gesto "OK", define o número de dedos levantados como 0
                if is_ok:
                    fingers_raised = 0
                else:
                    # Detecta os dedos levantados
                    fingers_raised = gesture_library.identify_fingers_raised(hand_landmarks, hand_type)

                if hand_type == "Esquerda":
                    left_fingers = fingers_raised
                    left_ok = is_ok
                else:
                    right_fingers = fingers_raised
                    right_ok = is_ok

                # Desenha os marcos na imagem
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Exibe os gestos e trata interações
        if left_fingers:
            cv2.putText(frame, f"Mão Esquerda: {left_fingers}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if right_fingers:
            cv2.putText(frame, f"Mão Direita: {right_fingers}", (w - 300, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if left_ok:
            cv2.putText(frame, "OK Esquerda", (10, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        if right_ok:
            cv2.putText(frame, "OK Direita", (w - 300, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Exibe o quadro
        cv2.imshow('Detecção de Gestos', frame)

        # Encerra o loop ao pressionar 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Libera os recursos
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
