import cv2
import time
from GestosMenu import GestosMenu  # Importa a classe GestosMenu

class Start:
    def __init__(self):
        self.gestos_menu = GestosMenu()  # Cria uma instância de GestosMenu
        self.last_menu_switch_time = time.time()  # Hora da última mudança de menu
        self.ignore_time = 2  # Ignora a detecção de mãos por 2 segundos após a troca de menus
        self.menu = 1  # Variável que controla qual menu está sendo exibido

    def start(self):
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            # Detecta as mãos e gestos
            results = self.gestos_menu.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            hands_landmarks = results.multi_hand_landmarks if results.multi_hand_landmarks else None

            left_fingers = 0
            right_fingers = 0
            left_ok = False
            right_ok = False

            if hands_landmarks:
                for hand_landmarks, hand_handedness in zip(hands_landmarks, results.multi_handedness):
                    hand_type = hand_handedness.classification[0].label

                    is_ok = self.gestos_menu.gesture_library.detect_ok_gesture(hand_landmarks)
                    fingers_raised = self.gestos_menu.gesture_library.identify_fingers_raised(hand_landmarks, hand_type)

                    if hand_type == "Left":
                        left_fingers = fingers_raised
                        left_ok = is_ok
                    else:
                        right_fingers = fingers_raised
                        right_ok = is_ok

                    self.gestos_menu.mp_draw.draw_landmarks(frame, hand_landmarks, self.gestos_menu.mp_hands.HAND_CONNECTIONS)

            # Exibe o título do Menu 1
            if self.menu == 1:
                cv2.putText(frame, "Jogo VR", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Exibe a contagem de dedos e o texto do gesto OK
            if left_fingers:
                cv2.putText(frame, f"Left Hand: {left_fingers}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if right_fingers:
                cv2.putText(frame, f"Right Hand: {right_fingers}", (w - 300, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if left_ok:
                cv2.putText(frame, "Left OK", (10, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            if right_ok:
                cv2.putText(frame, "Right OK", (w - 300, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Lógica de transição entre Menu 1 e Menu 2 (usando OK com ambas as mãos)
            if self.menu == 1:
                if left_ok and right_ok:  # Se ambas as mãos fizerem OK, transita para Menu 2
                    print("Transitioning to Menu 2")
                    self.menu = 2
                    self.last_menu_switch_time = time.time()

            # Menu 2: Opções de seleção
            if self.menu == 2:
                # Exibe as opções do Menu 2
                cv2.putText(frame, "Escolher Modo de Jogo", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.putText(frame, "1 - Teste", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.putText(frame, "2 - Objetos", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.putText(frame, "3 - Singleplayer", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.putText(frame, "4 - Multiplayer", (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.putText(frame, "5 - Retornar", (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                # Lógica para selecionar opções no Menu 2
                current_time = time.time()
                if current_time - self.last_menu_switch_time > self.ignore_time:
                    if left_fingers == 1 and right_ok:
                        print("Option 1 selected: Teste")
                        self.last_menu_switch_time = current_time
                    elif left_fingers == 2 and right_ok:
                        print("Option 2 selected: Objetos")
                        self.last_menu_switch_time = current_time
                    elif left_fingers == 3 and right_ok:
                        print("Option 3 selected: Singleplayer")
                        self.last_menu_switch_time = current_time
                    elif left_fingers == 4 and right_ok:
                        print("Option 4 selected: Multiplayer")
                        self.last_menu_switch_time = current_time
                    elif left_fingers == 5 and right_ok:
                        print("Option 5 selected: Returning to Menu 1")
                        self.menu = 1  # Retorna para o Menu 1
                        self.last_menu_switch_time = current_time

            # Exibe o frame
            cv2.imshow('Gesture Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

# Executa o programa
if __name__ == "__main__":
    start = Start()
    start.start()
