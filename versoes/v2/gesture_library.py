class GestureLibrary:
    def identify_fingers_raised(self, hand_landmarks, hand_type):
        """
        Identifica quantos dedos estão levantados com base nos marcos das mãos.
        Entrada:
            hand_landmarks: Dados dos marcos da mão (objeto de resultados de MediaPipe)
            hand_type: Tipo da mão ('Esquerda' ou 'Direita')
        Saída:
            fingers_raised: Número de dedos levantados (0 a 5)
        """
        finger_tips = [4, 8, 12, 16, 20]  # Dedo indicador, médio, anelar, mínimo
        finger_pips = [2, 6, 10, 14, 18]  # Articulações dos dedos

        fingers_raised = 0

        # Verifica cada dedo
        for tip, pip in zip(finger_tips[1:], finger_pips[1:]):  # Ignora o polegar por enquanto
            if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y:
                fingers_raised += 1

        # Verifica o polegar separadamente (com base nas coordenadas x)
        if hand_type == "Direita":
            if hand_landmarks.landmark[finger_tips[0]].x < hand_landmarks.landmark[finger_pips[0]].x:
                fingers_raised += 1
        else:  # Mão esquerda
            if hand_landmarks.landmark[finger_tips[0]].x > hand_landmarks.landmark[finger_pips[0]].x:
                fingers_raised += 1

        return fingers_raised

    def detect_ok_gesture(self, hand_landmarks):
        """
        Detecta se a mão está fazendo o gesto de 'OK'.
        Entrada:
            hand_landmarks: Dados dos marcos da mão (objeto de resultados de MediaPipe)
        Saída:
            True: Se o gesto 'OK' for detectado
            False: Caso contrário
        """
        thumb_tip = 4
        index_tip = 8
        middle_pip = 6
        ring_pip = 10
        pinky_pip = 14

        # Verifica se o polegar e o indicador estão tocando
        thumb_index_distance = self.calculate_distance(
            hand_landmarks.landmark[thumb_tip],
            hand_landmarks.landmark[index_tip]
        )
        touching = thumb_index_distance < 0.05

        # Verifica se outros dedos estão levantados
        middle_up = hand_landmarks.landmark[12].y < hand_landmarks.landmark[middle_pip].y
        ring_up = hand_landmarks.landmark[16].y < hand_landmarks.landmark[ring_pip].y
        pinky_up = hand_landmarks.landmark[20].y < hand_landmarks.landmark[pinky_pip].y

        return touching and middle_up and ring_up and pinky_up

    @staticmethod
    def calculate_distance(point1, point2):
        """
        Calcula a distância Euclidiana entre dois marcos.
        Entrada:
            point1: Primeiro ponto (objeto de marco)
            point2: Segundo ponto (objeto de marco)
        Saída:
            Distância Euclidiana entre os pontos
        """
        return ((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2) ** 0.5
