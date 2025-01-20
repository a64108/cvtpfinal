import cv2
import mediapipe as mp

# Configuração do MediaPipe para Objectron
mp_objectron = mp.solutions.objectron
mp_drawing = mp.solutions.drawing_utils

# Iniciar a detecção de objetos 3D
objectron = mp_objectron.Objectron(static_image_mode=False,
                                   max_num_objects=5,
                                   min_detection_confidence=0.5,
                                   min_tracking_confidence=0.99,
                                   model_name='Shoe')  # Você pode escolher entre 'Shoe', 'Chair', 'Cup', etc.

# Iniciar captura de vídeo
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convertendo a imagem de BGR para RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Processar a imagem para detectar objetos
    results = objectron.process(rgb_frame)

    # Desenhar os resultados
    if results.detected_objects:
        for detected_object in results.detected_objects:
            # Desenhar a caixa delimitadora 3D
            mp_drawing.draw_landmarks(frame, detected_object.landmarks_2d, mp_objectron.BOX_CONNECTIONS)

            # Obter os pontos dos marcos 2D
            landmarks = detected_object.landmarks_2d

            # Calcular as coordenadas da caixa delimitadora (bounding box)
            x_min = min([lm.x for lm in landmarks]) * frame.shape[1]
            y_min = min([lm.y for lm in landmarks]) * frame.shape[0]
            x_max = max([lm.x for lm in landmarks]) * frame.shape[1]
            y_max = max([lm.y for lm in landmarks]) * frame.shape[0]

            # Desenhar a caixa delimitadora (bounding box) no frame
            cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)

            # Adicionar o label sobre o objeto detectado
            label = 'Objeto Detectado'  # Você pode personalizar isso com base em outras informações
            cv2.putText(frame, label, (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Exibir a imagem com as anotações
    cv2.imshow('Detecção de Objetos 3D', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
