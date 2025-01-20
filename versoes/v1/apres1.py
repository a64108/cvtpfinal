import cv2
import mediapipe as mp
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
import glfw
import pywavefront

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Load the .obj file using PyWavefront
obj_mesh = pywavefront.Wavefront('TPFINAL/objetos/rock.obj', create_materials=True, collect_faces=True)

# State variables
left_hand_state = None
right_hand_state = None
left_object_created = False
right_object_created = False
left_object_position = None
right_object_position = None
rotation_angle = 0

# Function to draw the mesh
def draw_mesh(position=None, scale=0.8, rotation=0):
    """Draw the 3D object from the .obj file."""
    glPushMatrix()
    if position:
        glTranslatef(position[0], position[1], -2)  # Translate the object
    glScalef(scale, scale, scale)  # Scale the object
    glRotatef(rotation, 0, 1, 0)  # Rotate the object
    glColor4f(0.65, 0.33, 0.18, 1.0)  # Set object color to brown (opaque)
    for name, mesh in obj_mesh.meshes.items():
        glBegin(GL_TRIANGLES)
        for face in mesh.faces:
            for vertex_index in face:
                vertex = obj_mesh.vertices[vertex_index]
                glVertex3f(*vertex)
        glEnd()
    glPopMatrix()

# Function to detect hand state
def detect_hand_state(landmarks):
    """Detect the state of the hand (open or closed)."""
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP]
    wrist = landmarks[mp_hands.HandLandmark.WRIST]

    open_threshold = 0.4
    close_threshold = 0.25

    thumb_dist = np.linalg.norm(np.array([thumb_tip.x, thumb_tip.y]) - np.array([wrist.x, wrist.y]))
    index_dist = np.linalg.norm(np.array([index_tip.x, index_tip.y]) - np.array([wrist.x, wrist.y]))
    pinky_dist = np.linalg.norm(np.array([pinky_tip.x, pinky_tip.y]) - np.array([wrist.x, wrist.y]))

    if thumb_dist > open_threshold and index_dist > open_threshold and pinky_dist > open_threshold:
        return "Open"
    elif thumb_dist < close_threshold and index_dist < close_threshold and pinky_dist < close_threshold:
        return "Closed"
    else:
        return None

# Function to get object position based on landmarks
def get_object_position(landmarks):
    try:
        wrist = landmarks[mp_hands.HandLandmark.WRIST]
        return (wrist.x * 2 - 1, -(wrist.y * 2 - 1))
    except (AttributeError, IndexError):
        return None

# Function to render the OpenGL scene to a texture
def render_to_texture():
    global rotation_angle
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()

    if left_object_created and left_object_position is not None:
        draw_mesh(left_object_position, rotation=rotation_angle)

    if right_object_created and right_object_position is not None:
        draw_mesh(right_object_position, rotation=rotation_angle)

    rotation_angle += 1  # Slowly spin the object
    glFlush()

    width, height = 640, 480
    glPixelStorei(GL_PACK_ALIGNMENT, 1)
    data = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
    image = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)
    image = cv2.flip(image, 0)  # Flip vertically for OpenCV
    return image

# OpenCV video capture
cap = cv2.VideoCapture(0)

# Initialize GLFW
if not glfw.init():
    raise Exception("GLFW cannot be initialized!")

width, height = 640, 480
window = glfw.create_window(width, height, "OpenGL + MediaPipe", None, None)

if not window:
    glfw.terminate()
    raise Exception("GLFW window cannot be created!")

glfw.make_context_current(window)

# OpenGL perspective setup
glMatrixMode(GL_PROJECTION)
glLoadIdentity()
gluPerspective(45, width / height, 0.1, 50.0)
glMatrixMode(GL_MODELVIEW)
glEnable(GL_DEPTH_TEST)

# Main loop
while not glfw.window_should_close(window) and cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    left_hand_detected = False
    right_hand_detected = False

    if result.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = hand_landmarks.landmark
            hand_state = detect_hand_state(landmarks)
            hand_label = handedness.classification[0].label  # "Left" or "Right"

            if hand_label == "Left":
                left_hand_detected = True
                if left_hand_state == "Open" and hand_state == "Closed":
                    left_object_created = False
                    left_object_position = None
                    print("Left object thrown!")

                if hand_state == "Closed":
                    left_hand_state = "Closed"
                elif hand_state == "Open":
                    left_object_created = True
                    left_hand_state = "Open"
                    left_object_position = get_object_position(landmarks)

            elif hand_label == "Right":
                right_hand_detected = True
                if right_hand_state == "Open" and hand_state == "Closed":
                    right_object_created = False
                    right_object_position = None
                    print("Right object thrown!")

                if hand_state == "Closed":
                    right_hand_state = "Closed"
                elif hand_state == "Open":
                    right_object_created = True
                    right_hand_state = "Open"
                    right_object_position = get_object_position(landmarks)

    if not left_hand_detected:
        left_object_created = False
        left_object_position = None

    if not right_hand_detected:
        right_object_created = False
        right_object_position = None

    gl_image = render_to_texture()
    blended_frame = cv2.addWeighted(frame, 0.7, gl_image, 0.3, 0)

    if left_hand_detected and left_hand_state:
        cv2.putText(blended_frame, f"Left Hand: {left_hand_state}", (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    if right_hand_detected and right_hand_state:
        cv2.putText(blended_frame, f"Right Hand: {right_hand_state}", (w - 250, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('MediaPipe + OpenGL', blended_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
hands.close()
cv2.destroyAllWindows()
glfw.terminate()
