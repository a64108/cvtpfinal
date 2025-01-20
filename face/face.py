import cv2
import mediapipe as mp
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
import glfw
import pywavefront

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

obj_mesh = pywavefront.Wavefront('TPFINAL/objetos/hat.obj', create_materials=True, collect_faces=True)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

face_position = None
rotation_angle = 0

def draw_mesh(position=None, scale=0.8, rotation=0):
    glPushMatrix()
    if position:
        glTranslatef(position[0], position[1] + 0.3, -2)
    glScalef(scale * 1.25, scale * 1.25, scale * 1.25)
    glRotatef(rotation, 0, 0, 0)
    glColor4f(0.36, 0.25, 0.20, 1.0)
    for name, mesh in obj_mesh.meshes.items():
        glBegin(GL_TRIANGLES)
        for face in mesh.faces:
            for vertex_index in face:
                vertex = obj_mesh.vertices[vertex_index]
                glVertex3f(*vertex)
        glEnd()
    glPopMatrix()

def render_to_texture():
    global rotation_angle
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()

    if face_position:
        draw_mesh(face_position, scale=0.5, rotation=rotation_angle)

    rotation_angle += 1
    glFlush()

    width, height = 640, 480
    glPixelStorei(GL_PACK_ALIGNMENT, 1)
    data = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
    image = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)
    image = cv2.flip(image, 0)
    return image

cap = cv2.VideoCapture(0)

if not glfw.init():
    raise Exception("GLFW não pode ser inicializado!")

width, height = 640, 480
window = glfw.create_window(width, height, "OpenGL + MediaPipe", None, None)

if not window:
    glfw.terminate()
    raise Exception("Janela GLFW não pode ser criada!")

glfw.make_context_current(window)

glMatrixMode(GL_PROJECTION)
glLoadIdentity()
gluPerspective(45, width / height, 0.1, 50.0)
glMatrixMode(GL_MODELVIEW)
glEnable(GL_DEPTH_TEST)

while not glfw.window_should_close(window) and cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    face_position = None
    if len(faces) > 0:
        x, y, w, h = faces[0]
        face_position = ((x + w // 2) / 640 * 2 - 1, -(y + h // 2) / 480 * 2 + 1)

    gl_image = render_to_texture()
    blended_frame = cv2.addWeighted(frame, 0.7, gl_image, 0.3, 0)

    if face_position:
        cv2.putText(blended_frame, "Chapeu!!!", (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('MediaPipe + OpenGL', blended_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
hands.close()
cv2.destroyAllWindows()
glfw.terminate()
