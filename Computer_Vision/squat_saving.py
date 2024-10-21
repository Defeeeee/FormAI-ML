import csv
import math

import cv2
import mediapipe as mp

# Inicializar Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Capturar el video
cap = cv2.VideoCapture(
    '/Users/defeee/Downloads/stock-footage-online-workout-service-professional-trainer-explaining-exercise-virtual-video-tutorial-for (2).mp4')

# Abrir archivo CSV
with open('squat_dataset.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)

    # Escribir encabezado CSV
    writer.writerow(['angulo_rodilla_vertical', 'posicion_relativa_rodilla_dedos',
                     'angulo_torso_pierna', 'etapa', 'error_profundidad', 'error_rodilla', 'error_espalda'])

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Convertir la imagen a RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Procesar la imagen con Mediapipe Pose
        results = pose.process(image_rgb)

        # Dibujar los landmarks en la imagen
        if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Obtener los landmarks relevantes
            cadera = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
            rodilla = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
            tobillo = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
            hombro = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]

            # Calcular el ángulo entre la rodilla y la vertical
            angulo_rodilla_vertical = math.degrees(math.atan2(
                cadera.y - rodilla.y, cadera.x - rodilla.x))

            # Calcular la posición relativa de la rodilla y los dedos del pie
            posicion_relativa_rodilla_dedos = rodilla.x - tobillo.x

            # Calcular el ángulo entre el torso y la pierna
            angulo_torso_pierna = math.degrees(math.atan2(
                hombro.y - cadera.y, hombro.x - cadera.x) - math.atan2(
                tobillo.y - rodilla.y, tobillo.x - rodilla.x))

            # Detectar errores automáticamente (0 para "no", 1 para "si")
            error_profundidad = 1 if angulo_rodilla_vertical < 90 else 0
            error_rodilla = 1 if posicion_relativa_rodilla_dedos > 0 else 0
            # Ajusta el umbral según sea necesario para error_espalda
            error_espalda = 1 if angulo_torso_pierna < 160 else 0

            # Obtener etapa del squat (0: inicio, 1: bajando, 2: abajo)
            try:
                if angulo_rodilla_vertical < 90:
                    etapa = 0
                elif angulo_rodilla_vertical > 150:
                    etapa = 2
                else:
                    etapa = 1
            except:
                etapa = 0

            # Guardar datos solo cada 15 frames
            writer.writerow([angulo_rodilla_vertical, posicion_relativa_rodilla_dedos,
                             angulo_torso_pierna, etapa, error_profundidad, error_rodilla, error_espalda])

        # Mostrar la imagen
        cv2.imshow("Squat Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
