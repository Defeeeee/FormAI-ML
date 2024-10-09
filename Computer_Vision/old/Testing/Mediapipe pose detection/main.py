import cv2
import mediapipe as mp

# MediaPipe setup
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Local video path
video_path = ""

# Video capture from file
cap = cv2.VideoCapture('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR6d6nMYnFENB7YT8fsG9tU4Gjm0fVv5B7qrg&s')

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            break

        # Convert to RGB and process
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        # Draw landmarks
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Display the processed frame
        cv2.imshow('MediaPipe Pose', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        # print the landmarks
        if results.pose_landmarks:
            print(results.pose_landmarks)



        # wait for key press to continue
        while True:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
            break


cap.release()
cv2.destroyAllWindows()

