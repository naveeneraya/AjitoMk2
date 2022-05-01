
import cv2
import mediapipe as mp
import glm

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic


# For webcam input:
cap = cv2.VideoCapture(0)
cols = 1280
rows = 720
cap.set(cv2.CAP_PROP_FRAME_WIDTH, cols)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, rows)
drawPose = False

count = 0

showFaceLandmarks = True

with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as holistic:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = holistic.process(image)

    # Draw landmark annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image,
        results.face_landmarks,
        mp_holistic.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles
        .get_default_face_mesh_contours_style())
    if drawPose:
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles
            .get_default_pose_landmarks_style())
    mp_drawing.draw_landmarks(
        image,
        results.right_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style())

    mp_drawing.draw_landmarks(
        image,
        results.left_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style())

    #if showFaceLandmarks:
        #image = cv2.resize(image, (1920, 1080))

    eyeA0 = (0,0)
    eyeA1 = (0,0)
    eyeB0 = (0,0)
    eyeB1 = (0,0)

    upperLip = (0,0)
    bottomLip = (0,0)

    if results.face_landmarks:
        for id, lm in enumerate(results.face_landmarks.landmark):
            h, w, c = image.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            pt = (cx, cy)
            #cv2.circle(image, (cx,cy), 3, (0,255,0), 1)
            if showFaceLandmarks:
                w = 1920
                h = 1080
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.putText(image, str(id), pt, cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255,0,0))
            count = count + 1


            if id == 130:
                eyeA0 = pt
            if id == 243:
                eyeA1 = pt
            if id == 463:
                eyeB0 = pt
            if id == 359:
                eyeB1 = pt
            if id == 13:
                upperLip = pt
            if id == 14:
                bottomLip = pt

            if id == 130 or id == 243 or id == 463 or id == 359:
                cv2.circle(image, pt, 5, (0, 255, 0), 2)


    #eyeAlength = glm.distance(eyeA0, eyeA1)
    #eyeBlength = glm.distance(eyeB0, eyeB1)
    #eyeARadius = int(eyeAlength/2.0)
    #eyeBradius = int(eyeBlength/2.0)

    if glm.distance(upperLip, bottomLip) > 50:
        showFaceLandmarks = False
    if glm.distance(upperLip, bottomLip) < 5:
        showFaceLandmarks = True

    eyeColor = (255,255,150)
    radius = 20

    # Flip the image horizontally for a selfie-view display.
    #cv2.imshow('MediaPipe Holistic', cv2.flip(image, 1))

    cv2.imshow('image', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
