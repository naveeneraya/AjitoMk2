import PyOpenGL_Window
import PyGLPointCloud
import cv2
import glm
import PyOpenGL_PhotonSword
import My_MP_Holistic

cols = 1280
rows = 720
glWindow = PyOpenGL_Window.PyOpenGL_Window(cols, rows, PyOpenGL_Window.GL_WINDOW_MODE.CV_COORD_SYSTEM)
cv_to_gl = PyGLPointCloud.Quad_WithLayoutQualifiers(cols, rows)
cv_to_gl.disableDepth = True
pose = My_MP_Holistic.My_MP_Holistic(cols, rows)
#pose = My_MP_Pose_Manager.My_Mp_Pose_Manager(cols, rows)
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, cols)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, rows)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

saber = PyOpenGL_PhotonSword.PyOpenGL_PhotonSword(cols, rows)

while True:
    glWindow.preprocess()

    success, frame = cap.read()
    pose.process(frame, True)

    saberLength = saber.getLength()

    saberEndPt = saber.getEndPt()
    dist = glm.distance(pose.left_index_finger_mcp, pose.holistic_left_wrist_OpenCV)

    cv2.line(frame, wristPt, saberEndPt, (255,255,0), 20)
    cv2.line(frame, wristPt, saberEndPt, (255, 255, 255), 10)


    cv2.imshow("frame", frame)
    cv2.waitKey(1)

    projection = PyOpenGL_Window.PROJECTION_MODE.ORTHO
    projectionMat, viewMat = glWindow.getProjectionMat_and_ViewMat(projection)

    cv_to_gl.draw(frame, frame, viewMat, projectionMat)
    rot = glm.rotate(glm.mat4(1), glm.radians(90), glm.vec3(0,1,0))
    if dist > 0.001:
        diameter = dist
        saber.process(projectionMat, viewMat, wristPt[0], wristPt[1], 0, rot, diameter )
        saber.draw()

    glWindow.postprocess()
