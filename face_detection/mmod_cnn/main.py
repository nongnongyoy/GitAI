import cv2          #导入opencv
import dlib
cnn_face_detector = dlib.cnn_face_detection_model_v1('./model/mmod_human_face_detector.dat')

im_data = cv2.imread('../data/1.jpg')        #读取图像

dets = cnn_face_detector(im_data, 1)
for d in dets:
    face = d.rect
    print("confidence:", d.confidence)
    left = face.left()
    top = face.top()
    right = face.right()
    bottom = face.bottom()
    cv2.rectangle(im_data, (left, top), (right, bottom), (0, 0, 255), thickness=2)
cv2.namedWindow("result", cv2.WINDOW_NORMAL)
cv2.imshow("result", im_data)                                                       #结果展示
cv2.waitKey(0)