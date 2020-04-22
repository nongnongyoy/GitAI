import dlib
import cv2
im_data = cv2.imread('../data/1.jpg')        #读取图像
detector = dlib.get_frontal_face_detector()           #使用dlib库
predictor = dlib.shape_predictor('./face_landmark_dlib_model/shape_predictor_68_face_landmarks.dat')
rects = detector(im_data, 0)
res = []  # 列表用来保存人脸关键点的坐标
for face in rects:
    mark = predictor(im_data, face)
    for pt in mark.parts():
        pt_ops = (pt.x, pt.y)  # 返回的坐标是相对整张图像的坐标
        cv2.circle(im_data, (pt.x, pt.y), 1, (0, 0, 255),
                   thickness=2)
cv2.namedWindow("result", cv2.WINDOW_NORMAL)
cv2.imshow("result", im_data)                                                       #结果展示
cv2.waitKey(0)
