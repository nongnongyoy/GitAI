import dlib         #使用dlib库
import cv2          #导入opencv
detector = dlib.get_frontal_face_detector()    #声明检测器
im_data = cv2.imread('../data/1.jpg')        #读取图像
dets = detector(im_data)                    #检测
for face in dets:
    left = face.left()
    right = face.right()
    top = face.top()
    bottom = face.bottom()
    cv2.rectangle(im_data, (left, top), (right, bottom), (0, 0, 255), thickness=2)
cv2.namedWindow("result", cv2.WINDOW_NORMAL)
cv2.imshow("result", im_data)                                                       #结果展示
cv2.waitKey(0)