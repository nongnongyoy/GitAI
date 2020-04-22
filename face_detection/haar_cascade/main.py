import cv2          #导入opencv
im_data = cv2.imread('../data/1.jpg')       #读取图像
face_xml = cv2.CascadeClassifier('./model/haarcascade_frontalface_default.xml')       #导入人脸检测模型文件
eye_xml = cv2.CascadeClassifier('./model/haarcascade_eye.xml')                        #导入眼睛检测模型文件
gray = cv2.cvtColor(im_data, cv2.COLOR_BGR2GRAY)            #转为灰度图
faces = face_xml.detectMultiScale(gray, 1.3, 5)             #人脸检测
for (x, y, w, h) in faces:
    cv2.rectangle(im_data, (x, y), (x+w, y+h), (0, 0, 255), thickness=2)    #绘制人脸检测框
    roi_face = gray[y:y + h, x:x + w]                                       #获取人脸框用以检测眼睛
    eyes = eye_xml.detectMultiScale(roi_face)                               #眼睛检测
    for (e_x, e_y, e_w, e_h) in eyes:
        print(e_x, e_y, e_w, e_h)
        cv2.rectangle(im_data, (e_x+x, e_y+y), (e_w+e_x+x, e_y+y+e_h), (0, 0, 255), thickness=2)    #绘制眼睛检测框
cv2.namedWindow("result", cv2.WINDOW_NORMAL)
cv2.imshow("result", im_data)                                                       #结果展示
cv2.waitKey(0)