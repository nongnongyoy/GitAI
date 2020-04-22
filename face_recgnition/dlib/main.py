import cv2
import dlib
import numpy as np
detector = dlib.get_frontal_face_detector()           #使用dlib库
predictor = dlib.shape_predictor('model/shape_predictor_68_face_landmarks.dat')
face_rec = dlib.face_recognition_model_v1("model/dlib_face_recognition_resnet_model_v1.dat")
def face_feature_dlib(path):
    im_data = cv2.imread(path)
    features_cap_arr = []  # 人脸特征数组
    # 人脸检测
    sp = im_data.shape  ###SSD resnet10进行人脸检测
    img_gray = cv2.cvtColor(im_data, cv2.COLOR_RGB2GRAY)
    faces = detector(img_gray, 0)
    if (len(faces) == 1):
        # 计算人脸特征
        shape = predictor(im_data, faces[0])
        features_cap_arr.append(face_rec.compute_face_descriptor(im_data, shape))
        emb = np.array(features_cap_arr)
        print(emb)

if __name__ == '__main__':
    path = 'data/1.jpg'
    face_feature_dlib(path)         #输出人脸特征