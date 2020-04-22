import cv2          #导入opencv
import tensorflow as tf
import numpy as np
#人脸检测声明
modelPath = "./face_landmark_SENet_model/deploy.prototxt.txt"
weightPath = "./face_landmark_SENet_model/res10_300x300_ssd_iter_140000.caffemodel"
confidence = 0.5
net = cv2.dnn.readNetFromCaffe(modelPath, weightPath)
#SENet
pb_path = "./face_landmark_SENet_model/landmark.pb"
face_landmark_sess = tf.Session()
with face_landmark_sess.as_default():
    with tf.gfile.FastGFile(pb_path, "rb") as f:
        graph_def = face_landmark_sess.graph_def
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name="")
landmark = face_landmark_sess.graph.get_tensor_by_name("fully_connected_9/Relu:0")

im_data = cv2.imread('../data/1.jpg')        #读取图像
sp = im_data.shape
blob = cv2.dnn.blobFromImage(cv2.resize(im_data, (300, 300)), 1.0,
                                         (300, 300), (104.0, 177.0, 123.0))
net.setInput(blob)
detections = net.forward()  # 预测结果
res = []
for i in range(0, detections.shape[2]):
    # 获得置信度
    res_confidence = detections[0, 0, i, 2]
    # 过滤掉低置信度的像素
    if res_confidence > confidence:
    # 获得框的位置
        (x1, y1, x2, y2) = detections[0, 0, i, 3:7]
        y1 = int(y1 * sp[0])
        x1 = int(x1 * sp[1])
        y2 = int(y2 * sp[0])
        x2 = int(x2 * sp[1])
        # 获取人脸框
        face_data = im_data[y1:y2, x1:x2]
        img_data = cv2.resize(face_data, (128, 128))
        pred = face_landmark_sess.run(landmark, {"Placeholder:0":
                np.expand_dims(img_data, 0)})
        pred = pred[0]
        for i in range(0, 136, 2):
            cv2.circle(im_data, (int(pred[i] * (x2 - x1) + x1), int(pred[i + 1] * (y2 - y1) + y1)), 1, (0, 0, 255), thickness=2)
cv2.namedWindow("result", cv2.WINDOW_NORMAL)
cv2.imshow("result", im_data)                                                       #结果展示
cv2.waitKey(0)
