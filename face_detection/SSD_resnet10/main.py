import cv2          #导入opencv
modelPath = "./model/deploy.prototxt.txt"
weightPath = "./model/res10_300x300_ssd_iter_140000.caffemodel"
confidence = 0.5
net = cv2.dnn.readNetFromCaffe(modelPath, weightPath)

im_data = cv2.imread('../data/1.jpg')        #读取图像
img = im_data
blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
net.setInput(blob)
detections = net.forward()  # 前向传播预测结果
for i in range(0, detections.shape[2]):
    # 获得置信度
    res_confidence = detections[0, 0, i, 2]
    # 过滤掉低置信度的像素
    if res_confidence > confidence:
    # 获得框的位置
        (x1, y1, x2, y2) = detections[0, 0, i, 3:7]
        x1 = int(x1*im_data.shape[1])
        y1 = int(y1*im_data.shape[0])
        x2 = int(x2*im_data.shape[1])
        y2 = int(y2*im_data.shape[0])
        cv2.rectangle(im_data, (x1, y1), (x2, y2), (0, 0, 255), thickness=2)
cv2.namedWindow("result", cv2.WINDOW_NORMAL)
cv2.imshow("result", im_data)                                                       #结果展示
cv2.waitKey(0)