import numpy as np
import cv2
confThreshold = 0.5  #Confidence threshold
nmsThreshold = 0.5   #Non-maximum suppression threshold置信度阈值
inpWidth = 320       #Width of network's input image，改为320*320更快
inpHeight = 320      #Height of network's input image，改为608*608更准
classesFile = "./data/coco.names";
classes1 = None
with open(classesFile, 'rt') as f:
    classes1 = f.read().rstrip('\n').split('\n')
# Give the configuration and weight files for the model and load the network using them.
modelConfiguration = "./data/yolov3.cfg";
modelWeights = "./yolov3_model/yolov3.weights";
yolonet = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
yolonet.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
yolonet.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU) #可切换
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

cap = cv2.VideoCapture(0)
while (cap.isOpened()):
    ret, im_data = cap.read()
    if(ret):
        blob = cv2.dnn.blobFromImage(im_data, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)
        # Sets the input to the network
        yolonet.setInput(blob)
        # Runs the forward pass to get output of the output layers
        outs = yolonet.forward(getOutputsNames(yolonet))
        # Remove the bounding boxes with low confidence
        frameHeight = im_data.shape[0]
        frameWidth = im_data.shape[1]
        classIds = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > confThreshold:
                    center_x = int(detection[0] * frameWidth)
                    center_y = int(detection[1] * frameHeight)
                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])
        # Perform non maximum suppression to eliminate redundant overlapping boxes with
        # lower confidences.
        indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
        for i in indices:
            i = i[0]
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            cv2.rectangle(im_data, (left, top), (left + width, top + height), (255, 178, 50), 3)
            label = '%.2f' % confidences[i]
            # Get the label for the class name and its confidence
            if classes1:
                label = '%s:%s' % (classes1[classId], label)
            # Display the label at the top of the bounding box
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            top = max(top, labelSize[1])
            cv2.rectangle(im_data, (left, top - round(1.5 * labelSize[1])),
                          (left + round(1.5 * labelSize[0]), top + baseLine),
                          (255, 255, 255), cv2.FILLED)
            cv2.putText(im_data, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)
        # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
        t, _ = yolonet.getPerfProfile()
        label = 'yolov3 Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
        cv2.putText(im_data, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
        cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
        cv2.imshow("frame", im_data)
        cv2.waitKey(10)