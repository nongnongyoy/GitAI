import cv2          #导入opencv
import tensorflow as tf #导入tensorflow
import numpy as np
from object_detection.utils import ops as utils_ops
im_data = cv2.imread('../data/1.jpg')        #读取图像
sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = True
PATH_TO_FROZEN_GRAPH = "./model/frozen_inference_graph.pb"
PATH_TO_LABELS = "./model/face_label_map.pbtxt"
IMAGE_SIZE = (256, 256)
detection_sess = tf.Session()
with detection_sess.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
        ops = tf.get_default_graph().get_operations()
        all_tensor_names = {output.name for op in ops for output in op.outputs}
        tensor_dict = {}
        for key in [
            'num_detections', 'detection_boxes', 'detection_scores',
            'detection_classes', 'detection_masks'
        ]:
            tensor_name = key + ':0'
            if tensor_name in all_tensor_names:
                tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                    tensor_name)
        if 'detection_masks' in tensor_dict:
            # The following processing is only for single image
            detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
            detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
            # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
            real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
            detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
            detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
            detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                detection_masks, detection_boxes, IMAGE_SIZE[0], IMAGE_SIZE[1])
            detection_masks_reframed = tf.cast(
                tf.greater(detection_masks_reframed, 0.5), tf.uint8)
            # Follow the convention by adding back the batch dimension
            tensor_dict['detection_masks'] = tf.expand_dims(
                detection_masks_reframed, 0)
        image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

image_np = cv2.resize(im_data, IMAGE_SIZE)
output_dict = detection_sess.run(tensor_dict,
                    feed_dict={image_tensor: np.expand_dims(image_np, 0)})
# all outputs are float32 numpy arrays, so convert types as appropriate
output_dict['num_detections'] = int(output_dict['num_detections'][0])
output_dict['detection_classes'] = output_dict[
    'detection_classes'][0].astype(np.uint8)
output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
output_dict['detection_scores'] = output_dict['detection_scores'][0]
if 'detection_masks' in output_dict:
    output_dict['detection_masks'] = output_dict['detection_masks'][0]
x1 = 0
x2 = 0
y1 = 0
y2 = 0
# print(output_dict['detection_boxes'], output_dict['detection_classes'], output_dict['detection_scores'])
for i in range(len(output_dict['detection_scores'])):
    if output_dict['detection_scores'][i] > 0.1:
        bbox = output_dict['detection_boxes'][i]
        cate = output_dict['detection_classes'][i]
        y1 = bbox[0]
        x1 = bbox[1]
        y2 = bbox[2]
        x2 = bbox[3]
        x1 = int(x1 * im_data.shape[1])
        y1 = int(y1 * im_data.shape[0])
        x2 = int(x2 * im_data.shape[1])
        y2 = int(y2 * im_data.shape[0])
        cv2.rectangle(im_data, (x1, y1), (x2, y2), (0, 0, 255), thickness=2)
cv2.namedWindow("result", cv2.WINDOW_NORMAL)
cv2.imshow("result", im_data)                                                       #结果展示
cv2.waitKey(0)