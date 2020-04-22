import tensorflow as tf
from object_detection.utils import ops as utils_ops
import numpy as np
import cv2
face_feature_sess = tf.Session()
ff_pb_path = "model/face_recognition_model.pb"      #采用facenet官方模型
with face_feature_sess.as_default():
    ff_od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(ff_pb_path, 'rb') as fid:
        serialized_graph = fid.read()
        ff_od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(ff_od_graph_def, name='')

        ff_images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        ff_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

        ff_embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")


##############face_detection推理定义  SSD_Resnet50
sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = True
PATH_TO_FROZEN_GRAPH = "model/frozen_inference_graph.pb"
PATH_TO_LABELS = "data/face_label_map.pbtxt"
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


#############图像数据标准化
def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y

#############人脸特征提取函数
def face_feature(path):
    IMAGE_SIZE = (256, 256)
    im_data = cv2.imread(path)
    sp = im_data.shape
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
    # print(output_dict['detection_boxes'], output_dict['detection_classes'], output_dict['detection_scores'])
    for i in range(len(output_dict['detection_scores'])):
        if output_dict['detection_scores'][i] > 0.1:
            bbox = output_dict['detection_boxes'][i]
            cate = output_dict['detection_classes'][i]
            y1 = bbox[0]
            x1 = bbox[1]
            y2 = bbox[2]
            x2 = bbox[3]
            # print(output_dict['detection_scores'][i], x1, y1, x2, y2)
            y1 = int(y1 * sp[0])
            x1 = int(x1 * sp[1])
            y2 = int(y2 * sp[0])
            x2 = int(x2 * sp[1])
            # 获取人脸框
            face_data = im_data[y1:y2, x1:x2]
            # 数据标准化
            im_data = prewhiten(face_data)
            im_data = cv2.resize(im_data, (160, 160))
            # 特征向量检测
            im_data = np.expand_dims(im_data, axis=0)
            emb = face_feature_sess.run(ff_embeddings,
                                         feed_dict={ff_images_placeholder: im_data, ff_train_placeholder: False})
            print(emb)                      #人脸特征数据

if __name__ == '__main__':
    path = 'data/1.jpg'
    face_feature(path)              #输出人脸特征数据