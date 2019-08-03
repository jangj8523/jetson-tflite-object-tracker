import sys
import cv2
import paho.mqtt.client as mqtt
import threading
import imutils
import tensorflow as tf
import time
import numpy as np
import tensorflow.contrib.tensorrt as trt

sys.path.append("..")

from utils import label_map_util
from utils import visualization_utils as vis_util
from jetcam.csi_camera import CSICamera
from pyimagesearch.tensorTracker import TensorTracker
from collections import defaultdict
from matplotlib import pyplot as plt



# jetcam.csi_camera was found in the following link: https://github.com/NVIDIA-AI-IOT/jetcam
# Much faster CSICamera implementation, Frame processing is comparable with Deepstream SDK
# Deepstream SDK provides too much functionality compared to what we need. Deepstream SDK
# provides a comprehensive support for the Upstream infrastructure: (i.e. Video encoding-decoding, creating
# multiple data pipelines, 360 degree camera, etc.)

class SingleJetsonCam :
    MAX_DEPTH = 12000.0
    MIN_SCORE_THRESH = 0.4
    PATH_TO_LITE = 'coco_ssd_mobilenet_v1_1.0_quant_2018_06_29'
    LITE_MODEL_NAME = 'detect.tflite'
    NUM_CLASSES = 90


    def __init__(self):
        self._camera = None
        self._depth_list = []
        self._mqtt_cv = threading.Condition()
        self._lidar_message_received = False
        self._sub_channel = "intercept"
        self._pub_channel = "jetson"
        self._frame_tracker = TensorTracker()
        # Data has to be loaded. Hence its initialization is deferred
        self._category_index = []

    def _unnormalize(self, boxes):
        ymin, xmin, ymax, xmax = boxes
        (left, right, top, bottom) = (xmin * self._img_width, xmax * self._img_width,
                                      ymin * self._img_height, ymax * self._img_height)
        return [top, left, bottom, right]


    def _parse_config_files(self):
        class_label = label_map_util.load_labelmap(self.PATH_TO_LITE + '/label_map.pbtxt')
        categories = label_map_util.convert_label_map_to_categories(class_label, max_num_classes=self.NUM_CLASSES, use_display_name=True)
        self._category_index = label_map_util.create_category_index(categories)

    def _converter(self, entry):
        value = entry.astype(np.int32)
        return value + 1


    def _normalize_depth(self, depth):
        return depth / self.MAX_DEPTH

    def _on_message(self, client, userdata, message):
        self._mqtt_cv.acquire()
        raw_payload =  str(message.payload.decode("utf-8")).strip()
        parsed_payload = raw_payload.split(";")[:-1]
        payload = list(map(float, parsed_payload))

        # First entry of the payload is the unixTimeStamp
        unix_time_stamp = payload[0]

        self._depth_list = payload[2:]
        self._lidar_message_received = True
        self._mqtt_cv.notify()
        self._mqtt_cv.release()

    def _on_connect(self, client, userdata, flags, rc):
        if rc==0:
            print("connected OK Returned code=",rc)
        else:
            print("Bad connection Returned code=",rc)

    def _initialize_mqtt(self):
        host_name = "192.168.1.247"
        port = 1883
        client_id = "jjang_test"
        client = mqtt.Client(client_id=client_id)

        client.on_connect=self._on_connect
        client.on_message=self._on_message

        print ("===== CONNECT TO HOST ======")
        client.connect(host_name, port=port)
        client.loop_start()

        print ("===== SUBSCRIBE TO CHANNEL ======")
        client.subscribe(self._sub_channel)
        return client


    def _createCamera(self):
        camera = CSICamera(width=300, height=300, capture_width=1080, capture_height=720, capture_fps=30)
        # camera.running = True
        # camera.observe(self._callback, names='value')
        return camera

    def _create_payload(self, unix_time_stamp, bounding_box_list):
        payload = str(unix_time_stamp) + ';' + str(len(bounding_box_list)) + ';'
        for index in range(len(bounding_box_list)):
            payload += ','.join(map(str,bounding_box_list[index])) + ';'
        payload + ';'
        return payload

    def _runInferenceModel(self, mqtt_client, camera):
        with tf.Session() as sess:
            # create_inference_graph(
            #     input_saved_model_dir=input_saved_model_dir,
            #     precision_mode=”INT8",
            #     use_calibration=False)
            interpreter = tf.lite.Interpreter(model_path=self.PATH_TO_LITE + '/' + self.LITE_MODEL_NAME)
            interpreter.allocate_tensors()
            while True:
                unixTime = time.time()
                rects = []
                depth_list = ''
                bounding_box_list = []

                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()

                self._img_height = input_details[0]['shape'][1]
                self._img_width = input_details[0]['shape'][2]

                # Test model on random input data.
                img = camera.read()

                image_np = np.expand_dims(img, axis=0).astype(np.uint8)
                interpreter.set_tensor(input_details[0]['index'], image_np)
                interpreter.invoke()
                boxes = np.squeeze(interpreter.get_tensor(output_details[0]['index']))
                # print (boxes)
                boxes = np.array(list(map(self._unnormalize, boxes)))
                classes = np.array(list(map(self._converter, np.squeeze(interpreter.get_tensor(output_details[1]['index'])))))
                scores = np.squeeze(interpreter.get_tensor(output_details[2]['index']))
                num_detections = np.squeeze(interpreter.get_tensor(output_details[3]['index'])).astype(np.int8)

                # Bounding Box [ Top , left , bottom, right]
                # bounding_box_list = boxes
                bounding_box_list = [boxes[i] for i in range(num_detections) if scores is None or scores[i] > self.MIN_SCORE_THRESH]
                display_class_list = []
                self._depth_list =  np.empty((0,))

                if len(bounding_box_list) > 0:
                    #new_score, new_class = [list(i) for i in zip(*[(scores[j], classes[j]) for j in range(num_detections) if scores is None or scores[j] > self.MIN_SCORE_THRESH])]
                    new_score, new_class = scores, classes
                    payload = self._create_payload(unixTime, bounding_box_list)
                    print ("PUBLISH", payload, "\n")
                    mqtt_client.publish(self._pub_channel, payload)
                    self._mqtt_cv.acquire()
                    while not self._lidar_message_received:
                        self._mqtt_cv.wait()
                    self._lidar_message_received = False
                    self._mqtt_cv.release()
                    object_index, id_list = self._frame_tracker.update(bounding_box_list, self._depth_list)

                    # print ("object Index: ", object_index)
                    # print ("length of everyting else: ", len(bounding_box_list), len(new_score), len(new_class), len(depth_list) )

                    display_class_list = id_list
                    container = list(zip(*[(bounding_box_list[i], new_score[i], new_class[i], self._depth_list[i]) for i in object_index]))
                    boxes = np.array(container[0]) if len(container) != 0 else np.empty((0,))
                    scores = np.array(container[1]) if len(container) != 0 else np.array((0,))
                    classes = np.array(container[2]) if len(container) != 0 else np.array((0,))
                    depth_list = np.array(container[3]) if len(container) != 0 else np.array((0,))
                else:
                    object_index, id_list = ct.update(bounding_box_list, None)

                vis_util.visualize_boxes_and_labels_on_image_array(
                    img,
                    boxes,
                    classes,
                    scores,
                    self._category_index,
                    line_thickness=8,
                    depth_list = depth_list,
                    min_score_thresh = self.MIN_SCORE_THRESH,
                    track_ids=display_class_list)

                cv2.imshow('object detection', img)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                  cv2.destroyAllWindows()
                  break

    def runCamera (self):
        self._parse_config_files()
        client = self._initialize_mqtt()
        camera = self._createCamera()
        self._runInferenceModel(client, camera)

    # Unused callback. 
    # =======================================
    # def _callback(self, change):
    #     image = change['new']
    #     self.runInferenceModel(image)
    #     cv2.imshow('object detection', new_image)
    #     if cv2.waitKey(10) & 0xFF == ord('q'):
    #       cv2.destroyAllWindows()


singleJetCam = SingleJetsonCam()
singleJetCam.runCamera()




    # do some processing...
