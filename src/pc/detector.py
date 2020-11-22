import tensorflow as tf
import cv2
import numpy as np
import time

from settings import MODEL_PATH, CONFIDENCE


class SalmonDetector:

    def __init__(self):
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(MODEL_PATH, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            self.sess = tf.Session(graph=detection_graph)

        self.image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        self.boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        self.scores = detection_graph.get_tensor_by_name('detection_scores:0')
        self.classes = detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    def detect_objects(self, image_np):
        # Expand dimensions since the models expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)

        # Actual detection.
        return self.sess.run([self.boxes, self.scores, self.classes, self.num_detections],
                             feed_dict={self.image_tensor: image_np_expanded})

    def detect_from_images(self, frame):
        [frm_height, frm_width] = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st_time = time.time()

        (boxes, scores, classes, _) = self.detect_objects(frame_rgb)
        print(f"detection time: {time.time() - st_time}")
        detect_rect_list = []

        for i in range(len(scores[0])):
            if scores[0][i] > CONFIDENCE:
                x1, y1 = int(boxes[0][i][1] * frm_width), int(boxes[0][i][0] * frm_height)
                x2, y2 = int(boxes[0][i][3] * frm_width), int(boxes[0][i][2] * frm_height)
                detect_rect_list.append([x1, y1, x2, y2])

        return detect_rect_list


if __name__ == '__main__':
    image = cv2.imread("/media/main/Data/Task/SalmonDetectorNano/training_data/images/salmon0.jpg")
    rects = SalmonDetector().detect_from_images(frame=image)
    for rect in rects:
        left, top, right, bottom = rect
        cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 5)
    cv2.imshow("Salmon", cv2.resize(image, None, fx=0.2, fy=0.2))
    cv2.waitKey()
