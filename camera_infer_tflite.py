import cv2
import tensorflow as tf
import numpy as np
import time

class FaceMaskDetector:
    def __init__(self):
        self.mask_target_shape = (224, 224)
        self.color_map = {
            0: (0, 255, 0),
            1: (255, 0, 0)
        }
        self.class_map = {
            0: "Mask",
            1: "No Mask"
        }

        self.face_target_shape = (320, 320)
        self.face_interpreter = tf.lite.Interpreter(model_path="detector.tflite")
        self.face_interpreter.allocate_tensors()
        self.face_input_details = self.face_interpreter.get_input_details()
        self.face_output_details = self.face_interpreter.get_output_details()

    def preprocess(self, image, target_height=320, target_width=320):
        h, w, c = image.shape
        bimage = np.zeros(shape=[target_height, target_width, c], dtype=image.dtype) + np.array([127., 127., 127.], dtype=image.dtype)

        scale_y = target_height / h
        scale_x = target_width / w

        scale = min(scale_x, scale_y)

        image = cv2.resize(image, None, fx=scale, fy=scale)
        h_, w_, _ = image.shape

        dx = (target_width-w_)//2
        dy = (target_height-h_)//2
        bimage[dy:h_+dy, dx:w_+dx, :] = image

        return bimage, scale, scale, dx, dy

    def py_nms(self, bboxes, iou_thres, score_thres,max_boxes=1000):
        upper_thres = np.where(bboxes[:, 4] > score_thres)[0]
        bboxes = bboxes[upper_thres]

        x1 = bboxes[:, 0]
        y1 = bboxes[:, 1]
        x2 = bboxes[:, 2]
        y2 = bboxes[:, 3]

        order = np.argsort(bboxes[:, 4])[::-1]

        keep = []
        while order.shape[0] > 0:
            cur = order[0]
            keep.append(cur)

            area = (bboxes[cur, 2] - bboxes[cur, 0]) * (bboxes[cur, 3] - bboxes[cur, 1])

            x1_reain = x1[order[1:]]
            y1_reain = y1[order[1:]]
            x2_reain = x2[order[1:]]
            y2_reain = y2[order[1:]]

            xx1 = np.maximum(bboxes[cur, 0], x1_reain)
            yy1 = np.maximum(bboxes[cur, 1], y1_reain)
            xx2 = np.minimum(bboxes[cur, 2], x2_reain)
            yy2 = np.minimum(bboxes[cur, 3], y2_reain)

            intersection = np.maximum(0, yy2 - yy1) * np.maximum(0, xx2 - xx1)
            iou = intersection / (area + (y2_reain - y1_reain) * (x2_reain - x1_reain) - intersection)
            ##keep the low iou
            low_iou_position = np.where(iou < iou_thres)[0]
            order = order[low_iou_position + 1]

        return bboxes[keep]

    def face_detection(self, image):
        image_for_net, scale_x, scale_y, dx, dy = self.preprocess(image,
                                                                    target_height=self.face_target_shape[0],
                                                                    target_width=self.face_target_shape[1])
        image_for_net = np.expand_dims(image_for_net, 0).astype(np.float32)

        self.face_interpreter.set_tensor(self.face_input_details[0]["index"], image_for_net)
        self.face_interpreter.invoke()
        bboxes = self.face_interpreter.get_tensor(self.face_output_details[0]["index"])
        bboxes = self.py_nms(np.array(bboxes[0]), iou_thres=0.2, score_thres=0.8)

        boxes_scaler = np.array([(self.face_target_shape[1]) / scale_x,
                           (self.face_target_shape[0]) / scale_y,
                           (self.face_target_shape[1]) / scale_x,
                           (self.face_target_shape[0]) / scale_y,1.], dtype='float32')

        boxes_bias=np.array([dx / scale_x,
                           dy / scale_y,
                           dx / scale_x,
                           dy / scale_y,0.], dtype='float32')
        bboxes = bboxes * boxes_scaler-boxes_bias

        return bboxes

    def face_mask_inference(self, image):
        interpreter = tf.lite.Interpreter(model_path="mask_detector.tflite")
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        image_tensor = np.array(image, dtype=np.float32)
        interpreter.set_tensor(input_details[0]["index"], image_tensor)
        interpreter.invoke()
        probabilities = interpreter.get_tensor(output_details[0]["index"])
        return probabilities

    def inference(self, image):
        img = cv2.resize(image, self.mask_target_shape)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        probabilities = self.face_mask_inference(img)
        return probabilities

    def visualize_detections(self, image, boxes):
        image = np.array(image, dtype=np.uint8)
        boxes = np.array(boxes, dtype=np.uint16)
        output_lists = []
        for box in boxes:
            x1, y1, x2, y2, _ = box
            face_image = image[y1 : y2, x1 : x2]
            if face_image.shape[0] and face_image.shape[1]:
                face_image = image[y1 : y2, x1 : x2]
                output_info = self.inference(face_image)
                output_class = np.argmax(output_info)
                cv2.rectangle(image, (x1, y1), (x2, y2), self.color_map[output_class], 2)
                cv2.putText(image, self.class_map[output_class], (x1+2, y1-2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.color_map[output_class])
        return image

    def get_boxes(self, result):
        boxes = []
        for i in range(len(result)):
            boxes.append(result[i]["box"])
        boxes = np.array(boxes)
        return boxes

if __name__ == "__main__":
    face_mask_detector = FaceMaskDetector()
    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)
    fps = vc.get(cv2.CAP_PROP_FPS)

    if vc.isOpened():
        rval, frame = vc.read()
    else:
        rval = False
    
    while rval:
        start = time.time()

        rval, frame = vc.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes = face_mask_detector.face_detection(frame)
        frame = face_mask_detector.visualize_detections(frame, boxes)

        key = cv2.waitKey(20)
        if key == 27:
            break
        stop = time.time()
        cv2.putText(frame, "fps: %.2f - time: %.3f s" % ((1 / (stop - start)), stop - start), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0))
        cv2.imshow("image", frame[:, :, ::-1])
    vc.release()
    cv2.destroyWindow("preview")