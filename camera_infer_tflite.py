import cv2
import tensorflow as tf
import numpy as np
import time

class FaceMaskDetector:
    """
    Class representing the face mask detector.
    It contains all the fields and the methods that make it possible to make the detections.
    """
    def __init__(self):
        self.color_map = {
            0: (0, 255, 0),
            1: (255, 0, 0)
        }
        self.class_map = {
            0: "Mask",
            1: "No Mask"
        }

        # Definition of the Face Detection interpreter.
        self.face_target_shape = (320, 320)
        self.face_interpreter = tf.lite.Interpreter(model_path="models/detector.tflite")
        self.face_interpreter.allocate_tensors()
        self.face_input_details = self.face_interpreter.get_input_details()
        self.face_output_details = self.face_interpreter.get_output_details()

        # Definition of the Mask Detection Interpreter.
        self.mask_target_shape = (224, 224)
        self.mask_interpreter = tf.lite.Interpreter(model_path="models/mask_detector.tflite")
        self.mask_interpreter.allocate_tensors()
        self.mask_input_details = self.mask_interpreter.get_input_details()
        self.mask_output_details = self.mask_interpreter.get_output_details()

    def preprocess(self, image, target_height=320, target_width=320):
        """
        Preprocess of every frame in order to be given as input to the face detection model.

        Parameters
        ----------
        image
            The unprocessed image.
        target_height: int
            The target height for the image (default is 320).
        target_width: int
            The target width for the image (default is 320).
        
        Returns
        -------
        np.Array
            Array representing the processed image.
        float
            The x scale.
        float
            The y scale.
        float
            The x offset.
        float
            The y offset.

        """
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

    def py_nms(self, bboxes, iou_thres=0.2, score_thres=0.8, max_boxes=1000):
        """
        Application of non max suppression.

        Parameters
        ----------
        bboxes
            The faces' boxes.
        iou_thres
            The intersection over union threshold (default is 0.2).
        score_thres
            The score treshold (default is 0.8).
        max_boxes
            The maximum number of boxes (default is 1000).
        
        Returns
        -------
        list
            The effective boxes to keep and to be sent to the face mask detection model.
        """
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

    def detect_faces(self, image):
        """
        Detection of the faces in the webcam frames.

        Parameters
        ----------
        image
            The frame recorded by the webcam.
        
        Returns
        -------
        list
            The list of faces' boxes.
        """
        image_for_net, scale_x, scale_y, dx, dy = self.preprocess(image,
                                                                    target_height=self.face_target_shape[0],
                                                                    target_width=self.face_target_shape[1])
        image_for_net = np.expand_dims(image_for_net, 0).astype(np.float32)

        self.face_interpreter.set_tensor(self.face_input_details[0]["index"], image_for_net)
        self.face_interpreter.invoke()
        bboxes = self.face_interpreter.get_tensor(self.face_output_details[0]["index"])
        bboxes = self.py_nms(np.array(bboxes[0]))

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
        """
        Infers the face mask's precence or absence.

        Parameters
        ----------
        image
            The image to classify.
        
        Returns
        -------
        list
            The list of probabilities for both classes (Mask and NoMask).
        """
        img = cv2.resize(image, self.mask_target_shape)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        image_tensor = np.array(img, dtype=np.float32)
        self.mask_interpreter.set_tensor(self.mask_input_details[0]["index"], image_tensor)
        self.mask_interpreter.invoke()
        probabilities = self.mask_interpreter.get_tensor(self.mask_output_details[0]["index"])
        return probabilities

    def detect_masks(self, image, boxes):
        """
        Takes all the faces' boxes and classifies each one as Mask or NoMask.
        In the end, draws the boxes on the original frame.

        Parameters
        ----------
        image
            The frame recorded by the webcam.
        boxes
            The faces' boxes to classify.
        
        Returns:
        image
            The original frame with all the boxes drawn.
        """
        image = np.array(image, dtype=np.uint8)
        boxes = np.array(boxes, dtype=np.uint16)
        output_lists = []
        for box in boxes:
            x1, y1, x2, y2, _ = box
            face_image = image[y1 : y2, x1 : x2]
            if face_image.shape[0] and face_image.shape[1]:
                face_image = image[y1 : y2, x1 : x2]
                output_info = self.face_mask_inference(face_image)
                output_class = np.argmax(output_info)
                cv2.rectangle(image, (x1, y1), (x2, y2), self.color_map[output_class], 2)
                cv2.putText(image, self.class_map[output_class], (x1+2, y1-2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.color_map[output_class])
        return image

if __name__ == "__main__":
    """
    Main body of the application.
    """
    face_mask_detector = FaceMaskDetector()

    # Creates a new window and starts the webcam stream.
    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)
    fps = vc.get(cv2.CAP_PROP_FPS)

    rval, frame = vc.read() if vc.isOpened() else False, None
    
    ''' 
    Main loop:
        * Takes a frame from the webcam stream.
        * Detects faces.
        * Detects masks.
        * Visualize output.
    '''
    while rval:
        loop_start = time.time()
        rval, frame = vc.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        inference_start = time.time()
        boxes = face_mask_detector.detect_faces(frame)
        frame = face_mask_detector.detect_masks(frame, boxes)
        inference_stop = time.time()

        key = cv2.waitKey(20)
        if key == 27:
            break
        loop_stop = time.time()
        cv2.putText(frame, "fps: %.2f - infer time: %.3f s" % ((1 / (loop_stop - loop_start)), inference_stop - inference_start), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0))
        cv2.imshow("image", frame[:, :, ::-1])
        

    vc.release()
    cv2.destroyWindow("preview")