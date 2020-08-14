import cv2
import tensorflow as tf
import numpy as np
from fdet import RetinaFace
import time
from PIL import Image

detector = RetinaFace(backbone="MOBILENET")
face_cascade = cv2.CascadeClassifier()
face_cascade.load("haarcascade_frontalface_default.xml")
target_shape = (224, 224)
color_map = {
    0: (0, 255, 0),
    1: (255, 0, 0)
}
class_map = {
    0: "Mask",
    1: "No Mask"
}

def face_detection(image):
    #image_resized = cv2.resize(image, (640, 480))
    #image_np = image_resized / 255.0
    '''image_exp = np.expand_dims(image_resized, axis=0)
    interpreter = tf.lite.Interpreter(model_path="retinaface_r50_v1.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    image_tensor = np.array(image_exp, dtype=np.float32)
    interpreter.set_tensor(input_details[0]["index"], image_tensor)
    interpreter.invoke()
    result = interpreter.get_tensor(output_details[0]["index"])[0]
    print(result.shape)'''
    '''gt0 = values[0] > -10
    keep_idx = []
    for element in gt0:
        keep_idx.append(element[0])

    #print(keep_idx)
    bboxes = np.array(bboxes[0,keep_idx,:4], dtype=np.int)
    return bboxes'''
    faces = face_cascade.detectMultiScale(image, 1.1, 3)
    return faces

def face_mask_inference(image):
    interpreter = tf.lite.Interpreter(model_path="mask_detector.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    image_tensor = np.array(image, dtype=np.float32)
    interpreter.set_tensor(input_details[0]["index"], image_tensor)
    interpreter.invoke()
    probabilities = interpreter.get_tensor(output_details[0]["index"])
    return probabilities

def inference(image):
    img = cv2.resize(image, target_shape)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    probabilities = face_mask_inference(img)
    return probabilities

def visualize_detections(image, boxes):
    image = np.array(image, dtype=np.uint8)

    output_lists = []
    for box in boxes:
        x, y, w, h = box
        face_image = image[y : y+h, x : x+w]
        if face_image.shape[0] and face_image.shape[1]:
            output_info = inference(face_image)
            output_class = np.argmax(output_info)
            cv2.rectangle(image, (x, y), (x+w, y+h), color_map[output_class], 2)
            cv2.putText(image, class_map[output_class], (x+2, y+2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_map[output_class])
    return image

def get_boxes(result):
    boxes = []
    for i in range(len(result)):
        boxes.append(result[i]["box"])
    boxes = np.array(boxes)
    return boxes

if __name__ == "__main__":
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
        #boxes = face_detection(frame)
        result = detector.detect(frame)
        boxes = get_boxes(result)
        frame = visualize_detections(frame, boxes)

        key = cv2.waitKey(20)
        if key == 27:
            break
        stop = time.time()
        cv2.putText(frame, "fps: %.2f" % (1 / (stop - start)), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0))
        cv2.imshow("image", frame[:, :, ::-1])
        #rval = False
    vc.release()
    cv2.destroyWindow("preview")