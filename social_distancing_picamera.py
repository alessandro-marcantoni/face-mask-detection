import tflite_runtime.interpreter as tflite
import numpy as np
import cv2
import time
import picamera
import io
import math
from PIL import Image
from utils.annotation import Annotator
from itertools import combinations

threshold = 0.6
person_class = 0
camera_width = 640
camera_height = 480

interpreter = tflite.Interpreter("./models/detect.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
_, input_height, input_width, _ = interpreter.get_input_details()[0]["shape"]

def detect_people(frame):
	#frame = cv2.resize(frame, (input_width, input_height))
	frame = np.expand_dims(frame, 0)
	
	interpreter.set_tensor(input_details[0]["index"], frame)
	interpreter.invoke()
	
	boxes = np.squeeze(interpreter.get_tensor(output_details[0]["index"]))
	classes = np.squeeze(interpreter.get_tensor(output_details[1]["index"]))
	scores = np.squeeze(interpreter.get_tensor(output_details[2]["index"]))
	count = int(np.squeeze(interpreter.get_tensor(output_details[3]["index"])))
	
	people = []
	for i in range(count):
		if scores[i] >= threshold and classes[i] == person_class:
			person = {
				"bounding_box": boxes[i],
				"score": scores[i]
			}
			people.append(person)
	return people

def annotate_people(annotator, people):
	centroids = dict()
	for n, person in enumerate(people):
		ymin, xmin, ymax, xmax = person["bounding_box"]
		cX = (xmin + xmax) / 2.0
		cY = (ymin + ymax) / 2.0
		centroids[n] = (cX, cY)
	
	people_at_risk = set()
	for (id1, centroid1), (id2, centroid2) in combinations(centroids.items(), 2):
		dx, dy = centroid1[0] - centroid2[0], centroid1[1] - centroid2[1]
		distance = math.sqrt(dx * dx + dy * dy)
		if distance < 0.5:
			people_at_risk.add(id1)
			people_at_risk.add(id2)
	
	for n, person in enumerate(people):
		ymin, xmin, ymax, xmax = person["bounding_box"]
		xmin = int(xmin * camera_width)
		xmax = int(xmax * camera_width)
		ymin = int(ymin * camera_height)
		ymax = int(ymax * camera_height)
		if n in people_at_risk:
			annotator.bounding_box([xmin, ymin, xmax, ymax], outline="red")
		else:
			annotator.bounding_box([xmin, ymin, xmax, ymax], outline="green")
		
with picamera.PiCamera(resolution=(camera_width, camera_height), framerate=30) as camera:
	camera.start_preview()
	try:
		stream = io.BytesIO()
		annotator = Annotator(camera)
		for _ in camera.capture_continuous(stream, format="jpeg", use_video_port=True):
			stream.seek(0)
			frame = Image.open(stream).convert("RGB").resize((input_width, input_height), Image.ANTIALIAS)
			inference_start = time.time()
			people = detect_people(frame)
			inference_stop = time.time()
			
			annotator.clear()
			annotate_people(annotator, people)
			annotator.text([5, 0], "infer time: %.3f s" % (inference_stop - inference_start))
			annotator.update()
			
			stream.seek(0)
			stream.truncate()
	finally:
		camera.stop_preview()

'''cam = cv2.VideoCapture(0)
rval, frame = cam.read() if cam.isOpened() else False, None
		

while rval:
	loop_start = time.time()
	rval, frame = cam.read()
	camera_height, camera_width, _ = frame.shape
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	inference_start = time.time()
	people = detect_people(frame)
	frame = annotate_people(frame, people, camera_height, camera_width)
	inference_stop = time.time()

	key = cv2.waitKey(20)
	if key == 27:
		break
	loop_stop = time.time()
	cv2.putText(frame, "fps: %.2f - infer time: %.3f s" % ((1 / (loop_stop - loop_start)), inference_stop - inference_start), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0))
	cv2.imshow("image", frame[:, :, ::-1])'''
