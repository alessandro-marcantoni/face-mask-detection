import tflite_runtime.interpreter as tflite
import numpy as np
import cv2
import time
import math
from itertools import combinations

DISTANCE = 0.05
threshold = 0.5
person_class = 0
camera_width = 640
camera_height = 480

interpreter = tflite.Interpreter("./models/detect.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
_, input_height, input_width, _ = interpreter.get_input_details()[0]["shape"]

def detect_people(frame):
	frame = cv2.resize(frame, (input_width, input_height))
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

def annotate_people(frame, people, camera_height, camera_width):
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
		if distance < DISTANCE:
			people_at_risk.add(id1)
			people_at_risk.add(id2)
	
	for n, person in enumerate(people):
		ymin, xmin, ymax, xmax = person["bounding_box"]
		xmin = int(xmin * camera_width)
		xmax = int(xmax * camera_width)
		ymin = int(ymin * camera_height)
		ymax = int(ymax * camera_height)
		if n in people_at_risk:
			cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
		else:
			cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
		
	return frame

cam = cv2.VideoCapture("./samples/example.mp4")
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
	cv2.imshow("image", frame[:, :, ::-1])
