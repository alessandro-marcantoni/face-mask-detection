import cv2
import tflite_runtime.interpreter as tflite
import numpy as np
import glob
from xml.dom import minidom
from face_detector import FaceMaskDetector
import time

# Preparazione etichette
labels_paths = [path for path in glob.glob('hard_dataset/*.xml')]
labels_paths.sort()
labels = []
for label_path in labels_paths:
  labels.append([1 if l.firstChild.data=='face' else 0 for l in minidom.parse(label_path).getElementsByTagName('name')])

# Preparazione immagini
data_paths = [path for path in (glob.glob('hard_dataset/*.jpg') + glob.glob('hard_dataset/*.jpeg') + glob.glob('hard_dataset/*.png'))]
data_paths.sort()
data = []
for data_path in data_paths:
  img = cv2.cvtColor(cv2.imread(data_path), cv2.COLOR_BGR2RGB)
  data.append(img)

# Dichiarazione modelli
face_mask_detector = FaceMaskDetector()

# Esecuzione test
output = []

start = time.time()
for image in data:
  boxes = face_mask_detector.detect_faces(image)
  output_info, _ = face_mask_detector.detect_masks(image, boxes)
  output.append(output_info)
stop = time.time()

# Misure di accuratezza
wrong = 0
'''     people_not_detected:  persone presenti in label ma non riconosciute dalla rete
    people_wrongly_detected:  persone non presenti in label ma riconosciute dalla rete '''
people_not_detected, people_wrongly_detected = 0, 0
'''      mask_correct:  label mascherina e output mascherina        (TP)
       mask_incorrect:  label no mascherina e output mascherina     (FP)
      no_mask_correct:  label no mascherina e output no mascherina  (TN)
    no_mask_incorrect:  label mascherina e output no mascherina     (FN) '''
mask_correct, mask_incorrect, no_mask_correct, no_mask_incorrect = 0, 0, 0, 0
for list_idx, output_list in enumerate(output):
  longest, shortest = None, None
  if output_list != labels[list_idx]:
    wrong = wrong + 1
  if len(output_list) != len(labels[list_idx]):
    longest  = output_list if len(output_list) > len(labels[list_idx]) else labels[list_idx]
    shortest = output_list if len(output_list) < len(labels[list_idx]) else labels[list_idx]
    if longest == output_list:
      people_wrongly_detected = people_wrongly_detected + len(longest) - len(shortest)
    if longest == labels[list_idx]:
      people_not_detected = people_not_detected + len(longest) - len(shortest)
  else:
    shortest = output_list
  for idx in range(len(shortest)):
    if output_list[idx] == 0 and labels[list_idx][idx] == 0:
      mask_correct = mask_correct + 1
    if output_list[idx] == 1 and labels[list_idx][idx] == 1: 
      no_mask_correct = no_mask_correct + 1
    if output_list[idx] == 0 and labels[list_idx][idx] == 1:
      mask_incorrect = mask_incorrect + 1
    if output_list[idx] == 1 and labels[list_idx][idx] == 0:
      no_mask_incorrect = no_mask_incorrect + 1

accuracy            = (mask_correct + no_mask_correct) / (mask_correct + mask_incorrect + no_mask_correct + no_mask_incorrect)
precision_face_mask =    mask_correct / (   mask_correct +    mask_incorrect)
precision_face      = no_mask_correct / (no_mask_correct + no_mask_incorrect)
recall_face_mask    =    mask_correct / (   mask_correct + no_mask_incorrect)
recall_face         = no_mask_correct / (no_mask_correct +    mask_incorrect)
f1_face_mask        = 2 * precision_face_mask * recall_face_mask / (precision_face_mask + recall_face_mask)
f1_face             = 2 * precision_face      * recall_face      / (precision_face      + recall_face     )

print("           Accuracy:",            accuracy, "\n")  
print("Precision face-mask:", precision_face_mask      )
print("     Precision face:",      precision_face, "\n")
print("   Recall face-mask:",    recall_face_mask      )
print("        Recall face:",         recall_face, "\n")
print("       F1 face-mask:",        f1_face_mask      )
print("            F1 face:",             f1_face, "\n")

print("     Total Accuracy:", (len(output)-wrong)/len(output), "\n")

print("     Total Time (s):", stop - start)
print("   Average Time (s):", (stop -start) / len(data))
