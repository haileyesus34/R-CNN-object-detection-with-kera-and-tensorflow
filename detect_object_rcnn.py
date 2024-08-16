from imagesearch.nms import non_max_suppression
from imagesearch import config 
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input 
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model 
import numpy as np 
import imutils 
import pickle 
import cv2 

model = load_model(config.model_path)
lb = pickle.loads(open(config.encoder_path, 'rb').read())

image = cv2.imread('/Users/hanna m/machinelearning/deep_learning/cv/r-cnn/raccons/images/raccoon-11.jpg')
image = imutils.resize(image, width=500)

ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
ss.setBaseImage(image)
ss.switchToSelectiveSearchFast()
rects = ss.process()

proposals = []
boxes = []

for (x,y,w,h) in rects[:config.max_proposals_infer]:
    roi = image[y:y+h, x:x+w]
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    roi = cv2.resize(roi, config.input_dims, interpolation=cv2.INTER_CUBIC)

    roi = img_to_array(roi)
    roi = preprocess_input(roi)

    proposals.append(roi)
    boxes.append((x,y, x+w, y+h))

proposals = np.array(proposals, dtype='float32')
boxes = np.array(boxes, dtype='int32')
print(boxes)

proba = model.predict(proposals)
labels = lb.classes_[np.argmax(proba, axis=1)]
idxs = np.where(labels == 'raccoon')[0]

boxes = boxes[idxs]
proba = proba[idxs][:,1]

idxs = np.where(proba >= config.min_proba)
boxes = boxes[idxs]
proba = proba[idxs]

clone = image.copy()

for (box, prob) in zip(boxes, proba):
        print(box)
        x_min, y_min, x_max, y_max = box 
        cv2.rectangle(clone, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        y = y_min - 10 if x_min - 10 > 10 else y_min + 10
        text = 'Raccon: {:.2f}%'.format(prob*100)
        cv2.putText(clone, text, (x_min, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
    
cv2.imshow('Before NMS', clone)

boxIdxs = non_max_suppression(boxes, proba)
# loop over the bounding box indexes
for i in boxIdxs:
	# draw the bounding box, label, and probability on the image
	(startX, startY, endX, endY) = boxes[i]
	cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
	y = startY - 10 if startY - 10 > 10 else startY + 10
	text= "Raccoon: {:.2f}%".format(proba[i] * 100)
	cv2.putText(image, text, (startX, y),
		cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
# show the output image *after* running NMS
cv2.imshow("After NMS", image)
cv2.waitKey(0)