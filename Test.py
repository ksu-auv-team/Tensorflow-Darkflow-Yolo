#%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pprint as pp
from darkflow.net.build import TFNet
import cv2
def boxing(original_img, predictions):
    newImage = np.copy(original_img)

    for result in predictions:
        top_x = result['topleft']['x']
        top_y = result['topleft']['y']

        btm_x = result['bottomright']['x']
        btm_y = result['bottomright']['y']

        confidence = result['confidence']
        label = result['label'] + " " + str(round(confidence, 3))

        if confidence > 0.0:
            newImage = cv2.rectangle(newImage, (top_x, top_y), (btm_x, btm_y), (255,0,0), 3)
            newImage = cv2.putText(newImage, label, (top_x, top_y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL , 0.8, (0, 230, 0), 1, cv2.LINE_AA)

    return newImage


options = {"model": "cfg/ksyolo.cfg", "load": "bin/tiny-yolo-voc.weights", "threshold": 0.0}

tfnet = TFNet(options)


original_img = cv2.imread("./sample_img/gopro.jpg")
original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
results = tfnet.return_predict(original_img)

# pp.pprint(results)

fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(original_img)
plt.show()

fig, ax = plt.subplots(figsize=(20, 10))
ax.imshow(boxing(original_img, results))
plt.savefig('test.png')
plt.show()