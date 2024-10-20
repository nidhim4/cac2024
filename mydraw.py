import cv2
import numpy as np

def HED(image_path, output_path):
   hed_model = cv2.dnn.readNetFromCaffe("deploy.prototxt", "hed_pretrained_bsds.caffemodel")

   image = cv2.imread(image_path)

   height, width = image.shape[:2]
   max_size = max(height, width)
   scale = 1.0
   if max_size > 1000:
      scale = 1000.0 / max_size
   resized = cv2.resize(image, None, fx=scale, fy=scale)

   resized_height, resized_width = resized.shape[:2]
   hed_model.setInput(cv2.dnn.blobFromImage(resized, scalefactor=1.0, size=(resized_width, resized_height),
                                            mean=(104.00698793, 116.66876762, 122.67891434), swapRB=False, crop=False))
   hed = hed_model.forward()
   hed = np.squeeze(hed)

   hed = np.uint8(hed * 255)
   hed = cv2.bitwise_not(hed)

   cv2.imwrite(output_path, hed)

   return output_path