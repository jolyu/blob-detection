import cv2
import numpy as np

# read image in greyscale

img = cv2.imread('Penguins.jpg', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (650,500))

#determine cv version
print(cv2.__version__)
if cv2.__version__.startswith('2'):
    detector = cv2.SimpleBlobDetector()
else:
    detector = cv2.SimpleBlobDetector_create()

# detect suff
keypoints = detector.detect(img)

#draw detected keypoints as red circles
imgKeyPoints = cv2.drawKeypoints(img,keypoints, np.array([]),(0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

#display results
cv2.imshow("keypoints", imgKeyPoints)
cv2.waitKey(0)

cv2.destroyAllWindows()