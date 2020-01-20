import cv2
import numpy as np

# read image in greyscale

img = cv2.imread('Bilder/Birds.jpg', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (650,500))

# read image in Black/white
(thresh, blackAndWhiteImage) = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()

# Change thresholds
params.minThreshold = 100
params.maxThreshold = 255


# Filter by Area.
params.filterByArea = True
params.minArea = 1

# Filter by Circularity
params.filterByCircularity = False
params.minCircularity = 0.1

# Filter by Convexity
params.filterByConvexity = False
params.minConvexity = 0.87

# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.01


#determine cv version
print(cv2.__version__)
if cv2.__version__.startswith('2'):
    detector = cv2.SimpleBlobDetector(params)
else:
    detector = cv2.SimpleBlobDetector_create(params)


# detect suff
keypoints = detector.detect(img)
 # for black and white image
keypoints_1 = detector.detect(blackAndWhiteImage)

#draw detected keypoints as red circles
imgKeyPoints = cv2.drawKeypoints(img,keypoints, np.array([]),(0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

 # for black and White image 
blackAndWhiteImageKeyPoints = cv2.drawKeypoints(blackAndWhiteImage,keypoints_1, np.array([]),(0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

#display results
cv2.imshow("keypoints_grey", imgKeyPoints)
cv2.imshow("keypoints_BlackAndWhite",blackAndWhiteImageKeyPoints)
cv2.waitKey(0)

cv2.destroyAllWindows()