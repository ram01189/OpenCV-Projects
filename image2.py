import numpy as np
import cv2
#cap=cv2.VideoCapture(0)
#ret,frame=cap.read()
#cv2.imshow('image',cap)
cap=cv2.imread('1.jpg')
#convert image to grayscale
imggray=cv2.cvtColor(cap,cv2.COLOR_BGR2GRAY) 
#converts an Image to the Binary Image
ret,thresh1=cv2.threshold(imggray,100,255,cv2.THRESH_BINARY) 
blur=cv2.medianBlur(thresh1,5)
kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
erosion=cv2.erode(blur,kernel,iterations=6)
dilation1=cv2.dilate(erosion,kernel,iterations=4)
dilation= np.invert(dilation1)
edges=cv2.Canny(blur,100,150)
###########################################***************BLOB Detection********************################################
# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()
# Change thresholds
params.minThreshold = 10
params.maxThreshold = 100
# Filter by Area.
params.filterByArea = True
params.minArea = 1500
# Filter by Circularity
params.filterByCircularity = False
params.minCircularity = 0.1

# Filter by Convexity
params.filterByConvexity = False
params.minConvexity = 0.87
# Filter by Inertia
params.filterByInertia = False
params.minInertiaRatio = 0.01
detector = cv2.SimpleBlobDetector_create(params)
keypoints = detector.detect(dilation)
im_with_keypoints = cv2.drawKeypoints(dilation, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow("Keypoints", im_with_keypoints)
print len(keypoints)
###########################################################*************End Of Blob Detection********************
cv2.imshow('dilation_then_ersion',dilation)
cv2.waitKey(0)
cv2.destroyAllWindows()
