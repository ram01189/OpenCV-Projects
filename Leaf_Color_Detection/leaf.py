import numpy as np
import cv2
count=0
im=cv2.imread('22.jpg')           #reading an image from file
a=cv2.resize(im,None,fx=0.15,fy=0.15,interpolation=cv2.INTER_AREA)  #Reducing the size of an image
hsv=cv2.cvtColor(a,cv2.COLOR_BGR2HSV) #converting image to hsv
image_mask=cv2.inRange(hsv,np.array([20,50,50]),np.array([100,255,255])) # applying masking to detect the green colour 
out=cv2.bitwise_and(a,a,mask=image_mask) # performing and operation with image and mask to show green area of image
kernel = np.ones((5,5),np.uint8)
dilation = cv2.dilate(image_mask,kernel,iterations = 2) #dilating the making image
dil=dilation.ravel() # to convert the multidimensional array to the 1d array
for x in np.nditer(dil):
	if (x==255):
                count=count+1  # number of white pixels
total=float(np.size(dil)) # total number of pixels 
c=float(count) 
print((c/total)*100)   #calculating the percentage of the green pixels
cv2.imshow('im',im)
cv2.waitKey(0)
cv2.destroyAllWindows()
