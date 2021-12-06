# importing the module
from cv2 import cv2

# read the image and store the data in a variable
image=cv2.imread("_data/surface-cracks/Cracked/00001.jpg")

# make it grayscale
Gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

# Make it with the help of sobel
# make the sobel_horizontal
# For horizontal x axis=1 and yaxis=0
# for vertical x axis=0 and y axis=1
Horizontal=cv2.Sobel(Gray,0,1,0,cv2.CV_64F)

# the thresholds are like 
# (variable,0,<x axis>,<y axis>,cv2.CV_64F)
Vertical=cv2.Sobel(Gray,0,0,1,cv2.CV_64F)

# DO the Bitwise operation
Bitwise_Or=cv2.bitwise_or(Horizontal,Vertical)

# Show the Edged Image
cv2.imshow("Sobel Image",Bitwise_Or)
cv2.imshow("Original Image",Gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
