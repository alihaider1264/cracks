# importing the module
from cv2 import cv2

# read the image and store the data in a variable
image=cv2.imread("000400.jpg")

# make it grayscale
Gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

# Make canny Function
canny=cv2.Canny(Gray,40,140)

# the threshold is varies bw 0 and 255
cv2.imshow("Canny",canny)
cv2.imshow("Original",Gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
