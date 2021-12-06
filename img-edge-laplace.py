# importing the module
from cv2 import cv2

# read the image and store the data in a variable
image=cv2.imread("000400.jpg")

# make it grayscale
Gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

# Make Laplacian Function
Lappy=cv2.Laplacian(Gray,cv2.CV_64F)

cv2.imshow("Laplacian",Lappy)
cv2.imshow("Original",Gray)
cv2.waitKey(0)
cv2.destroyAllWindows()