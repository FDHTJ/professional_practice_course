import cv2
image=cv2.imread('10038.png')
image[image==1]=255
image[image==2]=123
cv2.imwrite('image.png',image)
