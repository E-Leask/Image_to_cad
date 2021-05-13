import cv2
import numpy as np

filename = 'output.png'
img = cv2.imread(filename)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

gray = np.float32(gray)
for i in range(1,10, 2):
    dst = cv2.cornerHarris(gray,2,i,0.1)

    #result is dilated for marking the corners, not important
    dst = cv2.dilate(dst,None)

    # Threshold for an optimal value, it may vary depending on the image.
    img[dst>0.01*dst.max()]=[0,0,255]

    cv2.imshow('dst' + str(i),img)


if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()