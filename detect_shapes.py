# USAGE
# python detect_shapes.py --image shapes_and_colors.png

# import the necessary packages
from pyimagesearch.shapedetector import ShapeDetector
import argparse
import imutils
import cv2 as cv

def show(image):
    cv.imshow("Image", image)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
args = vars(ap.parse_args())

# load the image and resize it to a smaller factor so that
# the shapes can be approximated better
image = cv.imread(args["image"])
resized = imutils.resize(image, width=300)
ratio = image.shape[0] / float(resized.shape[0])

# convert the resized image to grayscale, blur it slightly,
# and threshold it
gray = cv.cvtColor(resized, cv.COLOR_BGR2GRAY)
show(gray)
cv.waitKey(0)
blurred = cv.GaussianBlur(gray, (5 , 5), 0)
show(blurred)
cv.waitKey(0)
thresh = cv.threshold(blurred, 127, 255, cv.THRESH_BINARY_INV)[1]
show(thresh)
cv.waitKey(0)

# find contours in the thresholded image and initialize the
# shape detector
cnts = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL,
	cv.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
sd = ShapeDetector()

# loop over the contours
for c in cnts:
	# compute the center of the contour, then detect the name of the
	# shape using only the contour
	M = cv.moments(c)
	cX = int((M["m10"] / M["m00"]) * ratio)
	cY = int((M["m01"] / M["m00"]) * ratio)
	shape = sd.detect(c)

	# multiply the contour (x, y)-coordinates by the resize ratio,
	# then draw the contours and the name of the shape on the image
	c = c.astype("float")
	c *= ratio
	c = c.astype("int")
	cv.drawContours(image, [c], -1, (0, 255, 0), 2)
	cv.putText(image, shape, (cX, cY), cv.FONT_HERSHEY_SIMPLEX,
		0.5, (0, 0, 255), 2)

	# show the output image
	cv.imshow("Image", image)
	cv.waitKey(0)
 

     