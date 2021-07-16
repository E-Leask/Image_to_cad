import argparse
import imutils
import cv2 as cv
#============================================================================================
#INPUT
#============================================================================================
# construct the argument parse and parse the arguments
def input():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True,
        help="path to the input image")
    args = vars(ap.parse_args())

    # load the image and resize it to a smaller factor so that
    # the shapes can be approximated better
    image = cv.imread(args["image"])
    image=cv.flip(image, 0) #why flip????
    resized = imutils.resize(image, width=400)
    ratio = image.shape[0] / float(resized.shape[0])
    return ratio