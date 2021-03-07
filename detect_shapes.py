# USAGE
# python detect_shapes.py --image shapes_and_colors.png

# import the necessary packages
from six import u
from pyimagesearch.shapedetector import ShapeDetector
import argparse
import imutils
import cv2 as cv
from numpy import array
import openpyscad as ops
def show(image):
    cv.imshow("Image", image)

def get_tree(hier):
    tree={}
    roots=[]
    #Find all parents
    for h in range(0,len(hier)): #we have a parent
        if hier[h][2] != -1:
            tree[hier[h][2]]=[]
        if hier[h][3] == -1: #We have a root
            roots.append(h)
            
    for h in range(0,len(hier)):
        if hier[h][2] != -1:#Then we have a child node
            tree[hier[h][2]].append(h)
    return roots , tree

def clean_contours(hierarchy):
#find largest tree, throw out rest
    updateHierarchy = []
    updateCnts = []
    a = array(hierarchy)
    print (a.shape)
    for h in range(0,len(hierarchy[0])):
        if hierarchy[0][h][2] != -1 or hierarchy[0][h][3] != -1:
            updateHierarchy.append(hierarchy[0][h])
            updateCnts.append(cnts[h])
            cnts[h] = cnts[h].astype("float")
            cnts[h] *= ratio
            cnts[h] = cnts[h].astype("int")
            cv.drawContours(image, [cnts[h]], -1, (0, 255, 0), 2)
            updateCnts.append(cnts[h])
        else:
            cnts[h] = cnts[h].astype("float")
            cnts[h] *= ratio
            cnts[h] = cnts[h].astype("int")
            cv.drawContours(image, [cnts[h]], -1, (0, 0, 255), 2)
        cv.imshow("Image", image)
        
    cv.waitKey(0)
    return updateCnts , updateHierarchy

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
    help="path to the input image")
args = vars(ap.parse_args())

# load the image and resize it to a smaller factor so that
# the shapes can be approximated better
image = cv.imread(args["image"])
image=cv.flip(image, 0)
resized = imutils.resize(image, width=300)
ratio = image.shape[0] / float(resized.shape[0])

# convert the resized image to grayscale, blur it slightly,
# and threshold it
gray = cv.cvtColor(resized, cv.COLOR_BGR2GRAY)
show(gray)
cv.waitKey(0)
blurred = cv.GaussianBlur(gray, (11 , 11), 0)
show(blurred)
cv.waitKey(0)

thresh = cv.adaptiveThreshold(blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV,21,2)

show(thresh)
cv.waitKey(0)
# find contours in the thresholded image and initialize the
# shape detector
cnts = cv.findContours(thresh.copy(), cv.RETR_TREE	,
    cv.CHAIN_APPROX_SIMPLE)

hierarchy=cnts[1]
#[Next, Previous, First_Child, Parent]
cnts = imutils.grab_contours(cnts)
sd = ShapeDetector()
cnts, hierarchy=clean_contours(hierarchy)

roots,tree = get_tree(hierarchy)
#
#parents=tree.keys()
#duplicates=[]
#for p in parents:
#	if len(tree.get(p)) == 1: #We have found a potnetial duplicate
#		print(tree.get(p))
#		c=cnts[tree.get(p)[0]]
#		peri = cv.arcLength(c, True)
#		approx = cv.approxPolyDP(c, 0.04 * peri, True)
#		(x1, y1, w1, h1) = cv.boundingRect(approx)
#		
#		c=cnts[p]
#		peri = cv.arcLength(c, True)
#		approx = cv.approxPolyDP(c, 0.04 * peri, True)
#		(x2, y2, w2, h2) = cv.boundingRect(approx)
#		#(x, y, w, h)=abs(x2/x1-1), abs(y2/y1-1), abs(w2/w1-1), abs(h2/h1-1) 
#		(x, y, w, h)=abs(x2-x1), abs(y2-y1), abs(w2-w1), abs(h2-h1) 
#		#if w<=0.2 and y<=0.2: #we found a duplicate
#		if w<=10 and h<=10: #we found a duplicate
#				#list child for deletion
#			duplicates.append(h)
#			print("found!")
#duplicates.sort(reverse=True)	
#for d in duplicates:
#	del cnts[d] 
#	
#
cv.destroyAllWindows() 
#image = cv.imread(args["image"])     
# loop over the contours
cnts=cnts[::4]
shapes=[]
cad=ops.Difference()
for c in cnts:
    # compute the center of the contour, then detect the name of the
    # shape using only the contour

    M = cv.moments(c)
    if M["m00"] == 0:
        continue
    cX = int((M["m10"] / (M["m00"]+0.00001)) * ratio)
    cY = int((M["m01"] / (M["m00"]+0.00001)) * ratio)
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
    #Lets add it to the cad file
    peri = cv.arcLength(c, True)
    approx = cv.approxPolyDP(c, 0.04 * peri, True)
    (x, y, w, h) = cv.boundingRect(approx)
    shapes.append([shape,cX,cY,w,h])
    
    if shape == "rectangle" or shape == "square":
        cad.append(ops.Cube([w, h, 10]).translate([x,y,0]))
    else:
        cad.append(ops.Cylinder(h=20,d=w).translate([x+w/2,y+h/2,0]))
cad.write("sample.scad")
        
    
    
  
 

     