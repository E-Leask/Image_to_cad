# USAGE
# python detect_shapes.py --image gradient_basic.png

# import the necessary packages
#from six import u
#from pyimagesearch.shapedetector import ShapeDetector
import argparse
import imutils
import cv2 as cv
import numpy as np
from numpy import array
import openpyscad as ops
from scipy.spatial import distance
import math
import matplotlib.pyplot as plt
from shapedetector import *
from vertex import *
#from sklearn.datasets import make_blobs
#from sklearn.cluster import KMeans
#from sklearn.metrics import silhouette_score
#from sklearn.preprocessing import StandardScaler

from sympy import *



def show(image,name="Image"):
    image=image.astype(np.uint8)
    cv.imshow(name, image)

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

def check_same_point(point_a, point_b, max_dist):
    sq_dist=distance.sqeuclidean(point_a,point_b)
    if sq_dist > max_dist**2:
        return False
    return True

def check_point_intersects(point_c, line, max_dist):
    point_a=line[0]
    point_b=line[1]
    if np.all(point_c==point_a) or np.all(point_c==point_b):
        return False 
    #dist=(np.abs(np.cross(point_b-point_a, point_a-point_c)) / np.norm(point_b-point_a))
    if point_a[0]==point_b[0]:
        ab_line=[10000000,0]
    else:
        ab_line=[(point_a[1]-point_b[1])/(point_a[0]-point_b[0]),0]
    ab_line[1]=point_a[1]+(ab_line[0]*-point_a[0])
    if ab_line[0]==0:
        c_line=[0.000000001,0]
    else:
        c_line=[-1/ab_line[0],0]
    c_line[1]=point_c[1]+((c_line[0])*-point_c[0])
    
    
    x, y= symbols('x y')
    int_point,=linsolve([-y + ab_line[0]*x + ab_line[1],-y + c_line[0]*x + c_line[1]],(x,y))
    int_point=np.asarray(int_point, dtype=float)
    if int_point[0]<=point_a[0] and int_point[0]<=point_b[0] or int_point[1]<=point_a[1] and int_point[1]<=point_b[1]:
        return False
    
    dist=np.hypot((int_point[0]-point_c[0]),(int_point[1]-point_c[1]))
    if dist > max_dist:
        return False
    return True

def check_line_intersects(line1, line2, max_dist):
    point_a=line2[0]
    point_b=line2[1]
    point_c=line1[0]
    point_d=line1[1]
    
    #if there is a duplicate point the lines are connnected
    if np.all(point_c==point_a) or np.all(point_c==point_b):
        return None
    if np.all(point_d==point_a) or np.all(point_d==point_b):
        return None 
    #dist=(np.abs(np.cross(point_b-point_a, point_a-point_c)) / np.norm(point_b-point_a))
    
    #slope edge cases
    if point_a[0]==point_b[0]:
        ab_line=np.array([10000000,0],dtype=np.longlong)
    else:
        ab_line=np.array([(point_a[1]-point_b[1])/(point_a[0]-point_b[0]),0],dtype=np.longlong)
    ab_line[1]=point_a[1]+(ab_line[0]*-point_a[0])
    
    if point_c[0]==point_d[0]:
        cd_line=np.array([10000000,0],dtype=np.longlong)
    else:
        cd_line=np.array([(point_c[1]-point_d[1])/(point_c[0]-point_d[0]),0],dtype=np.longlong)
    cd_line[1]=point_c[1]+(cd_line[0]*-point_c[0])
    
    #Check if lines are parallel
    if np.all(ab_line[0]==cd_line[0]):
        return None 
    
    x, y= symbols('x y')
    int_point,=linsolve([-y + ab_line[0]*x + ab_line[1],-y + cd_line[0]*x + cd_line[1]],(x,y))
    int_point=np.asarray(int_point, dtype=np.intc)
    if int_point[0]<=point_a[0] and int_point[0]<=point_b[0] or int_point[1]<=point_a[1] and int_point[1]<=point_b[1]:
        return None
    
    dist=np.hypot((int_point[0]-point_c[0]),(int_point[1]-point_c[1]))
    if dist > max_dist:
        return None
    return int_point
#============================================================================================
#INPUT
#============================================================================================
# construct the argument parse and parse the arguments
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

#==========================================================================================
#FILTER
#==========================================================================================
# convert the resized image to grayscale, blur it slightly,
# and threshold it
fa = [[],[]]
gray = cv.cvtColor(resized, cv.COLOR_BGR2GRAY).astype(np.float32)
fa[0].append(gray)
fa[1].append("gray")

if True:
    i = 21
#for i in range(21,62, 20):
    gradient = cv.GaussianBlur(gray, (i , i), 0)

    gradRemove=cv.divide(gray,gradient)
    gradRemove = cv.normalize(gradRemove,None,0,255,cv.NORM_MINMAX)
    
    fa[0].append(gradRemove)
    fa[1].append("gradient remove")

    b=25
    bilat=cv.bilateralFilter(gradRemove, 9, b, b, cv.BORDER_DEFAULT)
    fa[0].append(bilat)
    fa[1].append("bilat")
    
    g=5
    blur = cv.GaussianBlur(bilat, (g , g), 0)
    fa[0].append(blur)
    fa[1].append("gauss")
    
    grad = np.gradient(blur)
    grad=np.hypot(grad[0],grad[1])
    grad = cv.normalize(grad,None,0,255,cv.NORM_MINMAX)
    fa[0].append(grad)
    fa[1].append("grad")
    filter=blur.astype(np.uint8)
    
    #blur=blur.astype(np.uint8)
    #equal = cv.equalizeHist(blur)
    #show( equal ,"equal" + str(i))
    #cv.waitKey(0)
    #filter=equal.astype(np.uint8)
    
    ret,thresh1 = cv.threshold(blur,127,255,cv.THRESH_BINARY)
    
for i in range(len(fa[0])):
    r=3
    #plt.subplot(int(np.ceil(len(fa[0])/r)),r,i+1),#plt.imshow(fa[0][i],'gray',vmin=0,vmax=255)
    #plt.title(fa[1][i])
    #plt.xticks([]),#plt.yticks([])
#plt.show()    
    

#=============================================================================
#Edge Detection
#=============================================================================
#Before locating edges with Canny Edge, need to determine a way to decide on threshold  values
#Lets try looking at the histogram #http://www.sci.utah.edu/~acoste/uou/Image/project1/Arthur_COSTE_Project_1_report.html
fa = [[],[]]
fa[0].append(gray)
fa[1].append("gray")
hist = cv.calcHist([filter],[0],None,[256],[0,256])

estwp=(np.where(grad<(55)))
estwp=len(estwp[0])

estbp=(np.where(grad>(200)))
estbp=len(estbp[0])

sumpixels=sum(hist)[0]
sumhist=0
midintensity=[]

for i in range(0,256,1):
    sumhist = sumhist + hist[i]
    if sumhist>estbp :
        midintensity.append(i)
        break
sumhist=0
for i in range(255,-1,-1):
    sumhist = sumhist + hist[i]
    if sumhist>estwp:
        midintensity.append(i)
        break
    
histnorm=np.divide(hist,sum(hist))
histnorm=histnorm[:,0]


x=list(range(0,256))
#plt.bar(x,histnorm)
#plt.plot(midintensity,np.array([histnorm[midintensity[0]], histnorm[midintensity[1]]]), 'o')
#plt.show()
np.column_stack((x, histnorm))

edges = cv.Canny(filter,midintensity[0],midintensity[1])
fa[0].append(edges)
fa[1].append("canny edge")
i=5
#cv.imshow("edges", edges)
#cv.waitKey(0)
closing = cv.morphologyEx(edges, cv.MORPH_CLOSE,  cv.getStructuringElement(cv.MORPH_RECT,(i,i)))
fa[0].append(closing)
fa[1].append("close morph" + str(i))
#cv.imshow("clsoing", closing)
#cv.waitKey(0)
#opening = cv.morphologyEx(closing, cv.MORPH_OPEN,  cv.getStructuringElement(cv.MORPH_RECT,(3,3)))
#fa[0].append(opening)
#fa[1].append("open morph" + str(3))
#cv.imshow("Image", opening)
#cv.waitKey(0)
#erosion = cv.erode(opening,cv.getStructuringElement(cv.MORPH_RECT,(3,3)),iterations = 1)
#fa[0].append(erosion)
#fa[1].append("erode morph")
#erosion  = cv.morphologyEx(erosion, cv.MORPH_OPEN,  cv.getStructuringElement(cv.MORPH_RECT,(3,3)))
thresh=closing
#for i in range(len(fa[0])):
#    r=3
    #plt.subplot(int(np.ceil(len(fa[0])/r)),r,i+1),#plt.imshow(fa[0][i],'gray',vmin=0,vmax=255)
    #plt.title(fa[1][i])
    #plt.xticks([]),#plt.yticks([])
#plt.show() 

#======================================================================================
#Find Contours
#======================================================================================
cnts, hierarchy = cv.findContours(thresh.copy(), cv.RETR_TREE,
	cv.CHAIN_APPROX_SIMPLE)
hierarchy=hierarchy[0]
#cnts = imutils.grab_contours(cnts)
sd = ShapeDetector()
# loop over the contours
print(type(cnts))
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
	#cv.drawContours(image, [c], -1, (0, 255, 0), 2)
	#cv.putText(image, shape, (cX, cY), cv.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 255), 2)

	# show the output image
	#cv.imshow("contours", image)
	#cv.waitKey(0)

print("test")
simplified_contours=[]
for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.1 * peri, True)
    simplified_contours.append(approx)
#Ok so I have all of these shapes that represent the shapes and will give information about inner data                       
#We need to simplify the data by estimating the polynomial.
#[Next, Previous, First_Child, Parent]
roots=[idx for idx, h in enumerate(hierarchy) if h[3]==-1]


#start with the outer most shape(we will deal with multiple shapes later)
#given the outer contour group of an object
#make our tree with how the hierarchy is connected
#[Next, Previous, First_Child, Parent]
#for r in roots:
r=roots[0]
children=[idx for idx, h in enumerate(hierarchy) if h[3]==r]
max_distance=10
#Working with simplified contours [contour_index, line_index, point_index]
#FIND CONNECTED POINTS
#for point1 in pointSet:

child_i=len(children)-1
while child_i > 0: #grab a child
    child=children[child_i]
    ch_l = len(child)-1
    i=child_i-1
    while i>=0: #grab other children
        compare=children[i]
        co_l= len(compare)-1
        while ch_l>=0:
            point1=child[ch_l]
            while co_l>=0:
                point2=compare[co_l] 
                if check_same_point(point1,point2,max_distance):
                    point2=point1
                    point1=[np.mean(point1[0],point2[0]),np.mean(point1[1],point2[1])]    
                co_l=co_l-1
            ch_l=ch_l-1
        i=i-1
    child_i=child_i-1

#POINTS AND LINES       
child_i=len(children)-1
i=child_i-1
while child_i > 0: #grab a child
    child=children[child_i]
    ch_l = len(child)-1
    i = len(children)-1
    while i>=0: #grab other children
        if i == child_i: #skip if referring to same child
            i=i-1
            continue
        compare=children[i]
        co_l= len(compare)-1
        while ch_l>=0:
            point1=child[ch_l]
            while co_l>=0:
                if check_point_intersects(point1,[compare[co_l],compare[(co_l-1)%len()]])
                co_l=co_l-1
            ch_l=ch_l-1
        i=i-1
    child_i=child_i-1
        
        
     
        

   
cp = cv.cvtColor(edge_detect, cv.COLOR_GRAY2BGR)
for i in range(0, len(lineSet)):
    li0 = lineSet[i,0]
    li1 = lineSet[i,1]
    cv.line(cp, (li0[0], li0[1]), (li1[0], li1[1]), (0,20*i,255), 2 , cv.LINE_AA)
plt.subplot(1,4,2)
plt.imshow(cp)

#FIND POINT INTERSECTING LINE  
i=len(pointSet)-1
while i >= 0: #Until I run out of points in the pointSet
    point1=pointSet[i]    
    #A point may potentially be intersecting with the line instead of endpoint    
    for idx,line in enumerate(lineSet):
        if check_point_intersects(point1,line,max_distance/2):
        #we break the line into line A and line B with endpoint pA<->p1 and p1<->pB
            div_line=np.asarray(line)
            l_index=np.argwhere(lineSet==div_line)
            line_a=[div_line[0],point1]
            line_b=[div_line[1],point1]
            #overwriting lines
            lineSet[idx]=line_a
            print (type(lineSet))
            lineSet=np.concatenate((lineSet,[line_b]))
            print(type(lineSet))
    i=i-1

elc = cv.cvtColor(edge_detect, cv.COLOR_GRAY2BGR)
for i in range(0, len(lineSet)):
    li0 = lineSet[i,0]
    li1 = lineSet[i,1]
    cv.line(elc, (li0[0], li0[1]), (li1[0], li1[1]), (20*i,0,255), 2 , cv.LINE_AA)
plt.subplot(1,4,3)
plt.imshow(elc)       





exit()
#======================================================================================
#Finding Lines
#======================================================================================

fa = [[],[]]
fa[0].append(gray)
fa[1].append("gray")

cdstP = cv.cvtColor(edge_detect, cv.COLOR_GRAY2BGR)
#cv.imshow("Source", cdstP)
j=1
if j:
#for j in range(1,52,10):
    linesP = cv.HoughLinesP(image=edge_detect, rho=j , theta=(np.pi / (180)/8), threshold=25 ,minLineLength=2, maxLineGap=5)
    cdstP = cv.cvtColor(edge_detect, cv.COLOR_GRAY2BGR)
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 2 , cv.LINE_AA)
    
    plt.subplot(1, 4, 1)
    plt.title(j)
    plt.imshow(cdstP)
    linesP=np.array(linesP)
    centers=[(linesP[:,0,2]+linesP[:,0,0])//2,(linesP[:,0,3]+linesP[:,0,1])//2]
    centers=np.swapaxes(centers, 0, 1)
    centers=np.squeeze(centers)
    ax1=plt.subplot(1, 4, 2)
    ax1.scatter(centers[:,0], centers[:,1],c='b')
    ax1.axis("equal")
    
    directions=[(linesP[:,0,2]-linesP[:,0,0]),(linesP[:,0,3]-linesP[:,0,1])]
    directions=np.swapaxes(directions, 0, 1)
    directions=np.squeeze(directions)
    plt.subplot(1, 4, 3)
    for i in range(0, len(centers)):
        plt.axis("equal")
        plt.quiver(centers[i,0], centers[i,1],directions[i,0], directions[i,1])
    plt.subplot(1, 4, 4)
    plt.scatter(linesP[:,0,0], linesP[:,0,1],c='r')
    plt.scatter(linesP[:,0,2], linesP[:,0,3],c='r')
    plt.axis("equal")
    plt.show()

    magnitudes=np.hypot(directions[:,0],directions[:,1])
    mi=np.flip(np.argsort(magnitudes)) #magnitude index from  largest to smallest 
    #group lines by direction of lines
    linesH=np.squeeze(linesP)
    lineSet=[]
    #linesF[index,0=lines,1=center x,y , 2=direction x,y]
    while len(mi)>0:
        lli=mi[0]#largestLineIndex
        groupLinesIndexes=[lli]
        mi=np.delete(mi,0,0)
        #r_linesH=np.delete(linesH,lli,0)
        #r_centers=np.delete(centers,lli,0)
        #r_directions=np.delete(directions,lli,0)
        r_linesH=linesH
        r_centers=centers
        r_directions=directions
        l = lli
        #lets go through all the lines
        i = len(r_linesH)-1
        for i in range(len(r_linesH)-1,-1,-1):
        #while i>0:
            line=r_linesH[i]
            center=r_centers[i]
            direction=r_directions[i]
            

            
            #lets compare the angle between the lines
            x=np.dot(direction,np.squeeze(directions[l]))
            y=np.dot(np.hypot(direction[0],direction[1]), np.hypot(directions[l,0],directions[l,1]))
            z=x/y
            ang=np.squeeze(np.arccos(z)*180/np.pi)

            if ang>90:
                ang=abs(180-ang)
            print("ang: " + str(ang))
            #print("angle: " + str(ang))
            if ang<15: #potentially part of same line 
                #TODO: Check for path between points in direction of gradient instead of this
            #distance between a point and line
            #c=yb-xa    
                c=linesH[l,1]*directions[l,0] - linesH[l,0]*(directions[l,1])
                eq_top=abs(directions[l,1]*center[0]-directions[l,0]*center[1]+c)#Am + Bn + C
                mag=np.hypot(directions[l,0],directions[l,1])
                d=eq_top/mag #this isn't reliable because as the point is farther away this line could be incorrect
                if d < 20: #This is classfied as a potential match
                    groupLinesIndexes.append(i)
                    #r_linesH=np.delete(r_linesH,i,0)
                    #r_centers=np.delete(r_centers,i,0)
                    #r_directions=np.delete(r_directions,i,0)
                    mi=mi[mi != i] 
            
            #Now that i have the group of lines I need to make the largest line             
            i=i-1    
        lineGroup=[r_linesH[index] for index in groupLinesIndexes]
        cdstP = cv.cvtColor(edge_detect, cv.COLOR_GRAY2BGR)
#        for i in range(0, len(lineGroup)):
#            li = lineGroup[i]
#            cv.line(cdstP, (li[0], li[1]), (li[2], li[3]), (0,0,255), 2 , cv.LINE_AA)
#            plt.imshow(cdstP)
#        plt.show()
        coordGroup=np.reshape(lineGroup,(-1, 2))
        maxCoords=[coordGroup[0],coordGroup[1]]
        coordGroup=np.delete(coordGroup,[0,1],0)
        for i in range(len(coordGroup)-1,-1,-1):
            #consider points a,b,c
            a=maxCoords[0]
            b=maxCoords[1]
            c=coordGroup[i]
            max_sol=np.argmax([distance.sqeuclidean(a,b),distance.sqeuclidean(a,c),distance.sqeuclidean(b,c)])
            #if a<->b is max then we update nothing 
            if max_sol==1: #a<->c is max so replace b with c in maxCoords
                maxCoords[1]=coordGroup[i]
            elif max_sol==2:#b<->c is max so replace a with c in maxCoords
                maxCoords[0]=coordGroup[i]
        lineSet.append(maxCoords)
        cdstP = cv.cvtColor(edge_detect, cv.COLOR_GRAY2BGR)
#        cv.line(cdstP, (maxCoords[0][0], maxCoords[0][1]), (maxCoords[1][0], maxCoords[1][1]), (0,255,0), 2 , cv.LINE_AA)
#        plt.imshow(cdstP)
#        plt.show()

#======================================================================================
#Connecting Points/Lines
#======================================================================================
#List of lines
lineSet=np.array(lineSet)
#List of points
pointSet=np.reshape(lineSet,(-1,2))
pointSet = np.unique(pointSet, axis=0)
max_distance=10
bfcp = cv.cvtColor(edge_detect, cv.COLOR_GRAY2BGR)
for i in range(0, len(lineSet)):
    li0 = lineSet[i,0]
    li1 = lineSet[i,1]
    cv.line(bfcp, (li0[0], li0[1]), (li1[0], li1[1]), (0,i*20,255), 2 , cv.LINE_AA)
plt.subplot(1,4,1)
plt.imshow(bfcp)


#FIND CONNECTED POINTS
#for point1 in pointSet:
i=len(pointSet)-1
while i >= 0: #Until I run out of points in the pointSet
    point1=pointSet[i]    
    j=len(pointSet)-1
    while j >= 0: #Until there are no other points
        point2=pointSet[j] 
        #matching_point=point1 == point2
        #if matching_point.all():
        if i == j: #skip points if they have the same index
            j=j-1
            continue
        
        #if two points are close enough to be a single point we return true
        if check_same_point(point1,point2,max_distance):
            #lets replace each instances of point2 with point1 
            #POTENTIAL BUGS
            for idx, line in enumerate(lineSet):
                for idy, point in enumerate(line):
                    matching_point= point2 == point
                    if matching_point.all():
                        lineSet[idx,idy]=point1
                        pointSet = np.delete(pointSet,j,0)
            j=j-1
            i=i-1      
        j=j-1
    i=i-1
    
cp = cv.cvtColor(edge_detect, cv.COLOR_GRAY2BGR)
for i in range(0, len(lineSet)):
    li0 = lineSet[i,0]
    li1 = lineSet[i,1]
    cv.line(cp, (li0[0], li0[1]), (li1[0], li1[1]), (0,20*i,255), 2 , cv.LINE_AA)
plt.subplot(1,4,2)
plt.imshow(cp)
 
#FIND POINT INTERSECTING LINE  
i=len(pointSet)-1
while i >= 0: #Until I run out of points in the pointSet
    point1=pointSet[i]    
    #A point may potentially be intersecting with the line instead of endpoint    
    for idx,line in enumerate(lineSet):
        if check_point_intersects(point1,line,max_distance/2):
        #we break the line into line A and line B with endpoint pA<->p1 and p1<->pB
            div_line=np.asarray(line)
            l_index=np.argwhere(lineSet==div_line)
            line_a=[div_line[0],point1]
            line_b=[div_line[1],point1]
            #overwriting lines
            lineSet[idx]=line_a
            print (type(lineSet))
            lineSet=np.concatenate((lineSet,[line_b]))
            print(type(lineSet))
    i=i-1

elc = cv.cvtColor(edge_detect, cv.COLOR_GRAY2BGR)
for i in range(0, len(lineSet)):
    li0 = lineSet[i,0]
    li1 = lineSet[i,1]
    cv.line(elc, (li0[0], li0[1]), (li1[0], li1[1]), (20*i,0,255), 2 , cv.LINE_AA)
plt.subplot(1,4,3)
plt.imshow(elc) 

#FIND LINE INTERSECTING LINE
i=len(lineSet)-1
while i >= 0: #Until I run out of points in the pointSet
    line1=lineSet[i]    
    #A point may potentially be intersecting with the line instead of endpoint    
    for idx,line in enumerate(lineSet):
        int_point = check_line_intersects(line1,line,max_distance/2)
        if np.all(int_point !=None):
        #we break the line into line A and line B with endpoint pA<->p1 and p1<->pB
            div_line=np.asarray(line)
            l_index=np.argwhere(lineSet==div_line)
            line_a=np.array([div_line[0],int_point])
            line_b=np.array([div_line[1],int_point])
            line_c=np.array([line1[0],int_point])
            line_d=np.array([line1[1],int_point])
            
            #overwriting lines
            lineSet[idx]=line_a
            lineSet[i]=line_c
            line1=line_c
            print (type(lineSet))
            print(lineSet.dtype)
            print(line_b.dtype)
            lineSet=np.concatenate((lineSet,[line_b]))
            lineSet=np.concatenate((lineSet,[line_c]))
            print(type(lineSet))
    i=i-1
print("Final;ized line set \n")
print(lineSet)

cil = cv.cvtColor(edge_detect, cv.COLOR_GRAY2BGR)
for i in range(0, len(lineSet)):
    li0 = lineSet[i,0]
    li1 = lineSet[i,1]
    cv.line(cil, (li0[0], li0[1]), (li1[0], li1[1]), (0,40*i,255), 2 , cv.LINE_AA)
plt.subplot(1,4,4)
plt.imshow(cil) 
plt.show()
cil = cv.cvtColor(edge_detect, cv.COLOR_GRAY2BGR)
for i in range(0, len(lineSet)):
    li0 = lineSet[i,0]
    li1 = lineSet[i,1]
    cv.line(cil, (li0[0], li0[1]), (li1[0], li1[1]), (0,40*i,255), 2 , cv.LINE_AA)       
    plt.imshow(cil) 
    plt.show()
exit()


#======================================================================================
#Create CAD
#======================================================================================


hierarchy=cnts[1]
#[Next, Previous, First_Child, Parent]
cnts = imutils.grab_contours(cnts)
sd = ShapeDetector()
cnts, hierarchy=clean_contours(hierarchy)

roots,tree = get_tree(hierarchy)

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
        
    
    
  
 

     