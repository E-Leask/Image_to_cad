def check_same_point(point_a, point_b, max_dist):
    sq_dist=distance.sqeuclidean(point_a,point_b)
    if sq_dist > max_dist**2:
        return False
    return True

def check_intersects(point_c, line, max_dist):
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
#======================================================================================
#Connecting Points
#======================================================================================
#List of lines
lineSet=np.array(lineSet)
#List of points
pointSet=np.reshape(lineSet,(-1,2))
pointSet = np.unique(pointSet, axis=0)
max_distance=10
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
    
i=len(pointSet)-1
while i >= 0: #Until I run out of points in the pointSet
    point1=pointSet[i]    
    #A point may potentially be intersecting with the line instead of endpoint    
    for idx,line in enumerate(lineSet):
        if check_intersects(point1,line,max_distance/2):
        #we break the line into line A and line B with endpoint pA<->p1 and p1<->pB
            div_line=np.asarray(line)
            l_index=np.argwhere(lineSet==div_line)
            line_a=[div_line[0],point1]
            line_b=[div_line[1],point1]
            lineSet[idx]=line_a
            print (type(lineSet))
            lineSet=np.append(lineSet,line_b)
            print(type(lineSet))
    i=i-1

print("Final;ized line set \n")
print(lineSet)

cdstP = cv.cvtColor(erosion, cv.COLOR_GRAY2BGR)
for i in range(0, len(lineSet)):
    li0 = lineSet[i,0]
    li1 = lineSet[i,1]
    cv.line(cdstP, (li0[0], li0[1]), (li1[0], li1[1]), (0,0,255), 2 , cv.LINE_AA)
    plt.imshow(cdstP)
plt.show()            
  
exit()
