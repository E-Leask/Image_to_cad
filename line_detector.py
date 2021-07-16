from connect_points import *
#======================================================================================
#Finding Lines
#======================================================================================
def line_detect(gray,erosion):
    fa = [[],[]]
    fa[0].append(gray)
    fa[1].append("gray")

    cdstP = cv.cvtColor(erosion, cv.COLOR_GRAY2BGR)
    #cv.imshow("Source", cdstP)
    j=1
    if j:
    #for j in range(1,52,10):
        linesP = cv.HoughLinesP(image=erosion, rho=j , theta=(np.pi / (180)/8), threshold=25 ,minLineLength=2, maxLineGap=5)
        cdstP = cv.cvtColor(erosion, cv.COLOR_GRAY2BGR)
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
            cdstP = cv.cvtColor(erosion, cv.COLOR_GRAY2BGR)
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
            cdstP = cv.cvtColor(erosion, cv.COLOR_GRAY2BGR)
    #        cv.line(cdstP, (maxCoords[0][0], maxCoords[0][1]), (maxCoords[1][0], maxCoords[1][1]), (0,255,0), 2 , cv.LINE_AA)
    #        plt.imshow(cdstP)
    #        plt.show()