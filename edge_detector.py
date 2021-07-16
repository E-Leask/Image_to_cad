#=============================================================================
#Edge Detection
#=============================================================================
#Before locating edges with Canny Edge, need to determine a way to decide on threshold  values
#Lets try looking at the histogram #http://www.sci.utah.edu/~acoste/uou/Image/project1/Arthur_COSTE_Project_1_report.html
def edge_detect(gray, filter, grad):
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
    closing = cv.morphologyEx(edges, cv.MORPH_CLOSE,  cv.getStructuringElement(cv.MORPH_RECT,(i,i)))
    fa[0].append(closing)
    fa[1].append("close morph" + str(i))
    opening = cv.morphologyEx(closing, cv.MORPH_OPEN,  cv.getStructuringElement(cv.MORPH_RECT,(3,3)))
    fa[0].append(opening)
    fa[1].append("open morph" + str(3))
    erosion = cv.erode(opening,cv.getStructuringElement(cv.MORPH_RECT,(3,3)),iterations = 1)
    fa[0].append(erosion)
    fa[1].append("erode morph")
    #erosion  = cv.morphologyEx(erosion, cv.MORPH_OPEN,  cv.getStructuringElement(cv.MORPH_RECT,(3,3)))

    for i in range(len(fa[0])):
        r=3
        #plt.subplot(int(np.ceil(len(fa[0])/r)),r,i+1),#plt.imshow(fa[0][i],'gray',vmin=0,vmax=255)
        #plt.title(fa[1][i])
        #plt.xticks([]),#plt.yticks([])
    #plt.show() 
    return erosion