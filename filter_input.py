#==========================================================================================
#FILTER
#==========================================================================================
# convert the resized image to grayscale, blur it slightly,
# and threshold it
def filter_input(resized):
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
    return gray,filter,thresh1,grad   