#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" @author: JGJ """

import numpy as np
import matplotlib.pyplot as plt
import cv2

############### MODULE LOCATION ###############################################
print(np.__file__)
print(plt.__file__)
print(cv2.__file__)

############### WHOLE FUNCTIONS IN A LIBRARY ##################################
dir(np)
help(np)
np.__all__  # OR
import inspect
all_functions = inspect.getmembers(np, inspect.isfunction)
print(all_functions)     # OR
np.__dict__     #OR
np.__dict__.keys()  #OR
np.__builtins__     #OR
for i in dir(np):
    print(i)
    
############### GET HELP FOR A FUNCTION IN A LIBRARY ##########################
help(cv2.THRESH_BINARY)
help(np.abs)
help(cv2.threshold)
help(cv2.Scharr)

############### LOCATION OF THE LIBRARY #######################################
cv2.__file__
np.__file__

############### GET FLAGS IN LIBABRY STARTING WITH ############################
# Get thresholding commands starting with color
flags = [i for i in dir(cv2) if i.startswith('COLOR_')]
print(flags)
# Get commands starting with imread
flags=[i for i in dir(cv2) if i.startswith('imread')]
print(flags)
#get commands starting with THRESH
flags=[i for i in dir(cv2) if i.startswith('THRESH')]
print(flags)
#numpy library ex
flags=[i for i in dir(np) if i.startswith('array')]
print(flags)
flags=[i for i in dir(np) if i.startswith('a')]
print(flags)
#matplotlib ex
flags=[i for i in dir(plt) if i.startswith('s')]
print(flags)

############### IMAGE THRESHOLDING ############################################
#Gray conversion use: cv2.COLOR_BGR2GRAY. for HSV: cv2.COLOR_BGR2HSV
img=cv2.imread('img0000.jpg',0)
plt.subplot(221),plt.imshow(img),plt.xticks([]),plt.yticks([]),plt.title('HSV')
plt.subplot(222),plt.imshow(img,cmap='gray'),plt.xticks([]),plt.yticks([]),plt.title('Gray')
ret,thresh1=cv2.threshold(img,100,255,cv2.THRESH_BINARY)
plt.subplot(223),plt.imshow(thresh1,cmap='gray'),plt.title('Binary'),plt.xticks([]),plt.yticks([])
cv2.imshow('Original',img)
cv2.imshow('Binary',thresh1)
cv2.waitKey(0)
cv2.destroyAllWindows()
#Different thresholding techniques and putting in a loop
img=cv2.imread('img0000.jpg',0)
ret,thresh1=cv2.threshold(img,100,255,cv2.THRESH_BINARY)
ret,thresh2=cv2.threshold(img,100,255,cv2.THRESH_BINARY_INV)
ret,thresh3=cv2.threshold(img,100,255,cv2.THRESH_MASK)
ret,thresh4=cv2.threshold(img,100,255,cv2.THRESH_TOZERO)
ret,thresh5=cv2.threshold(img,100,255,cv2.THRESH_TOZERO_INV)
ret,thresh6=cv2.threshold(img,100,255,cv2.THRESH_OTSU)
titles=['Original','Binary','Inverse_binary','Mask','Tozero','Inverse_tozero','Otsu']
images=[img,thresh1,thresh2,thresh3,thresh4,thresh5,thresh6]
for i in range(7):
    plt.subplot(3,4,i+1),plt.imshow(images[i],'gray')
    plt.xticks([]),plt.yticks([])
    plt.title(titles[i])     #plt.title(titles[i]
    plt.show()
#Adaptive and gaussian thresholding
img1=cv2.imread('img0000.jpg',0)
img=cv2.medianBlur(img1,6)
#plt.subplot(221),plt.imshow(img1,cmap='gray'),plt.title('Original')    
#plt.subplot(222),plt.imshow(img,cmap='gray'),plt.title('Median_blur')
ret,th1=cv2.threshold(img,100,255,cv2.THRESH_BINARY)
th2=cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
th3=cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
titles=['Original','Binary','Adaptive_mean','Adaptive_gaussian']
images=[img,th1,th2,th3]
for i in range(4):
    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i]),plt.xticks([]),plt.yticks([])
    plt.show()
#Otsu and gaussian thesholding for images with noise
img=cv2.imread('img0000.jpg',0)
# global thresholding
ret1,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
# Otsu's thresholding
ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# Otsu's thresholding after Gaussian filtering
blur=cv2.GaussianBlur(img,(5,5),0)
ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# plot all the images and their histograms
images = [img, 0, th1,
          img, 0, th2,
          blur, 0, th3]
titles = ['Original Noisy Image','Histogram','Global Thresholding (v=127)',
'Original Noisy Image','Histogram',"Otsu's Thresholding",
'Gaussian filtered Image','Histogram',"Otsu's Thresholding"]
for i in range(3):
    plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
    plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)
    plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
    plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
    plt.show()    
        
############### MORPHOLOGICAL OPERATIONS ######################################
img = cv2.imread('img0000.jpg',0)
#plt.imshow(img)    #kinda HSV 
plt.imshow(img,cmap='gray'),     #Gray scale image
kernel=np.ones((5,5),np.uint8)
#Erosion
erosion=cv2.erode(img,kernel,iterations=1)
plt.subplot(321),plt.imshow(erosion,cmap='gray'),plt.title('Erosion'),plt.xticks([]), plt.yticks([])
#Dilation
dilation=cv2.dilate(img,kernel,iterations=1)    
plt.subplot(322),plt.imshow(dilation,cmap='gray'),plt.title('Dilation'),plt.xticks([]), plt.yticks([])
#opening
opening=cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)
plt.subplot(323),plt.imshow(opening,cmap='gray'),plt.title('Opening'),plt.xticks([]), plt.yticks([])
#closing
closing=cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel)
plt.subplot(324),plt.imshow(closing,cmap='gray'),plt.title('Closing'),plt.xticks([]), plt.yticks([])
#tophat
tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
plt.subplot(325),plt.imshow(tophat,cmap='gray'),plt.title('Tophat'),plt.xticks([]), plt.yticks([])
#blackhat
blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
plt.subplot(326),plt.imshow(blackhat,cmap='gray'),plt.title('Blackhat'),plt.xticks([]), plt.yticks([])
#Laplacian, Sobel, Scharr etc..
#Laplacian
laplacian=cv2.Laplacian(img,cv2.CV_64F)
#SobelX & SobelY
sobelx=cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
sobely=cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
#Scharr
#scharr=cv2.Sc
#Canny edge
canny=cv2.Canny(img,100,200)
cv2.imshow('Laplacian',laplacian)
cv2.imshow('Sobel_X',sobelx)
cv2.imshow('Sobel_Y',sobely)
cv2.imshow('Canny',canny)
cv2.waitKey(0)
cv2.destroyAllWindows

############### CONTOURS ######################################################
im=cv2.imread("img0000.jpg",cv2.IMREAD_COLOR)
imgray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
ret,thresh=cv2.threshold(imgray,100,255,cv2.THRESH_BINARY)
#contours,hierarchy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) #first one is source image, second is contour retrieval mode
_, contours, _= cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
_, contours, _= cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
for data in contours:
    print("The contours have this data %r"%data)
cv2.drawContours(im,contours,-1,(0,255,0),3)
cv2.imshow('Output',im)
cv2.waitKey(0)
cv2.destroyAllWindows()
#while True:
#    if cv2.waitKey(6) & 0XFF==27:
#        break
#To draw all the contours in an image:
cv2.drawContours(im, contours, -1, (0,255,0), 3)
#To draw an individual contour, say 4th contour:
cv2.drawContours(im, contours, 3, (0,255,0), 3)
#But most of the time, below method will be useful:
cnt = contours[4]
cv2.drawContours(im, [cnt], 0, (0,255,0), 3)    
#MOMENTS: calculate centre of mass or area of the object etc..
img=cv2.imread('img0000.jpg',0)
#cv2.imshow('Original',img)
ret,thresh=cv2.threshold(img,100,255,0)
#cv2.imshow('Threshold',thresh)
_, contours, _= cv2.findContours(thresh,1,2)
cnt=contours[0]
m=cv2.moments(cnt)
print(m)
#Find centroid and area from this moment
cx=int(m['m10']/m['m00'])
cy=int(m['m01']/m['m00'])
print(cx,'\t', cy)  #print(cx,'\n', cy)
#AREA
ar=cv2.contourArea(cnt)
print(ar)
#CONTOUR PERIMETER or ARC-LENTH
arclen=cv2.arcLength(cnt,True)  #arclen=cv2.arcLength(cnt,False)
print(arclen)
#CONTOUR APPROXIMATION: using Douglas Peucker algo.
#epsilon: maximum distance from contour to approximated contour. an accuracy parameter & wise selection needed.
epsilon=0.1*cv2.arcLength(cnt,True)
approx=cv2.approxPolyDP(cnt,epsilon,True)
print(epsilon, '\n', approx)
#CONVEX HULL: hull = cv2.convexHull(points[, hull[, clockwise[, returnPoints]]
hull=cv2.convexHull(cnt) #to find convexity defects, you need to pass returnPoints = False.
print(hull)
#CONVEXITY chek
conv=cv2.isContourConvex(cnt)
print(conv)
#BOUNDING RECTANGLES: Straight bounding and rotated bounding
#Straight bounding
x,y,w,h=cv2.boundingRect(cnt)
img_rect=cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),4)
#Rotated rectangle
rect = cv2.minAreaRect(cnt)
box = cv2.boxPoints(rect)
box = np.int0(box)
img_rotate = cv2.drawContours(img,[box],0,(0,0,255),2)
#MINIMUM ENCLOSING CIRCLE: circle which completely covers the object with minimum area.
(x,y),radius = cv2.minEnclosingCircle(cnt)
center = (int(x),int(y))
radius = int(radius)
img = cv2.circle(img,center,radius,(0,255,0),2)
#ELLIPSE FITTING: returns the rotated rectangle in which the ellipse is inscribed.
ellipse = cv2.fitEllipse(cnt)
im = cv2.ellipse(im,ellipse,(0,255,0),2)
#LINE FITTING: fit a line to a set of points.
rows,cols = img.shape[:2]
[vx,vy,x,y] = cv2.fitLine(cnt, cv2.DIST_L2,0,0.01,0.01)
lefty = int((-x*vy/vx) + y)
righty = int(((cols-x)*vy/vx)+y)
img = cv2.line(img,(cols-1,righty),(0,lefty),(0,255,0),2)

plt.subplot(221),plt.imshow(img,'gray'),plt.title('Original')
plt.subplot(222),plt.imshow(thresh,'gray'),plt.title('Threshold')
#cx:'CentroidX'.cy:CentroidY',ar:'Contour_area',arclen:'Contour_perimeter', epsilon:'Contour_approx',approx:'Contour_approx',hull:'Convex_hull'

############### CONTOUR PROPERTIES ############################################
#ASPECT RATIO: width/height
x,y,w,h=cv2.boundingRect(cnt)
a_r=float(w)/h
#EXTENT: object area/bounding rectangel area
area=cv2.contourArea(cnt)
x,y,w,h=cv2.boundingRect(cnt)
rect_area=w*h
extent=float(area)/rect_area
#SOLIDITY: contour area/convex hull area
area=cv2.contourArea(cnt)
hull=cv2.convexHull(cnt)
hull_area=cv2.contourArea(hull)
solidity=float(area)/hull_area
#EQUIVALENT DIAMETER: diameter of the circle whose area is same as the contour area
#squareroot of(4*contour area/pi)
area=cv2.contourArea(cnt)
equi_diameter=np.sqrt(4*area/np.pi)
#ORIENTATION: the angle at which object is directed. also gives the Major Axis and Minor Axislengths.
(x,y),(MA,ma),angle=cv2.fitEllipse(cnt) #ERROR
#Orientation: the angle at which object is directed.  also gives the Major Axis and Minor Axislengths.
(x,y),(MA,ma),angle = cv2.fitEllipse(cnt)
#MASK & PIXEL POINTS: may need all the points which comprises that object. It's done by:
#using numpy function
mask = np.zeros(imgray.shape,np.uint8)
cv2.drawContours(mask,[cnt],0,255,-1)
pixelpoints = np.transpose(np.nonzero(mask))
#pixelpoints = cv2.findNonZero(mask)
#MAXIMUM & MINIMUM VALUES: using a mask image.
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(imgray,mask = mask)
#MEAN COLOR or INTENSITY: average color of an object. Or average intensity of the object in grayscale mode.
#EXTREME POINTS: topmost, bottommost, rightmost and leftmost points of the object.
leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])
#REGIONPROPS
#More contour functions
hull = cv2.convexHull(cnt,returnPoints = False)
defects = cv2.convexityDefects(cnt,hull)

############### 
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('img0000.jpg')
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(img_gray, 127, 255,0)
_,contours,hierarchy = cv2.findContours(thresh,2,1)
cnt = contours[0]
hull = cv2.convexHull(cnt,returnPoints = False)
defects = cv2.convexityDefects(cnt,hull)
for i in range(defects.shape[0]):
    s,e,f,d = defects[i,0]
    start = tuple(cnt[s][0])
    end = tuple(cnt[e][0])
    far = tuple(cnt[f][0])
    cv2.line(img,start,end,[0,255,0],2)
    cv2.circle(img,far,5,[0,0,255],-1)
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
#POINT POLYGON TEST : shortest distance between a point in the image and a contour
dist = cv2.pointPolygonTest(cnt,(50,50),True) 
#HU MOMENTS : 
img1 = cv2.imread('img0000.jpg',0)
img2 = cv2.imread('img0001.jpg',0)
ret, thresh = cv2.threshold(img1, 127, 255,0)
ret, thresh2 = cv2.threshold(img2, 127, 255,0)
_,contours,hierarchy = cv2.findContours(thresh,2,1)
cnt1 = contours[0]
_,contours,hierarchy = cv2.findContours(thresh2,2,1)
cnt2 = contours[0]
ret = cv2.matchShapes(cnt1,cnt2,1,0.0)
print(ret)

############### HISTOGRAM OPERATIONS ##########################################
img=cv2.imread('img0000.jpg',0)
hist=cv2.calcHist([img],[0],None,[256],[0,256])   # histogram using OpenCV
hist,bins=np.histogram(img.ravel(),256,[0,256])     # histogram using numpy
hist=np.bincount(img.ravel(),minlength=256)         # np.bincount is faster than np.histogram
# PLotting histogram
img = cv2.imread('img0000.jpg',0)
plt.hist(img.ravel(),256,[0,256]); plt.show()
#Get the boundary of histogram
img = cv2.imread('img0000.jpg')
color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([img],[i],None,[256],[0,256])
plt.plot(histr,color = col)
plt.xlim([0,256])
plt.show()
# Mask application
img=cv2.imread('img0000.jpg',0)
mask=np.zeros(img.shape[:2],np.uint8)
mask[80:800, 80:800]=255
masked_img=cv2.bitwise_and(img,img,mask=mask)
hist_full = cv2.calcHist([img],[0],None,[256],[0,256])
hist_mask = cv2.calcHist([img],[0],mask,[256],[0,256])
plt.subplot(221), plt.imshow(img, 'gray')
plt.subplot(222), plt.imshow(mask,'gray')
plt.subplot(223), plt.imshow(masked_img, 'gray')
plt.subplot(224), plt.plot(hist_full), plt.plot(hist_mask)
plt.xlim([0,256])
plt.show()

############### FEATURE DETECTION & DESCRIPTION ###############################
filename = 'img0000.jpg'
img = cv2.imread(filename)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)
dst = cv2.cornerHarris(gray,2,3,0.04)
#result is dilated for marking the corners, not important
dst = cv2.dilate(dst,None)
# Threshold for an optimal value, it may vary depending on the image.
img[dst>0.01*dst.max()]=[0,0,255]
cv2.imshow('dst',img)
if cv2.waitKey(0) & 0xff == 27: 
    cv2.destroyAllWindows()
# find Harris corners
gray = np.float32(gray)
dst = cv2.cornerHarris(gray,2,3,0.04)
dst = cv2.dilate(dst,None)
ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
dst = np.uint8(dst)
# find centroids
ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
# define the criteria to stop and refine the corners
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)
# Now draw them
res = np.hstack((centroids,corners))
res = np.int0(res)
img[res[:,1],res[:,0]]=[0,0,255]
img[res[:,3],res[:,2]] = [0,255,0]
cv2.imwrite('subpixel5.png',img)    
#Shi-Tomasi Corner Detector & Good Features to Track
corners = cv2.goodFeaturesToTrack(gray,25,0.01,10) # finding 25 best corners
corners = np.int0(corners)
for i in corners: 
    x,y = i.ravel()
    cv2.circle(img,(x,y),3,255,-1)
plt.imshow(img),plt.show()
#SIFT: Scale Invariant Feature Transform
sift = cv2.SIFT() #OR
sift = cv2.xfeatures2d.SIFT_create()
kp = sift.detect(gray,None)     #finds the keypoint in the images
img=cv2.drawKeypoints(gray,kp)
cv2.imwrite('sift_keypoints.jpg',img)
img=cv2.drawKeypoints(gray,kp,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite('sift_keypoints.jpg',img)
#SURF: Speeded-up Robust Feature
surf = cv2.SURF(400)    # set hessian threshold to 400
kp, des = surf.detectAndCompute(img,None)
len(kp)     #699
print(surf.hessianThreshold)     #400.0 # Check present Hessian threshold
# In actual cases, it is better to have a value 300-500
surf.hessianThreshold = 50000   #set Hessian threshold to 50000
#Again compute keypoints and check its number.
kp, des = surf.detectAndCompute(img,None)
print(len(kp))   ##if less than 50, draw it on the image.
img2 = cv2.drawKeypoints(img,kp,None,(255,0,0),4)
plt.imshow(img2),plt.show()
#SURF: blob detector, detects white blobs on wings
#Check upright flag, if it False, set it to True
print(surf.upright)  #False
surf.upright = True
#Recompute the feature points and draw it
kp = surf.detect(img,None)
img2 = cv2.drawKeypoints(img,kp,None,(255,0,0),4)
plt.imshow(img2),plt.show()
#check the descriptor size and change it to 128 if it is only 64-dim.
# Find size of descriptor
print(surf.descriptorSize()) #64
# That means flag, "extended" is False.
surf.extended #False
#Make it to True to get 128-dim descriptors.
surf.extended = True
kp, des = surf.detectAndCompute(img,None)
print(surf.descriptorSize())
#FAST algorithm for corner detection
fast=cv2.FastFeatureDetector()
#fast=cv2.FastFeatureDetector_create()
kp=fast.detect(img,None)
img2 = cv2.drawKeypoints(img, kp, color=(255,0,0))
#img2=cv2.drawKeypoints(img,kp,(255,0,0))
print("Threshold: ", fast.getInt('threshold'))
print("nonmaxSuppression: ", fast.getBool('nonmaxSuppression'))
print("neighborhood: ", fast.getInt('type'))
print("Total Keypoints with nonmaxSuppression: ", len(kp))
cv2.imwrite('fast_true.png',img2)
# Disable nonmaxSuppression
fast.setBool('nonmaxSuppression',0)
kp = fast.detect(img,None)
print("Total Keypoints without nonmaxSuppression: ", len(kp))
img3 = cv2.drawKeypoints(img, kp, color=(255,0,0))
cv2.imwrite('fast_false.png',img3)
#BRIEF: Binary Robust Independent Elementary Features
star = cv2.FeatureDetector_create("STAR") # Initiate STAR detector
brief = cv2.DescriptorExtractor_create("BRIEF") # Initiate BRIEF extractor
kp = star.detect(img,None) # find the keypoints with STAR
kp, des = brief.compute(img, kp) # compute the descriptors with BRIEF
print(brief.getInt('bytes'))
print(des.shape)
#ORB: Oriented FAST and Rotated BRIEF
orb = cv2.ORB() # Initiate STAR detector
kp = orb.detect(img,None) # find the keypoints with ORB
kp, des = orb.compute(img, kp) # compute the descriptors with ORB
img2 = cv2.drawKeypoints(img,kp,color=(0,255,0), flags=0) # draw only keypoints location,not size and orientation
plt.imshow(img2),plt.show()
##Feature matching
#Brute-Force matcher and FLANN Matcher
img1 = cv2.imread('box.png',0) # queryImage
img2 = cv2.imread('box_in_scene.png',0) # trainImage
orb = cv2.ORB() # Initiate SIFT detector
# find the keypoints and descriptors with SIFT
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) # create BFMatcher object
matches = bf.match(des1,des2) # Match descriptors.
matches = sorted(matches, key = lambda x:x.distance) # Sort them in the order of their distance.
#Draw first 10 matches.
img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10], flags=2)
plt.imshow(img3),plt.show()
# Brute-Force Matching with SIFT Descriptors and Ratio Test
img1 = cv2.imread('box.png',0) # queryImage
img2 = cv2.imread('box_in_scene.png',0) # trainImage
sift = cv2.SIFT()   # Initiate SIFT detector
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)
# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])
# cv2.drawMatchesKnn expects list of lists as matches.
img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,flags=2)
plt.imshow(img3),plt.show()
#FLANN based Matcher (Fast Library for Approximate Nearest Neighbors)
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12
                   key_size = 12, # 20
                   multi_probe_level = 1) #2
#Another way/type
img1 = cv2.imread('box.png',0) # queryImage
img2 = cv2.imread('box_in_scene.png',0) # trainImage
sift = cv2.SIFT()   # Initiate SIFT detector
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50) # or pass empty dictionary
flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)
matchesMask = [[0,0] for i in range(len(matches))] # create mask to draw good matches
# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        matchesMask[i]=[1,0]
        draw_params = dict(matchColor = (0,255,0),
                           singlePointColor = (255,0,0),
                           matchesMask = matchesMask,
                           flags = 0)
img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
plt.imshow(img3,),plt.show()
