import cv2
import serial
import xbee
import numpy as np
import time
import os
import itertools

'''cap = cv2.VideoCapture(3)
ret, frame = cap.read()
for i in xrange(5):
    ret, frame = cap.read()
'''


'''
*
*Function Name: get_perspective_image
*
*Input: frame->image captured from camera is passed to it and it gives cropped image
*Output: img ->cropped image
*Logic: image inside any rectangle is cropped into desired size-> here 400x400
*       it is same function used in task3 to crop the image
*Example Call: img = get_perspective_image(frame)   ->where frame is image captured
*
'''
def get_perspective_image(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lower = np.array([0, 0, 0])                                                             #black color mask
    upper = np.array([50, 50, 50])
    mask = cv2.inRange(frame, lower, upper)
    ret,thresh1 = cv2.threshold(mask,127,255,cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    biggest = 0
    max_area = 0
    min_size = thresh1.size/4
    index1 = 0
    for i in xrange(len(contours)):
        area = cv2.contourArea(contours[i])
        if area > 1000:
            peri = cv2.arcLength(contours[i],True)
        if area > max_area: 
            biggest = index1
            max_area = area
        index1 = index1 + 1

    approx = cv2.approxPolyDP(contours[biggest],0.05*peri,True)                                                                                #drawing the biggest polyline
    #cv2.polylines(frame, [approx], True, (0,255,0), 3)
    x1 = approx[0][0][0]
    y1 = approx[0][0][1]
    x2 = approx[1][0][0]
    y2 = approx[1][0][1]
    x3 = approx[3][0][0]
    y3 = approx[3][0][1]
    x4 = approx[2][0][0]
    y4 = approx[2][0][1]
                                                                                 #points remapped from source image from camera
                                                                                    #to cropped image try to match x1, y1,.... to the respective near values
    pts1 = np.float32([[x2,y2],[x4,y4],[x1,y1],[x3,y3]]) 
    pts2 = np.float32([[0,0],[0,389],[640,0],[640,389]])
    persM = cv2.getPerspectiveTransform(pts1,pts2)
    img = cv2.warpPerspective(frame,persM,(640,389))                                # resolution set to 400x400.....we are processing image on this resolution
    return img

def craters(frame):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    template = cv2.imread('crater.jpg', 0)
    w, h = template.shape[::-1]
    res = cv2.matchTemplate(frame_gray, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.9
    loc = np.where(res>=threshold)
    centroidx=centroidy=0
    for pt in zip(*loc[::-1]):
        centroidx = pt[0] + w/2
        centroidy = pt[1] + h/2
        if 50<=centroidy<=100:
            x = centroidx-107
            crater[0][x/38]=1
        elif 250<=centroidy<=290:
            x = centroidx-108
            crater[1][x/38]=1
        elif 290<=centroidy<=330:
            x = centroidx-108
            crater[2][x/38]=1

def obstacles(frame):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    template = cv2.imread('obstacle1.jpg', 0)
    w, h = template.shape[::-1]
    res = cv2.matchTemplate(frame_gray, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
    loc = np.where(res>=threshold)
    centroidx=centroidy=0
    for pt in zip(*loc[::-1]):
        centroidx = pt[0] + w/2
        centroidy = pt[1] + h/2
        if 243<=centroidy<=263:
            x = centroidx-108
            obstacle[0][x/38]=1
        elif 323<=centroidy<=343:
            x = centroidx-108
            obstacle[1][x/38]=1

def numbers(frame):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    file_list = os.listdir(r"digits/")
    old_path = os.getcwd()
    os.chdir(r"digits/")
    for file_name in file_list:
        template = cv2.imread(file_name, 0)
        w, h = template.shape[::-1]
        res = cv2.matchTemplate(frame_gray, template, cv2.TM_CCOEFF_NORMED)
        threshold = 0.8
        loc = np.where(res>=threshold)
        centroidx=centroidy=0
        for pt in zip(*loc[::-1]):
            centroidx = pt[0] + w/2
            centroidy = pt[1] + h/2
            if 10<=centroidy<=50:
                number[(centroidx/550)+2]=int(file_name[:1])
            elif 290<=centroidy<=330:
                if centroidx<550:
                    number[1]=int(file_name[:1])
                else:
                    number[0]=int(file_name[:1])
    os.chdir(old_path)

def bot(frame):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    template1 = cv2.imread('bot1.jpg', 0)
    template2 = cv2.imread('bot2.jpg', 0)
    w1, h1 = template1.shape[::-1]
    w2, h2 = template2.shape[::-1]
    res1 = cv2.matchTemplate(frame_gray, template1, cv2.TM_CCOEFF_NORMED)
    res2 = cv2.matchTemplate(frame_gray, template2, cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
    loc1 = np.where(res1>=threshold)
    loc2 = np.where(res2>=threshold)
    centroid1x=centroid1y=0
    centroid2x=centroid2y=0
    for pt in zip(*loc1[::-1]):
        centroid1x = pt[0] + w1/2
        centroid1y = pt[1] + h1/2
        print "1 at ", centroid1x, centroid1y
    for pt in zip(*loc2[::-1]):
        centroid2x = pt[0] + w2/2
        centroid2y = pt[1] + h2/2
        print "2 at ", centroid2x, centroid2y
    return (centroid2y-centroid1y)/(centroid2x-centroid1x)

def count(ar):
    j = 0
    for i in ar:
        if i==1:
            j+=1
    return j

def permutation(ar, r):
    permut = itertools.permutations(ar, r)
    for perm in permut:
        add = 0
        for element in perm:
            add+=element
        if add==Sum:
            return perm
    return None


#if cv2.waitKey(1) & 0xFF == ord('q'):
#        break
#out.release()
#cap.release()

frame = cv2.imread('test.jpg')
frame = get_perspective_image(frame)
crater = [[0 for i in range(7)] for j in range(3)]
obstacle = [[0 for i in range(7)] for j in range(2)]
number = [0 for i in range(4)]
craters(frame)
obstacles(frame)
numbers(frame)
bridge1 = count(crater[0])
bridge2a = count(crater[1])
bridge2b = count(crater[2])
obstacle2a = count(obstacle[0])
obstacle2b = count(obstacle[1])
Sum = 16
perm2 = permutation(number, bridge2a+bridge2b)
perm1 = permutation(number, bridge1)
if perm2!=None:
    print 'FOLLOW bridge2', perm2
elif perm1!=None:
    print 'FOLLOW bridge1', perm1
else:
    print 'Not Possible!!'
#slope = bot(frame)
#print 'slope ', slope
roi1 = frame[15:138, 104:381]
roi2 = frame[215:373, 105:383]

print 'bridge1 ', bridge1
print 'bridge2a ', bridge2a
print 'bridge2b ', bridge2b
print 'obstacle2a ', obstacle2a
print 'obstacle2b ', obstacle2b
print 'crater ', crater
print 'obstacle ', obstacle
print 'number ', number


cv2.imshow('frame',frame)
cv2.imshow('roi1', roi1)
cv2.imshow('roi2', roi2)
cv2.imwrite('frame.jpg',frame)
cv2.waitKey()
cv2.destroyAllWindows()
