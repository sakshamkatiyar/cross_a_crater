import cv2
import serial
import xbee
import numpy as np
import time

'''cap = cv2.VideoCapture(3)
ret, frame = cap.read()
for i in xrange(5):
    ret, frame = cap.read()
'''
frame = cv2.imread('test.jpg')
cv2.imshow('frame',frame)
cv2.waitKey()

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
    #cv2.imshow('msk',mask)
    #cv2.waitKey()
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
    '''
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    peri = cv2.arcLength(contours[0],True)
    '''
    approx = cv2.approxPolyDP(contours[biggest],0.05*peri,True)                                                                                #drawing the biggest polyline
    cv2.polylines(frame, [approx], True, (0,255,0), 3)
    x1 = approx[0][0][0]
    y1 = approx[0][0][1]
    x2 = approx[1][0][0]
    y2 = approx[1][0][1]
    x3 = approx[3][0][0]
    y3 = approx[3][0][1]
    x4 = approx[2][0][0]
    y4 = approx[2][0][1]

    #print x1, y1
    #print x2, y2
    #print x3, y3
    #print x4, y4
                                                                                    #points remapped from source image from camera
                                                                                    #to cropped image try to match x1, y1,.... to the respective near values
    pts1 = np.float32([[x2,y2],[x4,y4],[x1,y1],[x3,y3]]) 
    pts2 = np.float32([[0,0],[0,389],[640,0],[640,389]])
    persM = cv2.getPerspectiveTransform(pts1,pts2)
    img = cv2.warpPerspective(frame,persM,(640,389))                                # resolution set to 400x400.....we are processing image on this resolution
    return img

frame = get_perspective_image(frame)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 140, 255, 0)
blurred = cv2.medianBlur(thresh, 3)
#cv2.imshow('Blurred', blurred)
contours, hierarchy = cv2.findContours(blurred, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)
j=0
bridge1=bridge2a=bridge2b=0
crater=[[0 for i in range(7)] for j in range(3)]
for i in xrange(len(contours)):
    #print cv2.contourArea(contours[i])
    #print cnt
    if cv2.contourArea(contours[i])>=120:
        j+=1
        #simplified_cnt = cv2.approxPolyDP(contours[i],0.001*cv2.arcLength(contours[i],True),True)
        #x = simplified_cnt[-1][0][0]
        #y = simplified_cnt[-1][0][1]
        #cv2.drawContours(frame, contours, i, (0,255,0), 2)
    if 700<=cv2.contourArea(contours[i])<=875:
        M = cv2.moments(contours[i])                                                            #Moment of  the contour i
        centroidx = int(M['m10']/M['m00'])                                                      #X co-ordinate of the centroid of the contour i
        centroidy = int(M['m01']/M['m00'])
        b,g,r=(frame[centroidy,centroidx])
        print 'Crater found at ' , centroidx, ',' , centroidy, 'with value bgr as',(b,g,r)
        #print hierarchy[0][i]
        if 50<=centroidy<=100:
            bridge1+=1
            x = centroidx-107
            print 'b1', x
            crater[0][x/38]=1
        elif 250<=centroidy<=290:
            bridge2a+=1
            x = centroidx-128
            print 'b2a', x
            crater[1][x/38]=1
        elif 290<=centroidy<=330:
            bridge2b+=1
            x = centroidx-130
            print 'b2b', x
            crater[2][x/38]=1
    if 150<=cv2.contourArea(contours[i])<=200:
        M = cv2.moments(contours[i])                                                            #Moment of  the contour i
        centroidx = int(M['m10']/M['m00'])                                                      #X co-ordinate of the centroid of the contour i
        centroidy = int(M['m01']/M['m00'])
        b,g,r=(frame[centroidy,centroidx])
        print 'Obstacle found at ' , centroidx, ',' , centroidy, 'with value bgr as',(b,g,r)
        if 85<=centroidy<=125:
            bridge1+=1
            x = centroidx-127
            print 'b1', x
            #crater[0][x/35]=1
        elif 265<=centroidy<=300:
            bridge2a+=1
            x = centroidx-128
            print 'b2a', x
            #crater[1][x/35]=1
    cv2.imshow('frame',frame)
    #cv2.imshow('dst',dst)
    cv2.waitKey()

print 'bridge1 ', bridge1
print 'bridge2a ', bridge2a
print 'bridge2b ', bridge2b
print 'crater ', crater

#print len(contours)
print 'j ', j

#if cv2.waitKey(1) & 0xFF == ord('q'):
#        break
#out.release()
#cap.release()
cv2.imshow('frame',frame)
cv2.imwrite('frame.jpg',frame)
cv2.waitKey()
cv2.destroyAllWindows()
