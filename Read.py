import cv2 as cv 
#Reading images

# img = cv.imread('1310203.png')

# cv.imshow('Aizen', img)
#cv.waitKey(0)

#Reading Videos
capture = cv.VideoCapture(0)

while True:
    istrue, frame = capture.read()
    cv.imshow('Video', frame)
    
    if cv.waitKey(20) & 0xFF==ord('d'):
        break

capture.release()
cv.destroyAllWindows()