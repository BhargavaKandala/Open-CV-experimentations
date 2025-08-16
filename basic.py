import cv2 as cv

img = cv.imread('1310203.png')
cv.imshow('Image', img)

# Converting to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray Image', gray)


# Blurring an Image
blur = cv.GaussianBlur(img, (9, 9), cv.BORDER_DEFAULT)
cv.imshow("Blurr", blur)

# Edge Cascade
canny = cv.Canny(img, 125, 175)
cv.imshow("Canny Image", canny)

# Dilating an image
dilated = cv.dilate(canny, (7, 7), iterations=3)
cv.imshow("Dilated", dilated)

# Eroded
eroded = cv.erode(dilated, (3, 3), iterations=2)
cv.imshow("Eroded", eroded)

# Resize
resized = cv.resize(img, (500, 500), interpolation=cv.INTER_AREA)
cv.imshow( "Resized",resized)
# There is another function to crop the window and I am bored now
# so i am skipping that 💩


cv.waitKey(0)