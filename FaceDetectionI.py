# import 
import cv2

# load dataset
trained_data = cv2.CascadeClassifier('face.xml')

 
#choose image
img = cv2.imread('one.jpg')

#conversion to b&w
grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#detect faces
faceCoordinates = trained_data.detectMultiScale(grayimg, scaleFactor=1.2, minNeighbors=5)


for x,y,w,h in faceCoordinates:

    cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255),2)

cv2.imshow('window', img)
cv2.waitKey()

print('end')





