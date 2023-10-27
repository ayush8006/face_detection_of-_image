import cv2
  
#   *load our required xml file*
detect=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
#  for image capturing 
# 0/1--> webcam
# give picure name if present in same directory
#  
imp_img=cv2.VideoCapture("nis.jpg")
#this returns two values --first it returns true if it has read the image otherwise return false
#second we are going to get pixel dimensions of the image
res,img = imp_img.read()
#as haarcascade is trained for grey scale image so we need to convert img into grey scale
grey=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#for detecting faces of different sizes of images for this weuse detectmultiscale method inside detect
#which takes 3 input ---grey scale img,resizing command ,neighbouring code 
faces=detect.detectMultiScale(grey, 1.3, 5)

#5 paremeters inside rectangle method 5th one is width of border
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),4)
#to show the resut
cv2.imshow("Nishu Image ",img)
#after that do just 3 things ---give the wait key 0-means the timr
#for which you want image to stay and 1 means 1 ms of time,give release key and then 
#destroy the window
cv2.waitKey(0)
imp_img.release()
cv2.destroyAllWindow()



