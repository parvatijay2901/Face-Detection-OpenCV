import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

#Get the training data we previously made
data_path = 'faces/user/'
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path,f))]

#Create arrays for training data and labels

Training_Data, Labels = [],[]

#Open training images in our datapath
#Create a numpy array for training data
for i,files in enumerate(onlyfiles):
    image_path = data_path + onlyfiles[i]
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    Training_Data.append(np.asarray(images, dtype=np.uint8))
    Labels.append(i)
    
#Create a numpy array for both training and data lables
#Labels = np.asarray(images, dtype=np.int32)

#Initialize face recognizer
model = cv2.face.LBPHFaceRecognizer_create()

#Let's train our model
model.train(np.asarray(Training_Data), np.asarray(Labels))
print("Model trained successsfully")

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def face_detector(img,size=0.5):
    
    #convert image to grayscale
   # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(frame)
    if faces is ():
        return img,[]
    
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
        roi = img[y:y+h, x:x+w]
        roi = cv2.resize(roi, (200,200))
    return img,roi

#Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret,frame = cap.read()
    
    image, face = face_detector(frame)
    
    try:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            
            #Pass face to prediction model
            #"results" comprises of a tuple containing the label and the success value
        results = model.predict(face)
            
        if results[1] < 500:
            success = int( 100 * (1- (results[1])/400) )
            display_string = str(success) + '% success Authorised user'
            
        cv2.putText(image, display_string, (100,120), cv2.FONT_HERSHEY_COMPLEX, 1, (255,120,150), 2)
            
        if success >85:
            cv2.putText(image, "Unlocked", (250,450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            cv2.imshow('Face Recognition', image)
            
        else:
            cv2.putText(image, "Locked", (250,450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            cv2.imshow('Face Recognition', image)
            
    except:
        cv2.putText(image, "No face found", (220,120), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
        #cv2.putText(image, "Locked", (250,450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
        cv2.imshow("Face recognition", image)
        pass
    if cv2.waitKey(1) == 13:
        break
    
cap.release()
cv2.destroyAllWindows()




